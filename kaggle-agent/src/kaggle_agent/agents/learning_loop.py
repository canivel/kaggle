"""Self-learning experiment loop.

Wraps the base ExperimentLoop with learning capabilities:
- After each experiment, extract learnings (feature importance, model behavior)
- Propagate learnings to agent definitions
- Update competition skill with discovered insights
- Evolve strategies based on accumulated knowledge

This is the "always learning, always evolving" mechanism.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kaggle_agent.loop import ExperimentLoop, Strategy
from kaggle_agent.tracking.experiments import ExperimentResult
from kaggle_agent.agents.orchestration import (
    Learning,
    LearningStore,
    propagate_learnings,
    PIPELINE_PHASES,
)


class LearningExperimentLoop(ExperimentLoop):
    """Experiment loop that learns and evolves agents."""

    def __init__(
        self,
        config,
        base_dir: str | Path = ".",
        agents_dir: str | Path | None = None,
        skills_dir: str | Path | None = None,
    ):
        super().__init__(config, base_dir)
        self.learning_store = LearningStore(Path(base_dir) / "learnings")
        self.agents_dir = Path(agents_dir) if agents_dir else None
        self.skills_dir = Path(skills_dir) if skills_dir else None
        self.iteration = 0
        self._best_features: dict[str, float] = {}
        self._model_scores: dict[str, float] = {}
        self._ensemble_insights: list[str] = []

    def run_single(self, strategy: Strategy) -> ExperimentResult:
        """Run experiment and extract learnings."""
        result = super().run_single(strategy)
        self.iteration += 1

        # Extract learnings from this experiment
        self._extract_learnings(result, strategy)

        # Propagate learnings to agents every 5 experiments
        if self.iteration % 5 == 0 and self.agents_dir:
            self._propagate()

        return result

    def _extract_learnings(self, result: ExperimentResult, strategy: Strategy) -> None:
        """Extract learnings from an experiment result."""
        if result.status == "crashed":
            self._learn_from_crash(result, strategy)
            return

        if result.cv_score is None:
            return

        # Track model performance by type
        prev_best = self._model_scores.get(strategy.model_type)
        self._model_scores[strategy.model_type] = max(
            result.cv_score, prev_best or 0
        )

        # Learning: which model types work best
        if result.status == "kept":
            self._learn_from_improvement(result, strategy)

        if result.status == "discarded":
            self._learn_from_failure(result, strategy)

        # Learning: feature importance (if available)
        self._extract_feature_learnings(result, strategy)

    def _learn_from_improvement(self, result: ExperimentResult, strategy: Strategy) -> None:
        """Extract learnings when an experiment improves the score."""
        best = self.tracker.best_score(self.config.evaluation.metric_direction)

        # What made this work?
        params = strategy.params
        insight_parts = []

        if params.get("learning_rate", 0.05) < 0.03:
            insight_parts.append("low learning rate helps")
        if params.get("num_leaves", 31) > 60:
            insight_parts.append("more leaves improves capture of complex patterns")
        if params.get("n_estimators", 1000) > 2000:
            insight_parts.append("more iterations with patience helps")
        if params.get("reg_alpha", 0) > 0.5 or params.get("reg_lambda", 0) > 0.5:
            insight_parts.append("stronger regularization reduces overfitting")

        if insight_parts:
            learning = Learning(
                timestamp=datetime.datetime.now().isoformat(),
                source_agent="kaggle-model-trainer",
                phase="model_training",
                iteration=self.iteration,
                category="model",
                insight=f"{strategy.model_type}: {'; '.join(insight_parts)}",
                impact="high",
                evidence=f"CV improved to {result.cv_score:.6f}",
                action=f"Use similar params for {strategy.model_type}: {json.dumps(params)}",
                applied_to=["kaggle-model-trainer", "kaggle-ensembler"],
            )
            self.learning_store.add(learning)

    def _learn_from_failure(self, result: ExperimentResult, strategy: Strategy) -> None:
        """Extract learnings from a failed experiment."""
        best = self.tracker.best_score(self.config.evaluation.metric_direction)
        if best is None or result.cv_score is None:
            return

        gap = best - result.cv_score
        if gap > 0.01:  # Significant regression
            learning = Learning(
                timestamp=datetime.datetime.now().isoformat(),
                source_agent="kaggle-model-trainer",
                phase="model_training",
                iteration=self.iteration,
                category="model",
                insight=f"AVOID: {strategy.description} caused {gap:.4f} regression",
                impact="medium",
                evidence=f"CV dropped from {best:.6f} to {result.cv_score:.6f}",
                action=f"Don't use these params for {strategy.model_type}: {json.dumps(strategy.params)}",
                applied_to=["kaggle-model-trainer"],
            )
            self.learning_store.add(learning)

    def _learn_from_crash(self, result: ExperimentResult, strategy: Strategy) -> None:
        """Learn from crashed experiments."""
        learning = Learning(
            timestamp=datetime.datetime.now().isoformat(),
            source_agent="kaggle-model-trainer",
            phase="model_training",
            iteration=self.iteration,
            category="strategy",
            insight=f"CRASH: {strategy.description} - {result.notes}",
            impact="low",
            evidence=f"Experiment {result.experiment_id} crashed",
            action=f"Avoid configuration: {json.dumps(strategy.params)}",
            applied_to=["kaggle-model-trainer"],
        )
        self.learning_store.add(learning)

    def _extract_feature_learnings(self, result: ExperimentResult, strategy: Strategy) -> None:
        """Extract feature importance learnings."""
        # Load checkpoint if available
        checkpoint_dir = self.base_dir / self.config.experiment.checkpoint_dir / result.experiment_id
        meta_path = checkpoint_dir / "metadata.json"

        if not meta_path.exists():
            return

        # Check if we have models with feature importance
        try:
            import pickle
            models_path = checkpoint_dir / "models.pkl"
            if models_path.exists():
                with open(models_path, "rb") as f:
                    models = pickle.load(f)

                # Aggregate feature importance across folds
                all_importance = {}
                for model in models:
                    imp = model.feature_importance()
                    if imp:
                        for feat, score in imp.items():
                            if feat not in all_importance:
                                all_importance[feat] = []
                            all_importance[feat].append(score)

                if all_importance:
                    avg_importance = {
                        k: float(np.mean(v))
                        for k, v in all_importance.items()
                    }
                    # Sort and get top features
                    sorted_feats = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
                    top_10 = sorted_feats[:10]
                    bottom_5 = sorted_feats[-5:]

                    # Learn about important features
                    top_names = [f[0] for f in top_10]
                    if top_names != list(self._best_features.keys())[:10]:
                        self._best_features = avg_importance
                        learning = Learning(
                            timestamp=datetime.datetime.now().isoformat(),
                            source_agent="kaggle-model-trainer",
                            phase="model_training",
                            iteration=self.iteration,
                            category="feature",
                            insight=f"Top features: {', '.join(top_names[:5])}",
                            impact="high",
                            evidence=f"Feature importance from {strategy.model_type} (exp {result.experiment_id})",
                            action="Focus feature engineering on these features and their interactions",
                            applied_to=["kaggle-feature-engineer", "kaggle-eda"],
                        )
                        self.learning_store.add(learning)

                    # Learn about useless features
                    bottom_names = [f[0] for f in bottom_5]
                    if any(avg_importance[f[0]] < 1.0 for f in bottom_5):
                        learning = Learning(
                            timestamp=datetime.datetime.now().isoformat(),
                            source_agent="kaggle-model-trainer",
                            phase="model_training",
                            iteration=self.iteration,
                            category="feature",
                            insight=f"Low-value features: {', '.join(bottom_names)}",
                            impact="low",
                            evidence=f"Near-zero importance in {strategy.model_type}",
                            action="Consider dropping these features to reduce noise",
                            applied_to=["kaggle-feature-engineer"],
                        )
                        self.learning_store.add(learning)

                    # Save feature importance to disk
                    imp_path = self.base_dir / "learnings" / "feature_importance.json"
                    with open(imp_path, "w") as f:
                        json.dump(avg_importance, f, indent=2, sort_keys=True)

        except Exception:
            pass  # Feature importance extraction is best-effort

    def learn_from_ensemble(
        self,
        model_scores: dict[str, float],
        ensemble_score: float,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Record learnings from an ensemble experiment."""
        best_individual = max(model_scores.values())
        improvement = ensemble_score - best_individual

        if improvement > 0:
            learning = Learning(
                timestamp=datetime.datetime.now().isoformat(),
                source_agent="kaggle-ensembler",
                phase="ensemble",
                iteration=self.iteration,
                category="ensemble",
                insight=f"Ensemble gained +{improvement:.5f} over best individual model",
                impact="high" if improvement > 0.001 else "medium",
                evidence=f"Individual scores: {model_scores}, Ensemble: {ensemble_score:.6f}",
                action=f"Current best weights: {weights}" if weights else "Use equal-weight averaging",
                applied_to=["kaggle-ensembler", "kaggle-model-trainer", "kaggle-orchestrator"],
            )
        else:
            learning = Learning(
                timestamp=datetime.datetime.now().isoformat(),
                source_agent="kaggle-ensembler",
                phase="ensemble",
                iteration=self.iteration,
                category="ensemble",
                insight=f"Ensemble did NOT improve ({improvement:.5f}). Need more model diversity.",
                impact="high",
                evidence=f"Individual scores: {model_scores}, Ensemble: {ensemble_score:.6f}",
                action="Train models with different seeds, architectures, or feature subsets for diversity",
                applied_to=["kaggle-model-trainer", "kaggle-orchestrator"],
            )

        self.learning_store.add(learning)

    def learn_from_submission(
        self,
        cv_score: float,
        lb_score: float | None,
        submission_path: str,
    ) -> None:
        """Record learnings from a Kaggle submission."""
        if lb_score is not None:
            gap = cv_score - lb_score
            if abs(gap) > 0.005:
                learning = Learning(
                    timestamp=datetime.datetime.now().isoformat(),
                    source_agent="kaggle-orchestrator",
                    phase="submit_and_learn",
                    iteration=self.iteration,
                    category="strategy",
                    insight=f"CV-LB gap: {gap:.5f} ({'CV overestimates' if gap > 0 else 'CV underestimates'})",
                    impact="high",
                    evidence=f"CV={cv_score:.6f}, LB={lb_score:.6f}",
                    action="Adjust CV strategy to reduce gap" if gap > 0.01 else "Gap is acceptable",
                    applied_to=["kaggle-model-trainer", "kaggle-orchestrator"],
                )
                self.learning_store.add(learning)

    def _propagate(self) -> None:
        """Propagate learnings to all agent and skill files."""
        if self.agents_dir and self.agents_dir.exists():
            updated = propagate_learnings(
                self.learning_store,
                self.agents_dir,
                self.skills_dir,
            )
            if updated:
                from rich.console import Console
                console = Console()
                console.print(f"[bold cyan]Learnings propagated to: {list(updated.keys())}[/]")

    def get_learning_summary(self) -> str:
        """Get a summary of all learnings."""
        return self.learning_store.summary()
