"""Autonomous experiment loop.

Combines autoresearch's iterative experiment pattern with
deer-flow's multi-agent orchestration concept.

The loop:
1. Plan: Select next experiment (strategy library or agent-driven)
2. Execute: Run training + CV evaluation
3. Evaluate: Compare against best known score
4. Decide: Keep or discard
5. Log: Record in experiments.tsv
6. Repeat
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from kaggle_agent.config import CompetitionConfig
from kaggle_agent.pipeline.data import load_competition_data, preprocess_dataframe, apply_preprocessing
from kaggle_agent.pipeline.models import create_model, cross_validate, BaseModel
from kaggle_agent.pipeline.submission import generate_submission, blend_predictions
from kaggle_agent.tracking.experiments import ExperimentTracker, ExperimentResult

console = Console()


class Strategy:
    """A single experiment strategy definition."""

    def __init__(
        self,
        name: str,
        model_type: str,
        params: dict[str, Any] | None = None,
        feature_steps: list[dict] | None = None,
        description: str = "",
    ):
        self.name = name
        self.model_type = model_type
        self.params = params or {}
        self.feature_steps = feature_steps or []
        self.description = description or f"{model_type} with {params}"


class ExperimentLoop:
    """Main autonomous experiment loop.

    Usage:
        config = load_config("config.yaml")
        loop = ExperimentLoop(config)
        loop.add_strategy(Strategy("baseline_lgbm", "lgbm"))
        loop.add_strategy(Strategy("tuned_xgb", "xgb", {"max_depth": 8}))
        loop.run()
    """

    def __init__(self, config: CompetitionConfig, base_dir: str | Path = "."):
        self.config = config
        self.base_dir = Path(base_dir)
        self.tracker = ExperimentTracker(
            self.base_dir / config.experiment.results_file
        )
        self.strategies: list[Strategy] = []
        self._data_loaded = False
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.test_ids: pd.Series | None = None
        self.encoding_info: dict = {}

    def add_strategy(self, strategy: Strategy) -> None:
        """Add an experiment strategy to the queue."""
        self.strategies.append(strategy)

    def add_strategies(self, strategies: list[Strategy]) -> None:
        """Add multiple strategies."""
        self.strategies.extend(strategies)

    def load_data(self) -> None:
        """Load and preprocess competition data."""
        if self._data_loaded:
            return

        console.print("[bold blue]Loading competition data...[/]")

        self.X_train, self.X_test, self.y_train, self.test_ids = load_competition_data(
            train_path=self.config.data.train_path,
            test_path=self.config.data.test_path,
            target_column=self.config.data.target_column,
            id_column=self.config.data.id_column,
            original_data_path=self.config.data.original_data_path,
        )

        # Preprocess
        self.X_train, self.encoding_info = preprocess_dataframe(self.X_train)
        self.X_test = apply_preprocessing(self.X_test, self.encoding_info)

        console.print(
            f"  Train: {self.X_train.shape[0]:,} x {self.X_train.shape[1]} | "
            f"Test: {self.X_test.shape[0]:,} x {self.X_test.shape[1]}"
        )
        self._data_loaded = True

    def run_single(self, strategy: Strategy) -> ExperimentResult:
        """Run a single experiment strategy."""
        self.load_data()

        exp_id = self.tracker.next_id()
        console.print(f"\n[bold green]Experiment {exp_id}: {strategy.description}[/]")

        start_time = time.time()

        try:
            # Create model factory
            def model_factory():
                return create_model(strategy.model_type, params=strategy.params)

            # Cross-validate
            cv_result = cross_validate(
                model_factory=model_factory,
                X=self.X_train,
                y=self.y_train,
                metric=self.config.evaluation.metric,
                n_folds=self.config.evaluation.cv_folds,
                seed=self.config.evaluation.cv_seed,
                stratified=self.config.evaluation.cv_strategy == "stratified_kfold",
                return_oof=True,
            )

            duration = time.time() - start_time
            cv_score = cv_result["cv_score"]
            cv_std = cv_result["cv_std"]

            # Decide: keep or discard
            best = self.tracker.best_score(self.config.evaluation.metric_direction)
            if best is None:
                status = "kept"
            elif self.config.evaluation.metric_direction == "maximize":
                status = "kept" if cv_score > best else "discarded"
            else:
                status = "kept" if cv_score < best else "discarded"

            console.print(
                f"  CV: {cv_score:.6f} +/- {cv_std:.6f} | "
                f"Best: {best or 'N/A'} | Status: [{'green' if status == 'kept' else 'red'}]{status}[/]"
            )

            # If kept, save checkpoint
            if status == "kept":
                self._save_checkpoint(exp_id, cv_result, strategy)

            result = ExperimentResult(
                experiment_id=exp_id,
                timestamp=datetime.datetime.now().isoformat(),
                model_type=strategy.model_type,
                description=strategy.description,
                cv_score=cv_score,
                cv_std=cv_std,
                lb_score=None,
                status=status,
                duration_seconds=duration,
                n_features=self.X_train.shape[1],
                params=json.dumps(strategy.params),
            )

        except Exception as e:
            duration = time.time() - start_time
            console.print(f"  [red]CRASHED: {e}[/]")
            result = ExperimentResult(
                experiment_id=exp_id,
                timestamp=datetime.datetime.now().isoformat(),
                model_type=strategy.model_type,
                description=strategy.description,
                cv_score=None,
                cv_std=None,
                lb_score=None,
                status="crashed",
                duration_seconds=duration,
                n_features=self.X_train.shape[1] if self.X_train is not None else 0,
                params=json.dumps(strategy.params),
                notes=str(e),
            )

        self.tracker.log(result)
        return result

    def run(self, max_experiments: int | None = None) -> list[ExperimentResult]:
        """Run all queued strategies."""
        self.load_data()

        max_exp = max_experiments or self.config.experiment.max_experiments
        results = []

        console.print(f"\n[bold]Starting experiment loop ({len(self.strategies)} strategies)[/]")

        for i, strategy in enumerate(self.strategies):
            if i >= max_exp:
                break

            result = self.run_single(strategy)
            results.append(result)

        # Print summary
        self._print_summary()
        return results

    def generate_best_submission(self, output_path: str = "submission.csv") -> Path | None:
        """Generate submission using the best model checkpoint."""
        self.load_data()

        best = self.tracker.best_experiment(self.config.evaluation.metric_direction)
        if best is None:
            console.print("[red]No successful experiments found.[/]")
            return None

        checkpoint_dir = self.base_dir / self.config.experiment.checkpoint_dir / best.experiment_id
        if not checkpoint_dir.exists():
            console.print(f"[red]Checkpoint not found: {checkpoint_dir}[/]")
            return None

        # Load models and predict
        import pickle
        models_path = checkpoint_dir / "models.pkl"
        with open(models_path, "rb") as f:
            models = pickle.load(f)

        predictions = np.zeros(len(self.X_test))
        for model in models:
            predictions += model.predict(self.X_test)
        predictions /= len(models)

        path = generate_submission(
            test_ids=self.test_ids,
            predictions=predictions,
            id_column=self.config.data.id_column,
            target_column=self.config.data.target_column,
            output_path=output_path,
        )

        console.print(f"[green]Submission saved: {path} (from experiment {best.experiment_id})[/]")
        return path

    def _save_checkpoint(self, exp_id: str, cv_result: dict, strategy: Strategy) -> None:
        """Save model checkpoint."""
        import pickle
        checkpoint_dir = self.base_dir / self.config.experiment.checkpoint_dir / exp_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        with open(checkpoint_dir / "models.pkl", "wb") as f:
            pickle.dump(cv_result["models"], f)

        # Save metadata
        meta = {
            "experiment_id": exp_id,
            "model_type": strategy.model_type,
            "params": strategy.params,
            "cv_score": cv_result["cv_score"],
            "cv_std": cv_result["cv_std"],
            "fold_scores": cv_result["fold_scores"],
        }
        with open(checkpoint_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save OOF predictions
        if "oof_preds" in cv_result:
            np.save(checkpoint_dir / "oof_preds.npy", cv_result["oof_preds"])

    def _print_summary(self) -> None:
        """Print experiment summary table."""
        results = self.tracker.load_all()
        if not results:
            return

        table = Table(title="Experiment Results")
        table.add_column("ID", style="cyan")
        table.add_column("Model", style="green")
        table.add_column("CV Score", style="bold")
        table.add_column("CV Std")
        table.add_column("Status")
        table.add_column("Duration")
        table.add_column("Description")

        for r in results[-20:]:  # Last 20
            status_style = "green" if r.status == "kept" else "red" if r.status == "crashed" else "dim"
            table.add_row(
                r.experiment_id,
                r.model_type,
                f"{r.cv_score:.6f}" if r.cv_score else "N/A",
                f"{r.cv_std:.6f}" if r.cv_std else "N/A",
                f"[{status_style}]{r.status}[/]",
                f"{r.duration_seconds:.1f}s",
                r.description[:50],
            )

        console.print(table)
