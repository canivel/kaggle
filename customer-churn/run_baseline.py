"""Full autonomous pipeline for Customer Churn competition.

Run: uv run python run_baseline.py

Uses the self-learning experiment loop:
- Runs baselines → tuning → ensemble → submit
- Extracts learnings after each experiment
- Propagates learnings to agent definitions
- Documents everything in docs/competition_log.md
"""

import sys
sys.path.insert(0, "../kaggle-agent/src")

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_agent.config import load_config
from kaggle_agent.agents.learning_loop import LearningExperimentLoop
from kaggle_agent.loop import Strategy
from kaggle_agent.pipeline.data import load_competition_data, preprocess_dataframe, apply_preprocessing
from kaggle_agent.pipeline.features import FeatureEngineer
from kaggle_agent.pipeline.models import LGBMModel, XGBModel, CatBoostModel
from kaggle_agent.pipeline.submission import generate_submission
from kaggle_agent.ensemble.stacking import StackedEnsemble, WeightedEnsemble
from kaggle_agent.agents.strategies import tabular_binary_strategies


AGENTS_DIR = Path("../.claude/agents")
SKILLS_DIR = Path("../.claude/skills/competitions")


def build_features(X: pd.DataFrame, fe: FeatureEngineer | None = None, fit: bool = True, y=None):
    """Build competition-specific features."""
    if fe is None:
        fe = FeatureEngineer()

        # Service count (binary flags)
        service_cols = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        available = [c for c in service_cols if c in X.columns]
        if available:
            fe.add_count_features(available, name="service_count")

        # Numeric ratios
        fe.add_ratio_features([
            ("TotalCharges", "tenure"),
            ("MonthlyCharges", "tenure"),
        ])

        # Numeric interactions
        fe.add_interaction_features(
            ["tenure", "MonthlyCharges", "TotalCharges"],
            max_order=2,
        )

        # Frequency encoding for categoricals
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            fe.add_frequency_encoding(cat_cols)

        # Groupby stats
        for group_col in ["Contract", "InternetService", "PaymentMethod"]:
            if group_col in X.columns:
                fe.add_groupby_stats(
                    group_col,
                    ["MonthlyCharges", "TotalCharges", "tenure"],
                    ["mean", "std"],
                )

        # Binning
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            if col in X.columns:
                fe.add_binning(col, n_bins=10, strategy="quantile")

    if fit:
        X = fe.fit_transform(X, y=y)
    else:
        X = fe.transform(X)

    return X, fe


def update_competition_log(log_path: Path, entry: dict) -> None:
    """Append an entry to the competition documentation log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        header = """# Customer Churn Competition Log
## Kaggle Playground Series S6E3

Tracking all experiments, learnings, and results.

---

"""
        log_path.write_text(header, encoding="utf-8")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n### {entry['title']}\n")
        f.write(f"**Date**: {entry['timestamp']}\n\n")
        if "description" in entry:
            f.write(f"{entry['description']}\n\n")
        if "results" in entry:
            f.write(f"**Results**:\n")
            for k, v in entry["results"].items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        if "learnings" in entry:
            f.write(f"**Learnings**:\n")
            for l in entry["learnings"]:
                f.write(f"- {l}\n")
            f.write("\n")
        if "next_steps" in entry:
            f.write(f"**Next Steps**:\n")
            for s in entry["next_steps"]:
                f.write(f"- {s}\n")
            f.write("\n")
        f.write("---\n")


def main():
    config = load_config("config.yaml")
    log_path = Path("docs/competition_log.md")

    print("=" * 60)
    print("Customer Churn - Self-Learning Pipeline")
    print("=" * 60)

    # === PHASE 1: Load Data ===
    print("\n[Phase 1] Loading data...")
    X_train, X_test, y_train, test_ids = load_competition_data(
        train_path=config.data.train_path,
        test_path=config.data.test_path,
        target_column=config.data.target_column,
        id_column=config.data.id_column,
    )

    # Encode target
    if y_train.dtype == object:
        y_train = (y_train == "Yes").astype(int)

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Churn rate: {y_train.mean():.3f}")

    update_competition_log(log_path, {
        "title": "Data Loading",
        "timestamp": datetime.datetime.now().isoformat(),
        "results": {
            "train_shape": str(X_train.shape),
            "test_shape": str(X_test.shape),
            "churn_rate": f"{y_train.mean():.4f}",
            "null_count": str(X_train.isnull().sum().sum()),
        },
    })

    # === PHASE 2: Preprocess ===
    print("\n[Phase 2] Preprocessing...")
    X_train, encoding_info = preprocess_dataframe(X_train)
    X_test = apply_preprocessing(X_test, encoding_info)

    # === PHASE 3: Feature Engineering ===
    print("\n[Phase 3] Engineering features...")
    X_train, fe = build_features(X_train, fit=True, y=y_train)
    X_test, _ = build_features(X_test, fe=fe, fit=False)
    n_features = X_train.shape[1]
    print(f"  Features: {n_features}")

    update_competition_log(log_path, {
        "title": "Feature Engineering",
        "timestamp": datetime.datetime.now().isoformat(),
        "description": f"Created {n_features} features using ratios, interactions, frequency encoding, groupby stats, and binning.",
        "results": {"n_features": n_features, "feature_steps": len(fe.feature_names)},
    })

    # === PHASE 4: Self-Learning Experiment Loop ===
    print("\n[Phase 4] Running self-learning experiment loop...")
    loop = LearningExperimentLoop(
        config,
        base_dir=Path("."),
        agents_dir=AGENTS_DIR,
        skills_dir=SKILLS_DIR,
    )
    loop.X_train = X_train
    loop.X_test = X_test
    loop.y_train = y_train
    loop.test_ids = test_ids
    loop._data_loaded = True

    # Run ALL 16 strategies
    strategies = tabular_binary_strategies()
    loop.add_strategies(strategies)
    results = loop.run(max_experiments=len(strategies))

    # Log experiment results
    kept = [r for r in results if r.status == "kept"]
    update_competition_log(log_path, {
        "title": f"Experiment Loop - {len(strategies)} Strategies",
        "timestamp": datetime.datetime.now().isoformat(),
        "description": f"Ran {len(results)} experiments, {len(kept)} kept.",
        "results": {
            "total_experiments": len(results),
            "kept": len(kept),
            "best_cv": f"{max((r.cv_score for r in results if r.cv_score), default=0):.6f}",
        },
        "learnings": [
            f"{r.model_type} ({r.description}): {r.cv_score:.6f}" for r in kept
        ] if kept else ["No improvements found"],
    })

    # === PHASE 5: Stacked Ensemble ===
    print("\n[Phase 5] Building stacked ensemble...")
    ensemble = StackedEnsemble(
        base_model_factories={
            "lgbm": lambda: LGBMModel(),
            "xgb": lambda: XGBModel(),
            "catboost": lambda: CatBoostModel(),
        },
        meta_learner="logistic",
        n_folds=5,
        seed=42,
    )

    ensemble_result = ensemble.fit(X_train, y_train)
    print(f"  Individual: {ensemble_result['oof_scores']}")
    print(f"  Ensemble:   {ensemble_result['ensemble_score']:.6f}")

    # Record ensemble learnings
    loop.learn_from_ensemble(
        model_scores=ensemble_result["oof_scores"],
        ensemble_score=ensemble_result["ensemble_score"],
    )

    update_competition_log(log_path, {
        "title": "Stacked Ensemble",
        "timestamp": datetime.datetime.now().isoformat(),
        "results": {
            "individual_scores": json.dumps(ensemble_result["oof_scores"]),
            "ensemble_score": f"{ensemble_result['ensemble_score']:.6f}",
        },
        "learnings": [
            f"Ensemble {'improved' if ensemble_result['ensemble_score'] > max(ensemble_result['oof_scores'].values()) else 'did NOT improve'} over best individual",
        ],
    })

    # === PHASE 6: Generate & Submit ===
    print("\n[Phase 6] Generating submission...")
    test_preds = ensemble.predict(X_test)

    sub_path = generate_submission(
        test_ids=test_ids,
        predictions=test_preds,
        id_column="id",
        target_column="Churn",
        output_path="submissions/ensemble_v1.csv",
    )
    print(f"  Saved: {sub_path}")

    # === FINAL: Propagate Learnings & Summary ===
    print("\n[Phase 7] Propagating learnings to agents...")
    from kaggle_agent.agents.orchestration import propagate_learnings
    if AGENTS_DIR.exists():
        updated = propagate_learnings(loop.learning_store, AGENTS_DIR, SKILLS_DIR)
        print(f"  Updated agents: {list(updated.keys())}")

    update_competition_log(log_path, {
        "title": "Pipeline Complete",
        "timestamp": datetime.datetime.now().isoformat(),
        "results": {
            "best_ensemble_score": f"{ensemble_result['ensemble_score']:.6f}",
            "submission_file": str(sub_path),
            "total_learnings": str(len(loop.learning_store._learnings)),
        },
        "next_steps": [
            "Submit to Kaggle and check LB score",
            "Run Optuna tuning for each model type (50 trials each)",
            "Try target encoding for categorical features",
            "Add more model diversity (different seeds, DART boosting)",
            "Build multi-level stacking ensemble",
        ],
    })

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(loop.tracker.summary())
    print(f"\nLearnings: {loop.get_learning_summary()}")
    print(f"Documentation: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
