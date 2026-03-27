"""Iteration 2: Fix overfitting, add XGBoost, Optuna tuning, TabPFN.

Key learnings from iteration 1:
  - CV=0.91647 but LB=0.91380 → 0.00267 gap (SEVERE overfitting)
  - Groupby stats and frequency encoding are likely culprits
  - Stacking meta-learner may overfit

Strategy:
  A) MINIMAL features (original + 2 ratios) → expect lower CV but better LB
  B) MODERATE features (original + ratios + interactions, NO groupby/freq)
  C) FULL features (all 46) — baseline comparison
  Then ensemble A+B+C with rank averaging (most robust)
  Also: XGBoost now fixed, Optuna tuning, TabPFN exploration
"""

import sys
sys.path.insert(0, "../kaggle-agent/src")

import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from kaggle_agent.config import load_config
from kaggle_agent.pipeline.data import load_competition_data, preprocess_dataframe, apply_preprocessing
from kaggle_agent.pipeline.features import FeatureEngineer
from kaggle_agent.pipeline.models import (
    LGBMModel, XGBModel, CatBoostModel,
    cross_validate, METRIC_FUNCTIONS,
)
from kaggle_agent.pipeline.submission import generate_submission, rank_average
from kaggle_agent.pipeline.tuning import tune_model
from kaggle_agent.tracking.experiments import ExperimentTracker, ExperimentResult
from kaggle_agent.agents.orchestration import Learning, LearningStore, propagate_learnings

AGENTS_DIR = Path("../.claude/agents")
SKILLS_DIR = Path("../.claude/skills/competitions")
LOG_PATH = Path("docs/competition_log.md")


def log_entry(title: str, results: dict = None, learnings: list = None, next_steps: list = None):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n### {title}\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        if results:
            f.write("**Results**:\n")
            for k, v in results.items():
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        if learnings:
            f.write("**Learnings**:\n")
            for l in learnings:
                f.write(f"- {l}\n")
            f.write("\n")
        if next_steps:
            f.write("**Next Steps**:\n")
            for s in next_steps:
                f.write(f"- {s}\n")
            f.write("\n")
        f.write("---\n")


def make_feature_sets(X_train_raw, X_test_raw, y_train):
    """Create 3 feature sets: minimal, moderate, full."""
    results = {}

    # ===== MINIMAL: just label-encode + 2 ratios =====
    X_tr, enc = preprocess_dataframe(X_train_raw.copy())
    X_te = apply_preprocessing(X_test_raw.copy(), enc)

    fe_min = FeatureEngineer()
    fe_min.add_ratio_features([("TotalCharges", "tenure"), ("MonthlyCharges", "tenure")])
    X_tr_min = fe_min.fit_transform(X_tr, y=y_train)
    X_te_min = fe_min.transform(X_te)
    results["minimal"] = (X_tr_min, X_te_min, enc)
    print(f"  Minimal: {X_tr_min.shape[1]} features")

    # ===== MODERATE: + interactions, binning, service count. NO groupby/freq =====
    X_tr2, enc2 = preprocess_dataframe(X_train_raw.copy())
    X_te2 = apply_preprocessing(X_test_raw.copy(), enc2)

    fe_mod = FeatureEngineer()
    fe_mod.add_ratio_features([("TotalCharges", "tenure"), ("MonthlyCharges", "tenure")])
    fe_mod.add_interaction_features(["tenure", "MonthlyCharges", "TotalCharges"], max_order=2)
    service_cols = ["PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    avail = [c for c in service_cols if c in X_tr2.columns]
    if avail:
        fe_mod.add_count_features(avail, name="service_count")
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in X_tr2.columns:
            fe_mod.add_binning(col, n_bins=5, strategy="quantile")

    X_tr_mod = fe_mod.fit_transform(X_tr2, y=y_train)
    X_te_mod = fe_mod.transform(X_te2)
    results["moderate"] = (X_tr_mod, X_te_mod, enc2)
    print(f"  Moderate: {X_tr_mod.shape[1]} features")

    # ===== FULL: everything (same as iteration 1) =====
    X_tr3, enc3 = preprocess_dataframe(X_train_raw.copy())
    X_te3 = apply_preprocessing(X_test_raw.copy(), enc3)

    fe_full = FeatureEngineer()
    fe_full.add_ratio_features([("TotalCharges", "tenure"), ("MonthlyCharges", "tenure")])
    fe_full.add_interaction_features(["tenure", "MonthlyCharges", "TotalCharges"], max_order=2)
    if avail:
        fe_full.add_count_features(avail, name="service_count")
    cat_cols = X_tr3.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        fe_full.add_frequency_encoding(cat_cols)
    for g in ["Contract", "InternetService", "PaymentMethod"]:
        if g in X_tr3.columns:
            fe_full.add_groupby_stats(g, ["MonthlyCharges", "TotalCharges", "tenure"], ["mean", "std"])
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in X_tr3.columns:
            fe_full.add_binning(col, n_bins=10, strategy="quantile")

    X_tr_full = fe_full.fit_transform(X_tr3, y=y_train)
    X_te_full = fe_full.transform(X_te3)
    results["full"] = (X_tr_full, X_te_full, enc3)
    print(f"  Full: {X_tr_full.shape[1]} features")

    return results


def train_model_cv(name, model_factory, X, y, tracker, n_folds=5, seed=42):
    """Train a model with CV and return OOF predictions + test predictions."""
    print(f"  Training {name}...")
    start = time.time()

    result = cross_validate(
        model_factory=model_factory,
        X=X, y=y,
        metric="roc_auc",
        n_folds=n_folds,
        seed=seed,
        return_oof=True,
    )

    duration = time.time() - start
    print(f"    CV: {result['cv_score']:.6f} ± {result['cv_std']:.6f} ({duration:.0f}s)")

    # Log experiment
    exp_id = tracker.next_id()
    tracker.log(ExperimentResult(
        experiment_id=exp_id,
        timestamp=datetime.datetime.now().isoformat(),
        model_type=name.split("_")[0],
        description=name,
        cv_score=result["cv_score"],
        cv_std=result["cv_std"],
        lb_score=None,
        status="kept",
        duration_seconds=duration,
        n_features=X.shape[1],
        params="",
    ))

    return result


def main():
    config = load_config("config.yaml")
    tracker = ExperimentTracker(config.experiment.results_file)
    store = LearningStore("learnings")

    print("=" * 70)
    print("ITERATION 2: Fix Overfitting + XGBoost + Optuna + TabPFN")
    print("=" * 70)

    # === Load raw data ===
    print("\n[1] Loading data...")
    X_train_raw, X_test_raw, y_train, test_ids = load_competition_data(
        train_path=config.data.train_path,
        test_path=config.data.test_path,
        target_column=config.data.target_column,
        id_column=config.data.id_column,
    )
    if not pd.api.types.is_numeric_dtype(y_train):
        y_train = (y_train == "Yes").astype(int)

    # === Create 3 feature sets ===
    print("\n[2] Creating feature sets (minimal / moderate / full)...")
    feature_sets = make_feature_sets(X_train_raw, X_test_raw, y_train)

    # === Train diverse models on each feature set ===
    print("\n[3] Training models across feature sets...")

    all_oof = {}  # name -> oof_preds
    all_test = {}  # name -> test_preds
    all_cv = {}  # name -> cv_score

    model_configs = {
        "lgbm_default": lambda: LGBMModel(),
        "lgbm_reg": lambda: LGBMModel(params={
            "num_leaves": 31, "learning_rate": 0.01, "n_estimators": 3000,
            "reg_alpha": 1.0, "reg_lambda": 1.0,
        }),
        "lgbm_deep": lambda: LGBMModel(params={
            "num_leaves": 63, "max_depth": 8, "min_child_samples": 50,
        }),
        "xgb_default": lambda: XGBModel(),
        "xgb_reg": lambda: XGBModel(params={
            "max_depth": 5, "learning_rate": 0.01, "n_estimators": 3000,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
        }),
        "catboost_default": lambda: CatBoostModel(),
    }

    for fs_name, (X_tr, X_te, _) in feature_sets.items():
        print(f"\n--- Feature set: {fs_name} ({X_tr.shape[1]} features) ---")

        for model_name, factory in model_configs.items():
            full_name = f"{model_name}_{fs_name}"
            try:
                result = train_model_cv(full_name, factory, X_tr, y_train, tracker)

                all_oof[full_name] = result["oof_preds"]
                all_cv[full_name] = result["cv_score"]

                # Get test predictions
                test_preds = np.zeros(len(X_te))
                for m in result["models"]:
                    test_preds += m.predict(X_te)
                test_preds /= len(result["models"])
                all_test[full_name] = test_preds

            except Exception as e:
                print(f"    CRASHED: {e}")

    # === Optuna tuning on moderate features (best balance) ===
    print("\n[4] Optuna tuning (LGBM on moderate features, 30 trials)...")
    X_mod, X_mod_test, _ = feature_sets["moderate"]

    try:
        optuna_result = tune_model(
            "lgbm", X_mod, y_train,
            metric="roc_auc",
            n_trials=30,
            timeout=600,  # 10 min max
        )
        print(f"  Best Optuna LGBM: {optuna_result['best_score']:.6f}")
        print(f"  Best params: {optuna_result['best_params']}")

        # Train with best params
        best_lgbm = lambda: LGBMModel(params=optuna_result["best_params"])
        result = train_model_cv("lgbm_optuna_moderate", best_lgbm, X_mod, y_train, tracker)
        all_oof["lgbm_optuna_moderate"] = result["oof_preds"]
        all_cv["lgbm_optuna_moderate"] = result["cv_score"]
        test_preds = np.zeros(len(X_mod_test))
        for m in result["models"]:
            test_preds += m.predict(X_mod_test)
        all_test["lgbm_optuna_moderate"] = test_preds / len(result["models"])

        log_entry("Optuna LGBM Tuning", results={
            "best_score": f"{optuna_result['best_score']:.6f}",
            "n_trials": optuna_result["n_trials"],
            "best_params": json.dumps(optuna_result["best_params"]),
        })
    except Exception as e:
        print(f"  Optuna failed: {e}")

    # === Optuna XGBoost ===
    print("\n[5] Optuna tuning (XGBoost on moderate features, 30 trials)...")
    try:
        optuna_xgb = tune_model(
            "xgb", X_mod, y_train,
            metric="roc_auc",
            n_trials=30,
            timeout=600,
        )
        print(f"  Best Optuna XGB: {optuna_xgb['best_score']:.6f}")

        best_xgb = lambda: XGBModel(params=optuna_xgb["best_params"])
        result = train_model_cv("xgb_optuna_moderate", best_xgb, X_mod, y_train, tracker)
        all_oof["xgb_optuna_moderate"] = result["oof_preds"]
        all_cv["xgb_optuna_moderate"] = result["cv_score"]
        test_preds = np.zeros(len(X_mod_test))
        for m in result["models"]:
            test_preds += m.predict(X_mod_test)
        all_test["xgb_optuna_moderate"] = test_preds / len(result["models"])
    except Exception as e:
        print(f"  Optuna XGB failed: {e}")

    # === TabPFN (on minimal features — it works best with < 100 features, < 10K samples) ===
    print("\n[6] TabPFN exploration...")
    X_min, X_min_test, _ = feature_sets["minimal"]

    try:
        from tabpfn import TabPFNClassifier
        from sklearn.model_selection import StratifiedKFold

        # TabPFN can't handle 594K rows — subsample for OOF
        # Use it on a 10K subsample for diversity
        n_sub = min(10000, len(X_min))
        idx = np.random.RandomState(42).choice(len(X_min), n_sub, replace=False)
        X_sub = X_min.iloc[idx].reset_index(drop=True)
        y_sub = y_train.iloc[idx].reset_index(drop=True)

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tabpfn_oof = np.zeros(n_sub)
        tabpfn_test = np.zeros(len(X_min_test))

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_sub, y_sub)):
            clf = TabPFNClassifier(device="cuda")
            clf.fit(X_sub.iloc[tr_idx], y_sub.iloc[tr_idx])
            tabpfn_oof[va_idx] = clf.predict_proba(X_sub.iloc[va_idx])[:, 1]
            tabpfn_test += clf.predict_proba(X_min_test)[:, 1] / 5
            print(f"    Fold {fold+1}: done")

        from sklearn.metrics import roc_auc_score
        tabpfn_score = roc_auc_score(y_sub, tabpfn_oof)
        print(f"  TabPFN CV (10K subsample): {tabpfn_score:.6f}")

        all_test["tabpfn_minimal"] = tabpfn_test
        all_cv["tabpfn_minimal"] = tabpfn_score

        log_entry("TabPFN Exploration", results={
            "cv_score_10k": f"{tabpfn_score:.6f}",
            "subsample_size": n_sub,
            "feature_set": "minimal",
        }, learnings=[
            "TabPFN trained on 10K subsample for diversity",
            "Score may not be directly comparable to full-data models",
        ])
    except Exception as e:
        print(f"  TabPFN failed: {e}")

    # === Build ensembles ===
    print("\n[7] Building ensembles...")

    # Only use models that completed successfully
    valid_models = {k: v for k, v in all_test.items() if k in all_cv}
    print(f"  {len(valid_models)} models available for ensembling")

    for name, score in sorted(all_cv.items(), key=lambda x: x[1], reverse=True):
        print(f"    {name}: {score:.6f}")

    # A) Simple average of ALL models
    if len(valid_models) >= 2:
        avg_preds = np.mean(list(all_test.values()), axis=0)
        sub_avg = generate_submission(test_ids, avg_preds, "id", "Churn", "submissions/iter2_simple_avg.csv")
        print(f"\n  Simple average → {sub_avg}")

    # B) Rank average (most robust, reduces overfitting)
    if len(valid_models) >= 2:
        rank_preds = rank_average(list(all_test.values()))
        sub_rank = generate_submission(test_ids, rank_preds, "id", "Churn", "submissions/iter2_rank_avg.csv")
        print(f"  Rank average → {sub_rank}")

    # C) Best minimal-only ensemble (least overfitting)
    minimal_models = {k: v for k, v in all_test.items() if "minimal" in k}
    if len(minimal_models) >= 2:
        rank_min = rank_average(list(minimal_models.values()))
        sub_min = generate_submission(test_ids, rank_min, "id", "Churn", "submissions/iter2_minimal_rank.csv")
        print(f"  Minimal rank avg → {sub_min}")

    # D) Moderate-only ensemble
    moderate_models = {k: v for k, v in all_test.items() if "moderate" in k}
    if len(moderate_models) >= 2:
        rank_mod = rank_average(list(moderate_models.values()))
        sub_mod = generate_submission(test_ids, rank_mod, "id", "Churn", "submissions/iter2_moderate_rank.csv")
        print(f"  Moderate rank avg → {sub_mod}")

    # E) Top-5 models by CV (rank averaged)
    top5 = sorted(all_cv.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_preds = [all_test[name] for name, _ in top5]
    if len(top5_preds) >= 2:
        rank_top5 = rank_average(top5_preds)
        sub_top5 = generate_submission(test_ids, rank_top5, "id", "Churn", "submissions/iter2_top5_rank.csv")
        print(f"  Top-5 rank avg → {sub_top5}")
        print(f"    Models: {[n for n, _ in top5]}")

    # === Record learnings ===
    print("\n[8] Recording learnings...")

    # Compare feature set impact
    for fs in ["minimal", "moderate", "full"]:
        fs_scores = {k: v for k, v in all_cv.items() if fs in k}
        if fs_scores:
            avg = np.mean(list(fs_scores.values()))
            store.add(Learning(
                timestamp=datetime.datetime.now().isoformat(),
                source_agent="kaggle-model-trainer",
                phase="model_training",
                iteration=18,
                category="feature",
                insight=f"Feature set '{fs}': avg CV = {avg:.6f} across {len(fs_scores)} models",
                impact="high",
                evidence=json.dumps({k: f"{v:.6f}" for k, v in fs_scores.items()}),
                action=f"Use '{fs}' features if LB correlates with CV",
                applied_to=["kaggle-feature-engineer", "kaggle-model-trainer"],
            ))

    propagate_learnings(store, AGENTS_DIR, SKILLS_DIR)

    # === Final log ===
    log_entry(
        "Iteration 2 Complete",
        results={
            "total_models": len(valid_models),
            "best_cv": f"{max(all_cv.values()):.6f}" if all_cv else "N/A",
            "best_model": max(all_cv, key=all_cv.get) if all_cv else "N/A",
            "submissions_generated": 5,
        },
        learnings=[
            f"{len(valid_models)} diverse models trained across 3 feature sets",
            "Rank averaging used for all ensembles (most robust against overfitting)",
            "XGBoost now working after API fix",
            "Submissions: simple_avg, rank_avg, minimal_rank, moderate_rank, top5_rank",
        ],
        next_steps=[
            "Submit rank_avg and moderate_rank submissions to compare LB",
            "If minimal features ≈ moderate features on LB, overfitting confirmed",
            "Then: focus on moderate feature set + Optuna tuning + seed averaging",
        ],
    )

    print("\n" + "=" * 70)
    print("ITERATION 2 COMPLETE")
    print(f"Models trained: {len(valid_models)}")
    print(f"Best CV: {max(all_cv.values()):.6f} ({max(all_cv, key=all_cv.get)})")
    print("Submissions ready in submissions/")
    print("=" * 70)


if __name__ == "__main__":
    main()
