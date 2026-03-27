"""Iteration 3: Apply top notebook techniques.

Based on analysis of top 10 notebooks:
1. ORIG_proba from IBM Telco dataset (huge gain per BlamerX, Sagar)
2. enable_categorical=True for XGBoost (Deotte, BlamerX)
3. 50K+ estimators with low LR (0.003-0.03)
4. Very aggressive colsample_bytree (0.32) for regularization
5. Chris Deotte approach: 3 diverse model types (linear + tree + neural)
6. Multi-seed averaging (3 seeds)
7. Simple average ensemble (NOT stacking - avoid overfitting)
"""

import sys
sys.path.insert(0, "../kaggle-agent/src")

import datetime, json, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from kaggle_agent.pipeline.submission import generate_submission

LOG_PATH = Path("docs/competition_log.md")


def log_entry(title, **kwargs):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n### {title}\n**Date**: {datetime.datetime.now().isoformat()}\n\n")
        for key, val in kwargs.items():
            if isinstance(val, list):
                f.write(f"**{key}**:\n")
                for v in val:
                    f.write(f"- {v}\n")
            elif isinstance(val, dict):
                f.write(f"**{key}**:\n")
                for k, v in val.items():
                    f.write(f"- {k}: {v}\n")
            else:
                f.write(f"**{key}**: {val}\n")
        f.write("\n---\n")


def download_original_telco():
    """Download original IBM Telco dataset for ORIG_proba features."""
    orig_path = Path("data/telco_original.csv")
    if orig_path.exists():
        return pd.read_csv(orig_path)

    print("  Downloading original IBM Telco dataset...", flush=True)
    import subprocess
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", "blastchar/telco-customer-churn",
         "-p", "data/", "--unzip"],
        capture_output=True, text=True,
    )
    # The file is named WA_Fn-UseC_-Telco-Customer-Churn.csv
    for f in Path("data").glob("*Telco*"):
        f.rename(orig_path)
        break

    if orig_path.exists():
        return pd.read_csv(orig_path)
    return None


def compute_orig_proba(train_df, test_df, orig_df, target_col="Churn"):
    """Compute target probability per feature value from original dataset.

    This is the #1 feature engineering technique from top notebooks.
    For each column, lookup the churn rate from the original IBM Telco dataset.
    """
    # Encode target in original
    if orig_df[target_col].dtype == object or not pd.api.types.is_numeric_dtype(orig_df[target_col]):
        orig_df = orig_df.copy()
        orig_df[target_col] = (orig_df[target_col] == "Yes").astype(int)

    orig_features = []
    for col in train_df.columns:
        if col in orig_df.columns and col != target_col:
            proba = orig_df.groupby(col)[target_col].mean()
            feat_name = f"ORIG_proba_{col}"
            train_df[feat_name] = train_df[col].map(proba)
            test_df[feat_name] = test_df[col].map(proba)
            # Fill missing with global mean
            global_mean = orig_df[target_col].mean()
            train_df[feat_name] = train_df[feat_name].fillna(global_mean)
            test_df[feat_name] = test_df[feat_name].fillna(global_mean)
            orig_features.append(feat_name)

    return train_df, test_df, orig_features


def prepare_data():
    """Load and prepare data with top notebook techniques."""
    print("[1] Loading data...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]

    # Drop id and target
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    # Identify column types
    cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = train.select_dtypes(include=["number"]).columns.tolist()
    print(f"  Categorical: {len(cat_cols)}, Numeric: {len(num_cols)}", flush=True)

    # === ORIG_proba features ===
    print("[2] Computing ORIG_proba features...", flush=True)
    orig_df = download_original_telco()
    orig_features = []
    if orig_df is not None:
        # Need to match column dtypes - orig may have different TotalCharges handling
        if "TotalCharges" in orig_df.columns:
            orig_df["TotalCharges"] = pd.to_numeric(orig_df["TotalCharges"], errors="coerce")
            orig_df["TotalCharges"] = orig_df["TotalCharges"].fillna(0)

        train, test, orig_features = compute_orig_proba(train, test, orig_df)
        print(f"  Added {len(orig_features)} ORIG_proba features", flush=True)
    else:
        print("  WARNING: Could not download original dataset", flush=True)

    # === Arithmetic interactions ===
    print("[3] Engineering features...", flush=True)
    if "tenure" in train.columns and "TotalCharges" in train.columns:
        for df in [train, test]:
            df["avg_monthly_charges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
            df["monthly_to_total_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)
            df["charges_deviation"] = df["MonthlyCharges"] - df["avg_monthly_charges"]
            df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]

    # Service counts
    yes_no_cols = [c for c in cat_cols if set(train[c].dropna().unique()) & {"Yes", "No"}]
    for df in [train, test]:
        df["service_count"] = sum((df[c] == "Yes").astype(int) for c in yes_no_cols)
        df["has_internet"] = (df.get("InternetService", "No") != "No").astype(int)

    # === Ordinal encode categoricals ===
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[cat_cols] = oe.fit_transform(train[cat_cols]).astype("float32")
    test[cat_cols] = oe.transform(test[cat_cols]).astype("float32")

    print(f"  Total features: {train.shape[1]}", flush=True)
    return train, test, y, test_ids, cat_cols


def train_xgb_top_params(X, y, X_test, n_folds=5, seeds=[42, 11, 99]):
    """XGBoost with top notebook params + multi-seed."""
    import xgboost as xgb

    print("\n[XGBoost] Top notebook params, multi-seed...", flush=True)
    all_oof = np.zeros(len(X))
    all_test = np.zeros(len(X_test))
    all_scores = []

    for seed in seeds:
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            model = xgb.XGBClassifier(
                n_estimators=50000,
                learning_rate=0.01,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.35,
                reg_alpha=1.0,
                reg_lambda=1.0,
                gamma=0.1,
                early_stopping_rounds=300,
                device="cuda",
                enable_categorical=True,
                random_state=seed,
                verbosity=0,
                n_jobs=-1,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds

        score = roc_auc_score(y, oof)
        print(f"  Seed {seed}: CV={score:.6f}", flush=True)
        all_scores.append(score)
        all_oof += oof / len(seeds)
        all_test += test_preds / len(seeds)

    avg_score = roc_auc_score(y, all_oof)
    print(f"  Multi-seed avg: CV={avg_score:.6f}", flush=True)
    return all_oof, all_test, avg_score


def train_lgbm_top_params(X, y, X_test, n_folds=5, seeds=[42, 11, 99]):
    """LightGBM with top notebook params + multi-seed."""
    import lightgbm as lgb

    print("\n[LightGBM] Top notebook params, multi-seed...", flush=True)
    all_oof = np.zeros(len(X))
    all_test = np.zeros(len(X_test))
    all_scores = []

    for seed in seeds:
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            model = lgb.LGBMClassifier(
                n_estimators=50000,
                learning_rate=0.02,
                num_leaves=63,
                max_depth=7,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=seed,
                verbosity=-1,
                n_jobs=-1,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(300, verbose=False)],
            )
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / n_folds

        score = roc_auc_score(y, oof)
        print(f"  Seed {seed}: CV={score:.6f}", flush=True)
        all_scores.append(score)
        all_oof += oof / len(seeds)
        all_test += test_preds / len(seeds)

    avg_score = roc_auc_score(y, all_oof)
    print(f"  Multi-seed avg: CV={avg_score:.6f}", flush=True)
    return all_oof, all_test, avg_score


def train_catboost_top_params(X, y, X_test, cat_cols, n_folds=5, seed=42):
    """CatBoost with top notebook params."""
    from catboost import CatBoostClassifier

    print("\n[CatBoost] Top notebook params...", flush=True)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model = CatBoostClassifier(
            iterations=10000,
            learning_rate=0.02,
            depth=4,
            min_data_in_leaf=20,
            subsample=0.9,
            random_seed=seed,
            verbose=0,
            eval_metric="Logloss",
            task_type="GPU",
            devices="0",
            early_stopping_rounds=300,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=0)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_folds

    score = roc_auc_score(y, oof)
    print(f"  CV={score:.6f}", flush=True)
    return oof, test_preds, score


def train_logistic(X, y, X_test, n_folds=5, seed=42):
    """Logistic Regression for diversity (Chris Deotte approach)."""
    print("\n[LogisticRegression] For diversity...", flush=True)
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    scaler = StandardScaler()

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
        X_tr = scaler.fit_transform(X.iloc[tr_idx])
        X_va = scaler.transform(X.iloc[va_idx])
        X_te = scaler.transform(X_test)

        model = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        model.fit(X_tr, y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_te)[:, 1] / n_folds

    score = roc_auc_score(y, oof)
    print(f"  CV={score:.6f}", flush=True)
    return oof, test_preds, score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 3: Top Notebook Techniques", flush=True)
    print("=" * 70, flush=True)

    X, X_test, y, test_ids, cat_cols = prepare_data()

    # Train 4 diverse model types (Chris Deotte approach)
    results = {}

    # 1. XGBoost (multi-seed)
    xgb_oof, xgb_test, xgb_score = train_xgb_top_params(X, y, X_test, seeds=[42, 11, 99])
    results["xgb"] = {"oof": xgb_oof, "test": xgb_test, "score": xgb_score}

    # 2. LightGBM (multi-seed)
    lgbm_oof, lgbm_test, lgbm_score = train_lgbm_top_params(X, y, X_test, seeds=[42, 11, 99])
    results["lgbm"] = {"oof": lgbm_oof, "test": lgbm_test, "score": lgbm_score}

    # 3. CatBoost
    cat_oof, cat_test, cat_score = train_catboost_top_params(X, y, X_test, cat_cols)
    results["catboost"] = {"oof": cat_oof, "test": cat_test, "score": cat_score}

    # 4. Logistic Regression (for diversity)
    lr_oof, lr_test, lr_score = train_logistic(X, y, X_test)
    results["logistic"] = {"oof": lr_oof, "test": lr_test, "score": lr_score}

    # === Ensembles ===
    print("\n[Ensembles]", flush=True)

    # Simple average (Chris Deotte style)
    all_test = [r["test"] for r in results.values()]
    all_oof = [r["oof"] for r in results.values()]

    simple_avg = np.mean(all_test, axis=0)
    simple_oof = np.mean(all_oof, axis=0)
    simple_score = roc_auc_score(y, simple_oof)
    print(f"  Simple avg (4 models): CV={simple_score:.6f}", flush=True)

    # Tree-only ensemble (XGB + LGBM + CatBoost)
    tree_test = np.mean([results["xgb"]["test"], results["lgbm"]["test"], results["catboost"]["test"]], axis=0)
    tree_oof = np.mean([results["xgb"]["oof"], results["lgbm"]["oof"], results["catboost"]["oof"]], axis=0)
    tree_score = roc_auc_score(y, tree_oof)
    print(f"  Tree-only avg (3 models): CV={tree_score:.6f}", flush=True)

    # Rank average
    rank_test = np.mean([rankdata(t) / len(t) for t in all_test], axis=0)
    rank_oof = np.mean([rankdata(o) / len(o) for o in all_oof], axis=0)
    rank_score = roc_auc_score(y, rank_oof)
    print(f"  Rank avg (4 models): CV={rank_score:.6f}", flush=True)

    # === Save submissions ===
    print("\n[Submissions]", flush=True)
    generate_submission(test_ids, simple_avg, "id", "Churn", "submissions/iter3_simple_avg.csv")
    generate_submission(test_ids, tree_test, "id", "Churn", "submissions/iter3_tree_avg.csv")
    generate_submission(test_ids, rank_test, "id", "Churn", "submissions/iter3_rank_avg.csv")
    print("  Saved: iter3_simple_avg.csv, iter3_tree_avg.csv, iter3_rank_avg.csv", flush=True)

    # === Log results ===
    log_entry(
        "Iteration 3 - Top Notebook Techniques",
        Results={
            "XGBoost (3-seed)": f"{xgb_score:.6f}",
            "LightGBM (3-seed)": f"{lgbm_score:.6f}",
            "CatBoost": f"{cat_score:.6f}",
            "LogisticRegression": f"{lr_score:.6f}",
            "Simple avg (4 models)": f"{simple_score:.6f}",
            "Tree avg (3 models)": f"{tree_score:.6f}",
            "Rank avg (4 models)": f"{rank_score:.6f}",
        },
        Techniques=[
            "ORIG_proba from IBM Telco dataset",
            "XGBoost: 50K trees, lr=0.01, colsample=0.35, enable_categorical",
            "LightGBM: 50K trees, lr=0.02, num_leaves=63",
            "CatBoost: 10K trees, lr=0.02, depth=4",
            "LogisticRegression: C=0.5 for diversity",
            "Multi-seed averaging (seeds 42, 11, 99)",
            "Simple average ensemble (avoids overfitting vs stacking)",
        ],
    )

    print("\n" + "=" * 70, flush=True)
    print("ITERATION 3 COMPLETE", flush=True)
    for name, r in results.items():
        print(f"  {name}: {r['score']:.6f}", flush=True)
    print(f"  Ensemble (simple avg): {simple_score:.6f}", flush=True)
    print(f"  Ensemble (rank avg): {rank_score:.6f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
