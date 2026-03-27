"""Iteration 3 (fast version): LGBM multi-seed + CatBoost + LogReg with ORIG_proba.

Avoids XGBoost GPU/CPU OOM issues on our 3080.
Uses 3x LightGBM (different seeds/configs) + CatBoost + LogReg for diversity.
"""

import sys
sys.path.insert(0, "../kaggle-agent/src")

import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from kaggle_agent.pipeline.submission import generate_submission


def main():
    print("Loading data...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    # ORIG_proba from IBM Telco
    print("Computing ORIG_proba...", flush=True)
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    orig_churn = (orig["Churn"] == "Yes").astype(int)
    global_mean = orig_churn.mean()

    for col in train.columns:
        if col in orig.columns:
            proba = orig.copy()
            proba["_y"] = orig_churn
            proba = proba.groupby(col)["_y"].mean()
            train[f"ORIG_{col}"] = train[col].map(proba).fillna(global_mean)
            test[f"ORIG_{col}"] = test[col].map(proba).fillna(global_mean)

    # Basic features
    for df in [train, test]:
        df["avg_monthly"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
        df["charge_ratio"] = df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)
        df["charge_dev"] = df["MonthlyCharges"] - df["avg_monthly"]

    # Encode categoricals
    cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[cat_cols] = oe.fit_transform(train[cat_cols]).astype("float32")
    test[cat_cols] = oe.transform(test[cat_cols]).astype("float32")

    print(f"Features: {train.shape[1]}", flush=True)

    n_folds = 5
    all_oof = {}
    all_test = {}

    # === LightGBM x3 seeds ===
    import lightgbm as lgb

    configs = [
        ("lgbm_s42", 42, {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.02,
                          "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0,
                          "min_child_samples": 20}),
        ("lgbm_s11", 11, {"num_leaves": 63, "max_depth": 7, "learning_rate": 0.02,
                          "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0,
                          "min_child_samples": 20}),
        ("lgbm_reg", 99, {"num_leaves": 31, "max_depth": 5, "learning_rate": 0.01,
                          "colsample_bytree": 0.5, "reg_alpha": 1.0, "reg_lambda": 2.0,
                          "min_child_samples": 50}),
    ]

    for name, seed, params in configs:
        print(f"\n[{name}]...", flush=True)
        oof = np.zeros(len(train))
        test_preds = np.zeros(len(test))

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
            model = lgb.LGBMClassifier(
                n_estimators=3000, subsample=0.8, random_state=seed,
                verbosity=-1, n_jobs=-1, **params,
            )
            model.fit(
                train.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(train.iloc[va_idx], y.iloc[va_idx])],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            oof[va_idx] = model.predict_proba(train.iloc[va_idx])[:, 1]
            test_preds += model.predict_proba(test)[:, 1] / n_folds
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)

        score = roc_auc_score(y, oof)
        print(f"  {name} CV: {score:.6f}", flush=True)
        all_oof[name] = oof
        all_test[name] = test_preds

    # === CatBoost ===
    print("\n[CatBoost]...", flush=True)
    from catboost import CatBoostClassifier

    oof_cb = np.zeros(len(train))
    test_cb = np.zeros(len(test))
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        model = CatBoostClassifier(
            iterations=3000, learning_rate=0.02, depth=4,
            min_data_in_leaf=20, subsample=0.9, random_seed=42,
            verbose=0, eval_metric="Logloss", task_type="CPU",
            early_stopping_rounds=200,
        )
        model.fit(train.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=(train.iloc[va_idx], y.iloc[va_idx]), verbose=0)
        oof_cb[va_idx] = model.predict_proba(train.iloc[va_idx])[:, 1]
        test_cb += model.predict_proba(test)[:, 1] / n_folds
        print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof_cb[va_idx]):.6f}", flush=True)

    cb_score = roc_auc_score(y, oof_cb)
    print(f"  CatBoost CV: {cb_score:.6f}", flush=True)
    all_oof["catboost"] = oof_cb
    all_test["catboost"] = test_cb

    # === LogisticRegression ===
    print("\n[LogReg]...", flush=True)
    oof_lr = np.zeros(len(train))
    test_lr = np.zeros(len(test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        sc = StandardScaler()
        X_tr = sc.fit_transform(train.iloc[tr_idx])
        X_va = sc.transform(train.iloc[va_idx])
        X_te = sc.transform(test)
        model = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        model.fit(X_tr, y.iloc[tr_idx])
        oof_lr[va_idx] = model.predict_proba(X_va)[:, 1]
        test_lr += model.predict_proba(X_te)[:, 1] / n_folds

    lr_score = roc_auc_score(y, oof_lr)
    print(f"  LogReg CV: {lr_score:.6f}", flush=True)
    all_oof["logistic"] = oof_lr
    all_test["logistic"] = test_lr

    # === Ensembles ===
    print("\n=== ENSEMBLES ===", flush=True)
    all_t = list(all_test.values())
    all_o = list(all_oof.values())

    # Simple average
    simple_t = np.mean(all_t, axis=0)
    simple_o = np.mean(all_o, axis=0)
    simple_score = roc_auc_score(y, simple_o)
    print(f"Simple avg (5 models): {simple_score:.6f}", flush=True)

    # Rank average
    rank_t = np.mean([rankdata(t) / len(t) for t in all_t], axis=0)
    rank_o = np.mean([rankdata(o) / len(o) for o in all_o], axis=0)
    rank_score = roc_auc_score(y, rank_o)
    print(f"Rank avg (5 models):   {rank_score:.6f}", flush=True)

    # Tree-only (no logistic)
    tree_keys = ["lgbm_s42", "lgbm_s11", "lgbm_reg", "catboost"]
    tree_t = np.mean([all_test[k] for k in tree_keys], axis=0)
    tree_o = np.mean([all_oof[k] for k in tree_keys], axis=0)
    tree_score = roc_auc_score(y, tree_o)
    print(f"Tree avg (4 models):   {tree_score:.6f}", flush=True)

    # Save submissions
    generate_submission(test_ids, simple_t, "id", "Churn", "submissions/iter3_simple.csv")
    generate_submission(test_ids, rank_t, "id", "Churn", "submissions/iter3_rank.csv")
    generate_submission(test_ids, tree_t, "id", "Churn", "submissions/iter3_tree.csv")

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 3 Results (with ORIG_proba)\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        for name in all_oof:
            f.write(f"- {name}: {roc_auc_score(y, all_oof[name]):.6f}\n")
        f.write(f"- **Simple avg**: {simple_score:.6f}\n")
        f.write(f"- **Rank avg**: {rank_score:.6f}\n")
        f.write(f"- **Tree avg**: {tree_score:.6f}\n")
        f.write(f"\n---\n")

    print("\nSubmissions saved to submissions/", flush=True)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
