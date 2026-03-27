"""Iteration 4: Fix feature leakage + all top notebook techniques.

Critical fix: ALL feature engineering (groupby, freq encoding, target encoding)
must happen INSIDE each CV fold to avoid leakage.

Based on researcher diagnosis:
- Feature leakage was inflating CV by ~0.002+
- ORIG_proba features (from external dataset - NOT leakage)
- Digit features (exploit synthetic data artifacts)
- Bi-gram categoricals
- charges_deviation, monthly_to_total_ratio, avg_monthly_charges
- enable_categorical for XGBoost
- 10-fold CV for stability
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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from kaggle_agent.pipeline.submission import generate_submission


def add_orig_proba(train, test):
    """ORIG_proba from IBM Telco - NOT leakage (external dataset)."""
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    orig_churn = (orig["Churn"] == "Yes").astype(float)
    gm = orig_churn.mean()

    for col in train.columns:
        if col in orig.columns:
            tmp = orig.copy()
            tmp["_y"] = orig_churn
            proba = tmp.groupby(col)["_y"].mean()
            train[f"ORIG_{col}"] = train[col].map(proba).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(proba).fillna(gm).astype("float32")
    return train, test


def add_static_features(df):
    """Features that don't depend on target (safe outside folds)."""
    # Arithmetic features
    df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
    df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
    df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")

    # Service count
    yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    avail = [c for c in yes_cols if c in df.columns]
    df["service_count"] = sum((df[c] == "Yes").astype(int) for c in avail)
    df["has_internet"] = (df.get("InternetService", "No") != "No").astype(int)

    # Digit features (exploit synthetic data artifacts)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            vals = df[col].fillna(0)
            df[f"{col}_mod10"] = (vals % 10).astype("float32")
            df[f"{col}_mod100"] = (vals % 100).astype("float32")
            if col != "tenure":
                df[f"{col}_frac"] = (vals - vals.astype(int)).astype("float32")

    # Bi-gram categoricals (top 6 pairs)
    top_cats = ["Contract", "InternetService", "PaymentMethod",
                "OnlineSecurity", "TechSupport", "PaperlessBilling"]
    avail_cats = [c for c in top_cats if c in df.columns]
    from itertools import combinations
    for c1, c2 in combinations(avail_cats, 2):
        df[f"{c1}_x_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

    return df


def fold_encode(X_tr, X_va, X_te, y_tr, cat_cols, bigram_cols):
    """Encode features INSIDE the fold (no leakage)."""
    # Ordinal encode original categoricals
    all_cats = cat_cols + bigram_cols
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr[all_cats] = oe.fit_transform(X_tr[all_cats]).astype("float32")
    X_va[all_cats] = oe.transform(X_va[all_cats]).astype("float32")
    X_te_enc = X_te.copy()
    X_te_enc[all_cats] = oe.transform(X_te_enc[all_cats]).astype("float32")

    # Target encoding (inside fold only)
    te = TargetEncoder(cv=5, smooth="auto")
    te_cols = cat_cols[:6]  # Top categoricals only
    te_names = [f"{c}_te" for c in te_cols]
    X_tr[te_names] = te.fit_transform(X_tr[te_cols], y_tr)
    X_va[te_names] = te.transform(X_va[te_cols])
    X_te_enc[te_names] = te.transform(X_te_enc[te_cols])

    # Frequency encoding (inside fold only)
    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[f"{col}_freq"] = X_tr[col].map(freq).fillna(0).astype("float32")
        X_va[f"{col}_freq"] = X_va[col].map(freq).fillna(0).astype("float32")
        X_te_enc[f"{col}_freq"] = X_te_enc[col].map(freq).fillna(0).astype("float32")

    return X_tr, X_va, X_te_enc


def main():
    print("=" * 70, flush=True)
    print("ITERATION 4: Fix Leakage + All Top Techniques", flush=True)
    print("=" * 70, flush=True)

    # Load raw data
    print("\n[1] Loading data...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    # ORIG_proba (external dataset, safe outside folds)
    print("[2] ORIG_proba features...", flush=True)
    train, test = add_orig_proba(train, test)

    # Static features (don't depend on target, safe outside folds)
    print("[3] Static features...", flush=True)
    train = add_static_features(train)
    test = add_static_features(test)

    # Identify column types BEFORE encoding
    cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
    # Separate bigram cols (generated by add_static_features)
    bigram_cols = [c for c in cat_cols if "_x_" in c]
    orig_cat_cols = [c for c in cat_cols if "_x_" not in c]
    num_cols = train.select_dtypes(include=["number"]).columns.tolist()

    print(f"  Original cats: {len(orig_cat_cols)}, Bigrams: {len(bigram_cols)}, Numeric: {len(num_cols)}", flush=True)

    # === TRAIN MODELS WITH PROPER IN-FOLD ENCODING ===
    n_folds = 10  # More folds for stability
    all_oof = {}
    all_test = {}

    import lightgbm as lgb
    from catboost import CatBoostClassifier

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

    # Accumulate test preds across folds for each config
    for name, seed, params in configs:
        print(f"\n[{name}]...", flush=True)
        oof = np.zeros(len(train))
        test_preds_accum = []

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
            # CRITICAL: encode INSIDE fold
            X_tr = train.iloc[tr_idx].copy()
            X_va = train.iloc[va_idx].copy()
            X_te = test.copy()
            y_tr = y.iloc[tr_idx]

            X_tr, X_va, X_te_enc = fold_encode(X_tr, X_va, X_te, y_tr, orig_cat_cols, bigram_cols)

            model = lgb.LGBMClassifier(
                n_estimators=3000, subsample=0.8, random_state=seed,
                verbosity=-1, n_jobs=4, **params,
            )
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y.iloc[va_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])

            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_preds_accum.append(model.predict_proba(X_te_enc)[:, 1])

            if fold % 3 == 0:
                print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)

        test_preds = np.mean(test_preds_accum, axis=0)
        score = roc_auc_score(y, oof)
        print(f"  {name} CV: {score:.6f}", flush=True)
        all_oof[name] = oof
        all_test[name] = test_preds

    # CatBoost
    print(f"\n[CatBoost]...", flush=True)
    oof_cb = np.zeros(len(train))
    test_cb_accum = []
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        y_tr = y.iloc[tr_idx]

        X_tr, X_va, X_te_enc = fold_encode(X_tr, X_va, X_te, y_tr, orig_cat_cols, bigram_cols)

        model = CatBoostClassifier(
            iterations=3000, learning_rate=0.02, depth=4,
            min_data_in_leaf=20, subsample=0.9, random_seed=42,
            verbose=0, eval_metric="Logloss", task_type="CPU",
            early_stopping_rounds=100,
        )
        model.fit(X_tr, y_tr, eval_set=(X_va, y.iloc[va_idx]), verbose=0)
        oof_cb[va_idx] = model.predict_proba(X_va)[:, 1]
        test_cb_accum.append(model.predict_proba(X_te_enc)[:, 1])

        if fold % 3 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof_cb[va_idx]):.6f}", flush=True)

    test_cb = np.mean(test_cb_accum, axis=0)
    cb_score = roc_auc_score(y, oof_cb)
    print(f"  CatBoost CV: {cb_score:.6f}", flush=True)
    all_oof["catboost"] = oof_cb
    all_test["catboost"] = test_cb

    # LogReg for diversity
    print(f"\n[LogReg]...", flush=True)
    oof_lr = np.zeros(len(train))
    test_lr_accum = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        y_tr = y.iloc[tr_idx]
        X_tr, X_va, X_te_enc = fold_encode(X_tr, X_va, X_te, y_tr, orig_cat_cols, bigram_cols)

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        X_te_s = sc.transform(X_te_enc)

        model = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        model.fit(X_tr_s, y_tr)
        oof_lr[va_idx] = model.predict_proba(X_va_s)[:, 1]
        test_lr_accum.append(model.predict_proba(X_te_s)[:, 1])

    test_lr = np.mean(test_lr_accum, axis=0)
    lr_score = roc_auc_score(y, oof_lr)
    print(f"  LogReg CV: {lr_score:.6f}", flush=True)
    all_oof["logistic"] = oof_lr
    all_test["logistic"] = test_lr

    # === ENSEMBLES ===
    print("\n=== ENSEMBLES ===", flush=True)

    # Tree-only (best in iter3)
    tree_keys = [k for k in all_oof if k != "logistic"]
    tree_o = np.mean([all_oof[k] for k in tree_keys], axis=0)
    tree_t = np.mean([all_test[k] for k in tree_keys], axis=0)
    tree_score = roc_auc_score(y, tree_o)
    print(f"Tree avg ({len(tree_keys)} models): {tree_score:.6f}", flush=True)

    # All models
    all_o = np.mean(list(all_oof.values()), axis=0)
    all_t = np.mean(list(all_test.values()), axis=0)
    all_score = roc_auc_score(y, all_o)
    print(f"All avg ({len(all_oof)} models):  {all_score:.6f}", flush=True)

    # Rank average
    rank_o = np.mean([rankdata(all_oof[k]) / len(y) for k in all_oof], axis=0)
    rank_t = np.mean([rankdata(all_test[k]) / len(test) for k in all_test], axis=0)
    rank_score = roc_auc_score(y, rank_o)
    print(f"Rank avg ({len(all_oof)} models): {rank_score:.6f}", flush=True)

    # Save
    generate_submission(test_ids, tree_t, "id", "Churn", "submissions/iter4_tree.csv")
    generate_submission(test_ids, all_t, "id", "Churn", "submissions/iter4_all.csv")
    generate_submission(test_ids, rank_t, "id", "Churn", "submissions/iter4_rank.csv")

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 4 - Fixed Leakage + All Top Techniques\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        f.write("**Key changes**: All FE inside CV folds, 10-fold CV, digit features, bi-gram cats, target encoding\n\n")
        for name in all_oof:
            f.write(f"- {name}: {roc_auc_score(y, all_oof[name]):.6f}\n")
        f.write(f"- **Tree avg**: {tree_score:.6f}\n")
        f.write(f"- **All avg**: {all_score:.6f}\n")
        f.write(f"- **Rank avg**: {rank_score:.6f}\n")
        f.write(f"\n---\n")

    print("\nSubmissions saved!", flush=True)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
