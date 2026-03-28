"""Iteration 12: Breakthrough attempt toward 0.93.

Analysis of plateau at 0.918-0.919:
- BlamerX exact params: 0.9187 (20-fold)
- Lean features 30-fold: 0.9181
- More features hurt, fewer features slightly hurt
- The 0.918-0.919 ceiling seems like the dataset's limit for GBDT

To break through, we need:
1. STACKING: Use model predictions as features for a meta-learner
   (not simple averaging which maxes at ~0.9181)
2. DEEPER INTERACTIONS: Exhaustive pairwise feature interactions
3. MORE ORIG_proba SIGNAL: Use original dataset more aggressively
   (conditional probabilities, not just marginal)
4. CONDITIONAL FEATURES: Features that only activate for certain customer segments
5. FEATURE ABLATION: Find the exact optimal feature set
"""

import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")

import gc
import warnings
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations, product
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")

N_FOLDS_OUTER = 20
N_FOLDS_INNER = 5


def prepare_with_conditional_orig():
    """Enhanced features with CONDITIONAL original dataset probabilities."""
    print("[1] Enhanced features with conditional ORIG...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"] == "Yes").astype(float)
    gm = oc.mean()

    # Standard ORIG_proba
    for col in train.columns:
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            proba = tmp.groupby(col)["_y"].mean()
            train[f"ORIG_{col}"] = train[col].map(proba).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(proba).fillna(gm).astype("float32")

    # CONDITIONAL ORIG_proba: P(churn | feature1, feature2)
    # This captures interactions in the original dataset
    key_pairs = [
        ("Contract", "InternetService"),
        ("Contract", "PaymentMethod"),
        ("InternetService", "PaymentMethod"),
        ("Contract", "PaperlessBilling"),
        ("InternetService", "OnlineSecurity"),
        ("Contract", "TechSupport"),
    ]
    for c1, c2 in key_pairs:
        if c1 in orig.columns and c2 in orig.columns and c1 in train.columns and c2 in train.columns:
            pair_key = orig[c1].astype(str) + "_" + orig[c2].astype(str)
            tmp = pd.DataFrame({"pair": pair_key, "y": oc.values})
            proba = tmp.groupby("pair")["y"].mean()
            name = f"ORIG_{c1}_{c2}"
            for df in [train, test]:
                df[name] = (df[c1].astype(str) + "_" + df[c2].astype(str)).map(proba).fillna(gm).astype("float32")

    # Distribution features
    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    ch_tc = np.sort(orig_tc[orig["Churn"] == "Yes"].values)
    nc_tc = np.sort(orig_tc[orig["Churn"] != "Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")
        df["pctrank_gap"] = df["pctrank_ch"] - df["pctrank_nc"]

    # Core features
    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")

        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        avail = [c for c in yes_cols if c in df.columns]
        df["svc_count"] = sum((df[c] == "Yes").astype(int) for c in avail)

        # Digit features
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v % 10).astype("float32")
            if col != "tenure":
                df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure":
                df[f"{col}_m12"] = (v % 12).astype("float32")

        # Bi-grams
        top_cats = ["Contract", "InternetService", "PaymentMethod",
                    "OnlineSecurity", "TechSupport", "PaperlessBilling"]
        avail_cats = [c for c in top_cats if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                if not c.startswith("BG_")]
    ngram_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                  if c.startswith("BG_")]

    print(f"  Features: {train.shape[1]}", flush=True)
    return train, test, y, test_ids, cat_cols, ngram_cols


def fold_process(X_tr, X_va, X_te, y_tr, cat_cols, ngram_cols):
    """In-fold encoding + Ridge."""
    all_cats = cat_cols + ngram_cols
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    X_tr[all_cats] = oe.fit_transform(X_tr[all_cats])
    X_va[all_cats] = oe.transform(X_va[all_cats])
    X_te[all_cats] = oe.transform(X_te[all_cats])

    gm = y_tr.mean()
    smooth = 10.0
    te_cols = cat_cols[:8] + ngram_cols[:10]
    for col in te_cols:
        tmp = pd.DataFrame({"c": X_tr[col], "y": y_tr.values})
        agg = tmp.groupby("c")["y"].agg(["mean", "count"])
        sm = (agg["count"] * agg["mean"] + smooth * gm) / (agg["count"] + smooth)
        X_tr[f"{col}_te"] = X_tr[col].map(sm).fillna(gm).astype("float32")
        X_va[f"{col}_te"] = X_va[col].map(sm).fillna(gm).astype("float32")
        X_te[f"{col}_te"] = X_te[col].map(sm).fillna(gm).astype("float32")

    for col in cat_cols[:6]:
        for stat in ["std"]:
            tmp = pd.DataFrame({"c": X_tr[col], "y": y_tr.values})
            agg = tmp.groupby("c")["y"].agg(stat)
            X_tr[f"{col}_{stat}"] = X_tr[col].map(agg).fillna(0).astype("float32")
            X_va[f"{col}_{stat}"] = X_va[col].map(agg).fillna(0).astype("float32")
            X_te[f"{col}_{stat}"] = X_te[col].map(agg).fillna(0).astype("float32")

    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[f"{col}_f"] = X_tr[col].map(freq).fillna(0).astype("float32")
        X_va[f"{col}_f"] = X_va[col].map(freq).fillna(0).astype("float32")
        X_te[f"{col}_f"] = X_te[col].map(freq).fillna(0).astype("float32")

    sc = StandardScaler()
    r = Ridge(alpha=10.0)
    r.fit(sc.fit_transform(X_tr.fillna(0)), y_tr)
    X_tr["ridge"] = np.clip(r.predict(sc.transform(X_tr.fillna(0))), 0, 1).astype("float32")
    X_va["ridge"] = np.clip(r.predict(sc.transform(X_va.fillna(0))), 0, 1).astype("float32")
    X_te["ridge"] = np.clip(r.predict(sc.transform(X_te.fillna(0))), 0, 1).astype("float32")

    return X_tr.fillna(0), X_va.fillna(0), X_te.fillna(0)


def two_level_stacking(train_df, test_df, y, cat_cols, ngram_cols):
    """Two-level stacking: L0 models → L1 Ridge meta-learner.

    L0: XGB, LGBM, XGB-different-params, LGBM-different-params
    L1: Ridge on OOF predictions
    """
    print("\n[2] Two-level stacking...", flush=True)

    l0_configs = [
        ("xgb_bx", "xgb", dict(
            n_estimators=50000, learning_rate=0.0063, max_depth=5,
            min_child_weight=6, subsample=0.81, colsample_bytree=0.32,
            reg_alpha=3.5, reg_lambda=1.3, gamma=0.79,
            early_stopping_rounds=500, device="cuda",
            random_state=42, verbosity=0, n_jobs=-1,
        )),
        ("xgb_deep", "xgb", dict(
            n_estimators=10000, learning_rate=0.01, max_depth=7,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.5,
            reg_alpha=0.5, reg_lambda=1.0, gamma=0.1,
            early_stopping_rounds=300, device="cuda",
            random_state=42, verbosity=0, n_jobs=-1,
        )),
        ("lgbm_reg", "lgbm", dict(
            n_estimators=10000, learning_rate=0.01, num_leaves=31,
            max_depth=5, subsample=0.8, colsample_bytree=0.5,
            reg_alpha=1.0, reg_lambda=2.0, min_child_samples=50,
            random_state=42, verbosity=-1, n_jobs=4,
        )),
        ("lgbm_wide", "lgbm", dict(
            n_estimators=5000, learning_rate=0.02, num_leaves=63,
            max_depth=7, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=20,
            random_state=42, verbosity=-1, n_jobs=4,
        )),
    ]

    # L0: Generate OOF predictions
    l0_oof = {}
    l0_test = {}

    for name, model_cls, params in l0_configs:
        print(f"  L0: {name}...", flush=True)
        oof = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))

        kf = StratifiedKFold(n_splits=N_FOLDS_OUTER, shuffle=True, random_state=42)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
            X_tr = train_df.iloc[tr_idx].copy()
            X_va = train_df.iloc[va_idx].copy()
            X_te = test_df.copy()
            X_tr, X_va, X_te = fold_process(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)

            if model_cls == "xgb":
                m = xgb.XGBClassifier(**params)
                m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
            else:
                m = lgb.LGBMClassifier(**params)
                m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])],
                      callbacks=[lgb.early_stopping(300, verbose=False)])

            oof[va_idx] = m.predict_proba(X_va)[:, 1]
            test_preds += m.predict_proba(X_te)[:, 1] / N_FOLDS_OUTER

            if fold % 5 == 0:
                print(f"    F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
            del m; gc.collect()

        score = roc_auc_score(y, oof)
        print(f"    {name} L0 CV: {score:.6f}", flush=True)
        l0_oof[name] = oof
        l0_test[name] = test_preds

    # L1: Ridge stacking
    print("\n  L1: Ridge stacking...", flush=True)
    L1_X = np.column_stack(list(l0_oof.values()))
    L1_X_test = np.column_stack(list(l0_test.values()))

    l1_oof = np.zeros(len(y))
    l1_test = np.zeros(len(test_df))

    kf = StratifiedKFold(n_splits=N_FOLDS_INNER, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(L1_X, y)):
        ridge = Ridge(alpha=1.0)
        ridge.fit(L1_X[tr_idx], y.iloc[tr_idx])
        l1_oof[va_idx] = np.clip(ridge.predict(L1_X[va_idx]), 0, 1)
        l1_test += np.clip(ridge.predict(L1_X_test), 0, 1) / N_FOLDS_INNER

    l1_score = roc_auc_score(y, l1_oof)
    print(f"  L1 Ridge CV: {l1_score:.6f}", flush=True)

    # Also try LogReg meta-learner
    l1_lr_oof = np.zeros(len(y))
    l1_lr_test = np.zeros(len(test_df))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(L1_X, y)):
        sc = StandardScaler()
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(sc.fit_transform(L1_X[tr_idx]), y.iloc[tr_idx])
        l1_lr_oof[va_idx] = lr.predict_proba(sc.transform(L1_X[va_idx]))[:, 1]
        l1_lr_test += lr.predict_proba(sc.transform(L1_X_test))[:, 1] / N_FOLDS_INNER

    l1_lr_score = roc_auc_score(y, l1_lr_oof)
    print(f"  L1 LogReg CV: {l1_lr_score:.6f}", flush=True)

    # Simple average for comparison
    simple_oof = np.mean(list(l0_oof.values()), axis=0)
    simple_test = np.mean(list(l0_test.values()), axis=0)
    simple_score = roc_auc_score(y, simple_oof)
    print(f"  Simple avg: {simple_score:.6f}", flush=True)

    return {
        "l0": l0_oof, "l0_test": l0_test,
        "l1_ridge": (l1_oof, l1_test, l1_score),
        "l1_logreg": (l1_lr_oof, l1_lr_test, l1_lr_score),
        "simple": (simple_oof, simple_test, simple_score),
    }


def main():
    print("=" * 70, flush=True)
    print("ITERATION 12: Breakthrough - Stacking + Conditional ORIG", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols, ngram_cols = prepare_with_conditional_orig()

    # Two-level stacking
    stack_results = two_level_stacking(train, test, y, cat_cols, ngram_cols)

    # Save submissions
    l1_oof, l1_test, l1_score = stack_results["l1_ridge"]
    l1_lr_oof, l1_lr_test, l1_lr_score = stack_results["l1_logreg"]
    simple_oof, simple_test, simple_score = stack_results["simple"]

    generate_submission(test_ids, l1_test, "id", "Churn", "submissions/iter12_stack_ridge.csv")
    generate_submission(test_ids, l1_lr_test, "id", "Churn", "submissions/iter12_stack_logreg.csv")
    generate_submission(test_ids, simple_test, "id", "Churn", "submissions/iter12_simple.csv")

    # Blend stacking with iter6
    try:
        iter6 = pd.read_csv("submissions/iter6_blamerx.csv")["Churn"].values
        blend = (l1_test + iter6) / 2
        generate_submission(test_ids, blend, "id", "Churn", "submissions/iter12_stack_i6.csv")
    except Exception:
        pass

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 12 - Stacking + Conditional ORIG\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        for name in stack_results["l0"]:
            cv = roc_auc_score(y, stack_results["l0"][name])
            f.write(f"- L0 {name}: {cv:.6f}\n")
        f.write(f"- **L1 Ridge stack**: {l1_score:.6f}\n")
        f.write(f"- **L1 LogReg stack**: {l1_lr_score:.6f}\n")
        f.write(f"- Simple avg: {simple_score:.6f}\n")
        f.write(f"\n---\n")

    print("\n" + "=" * 70, flush=True)
    print("ITERATION 12 COMPLETE", flush=True)
    print(f"  L1 Ridge stack: {l1_score:.6f}", flush=True)
    print(f"  L1 LogReg stack: {l1_lr_score:.6f}", flush=True)
    print(f"  Simple avg: {simple_score:.6f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
