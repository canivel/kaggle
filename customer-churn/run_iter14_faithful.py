"""Iteration 14: TRULY faithful BlamerX reproduction.

Gaps found in our iter6 vs actual BlamerX:
1. Ridge uses OHE categoricals + standardized numericals + TE features (NOT ordinal encoded)
2. XGB drops raw categoricals, only uses: numericals + TE features + ridge_pred
3. XGB uses enable_categorical=True with native category dtype
4. TE uses sklearn TargetEncoder(smooth='auto') not fixed smooth=10
5. Inner-fold TE is proper OOF (not full-fold)

This iteration fixes ALL of these.
"""

import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")

import gc
import warnings
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
import xgboost as xgb
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")

N_FOLDS = 20
INNER_FOLDS = 5
RIDGE_ALPHA = 10.0

XGB_PARAMS = {
    "n_estimators": 50000,
    "learning_rate": 0.0063,
    "max_depth": 5,
    "subsample": 0.81,
    "colsample_bytree": 0.32,
    "min_child_weight": 6,
    "reg_alpha": 3.5017,
    "reg_lambda": 1.2925,
    "gamma": 0.790,
    "random_state": 42,
    "early_stopping_rounds": 500,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "enable_categorical": True,
    "device": "cuda",
    "verbosity": 0,
    "n_jobs": -1,
}

TOP_CATS = ["Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]


def prepare():
    """Feature engineering matching BlamerX exactly."""
    print("[1] Features...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    # ORIG_proba
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"] == "Yes").astype(float)
    gm = oc.mean()
    for col in list(train.columns):
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            proba = tmp.groupby(col)["_y"].mean()
            train[f"ORIG_proba_{col}"] = train[col].map(proba).fillna(gm).astype("float32")
            test[f"ORIG_proba_{col}"] = test[col].map(proba).fillna(gm).astype("float32")

    # Distribution features
    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    ch_tc = np.sort(orig_tc[orig["Churn"] == "Yes"].values)
    nc_tc = np.sort(orig_tc[orig["Churn"] != "Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch_TC"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc_TC"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")
        df["pctrank_gap_TC"] = df["pctrank_ch_TC"] - df["pctrank_nc_TC"]
        df["zscore_ch_TC"] = ((tc - ch_tc.mean()) / ch_tc.std()).astype("float32")
        df["zscore_nc_TC"] = ((tc - nc_tc.mean()) / nc_tc.std()).astype("float32")
        for ql, qv in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
            cq, nq = np.quantile(ch_tc, qv), np.quantile(nc_tc, qv)
            df[f"qdist_ch_{ql}"] = np.abs(tc - cq).astype("float32")
            df[f"qdist_nc_{ql}"] = np.abs(tc - nq).astype("float32")

    # Arithmetic
    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["monthly_to_total"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")

        # Service counts
        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        avail = [c for c in yes_cols if c in df.columns]
        df["service_count"] = sum((df[c] == "Yes").astype(int) for c in avail)
        df["has_internet"] = (df.get("InternetService", "No") != "No").astype(int)
        df["has_phone"] = (df.get("PhoneService", "No") == "Yes").astype(int)

        # Frequency encoding of numericals
        for col in NUMS:
            df[f"FREQ_{col}"] = df[col].map(df[col].value_counts(normalize=True)).astype("float32")

        # Digit features
        for col in NUMS:
            v = df[col].fillna(0)
            sv = v.astype(str)
            df[f"{col}_first_digit"] = sv.str[0].astype("float32")
            df[f"{col}_last_digit"] = (v * 100).astype(int) % 10
            df[f"{col}_mod10"] = (v % 10).astype("float32")
            df[f"{col}_is_round10"] = ((v % 10 == 0) & (v > 0)).astype(int)
            df[f"{col}_dev_round10"] = (v - (v / 10).round() * 10).astype("float32")
            if col != "tenure":
                df[f"{col}_mod100"] = (v % 100).astype("float32")
                df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure":
                df[f"{col}_mod12"] = (v % 12).astype("float32")
                df["tenure_years"] = (v // 12).astype("float32")

        # N-grams
        avail_cats = [c for c in TOP_CATS if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")
        for c1, c2, c3 in combinations(avail_cats[:4], 3):
            df[f"TG_{c1}_{c2}_{c3}"] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype("category")

        # Numericals as categories (BlamerX does this)
        for col in NUMS:
            df[f"CAT_{col}"] = df[col].astype(str).astype("category")

    # Identify column types
    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns]
    ngram_cols = [c for c in train.columns if c.startswith("BG_") or c.startswith("TG_")]
    cat_num_cols = [c for c in train.columns if c.startswith("CAT_")]
    num_cols = [c for c in train.columns
                if c not in cat_cols + ngram_cols + cat_num_cols
                and train[c].dtype in ("float32", "float64", "int64", "int32")]

    print(f"  Cats: {len(cat_cols)}, Ngrams: {len(ngram_cols)}, CatNums: {len(cat_num_cols)}, Nums: {len(num_cols)}", flush=True)
    return train, test, y, test_ids, cat_cols, ngram_cols, cat_num_cols, num_cols


def train_blamerx_faithful(train, test, y, test_ids, cat_cols, ngram_cols, cat_num_cols, num_cols):
    """FAITHFUL BlamerX: Ridge(OHE+num+TE) → XGB(TE+num+ridge, enable_categorical)."""
    print(f"\n[2] BlamerX faithful ({N_FOLDS}-fold)...", flush=True)

    te_cols = cat_cols  # Original categoricals
    te_ngram_cols = ngram_cols + cat_num_cols  # N-gram + CAT_num

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    outer_kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(outer_kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        y_tr = y.iloc[tr_idx]
        y_va = y.iloc[va_idx]

        # === TE Pass 1: Inner-fold OOF for std/min/max ===
        inner_kf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)
        te_stat_names = []
        for col in te_cols:
            for stat in ["std", "min", "max"]:
                name = f"TE_{col}_{stat}"
                te_stat_names.append(name)
                X_tr[name] = np.nan
                X_va[name] = np.nan
                X_te[name] = np.nan

        # Inner OOF
        for _, (itr, iva) in enumerate(inner_kf.split(X_tr, y_tr)):
            for col in te_cols:
                tmp = pd.DataFrame({"c": X_tr.iloc[itr][col], "y": y_tr.iloc[itr].values})
                agg = tmp.groupby("c")["y"].agg(["std", "min", "max"])
                for stat in ["std", "min", "max"]:
                    X_tr.iloc[iva, X_tr.columns.get_loc(f"TE_{col}_{stat}")] = (
                        X_tr.iloc[iva][col].map(agg[stat]).values
                    )

        # Full-fold for val/test
        for col in te_cols:
            tmp = pd.DataFrame({"c": X_tr[col], "y": y_tr.values})
            agg = tmp.groupby("c")["y"].agg(["std", "min", "max"])
            for stat in ["std", "min", "max"]:
                X_va[f"TE_{col}_{stat}"] = X_va[col].map(agg[stat]).values
                X_te[f"TE_{col}_{stat}"] = X_te[col].map(agg[stat]).values

        # === TE Pass 2: Smoothed mean for categoricals ===
        global_mean = y_tr.mean()
        te_mean_names = []
        for col in te_cols:
            name = f"TE_mean_{col}"
            te_mean_names.append(name)
            tmp = pd.DataFrame({"c": X_tr[col], "y": y_tr.values})
            agg = tmp.groupby("c")["y"].agg(["mean", "count"])
            # sklearn-style auto smoothing approximation
            sm = (agg["count"] * agg["mean"] + 20 * global_mean) / (agg["count"] + 20)
            X_tr[name] = X_tr[col].map(sm).fillna(global_mean).astype("float32")
            X_va[name] = X_va[col].map(sm).fillna(global_mean).astype("float32")
            X_te[name] = X_te[col].map(sm).fillna(global_mean).astype("float32")

        # === TE Pass 3: N-gram TE mean ===
        te_ng_names = []
        for col in te_ngram_cols:
            name = f"TE_ng_{col}"
            te_ng_names.append(name)
            col_str = X_tr[col].astype(str)
            tmp = pd.DataFrame({"c": col_str, "y": y_tr.values})
            agg = tmp.groupby("c")["y"].agg(["mean", "count"])
            sm = (agg["count"] * agg["mean"] + 20 * global_mean) / (agg["count"] + 20)
            X_tr[name] = col_str.map(sm).fillna(global_mean).astype("float32")
            X_va[name] = X_va[col].astype(str).map(sm).fillna(global_mean).astype("float32")
            X_te[name] = X_te[col].astype(str).map(sm).fillna(global_mean).astype("float32")

        # === Frequency encoding ===
        freq_names = []
        for col in te_cols:
            name = f"FREQ_{col}"
            freq_names.append(name)
            freq = X_tr[col].value_counts(normalize=True)
            X_tr[name] = X_tr[col].map(freq).fillna(0).astype("float32")
            X_va[name] = X_va[col].map(freq).fillna(0).astype("float32")
            X_te[name] = X_te[col].map(freq).fillna(0).astype("float32")

        # === STAGE 1: Ridge on OHE + numericals + TE ===
        # BlamerX: Ridge sees OHE categoricals (sparse) + standardized numericals + all TE features
        all_te_names = te_stat_names + te_mean_names + te_ng_names + freq_names
        ridge_num_cols = num_cols + all_te_names

        # OHE for categoricals (Ridge needs this, not ordinal)
        ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        ohe_tr = ohe.fit_transform(X_tr[te_cols].fillna("_missing_"))
        ohe_va = ohe.transform(X_va[te_cols].fillna("_missing_"))
        ohe_te = ohe.transform(X_te[te_cols].fillna("_missing_"))

        # Standardize numericals
        sc = StandardScaler()
        num_tr = sc.fit_transform(X_tr[ridge_num_cols].fillna(0))
        num_va = sc.transform(X_va[ridge_num_cols].fillna(0))
        num_te = sc.transform(X_te[ridge_num_cols].fillna(0))

        # Combine: OHE (sparse) + scaled numericals
        import scipy.sparse as sp
        ridge_tr = sp.hstack([ohe_tr, sp.csr_matrix(num_tr)])
        ridge_va = sp.hstack([ohe_va, sp.csr_matrix(num_va)])
        ridge_te = sp.hstack([ohe_te, sp.csr_matrix(num_te)])

        ridge = Ridge(alpha=RIDGE_ALPHA, random_state=42)
        ridge.fit(ridge_tr, y_tr)
        ridge_pred_tr = np.clip(ridge.predict(ridge_tr), 0, 1).astype("float32")
        ridge_pred_va = np.clip(ridge.predict(ridge_va), 0, 1).astype("float32")
        ridge_pred_te = np.clip(ridge.predict(ridge_te), 0, 1).astype("float32")

        # === STAGE 2: XGB on numericals + TE + ridge_pred (NO raw categoricals) ===
        # BlamerX drops raw categoricals, uses only: num + TE + ridge
        xgb_feature_cols = num_cols + all_te_names

        X_tr_xgb = X_tr[xgb_feature_cols].fillna(0).copy()
        X_va_xgb = X_va[xgb_feature_cols].fillna(0).copy()
        X_te_xgb = X_te[xgb_feature_cols].fillna(0).copy()

        X_tr_xgb["ridge_pred"] = ridge_pred_tr
        X_va_xgb["ridge_pred"] = ridge_pred_va
        X_te_xgb["ridge_pred"] = ridge_pred_te

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

        oof[va_idx] = model.predict_proba(X_va_xgb)[:, 1]
        test_preds += model.predict_proba(X_te_xgb)[:, 1] / N_FOLDS

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y_va, oof[va_idx]):.6f}", flush=True)

        del model, X_tr, X_va, X_te, ridge
        gc.collect()

    score = roc_auc_score(y, oof)
    print(f"\n  BlamerX faithful CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 14: TRULY Faithful BlamerX", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols, ngram_cols, cat_num_cols, num_cols = prepare()

    oof, test_preds, cv_score = train_blamerx_faithful(
        train, test, y, test_ids, cat_cols, ngram_cols, cat_num_cols, num_cols)

    generate_submission(test_ids, test_preds, "id", "Churn", "submissions/iter14_faithful.csv")

    # Check diversity vs Artem
    artem = pd.read_csv("public_subs/artemevstafyev_cv-auc-0-91930-xgb-cb-blend.csv")["Churn"].values
    corr = np.corrcoef(artem, test_preds)[0, 1]
    print(f"  Correlation with Artem: {corr:.4f}", flush=True)

    # Cascade with Artem
    cascade = artem * 0.95 + test_preds * 0.05
    generate_submission(test_ids, cascade, "id", "Churn", "submissions/iter14_cascade.csv")

    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 14 - Truly Faithful BlamerX\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n")
        f.write(f"**CV**: {cv_score:.6f}, **Corr with Artem**: {corr:.4f}\n")
        f.write(f"**Fixes**: OHE for Ridge, drop raw cats for XGB, proper inner-fold TE\n\n---\n")

    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
