"""Iteration 15: Fix iter14 - keep categoricals for XGB with enable_categorical.

Iter14 dropped raw categoricals → CV dropped to 0.9152.
iter6 kept everything ordinal-encoded → CV 0.9188.

The truth: BlamerX uses enable_categorical=True which means XGB
handles categoricals NATIVELY as category dtype. Not dropping them,
not ordinal encoding them.

Fix: Keep cats as category dtype for XGB + all TE features + Ridge pred.
"""

import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")

import gc, warnings, datetime
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
import scipy.sparse as sp
import xgboost as xgb
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")

N_FOLDS = 20
INNER_FOLDS = 5

XGB_PARAMS = {
    "n_estimators": 50000, "learning_rate": 0.0063, "max_depth": 5,
    "subsample": 0.81, "colsample_bytree": 0.32, "min_child_weight": 6,
    "reg_alpha": 3.5017, "reg_lambda": 1.2925, "gamma": 0.790,
    "early_stopping_rounds": 500, "enable_categorical": True,
    "device": "cuda", "random_state": 42, "verbosity": 0, "n_jobs": -1,
}

TOP_CATS = ["Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]


def prepare():
    """Same feature engineering as iter14."""
    print("[1] Features...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"] == "Yes").astype(float); gm = oc.mean()
    for col in list(train.columns):
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            train[f"ORIG_{col}"] = train[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")

    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    ch_tc = np.sort(orig_tc[orig["Churn"] == "Yes"].values)
    nc_tc = np.sort(orig_tc[orig["Churn"] != "Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")
        df["pctrank_gap"] = df["pctrank_ch"] - df["pctrank_nc"]

    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        df["svc_count"] = sum((df[c] == "Yes").astype(int) for c in yes_cols if c in df.columns)
        df["has_internet"] = (df.get("InternetService", "No") != "No").astype(int)
        for col in NUMS:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v % 10).astype("float32")
            df[f"{col}_d1"] = v.astype(str).str[0].astype("float32")
            df[f"{col}_round10"] = ((v % 10 == 0) & (v > 0)).astype(int)
            if col != "tenure": df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure": df[f"{col}_m12"] = (v % 12).astype("float32")
        avail_cats = [c for c in TOP_CATS if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")
        for c1, c2, c3 in combinations(avail_cats[:4], 3):
            df[f"TG_{c1}_{c2}_{c3}"] = (df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)).astype("category")
        for col in NUMS:
            df[f"CAT_{col}"] = df[col].astype(str).astype("category")

    # Convert original categoricals to category dtype (for XGB enable_categorical)
    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns]
    for col in cat_cols:
        train[col] = train[col].astype("category")
        test[col] = test[col].astype("category")

    ngram_cols = [c for c in train.columns if c.startswith("BG_") or c.startswith("TG_")]
    cat_num_cols = [c for c in train.columns if c.startswith("CAT_")]
    all_cat_cols = cat_cols + ngram_cols + cat_num_cols
    num_cols = [c for c in train.columns if c not in all_cat_cols
                and train[c].dtype in ("float32", "float64", "int64", "int32")]

    print(f"  Total: {train.shape[1]}, Cats(category dtype): {len(all_cat_cols)}, Nums: {len(num_cols)}", flush=True)
    return train, test, y, test_ids, cat_cols, ngram_cols, cat_num_cols, num_cols


def main():
    print("=" * 70, flush=True)
    print("ITERATION 15: Fixed BlamerX (cats as category dtype for XGB)", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols, ngram_cols, cat_num_cols, num_cols = prepare()

    te_cols = cat_cols
    te_ngram_cols = ngram_cols + cat_num_cols
    all_cat = cat_cols + ngram_cols + cat_num_cols

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    outer_kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(outer_kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # TE: smoothed mean for all categoricals
        gm = y_tr.mean()
        te_names = []
        for col in te_cols + te_ngram_cols:
            name = f"TE_{col}"
            te_names.append(name)
            col_str = X_tr[col].astype(str)
            tmp = pd.DataFrame({"c": col_str, "y": y_tr.values})
            agg = tmp.groupby("c")["y"].agg(["mean", "count"])
            sm = (agg["count"] * agg["mean"] + 20 * gm) / (agg["count"] + 20)
            X_tr[name] = col_str.map(sm).fillna(gm).astype("float32")
            X_va[name] = X_va[col].astype(str).map(sm).fillna(gm).astype("float32")
            X_te[name] = X_te[col].astype(str).map(sm).fillna(gm).astype("float32")

        # TE stats (std only - most useful)
        te_stat_names = []
        for col in te_cols[:8]:
            name = f"TE_std_{col}"
            te_stat_names.append(name)
            tmp = pd.DataFrame({"c": X_tr[col].astype(str), "y": y_tr.values})
            agg = tmp.groupby("c")["y"].std().fillna(0)
            X_tr[name] = X_tr[col].astype(str).map(agg).fillna(0).astype("float32")
            X_va[name] = X_va[col].astype(str).map(agg).fillna(0).astype("float32")
            X_te[name] = X_te[col].astype(str).map(agg).fillna(0).astype("float32")

        # Frequency
        freq_names = []
        for col in te_cols:
            name = f"FRQ_{col}"
            freq_names.append(name)
            freq = X_tr[col].astype(str).value_counts(normalize=True)
            X_tr[name] = X_tr[col].astype(str).map(freq).fillna(0).astype("float32")
            X_va[name] = X_va[col].astype(str).map(freq).fillna(0).astype("float32")
            X_te[name] = X_te[col].astype(str).map(freq).fillna(0).astype("float32")

        # Ridge Stage 1 (OHE + nums + TE)
        ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        ohe_tr = ohe.fit_transform(X_tr[te_cols].astype(str))
        ohe_va = ohe.transform(X_va[te_cols].astype(str))
        ohe_te = ohe.transform(X_te[te_cols].astype(str))

        ridge_feats = num_cols + te_names + te_stat_names + freq_names
        sc = StandardScaler()
        num_tr = sc.fit_transform(X_tr[ridge_feats].fillna(0))
        num_va = sc.transform(X_va[ridge_feats].fillna(0))
        num_te = sc.transform(X_te[ridge_feats].fillna(0))

        ridge_tr = sp.hstack([ohe_tr, sp.csr_matrix(num_tr)])
        ridge_va = sp.hstack([ohe_va, sp.csr_matrix(num_va)])
        ridge_te = sp.hstack([ohe_te, sp.csr_matrix(num_te)])

        ridge = Ridge(alpha=10.0)
        ridge.fit(ridge_tr, y_tr)
        X_tr["ridge_pred"] = np.clip(ridge.predict(ridge_tr), 0, 1).astype("float32")
        X_va["ridge_pred"] = np.clip(ridge.predict(ridge_va), 0, 1).astype("float32")
        X_te["ridge_pred"] = np.clip(ridge.predict(ridge_te), 0, 1).astype("float32")

        # XGB Stage 2: ALL features including raw cats as category dtype + TE + ridge
        # This is the key: enable_categorical + category dtype = XGB handles cats natively
        xgb_cols = all_cat + num_cols + te_names + te_stat_names + freq_names + ["ridge_pred"]
        X_tr_xgb = X_tr[xgb_cols].copy()
        X_va_xgb = X_va[xgb_cols].copy()
        X_te_xgb = X_te[xgb_cols].copy()

        # Fill NaN in numeric columns only
        for col in X_tr_xgb.select_dtypes(include=["float32", "float64", "int64"]).columns:
            X_tr_xgb[col] = X_tr_xgb[col].fillna(0)
            X_va_xgb[col] = X_va_xgb[col].fillna(0)
            X_te_xgb[col] = X_te_xgb[col].fillna(0)

        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr_xgb, y_tr, eval_set=[(X_va_xgb, y_va)], verbose=False)

        oof[va_idx] = model.predict_proba(X_va_xgb)[:, 1]
        test_preds += model.predict_proba(X_te_xgb)[:, 1] / N_FOLDS

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y_va, oof[va_idx]):.6f}", flush=True)
        del model, ridge; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"\n  Iter15 CV: {score:.6f}", flush=True)

    # Diversity check
    artem = pd.read_csv("public_subs/artemevstafyev_cv-auc-0-91930-xgb-cb-blend.csv")["Churn"].values
    corr = np.corrcoef(artem, test_preds)[0, 1]
    print(f"  Corr with Artem: {corr:.4f}", flush=True)

    generate_submission(test_ids, test_preds, "id", "Churn", "submissions/iter15_fixed.csv")

    # Cascade
    cascade = artem * 0.95 + test_preds * 0.05
    generate_submission(test_ids, cascade, "id", "Churn", "submissions/iter15_cascade.csv")

    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 15 - Fixed BlamerX (cats as category dtype)\n")
        f.write(f"**CV**: {score:.6f}, **Corr**: {corr:.4f}\n\n---\n")

    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
