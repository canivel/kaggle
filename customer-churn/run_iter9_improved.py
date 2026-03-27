"""Iteration 9: Improved BlamerX + RealMLP diversity.

Key learnings applied:
- Seed diversity is FAKE diversity (LB 0.91580 < 0.91603)
- Only TRUE architecture diversity helps on LB
- 20-fold BlamerX generalizes best
- Focus: improve the single best approach + add ONE truly diverse model

Approach:
1. BlamerX XGB with 20-fold (our best: LB 0.91603)
2. BlamerX LGBM with 20-fold (different algo, same framework)
3. RealMLP (neural, truly diverse architecture)
4. Simple 1/3 average of 3 DIVERSE models (Deotte's winning formula)
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
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")

N_FOLDS = 20
TOP_CATS = ["Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]


def prepare_data():
    """Full feature engineering."""
    print("[1] Loading & engineering features...", flush=True)
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
    for col in train.columns:
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            proba = tmp.groupby(col)["_y"].mean()
            train[f"ORIG_{col}"] = train[col].map(proba).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(proba).fillna(gm).astype("float32")

    # Distribution + quantile distance
    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    ch_tc = np.sort(orig_tc[orig["Churn"] == "Yes"].values)
    nc_tc = np.sort(orig_tc[orig["Churn"] != "Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")
        df["pctrank_gap"] = df["pctrank_ch"] - df["pctrank_nc"]
        df["zscore_ch"] = ((tc - ch_tc.mean()) / ch_tc.std()).astype("float32")
        df["zscore_nc"] = ((tc - nc_tc.mean()) / nc_tc.std()).astype("float32")
        for ql, qv in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
            cq = np.quantile(ch_tc, qv)
            nq = np.quantile(nc_tc, qv)
            df[f"qdist_ch_{ql}"] = np.abs(tc - cq).astype("float32")
            df[f"qdist_nc_{ql}"] = np.abs(tc - nq).astype("float32")
            df[f"qdist_gap_{ql}"] = (df[f"qdist_nc_{ql}"] - df[f"qdist_ch_{ql}"]).astype("float32")

    # Static features
    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["tenure_x_mc"] = (df["tenure"] * df["MonthlyCharges"]).astype("float32")
        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        avail = [c for c in yes_cols if c in df.columns]
        df["svc_count"] = sum((df[c] == "Yes").astype(int) for c in avail)
        df["has_internet"] = (df.get("InternetService", "No") != "No").astype(int)
        for col in NUMS:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v % 10).astype("float32")
            df[f"{col}_m100"] = (v % 100).astype("float32")
            df[f"{col}_d1"] = v.astype(str).str[0].astype("float32")
            df[f"{col}_ld"] = (v * 100).astype(int) % 10
            df[f"{col}_is_round"] = ((v % 10 == 0) & (v > 0)).astype(int)
            if col != "tenure":
                df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
                df[f"{col}_dev_r10"] = (v - (v / 10).round() * 10).astype("float32")
            if col == "tenure":
                df[f"{col}_m12"] = (v % 12).astype("float32")
                df["tenure_years"] = (v // 12).astype("float32")
        avail_cats = [c for c in TOP_CATS if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
        for c1, c2, c3 in combinations(avail_cats[:4], 3):
            df[f"TG_{c1}_{c2}_{c3}"] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                if "BG_" not in c and "TG_" not in c]
    ngram_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                  if "BG_" in c or "TG_" in c]
    print(f"  Features: {train.shape[1]}, Cats: {len(cat_cols)}, Ngrams: {len(ngram_cols)}", flush=True)
    return train, test, y, test_ids, cat_cols, ngram_cols


def fold_process(X_tr, X_va, X_te, y_tr, cat_cols, ngram_cols, use_ridge=True):
    """Full in-fold encoding + Ridge stage 1."""
    all_cats = cat_cols + ngram_cols
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    X_tr[all_cats] = oe.fit_transform(X_tr[all_cats])
    X_va[all_cats] = oe.transform(X_va[all_cats])
    X_te[all_cats] = oe.transform(X_te[all_cats])

    # Smoothed TE
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

    # TE std/min/max
    for col in cat_cols[:6]:
        for stat in ["std", "min", "max"]:
            tmp = pd.DataFrame({"c": X_tr[col], "y": y_tr.values})
            agg = tmp.groupby("c")["y"].agg(stat)
            name = f"{col}_te_{stat}"
            X_tr[name] = X_tr[col].map(agg).fillna(0).astype("float32")
            X_va[name] = X_va[col].map(agg).fillna(0).astype("float32")
            X_te[name] = X_te[col].map(agg).fillna(0).astype("float32")

    # Frequency
    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[f"{col}_f"] = X_tr[col].map(freq).fillna(0).astype("float32")
        X_va[f"{col}_f"] = X_va[col].map(freq).fillna(0).astype("float32")
        X_te[f"{col}_f"] = X_te[col].map(freq).fillna(0).astype("float32")

    if use_ridge:
        sc = StandardScaler()
        r = Ridge(alpha=10.0)
        r.fit(sc.fit_transform(X_tr), y_tr)
        X_tr["ridge"] = np.clip(r.predict(sc.transform(X_tr)), 0, 1).astype("float32")
        X_va["ridge"] = np.clip(r.predict(sc.transform(X_va)), 0, 1).astype("float32")
        X_te["ridge"] = np.clip(r.predict(sc.transform(X_te)), 0, 1).astype("float32")

    X_tr = X_tr.fillna(0)
    X_va = X_va.fillna(0)
    X_te = X_te.fillna(0)
    return X_tr, X_va, X_te


def train_model_20fold(name, model_cls, model_params, train_df, test_df, y, cat_cols, ngram_cols):
    """20-fold CV with in-fold encoding + Ridge."""
    print(f"\n[{name}] 20-fold...", flush=True)
    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
        X_tr = train_df.iloc[tr_idx].copy()
        X_va = train_df.iloc[va_idx].copy()
        X_te = test_df.copy()
        X_tr, X_va, X_te = fold_process(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)

        if model_cls == "xgb":
            m = xgb.XGBClassifier(**model_params)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        elif model_cls == "lgbm":
            m = lgb.LGBMClassifier(**model_params)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(300, verbose=False)])

        oof[va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds += m.predict_proba(X_te)[:, 1] / N_FOLDS

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
        del m, X_tr, X_va, X_te; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"  {name} CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def train_realmlp_20fold(train_df, test_df, y, cat_cols, ngram_cols):
    """RealMLP from pytabkit - true neural diversity."""
    print(f"\n[RealMLP] 20-fold...", flush=True)

    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
        X_tr = train_df.iloc[tr_idx].copy()
        X_va = train_df.iloc[va_idx].copy()
        X_te = test_df.copy()
        X_tr, X_va, X_te = fold_process(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols, use_ridge=False)

        from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier

        m = RealMLP_TD_Classifier(
            n_epochs=3,
            batch_size=256,
            n_ens=4,
            verbosity=0,
            device="cuda",
        )
        m.fit(X_tr.values.astype(np.float32), y.iloc[tr_idx].values)
        oof[va_idx] = m.predict_proba(X_va.values.astype(np.float32))[:, 1]
        test_preds += m.predict_proba(X_te.values.astype(np.float32))[:, 1] / N_FOLDS

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
        del m; gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score = roc_auc_score(y, oof)
    print(f"  RealMLP CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 9: Improved BlamerX + RealMLP (Deotte-style 3 models)", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols, ngram_cols = prepare_data()

    results = {}

    # Model 1: BlamerX XGB (our best LB approach)
    xgb_params = dict(
        n_estimators=50000, learning_rate=0.0063, max_depth=5,
        min_child_weight=6, subsample=0.81, colsample_bytree=0.32,
        reg_alpha=3.5017, reg_lambda=1.2925, gamma=0.790,
        early_stopping_rounds=500, device="cuda",
        random_state=42, verbosity=0, n_jobs=-1,
    )
    xgb_oof, xgb_test, xgb_cv = train_model_20fold(
        "BlamerX_XGB", "xgb", xgb_params, train, test, y, cat_cols, ngram_cols)
    results["xgb"] = {"oof": xgb_oof, "test": xgb_test, "cv": xgb_cv}

    # Model 2: LGBM with same 20-fold framework (different algo = diversity)
    lgbm_params = dict(
        n_estimators=10000, learning_rate=0.01, num_leaves=31,
        max_depth=5, subsample=0.8, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=2.0, min_child_samples=50,
        random_state=42, verbosity=-1, n_jobs=4,
    )
    lgbm_oof, lgbm_test, lgbm_cv = train_model_20fold(
        "BlamerX_LGBM", "lgbm", lgbm_params, train, test, y, cat_cols, ngram_cols)
    results["lgbm"] = {"oof": lgbm_oof, "test": lgbm_test, "cv": lgbm_cv}

    # Model 3: RealMLP (TRUE neural diversity)
    try:
        mlp_oof, mlp_test, mlp_cv = train_realmlp_20fold(train, test, y, cat_cols, ngram_cols)
        results["realmlp"] = {"oof": mlp_oof, "test": mlp_test, "cv": mlp_cv}
    except Exception as e:
        print(f"  RealMLP failed: {e}", flush=True)

    # === ENSEMBLES (Deotte style: equal weight diverse models) ===
    print("\n=== ENSEMBLES ===", flush=True)

    # XGB standalone (our current best LB approach)
    generate_submission(test_ids, xgb_test, "id", "Churn", "submissions/iter9_xgb.csv")

    # 2-model: XGB + LGBM (same family but different algo)
    blend2 = (xgb_test + lgbm_test) / 2
    blend2_oof = (xgb_oof + lgbm_oof) / 2
    print(f"  XGB+LGBM avg: {roc_auc_score(y, blend2_oof):.6f}", flush=True)
    generate_submission(test_ids, blend2, "id", "Churn", "submissions/iter9_xgb_lgbm.csv")

    if "realmlp" in results:
        # 3-model Deotte-style (1/3 each)
        blend3 = (xgb_test + lgbm_test + results["realmlp"]["test"]) / 3
        blend3_oof = (xgb_oof + lgbm_oof + results["realmlp"]["oof"]) / 3
        print(f"  XGB+LGBM+RealMLP (Deotte): {roc_auc_score(y, blend3_oof):.6f}", flush=True)
        generate_submission(test_ids, blend3, "id", "Churn", "submissions/iter9_deotte3.csv")

        # Rank average (most robust)
        rank3 = np.mean([
            rankdata(xgb_test) / len(xgb_test),
            rankdata(lgbm_test) / len(lgbm_test),
            rankdata(results["realmlp"]["test"]) / len(test),
        ], axis=0)
        generate_submission(test_ids, rank3, "id", "Churn", "submissions/iter9_rank3.csv")

    # Blend with iter6 (if different enough)
    try:
        iter6 = pd.read_csv("submissions/iter6_blamerx.csv")["Churn"].values
        # Average new XGB with iter6 (essentially 2 independent BlamerX runs)
        avg_i6 = (xgb_test + iter6) / 2
        generate_submission(test_ids, avg_i6, "id", "Churn", "submissions/iter9_xgb_i6.csv")
        print(f"  XGB + iter6 avg saved", flush=True)
    except Exception:
        pass

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 9 - Improved BlamerX + RealMLP\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        for name, r in results.items():
            f.write(f"- {name}: {r['cv']:.6f}\n")
        f.write(f"- XGB+LGBM avg: {roc_auc_score(y, blend2_oof):.6f}\n")
        if "realmlp" in results:
            f.write(f"- **Deotte 3-model**: {roc_auc_score(y, blend3_oof):.6f}\n")
        f.write(f"\n---\n")

    print("\n" + "=" * 70, flush=True)
    print("ITERATION 9 COMPLETE", flush=True)
    for name, r in results.items():
        print(f"  {name}: {r['cv']:.6f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
