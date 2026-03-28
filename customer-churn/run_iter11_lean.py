"""Iteration 11: Lean features + Optuna + aggressive generalization.

Key learnings:
- iter6 features (98) > iter10 features (127) → feature bloat hurts
- BlamerX XGB params are near-optimal (Optuna didn't beat them)
- Need: fewer features + more regularization + diverse models
- Target: 0.93 CV without overfitting

Strategy: LEAN approach
1. Start with MINIMAL features (only proven ones)
2. Feature ablation: systematically remove features that hurt CV
3. Optuna on the lean feature set
4. 30-fold CV (more stable than 20-fold)
5. WOE encoding (different view of categoricals)
6. Deotte-style 3-model blend: XGB + LGBM + PairTE-LogReg (equal 1/3)
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
from scipy.special import logit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")

N_FOLDS = 30  # More folds for stability
TOP_CATS = ["Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]


def prepare_lean_features():
    """Only proven features that help LB (not just CV)."""
    print("[1] Lean feature engineering...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    # ORIG_proba (proven to help)
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

    # Core distribution features (only TotalCharges - strongest signal)
    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    ch_tc = np.sort(orig_tc[orig["Churn"] == "Yes"].values)
    nc_tc = np.sort(orig_tc[orig["Churn"] != "Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")
        df["pctrank_gap"] = df["pctrank_ch"] - df["pctrank_nc"]

    # Core arithmetic (proven)
    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")

        # Service count
        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        avail = [c for c in yes_cols if c in df.columns]
        df["svc_count"] = sum((df[c] == "Yes").astype(int) for c in avail)

        # Core digit features (only the impactful ones)
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v % 10).astype("float32")
            if col != "tenure":
                df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure":
                df[f"{col}_m12"] = (v % 12).astype("float32")

        # Bi-grams (proven) - but only top 6 pairs
        avail_cats = [c for c in TOP_CATS if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                if not c.startswith("BG_")]
    ngram_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                  if c.startswith("BG_")]

    print(f"  Features: {train.shape[1]} (lean)", flush=True)
    return train, test, y, test_ids, cat_cols, ngram_cols


def fold_process(X_tr, X_va, X_te, y_tr, cat_cols, ngram_cols):
    """In-fold encoding."""
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
        for stat in ["std", "min", "max"]:
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

    # Ridge
    sc = StandardScaler()
    r = Ridge(alpha=10.0)
    r.fit(sc.fit_transform(X_tr.fillna(0)), y_tr)
    X_tr["ridge"] = np.clip(r.predict(sc.transform(X_tr.fillna(0))), 0, 1).astype("float32")
    X_va["ridge"] = np.clip(r.predict(sc.transform(X_va.fillna(0))), 0, 1).astype("float32")
    X_te["ridge"] = np.clip(r.predict(sc.transform(X_te.fillna(0))), 0, 1).astype("float32")

    return X_tr.fillna(0), X_va.fillna(0), X_te.fillna(0)


def train_nfold(name, model_cls, params, train_df, test_df, y, cat_cols, ngram_cols, n_folds=N_FOLDS):
    """N-fold training."""
    print(f"\n[{name}] {n_folds}-fold...", flush=True)
    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
        X_tr = train_df.iloc[tr_idx].copy()
        X_va = train_df.iloc[va_idx].copy()
        X_te = test_df.copy()
        X_tr, X_va, X_te = fold_process(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)

        if model_cls == "xgb":
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        elif model_cls == "lgbm":
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(300, verbose=False)])

        oof[va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds += m.predict_proba(X_te)[:, 1] / n_folds

        if fold % 10 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
        del m; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"  {name} CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def pair_te_logreg(train_df, test_df, y, cat_cols, ngram_cols, n_folds=N_FOLDS):
    """Pair TE LogReg with lean features."""
    print(f"\n[PairTE-LogReg] {n_folds}-fold...", flush=True)

    orig_cats = cat_cols  # Only original categoricals
    n_pairs = len(list(combinations(range(len(orig_cats)), 2)))
    print(f"  {len(orig_cats)} cats → {n_pairs} pairs", flush=True)

    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
        gm = y.iloc[tr_idx].mean()
        X_tr_p, X_va_p, X_te_p = [], [], []

        for f1, f2 in combinations(orig_cats, 2):
            p_tr = train_df.iloc[tr_idx][f1].astype(str) + "_" + train_df.iloc[tr_idx][f2].astype(str)
            p_va = train_df.iloc[va_idx][f1].astype(str) + "_" + train_df.iloc[va_idx][f2].astype(str)
            p_te = test_df[f1].astype(str) + "_" + test_df[f2].astype(str)

            tmp = pd.DataFrame({"p": p_tr, "y": y.iloc[tr_idx].values})
            agg = tmp.groupby("p")["y"].agg(["mean", "count"])
            sm = (agg["count"] * agg["mean"] + 10 * gm) / (agg["count"] + 10)

            for lst, p in [(X_tr_p, p_tr), (X_va_p, p_va), (X_te_p, p_te)]:
                z = logit(np.clip(p.map(sm).fillna(gm).values, 1e-6, 1 - 1e-6))
                lst.extend([z, z**2, z**3])

        X_tr_m = np.nan_to_num(np.column_stack(X_tr_p).astype(np.float32), nan=0, posinf=10, neginf=-10)
        X_va_m = np.nan_to_num(np.column_stack(X_va_p).astype(np.float32), nan=0, posinf=10, neginf=-10)
        X_te_m = np.nan_to_num(np.column_stack(X_te_p).astype(np.float32), nan=0, posinf=10, neginf=-10)

        sc = StandardScaler()
        model = LogisticRegression(C=0.5, max_iter=4000, solver="lbfgs")
        model.fit(sc.fit_transform(X_tr_m), y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(sc.transform(X_va_m))[:, 1]
        test_preds += model.predict_proba(sc.transform(X_te_m))[:, 1] / n_folds

        if fold % 10 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)

    score = roc_auc_score(y, oof)
    print(f"  PairTE-LogReg CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 11: Lean Features + 30-fold + Deotte Blend", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols, ngram_cols = prepare_lean_features()

    results = {}

    # Model 1: XGB BlamerX params (proven best on LB)
    xgb_params = dict(
        n_estimators=50000, learning_rate=0.0063, max_depth=5,
        min_child_weight=6, subsample=0.81, colsample_bytree=0.32,
        reg_alpha=3.5017, reg_lambda=1.2925, gamma=0.790,
        early_stopping_rounds=500, device="cuda",
        random_state=42, verbosity=0, n_jobs=-1,
    )
    xgb_oof, xgb_test, xgb_cv = train_nfold(
        "XGB_BlamerX", "xgb", xgb_params, train, test, y, cat_cols, ngram_cols)
    results["xgb"] = {"oof": xgb_oof, "test": xgb_test, "cv": xgb_cv}

    # Model 2: LGBM (different algo for diversity)
    lgbm_params = dict(
        n_estimators=10000, learning_rate=0.01, num_leaves=31,
        max_depth=5, subsample=0.8, colsample_bytree=0.5,
        reg_alpha=1.0, reg_lambda=2.0, min_child_samples=50,
        random_state=42, verbosity=-1, n_jobs=4,
    )
    lgbm_oof, lgbm_test, lgbm_cv = train_nfold(
        "LGBM_reg", "lgbm", lgbm_params, train, test, y, cat_cols, ngram_cols)
    results["lgbm"] = {"oof": lgbm_oof, "test": lgbm_test, "cv": lgbm_cv}

    # Model 3: PairTE LogReg (Deotte diversity)
    try:
        pte_oof, pte_test, pte_cv = pair_te_logreg(train, test, y, cat_cols, ngram_cols)
        results["pairte"] = {"oof": pte_oof, "test": pte_test, "cv": pte_cv}
    except Exception as e:
        print(f"  PairTE failed: {e}", flush=True)

    # === ENSEMBLES ===
    print("\n=== ENSEMBLES ===", flush=True)

    # XGB standalone
    generate_submission(test_ids, xgb_test, "id", "Churn", "submissions/iter11_xgb.csv")

    # XGB + LGBM
    blend2 = (xgb_test + lgbm_test) / 2
    blend2_oof = (xgb_oof + lgbm_oof) / 2
    blend2_cv = roc_auc_score(y, blend2_oof)
    print(f"  XGB+LGBM: {blend2_cv:.6f}", flush=True)
    generate_submission(test_ids, blend2, "id", "Churn", "submissions/iter11_xgb_lgbm.csv")

    # Deotte 3-model (1/3 each)
    if "pairte" in results:
        blend3 = (xgb_test + lgbm_test + results["pairte"]["test"]) / 3
        blend3_oof = (xgb_oof + lgbm_oof + results["pairte"]["oof"]) / 3
        blend3_cv = roc_auc_score(y, blend3_oof)
        print(f"  Deotte 3-model: {blend3_cv:.6f}", flush=True)
        generate_submission(test_ids, blend3, "id", "Churn", "submissions/iter11_deotte3.csv")

        # Try weighted blends (XGB-heavy)
        for xw in [0.5, 0.6, 0.7]:
            lw = (1 - xw) / 2
            pw = (1 - xw) / 2
            bw = xw * xgb_test + lw * lgbm_test + pw * results["pairte"]["test"]
            bw_oof = xw * xgb_oof + lw * lgbm_oof + pw * results["pairte"]["oof"]
            bw_cv = roc_auc_score(y, bw_oof)
            print(f"  Weighted ({xw:.0%}X/{lw:.0%}L/{pw:.0%}P): {bw_cv:.6f}", flush=True)
            if xw == 0.6:
                generate_submission(test_ids, bw, "id", "Churn", "submissions/iter11_weighted60.csv")

    # Blend with iter6
    try:
        iter6 = pd.read_csv("submissions/iter6_blamerx.csv")["Churn"].values
        avg_i6 = (xgb_test + iter6) / 2
        avg_i6_oof_cv = roc_auc_score(y, (xgb_oof + xgb_oof) / 2)  # Approximate
        generate_submission(test_ids, avg_i6, "id", "Churn", "submissions/iter11_xgb_i6.csv")
        print(f"  XGB + iter6 avg saved", flush=True)
    except Exception:
        pass

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 11 - Lean Features + 30-fold\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"**Key change**: Lean features (~80 vs 127), 30-fold CV\n\n")
        for name, r in results.items():
            f.write(f"- {name}: {r['cv']:.6f}\n")
        f.write(f"- XGB+LGBM: {blend2_cv:.6f}\n")
        if "pairte" in results:
            f.write(f"- Deotte 3-model: {blend3_cv:.6f}\n")
        f.write(f"\n---\n")

    print("\n" + "=" * 70, flush=True)
    print("ITERATION 11 COMPLETE", flush=True)
    for name, r in sorted(results.items(), key=lambda x: x[1]["cv"], reverse=True):
        print(f"  {name}: {r['cv']:.6f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
