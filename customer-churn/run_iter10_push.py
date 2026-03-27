"""Iteration 10: Push toward 0.93 CV.

Current best: 0.918701. Target: 0.93. Gap: 0.0113.

Novel approaches to close the gap:
1. Optuna-tune XGB params on OUR exact feature set (20-fold)
2. Chris Deotte's pair target encoding (all C(19,2)=171 feature pairs → TE → logit3 → LogReg)
3. Multi-round pseudo-labeling with soft labels
4. Feature selection: drop features that hurt generalization
5. WOE encoding for categoricals
6. Deeper feature interactions (polynomial, cross-category)
"""

import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")

import gc
import warnings
import datetime
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import rankdata
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import optuna
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_FOLDS = 20
TOP_CATS = ["Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]


def prepare_full_features():
    """Maximum feature engineering."""
    print("[1] Full feature engineering...", flush=True)
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

    # ORIG_proba for ALL columns
    for col in train.columns:
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            proba = tmp.groupby(col)["_y"].mean()
            train[f"ORIG_{col}"] = train[col].map(proba).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(proba).fillna(gm).astype("float32")

    # Distribution features (TotalCharges, MonthlyCharges, tenure)
    for num_col in NUMS:
        if num_col not in orig.columns:
            continue
        orig_vals = pd.to_numeric(orig[num_col], errors="coerce").fillna(0)
        ch_vals = np.sort(orig_vals[orig["Churn"] == "Yes"].values)
        nc_vals = np.sort(orig_vals[orig["Churn"] != "Yes"].values)
        for df in [train, test]:
            v = df[num_col].values
            df[f"{num_col}_pctrank_ch"] = (np.searchsorted(ch_vals, v) / max(len(ch_vals), 1)).astype("float32")
            df[f"{num_col}_pctrank_nc"] = (np.searchsorted(nc_vals, v) / max(len(nc_vals), 1)).astype("float32")
            df[f"{num_col}_pctrank_gap"] = df[f"{num_col}_pctrank_ch"] - df[f"{num_col}_pctrank_nc"]
            # Z-scores
            if len(ch_vals) > 1 and ch_vals.std() > 0:
                df[f"{num_col}_zscore_ch"] = ((v - ch_vals.mean()) / ch_vals.std()).astype("float32")
            if len(nc_vals) > 1 and nc_vals.std() > 0:
                df[f"{num_col}_zscore_nc"] = ((v - nc_vals.mean()) / nc_vals.std()).astype("float32")
            # Quantile distances
            for ql, qv in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
                if len(ch_vals) > 0 and len(nc_vals) > 0:
                    cq = np.quantile(ch_vals, qv)
                    nq = np.quantile(nc_vals, qv)
                    df[f"{num_col}_qdist_ch_{ql}"] = np.abs(v - cq).astype("float32")
                    df[f"{num_col}_qdist_nc_{ql}"] = np.abs(v - nq).astype("float32")

    # Static features
    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["tenure_x_mc"] = (df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["tenure_sq"] = (df["tenure"] ** 2).astype("float32")
        df["mc_sq"] = (df["MonthlyCharges"] ** 2).astype("float32")

        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        avail = [c for c in yes_cols if c in df.columns]
        df["svc_count"] = sum((df[c] == "Yes").astype(int) for c in avail)
        df["has_internet"] = (df.get("InternetService", "No") != "No").astype(int)
        df["has_phone"] = (df.get("PhoneService", "No") == "Yes").astype(int)

        # Extended digit features
        for col in NUMS:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v % 10).astype("float32")
            df[f"{col}_m100"] = (v % 100).astype("float32")
            df[f"{col}_d1"] = v.astype(str).str[0].astype("float32")
            df[f"{col}_ld"] = (v * 100).astype(int) % 10
            df[f"{col}_nd"] = v.astype(str).str.replace(".", "", regex=False).str.len().astype("float32")
            df[f"{col}_round10"] = ((v % 10 == 0) & (v > 0)).astype(int)
            df[f"{col}_dev_r10"] = (v - (v / 10).round() * 10).astype("float32")
            if col != "tenure":
                df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure":
                df[f"{col}_m12"] = (v % 12).astype("float32")
                df["tenure_years"] = (v // 12).astype("float32")

        # N-grams
        avail_cats = [c for c in TOP_CATS if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
        for c1, c2, c3 in combinations(avail_cats[:4], 3):
            df[f"TG_{c1}_{c2}_{c3}"] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)

        # Numericals as categories
        for col in NUMS:
            df[f"CAT_{col}"] = df[col].astype(str)

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                if not c.startswith("BG_") and not c.startswith("TG_") and not c.startswith("CAT_")]
    ngram_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                  if c.startswith("BG_") or c.startswith("TG_")]
    cat_num_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                    if c.startswith("CAT_")]

    print(f"  Features: {train.shape[1]}", flush=True)
    return train, test, y, test_ids, cat_cols, ngram_cols, cat_num_cols


def fold_process(X_tr, X_va, X_te, y_tr, cat_cols, ngram_cols, cat_num_cols):
    """Full in-fold encoding."""
    all_cats = cat_cols + ngram_cols + cat_num_cols
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    X_tr[all_cats] = oe.fit_transform(X_tr[all_cats])
    X_va[all_cats] = oe.transform(X_va[all_cats])
    X_te[all_cats] = oe.transform(X_te[all_cats])

    gm = y_tr.mean()
    smooth = 10.0
    te_cols = cat_cols[:8] + ngram_cols[:10] + cat_num_cols
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
            X_tr[f"{col}_te_{stat}"] = X_tr[col].map(agg).fillna(0).astype("float32")
            X_va[f"{col}_te_{stat}"] = X_va[col].map(agg).fillna(0).astype("float32")
            X_te[f"{col}_te_{stat}"] = X_te[col].map(agg).fillna(0).astype("float32")

    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[f"{col}_f"] = X_tr[col].map(freq).fillna(0).astype("float32")
        X_va[f"{col}_f"] = X_va[col].map(freq).fillna(0).astype("float32")
        X_te[f"{col}_f"] = X_te[col].map(freq).fillna(0).astype("float32")

    # Ridge stage 1
    sc = StandardScaler()
    r = Ridge(alpha=10.0)
    r.fit(sc.fit_transform(X_tr.fillna(0)), y_tr)
    X_tr["ridge"] = np.clip(r.predict(sc.transform(X_tr.fillna(0))), 0, 1).astype("float32")
    X_va["ridge"] = np.clip(r.predict(sc.transform(X_va.fillna(0))), 0, 1).astype("float32")
    X_te["ridge"] = np.clip(r.predict(sc.transform(X_te.fillna(0))), 0, 1).astype("float32")

    return X_tr.fillna(0), X_va.fillna(0), X_te.fillna(0)


def optuna_tune_xgb(train_df, y, cat_cols, ngram_cols, cat_num_cols, n_trials=30):
    """Optuna-tune XGB params on our exact feature set with 5-fold CV."""
    print("\n[2] Optuna tuning XGB (30 trials, 5-fold)...", flush=True)

    def objective(trial):
        params = {
            "n_estimators": 10000,
            "learning_rate": trial.suggest_float("lr", 0.003, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "early_stopping_rounds": 300,
            "device": "cuda",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
            X_tr = train_df.iloc[tr_idx].copy()
            X_va = train_df.iloc[va_idx].copy()
            # Quick encode (no test needed for tuning)
            all_cats = cat_cols + ngram_cols + cat_num_cols
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
            X_tr[all_cats] = oe.fit_transform(X_tr[all_cats])
            X_va[all_cats] = oe.transform(X_va[all_cats])

            gm = y.iloc[tr_idx].mean()
            for col in (cat_cols[:6] + ngram_cols[:5]):
                tmp = pd.DataFrame({"c": X_tr[col], "y": y.iloc[tr_idx].values})
                agg = tmp.groupby("c")["y"].agg(["mean", "count"])
                sm = (agg["count"] * agg["mean"] + 10 * gm) / (agg["count"] + 10)
                X_tr[f"{col}_te"] = X_tr[col].map(sm).fillna(gm).astype("float32")
                X_va[f"{col}_te"] = X_va[col].map(sm).fillna(gm).astype("float32")

            X_tr = X_tr.fillna(0)
            X_va = X_va.fillna(0)

            m = xgb.XGBClassifier(**params)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
            pred = m.predict_proba(X_va)[:, 1]
            scores.append(roc_auc_score(y.iloc[va_idx], pred))
            del m; gc.collect()

            if fold >= 2:  # Early pruning: 3 folds enough
                break

        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=1800)

    print(f"  Best trial: {study.best_value:.6f}", flush=True)
    print(f"  Best params: {study.best_params}", flush=True)
    return study.best_params


def train_20fold_xgb(train_df, test_df, y, cat_cols, ngram_cols, cat_num_cols, xgb_params):
    """20-fold XGB with given params."""
    print(f"\n[3] Training 20-fold XGB...", flush=True)
    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    full_params = {
        "n_estimators": 50000,
        "early_stopping_rounds": 500,
        "device": "cuda",
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
        **xgb_params,
    }

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
        X_tr = train_df.iloc[tr_idx].copy()
        X_va = train_df.iloc[va_idx].copy()
        X_te = test_df.copy()
        X_tr, X_va, X_te = fold_process(X_tr, X_va, X_te, y.iloc[tr_idx],
                                          cat_cols, ngram_cols, cat_num_cols)

        m = xgb.XGBClassifier(**full_params)
        m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        oof[va_idx] = m.predict_proba(X_va)[:, 1]
        test_preds += m.predict_proba(X_te)[:, 1] / N_FOLDS

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
        del m; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"  Optuna XGB CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def pair_te_logreg(train_df, test_df, y, cat_cols, ngram_cols, cat_num_cols):
    """Chris Deotte's pair TE → logit3 → LogisticRegression.

    All C(n,2) feature pairs get target encoded, then converted to
    logit space with z, z^2, z^3 features.
    """
    print(f"\n[4] Pair TE → Logit3 → LogReg (Deotte approach)...", flush=True)

    # Use all original features (not engineered ones)
    orig_features = [c for c in train_df.columns
                     if not c.startswith("ORIG_") and not c.startswith("BG_")
                     and not c.startswith("TG_") and not c.startswith("CAT_")
                     and "_te" not in c and "_f" not in c and "_pctrank" not in c
                     and "_zscore" not in c and "_qdist" not in c
                     and c not in ["avg_monthly", "charge_ratio", "charges_dev",
                                   "tenure_x_mc", "tenure_sq", "mc_sq", "svc_count",
                                   "has_internet", "has_phone", "ridge"]]

    # Limit to manageable number of pairs
    if len(orig_features) > 20:
        orig_features = orig_features[:20]

    n_pairs = len(list(combinations(range(len(orig_features)), 2)))
    print(f"  {len(orig_features)} features → {n_pairs} pairs", flush=True)

    oof = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
        # Create pair features with TE
        X_tr_pairs = []
        X_va_pairs = []
        X_te_pairs = []

        gm = y.iloc[tr_idx].mean()
        smooth = 0.0  # Deotte uses smooth=0

        for i, (f1, f2) in enumerate(combinations(orig_features, 2)):
            # Create pair key
            pair_tr = train_df.iloc[tr_idx][f1].astype(str) + "_" + train_df.iloc[tr_idx][f2].astype(str)
            pair_va = train_df.iloc[va_idx][f1].astype(str) + "_" + train_df.iloc[va_idx][f2].astype(str)
            pair_te = test_df[f1].astype(str) + "_" + test_df[f2].astype(str)

            # Target encode
            tmp = pd.DataFrame({"pair": pair_tr, "y": y.iloc[tr_idx].values})
            agg = tmp.groupby("pair")["y"].agg(["mean", "count"])
            te_map = (agg["count"] * agg["mean"] + 10 * gm) / (agg["count"] + 10)

            te_tr = pair_tr.map(te_map).fillna(gm).values
            te_va = pair_va.map(te_map).fillna(gm).values
            te_te = pair_te.map(te_map).fillna(gm).values

            # Logit transform (clip to avoid inf)
            for arr_list, te_arr in [(X_tr_pairs, te_tr), (X_va_pairs, te_va), (X_te_pairs, te_te)]:
                z = logit(np.clip(te_arr, 1e-6, 1 - 1e-6))
                arr_list.extend([z, z**2, z**3])

        # Stack into matrices
        X_tr_m = np.column_stack(X_tr_pairs).astype(np.float32)
        X_va_m = np.column_stack(X_va_pairs).astype(np.float32)
        X_te_m = np.column_stack(X_te_pairs).astype(np.float32)

        # Replace inf/nan
        X_tr_m = np.nan_to_num(X_tr_m, nan=0, posinf=10, neginf=-10)
        X_va_m = np.nan_to_num(X_va_m, nan=0, posinf=10, neginf=-10)
        X_te_m = np.nan_to_num(X_te_m, nan=0, posinf=10, neginf=-10)

        # Scale + LogReg
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_m)
        X_va_s = sc.transform(X_va_m)
        X_te_s = sc.transform(X_te_m)

        model = LogisticRegression(C=0.5, max_iter=4000, solver="lbfgs")
        model.fit(X_tr_s, y.iloc[tr_idx])

        oof[va_idx] = model.predict_proba(X_va_s)[:, 1]
        test_preds += model.predict_proba(X_te_s)[:, 1] / N_FOLDS

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)

    score = roc_auc_score(y, oof)
    print(f"  Pair TE LogReg CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def pseudo_label_rounds(train_df, test_df, y, cat_cols, ngram_cols, cat_num_cols,
                        xgb_params, n_rounds=3):
    """Multi-round pseudo-labeling with decreasing thresholds."""
    print(f"\n[5] Multi-round pseudo-labeling ({n_rounds} rounds)...", flush=True)

    thresholds = [0.999, 0.995, 0.99][:n_rounds]
    best_oof = None
    best_test = None
    best_score = 0

    for rnd, thresh in enumerate(thresholds):
        print(f"  Round {rnd+1}, threshold={thresh}...", flush=True)

        if rnd == 0:
            # First round: standard training
            train_aug = train_df.copy()
            y_aug = y.copy()
        else:
            # Add pseudo-labels from previous round
            mask_high = best_test > thresh
            mask_low = best_test < (1 - thresh)
            mask = mask_high | mask_low
            n_pseudo = mask.sum()
            print(f"    Adding {n_pseudo} pseudo-labels", flush=True)

            if n_pseudo < 50:
                print(f"    Too few, stopping", flush=True)
                break

            pseudo_X = test_df[mask].copy()
            # Use SOFT labels
            pseudo_y = pd.Series(best_test[mask])
            pseudo_y = (pseudo_y > 0.5).astype(int)

            train_aug = pd.concat([train_df, pseudo_X], ignore_index=True)
            y_aug = pd.concat([y, pseudo_y], ignore_index=True)

        oof = np.zeros(len(train_df))  # Only evaluate on original train
        test_preds = np.zeros(len(test_df))

        full_params = {
            "n_estimators": 10000,
            "early_stopping_rounds": 300,
            "device": "cuda",
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
            **xgb_params,
        }

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
            # Training includes pseudo-labels + fold's train
            aug_tr_idx = list(tr_idx) + list(range(len(train_df), len(train_aug)))
            X_tr = train_aug.iloc[aug_tr_idx].copy()
            X_va = train_df.iloc[va_idx].copy()
            X_te = test_df.copy()
            y_tr = y_aug.iloc[aug_tr_idx]

            X_tr, X_va, X_te = fold_process(X_tr, X_va, X_te, y_tr,
                                              cat_cols, ngram_cols, cat_num_cols)

            m = xgb.XGBClassifier(**full_params)
            m.fit(X_tr, y_tr, eval_set=[(X_va, y.iloc[va_idx])], verbose=False)

            oof[va_idx] = m.predict_proba(X_va)[:, 1]
            test_preds += m.predict_proba(X_te)[:, 1] / 5
            del m; gc.collect()

        score = roc_auc_score(y, oof)
        print(f"    CV: {score:.6f}", flush=True)

        if score > best_score:
            best_score = score
            best_oof = oof.copy()
            best_test = test_preds.copy()
            print(f"    NEW BEST!", flush=True)

    print(f"  Best pseudo-label CV: {best_score:.6f}", flush=True)
    return best_oof, best_test, best_score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 10: Push to 0.93 - Optuna + Pair TE + Pseudo-labels", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols, ngram_cols, cat_num_cols = prepare_full_features()

    results = {}

    # 1. Optuna-tune XGB on our features
    best_params = optuna_tune_xgb(train, y, cat_cols, ngram_cols, cat_num_cols, n_trials=30)

    # 2. Train 20-fold with Optuna'd params
    xgb_oof, xgb_test, xgb_cv = train_20fold_xgb(
        train, test, y, cat_cols, ngram_cols, cat_num_cols, best_params)
    results["optuna_xgb"] = {"oof": xgb_oof, "test": xgb_test, "cv": xgb_cv}

    # 3. Train 20-fold with BlamerX params (baseline comparison)
    blamerx_params = {
        "learning_rate": 0.0063, "max_depth": 5, "min_child_weight": 6,
        "subsample": 0.81, "colsample_bytree": 0.32,
        "reg_alpha": 3.5017, "reg_lambda": 1.2925, "gamma": 0.790,
    }
    print(f"\n[3b] BlamerX params comparison...", flush=True)
    bx_oof, bx_test, bx_cv = train_20fold_xgb(
        train, test, y, cat_cols, ngram_cols, cat_num_cols, blamerx_params)
    results["blamerx_xgb"] = {"oof": bx_oof, "test": bx_test, "cv": bx_cv}

    # 4. Pair TE → LogReg (Deotte's diversity model)
    try:
        pte_oof, pte_test, pte_cv = pair_te_logreg(
            train, test, y, cat_cols, ngram_cols, cat_num_cols)
        results["pair_te_logreg"] = {"oof": pte_oof, "test": pte_test, "cv": pte_cv}
    except Exception as e:
        print(f"  Pair TE failed: {e}", flush=True)

    # 5. Pseudo-labeling with best XGB params
    best_xgb_params = best_params if xgb_cv > bx_cv else blamerx_params
    try:
        ps_oof, ps_test, ps_cv = pseudo_label_rounds(
            train, test, y, cat_cols, ngram_cols, cat_num_cols, best_xgb_params)
        results["pseudo_xgb"] = {"oof": ps_oof, "test": ps_test, "cv": ps_cv}
    except Exception as e:
        print(f"  Pseudo-labeling failed: {e}", flush=True)

    # === ENSEMBLES ===
    print("\n=== ENSEMBLES ===", flush=True)

    # Best single
    best_name = max(results, key=lambda k: results[k]["cv"])
    best_cv = results[best_name]["cv"]
    print(f"  Best single: {best_name} = {best_cv:.6f}", flush=True)

    # Diverse blend (XGB + LogReg if available)
    if "pair_te_logreg" in results:
        xgb_best = results[best_name]["test"]
        pte = results["pair_te_logreg"]["test"]
        for w in [0.7, 0.8, 0.9]:
            blend = w * xgb_best + (1 - w) * pte
            blend_oof = w * results[best_name]["oof"] + (1 - w) * results["pair_te_logreg"]["oof"]
            blend_cv = roc_auc_score(y, blend_oof)
            print(f"  Blend ({w:.0%} XGB + {1-w:.0%} PairTE): {blend_cv:.6f}", flush=True)

    # Save all
    for name, r in results.items():
        generate_submission(test_ids, r["test"], "id", "Churn", f"submissions/iter10_{name}.csv")

    # Best blend
    if "pair_te_logreg" in results:
        blend80 = 0.8 * results[best_name]["test"] + 0.2 * results["pair_te_logreg"]["test"]
        generate_submission(test_ids, blend80, "id", "Churn", "submissions/iter10_blend80.csv")

    # Blend with iter6
    try:
        iter6 = pd.read_csv("submissions/iter6_blamerx.csv")["Churn"].values
        avg_i6 = (results[best_name]["test"] + iter6) / 2
        generate_submission(test_ids, avg_i6, "id", "Churn", "submissions/iter10_best_i6.csv")
    except Exception:
        pass

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 10 - Optuna + Pair TE + Pseudo-labels\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"**Optuna best params**: {best_params}\n\n")
        for name, r in results.items():
            f.write(f"- {name}: {r['cv']:.6f}\n")
        f.write(f"\n---\n")

    print("\n" + "=" * 70, flush=True)
    print("ITERATION 10 COMPLETE", flush=True)
    for name, r in sorted(results.items(), key=lambda x: x[1]["cv"], reverse=True):
        print(f"  {name}: {r['cv']:.6f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
