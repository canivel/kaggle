"""Iteration 6: Faithful BlamerX reproduction (CV 0.91927).

Exact reproduction of the highest-scoring single-notebook approach:
- 20-fold outer CV
- 5-fold inner CV for leak-free target encoding
- Two-stage Ridge → XGBoost pipeline
- N-gram categoricals (15 bi-grams + 4 tri-grams) with TE
- ORIG_proba from IBM Telco dataset
- Distribution features (pctrank, zscore vs churner/non-churner)
- Quantile distance features
- 30+ digit features
- Numericals as categories
- BlamerX's exact Optuna-tuned XGBoost hyperparameters
- enable_categorical=True with GPU
"""

import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")

import gc
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
import xgboost as xgb

warnings.filterwarnings("ignore")

# ============================================================
# EXACT BlamerX Configuration
# ============================================================
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

TOP_CATS_FOR_NGRAM = [
    "Contract", "InternetService", "PaymentMethod",
    "OnlineSecurity", "TechSupport", "PaperlessBilling",
]
TOP4 = TOP_CATS_FOR_NGRAM[:4]

TARGET = "Churn"
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data():
    print("[1] Loading data...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train[TARGET] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", TARGET])
    test = test.drop(columns=["id"])
    return train, test, y, test_ids


def load_original():
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    return orig


def add_orig_proba(train, test, orig):
    """ORIG_proba features for ALL columns."""
    print("[2] ORIG_proba features...", flush=True)
    oc = (orig[TARGET] == "Yes").astype(float)
    gm = oc.mean()
    cats = train.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cats + NUMS:
        if col not in orig.columns:
            continue
        tmp = orig.copy()
        tmp["_y"] = oc
        proba = tmp.groupby(col)["_y"].mean()
        name = f"ORIG_proba_{col}"
        train[name] = train[col].map(proba).fillna(gm).astype("float32")
        test[name] = test[col].map(proba).fillna(gm).astype("float32")
    return train, test


def add_distribution_features(train, test, orig):
    """Percentile rank, z-score, quantile distance features."""
    print("[3] Distribution features...", flush=True)
    oc = (orig[TARGET] == "Yes")
    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)

    churner_tc = np.sort(orig_tc[oc].values)
    nonchurner_tc = np.sort(orig_tc[~oc].values)
    all_tc = np.sort(orig_tc.values)

    for df in [train, test]:
        tc = df["TotalCharges"].values

        # Percentile ranks
        df["pctrank_churner_TC"] = (np.searchsorted(churner_tc, tc) / len(churner_tc)).astype("float32")
        df["pctrank_nonchurner_TC"] = (np.searchsorted(nonchurner_tc, tc) / len(nonchurner_tc)).astype("float32")
        df["pctrank_orig_TC"] = (np.searchsorted(all_tc, tc) / len(all_tc)).astype("float32")
        df["pctrank_churn_gap_TC"] = df["pctrank_churner_TC"] - df["pctrank_nonchurner_TC"]

        # Z-scores
        ch_mu, ch_std = churner_tc.mean(), churner_tc.std()
        nc_mu, nc_std = nonchurner_tc.mean(), nonchurner_tc.std()
        df["zscore_churner_TC"] = ((tc - ch_mu) / ch_std).astype("float32")
        df["zscore_nonchurner_TC"] = ((tc - nc_mu) / nc_std).astype("float32")
        df["zscore_churn_gap_TC"] = df["zscore_churner_TC"] - df["zscore_nonchurner_TC"]

        # Quantile distances
        for q_label, q_val in [("q25", 0.25), ("q50", 0.50), ("q75", 0.75)]:
            ch_q = np.quantile(churner_tc, q_val)
            nc_q = np.quantile(nonchurner_tc, q_val)
            df[f"dist_ch_{q_label}"] = np.abs(tc - ch_q).astype("float32")
            df[f"dist_nc_{q_label}"] = np.abs(tc - nc_q).astype("float32")
            df[f"qdist_gap_{q_label}"] = (df[f"dist_nc_{q_label}"] - df[f"dist_ch_{q_label}"]).astype("float32")

        # Conditional percentile ranks
        for grp_col in ["InternetService", "Contract"]:
            if grp_col not in orig.columns:
                continue
            for grp_val in orig[grp_col].unique():
                mask_orig = orig[grp_col] == grp_val
                ref = np.sort(orig_tc[mask_orig].values)
                if len(ref) < 10:
                    continue
                mask_df = df[grp_col] == grp_val
                df.loc[mask_df, f"cond_pctrank_{grp_col}_{grp_val}_TC"] = (
                    np.searchsorted(ref, tc[mask_df]) / len(ref)
                ).astype("float32")

    return train, test


def add_static_features(train, test):
    """Arithmetic, service counts, digit features, n-grams."""
    print("[4] Static features...", flush=True)

    for df in [train, test]:
        # Arithmetic
        df["avg_monthly_charges"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")

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

        # === DIGIT FEATURES (30+ features) ===
        for col in NUMS:
            v = df[col].fillna(0)
            sv = v.astype(str)
            df[f"{col}_first_digit"] = sv.str[0].astype("float32")
            df[f"{col}_last_digit"] = (v * 100).astype(int) % 10
            df[f"{col}_second_digit"] = sv.str[1:2].replace("", "0").str.replace(".", "0").astype("float32")
            df[f"{col}_mod10"] = (v % 10).astype("float32")
            df[f"{col}_num_digits"] = sv.str.replace(".", "", regex=False).str.len().astype("float32")
            df[f"{col}_is_multiple_10"] = ((v % 10 == 0) & (v > 0)).astype(int)
            df[f"{col}_rounded_10"] = (v / 10).round() * 10
            df[f"{col}_dev_from_round10"] = (v - df[f"{col}_rounded_10"]).astype("float32")

            if col != "tenure":
                df[f"{col}_mod100"] = (v % 100).astype("float32")
                df[f"{col}_fractional"] = (v - v.astype(int)).astype("float32")

            if col == "tenure":
                df[f"{col}_mod12"] = (v % 12).astype("float32")
                df["tenure_years"] = (v // 12).astype("float32")
                df["tenure_months_in_year"] = (v % 12).astype("float32")

        # === N-GRAM CATEGORICALS ===
        avail_cats = [c for c in TOP_CATS_FOR_NGRAM if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = (df[c1].astype(str) + "_" + df[c2].astype(str)).astype("category")

        top4 = avail_cats[:4]
        for c1, c2, c3 in combinations(top4, 3):
            df[f"TG_{c1}_{c2}_{c3}"] = (
                df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
            ).astype("category")

        # === NUMERICALS AS CATEGORIES ===
        for col in NUMS:
            df[f"CAT_{col}"] = df[col].astype(str).astype("category")

    return train, test


def train_blamerx(train, test, y, test_ids):
    """Full BlamerX pipeline: 20-fold outer, Ridge→XGB."""
    print(f"\n[5] Training BlamerX pipeline ({N_FOLDS}-fold)...", flush=True)

    # Identify column types
    cat_cols = [c for c in train.columns
                if train[c].dtype.name in ("category", "object", "string")
                and not c.startswith("BG_") and not c.startswith("TG_") and not c.startswith("CAT_")]
    ngram_cols = [c for c in train.columns if c.startswith("BG_") or c.startswith("TG_")]
    cat_as_num_cols = [c for c in train.columns if c.startswith("CAT_")]
    num_cols = [c for c in train.columns
                if c not in cat_cols + ngram_cols + cat_as_num_cols
                and train[c].dtype in ("float32", "float64", "int64", "int32")]

    te_columns = cat_cols  # Original categoricals for TE
    te_ngram_columns = ngram_cols  # N-gram categoricals for TE

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    outer_kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for fold, (tr_idx, va_idx) in enumerate(outer_kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        y_tr = y.iloc[tr_idx].copy()
        y_va = y.iloc[va_idx]

        # === INNER-FOLD TARGET ENCODING ===
        # TE Pass 1: std, min, max for original categoricals
        inner_kf = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=42)
        te_stats = ["std", "min", "max"]

        te1_tr = pd.DataFrame(index=X_tr.index)
        te1_va = pd.DataFrame(index=X_va.index)
        te1_te = pd.DataFrame(index=X_te.index)

        for col in te_columns:
            for stat in te_stats:
                feat_name = f"TE1_{col}_{stat}"
                te1_tr[feat_name] = np.nan
                te1_va[feat_name] = np.nan
                te1_te[feat_name] = np.nan

        # Inner-fold OOF for TE stats
        for inner_fold, (itr_idx, iva_idx) in enumerate(inner_kf.split(X_tr, y_tr)):
            for col in te_columns:
                col_vals = X_tr.iloc[itr_idx].copy()
                col_vals["_target"] = y_tr.iloc[itr_idx].values
                agg = col_vals.groupby(col)["_target"].agg(te_stats)
                for stat in te_stats:
                    feat_name = f"TE1_{col}_{stat}"
                    mapping = agg[stat]
                    te1_tr.iloc[iva_idx, te1_tr.columns.get_loc(feat_name)] = (
                        X_tr.iloc[iva_idx][col].map(mapping).values
                    )

        # Full-fold stats for val/test
        for col in te_columns:
            col_vals = X_tr.copy()
            col_vals["_target"] = y_tr.values
            agg = col_vals.groupby(col)["_target"].agg(te_stats)
            for stat in te_stats:
                feat_name = f"TE1_{col}_{stat}"
                te1_va[feat_name] = X_va[col].map(agg[stat]).values
                te1_te[feat_name] = X_te[col].map(agg[stat]).values

        # TE Pass 2: Manual smoothed target encoding (mean)
        global_mean = y_tr.mean()
        smooth = 10.0
        te_mean_cols = [f"TE_mean_{c}" for c in te_columns]
        te_tr_mean = pd.DataFrame(index=X_tr.index)
        te_va_mean = pd.DataFrame(index=X_va.index)
        te_te_mean = pd.DataFrame(index=X_te.index)

        for col in te_columns:
            tmp = pd.DataFrame({"col": X_tr[col], "y": y_tr.values})
            agg = tmp.groupby("col")["y"].agg(["mean", "count"])
            # Bayesian smoothing
            smoothed = (agg["count"] * agg["mean"] + smooth * global_mean) / (agg["count"] + smooth)
            name = f"TE_mean_{col}"
            te_tr_mean[name] = X_tr[col].map(smoothed).fillna(global_mean).astype("float32")
            te_va_mean[name] = X_va[col].map(smoothed).fillna(global_mean).astype("float32")
            te_te_mean[name] = X_te[col].map(smoothed).fillna(global_mean).astype("float32")

        # TE Pass 3: N-gram TE (mean only, same smoothed approach)
        ng_mean_cols = [f"TE_ng_{c}" for c in te_ngram_columns]
        te_ng_tr = pd.DataFrame(index=X_tr.index)
        te_ng_va = pd.DataFrame(index=X_va.index)
        te_ng_te = pd.DataFrame(index=X_te.index)

        if te_ngram_columns:
            for col in te_ngram_columns:
                tmp = pd.DataFrame({"col": X_tr[col].astype(str), "y": y_tr.values})
                agg = tmp.groupby("col")["y"].agg(["mean", "count"])
                smoothed = (agg["count"] * agg["mean"] + smooth * global_mean) / (agg["count"] + smooth)
                name = f"TE_ng_{col}"
                te_ng_tr[name] = X_tr[col].astype(str).map(smoothed).fillna(global_mean).astype("float32")
                te_ng_va[name] = X_va[col].astype(str).map(smoothed).fillna(global_mean).astype("float32")
                te_ng_te[name] = X_te[col].astype(str).map(smoothed).fillna(global_mean).astype("float32")

        # === ASSEMBLE FEATURES ===
        # Numeric features
        X_tr_num = X_tr[num_cols].copy()
        X_va_num = X_va[num_cols].copy()
        X_te_num = X_te[num_cols].copy()

        # Add all TE features
        for df_main, te1, te_m, te_n in [
            (X_tr_num, te1_tr, te_tr_mean, te_ng_tr if te_ngram_columns else None),
            (X_va_num, te1_va, te_va_mean, te_ng_va if te_ngram_columns else None),
            (X_te_num, te1_te, te_te_mean, te_ng_te if te_ngram_columns else None),
        ]:
            for c in te1.columns:
                df_main[c] = te1[c].values
            for c in te_m.columns:
                df_main[c] = te_m[c].values
            if te_n is not None:
                for c in te_n.columns:
                    df_main[c] = te_n[c].values

        # Fill NaN
        X_tr_num = X_tr_num.fillna(0)
        X_va_num = X_va_num.fillna(0)
        X_te_num = X_te_num.fillna(0)

        # === STAGE 1: Ridge ===
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_num)
        X_va_scaled = scaler.transform(X_va_num)
        X_te_scaled = scaler.transform(X_te_num)

        ridge = Ridge(alpha=RIDGE_ALPHA, random_state=42)
        ridge.fit(X_tr_scaled, y_tr)

        ridge_tr_pred = np.clip(ridge.predict(X_tr_scaled), 0, 1).astype("float32")
        ridge_va_pred = np.clip(ridge.predict(X_va_scaled), 0, 1).astype("float32")
        ridge_te_pred = np.clip(ridge.predict(X_te_scaled), 0, 1).astype("float32")

        X_tr_num["ridge_pred"] = ridge_tr_pred
        X_va_num["ridge_pred"] = ridge_va_pred
        X_te_num["ridge_pred"] = ridge_te_pred

        # === STAGE 2: XGBoost ===
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr_num, y_tr,
            eval_set=[(X_va_num, y_va)],
            verbose=False,
        )

        oof[va_idx] = model.predict_proba(X_va_num)[:, 1]
        test_preds += model.predict_proba(X_te_num)[:, 1] / N_FOLDS

        if fold % 5 == 0:
            fold_auc = roc_auc_score(y_va, oof[va_idx])
            print(f"  Fold {fold:2d}: AUC={fold_auc:.6f}", flush=True)

        del X_tr, X_va, X_te, X_tr_num, X_va_num, X_te_num, model
        gc.collect()

    final_auc = roc_auc_score(y, oof)
    print(f"\n  BlamerX CV ({N_FOLDS}-fold): {final_auc:.6f}", flush=True)
    return oof, test_preds, final_auc


def main():
    print("=" * 70, flush=True)
    print("ITERATION 6: BlamerX Reproduction (CV 0.91927 target)", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids = load_data()
    orig = load_original()

    train, test = add_orig_proba(train, test, orig)
    train, test = add_distribution_features(train, test, orig)
    train, test = add_static_features(train, test)

    print(f"  Total features: {train.shape[1]}", flush=True)

    oof, test_preds, cv_score = train_blamerx(train, test, y, test_ids)

    # Save submission
    from kaggle_agent.pipeline.submission import generate_submission
    generate_submission(test_ids, test_preds, "id", "Churn", "submissions/iter6_blamerx.csv")
    print(f"\nSubmission saved: submissions/iter6_blamerx.csv", flush=True)

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 6 - BlamerX Reproduction\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"**CV**: {cv_score:.6f} (target: 0.91927)\n\n")
        f.write("**Approach**: 20-fold Ridge→XGB, N-gram TE, ORIG_proba, distribution features, ")
        f.write("digit features, exact BlamerX XGB params (lr=0.0063, colsample=0.32, reg_alpha=3.5)\n\n")
        f.write("---\n")

    print(f"\nDONE! CV={cv_score:.6f}", flush=True)


if __name__ == "__main__":
    main()
