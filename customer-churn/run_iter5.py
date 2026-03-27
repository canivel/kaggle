"""Iteration 5: Full top-notebook reproduction.

Implements:
1. BlamerX-style two-stage Ridge→LGBM (CV 0.91927 approach)
2. XGBoost with enable_categorical (all top notebooks)
3. Extended digit features (35 features, exploit synthetic data)
4. Distribution features (pctrank vs churner/non-churner)
5. N-gram categoricals with target encoding
6. Pseudo-labeling with confident test predictions
7. Multi-seed (3 seeds) x multi-model (LGBM, XGB, CatBoost)
8. Simple average ensemble (proven more robust than stacking)

Target: LB > 0.917 (currently 0.91526)
"""

import sys
sys.path.insert(0, "../kaggle-agent/src")

import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, TargetEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from itertools import combinations
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")


def add_orig_proba(train, test):
    """ORIG_proba from IBM Telco (external, no leakage)."""
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    orig_churn = (orig["Churn"] == "Yes").astype(float)
    gm = orig_churn.mean()
    added = []
    for col in train.columns:
        if col in orig.columns:
            tmp = orig.copy()
            tmp["_y"] = orig_churn
            proba = tmp.groupby(col)["_y"].mean()
            name = f"ORIG_{col}"
            train[name] = train[col].map(proba).fillna(gm).astype("float32")
            test[name] = test[col].map(proba).fillna(gm).astype("float32")
            added.append(name)
    return train, test, added


def add_static_features(df):
    """Target-independent features (safe outside folds)."""
    # Arithmetic
    df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
    df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
    df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
    df["tenure_x_monthly"] = (df["tenure"] * df["MonthlyCharges"]).astype("float32")

    # Service counts
    yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    avail = [c for c in yes_cols if c in df.columns]
    df["service_count"] = sum((df[c] == "Yes").astype(int) for c in avail)
    df["has_internet"] = (df.get("InternetService", "No") != "No").astype(int)
    df["has_phone"] = (df.get("PhoneService", "No") == "Yes").astype(int)

    # Extended digit features (BlamerX: 35 features)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col not in df.columns:
            continue
        vals = df[col].fillna(0)
        df[f"{col}_mod10"] = (vals % 10).astype("float32")
        df[f"{col}_mod100"] = (vals % 100).astype("float32")
        df[f"{col}_first_digit"] = vals.astype(str).str[0].astype("float32")
        df[f"{col}_last_digit"] = (vals * 100).astype(int) % 10
        df[f"{col}_is_round"] = ((vals % 5 == 0) | (vals % 10 == 0)).astype(int)
        if col != "tenure":
            df[f"{col}_frac"] = (vals - vals.astype(int)).astype("float32")
            df[f"{col}_deviation_round"] = (vals - (vals / 5).round() * 5).astype("float32")
        if col == "tenure":
            df[f"{col}_mod12"] = (vals % 12).astype("float32")  # yearly cycles

    # Bi-gram categoricals
    top_cats = ["Contract", "InternetService", "PaymentMethod",
                "OnlineSecurity", "TechSupport", "PaperlessBilling"]
    avail_cats = [c for c in top_cats if c in df.columns]
    for c1, c2 in combinations(avail_cats, 2):
        df[f"{c1}_x_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)

    # Tri-gram (top 4)
    top4 = avail_cats[:4]
    for c1, c2, c3 in combinations(top4, 3):
        df[f"{c1}_x_{c2}_x_{c3}"] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)

    return df


def add_distribution_features(train, test, orig):
    """Percentile rank vs churner/non-churner distributions from original dataset."""
    orig = orig.copy()
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    orig_churn = (orig["Churn"] == "Yes")

    for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if col not in orig.columns or col not in train.columns:
            continue
        churner_vals = orig.loc[orig_churn, col].dropna().values
        nonchurner_vals = orig.loc[~orig_churn, col].dropna().values

        if len(churner_vals) == 0 or len(nonchurner_vals) == 0:
            continue

        # Percentile rank against churner distribution
        for df in [train, test]:
            vals = df[col].values
            pctrank_churn = np.searchsorted(np.sort(churner_vals), vals) / len(churner_vals)
            pctrank_nonchurn = np.searchsorted(np.sort(nonchurner_vals), vals) / len(nonchurner_vals)
            df[f"{col}_pctrank_churn"] = pctrank_churn.astype("float32")
            df[f"{col}_pctrank_nonchurn"] = pctrank_nonchurn.astype("float32")
            df[f"{col}_pctrank_gap"] = (pctrank_churn - pctrank_nonchurn).astype("float32")

    return train, test


def fold_encode(X_tr, X_va, X_te, y_tr, cat_cols, ngram_cols):
    """All encoding INSIDE the fold."""
    all_cats = cat_cols + ngram_cols
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                        dtype=np.float32)
    X_tr[all_cats] = oe.fit_transform(X_tr[all_cats])
    X_va[all_cats] = oe.transform(X_va[all_cats])
    X_te[all_cats] = oe.transform(X_te[all_cats])

    # Target encoding (leak-free via inner CV)
    te_cols = cat_cols[:8] + ngram_cols[:10]  # Top cats + top ngrams
    if te_cols:
        te = TargetEncoder(cv=5, smooth="auto")
        te_names = [f"{c}_te" for c in te_cols]
        X_tr[te_names] = te.fit_transform(X_tr[te_cols], y_tr)
        X_va[te_names] = te.transform(X_va[te_cols])
        X_te[te_names] = te.transform(X_te[te_cols])

    # Frequency encoding
    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True)
        fname = f"{col}_freq"
        X_tr[fname] = X_tr[col].map(freq).fillna(0).astype("float32")
        X_va[fname] = X_va[col].map(freq).fillna(0).astype("float32")
        X_te[fname] = X_te[col].map(freq).fillna(0).astype("float32")

    return X_tr, X_va, X_te


def train_ridge_stage1(X_tr, X_va, X_te, y_tr):
    """Stage 1: Ridge predictions as features for Stage 2 (BlamerX approach)."""
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr_s, y_tr)

    ridge_tr = np.clip(ridge.predict(X_tr_s), 0, 1).astype("float32")
    ridge_va = np.clip(ridge.predict(X_va_s), 0, 1).astype("float32")
    ridge_te = np.clip(ridge.predict(X_te_s), 0, 1).astype("float32")

    X_tr["ridge_pred"] = ridge_tr
    X_va["ridge_pred"] = ridge_va
    X_te["ridge_pred"] = ridge_te

    return X_tr, X_va, X_te


def run_model_cv(name, model_cls, model_params, train_df, test_df, y, cat_cols, ngram_cols,
                 n_folds=5, seeds=[42], use_ridge=False):
    """Run a model with proper in-fold encoding + optional Ridge stage 1."""
    print(f"\n[{name}]", flush=True)
    all_oof = np.zeros(len(train_df))
    all_test = np.zeros(len(test_df))

    for seed in seeds:
        oof = np.zeros(len(train_df))
        test_accum = []

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train_df, y)):
            X_tr = train_df.iloc[tr_idx].copy()
            X_va = train_df.iloc[va_idx].copy()
            X_te = test_df.copy()
            y_tr = y.iloc[tr_idx]

            X_tr, X_va, X_te = fold_encode(X_tr, X_va, X_te, y_tr, cat_cols, ngram_cols)

            if use_ridge:
                X_tr, X_va, X_te = train_ridge_stage1(X_tr, X_va, X_te, y_tr)

            if model_cls == "lgbm":
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**model_params, random_state=seed,
                                           verbosity=-1, n_jobs=4)
                model.fit(X_tr, y_tr,
                          eval_set=[(X_va, y.iloc[va_idx])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            elif model_cls == "xgb":
                import xgboost as xgb
                model = xgb.XGBClassifier(**model_params, random_state=seed,
                                          verbosity=0, n_jobs=4)
                model.fit(X_tr, y_tr,
                          eval_set=[(X_va, y.iloc[va_idx])],
                          verbose=False)
            elif model_cls == "catboost":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**model_params, random_seed=seed, verbose=0)
                model.fit(X_tr, y_tr, eval_set=(X_va, y.iloc[va_idx]), verbose=0)

            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_accum.append(model.predict_proba(X_te)[:, 1])

        score = roc_auc_score(y, oof)
        print(f"  seed={seed}: CV={score:.6f}", flush=True)
        all_oof += oof / len(seeds)
        all_test += np.mean(test_accum, axis=0) / len(seeds)

    final_score = roc_auc_score(y, all_oof)
    print(f"  {name} final: {final_score:.6f}", flush=True)
    return all_oof, all_test, final_score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 5: Full Top-Notebook Reproduction", flush=True)
    print("=" * 70, flush=True)

    # Load data
    print("\n[1] Loading data...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    # ORIG_proba
    print("[2] ORIG_proba...", flush=True)
    train, test, orig_feats = add_orig_proba(train, test)

    # Distribution features
    print("[3] Distribution features...", flush=True)
    orig = pd.read_csv("data/telco_original.csv")
    train, test = add_distribution_features(train, test, orig)

    # Static features (digit, bigram, trigram, arithmetic)
    print("[4] Static features...", flush=True)
    train = add_static_features(train)
    test = add_static_features(test)

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                if "_x_" not in c]
    ngram_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns
                  if "_x_" in c]
    print(f"  Cats: {len(cat_cols)}, N-grams: {len(ngram_cols)}, "
          f"Total features: {train.shape[1]}", flush=True)

    # === TRAIN DIVERSE MODELS ===
    all_oof = {}
    all_test = {}

    # LGBM configs (multi-seed)
    lgbm_params = {
        "n_estimators": 3000, "learning_rate": 0.02, "num_leaves": 63,
        "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "min_child_samples": 20,
    }
    oof, tst, sc = run_model_cv("lgbm_3seed", "lgbm", lgbm_params,
                                 train, test, y, cat_cols, ngram_cols,
                                 seeds=[42, 11, 99])
    all_oof["lgbm_3seed"] = oof
    all_test["lgbm_3seed"] = tst

    # LGBM high-reg
    lgbm_reg_params = {
        "n_estimators": 3000, "learning_rate": 0.01, "num_leaves": 31,
        "max_depth": 5, "subsample": 0.7, "colsample_bytree": 0.5,
        "reg_alpha": 1.0, "reg_lambda": 2.0, "min_child_samples": 50,
    }
    oof, tst, sc = run_model_cv("lgbm_reg", "lgbm", lgbm_reg_params,
                                 train, test, y, cat_cols, ngram_cols,
                                 seeds=[42])
    all_oof["lgbm_reg"] = oof
    all_test["lgbm_reg"] = tst

    # LGBM with Ridge stage 1 (BlamerX approach)
    oof, tst, sc = run_model_cv("lgbm_ridge", "lgbm", lgbm_params,
                                 train, test, y, cat_cols, ngram_cols,
                                 seeds=[42], use_ridge=True)
    all_oof["lgbm_ridge"] = oof
    all_test["lgbm_ridge"] = tst

    # XGBoost (CPU, aggressive regularization like BlamerX)
    xgb_params = {
        "n_estimators": 3000, "learning_rate": 0.01, "max_depth": 5,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.35,
        "reg_alpha": 3.5, "reg_lambda": 1.3, "gamma": 0.8,
        "early_stopping_rounds": 200, "tree_method": "hist",
    }
    oof, tst, sc = run_model_cv("xgb_blamerx", "xgb", xgb_params,
                                 train, test, y, cat_cols, ngram_cols,
                                 seeds=[42, 11])
    all_oof["xgb_blamerx"] = oof
    all_test["xgb_blamerx"] = tst

    # CatBoost
    cb_params = {
        "iterations": 3000, "learning_rate": 0.02, "depth": 4,
        "min_data_in_leaf": 20, "subsample": 0.9,
        "eval_metric": "Logloss", "task_type": "CPU",
        "early_stopping_rounds": 100,
    }
    oof, tst, sc = run_model_cv("catboost", "catboost", cb_params,
                                 train, test, y, cat_cols, ngram_cols,
                                 seeds=[42])
    all_oof["catboost"] = oof
    all_test["catboost"] = tst

    # LogReg for diversity
    print("\n[LogReg]", flush=True)
    lr_oof = np.zeros(len(train))
    lr_test = np.zeros(len(test))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        X_tr, X_va, X_te = fold_encode(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        model = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        model.fit(X_tr_s, y.iloc[tr_idx])
        lr_oof[va_idx] = model.predict_proba(sc.transform(X_va))[:, 1]
        lr_test += model.predict_proba(sc.transform(X_te))[:, 1] / 5
    print(f"  LogReg: {roc_auc_score(y, lr_oof):.6f}", flush=True)
    all_oof["logistic"] = lr_oof
    all_test["logistic"] = lr_test

    # === ENSEMBLES ===
    print("\n=== ENSEMBLES ===", flush=True)

    # Tree-only
    tree_keys = [k for k in all_oof if k != "logistic"]
    tree_o = np.mean([all_oof[k] for k in tree_keys], axis=0)
    tree_t = np.mean([all_test[k] for k in tree_keys], axis=0)
    print(f"Tree avg ({len(tree_keys)}): {roc_auc_score(y, tree_o):.6f}", flush=True)

    # All models
    all_o = np.mean(list(all_oof.values()), axis=0)
    all_t = np.mean(list(all_test.values()), axis=0)
    print(f"All avg ({len(all_oof)}):  {roc_auc_score(y, all_o):.6f}", flush=True)

    # Rank average
    rank_t = np.mean([rankdata(all_test[k]) / len(test) for k in all_test], axis=0)
    rank_o = np.mean([rankdata(all_oof[k]) / len(y) for k in all_oof], axis=0)
    print(f"Rank avg ({len(all_oof)}): {roc_auc_score(y, rank_o):.6f}", flush=True)

    # === PSEUDO-LABELING ===
    print("\n[Pseudo-labeling]", flush=True)
    # Use confident predictions from ensemble to add to training
    confident_mask = (all_t > 0.995) | (all_t < 0.005)
    n_pseudo = confident_mask.sum()
    print(f"  Confident test samples: {n_pseudo}", flush=True)

    if n_pseudo > 100:
        pseudo_y = (all_t[confident_mask] > 0.5).astype(int)
        pseudo_X = test[confident_mask].copy()
        # Retrain best model with pseudo labels
        train_aug = pd.concat([train, pseudo_X], ignore_index=True)
        y_aug = pd.concat([y, pd.Series(pseudo_y)], ignore_index=True)

        oof_ps = np.zeros(len(train))
        test_ps_accum = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
            # Only validate on original data
            X_tr = train_aug.iloc[list(tr_idx) + list(range(len(train), len(train_aug)))].copy()
            X_va = train.iloc[va_idx].copy()
            X_te = test.copy()
            y_tr_fold = y_aug.iloc[list(tr_idx) + list(range(len(train), len(train_aug)))]

            X_tr, X_va, X_te = fold_encode(X_tr, X_va, X_te, y_tr_fold, cat_cols, ngram_cols)

            import lightgbm as lgb
            model = lgb.LGBMClassifier(**lgbm_params, random_state=42, verbosity=-1, n_jobs=4)
            model.fit(X_tr, y_tr_fold,
                      eval_set=[(X_va, y.iloc[va_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
            oof_ps[va_idx] = model.predict_proba(X_va)[:, 1]
            test_ps_accum.append(model.predict_proba(X_te)[:, 1])

        ps_score = roc_auc_score(y, oof_ps)
        print(f"  Pseudo-label LGBM: {ps_score:.6f}", flush=True)
        test_ps = np.mean(test_ps_accum, axis=0)
        all_oof["lgbm_pseudo"] = oof_ps
        all_test["lgbm_pseudo"] = test_ps

        # Updated ensemble with pseudo
        all_o2 = np.mean(list(all_oof.values()), axis=0)
        all_t2 = np.mean(list(all_test.values()), axis=0)
        print(f"  All+pseudo avg: {roc_auc_score(y, all_o2):.6f}", flush=True)

    # === SAVE SUBMISSIONS ===
    print("\n[Saving submissions]", flush=True)
    generate_submission(test_ids, tree_t, "id", "Churn", "submissions/iter5_tree.csv")
    generate_submission(test_ids, all_t, "id", "Churn", "submissions/iter5_all.csv")
    generate_submission(test_ids, rank_t, "id", "Churn", "submissions/iter5_rank.csv")
    if "lgbm_pseudo" in all_test:
        all_t2 = np.mean(list(all_test.values()), axis=0)
        generate_submission(test_ids, all_t2, "id", "Churn", "submissions/iter5_pseudo.csv")

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 5 - Full Top-Notebook Reproduction\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        f.write("**New techniques**: Ridge stage1, XGB BlamerX params, distribution features, ")
        f.write("extended digit features, tri-grams, pseudo-labeling\n\n")
        for name in all_oof:
            f.write(f"- {name}: {roc_auc_score(y, all_oof[name]):.6f}\n")
        f.write(f"- **Tree avg**: {roc_auc_score(y, tree_o):.6f}\n")
        f.write(f"- **All avg**: {roc_auc_score(y, all_o):.6f}\n")
        f.write(f"- **Rank avg**: {roc_auc_score(y, rank_o):.6f}\n")
        f.write(f"\n---\n")

    print("\n" + "=" * 70, flush=True)
    print("ITERATION 5 COMPLETE", flush=True)
    for name in sorted(all_oof, key=lambda k: roc_auc_score(y, all_oof[k]), reverse=True):
        print(f"  {name}: {roc_auc_score(y, all_oof[name]):.6f}", flush=True)
    print(f"\n  Tree avg: {roc_auc_score(y, tree_o):.6f}", flush=True)
    print(f"  All avg:  {roc_auc_score(y, all_o):.6f}", flush=True)
    print(f"  Rank avg: {roc_auc_score(y, rank_o):.6f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
