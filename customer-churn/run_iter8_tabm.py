"""Iteration 8: TabM neural model + 3-level stacking.

TabM: Parameter-efficient MLP ensemble (ICLR 2025, Yandex Research).
Won $60K in Kaggle prize money. Fundamentally different from GBDTs.

Strategy:
1. Train TabM on same features as iter7 → OOF predictions
2. Load iter7 OOF predictions (31 models)
3. 3-level stacking: Level-0 (models) → Level-1 (Ridge) → Level-2 (Ridge)
4. Hill climbing on ALL predictions including TabM
5. Generate diverse submission variants
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
from sklearn.linear_model import Ridge, LogisticRegression
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")

TOP_CATS = ["Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]


def prepare_data():
    """Same feature engineering as iter7."""
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

    # Distribution features
    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    ch_tc = np.sort(orig_tc[orig["Churn"] == "Yes"].values)
    nc_tc = np.sort(orig_tc[orig["Churn"] != "Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")

    # Static features
    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        avail = [c for c in yes_cols if c in df.columns]
        df["svc_count"] = sum((df[c] == "Yes").astype(int) for c in avail)
        for col in NUMS:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v % 10).astype("float32")
            df[f"{col}_m100"] = (v % 100).astype("float32")
            df[f"{col}_d1"] = v.astype(str).str[0].astype("float32")
            if col != "tenure": df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure": df[f"{col}_m12"] = (v % 12).astype("float32")
        avail_cats = [c for c in TOP_CATS if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
        for c1, c2, c3 in combinations(avail_cats[:4], 3):
            df[f"TG_{c1}_{c2}_{c3}"] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)

    # Encode ALL categoricals
    cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    train[cat_cols] = oe.fit_transform(train[cat_cols])
    test[cat_cols] = oe.transform(test[cat_cols])

    return train, test, y, test_ids


def train_tabm(X_train, y_train, X_test, n_folds=5, seed=42):
    """Train TabM neural model."""
    print("\n[TabM] Training...", flush=True)
    import torch
    from tabm import TabMClassifier

    oof = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))

    # Convert to numpy
    X_np = X_train.values.astype(np.float32)
    X_te_np = X_test.values.astype(np.float32)
    y_np = y_train.values

    # Fill NaN
    X_np = np.nan_to_num(X_np, nan=0.0)
    X_te_np = np.nan_to_num(X_te_np, nan=0.0)

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_np, y_np)):
        print(f"  Fold {fold}...", flush=True)

        model = TabMClassifier(
            n_ensemble_members=16,
            n_epochs=50,
            batch_size=256,
            lr=0.001,
            patience=10,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        model.fit(X_np[tr_idx], y_np[tr_idx],
                  X_val=X_np[va_idx], y_val=y_np[va_idx])

        oof[va_idx] = model.predict_proba(X_np[va_idx])[:, 1]
        test_preds += model.predict_proba(X_te_np)[:, 1] / n_folds

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    score = roc_auc_score(y_train, oof)
    print(f"  TabM CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def train_gbdt_seeds(X_train, y_train, X_test, n_seeds=5):
    """Train LGBM with multiple seeds for stacking input."""
    import lightgbm as lgb
    print(f"\n[LGBM {n_seeds} seeds]...", flush=True)

    all_oof = []
    all_test = []

    params = dict(
        n_estimators=3000, learning_rate=0.02, num_leaves=63, max_depth=7,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_samples=20,
    )

    for seed in range(n_seeds):
        oof = np.zeros(len(X_train))
        test_acc = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 7 + 42)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):
            m = lgb.LGBMClassifier(**params, random_state=seed, verbosity=-1, n_jobs=4)
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx],
                  eval_set=[(X_train.iloc[va_idx], y_train.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
            test_acc.append(m.predict_proba(X_test)[:, 1])
            del m; gc.collect()
        all_oof.append(oof)
        all_test.append(np.mean(test_acc, axis=0))
        if seed % 2 == 0:
            print(f"  seed {seed}: {roc_auc_score(y_train, oof):.6f}", flush=True)

    return all_oof, all_test


def three_level_stack(oof_list, y_true, test_list):
    """3-level stacking with Ridge meta-learner."""
    print("\n[3-Level Stacking]...", flush=True)

    # Level 1: OOF predictions as features
    L1_train = np.column_stack(oof_list)
    L1_test = np.column_stack(test_list)
    print(f"  L1 features: {L1_train.shape[1]}", flush=True)

    # Level 2: Ridge on L1 OOF
    L2_oof = np.zeros(len(y_true))
    L2_test = np.zeros(L1_test.shape[0])
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(L1_train, y_true)):
        ridge = Ridge(alpha=1.0)
        ridge.fit(L1_train[tr_idx], y_true.iloc[tr_idx])
        L2_oof[va_idx] = np.clip(ridge.predict(L1_train[va_idx]), 0, 1)
        L2_test += np.clip(ridge.predict(L1_test), 0, 1) / 5

    L2_score = roc_auc_score(y_true, L2_oof)
    print(f"  L2 Ridge CV: {L2_score:.6f}", flush=True)

    # Level 3: Blend L2 with best L1
    best_l1_idx = max(range(len(oof_list)), key=lambda i: roc_auc_score(y_true, oof_list[i]))
    L3_train = np.column_stack([L2_oof, oof_list[best_l1_idx]])
    L3_test_feats = np.column_stack([L2_test, test_list[best_l1_idx]])

    L3_oof = np.zeros(len(y_true))
    L3_test = np.zeros(len(L2_test))
    for fold, (tr_idx, va_idx) in enumerate(kf.split(L3_train, y_true)):
        ridge = Ridge(alpha=0.5)
        ridge.fit(L3_train[tr_idx], y_true.iloc[tr_idx])
        L3_oof[va_idx] = np.clip(ridge.predict(L3_train[va_idx]), 0, 1)
        L3_test += np.clip(ridge.predict(L3_test_feats), 0, 1) / 5

    L3_score = roc_auc_score(y_true, L3_oof)
    print(f"  L3 Ridge CV: {L3_score:.6f}", flush=True)

    return L2_test, L3_test, L2_score, L3_score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 8: TabM + 3-Level Stacking", flush=True)
    print("=" * 70, flush=True)

    print("[1] Preparing data...", flush=True)
    X_train, X_test, y, test_ids = prepare_data()
    print(f"  Features: {X_train.shape[1]}", flush=True)

    # Fill NaN
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    all_oof = []
    all_test = []
    names = []

    # TabM
    try:
        tabm_oof, tabm_test, tabm_score = train_tabm(X_train, y, X_test)
        all_oof.append(tabm_oof)
        all_test.append(tabm_test)
        names.append("tabm")
    except Exception as e:
        print(f"  TabM failed: {e}", flush=True)

    # LGBM 5 seeds
    lgbm_oof, lgbm_test = train_gbdt_seeds(X_train, y, X_test, n_seeds=5)
    for i, (o, t) in enumerate(zip(lgbm_oof, lgbm_test)):
        all_oof.append(o)
        all_test.append(t)
        names.append(f"lgbm_s{i}")

    # XGB 3 seeds
    import xgboost as xgb
    print("\n[XGB 3 seeds]...", flush=True)
    xgb_params = dict(
        n_estimators=5000, learning_rate=0.0063, max_depth=5,
        min_child_weight=6, subsample=0.81, colsample_bytree=0.32,
        reg_alpha=3.5, reg_lambda=1.3, gamma=0.79,
        early_stopping_rounds=300, device="cuda", verbosity=0,
    )
    for seed in range(3):
        oof = np.zeros(len(X_train))
        test_acc = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 13 + 7)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y)):
            m = xgb.XGBClassifier(**xgb_params, random_state=seed, n_jobs=-1)
            m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
                  eval_set=[(X_train.iloc[va_idx], y.iloc[va_idx])], verbose=False)
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
            test_acc.append(m.predict_proba(X_test)[:, 1])
            del m; gc.collect()
        all_oof.append(oof)
        all_test.append(np.mean(test_acc, axis=0))
        names.append(f"xgb_s{seed}")
        print(f"  xgb_s{seed}: {roc_auc_score(y, oof):.6f}", flush=True)

    # LogReg
    print("\n[LogReg]...", flush=True)
    lr_oof = np.zeros(len(X_train))
    lr_test = np.zeros(len(X_test))
    sc = StandardScaler()
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y)):
        X_tr_s = sc.fit_transform(X_train.iloc[tr_idx])
        m = LogisticRegression(C=0.5, max_iter=2000)
        m.fit(X_tr_s, y.iloc[tr_idx])
        lr_oof[va_idx] = m.predict_proba(sc.transform(X_train.iloc[va_idx]))[:, 1]
        lr_test += m.predict_proba(sc.transform(X_test))[:, 1] / 5
    all_oof.append(lr_oof)
    all_test.append(lr_test)
    names.append("logreg")
    print(f"  logreg: {roc_auc_score(y, lr_oof):.6f}", flush=True)

    # === 3-Level Stacking ===
    L2_test, L3_test, L2_score, L3_score = three_level_stack(all_oof, y, all_test)

    # === Simple averages ===
    print("\n[Ensembles]...", flush=True)
    simple = np.mean(all_test, axis=0)
    simple_oof = np.mean(all_oof, axis=0)
    print(f"  Simple avg: {roc_auc_score(y, simple_oof):.6f}", flush=True)

    # === Save ===
    generate_submission(test_ids, simple, "id", "Churn", "submissions/iter8_simple.csv")
    generate_submission(test_ids, L2_test, "id", "Churn", "submissions/iter8_stack_L2.csv")
    generate_submission(test_ids, L3_test, "id", "Churn", "submissions/iter8_stack_L3.csv")

    # Blend with iter6
    try:
        iter6 = pd.read_csv("submissions/iter6_blamerx.csv")["Churn"].values
        blend = 0.5 * simple + 0.5 * iter6
        generate_submission(test_ids, blend, "id", "Churn", "submissions/iter8_blend_i6.csv")
        print("  Saved iter8_blend_i6.csv", flush=True)
    except Exception:
        pass

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 8 - TabM + 3-Level Stacking\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        for i, name in enumerate(names):
            f.write(f"- {name}: {roc_auc_score(y, all_oof[i]):.6f}\n")
        f.write(f"- Simple avg: {roc_auc_score(y, simple_oof):.6f}\n")
        f.write(f"- L2 Stack: {L2_score:.6f}\n")
        f.write(f"- L3 Stack: {L3_score:.6f}\n")
        f.write(f"\n---\n")

    print("\nDONE!", flush=True)


if __name__ == "__main__":
    main()
