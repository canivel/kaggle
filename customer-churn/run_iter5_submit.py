"""Iter5 submission generator - minimal, focused.

Just trains the 2 best configs and generates ensemble submission.
- LGBM 3-seed (CV 0.918007)
- XGB BlamerX params (CV 0.917782)
"""

import sys
sys.path.insert(0, "../kaggle-agent/src")

import gc
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")


def prepare():
    print("Loading...", flush=True)
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
            p = tmp.groupby(col)["_y"].mean()
            train[f"ORIG_{col}"] = train[col].map(p).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(p).fillna(gm).astype("float32")

    # Distribution features
    for col in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if col not in orig.columns:
            continue
        ch = orig.loc[orig["Churn"] == "Yes", col].dropna()
        nc = orig.loc[orig["Churn"] != "Yes", col].dropna()
        if col == "TotalCharges":
            ch = pd.to_numeric(ch, errors="coerce").dropna()
            nc = pd.to_numeric(nc, errors="coerce").dropna()
        ch = np.sort(ch.values)
        nc = np.sort(nc.values)
        for df in [train, test]:
            v = df[col].values
            df[f"{col}_pctrank_ch"] = (np.searchsorted(ch, v) / len(ch)).astype("float32")
            df[f"{col}_pctrank_nc"] = (np.searchsorted(nc, v) / len(nc)).astype("float32")

    # Static features
    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        avail = [c for c in yes_cols if c in df.columns]
        df["svc_count"] = sum((df[c] == "Yes").astype(int) for c in avail)
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v % 10).astype("float32")
            df[f"{col}_m100"] = (v % 100).astype("float32")
            df[f"{col}_d1"] = v.astype(str).str[0].astype("float32")
            if col != "tenure":
                df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure":
                df[f"{col}_m12"] = (v % 12).astype("float32")
        top = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "PaperlessBilling"]
        avail = [c for c in top if c in df.columns]
        for c1, c2 in combinations(avail, 2):
            df[f"{c1}_x_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
        for c1, c2, c3 in combinations(avail[:4], 3):
            df[f"{c1}_x_{c2}_x_{c3}"] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns if "_x_" not in c]
    ngram_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns if "_x_" in c]
    print(f"Features: {train.shape[1]}", flush=True)
    return train, test, y, test_ids, cat_cols, ngram_cols


def fold_enc(X_tr, X_va, X_te, y_tr, cat_cols, ngram_cols):
    all_cats = cat_cols + ngram_cols
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    X_tr[all_cats] = oe.fit_transform(X_tr[all_cats])
    X_va[all_cats] = oe.transform(X_va[all_cats])
    X_te[all_cats] = oe.transform(X_te[all_cats])
    te_cols = cat_cols[:8] + ngram_cols[:10]
    if te_cols:
        te = TargetEncoder(cv=5, smooth="auto")
        names = [f"{c}_te" for c in te_cols]
        X_tr[names] = te.fit_transform(X_tr[te_cols], y_tr)
        X_va[names] = te.transform(X_va[te_cols])
        X_te[names] = te.transform(X_te[te_cols])
    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[f"{col}_f"] = X_tr[col].map(freq).fillna(0).astype("float32")
        X_va[f"{col}_f"] = X_va[col].map(freq).fillna(0).astype("float32")
        X_te[f"{col}_f"] = X_te[col].map(freq).fillna(0).astype("float32")
    return X_tr, X_va, X_te


def main():
    train, test, y, test_ids, cat_cols, ngram_cols = prepare()
    all_test = []

    # LGBM with 3 seeds
    import lightgbm as lgb
    params = dict(n_estimators=3000, learning_rate=0.02, num_leaves=63, max_depth=7,
                  subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                  min_child_samples=20)

    for seed in [42, 11, 99]:
        print(f"LGBM seed={seed}...", flush=True)
        oof = np.zeros(len(train))
        test_acc = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
            X_tr = train.iloc[tr_idx].copy()
            X_va = train.iloc[va_idx].copy()
            X_te = test.copy()
            X_tr, X_va, X_te = fold_enc(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)
            m = lgb.LGBMClassifier(**params, random_state=seed, verbosity=-1, n_jobs=4)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va_idx] = m.predict_proba(X_va)[:, 1]
            test_acc.append(m.predict_proba(X_te)[:, 1])
            del m, X_tr, X_va, X_te; gc.collect()
        print(f"  CV: {roc_auc_score(y, oof):.6f}", flush=True)
        all_test.append(np.mean(test_acc, axis=0))
        del test_acc; gc.collect()

    # XGB with BlamerX params
    import xgboost as xgb
    xgb_params = dict(n_estimators=3000, learning_rate=0.01, max_depth=5,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.35,
                      reg_alpha=3.5, reg_lambda=1.3, gamma=0.8,
                      early_stopping_rounds=200, tree_method="hist")

    for seed in [42, 11]:
        print(f"XGB seed={seed}...", flush=True)
        oof = np.zeros(len(train))
        test_acc = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
            X_tr = train.iloc[tr_idx].copy()
            X_va = train.iloc[va_idx].copy()
            X_te = test.copy()
            X_tr, X_va, X_te = fold_enc(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)
            m = xgb.XGBClassifier(**xgb_params, random_state=seed, verbosity=0, n_jobs=4)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
            oof[va_idx] = m.predict_proba(X_va)[:, 1]
            test_acc.append(m.predict_proba(X_te)[:, 1])
            del m, X_tr, X_va, X_te; gc.collect()
        print(f"  CV: {roc_auc_score(y, oof):.6f}", flush=True)
        all_test.append(np.mean(test_acc, axis=0))
        del test_acc; gc.collect()

    # Ensemble
    avg = np.mean(all_test, axis=0)
    generate_submission(test_ids, avg, "id", "Churn", "submissions/iter5_lgbm3_xgb2.csv")
    print("Saved: iter5_lgbm3_xgb2.csv", flush=True)
    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
