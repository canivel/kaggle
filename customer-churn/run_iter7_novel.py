"""Iteration 7: Novel approaches from cutting-edge research.

Based on arxiv research, the 3 highest-impact techniques:
1. Multi-seed (20 seeds) for variance reduction (tighten CV-LB gap)
2. Hill-climbing ensemble selection (instead of naive averaging)
3. TabM neural model for non-tree diversity

The CV-LB gap (0.0028) is the real enemy. More seeds + smarter
ensemble selection should close it.
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
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")

# Reuse iter6 feature engineering
TOP_CATS = ["Contract", "InternetService", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "PaperlessBilling"]
NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]


def prepare_data():
    """Full feature engineering (same as iter6)."""
    print("[1] Loading data...", flush=True)
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
        df["pctrank_ch_TC"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc_TC"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")
        df["pctrank_gap_TC"] = df["pctrank_ch_TC"] - df["pctrank_nc_TC"]

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
            if col != "tenure":
                df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure":
                df[f"{col}_m12"] = (v % 12).astype("float32")
        avail_cats = [c for c in TOP_CATS if c in df.columns]
        for c1, c2 in combinations(avail_cats, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
        for c1, c2, c3 in combinations(avail_cats[:4], 3):
            df[f"TG_{c1}_{c2}_{c3}"] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns if "BG_" not in c and "TG_" not in c]
    ngram_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns if "BG_" in c or "TG_" in c]
    print(f"  Features: {train.shape[1]}, Cats: {len(cat_cols)}, Ngrams: {len(ngram_cols)}", flush=True)
    return train, test, y, test_ids, cat_cols, ngram_cols


def fold_encode(X_tr, X_va, X_te, y_tr, cat_cols, ngram_cols):
    """In-fold encoding with smoothed TE."""
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

    for col in cat_cols:
        freq = X_tr[col].value_counts(normalize=True)
        X_tr[f"{col}_f"] = X_tr[col].map(freq).fillna(0).astype("float32")
        X_va[f"{col}_f"] = X_va[col].map(freq).fillna(0).astype("float32")
        X_te[f"{col}_f"] = X_te[col].map(freq).fillna(0).astype("float32")
    return X_tr, X_va, X_te


def ridge_stage1(X_tr, X_va, X_te, y_tr):
    """Ridge predictions as features."""
    sc = StandardScaler()
    r = Ridge(alpha=10.0)
    r.fit(sc.fit_transform(X_tr), y_tr)
    X_tr["ridge"] = np.clip(r.predict(sc.transform(X_tr)), 0, 1).astype("float32")
    X_va["ridge"] = np.clip(r.predict(sc.transform(X_va)), 0, 1).astype("float32")
    X_te["ridge"] = np.clip(r.predict(sc.transform(X_te)), 0, 1).astype("float32")
    return X_tr, X_va, X_te


def hill_climb(oof_dict, y_true, test_dict, max_rounds=100):
    """Greedy forward hill-climbing ensemble selection."""
    print("\n=== HILL CLIMBING ===", flush=True)
    names = list(oof_dict.keys())
    best_name = max(names, key=lambda n: roc_auc_score(y_true, oof_dict[n]))
    selected = [best_name]
    sel_oof = oof_dict[best_name].copy()
    sel_test = test_dict[best_name].copy()
    best_score = roc_auc_score(y_true, sel_oof)
    print(f"  Start: {best_name} = {best_score:.6f}", flush=True)

    for rnd in range(max_rounds):
        best_imp = 0
        best_cand = None
        n = len(selected)
        for name in names:
            cand_oof = (sel_oof * n + oof_dict[name]) / (n + 1)
            cand_score = roc_auc_score(y_true, cand_oof)
            imp = cand_score - best_score
            if imp > best_imp:
                best_imp = imp
                best_cand = name
        if best_cand is None or best_imp < 1e-7:
            break
        selected.append(best_cand)
        n2 = len(selected)
        sel_oof = (sel_oof * (n2 - 1) + oof_dict[best_cand]) / n2
        sel_test = (sel_test * (n2 - 1) + test_dict[best_cand]) / n2
        best_score = roc_auc_score(y_true, sel_oof)
        print(f"  +{best_cand} = {best_score:.6f} (+{best_imp:.7f})", flush=True)

    print(f"  Final: {len(selected)} selections, AUC={best_score:.6f}", flush=True)
    return sel_test, best_score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 7: 20-Seed + Hill Climbing + Novel Models", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols, ngram_cols = prepare_data()

    oof_dict = {}
    test_dict = {}

    # === 20-SEED LGBM (best config from iter6) ===
    import lightgbm as lgb
    lgbm_params = dict(
        n_estimators=3000, learning_rate=0.02, num_leaves=63, max_depth=7,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
        min_child_samples=20,
    )

    print("\n[2] Training 20-seed LGBM...", flush=True)
    for seed in range(20):
        oof = np.zeros(len(train))
        test_acc = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 7 + 42)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
            X_tr = train.iloc[tr_idx].copy()
            X_va = train.iloc[va_idx].copy()
            X_te = test.copy()
            X_tr, X_va, X_te = fold_encode(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)
            X_tr, X_va, X_te = ridge_stage1(X_tr, X_va, X_te, y.iloc[tr_idx])
            m = lgb.LGBMClassifier(**lgbm_params, random_state=seed, verbosity=-1, n_jobs=4)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
            oof[va_idx] = m.predict_proba(X_va)[:, 1]
            test_acc.append(m.predict_proba(X_te)[:, 1])
            del m; gc.collect()
        name = f"lgbm_s{seed}"
        cv = roc_auc_score(y, oof)
        oof_dict[name] = oof
        test_dict[name] = np.mean(test_acc, axis=0)
        if seed % 5 == 0:
            print(f"  {name}: {cv:.6f}", flush=True)
        del test_acc; gc.collect()

    # === 10-SEED XGB (BlamerX params) ===
    import xgboost as xgb
    xgb_params = dict(
        n_estimators=5000, learning_rate=0.0063, max_depth=5,
        min_child_weight=6, subsample=0.81, colsample_bytree=0.32,
        reg_alpha=3.5, reg_lambda=1.3, gamma=0.79,
        early_stopping_rounds=300, device="cuda", verbosity=0,
    )

    print("\n[3] Training 10-seed XGB (GPU)...", flush=True)
    for seed in range(10):
        oof = np.zeros(len(train))
        test_acc = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed * 13 + 7)
        for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
            X_tr = train.iloc[tr_idx].copy()
            X_va = train.iloc[va_idx].copy()
            X_te = test.copy()
            X_tr, X_va, X_te = fold_encode(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)
            X_tr, X_va, X_te = ridge_stage1(X_tr, X_va, X_te, y.iloc[tr_idx])
            m = xgb.XGBClassifier(**xgb_params, random_state=seed, n_jobs=-1)
            m.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
            oof[va_idx] = m.predict_proba(X_va)[:, 1]
            test_acc.append(m.predict_proba(X_te)[:, 1])
            del m; gc.collect()
        name = f"xgb_s{seed}"
        cv = roc_auc_score(y, oof)
        oof_dict[name] = oof
        test_dict[name] = np.mean(test_acc, axis=0)
        if seed % 3 == 0:
            print(f"  {name}: {cv:.6f}", flush=True)
        del test_acc; gc.collect()

    # === LOGISTIC REGRESSION for diversity ===
    print("\n[4] LogReg...", flush=True)
    oof_lr = np.zeros(len(train))
    test_lr = np.zeros(len(test))
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        X_tr, X_va, X_te = fold_encode(X_tr, X_va, X_te, y.iloc[tr_idx], cat_cols, ngram_cols)
        sc = StandardScaler()
        m = LogisticRegression(C=0.5, max_iter=2000)
        m.fit(sc.fit_transform(X_tr), y.iloc[tr_idx])
        oof_lr[va_idx] = m.predict_proba(sc.transform(X_va))[:, 1]
        test_lr += m.predict_proba(sc.transform(X_te))[:, 1] / 5
    oof_dict["logreg"] = oof_lr
    test_dict["logreg"] = test_lr
    print(f"  logreg: {roc_auc_score(y, oof_lr):.6f}", flush=True)

    # === SUMMARY ===
    print(f"\n=== {len(oof_dict)} models trained ===", flush=True)
    scores = {n: roc_auc_score(y, o) for n, o in oof_dict.items()}
    for n in sorted(scores, key=scores.get, reverse=True)[:10]:
        print(f"  {n}: {scores[n]:.6f}", flush=True)

    # === SIMPLE AVERAGES ===
    print("\n[5] Ensembles...", flush=True)
    all_avg = np.mean(list(test_dict.values()), axis=0)
    all_oof = np.mean(list(oof_dict.values()), axis=0)
    print(f"  Simple avg ({len(oof_dict)} models): {roc_auc_score(y, all_oof):.6f}", flush=True)

    # Top-10 models by CV
    top10 = sorted(scores, key=scores.get, reverse=True)[:10]
    top10_avg = np.mean([test_dict[n] for n in top10], axis=0)
    top10_oof = np.mean([oof_dict[n] for n in top10], axis=0)
    print(f"  Top-10 avg: {roc_auc_score(y, top10_oof):.6f}", flush=True)

    # === HILL CLIMBING ===
    hc_test, hc_score = hill_climb(oof_dict, y, test_dict)

    # === SAVE SUBMISSIONS ===
    print("\n[6] Saving submissions...", flush=True)
    generate_submission(test_ids, all_avg, "id", "Churn", "submissions/iter7_all_avg.csv")
    generate_submission(test_ids, top10_avg, "id", "Churn", "submissions/iter7_top10.csv")
    generate_submission(test_ids, hc_test, "id", "Churn", "submissions/iter7_hillclimb.csv")

    # Rank-calibrate hill climb with iter6
    try:
        iter6 = pd.read_csv("submissions/iter6_blamerx.csv")["Churn"].values
        r_hc = rankdata(hc_test) / len(hc_test)
        r_i6 = rankdata(iter6) / len(iter6)
        blend = 0.7 * r_hc + 0.3 * r_i6
        # Map back to probability scale using hc distribution
        order = np.argsort(blend)
        calibrated = np.empty_like(blend)
        calibrated[order] = np.sort(hc_test)
        generate_submission(test_ids, calibrated, "id", "Churn", "submissions/iter7_hc_i6_blend.csv")
        print("  Saved: iter7_hc_i6_blend.csv", flush=True)
    except Exception as e:
        print(f"  Blend failed: {e}", flush=True)

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 7 - 20-Seed + Hill Climbing\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"- 20-seed LGBM + 10-seed XGB + LogReg = {len(oof_dict)} models\n")
        f.write(f"- Simple avg: {roc_auc_score(y, all_oof):.6f}\n")
        f.write(f"- Top-10 avg: {roc_auc_score(y, top10_oof):.6f}\n")
        f.write(f"- **Hill climb**: {hc_score:.6f}\n")
        f.write(f"\n---\n")

    print("\n" + "=" * 70, flush=True)
    print("ITERATION 7 COMPLETE", flush=True)
    print(f"  Hill climb: {hc_score:.6f}", flush=True)
    print(f"  All avg: {roc_auc_score(y, all_oof):.6f}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
