"""Iteration 13: Build a maximally diverse model.

Our iter6 correlates 0.9935 with Artem. We need <0.99 correlation
while maintaining 0.916+ LB. This means a FUNDAMENTALLY different approach.

Deotte's LogReg (pair TE + logit3) correlates 0.9874 with Artem.
His MLP correlates 0.9890. These are the most diverse strong models.

Strategy: Reproduce Deotte's approach locally:
1. Pair TE with logit3 features → LogisticRegression (his Model A)
2. CatBoost with different hyperparams (NOT XGBoost like everyone else)
3. Simple MLP with embeddings
4. Average our 3 diverse models
5. Then blend with Artem's public output

Target: correlation with Artem < 0.99, LB > 0.916
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
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")

N_FOLDS = 20


def load_and_prepare():
    """Load raw data with minimal processing."""
    print("[1] Loading...", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    # Identify original feature columns
    cat_cols = train.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = train.select_dtypes(include=["number"]).columns.tolist()

    print(f"  {len(cat_cols)} cats, {len(num_cols)} nums", flush=True)
    return train, test, y, test_ids, cat_cols, num_cols


def model_a_pair_te_logreg(train, test, y, cat_cols, num_cols):
    """Deotte's Model A: All C(n,2) feature pairs → TE → logit3 → LogReg.

    This is fundamentally different from tree-based approaches.
    """
    print("\n[Model A] Pair TE → Logit3 → LogReg...", flush=True)

    all_features = cat_cols + num_cols
    n_pairs = len(list(combinations(range(len(all_features)), 2)))
    print(f"  {len(all_features)} features → {n_pairs} pairs", flush=True)

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        gm = y.iloc[tr_idx].mean()

        X_tr_feats, X_va_feats, X_te_feats = [], [], []

        for f1, f2 in combinations(all_features, 2):
            # Create pair key
            p_tr = train.iloc[tr_idx][f1].astype(str) + "_" + train.iloc[tr_idx][f2].astype(str)
            p_va = train.iloc[va_idx][f1].astype(str) + "_" + train.iloc[va_idx][f2].astype(str)
            p_te = test[f1].astype(str) + "_" + test[f2].astype(str)

            # Target encode with smoothing=0 (Deotte uses smooth=0)
            tmp = pd.DataFrame({"p": p_tr, "y": y.iloc[tr_idx].values})
            te_map = tmp.groupby("p")["y"].mean()

            te_tr = np.clip(p_tr.map(te_map).fillna(gm).values, 1e-6, 1 - 1e-6)
            te_va = np.clip(p_va.map(te_map).fillna(gm).values, 1e-6, 1 - 1e-6)
            te_te = np.clip(p_te.map(te_map).fillna(gm).values, 1e-6, 1 - 1e-6)

            # Logit3: z, z^2, z^3
            for arr_list, te_arr in [(X_tr_feats, te_tr), (X_va_feats, te_va), (X_te_feats, te_te)]:
                z = logit(te_arr)
                arr_list.extend([z, z ** 2, z ** 3])

        X_tr_m = np.nan_to_num(np.column_stack(X_tr_feats).astype(np.float32), nan=0, posinf=10, neginf=-10)
        X_va_m = np.nan_to_num(np.column_stack(X_va_feats).astype(np.float32), nan=0, posinf=10, neginf=-10)
        X_te_m = np.nan_to_num(np.column_stack(X_te_feats).astype(np.float32), nan=0, posinf=10, neginf=-10)

        sc = StandardScaler()
        model = LogisticRegression(C=0.5, max_iter=4000, solver="lbfgs")
        model.fit(sc.fit_transform(X_tr_m), y.iloc[tr_idx])
        oof[va_idx] = model.predict_proba(sc.transform(X_va_m))[:, 1]
        test_preds += model.predict_proba(sc.transform(X_te_m))[:, 1] / N_FOLDS

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)

    score = roc_auc_score(y, oof)
    print(f"  Model A CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def model_b_catboost(train, test, y, cat_cols, num_cols):
    """CatBoost with native categoricals (different from XGB/LGBM)."""
    print("\n[Model B] CatBoost native categoricals...", flush=True)

    # Add ORIG_proba
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"] == "Yes").astype(float)
    gm = oc.mean()

    train_cb = train.copy()
    test_cb = test.copy()
    for col in train.columns:
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            proba = tmp.groupby(col)["_y"].mean()
            train_cb[f"ORIG_{col}"] = train_cb[col].map(proba).fillna(gm).astype("float32")
            test_cb[f"ORIG_{col}"] = test_cb[col].map(proba).fillna(gm).astype("float32")

    # Encode cats as integers for CatBoost
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_cb[cat_cols] = oe.fit_transform(train_cb[cat_cols]).astype(int)
    test_cb[cat_cols] = oe.transform(test_cb[cat_cols]).astype(int)

    cat_indices = [train_cb.columns.get_loc(c) for c in cat_cols]

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train_cb, y)):
        model = CatBoostClassifier(
            iterations=5000, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, subsample=0.8, random_seed=42,
            verbose=0, eval_metric="AUC", task_type="CPU",
            early_stopping_rounds=200,
            cat_features=cat_indices,
        )
        model.fit(
            train_cb.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(train_cb.iloc[va_idx], y.iloc[va_idx]),
            verbose=0,
        )
        oof[va_idx] = model.predict_proba(train_cb.iloc[va_idx])[:, 1]
        test_preds += model.predict_proba(test_cb)[:, 1] / N_FOLDS

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
        del model; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"  Model B CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 13: Maximally Diverse Model", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols, num_cols = load_and_prepare()

    results = {}

    # Model A: Pair TE LogReg (most diverse from GBDT)
    a_oof, a_test, a_cv = model_a_pair_te_logreg(train, test, y, cat_cols, num_cols)
    results["pair_te_logreg"] = {"oof": a_oof, "test": a_test, "cv": a_cv}

    # Model B: CatBoost native (different from XGB)
    b_oof, b_test, b_cv = model_b_catboost(train, test, y, cat_cols, num_cols)
    results["catboost"] = {"oof": b_oof, "test": b_test, "cv": b_cv}

    # === Check diversity vs Artem ===
    print("\n=== Diversity check ===", flush=True)
    artem = pd.read_csv("public_subs/artemevstafyev_cv-auc-0-91930-xgb-cb-blend.csv")["Churn"].values
    for name, r in results.items():
        corr = np.corrcoef(artem, r["test"])[0, 1]
        print(f"  {name} vs Artem: corr={corr:.4f}", flush=True)

    # === Ensembles ===
    print("\n=== Ensembles ===", flush=True)

    # Our diverse 2-model average
    our_diverse = (a_test + b_test) / 2
    our_diverse_oof = (a_oof + b_oof) / 2
    div_corr = np.corrcoef(artem, our_diverse)[0, 1]
    print(f"  Our diverse avg: CV={roc_auc_score(y, our_diverse_oof):.6f}, corr_artem={div_corr:.4f}", flush=True)

    # Blend our diverse model with Artem (should help because lower correlation!)
    for w in [0.3, 0.4, 0.5]:
        blend = (1 - w) * artem + w * our_diverse
        print(f"  Artem {1-w:.0%} + Ours {w:.0%}: correlation matters, not CV here", flush=True)
        generate_submission(test_ids, blend, "id", "Churn", f"submissions/diverse13_{int(w*100)}.csv")

    # Save standalone
    generate_submission(test_ids, our_diverse, "id", "Churn", "submissions/iter13_diverse.csv")
    generate_submission(test_ids, a_test, "id", "Churn", "submissions/iter13_pair_te.csv")
    generate_submission(test_ids, b_test, "id", "Churn", "submissions/iter13_catboost.csv")

    # Best cascade: Artem base + our diverse model at higher weight
    cascade = artem.copy()
    cascade = cascade * 0.85 + our_diverse * 0.15  # 15% our diverse model (higher than 5%)
    cascade = cascade * 0.95 + pd.read_csv("public_subs/blamerx_s6e3-ridge-xgb-n-gram-0-91927-cv.csv")["Churn"].values * 0.05
    generate_submission(test_ids, cascade, "id", "Churn", "submissions/diverse13_cascade.csv")

    # Log
    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 13 - Maximally Diverse Model\n")
        f.write(f"**Date**: {datetime.datetime.now().isoformat()}\n\n")
        for name, r in results.items():
            corr = np.corrcoef(artem, r["test"])[0, 1]
            f.write(f"- {name}: CV={r['cv']:.6f}, corr_artem={corr:.4f}\n")
        f.write(f"- Diverse avg: CV={roc_auc_score(y, our_diverse_oof):.6f}, corr={div_corr:.4f}\n")
        f.write(f"\n---\n")

    print("\n" + "=" * 70, flush=True)
    print("ITERATION 13 COMPLETE", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
