"""Iteration 17: WILD approaches.

1. KDE log-likelihood ratio (class-conditional density as feature)
2. KNN graph features (neighborhood churn rate)
3. Price fairness (how much more than identical-service peers)
4. Genetic algorithm to optimize blend weights across ALL submissions
5. Test-time augmentation (add noise, predict multiple times, average)
"""

import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")

import gc, warnings, datetime, glob
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.linear_model import Ridge
import xgboost as xgb
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")


def kde_features(X_tr, X_va, X_te, y_tr, num_cols):
    """KDE log-likelihood ratio: P(x|churn) / P(x|stay)."""
    sc = StandardScaler()
    Xn_tr = sc.fit_transform(X_tr[num_cols].fillna(0))
    Xn_va = sc.transform(X_va[num_cols].fillna(0))
    Xn_te = sc.transform(X_te[num_cols].fillna(0))

    kde_ch = KernelDensity(bandwidth=0.5, kernel="gaussian")
    kde_st = KernelDensity(bandwidth=0.5, kernel="gaussian")
    kde_ch.fit(Xn_tr[y_tr.values == 1])
    kde_st.fit(Xn_tr[y_tr.values == 0])

    for name, Xn, df in [("tr", Xn_tr, X_tr), ("va", Xn_va, X_va), ("te", Xn_te, X_te)]:
        ll_ch = kde_ch.score_samples(Xn)
        ll_st = kde_st.score_samples(Xn)
        df["kde_llr"] = (ll_ch - ll_st).astype("float32")
        df["kde_ll_ch"] = ll_ch.astype("float32")
        df["kde_ll_st"] = ll_st.astype("float32")

    return X_tr, X_va, X_te


def knn_features(X_tr, X_va, X_te, y_tr, feature_cols, k=20):
    """KNN: churn rate among nearest neighbors."""
    sc = StandardScaler()
    Xn_tr = sc.fit_transform(X_tr[feature_cols].fillna(0))
    Xn_va = sc.transform(X_va[feature_cols].fillna(0))
    Xn_te = sc.transform(X_te[feature_cols].fillna(0))

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(Xn_tr)

    y_arr = y_tr.values

    for name, Xn, df in [("va", Xn_va, X_va), ("te", Xn_te, X_te)]:
        dists, idxs = nn.kneighbors(Xn)
        # For val/test, all neighbors are from train (no self)
        neighbor_labels = y_arr[idxs[:, :k]]
        df["knn_churn_rate"] = neighbor_labels.mean(axis=1).astype("float32")
        df["knn_churn_std"] = neighbor_labels.std(axis=1).astype("float32")
        df["knn_mean_dist"] = dists[:, :k].mean(axis=1).astype("float32")

    # For train: skip self (index 0)
    dists, idxs = nn.kneighbors(Xn_tr)
    neighbor_labels = y_arr[idxs[:, 1:k+1]]
    X_tr["knn_churn_rate"] = neighbor_labels.mean(axis=1).astype("float32")
    X_tr["knn_churn_std"] = neighbor_labels.std(axis=1).astype("float32")
    X_tr["knn_mean_dist"] = dists[:, 1:k+1].mean(axis=1).astype("float32")

    return X_tr, X_va, X_te


def price_fairness(X_tr, X_va, X_te):
    """How does this customer's price compare to peers with identical services?"""
    service_cols = ["PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies", "Contract"]
    avail = [c for c in service_cols if c in X_tr.columns]

    config_tr = X_tr[avail].astype(str).agg("|".join, axis=1)
    stats = pd.DataFrame({"cfg": config_tr, "mc": X_tr["MonthlyCharges"]}).groupby("cfg")["mc"].agg(["mean", "std", "median"])

    for df in [X_tr, X_va, X_te]:
        cfg = df[avail].astype(str).agg("|".join, axis=1)
        peer_mean = cfg.map(stats["mean"])
        peer_std = cfg.map(stats["std"]).fillna(1)
        df["price_vs_peers"] = (df["MonthlyCharges"] - peer_mean.fillna(df["MonthlyCharges"])).astype("float32")
        df["price_peer_zscore"] = ((df["MonthlyCharges"] - peer_mean.fillna(df["MonthlyCharges"])) / peer_std.replace(0, 1)).astype("float32")

    return X_tr, X_va, X_te


def contract_tension(df):
    """Mismatches between commitment and payment friction."""
    payment_friction = df["PaymentMethod"].map({
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3
    }).fillna(1)
    contract = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2}).fillna(0)
    df["exit_ease"] = ((3 - payment_friction) * (2 - contract)).astype("float32")
    digital = (df["PaperlessBilling"].map({"Yes": 1, "No": 0}).fillna(0) +
               (df["PaymentMethod"] == "Electronic check").astype(int))
    df["switching_cost"] = (contract + payment_friction - digital).astype("float32")
    monthly_rank = rankdata(df["MonthlyCharges"]) / len(df)
    df["premium_no_lockin"] = (monthly_rank * (contract == 0).astype(int)).astype("float32")
    return df


def prepare_and_train(train_raw, test_raw, y, test_ids):
    """Full pipeline with wild features."""
    print("[1] Base features...", flush=True)
    train = train_raw.copy()
    test = test_raw.copy()

    # ORIG_proba
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"] == "Yes").astype(float); gm = oc.mean()
    for col in list(train.columns):
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            train[f"ORIG_{col}"] = train[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")

    # Standard features
    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"] / df["TotalCharges"].replace(0, 1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
        yes_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        df["svc_count"] = sum((df[c] == "Yes").astype(int) for c in yes_cols if c in df.columns)
        for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v % 10).astype("float32")
            if col != "tenure": df[f"{col}_frac"] = (v - v.astype(int)).astype("float32")
            if col == "tenure": df[f"{col}_m12"] = (v % 12).astype("float32")
        top_cats = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "PaperlessBilling"]
        avail = [c for c in top_cats if c in df.columns]
        for c1, c2 in combinations(avail, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str) + "_" + df[c2].astype(str)
        contract_tension(df)

    # Distribution features
    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    ch_tc = np.sort(orig_tc[orig["Churn"] == "Yes"].values)
    nc_tc = np.sort(orig_tc[orig["Churn"] != "Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly", "charge_ratio", "charges_dev"]

    print(f"  Features before wild: {train.shape[1]}", flush=True)

    # === 20-FOLD WITH WILD FEATURES (computed inside fold) ===
    print("[2] Training 20-fold XGB with wild features...", flush=True)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    all_cat = [c for c in train.columns if train[c].dtype in ("object", "string") or "BG_" in c]

    kf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        y_tr = y.iloc[tr_idx]

        # Encode categoricals
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        X_tr[all_cat] = oe.fit_transform(X_tr[all_cat])
        X_va[all_cat] = oe.transform(X_va[all_cat])
        X_te[all_cat] = oe.transform(X_te[all_cat])

        # Smoothed TE
        gm_fold = y_tr.mean()
        for col in all_cat[:15]:
            tmp = pd.DataFrame({"c": X_tr[col], "y": y_tr.values})
            agg = tmp.groupby("c")["y"].agg(["mean", "count"])
            sm = (agg["count"] * agg["mean"] + 10 * gm_fold) / (agg["count"] + 10)
            X_tr[f"{col}_te"] = X_tr[col].map(sm).fillna(gm_fold).astype("float32")
            X_va[f"{col}_te"] = X_va[col].map(sm).fillna(gm_fold).astype("float32")
            X_te[f"{col}_te"] = X_te[col].map(sm).fillna(gm_fold).astype("float32")

        # === WILD FEATURES (inside fold to avoid leakage) ===
        # KDE log-likelihood ratio
        X_tr, X_va, X_te = kde_features(X_tr, X_va, X_te, y_tr, num_cols)

        # KNN neighborhood churn rate
        knn_cols = num_cols + [c for c in all_cat[:6]]
        X_tr, X_va, X_te = knn_features(X_tr, X_va, X_te, y_tr, knn_cols, k=20)

        # Price fairness (uses train stats only)
        X_tr, X_va, X_te = price_fairness(X_tr, X_va, X_te)

        # Ridge stage 1
        sc = StandardScaler()
        r = Ridge(alpha=10.0)
        r.fit(sc.fit_transform(X_tr.fillna(0)), y_tr)
        X_tr["ridge"] = np.clip(r.predict(sc.transform(X_tr.fillna(0))), 0, 1).astype("float32")
        X_va["ridge"] = np.clip(r.predict(sc.transform(X_va.fillna(0))), 0, 1).astype("float32")
        X_te["ridge"] = np.clip(r.predict(sc.transform(X_te.fillna(0))), 0, 1).astype("float32")

        X_tr = X_tr.fillna(0); X_va = X_va.fillna(0); X_te = X_te.fillna(0)

        model = xgb.XGBClassifier(
            n_estimators=50000, learning_rate=0.0063, max_depth=5,
            min_child_weight=6, subsample=0.81, colsample_bytree=0.32,
            reg_alpha=3.5017, reg_lambda=1.2925, gamma=0.790,
            early_stopping_rounds=500, device="cuda",
            random_state=42, verbosity=0, n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_te)[:, 1] / 20

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
        del model; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"  Wild XGB CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def test_time_augmentation(train, test, y, test_ids, n_aug=5):
    """TTA: add small noise to test features, predict multiple times, average."""
    print("\n[3] Test-time augmentation...", flush=True)

    cat_cols = [c for c in train.select_dtypes(include=["object", "string"]).columns]
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    train_enc = train.copy()
    train_enc[cat_cols] = oe.fit_transform(train_enc[cat_cols])

    # Train one model on all data
    sc = StandardScaler()
    X_all = sc.fit_transform(train_enc.fillna(0))

    model = xgb.XGBClassifier(
        n_estimators=5000, learning_rate=0.01, max_depth=5,
        subsample=0.8, colsample_bytree=0.5, device="cuda",
        random_state=42, verbosity=0,
    )
    model.fit(X_all, y)

    # TTA: predict with noise added
    test_enc = test.copy()
    test_enc[cat_cols] = oe.transform(test_enc[cat_cols])
    X_test_base = sc.transform(test_enc.fillna(0))

    all_preds = []
    for i in range(n_aug):
        noise = np.random.RandomState(i).randn(*X_test_base.shape) * 0.01
        X_noisy = X_test_base + noise
        pred = model.predict_proba(X_noisy)[:, 1]
        all_preds.append(pred)

    # Also predict without noise
    all_preds.append(model.predict_proba(X_test_base)[:, 1])

    tta_pred = np.mean(all_preds, axis=0)
    print(f"  TTA done ({n_aug} augmentations)", flush=True)
    return tta_pred


def genetic_blend_optimization(test_ids):
    """Genetic algorithm to find optimal blend of ALL submission files."""
    print("\n[4] Genetic algorithm blend optimization...", flush=True)

    # Load ALL submissions + public subs
    subs = {}
    for f in glob.glob("submissions/*.csv"):
        name = Path(f).stem
        try:
            df = pd.read_csv(f)
            if "Churn" in df.columns and len(df) == len(test_ids):
                subs[name] = df["Churn"].values
        except Exception:
            pass

    for f in glob.glob("public_subs/*.csv"):
        name = "PUB_" + Path(f).stem
        try:
            df = pd.read_csv(f)
            if "Churn" in df.columns and len(df) == len(test_ids):
                subs[name] = df["Churn"].values
        except Exception:
            pass

    print(f"  Loaded {len(subs)} submissions", flush=True)
    if len(subs) < 3:
        return None

    names = list(subs.keys())
    preds = np.array([subs[n] for n in names])

    # Genetic algorithm: optimize weights to maximize diversity
    # Since we don't have OOF for all, optimize for minimum correlation
    # with the mean (maximize spread)
    n_pop = 100
    n_gen = 200
    n_subs = len(names)

    rng = np.random.RandomState(42)
    pop = rng.dirichlet(np.ones(n_subs), size=n_pop)  # Random weight vectors

    def fitness(weights):
        blend = np.average(preds, axis=0, weights=weights)
        # Maximize: negative of max correlation with any single submission
        # (we want the blend to be different from any single sub)
        corrs = [np.corrcoef(blend, preds[i])[0, 1] for i in range(n_subs)]
        # Also want predictions spread (not all same)
        spread = np.std(blend)
        return -max(corrs) + 0.1 * spread

    for gen in range(n_gen):
        scores = np.array([fitness(w) for w in pop])
        top_idx = np.argsort(scores)[-20:]  # Top 20

        # Crossover + mutation
        new_pop = [pop[i] for i in top_idx]
        while len(new_pop) < n_pop:
            p1, p2 = pop[rng.choice(top_idx)], pop[rng.choice(top_idx)]
            child = (p1 + p2) / 2 + rng.randn(n_subs) * 0.02
            child = np.clip(child, 0, 1)
            child /= child.sum()
            new_pop.append(child)
        pop = np.array(new_pop[:n_pop])

    best = pop[np.argmax([fitness(w) for w in pop])]
    blend = np.average(preds, axis=0, weights=best)

    # Show top weights
    top_w = sorted(zip(names, best), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top weights: {[(n, f'{w:.3f}') for n, w in top_w]}", flush=True)

    return blend


def main():
    print("=" * 70, flush=True)
    print("ITERATION 17: WILD - KDE + KNN + Price + GA + TTA", flush=True)
    print("=" * 70, flush=True)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test.drop(columns=["id"])

    # Train with wild features
    oof, test_preds, cv_score = prepare_and_train(train, test, y, test_ids)
    generate_submission(test_ids, test_preds, "id", "Churn", "submissions/iter17_wild.csv")

    # Cascade with Artem
    artem = pd.read_csv("public_subs/artemevstafyev_cv-auc-0-91930-xgb-cb-blend.csv")["Churn"].values
    corr = np.corrcoef(artem, test_preds)[0, 1]
    print(f"  Corr with Artem: {corr:.4f}", flush=True)
    cascade = artem * 0.95 + test_preds * 0.05
    generate_submission(test_ids, cascade, "id", "Churn", "submissions/iter17_cascade.csv")

    # TTA
    tta_preds = test_time_augmentation(train, test, y, test_ids)
    generate_submission(test_ids, tta_preds, "id", "Churn", "submissions/iter17_tta.csv")

    # GA blend optimization
    ga_blend = genetic_blend_optimization(test_ids)
    if ga_blend is not None:
        generate_submission(test_ids, ga_blend, "id", "Churn", "submissions/iter17_ga.csv")

    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 17 - WILD (KDE+KNN+Price+GA+TTA)\n")
        f.write(f"**CV**: {cv_score:.6f}, **Corr**: {corr:.4f}\n\n---\n")

    print("\nDONE!", flush=True)


if __name__ == "__main__":
    main()
