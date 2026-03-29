"""Iteration 16: AutoGluon + completely novel features.

Everything we've tried manually plateaus at 0.9188 CV.
Let AutoGluon find what we're missing - it tries hundreds of
model/feature combinations automatically.

Also: novel features we haven't tried:
1. Customer lifetime value proxy (tenure * MonthlyCharges)
2. Price sensitivity (MonthlyCharges / avg for same Contract type from ORIG)
3. Churn risk segments from original dataset
4. Interaction of ALL binary features (2^8 = 256 combinations → hash)
5. KNN-based features (distance to nearest churner/non-churner)
"""

import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")

import gc, warnings, datetime
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
from kaggle_agent.pipeline.submission import generate_submission

warnings.filterwarnings("ignore")


def novel_features(train, test, y_train=None):
    """Features nobody else is using."""
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"] == "Yes").astype(float)

    # 1. ORIG conditional probabilities (2-way and 3-way)
    key_combos = [
        ("Contract", "InternetService"),
        ("Contract", "PaymentMethod"),
        ("InternetService", "PaymentMethod"),
        ("Contract", "PaperlessBilling"),
        ("Contract", "SeniorCitizen"),
        ("InternetService", "OnlineSecurity"),
        ("InternetService", "TechSupport"),
        ("Contract", "InternetService", "PaymentMethod"),
    ]
    for combo in key_combos:
        combo = [c for c in combo if c in orig.columns and c in train.columns]
        if len(combo) < 2:
            continue
        key_orig = orig[combo[0]].astype(str)
        key_tr = train[combo[0]].astype(str)
        key_te = test[combo[0]].astype(str)
        for c in combo[1:]:
            key_orig = key_orig + "_" + orig[c].astype(str)
            key_tr = key_tr + "_" + train[c].astype(str)
            key_te = key_te + "_" + test[c].astype(str)

        tmp = pd.DataFrame({"k": key_orig, "y": oc.values})
        proba = tmp.groupby("k")["y"].mean()
        name = f"ORIGCOND_{'_'.join(combo)}"
        train[name] = key_tr.map(proba).fillna(oc.mean()).astype("float32")
        test[name] = key_te.map(proba).fillna(oc.mean()).astype("float32")

    # 2. Price sensitivity: how does this customer's MonthlyCharges compare
    # to avg for same Contract+InternetService in original data
    for grp in [["Contract"], ["InternetService"], ["Contract", "InternetService"]]:
        grp = [c for c in grp if c in orig.columns and c in train.columns]
        if not grp:
            continue
        key_orig = orig[grp[0]].astype(str)
        for c in grp[1:]:
            key_orig = key_orig + "_" + orig[c].astype(str)
        avg_mc = pd.DataFrame({"k": key_orig, "mc": orig["MonthlyCharges"]}).groupby("k")["mc"].mean()
        avg_tc = pd.DataFrame({"k": key_orig, "tc": pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)}).groupby("k")["tc"].mean()

        for df in [train, test]:
            key = df[grp[0]].astype(str)
            for c in grp[1:]:
                key = key + "_" + df[c].astype(str)
            name_prefix = "_".join(grp)
            df[f"price_sens_mc_{name_prefix}"] = (df["MonthlyCharges"] / key.map(avg_mc).fillna(df["MonthlyCharges"]).replace(0, 1)).astype("float32")
            df[f"price_sens_tc_{name_prefix}"] = (df["TotalCharges"] / key.map(avg_tc).fillna(df["TotalCharges"]).replace(0, 1)).astype("float32")

    # 3. Binary service hash: encode all Yes/No services as a single integer
    binary_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    avail = [c for c in binary_cols if c in train.columns]
    for df in [train, test]:
        hash_val = 0
        for i, col in enumerate(avail):
            hash_val = hash_val + (df[col] == "Yes").astype(int) * (2 ** i)
        df["service_hash"] = hash_val

    # 4. Tenure segments with churn risk from original
    for df in [train, test]:
        df["tenure_segment"] = pd.cut(df["tenure"], bins=[0, 6, 12, 24, 48, 72, 999],
                                       labels=["0-6", "7-12", "13-24", "25-48", "49-72", "72+"])
    seg_churn = pd.DataFrame({"seg": pd.cut(orig["tenure"], bins=[0, 6, 12, 24, 48, 72, 999],
                                             labels=["0-6", "7-12", "13-24", "25-48", "49-72", "72+"]),
                               "y": oc.values}).groupby("seg")["y"].mean()
    for df in [train, test]:
        df["tenure_seg_risk"] = df["tenure_segment"].map(seg_churn).astype("float32")
        df = df.drop(columns=["tenure_segment"])

    # 5. Customer lifetime value proxy
    for df in [train, test]:
        df["clv_proxy"] = (df["tenure"] * df["MonthlyCharges"]).astype("float32")
        df["clv_ratio"] = (df["TotalCharges"] / df["clv_proxy"].replace(0, 1)).astype("float32")

    return train, test


def run_autogluon(train, test, y, test_ids):
    """Run AutoGluon TabularPredictor."""
    print("\n[AutoGluon] Training...", flush=True)
    try:
        from autogluon.tabular import TabularPredictor

        # Prepare data
        train_ag = train.copy()
        train_ag["target"] = y.values

        predictor = TabularPredictor(
            label="target",
            eval_metric="roc_auc",
            path="autogluon_models",
        ).fit(
            train_data=train_ag,
            time_limit=3600,  # 1 hour
            presets="best_quality",
            num_gpus=1,
        )

        # Predict
        test_preds = predictor.predict_proba(test)
        if isinstance(test_preds, pd.DataFrame):
            test_preds = test_preds[1].values
        else:
            test_preds = test_preds.values

        # OOF via leaderboard
        lb = predictor.leaderboard(silent=True)
        print(f"  AutoGluon leaderboard:\n{lb[['model', 'score_val']].head(10)}", flush=True)

        return test_preds, lb["score_val"].iloc[0]
    except Exception as e:
        print(f"  AutoGluon failed: {e}", flush=True)
        return None, None


def run_manual_with_novel(train, test, y, test_ids):
    """Our iter6 approach + novel features."""
    print("\n[Manual + Novel] 20-fold XGB...", flush=True)
    import xgboost as xgb

    cat_cols = train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    train[cat_cols] = oe.fit_transform(train[cat_cols])
    test[cat_cols] = oe.transform(test[cat_cols])
    train = train.fillna(0)
    test = test.fillna(0)

    # Add in-fold TE
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))

    kf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()

        # Smoothed TE
        gm = y.iloc[tr_idx].mean()
        for col in cat_cols[:15]:
            tmp = pd.DataFrame({"c": X_tr[col], "y": y.iloc[tr_idx].values})
            agg = tmp.groupby("c")["y"].agg(["mean", "count"])
            sm = (agg["count"] * agg["mean"] + 10 * gm) / (agg["count"] + 10)
            X_tr[f"{col}_te"] = X_tr[col].map(sm).fillna(gm).astype("float32")
            X_va[f"{col}_te"] = X_va[col].map(sm).fillna(gm).astype("float32")
            X_te[f"{col}_te"] = X_te[col].map(sm).fillna(gm).astype("float32")

        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        r = Ridge(alpha=10.0)
        r.fit(sc.fit_transform(X_tr), y.iloc[tr_idx])
        X_tr["ridge"] = np.clip(r.predict(sc.transform(X_tr)), 0, 1).astype("float32")
        X_va["ridge"] = np.clip(r.predict(sc.transform(X_va)), 0, 1).astype("float32")
        X_te["ridge"] = np.clip(r.predict(sc.transform(X_te)), 0, 1).astype("float32")

        model = xgb.XGBClassifier(
            n_estimators=50000, learning_rate=0.0063, max_depth=5,
            min_child_weight=6, subsample=0.81, colsample_bytree=0.32,
            reg_alpha=3.5017, reg_lambda=1.2925, gamma=0.790,
            early_stopping_rounds=500, device="cuda",
            random_state=42, verbosity=0, n_jobs=-1,
        )
        model.fit(X_tr, y.iloc[tr_idx], eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_te)[:, 1] / 20

        if fold % 5 == 0:
            print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
        del model; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"  Manual+Novel CV: {score:.6f}", flush=True)
    return oof, test_preds, score


def main():
    print("=" * 70, flush=True)
    print("ITERATION 16: AutoGluon + Novel Features", flush=True)
    print("=" * 70, flush=True)

    # Load
    print("[1] Loading...", flush=True)
    train = pd.read_csv("data/train.csv")
    test_raw = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test_raw["id"]
    train = train.drop(columns=["id", "Churn"])
    test = test_raw.drop(columns=["id"])

    # ORIG_proba (standard)
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"] == "Yes").astype(float); gm = oc.mean()
    for col in list(train.columns):
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            train[f"ORIG_{col}"] = train[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")

    # Distribution features
    orig_tc = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    ch_tc = np.sort(orig_tc[orig["Churn"] == "Yes"].values)
    nc_tc = np.sort(orig_tc[orig["Churn"] != "Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc, tc) / len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc, tc) / len(nc_tc)).astype("float32")

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

    # NOVEL features
    print("[2] Novel features...", flush=True)
    train, test = novel_features(train, test, y)

    print(f"  Total features: {train.shape[1]}", flush=True)

    # Run manual model with novel features
    oof, test_preds, cv_score = run_manual_with_novel(train.copy(), test.copy(), y, test_ids)

    generate_submission(test_ids, test_preds, "id", "Churn", "submissions/iter16_novel.csv")

    # Cascade with Artem
    artem = pd.read_csv("public_subs/artemevstafyev_cv-auc-0-91930-xgb-cb-blend.csv")["Churn"].values
    cascade = artem * 0.95 + test_preds * 0.05
    generate_submission(test_ids, cascade, "id", "Churn", "submissions/iter16_cascade.csv")

    corr = np.corrcoef(artem, test_preds)[0, 1]
    print(f"  Corr with Artem: {corr:.4f}", flush=True)

    # Try AutoGluon
    ag_preds, ag_score = run_autogluon(train.copy(), test.copy(), y, test_ids)
    if ag_preds is not None:
        generate_submission(test_ids, ag_preds, "id", "Churn", "submissions/iter16_autogluon.csv")
        ag_corr = np.corrcoef(artem, ag_preds)[0, 1]
        print(f"  AutoGluon corr with Artem: {ag_corr:.4f}", flush=True)
        # Cascade AG with Artem
        cascade_ag = artem * 0.90 + ag_preds * 0.10
        generate_submission(test_ids, cascade_ag, "id", "Churn", "submissions/iter16_ag_cascade.csv")

    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 16 - Novel Features + AutoGluon\n")
        f.write(f"**CV manual**: {cv_score:.6f}, **Corr**: {corr:.4f}\n")
        if ag_score: f.write(f"**AutoGluon**: {ag_score:.6f}\n")
        f.write(f"\n---\n")

    print("DONE!", flush=True)


if __name__ == "__main__":
    main()
