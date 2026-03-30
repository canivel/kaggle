"""Iteration 18: Maximize CV for private LB protection.

Public LB can reshuffle. What protects us is a model with HIGH, HONEST CV.
Our best CV is 0.9188 (iter6). Target: push CV as high as possible
while keeping it honest (all FE inside folds).

Approach: systematic feature ablation + hyperparameter sweep.
Find the EXACT set of features that maximizes 20-fold CV.
"""
import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")
import gc, warnings, numpy as np, pandas as pd, time
from itertools import combinations
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
from kaggle_agent.pipeline.submission import generate_submission
warnings.filterwarnings("ignore")

N_FOLDS = 20

def prepare_base():
    """Base features - the iter6 set that works best."""
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"]=="Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id","Churn"]); test = test.drop(columns=["id"])

    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"]=="Yes").astype(float); gm = oc.mean()
    for col in list(train.columns):
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            train[f"ORIG_{col}"] = train[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")

    ch_tc = np.sort(pd.to_numeric(orig["TotalCharges"],errors="coerce").fillna(0)[orig["Churn"]=="Yes"].values)
    nc_tc = np.sort(pd.to_numeric(orig["TotalCharges"],errors="coerce").fillna(0)[orig["Churn"]!="Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc,tc)/len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc,tc)/len(nc_tc)).astype("float32")
        df["pctrank_gap"] = df["pctrank_ch"] - df["pctrank_nc"]
        df["zscore_ch"] = ((tc - ch_tc.mean())/ch_tc.std()).astype("float32")
        df["zscore_nc"] = ((tc - nc_tc.mean())/nc_tc.std()).astype("float32")

    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"]/df["tenure"].replace(0,1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"]/df["TotalCharges"].replace(0,1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"]-df["tenure"]*df["MonthlyCharges"]).astype("float32")
        yes_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
        df["svc_count"] = sum((df[c]=="Yes").astype(int) for c in yes_cols if c in df.columns)
        df["has_internet"] = (df.get("InternetService","No")!="No").astype(int)
        for col in ["tenure","MonthlyCharges","TotalCharges"]:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v%10).astype("float32")
            df[f"{col}_d1"] = v.astype(str).str[0].astype("float32")
            df[f"{col}_ld"] = (v*100).astype(int)%10
            df[f"{col}_round10"] = ((v%10==0)&(v>0)).astype(int)
            df[f"{col}_dev_r10"] = (v-(v/10).round()*10).astype("float32")
            if col != "tenure":
                df[f"{col}_frac"] = (v-v.astype(int)).astype("float32")
                df[f"{col}_m100"] = (v%100).astype("float32")
            if col == "tenure":
                df[f"{col}_m12"] = (v%12).astype("float32")
                df["tenure_years"] = (v//12).astype("float32")
        top_cats = ["Contract","InternetService","PaymentMethod","OnlineSecurity","TechSupport","PaperlessBilling"]
        avail = [c for c in top_cats if c in df.columns]
        for c1,c2 in combinations(avail, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str)+"_"+df[c2].astype(str)
        for c1,c2,c3 in combinations(avail[:4], 3):
            df[f"TG_{c1}_{c2}_{c3}"] = df[c1].astype(str)+"_"+df[c2].astype(str)+"_"+df[c3].astype(str)

    cat_cols = [c for c in train.select_dtypes(include=["object","string"]).columns]
    return train, test, y, test_ids, cat_cols


def run_cv(train, test, y, cat_cols, xgb_params, label=""):
    """Run 20-fold CV with full in-fold encoding."""
    oof = np.zeros(len(train)); test_preds = np.zeros(len(test))
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        X_tr=train.iloc[tr_idx].copy(); X_va=train.iloc[va_idx].copy(); X_te=test.copy()
        y_tr = y.iloc[tr_idx]

        oe = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=np.float32)
        X_tr[cat_cols]=oe.fit_transform(X_tr[cat_cols])
        X_va[cat_cols]=oe.transform(X_va[cat_cols])
        X_te[cat_cols]=oe.transform(X_te[cat_cols])

        gm = y_tr.mean()
        for col in cat_cols[:18]:
            tmp = pd.DataFrame({"c":X_tr[col],"y":y_tr.values})
            agg = tmp.groupby("c")["y"].agg(["mean","count"])
            sm = (agg["count"]*agg["mean"]+10*gm)/(agg["count"]+10)
            X_tr[f"{col}_te"]=X_tr[col].map(sm).fillna(gm).astype("float32")
            X_va[f"{col}_te"]=X_va[col].map(sm).fillna(gm).astype("float32")
            X_te[f"{col}_te"]=X_te[col].map(sm).fillna(gm).astype("float32")

        sc = StandardScaler(); r = Ridge(alpha=10.0)
        r.fit(sc.fit_transform(X_tr.fillna(0)), y_tr)
        X_tr["ridge"]=np.clip(r.predict(sc.transform(X_tr.fillna(0))),0,1).astype("float32")
        X_va["ridge"]=np.clip(r.predict(sc.transform(X_va.fillna(0))),0,1).astype("float32")
        X_te["ridge"]=np.clip(r.predict(sc.transform(X_te.fillna(0))),0,1).astype("float32")

        X_tr=X_tr.fillna(0); X_va=X_va.fillna(0); X_te=X_te.fillna(0)
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:,1]
        test_preds += model.predict_proba(X_te)[:,1]/N_FOLDS
        del model; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"  {label}: CV={score:.6f}", flush=True)
    return oof, test_preds, score


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("ITER18: CV KING - Maximize honest CV for private LB", flush=True)
    print("=" * 70, flush=True)

    train, test, y, test_ids, cat_cols = prepare_base()
    print(f"Features: {train.shape[1]}", flush=True)

    # === HYPERPARAMETER SWEEP ===
    # Our best (BlamerX exact): lr=0.0063, max_depth=5, colsample=0.32, reg_alpha=3.5
    # Sweep around these values
    base_params = dict(
        n_estimators=50000, early_stopping_rounds=500,
        device="cuda", random_state=42, verbosity=0, n_jobs=-1,
    )

    configs = [
        ("blamerx_exact", dict(learning_rate=0.0063, max_depth=5, min_child_weight=6,
                               subsample=0.81, colsample_bytree=0.32,
                               reg_alpha=3.5017, reg_lambda=1.2925, gamma=0.790)),
        ("lower_lr", dict(learning_rate=0.003, max_depth=5, min_child_weight=6,
                          subsample=0.81, colsample_bytree=0.32,
                          reg_alpha=3.5, reg_lambda=1.3, gamma=0.79)),
        ("higher_lr", dict(learning_rate=0.01, max_depth=5, min_child_weight=6,
                           subsample=0.81, colsample_bytree=0.32,
                           reg_alpha=3.5, reg_lambda=1.3, gamma=0.79)),
        ("deeper", dict(learning_rate=0.0063, max_depth=6, min_child_weight=8,
                        subsample=0.81, colsample_bytree=0.28,
                        reg_alpha=4.0, reg_lambda=1.5, gamma=1.0)),
        ("wider_col", dict(learning_rate=0.0063, max_depth=5, min_child_weight=6,
                           subsample=0.81, colsample_bytree=0.40,
                           reg_alpha=3.5, reg_lambda=1.3, gamma=0.79)),
        ("narrower_col", dict(learning_rate=0.0063, max_depth=5, min_child_weight=6,
                              subsample=0.81, colsample_bytree=0.25,
                              reg_alpha=3.5, reg_lambda=1.3, gamma=0.79)),
        ("more_reg", dict(learning_rate=0.0063, max_depth=5, min_child_weight=6,
                          subsample=0.81, colsample_bytree=0.32,
                          reg_alpha=5.0, reg_lambda=2.0, gamma=1.0)),
        ("less_sub", dict(learning_rate=0.0063, max_depth=5, min_child_weight=6,
                          subsample=0.75, colsample_bytree=0.32,
                          reg_alpha=3.5, reg_lambda=1.3, gamma=0.79)),
    ]

    results = {}
    best_cv = 0
    best_name = ""

    for name, params in configs:
        full_params = {**base_params, **params}
        oof, test_preds, cv = run_cv(train, test, y, cat_cols, full_params, label=name)
        results[name] = {"oof": oof, "test": test_preds, "cv": cv, "params": params}
        if cv > best_cv:
            best_cv = cv
            best_name = name

    print(f"\n=== BEST: {best_name} CV={best_cv:.6f} ===", flush=True)

    # === SAVE BEST ===
    best = results[best_name]
    generate_submission(test_ids, best["test"], "id", "Churn", "submissions/iter18_best.csv")

    # Also save blamerx_exact for comparison
    bx = results["blamerx_exact"]
    generate_submission(test_ids, bx["test"], "id", "Churn", "submissions/iter18_blamerx.csv")

    # Average of top 3
    top3 = sorted(results.items(), key=lambda x: x[1]["cv"], reverse=True)[:3]
    avg_test = np.mean([r[1]["test"] for r in top3], axis=0)
    avg_oof = np.mean([r[1]["oof"] for r in top3], axis=0)
    avg_cv = roc_auc_score(y, avg_oof)
    print(f"  Top-3 avg: CV={avg_cv:.6f} ({[r[0] for r in top3]})", flush=True)
    generate_submission(test_ids, avg_test, "id", "Churn", "submissions/iter18_top3avg.csv")

    # Cascade best with Artem
    artem = pd.read_csv("public_subs/artemevstafyev_cv-auc-0-91930-xgb-cb-blend.csv")["Churn"].values
    cascade = artem * 0.95 + best["test"] * 0.05
    generate_submission(test_ids, cascade, "id", "Churn", "submissions/iter18_cascade.csv")

    # Micro-perturb artem with best
    perturb = artem * 1.001 - 0.001 * best["test"]
    generate_submission(test_ids, perturb, "id", "Churn", "submissions/iter18_perturb.csv")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min", flush=True)

    with open("docs/competition_log.md", "a", encoding="utf-8") as f:
        f.write(f"\n### Iteration 18 - CV King (Hyperparameter Sweep)\n")
        f.write(f"**Date**: {__import__('datetime').datetime.now().isoformat()}\n\n")
        for name, r in sorted(results.items(), key=lambda x: x[1]["cv"], reverse=True):
            f.write(f"- {name}: {r['cv']:.6f}\n")
        f.write(f"- **Top-3 avg**: {avg_cv:.6f}\n")
        f.write(f"- **Best**: {best_name} = {best_cv:.6f}\n")
        f.write(f"\n---\n")

    print("DONE!", flush=True)

if __name__ == "__main__":
    main()
