"""Iter17 domain-only: price fairness + contract tension + service coherence.
NO KDE, NO KNN - just fast domain features that add new signal."""
import sys
sys.path.insert(0, "../kaggle-agent/src" if sys.platform == "win32" else "/app/kaggle-agent/src")
import gc, warnings, numpy as np, pandas as pd
from itertools import combinations
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb
from kaggle_agent.pipeline.submission import generate_submission
warnings.filterwarnings("ignore")

def main():
    print("=== ITER17 DOMAIN FEATURES ===", flush=True)
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = (train["Churn"] == "Yes").astype(int)
    test_ids = test["id"]
    train = train.drop(columns=["id","Churn"]); test = test.drop(columns=["id"])

    # ORIG_proba
    orig = pd.read_csv("data/telco_original.csv")
    orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce").fillna(0)
    oc = (orig["Churn"]=="Yes").astype(float); gm = oc.mean()
    for col in list(train.columns):
        if col in orig.columns:
            tmp = orig.copy(); tmp["_y"] = oc
            train[f"ORIG_{col}"] = train[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")
            test[f"ORIG_{col}"] = test[col].map(tmp.groupby(col)["_y"].mean()).fillna(gm).astype("float32")

    # Distribution
    ch_tc = np.sort(pd.to_numeric(orig["TotalCharges"],errors="coerce").fillna(0)[orig["Churn"]=="Yes"].values)
    nc_tc = np.sort(pd.to_numeric(orig["TotalCharges"],errors="coerce").fillna(0)[orig["Churn"]!="Yes"].values)
    for df in [train, test]:
        tc = df["TotalCharges"].values
        df["pctrank_ch"] = (np.searchsorted(ch_tc,tc)/len(ch_tc)).astype("float32")
        df["pctrank_nc"] = (np.searchsorted(nc_tc,tc)/len(nc_tc)).astype("float32")

    for df in [train, test]:
        df["avg_monthly"] = (df["TotalCharges"]/df["tenure"].replace(0,1)).astype("float32")
        df["charge_ratio"] = (df["MonthlyCharges"]/df["TotalCharges"].replace(0,1)).astype("float32")
        df["charges_dev"] = (df["TotalCharges"]-df["tenure"]*df["MonthlyCharges"]).astype("float32")
        yes_cols = ["PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
        df["svc_count"] = sum((df[c]=="Yes").astype(int) for c in yes_cols if c in df.columns)
        for col in ["tenure","MonthlyCharges","TotalCharges"]:
            v = df[col].fillna(0)
            df[f"{col}_m10"] = (v%10).astype("float32")
            if col != "tenure": df[f"{col}_frac"] = (v-v.astype(int)).astype("float32")
            if col == "tenure": df[f"{col}_m12"] = (v%12).astype("float32")
        top_cats = ["Contract","InternetService","PaymentMethod","OnlineSecurity","TechSupport","PaperlessBilling"]
        avail = [c for c in top_cats if c in df.columns]
        for c1,c2 in combinations(avail, 2):
            df[f"BG_{c1}_{c2}"] = df[c1].astype(str)+"_"+df[c2].astype(str)

        # === NOVEL DOMAIN FEATURES ===
        # Contract tension
        pf = df["PaymentMethod"].map({"Electronic check":0,"Mailed check":1,"Bank transfer (automatic)":2,"Credit card (automatic)":3}).fillna(1)
        ct = df["Contract"].map({"Month-to-month":0,"One year":1,"Two year":2}).fillna(0)
        df["exit_ease"] = ((3-pf)*(2-ct)).astype("float32")
        dig = (df["PaperlessBilling"].map({"Yes":1,"No":0}).fillna(0)+(df["PaymentMethod"]=="Electronic check").astype(int))
        df["switching_cost"] = (ct+pf-dig).astype("float32")
        df["premium_no_lockin"] = (rankdata(df["MonthlyCharges"])/len(df)*(ct==0).astype(int)).astype("float32")

        # Service coherence
        has_int = (df["InternetService"]!="No").astype(int) if "InternetService" in df.columns else 0
        prot = sum((df[c]=="Yes").astype(int) for c in ["OnlineSecurity","OnlineBackup","DeviceProtection"] if c in df.columns)
        df["protection_completeness"] = np.where(has_int>0, prot/3, 0).astype("float32")
        ent = sum((df[c]=="Yes").astype(int) for c in ["StreamingTV","StreamingMovies"] if c in df.columns)
        df["entertainment_completeness"] = np.where(has_int>0, ent/2, 0).astype("float32")
        df["has_support"] = ((df.get("TechSupport","No")=="Yes") | (df.get("OnlineSecurity","No")=="Yes")).astype(int)

    # Price fairness (fast - no KDE)
    svc = ["PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract"]
    av = [c for c in svc if c in train.columns]
    for df in [train, test]:
        df["_cfg"] = df[av].astype(str).agg("|".join, axis=1)
    stats = train.groupby("_cfg")["MonthlyCharges"].agg(["mean","std","count"])
    for df in [train, test]:
        pm = df["_cfg"].map(stats["mean"]); ps = df["_cfg"].map(stats["std"]).fillna(1).replace(0,1)
        df["price_vs_peers"] = (df["MonthlyCharges"]-pm.fillna(df["MonthlyCharges"])).astype("float32")
        df["price_zscore"] = ((df["MonthlyCharges"]-pm.fillna(df["MonthlyCharges"]))/ps).astype("float32")
        df["peer_group_size"] = df["_cfg"].map(stats["count"]).fillna(0).astype("float32")
        df.drop(columns=["_cfg"], inplace=True)

    cat_cols = [c for c in train.select_dtypes(include=["object","string"]).columns]
    print(f"Features: {train.shape[1]}", flush=True)

    oof = np.zeros(len(train)); test_preds = np.zeros(len(test))
    kf = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(train, y)):
        X_tr=train.iloc[tr_idx].copy(); X_va=train.iloc[va_idx].copy(); X_te=test.copy()
        y_tr = y.iloc[tr_idx]
        oe = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1,dtype=np.float32)
        X_tr[cat_cols]=oe.fit_transform(X_tr[cat_cols]); X_va[cat_cols]=oe.transform(X_va[cat_cols]); X_te[cat_cols]=oe.transform(X_te[cat_cols])
        gm_f = y_tr.mean()
        for col in cat_cols[:15]:
            tmp = pd.DataFrame({"c":X_tr[col],"y":y_tr.values})
            agg = tmp.groupby("c")["y"].agg(["mean","count"])
            sm = (agg["count"]*agg["mean"]+10*gm_f)/(agg["count"]+10)
            X_tr[f"{col}_te"]=X_tr[col].map(sm).fillna(gm_f).astype("float32")
            X_va[f"{col}_te"]=X_va[col].map(sm).fillna(gm_f).astype("float32")
            X_te[f"{col}_te"]=X_te[col].map(sm).fillna(gm_f).astype("float32")
        sc = StandardScaler(); r = Ridge(alpha=10.0)
        r.fit(sc.fit_transform(X_tr.fillna(0)), y_tr)
        X_tr["ridge"]=np.clip(r.predict(sc.transform(X_tr.fillna(0))),0,1).astype("float32")
        X_va["ridge"]=np.clip(r.predict(sc.transform(X_va.fillna(0))),0,1).astype("float32")
        X_te["ridge"]=np.clip(r.predict(sc.transform(X_te.fillna(0))),0,1).astype("float32")
        X_tr=X_tr.fillna(0); X_va=X_va.fillna(0); X_te=X_te.fillna(0)
        model = xgb.XGBClassifier(n_estimators=50000,learning_rate=0.0063,max_depth=5,min_child_weight=6,subsample=0.81,colsample_bytree=0.32,reg_alpha=3.5017,reg_lambda=1.2925,gamma=0.790,early_stopping_rounds=500,device="cuda",random_state=42,verbosity=0,n_jobs=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y.iloc[va_idx])], verbose=False)
        oof[va_idx] = model.predict_proba(X_va)[:,1]
        test_preds += model.predict_proba(X_te)[:,1]/20
        if fold%5==0: print(f"  F{fold}: {roc_auc_score(y.iloc[va_idx], oof[va_idx]):.6f}", flush=True)
        del model; gc.collect()

    score = roc_auc_score(y, oof)
    print(f"CV: {score:.6f}", flush=True)
    artem = pd.read_csv("public_subs/artemevstafyev_cv-auc-0-91930-xgb-cb-blend.csv")["Churn"].values
    corr = np.corrcoef(artem, test_preds)[0,1]
    print(f"Corr with Artem: {corr:.4f}", flush=True)
    generate_submission(test_ids, test_preds, "id", "Churn", "submissions/iter17_domain.csv")
    cascade = artem*0.95 + test_preds*0.05
    generate_submission(test_ids, cascade, "id", "Churn", "submissions/iter17_cascade.csv")
    print("DONE!", flush=True)

if __name__ == "__main__":
    main()
