# ====== CODE CELL 3 ======
try:
    get_ipython().run_line_magic("load_ext", "cudf.pandas")
    print("cudf.pandas loaded ✅ — GPU-accelerated pandas active")
except Exception as e:
    print(f"cudf.pandas unavailable ({e.__class__.__name__}) — using standard pandas ✅")

import numpy as np
import pandas as pd
import warnings, gc, time

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# ====== CODE CELL 5 ======
# CONFIG = {
#     "N_FOLDS": 10,
#     "INNER_FOLDS": 10,
#     "SEEDS": [11, 42, 99],
#     "TARGET": "Churn",
#     "PSEUDO_LABELS": True,
#     "TRES": 0.998,
# }

# XGB_PARAMS = {
#     "n_estimators": 50000,
#     "learning_rate": 0.03,
#     "max_depth": 6,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "min_child_weight": 5,
#     "reg_alpha": 0.1,
#     "reg_lambda": 1.0,
#     "gamma": 0.05,
#     "early_stopping_rounds": 500,
#     "objective": "binary:logistic",
#     "eval_metric": "auc",
#     "enable_categorical": True,
#     "device": "cuda",
#     "tree_method": "hist",
#     "verbosity": 0,
# }

# LGB_PARAMS = {
#     "n_estimators": 50000,
#     "learning_rate": 0.03,
#     "max_depth": 7,
#     "num_leaves": 63,
#     "subsample": 0.8,
#     "colsample_bytree": 0.8,
#     "reg_alpha": 0.1,
#     "reg_lambda": 1.0,
#     "min_child_samples": 20,
#     "objective": "binary",
#     "metric": "auc",
#     "n_jobs": -1,
#     "verbose": -1,
# }

# print("=" * 70)
# print("CONFIGURATION:")
# for k, v in CONFIG.items():
#     print(f"  {k:<20} : {v}")
# print(f"\n  XGB lr={XGB_PARAMS['learning_rate']}, depth={XGB_PARAMS['max_depth']}, es={XGB_PARAMS['early_stopping_rounds']}")
# print(f"  LGB lr={LGB_PARAMS['learning_rate']}, depth={LGB_PARAMS['max_depth']}, leaves={LGB_PARAMS['num_leaves']}")
# print(f"  Seeds: {CONFIG['SEEDS']} ({len(CONFIG['SEEDS'])} seeds)")
# print("=" * 70)

# ====== CODE CELL 7 ======
# TRAIN_PATH = "/kaggle/input/competitions/playground-series-s6e3/train.csv"
# TEST_PATH = "/kaggle/input/competitions/playground-series-s6e3/test.csv"
# ORIGINAL_PATH = "/kaggle/input/datasets/thedrzee/customer-churn-in-telecom-sample-dataset-by-ibm/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# print("[1/3] Loading competition train set...")
# train = pd.read_csv(TRAIN_PATH)
# print(f"Shape: {train.shape} | Memory: {train.memory_usage(deep=True).sum()/1024**2:.2f} MB")

# print("[2/3] Loading competition test set...")
# test = pd.read_csv(TEST_PATH)
# print(f"Shape: {test.shape}")

# print("[3/3] Loading original IBM Telco dataset...")
# orig = pd.read_csv(ORIGINAL_PATH)
# print(f"Shape: {orig.shape}")

# train[CONFIG["TARGET"]] = train[CONFIG["TARGET"]].astype(str).str.strip().map({"No": 0, "Yes": 1})
# orig[CONFIG["TARGET"]] = orig[CONFIG["TARGET"]].astype(str).str.strip().map({"No": 0, "Yes": 1})

# train[CONFIG["TARGET"]] = train[CONFIG["TARGET"]].fillna(0).astype("int8")
# orig[CONFIG["TARGET"]] = orig[CONFIG["TARGET"]].fillna(0).astype("int8")

# if "TotalCharges" in orig.columns:
#     orig["TotalCharges"] = pd.to_numeric(orig["TotalCharges"], errors="coerce")
#     orig["TotalCharges"] = orig["TotalCharges"].fillna(orig["TotalCharges"].median())

# if "customerID" in orig.columns:
#     orig.drop(columns=["customerID"], inplace=True)

# train_ids = train["id"].copy()
# test_ids = test["id"].copy()

# print(f"\nTrain: {len(train):,} rows | Churn rate: {train[CONFIG['TARGET']].mean()*100:.2f}%")
# print(f"Orig : {len(orig):,} rows | Churn rate: {orig[CONFIG['TARGET']].mean()*100:.2f}%")
# print(f"Test : {len(test):,} rows")
# print("=" * 70)

# ====== CODE CELL 9 ======
# CATS = [
#     "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
#     "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
#     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
#     "Contract", "PaperlessBilling", "PaymentMethod",
# ]
# NUMS = ["tenure", "MonthlyCharges", "TotalCharges"]

# NEW_NUMS, NEW_CATS, NUM_AS_CAT, NON_TE_CATS, TO_REMOVE = [], [], [], [], []

# for df in (train, test, orig):
#     for c in CATS:
#         if c in df.columns:
#             df[c] = df[c].astype(str).fillna("missing").str.strip()
#     for c in NUMS:
#         if c in df.columns:
#             df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

# for df in (train, test, orig):
#     if "TotalCharges" in df.columns:
#         df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# for col in NUMS:
#     freq = pd.concat([train[col], orig[col], test[col]], axis=0).value_counts(normalize=True)
#     for df in (train, test, orig):
#         df[f"FREQ_{col}"] = df[col].map(freq).fillna(0).astype("float32")
#     NEW_NUMS.append(f"FREQ_{col}")

# _all_num = pd.concat([train[NUMS], test[NUMS], orig[NUMS]], axis=0, ignore_index=True)
# for col in NUMS:
#     r = _all_num[col].rank(method="average", pct=True)
#     n_tr, n_te = len(train), len(test)
#     train[f"RANK_{col}"] = r.iloc[:n_tr].astype("float32").values
#     test[f"RANK_{col}"] = r.iloc[n_tr:n_tr + n_te].astype("float32").values
#     orig[f"RANK_{col}"] = r.iloc[n_tr + n_te:].astype("float32").values
# NEW_NUMS += [f"RANK_{c}" for c in NUMS]

# for df in (train, test, orig):
#     for col in NUMS:
#         v = df[col].astype("float32")
#         df[f"LOG1P_{col}"] = np.log1p(v.clip(lower=0)).astype("float32")
#         df[f"SQRT_{col}"] = np.sqrt(v.clip(lower=0)).astype("float32")
#         df[f"INV1P_{col}"] = (1.0 / (1.0 + v.clip(lower=0))).astype("float32")
# NEW_NUMS += [f"LOG1P_{c}" for c in NUMS] + [f"SQRT_{c}" for c in NUMS] + [f"INV1P_{c}" for c in NUMS]

# for df in (train, test, orig):
#     df["charges_deviation"] = (df["TotalCharges"] - df["tenure"] * df["MonthlyCharges"]).astype("float32")
#     df["abs_charges_dev"] = np.abs(df["charges_deviation"]).astype("float32")
#     df["monthly_to_total_ratio"] = (df["MonthlyCharges"] / (df["TotalCharges"] + 1)).astype("float32")
#     df["total_to_monthly_ratio"] = (df["TotalCharges"] / (df["MonthlyCharges"] + 1)).astype("float32")
#     df["avg_monthly_charges"] = (df["TotalCharges"] / (df["tenure"] + 1)).astype("float32")
#     df["tenure_x_monthly"] = (df["tenure"] * df["MonthlyCharges"]).astype("float32")
#     df["tenure_x_total"] = (df["tenure"] * df["TotalCharges"]).astype("float32")
# NEW_NUMS += [
#     "charges_deviation", "abs_charges_dev",
#     "monthly_to_total_ratio", "total_to_monthly_ratio",
#     "avg_monthly_charges", "tenure_x_monthly", "tenure_x_total"
# ]

# SERVICE_COLS = [
#     "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
#     "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
# ]
# for df in (train, test, orig):
#     for c in SERVICE_COLS + ["InternetService"]:
#         if c in df.columns:
#             df[c] = df[c].astype(str).fillna("missing").str.strip()
#     df["service_yes_count"] = (df[SERVICE_COLS] == "Yes").sum(axis=1).astype("float32")
#     df["service_no_count"] = (df[SERVICE_COLS] == "No").sum(axis=1).astype("float32")
#     df["service_other_count"] = (df[SERVICE_COLS].isin(["No phone service", "No internet service"])).sum(axis=1).astype("float32")
#     df["service_count"] = df["service_yes_count"].astype("float32")
#     df["has_internet"] = (df["InternetService"] != "No").astype("float32")
#     df["has_phone"] = (df["PhoneService"] == "Yes").astype("float32")
# NEW_NUMS += [
#     "service_yes_count", "service_no_count", "service_other_count",
#     "service_count", "has_internet", "has_phone"
# ]

# TENURE_BINS = [0, 1, 3, 6, 12, 24, 36, 48, 60, 72, 10_000]
# MC_BINS = 40
# TC_BINS = 60

# for df in (train, test, orig):
#     df["tenure_bin"] = pd.cut(df["tenure"], bins=TENURE_BINS, include_lowest=True).astype(str).astype("category")

# mc_bins = pd.qcut(pd.concat([train["MonthlyCharges"], test["MonthlyCharges"], orig["MonthlyCharges"]]), q=MC_BINS, retbins=True, duplicates="drop")[1]
# tc_bins = pd.qcut(pd.concat([train["TotalCharges"], test["TotalCharges"], orig["TotalCharges"]]), q=TC_BINS, retbins=True, duplicates="drop")[1]

# for df in (train, test, orig):
#     df["MonthlyCharges_bin"] = pd.cut(df["MonthlyCharges"], bins=mc_bins, include_lowest=True).astype(str).astype("category")
#     df["TotalCharges_bin"] = pd.cut(df["TotalCharges"], bins=tc_bins, include_lowest=True).astype(str).astype("category")

# NEW_CATS += ["tenure_bin", "MonthlyCharges_bin", "TotalCharges_bin"]

# for df in (train, test, orig):
#     for col in NUMS:
#         df[f"MISS_{col}"] = pd.to_numeric(df[col], errors="coerce").isna().astype("int8")
# NEW_NUMS += [f"MISS_{c}" for c in NUMS]

# YN_COLS = [
#     "Partner", "Dependents", "PhoneService", "PaperlessBilling",
#     "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
#     "StreamingTV", "StreamingMovies", "MultipleLines",
# ]
# for df in (train, test, orig):
#     for c in YN_COLS:
#         if c in df.columns:
#             s = df[c].astype(str)
#             df[f"ISYES_{c}"] = (s == "Yes").astype("float32")
#             df[f"ISNO_{c}"] = (s == "No").astype("float32")
#             df[f"ISOTHER_{c}"] = (~s.isin(["Yes", "No"])).astype("float32")
# NEW_NUMS += (
#     [f"ISYES_{c}" for c in YN_COLS if c in train.columns]
#     + [f"ISNO_{c}" for c in YN_COLS if c in train.columns]
#     + [f"ISOTHER_{c}" for c in YN_COLS if c in train.columns]
# )

# CROSS_PAIRS = [
#     ("Contract", "InternetService"),
#     ("PaymentMethod", "Contract"),
#     ("InternetService", "OnlineSecurity"),
#     ("PaymentMethod", "PaperlessBilling"),
#     ("Contract", "PaperlessBilling"),
#     ("InternetService", "TechSupport"),
# ]
# for a, b in CROSS_PAIRS:
#     if a in train.columns and b in train.columns:
#         name = f"{a}__{b}"
#         for df in (train, test, orig):
#             df[name] = (df[a].astype(str) + "|" + df[b].astype(str)).astype("category")
#         NEW_CATS.append(name)

# TRIPLE = [("Contract", "InternetService", "PaymentMethod")]
# for a, b, c in TRIPLE:
#     if a in train.columns and b in train.columns and c in train.columns:
#         name = f"{a}__{b}__{c}"
#         for df in (train, test, orig):
#             df[name] = (df[a].astype(str) + "|" + df[b].astype(str) + "|" + df[c].astype(str)).astype("category")
#         NEW_CATS.append(name)

# ALL_CATS_FOR_COUNT = CATS + NEW_CATS
# _all_cat = pd.concat([train[ALL_CATS_FOR_COUNT], test[ALL_CATS_FOR_COUNT], orig[ALL_CATS_FOR_COUNT]], axis=0, ignore_index=True)
# for c in ALL_CATS_FOR_COUNT:
#     vc = _all_cat[c].value_counts(dropna=False)
#     for df in (train, test, orig):
#         df[f"CAT_CNT_{c}"] = df[c].map(vc).fillna(0).astype("float32")
#         df[f"CAT_RARE_{c}"] = (df[f"CAT_CNT_{c}"] <= 50).astype("float32")
#     NEW_NUMS += [f"CAT_CNT_{c}", f"CAT_RARE_{c}"]

# for col in CATS + NUMS + NEW_CATS:
#     if col in orig.columns:
#         tmp = orig.groupby(col, observed=False)[CONFIG["TARGET"]].mean()
#         name = f"ORIG_proba_{col}"
#         train = train.merge(tmp.rename(name), on=col, how="left")
#         test = test.merge(tmp.rename(name), on=col, how="left")
#         orig = orig.merge(tmp.rename(name), on=col, how="left")
#         for df in (train, test, orig):
#             df[name] = df[name].fillna(0.5).astype("float32")
#         NEW_NUMS.append(name)

# for col in NUMS:
#     _new = f"CAT_{col}"
#     NUM_AS_CAT.append(_new)
#     for df in (train, test, orig):
#         df[_new] = df[col].astype(str).astype("category")

# FEATURES = NUMS + CATS + NEW_NUMS + NEW_CATS + NUM_AS_CAT + NON_TE_CATS

# ====== CODE CELL 11 ======
# %%time

# TE_COLUMNS = NUM_AS_CAT + CATS + NEW_CATS
# TO_REMOVE = NUM_AS_CAT + CATS + NEW_CATS
# STATS = ["std", "min", "max"]

# SEEDS = CONFIG["SEEDS"]
# N_FOLDS = CONFIG["N_FOLDS"]
# INNER_FOLDS = CONFIG["INNER_FOLDS"]
# TARGET = CONFIG["TARGET"]

# oof_xgb_all = np.zeros(len(train), dtype=np.float64)
# oof_lgb_all = np.zeros(len(train), dtype=np.float64)
# pred_xgb_all = np.zeros(len(test), dtype=np.float64)
# pred_lgb_all = np.zeros(len(test), dtype=np.float64)
# all_importances = []
# fold_scores_xgb_first = []
# fold_scores_lgb_first = []

# t0 = time.time()
# print(f"\n{'='*80}")
# print(f"⚡ TRAINING: XGB(CUDA) + LGB(CPU) × {len(SEEDS)} seeds × {N_FOLDS} folds")
# print("=" * 80)

# for seed_idx, SEED in enumerate(SEEDS):
#     print(f"\n{'~'*60}")
#     print(f"SEED {seed_idx+1}/{len(SEEDS)}: {SEED}")
#     print(f"{'~'*60}")

#     xgb_params = {**XGB_PARAMS, "random_state": SEED}
#     lgb_params = {**LGB_PARAMS, "random_state": SEED}

#     skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

#     oof_xgb_seed = np.zeros(len(train), dtype=np.float64)
#     oof_lgb_seed = np.zeros(len(train), dtype=np.float64)
#     pred_xgb_seed = np.zeros(len(test), dtype=np.float64)
#     pred_lgb_seed = np.zeros(len(test), dtype=np.float64)

#     for i, (train_idx, val_idx) in enumerate(skf.split(train, train[TARGET])):
#         t_fold = time.time()

#         # === SPLIT ===
#         X_tr = train.loc[train_idx, FEATURES + [TARGET]].reset_index(drop=True).copy()
#         y_tr = train.loc[train_idx, TARGET].values.copy()
#         X_val = train.loc[val_idx, FEATURES].reset_index(drop=True).copy()
#         y_val = train.loc[val_idx, TARGET].values.copy()
#         X_te = test[FEATURES].reset_index(drop=True).copy()

#         # === INNER K-FOLD TARGET ENCODING ===
#         skf_inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED)
#         for j, (in_tr, in_va) in enumerate(skf_inner.split(X_tr, y_tr)):
#             X_tr2 = X_tr.loc[in_tr, FEATURES + [TARGET]].copy()
#             X_va2 = X_tr.loc[in_va, FEATURES].copy()
#             for col in TE_COLUMNS:
#                 tmp = X_tr2.groupby(col, observed=False)[TARGET].agg(STATS)
#                 tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
#                 X_va2 = X_va2.merge(tmp, on=col, how="left")
#                 for c in tmp.columns:
#                     X_tr.loc[in_va, c] = X_va2[c].values.astype("float32")

#         # Apply TE to val/test
#         for col in TE_COLUMNS:
#             tmp = X_tr.groupby(col, observed=False)[TARGET].agg(STATS)
#             tmp.columns = [f"TE1_{col}_{s}" for s in STATS]
#             tmp = tmp.astype("float32")
#             X_val = X_val.merge(tmp, on=col, how="left")
#             X_te = X_te.merge(tmp, on=col, how="left")
#             for c in tmp.columns:
#                 for df_ in [X_tr, X_val, X_te]:
#                     df_[c] = df_[c].fillna(0)

#         # sklearn TargetEncoder (mean)
#         TE_MEAN_COLS = [f"TE_{col}" for col in TE_COLUMNS]
#         te = TargetEncoder(cv=INNER_FOLDS, shuffle=True, smooth="auto",
#                            target_type="binary", random_state=SEED)
#         X_tr[TE_MEAN_COLS] = te.fit_transform(X_tr[TE_COLUMNS], y_tr)
#         X_val[TE_MEAN_COLS] = te.transform(X_val[TE_COLUMNS])
#         X_te[TE_MEAN_COLS] = te.transform(X_te[TE_COLUMNS])

#         # Convert cats for XGB, then drop for final features
#         for df_ in [X_tr, X_val, X_te]:
#             df_[CATS + NUM_AS_CAT] = df_[CATS + NUM_AS_CAT].astype(str).astype("category")
#             df_.drop(columns=TO_REMOVE, inplace=True)
#         X_tr.drop(columns=[TARGET], inplace=True)
#         COLS = X_tr.columns

#         # =============================================
#         # 1) TRAIN LIGHTGBM (CPU)
#         # =============================================
#         lgb_model = lgb.LGBMClassifier(**lgb_params)
#         lgb_model.fit(
#             X_tr.values, y_tr,
#             eval_set=[(X_val.values, y_val)],
#             callbacks=[lgb.early_stopping(500), lgb.log_evaluation(0)]
#         )
#         lgb_oof_p = lgb_model.predict_proba(X_val.values)[:, 1]
#         lgb_auc = roc_auc_score(y_val, lgb_oof_p)

#         # Pseudo-labels for LGB
#         if CONFIG["PSEUDO_LABELS"]:
#             lgb_test_p = lgb_model.predict_proba(X_te.values)[:, 1]
#             mask = (lgb_test_p > CONFIG["TRES"]) | (lgb_test_p < 1 - CONFIG["TRES"])
#             if mask.sum() > 0:
#                 X_pl = pd.concat([X_tr, X_te[mask]], axis=0)
#                 y_pl = np.concatenate([y_tr, (lgb_test_p[mask] > 0.5).astype(int)])
#                 lgb_model2 = lgb.LGBMClassifier(**lgb_params)
#                 lgb_model2.fit(
#                     X_pl.values, y_pl,
#                     eval_set=[(X_val.values, y_val)],
#                     callbacks=[lgb.early_stopping(500), lgb.log_evaluation(0)]
#                 )
#                 lgb_oof_p2 = lgb_model2.predict_proba(X_val.values)[:, 1]
#                 lgb_auc2 = roc_auc_score(y_val, lgb_oof_p2)
#                 if lgb_auc2 > lgb_auc:
#                     lgb_model, lgb_oof_p, lgb_auc = lgb_model2, lgb_oof_p2, lgb_auc2
#                 del X_pl, y_pl

#         # =============================================
#         # 2) TRAIN XGBOOST (GPU CUDA)
#         # =============================================
#         xgb_model = xgb.XGBClassifier(**xgb_params)
#         xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
#         xgb_oof_p = xgb_model.predict_proba(X_val)[:, 1]
#         xgb_auc = roc_auc_score(y_val, xgb_oof_p)

#         # Pseudo-labels for XGB
#         if CONFIG["PSEUDO_LABELS"]:
#             xgb_test_p = xgb_model.predict_proba(X_te)[:, 1]
#             mask = (xgb_test_p > CONFIG["TRES"]) | (xgb_test_p < 1 - CONFIG["TRES"])
#             if mask.sum() > 0:
#                 X_pl = pd.concat([X_tr, X_te[mask]], axis=0)
#                 y_pl = np.concatenate([y_tr, (xgb_test_p[mask] > 0.5).astype(int)])
#                 xgb_model2 = xgb.XGBClassifier(**xgb_params)
#                 xgb_model2.fit(X_pl, y_pl, eval_set=[(X_val, y_val)], verbose=False)
#                 xgb_oof_p2 = xgb_model2.predict_proba(X_val)[:, 1]
#                 xgb_auc2 = roc_auc_score(y_val, xgb_oof_p2)
#                 if xgb_auc2 > xgb_auc:
#                     xgb_model, xgb_oof_p, xgb_auc = xgb_model2, xgb_oof_p2, xgb_auc2
#                 del X_pl, y_pl

#         elapsed = time.time() - t_fold
#         print(f"  Fold {i+1:>2}/{N_FOLDS} [{elapsed:.0f}s] XGB={xgb_auc:.5f} LGB={lgb_auc:.5f}")

#         # Store OOF
#         oof_xgb_seed[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
#         oof_lgb_seed[val_idx] = lgb_model.predict_proba(X_val.values)[:, 1]

#         # Store test preds
#         pred_xgb_seed += xgb_model.predict_proba(X_te[COLS])[:, 1]
#         pred_lgb_seed += lgb_model.predict_proba(X_te[COLS].values)[:, 1]

#         if seed_idx == 0:
#             fold_scores_xgb_first.append(xgb_auc)
#             fold_scores_lgb_first.append(lgb_auc)
#             all_importances.append(xgb_model.get_booster().get_score(importance_type="gain"))

#         del X_tr, X_val, X_te, y_tr, y_val, xgb_model, lgb_model
#         gc.collect()

#     pred_xgb_seed /= N_FOLDS
#     pred_lgb_seed /= N_FOLDS
#     oof_xgb_all += oof_xgb_seed
#     oof_lgb_all += oof_lgb_seed
#     pred_xgb_all += pred_xgb_seed
#     pred_lgb_all += pred_lgb_seed

#     seed_xgb_auc = roc_auc_score(train[TARGET].values, oof_xgb_seed)
#     seed_lgb_auc = roc_auc_score(train[TARGET].values, oof_lgb_seed)
#     print(f"\n  >>> Seed {SEED}: XGB={seed_xgb_auc:.5f}  LGB={seed_lgb_auc:.5f}")

# # === Average across seeds ===
# oof_xgb = oof_xgb_all / len(SEEDS)
# oof_lgb = oof_lgb_all / len(SEEDS)
# pred_xgb = pred_xgb_all / len(SEEDS)
# pred_lgb = pred_lgb_all / len(SEEDS)

# # === Optimal blend weight ===
# def neg_auc(w):
#     return -roc_auc_score(train[TARGET].values, w * oof_xgb + (1 - w) * oof_lgb)

# res = minimize_scalar(neg_auc, bounds=(0, 1), method="bounded")
# best_w = res.x

# oof = best_w * oof_xgb + (1 - best_w) * oof_lgb
# pred = best_w * pred_xgb + (1 - best_w) * pred_lgb

# total_time = time.time() - t0
# print(f"\n{'='*80}")
# print(f"⚡ DONE in {total_time/60:.1f} min")
# print(f"   XGB OOF AUC   : {roc_auc_score(train[TARGET].values, oof_xgb):.5f}")
# print(f"   LGB OOF AUC   : {roc_auc_score(train[TARGET].values, oof_lgb):.5f}")
# print(f"   BLEND OOF AUC : {roc_auc_score(train[TARGET].values, oof):.5f}  (w_xgb={best_w:.3f})")
# print(f"   XGB Fold AUC  : {np.mean(fold_scores_xgb_first):.5f} ± {np.std(fold_scores_xgb_first):.5f}")
# print(f"   LGB Fold AUC  : {np.mean(fold_scores_lgb_first):.5f} ± {np.std(fold_scores_lgb_first):.5f}")
# print("=" * 80)

# ====== CODE CELL 13 ======
# # Save OOF
# pd.DataFrame({"xgb": oof_xgb, "lgb": oof_lgb, "blend": oof}).to_csv("oof_predictions.csv", index=False)

# # Submission
# sub = pd.DataFrame({"id": test_ids, CONFIG["TARGET"]: pred})
# sub.to_csv("submission.csv", index=False)

# blend_auc = roc_auc_score(train[CONFIG["TARGET"]].values, oof)
# print(f"✅ BLEND OOF AUC : {blend_auc:.5f}")
# print(f"   Blend weight  : {best_w:.3f} XGB + {1-best_w:.3f} LGB")
# print(f"   Pred range    : [{pred.min():.5f}, {pred.max():.5f}]")
# print(f"   Mean churn    : {pred.mean():.5f}")
# print(f"   Features used : {len(COLS)}")
# print(f"   Seeds         : {CONFIG['SEEDS']}")
# print("Saved: submission.csv, oof_predictions.csv")

# ====== CODE CELL 14 ======


sub = pd.read_csv('/kaggle/input/datasets/badalkrsharma/0-91680-s6e3/auc-0-91925-xgboost-bi-tri-target-encoding.ipynb')

# Save with EXACT required name
sub.to_csv('submission.csv', index=False)

print("submission.csv created successfully")
sub.head()
