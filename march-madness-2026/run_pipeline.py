#!/usr/bin/env python3
"""
March Madness 2026 - Autoresearch Pipeline Runner
Iteratively builds, evaluates, and improves prediction models.
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path
from functools import partial

DATA_DIR = Path("data")
EXPERIMENTS_FILE = Path("experiments.tsv")
SUBMISSION_FILE = Path("submission.csv")
BRIER_THRESHOLD = 0.1648

TRAIN_SEASONS = list(range(2003, 2026))
PREDICT_SEASON = 2026


def load_env():
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().strip().split("\n"):
            if "=" in line and not line.startswith("#"):
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()


def log_experiment(exp_id, description, brier, per_season, notes, kept):
    row = {
        "experiment_id": exp_id,
        "description": description,
        "brier_score": f"{brier:.6f}",
        "per_season_scores": json.dumps({str(k): round(v, 6) for k, v in per_season.items()}),
        "notes": notes,
        "kept": "yes" if kept else "no",
    }
    df = pd.DataFrame([row])
    if EXPERIMENTS_FILE.exists() and EXPERIMENTS_FILE.stat().st_size > 50:
        existing = pd.read_csv(EXPERIMENTS_FILE, sep="\t")
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(EXPERIMENTS_FILE, sep="\t", index=False)


def get_next_id():
    if EXPERIMENTS_FILE.exists() and EXPERIMENTS_FILE.stat().st_size > 50:
        df = pd.read_csv(EXPERIMENTS_FILE, sep="\t")
        return f"{len(df) + 1:03d}"
    return "001"


def fill_nans(X):
    """Fill NaN values for models that can't handle them."""
    return X.fillna(0)


def run_cv(factory, X, y, seasons, label):
    """Run leave-one-season-out CV with a model factory."""
    from model import brier_score

    unique = sorted(seasons.unique())
    all_preds, all_true = [], []
    per_season = {}

    for s in unique:
        test_mask = (seasons == s).values
        train_mask = ~test_mask
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]
        if len(Xte) == 0:
            continue
        model = factory()
        model.train(Xtr, ytr)
        preds = model.predict(Xte)
        preds = np.clip(preds, 0.001, 0.999)
        score = brier_score(yte.values, preds)
        per_season[int(s)] = score
        all_preds.extend(preds)
        all_true.extend(yte.values)
        print(f"  Season {int(s)}: Brier={score:.6f} (n={len(Xte)})")

    mean_b = brier_score(np.array(all_true), np.array(all_preds))
    print(f"  {label} Overall: {mean_b:.6f} (n={len(all_true)})")
    return {"mean_brier": mean_b, "per_season_brier": per_season,
            "all_predictions": np.array(all_preds), "all_actuals": np.array(all_true)}


def main():
    load_env()
    start = time.time()

    from features import FeatureBuilder
    from model import MarchMadnessModel, NeuralNetModel, brier_score

    print("=" * 60)
    print("MARCH MADNESS 2026 - AUTORESEARCH PIPELINE")
    print("=" * 60)

    fb = FeatureBuilder(str(DATA_DIR))
    print("Building training data...")
    X_raw, y, seasons = fb.build_training_data(TRAIN_SEASONS)
    X = fill_nans(X_raw)
    print(f"Training data: {X.shape[0]} games, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")

    results = {}

    # ================================================================
    # ROUND 1: Baseline models
    # ================================================================
    print("\n>>> ROUND 1: Baseline models")

    # XGBoost
    print("\n--- XGBoost default ---")
    r = run_cv(lambda: MarchMadnessModel("xgb"), X, y, seasons, "XGB")
    results["XGB_default"] = r["mean_brier"]
    log_experiment(get_next_id(), "XGBoost default", r["mean_brier"], r["per_season_brier"],
                   f"{X.shape[1]} features", True)

    # LightGBM
    print("\n--- LightGBM default ---")
    r_lgbm = run_cv(lambda: MarchMadnessModel("lgbm"), X, y, seasons, "LGBM")
    results["LGBM_default"] = r_lgbm["mean_brier"]
    log_experiment(get_next_id(), "LightGBM default", r_lgbm["mean_brier"], r_lgbm["per_season_brier"],
                   f"{X.shape[1]} features", True)

    # Logistic Regression
    print("\n--- Logistic Regression ---")
    r_lr = run_cv(lambda: MarchMadnessModel("logistic"), X, y, seasons, "LR")
    results["Logistic"] = r_lr["mean_brier"]
    log_experiment(get_next_id(), "Logistic Regression", r_lr["mean_brier"], r_lr["per_season_brier"],
                   f"{X.shape[1]} features", False)

    # Ensemble
    print("\n--- Ensemble (XGB+LGBM+LR) ---")
    r_ens = run_cv(lambda: MarchMadnessModel("ensemble"), X, y, seasons, "Ensemble")
    results["Ensemble_default"] = r_ens["mean_brier"]
    log_experiment(get_next_id(), "Ensemble XGB+LGBM+LR", r_ens["mean_brier"], r_ens["per_season_brier"],
                   f"{X.shape[1]} features", True)

    # Feature importance from baseline
    xgb_full = MarchMadnessModel("xgb")
    xgb_full.train(X, y)
    imp = xgb_full.get_feature_importance()
    sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
    print("\nTop features (XGB):")
    for f, v in sorted_imp[:15]:
        print(f"  {f}: {v:.4f}")

    # ================================================================
    # ROUND 2: Hyperparameter tuning
    # ================================================================
    print("\n>>> ROUND 2: Hyperparameter tuning")

    xgb_configs = [
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 800, "min_child_weight": 5,
         "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 600, "min_child_weight": 3,
         "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.3, "reg_lambda": 1.5},
        {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 1000, "min_child_weight": 7,
         "subsample": 0.75, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 3.0},
        {"max_depth": 4, "learning_rate": 0.04, "n_estimators": 700, "min_child_weight": 4,
         "subsample": 0.75, "colsample_bytree": 0.75, "reg_alpha": 0.2, "reg_lambda": 1.0},
    ]

    best_xgb_brier = 1.0
    best_xgb_config = {}
    for i, cfg in enumerate(xgb_configs):
        print(f"\n--- XGB config {i+1} ---")
        params = {"objective": "binary:logistic", "eval_metric": "logloss",
                  "random_state": 42, "verbosity": 0, **cfg}
        r = run_cv(lambda p=params: MarchMadnessModel("xgb", p), X, y, seasons, f"XGB-{i+1}")
        if r["mean_brier"] < best_xgb_brier:
            best_xgb_brier = r["mean_brier"]
            best_xgb_config = cfg
    results["XGB_tuned"] = best_xgb_brier
    log_experiment(get_next_id(), "Tuned XGBoost", best_xgb_brier, r["per_season_brier"],
                   str(best_xgb_config), True)

    lgbm_configs = [
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 800, "min_child_weight": 5,
         "subsample": 0.7, "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 2.0, "num_leaves": 31},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 600, "min_child_weight": 3,
         "subsample": 0.8, "colsample_bytree": 0.6, "reg_alpha": 0.3, "reg_lambda": 1.5, "num_leaves": 20},
        {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 1000, "min_child_weight": 7,
         "subsample": 0.75, "colsample_bytree": 0.8, "reg_alpha": 1.0, "reg_lambda": 3.0, "num_leaves": 15},
    ]

    best_lgbm_brier = 1.0
    best_lgbm_config = {}
    for i, cfg in enumerate(lgbm_configs):
        print(f"\n--- LGBM config {i+1} ---")
        params = {"objective": "binary", "metric": "binary_logloss",
                  "random_state": 42, "verbose": -1, **cfg}
        r = run_cv(lambda p=params: MarchMadnessModel("lgbm", p), X, y, seasons, f"LGBM-{i+1}")
        if r["mean_brier"] < best_lgbm_brier:
            best_lgbm_brier = r["mean_brier"]
            best_lgbm_config = cfg
    results["LGBM_tuned"] = best_lgbm_brier
    log_experiment(get_next_id(), "Tuned LightGBM", best_lgbm_brier, r["per_season_brier"],
                   str(best_lgbm_config), True)

    # ================================================================
    # ROUND 3: Neural net + mega ensemble
    # ================================================================
    print("\n>>> ROUND 3: Neural net + mega ensemble")

    # Neural net
    print("\n--- Neural Network ---")

    class NNWrapper:
        def __init__(self):
            self._model = None
        def train(self, Xtr, ytr):
            self._model = NeuralNetModel()
            self._model.train(Xtr, ytr)
        def predict(self, Xte):
            return self._model.predict(Xte)

    r_nn = run_cv(NNWrapper, X, y, seasons, "NN")
    results["NeuralNet"] = r_nn["mean_brier"]
    log_experiment(get_next_id(), "Neural Network MLP", r_nn["mean_brier"], r_nn["per_season_brier"],
                   "3-layer MLP with BatchNorm", False)

    # Mega ensemble: per-fold, train all models and average
    print("\n--- Mega Ensemble ---")
    unique = sorted(seasons.unique())
    all_preds, all_true = [], []
    mega_per_season = {}

    for s in unique:
        test_mask = (seasons == s).values
        train_mask = ~test_mask
        Xtr, ytr = X[train_mask], y[train_mask]
        Xte, yte = X[test_mask], y[test_mask]
        if len(Xte) == 0:
            continue

        preds_list = []

        # Best XGB
        xgb_params = {"objective": "binary:logistic", "eval_metric": "logloss",
                      "random_state": 42, "verbosity": 0, **best_xgb_config}
        m = MarchMadnessModel("xgb", xgb_params)
        m.train(Xtr, ytr)
        preds_list.append(("xgb", m.predict(Xte), 0.30))

        # Best LGBM
        lgbm_params = {"objective": "binary", "metric": "binary_logloss",
                       "random_state": 42, "verbose": -1, **best_lgbm_config}
        m = MarchMadnessModel("lgbm", lgbm_params)
        m.train(Xtr, ytr)
        preds_list.append(("lgbm", m.predict(Xte), 0.30))

        # Default XGB (diversity)
        m = MarchMadnessModel("xgb")
        m.train(Xtr, ytr)
        preds_list.append(("xgb_default", m.predict(Xte), 0.10))

        # Logistic
        m = MarchMadnessModel("logistic")
        m.train(Xtr, ytr)
        preds_list.append(("lr", m.predict(Xte), 0.10))

        # NN
        nn = NeuralNetModel()
        nn.train(Xtr, ytr)
        preds_list.append(("nn", nn.predict(Xte), 0.20))

        ens_pred = np.zeros(len(Xte))
        for name, p, w in preds_list:
            ens_pred += w * p
        ens_pred = np.clip(ens_pred, 0.001, 0.999)

        score = brier_score(yte.values, ens_pred)
        mega_per_season[int(s)] = score
        all_preds.extend(ens_pred)
        all_true.extend(yte.values)
        print(f"  Season {int(s)}: Brier={score:.6f}")

    mega_brier = brier_score(np.array(all_true), np.array(all_preds))
    print(f"  Mega Ensemble Overall: {mega_brier:.6f}")
    results["Mega_Ensemble"] = mega_brier
    log_experiment(get_next_id(), "Mega Ensemble: XGB+LGBM+XGB_def+LR+NN",
                   mega_brier, mega_per_season,
                   "weights: 0.30+0.30+0.10+0.10+0.20", True)

    # ================================================================
    # ROUND 4: Seed-only baseline and blend
    # ================================================================
    print("\n>>> ROUND 4: Seed-based predictions + blend")

    # Historical seed win rates are extremely powerful for March Madness
    # Build a seed-only model as a strong baseline
    from features import FeatureBuilder as FB
    seed_feats = fb.build_seed_features()

    # For tournament games, compute historical seed vs seed win rates
    m_tourney = pd.read_csv(DATA_DIR / "MNCAATourneyCompactResults.csv")
    w_tourney = pd.read_csv(DATA_DIR / "WNCAATourneyCompactResults.csv")

    m_seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    w_seeds = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv")

    def parse_seed_num(s):
        return int(s[1:3])

    m_seeds['SeedNum'] = m_seeds['Seed'].apply(parse_seed_num)
    w_seeds['SeedNum'] = w_seeds['Seed'].apply(parse_seed_num)

    all_tourney = pd.concat([m_tourney, w_tourney], ignore_index=True)
    all_seeds = pd.concat([m_seeds, w_seeds], ignore_index=True)

    all_tourney = all_tourney.merge(
        all_seeds[['Season', 'TeamID', 'SeedNum']].rename(columns={'TeamID': 'WTeamID', 'SeedNum': 'WSeed'}),
        on=['Season', 'WTeamID'], how='left'
    ).merge(
        all_seeds[['Season', 'TeamID', 'SeedNum']].rename(columns={'TeamID': 'LTeamID', 'SeedNum': 'LSeed'}),
        on=['Season', 'LTeamID'], how='left'
    )

    # Compute historical seed vs seed win probability
    seed_matchup_wins = {}
    seed_matchup_total = {}
    for _, g in all_tourney.dropna(subset=['WSeed', 'LSeed']).iterrows():
        s1, s2 = int(min(g['WSeed'], g['LSeed'])), int(max(g['WSeed'], g['LSeed']))
        key = (s1, s2)
        seed_matchup_total[key] = seed_matchup_total.get(key, 0) + 1
        if g['WSeed'] == s1:  # better seed won
            seed_matchup_wins[key] = seed_matchup_wins.get(key, 0) + 1

    seed_win_rates = {}
    for key in seed_matchup_total:
        wins = seed_matchup_wins.get(key, 0)
        total = seed_matchup_total[key]
        seed_win_rates[key] = wins / total

    print(f"  Computed {len(seed_win_rates)} seed matchup probabilities")

    # ================================================================
    # RESULTS SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, score in sorted(results.items(), key=lambda x: x[1]):
        marker = " <<<" if score == min(results.values()) else ""
        thr = " [BELOW THRESHOLD]" if score < BRIER_THRESHOLD else ""
        print(f"  {name:25s}: {score:.6f}{marker}{thr}")

    best_name = min(results, key=results.get)
    best_score = results[best_name]
    print(f"\nBest: {best_name} = {best_score:.6f}")

    # ================================================================
    # GENERATE SUBMISSION
    # ================================================================
    print("\n>>> GENERATING SUBMISSION")

    # Train all models on full data
    xgb_params_final = {"objective": "binary:logistic", "eval_metric": "logloss",
                        "random_state": 42, "verbosity": 0, **best_xgb_config}
    lgbm_params_final = {"objective": "binary", "metric": "binary_logloss",
                         "random_state": 42, "verbose": -1, **best_lgbm_config}

    models_full = []

    m1 = MarchMadnessModel("xgb", xgb_params_final)
    m1.train(X, y)
    models_full.append(("xgb_tuned", m1, 0.30))

    m2 = MarchMadnessModel("lgbm", lgbm_params_final)
    m2.train(X, y)
    models_full.append(("lgbm_tuned", m2, 0.30))

    m3 = MarchMadnessModel("xgb")
    m3.train(X, y)
    models_full.append(("xgb_default", m3, 0.10))

    m4 = MarchMadnessModel("logistic")
    m4.train(X, y)
    models_full.append(("lr", m4, 0.10))

    nn_full = NeuralNetModel()
    nn_full.train(X, y)

    # Build Stage 2 features
    print("Building submission features for 2026...")
    sub_df = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    X_sub = fb.build_submission_features(str(DATA_DIR / "SampleSubmissionStage2.csv"))
    X_sub = fill_nans(X_sub)
    print(f"Submission: {len(X_sub)} matchups")

    preds = np.zeros(len(X_sub))
    for name, model, weight in models_full:
        p = model.predict(X_sub)
        preds += weight * p
        print(f"  {name}: mean={p.mean():.4f} std={p.std():.4f}")

    nn_p = nn_full.predict(X_sub)
    preds += 0.20 * nn_p
    print(f"  nn: mean={nn_p.mean():.4f} std={nn_p.std():.4f}")

    preds = np.clip(preds, 0.01, 0.99)
    print(f"  Final: mean={preds.mean():.4f} std={preds.std():.4f}")

    sub_df["Pred"] = preds
    sub_df.to_csv(SUBMISSION_FILE, index=False)
    print(f"Saved {SUBMISSION_FILE}")

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}m)")

    if best_score < BRIER_THRESHOLD:
        print(f"\nBrier {best_score:.6f} < {BRIER_THRESHOLD} - SUBMITTING!")
        os.system(
            f'kaggle competitions submit -c march-machine-learning-mania-2026 '
            f'-f submission.csv -m "Autoresearch ensemble Brier={best_score:.6f}"'
        )
    else:
        print(f"\nBrier {best_score:.6f} >= {BRIER_THRESHOLD} - not submitting yet.")


if __name__ == "__main__":
    main()
