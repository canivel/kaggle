"""Adversarial validation + novel post-processing.

1. Adversarial validation: Can we distinguish train from test?
   If yes, the distributions differ and we need to account for it.

2. Test-time calibration: Isotonic regression, Platt scaling
   to better calibrate predictions.

3. Hill-climbing ensemble selection: Greedy forward selection
   of models/folds that maximize OOF AUC.
"""

import sys
sys.path.insert(0, "../kaggle-agent/src")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from scipy.stats import rankdata
import lightgbm as lgb


def adversarial_validation(train, test):
    """Check if train/test distributions differ."""
    print("=== ADVERSARIAL VALIDATION ===", flush=True)

    # Label: 0=train, 1=test
    train_av = train.copy()
    test_av = test.copy()
    train_av["is_test"] = 0
    test_av["is_test"] = 1

    combined = pd.concat([train_av, test_av], ignore_index=True)
    y_av = combined["is_test"]
    X_av = combined.drop(columns=["is_test"])

    # Encode categoricals
    for col in X_av.select_dtypes(include=["object", "string"]).columns:
        X_av[col] = X_av[col].astype("category").cat.codes

    model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=31,
                                verbosity=-1, n_jobs=4)

    scores = cross_val_score(model, X_av, y_av, cv=5, scoring="roc_auc")
    print(f"  AV AUC: {scores.mean():.4f} +/- {scores.std():.4f}", flush=True)
    print(f"  (0.50 = identical, >0.55 = different distributions)", flush=True)

    if scores.mean() > 0.55:
        # Find which features differ most
        model.fit(X_av, y_av)
        imp = dict(zip(X_av.columns, model.feature_importances_))
        top_diff = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:10]
        print("  Top differing features:", flush=True)
        for feat, score in top_diff:
            print(f"    {feat}: {score}", flush=True)

    return scores.mean()


def hill_climbing_ensemble(oof_dict, y_true, test_dict, n_rounds=50):
    """Greedy forward selection of predictions to maximize OOF AUC.

    Each round, try adding each model to the current ensemble and keep
    the one that improves the most.
    """
    print("\n=== HILL-CLIMBING ENSEMBLE ===", flush=True)

    names = list(oof_dict.keys())
    n_models = len(names)

    # Start with best single model
    best_single = max(names, key=lambda n: roc_auc_score(y_true, oof_dict[n]))
    selected = [best_single]
    selected_oof = oof_dict[best_single].copy()
    selected_test = test_dict[best_single].copy()
    best_score = roc_auc_score(y_true, selected_oof)
    print(f"  Start: {best_single} = {best_score:.6f}", flush=True)

    for round_idx in range(n_rounds):
        best_improvement = 0
        best_candidate = None

        for name in names:
            # Try adding this model with equal weight
            n_selected = len(selected)
            candidate_oof = (selected_oof * n_selected + oof_dict[name]) / (n_selected + 1)
            candidate_score = roc_auc_score(y_true, candidate_oof)
            improvement = candidate_score - best_score

            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = name

        if best_candidate is None or best_improvement < 1e-7:
            break

        selected.append(best_candidate)
        n_sel = len(selected)
        selected_oof = (selected_oof * (n_sel - 1) + oof_dict[best_candidate]) / n_sel
        selected_test = (selected_test * (n_sel - 1) + test_dict[best_candidate]) / n_sel
        best_score = roc_auc_score(y_true, selected_oof)
        print(f"  Round {round_idx+1}: +{best_candidate} = {best_score:.6f} (+{best_improvement:.7f})", flush=True)

    print(f"  Final: {len(selected)} models, AUC={best_score:.6f}", flush=True)
    return selected_test, selected, best_score


def isotonic_calibration(oof_preds, y_true, test_preds):
    """Calibrate predictions using isotonic regression on OOF."""
    print("\n=== ISOTONIC CALIBRATION ===", flush=True)

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_preds, y_true)

    calibrated_oof = ir.predict(oof_preds)
    calibrated_test = ir.predict(test_preds)

    before = roc_auc_score(y_true, oof_preds)
    after = roc_auc_score(y_true, calibrated_oof)
    print(f"  Before: {before:.6f}, After: {after:.6f}", flush=True)
    print(f"  (Isotonic doesn't change AUC but may help with LB calibration)", flush=True)

    return calibrated_test


def rank_calibration(pred1, pred2, weight1=0.99, weight2=0.01):
    """Rank-based calibration (Artem's technique from top notebook).

    Blend in rank space, then map back to probability space using pred1's distribution.
    """
    print("\n=== RANK CALIBRATION ===", flush=True)

    # Compute weighted rank
    rank1 = rankdata(pred1)
    rank2 = rankdata(pred2)
    blended_rank = weight1 * rank1 + weight2 * rank2

    # Map blended ranks back to pred1's probability space
    # Sort pred1 by rank, create monotonic mapping
    order = np.argsort(blended_rank)
    sorted_pred1 = np.sort(pred1)

    calibrated = np.empty_like(pred1)
    calibrated[order] = sorted_pred1

    print(f"  Blended with weights ({weight1:.2f}, {weight2:.2f})", flush=True)
    return calibrated


def main():
    print("=" * 70, flush=True)
    print("NOVEL APPROACHES: Adversarial Validation + Post-Processing", flush=True)
    print("=" * 70, flush=True)

    # Load data
    train = pd.read_csv("data/train.csv").drop(columns=["id", "Churn"])
    test = pd.read_csv("data/test.csv").drop(columns=["id"])
    y = (pd.read_csv("data/train.csv")["Churn"] == "Yes").astype(int)
    test_ids = pd.read_csv("data/test.csv")["id"]

    # 1. Adversarial validation
    av_score = adversarial_validation(train, test)

    # 2. Load existing predictions for ensemble experiments
    print("\n=== Loading existing submissions ===", flush=True)
    subs = {}
    for fname in ["iter5_tree", "iter6_blamerx", "iter5_all", "iter5_rank"]:
        path = f"submissions/{fname}.csv"
        try:
            df = pd.read_csv(path)
            subs[fname] = df["Churn"].values
            print(f"  Loaded {fname}: range [{df['Churn'].min():.4f}, {df['Churn'].max():.4f}]", flush=True)
        except FileNotFoundError:
            pass

    if len(subs) < 2:
        print("  Not enough submissions for ensemble experiments", flush=True)
        return

    # 3. Rank calibration between iter6 and iter5
    if "iter6_blamerx" in subs and "iter5_tree" in subs:
        calibrated = rank_calibration(subs["iter6_blamerx"], subs["iter5_tree"], 0.95, 0.05)
        from kaggle_agent.pipeline.submission import generate_submission
        generate_submission(test_ids, calibrated, "id", "Churn",
                          "submissions/iter6_rank_calibrated.csv")
        print("  Saved: iter6_rank_calibrated.csv", flush=True)

    # 4. Rank calibration variants
    for w1 in [0.90, 0.80, 0.70]:
        w2 = 1 - w1
        cal = rank_calibration(subs["iter6_blamerx"], subs["iter5_tree"], w1, w2)
        name = f"submissions/rank_cal_{int(w1*100)}_{int(w2*100)}.csv"
        generate_submission(test_ids, cal, "id", "Churn", name)
        print(f"  Saved: {name}", flush=True)

    print("\nDONE!", flush=True)


if __name__ == "__main__":
    main()
