"""
Research experiment: TabPFN for March Madness prediction.

TabPFN is a pretrained transformer for tabular data that performs
Bayesian inference in a single forward pass. It handles small datasets
well (up to ~10K samples, ~500 features) -- our tournament data
(~2851 samples, 44 features) is a perfect fit.

Usage: uv run python research_tabpfn.py
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from features import FeatureBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def brier_score(y_true, y_pred):
    """Mean squared error between true labels and predicted probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def log_experiment(exp_id, description, overall_brier, per_season, notes, kept="no"):
    """Append a row to experiments.tsv."""
    row = (
        f"{exp_id}\t{description}\t{overall_brier:.6f}\t"
        f"{json.dumps(per_season)}\t{notes}\t{kept}\n"
    )
    with open("experiments.tsv", "a", encoding="utf-8") as f:
        f.write(row)
    print(f"  -> Logged to experiments.tsv as experiment {exp_id}", flush=True)


def next_experiment_id():
    """Read experiments.tsv and return the next integer ID, zero-padded."""
    try:
        df = pd.read_csv("experiments.tsv", sep="\t")
        max_id = df["experiment_id"].astype(int).max()
        return f"{max_id + 1:03d}"
    except Exception:
        return "002"


# ---------------------------------------------------------------------------
# Leave-one-season-out CV
# ---------------------------------------------------------------------------

def run_cv(X, y, seasons, model_fn, config_name):
    """Run leave-one-season-out CV for seasons 2021-2025.

    Parameters
    ----------
    X : pd.DataFrame -- feature matrix
    y : array-like   -- binary target
    seasons : array-like -- season labels
    model_fn : callable(X_train, y_train, X_test) -> y_pred
    config_name : str -- name for printing

    Returns
    -------
    overall_brier : float
    per_season : dict  {season_str: brier}
    """
    eval_seasons = [2021, 2022, 2023, 2024, 2025]
    per_season = {}
    all_preds = []
    all_true = []

    print(f"\n{'='*60}", flush=True)
    print(f"  Config: {config_name}", flush=True)
    print(f"{'='*60}", flush=True)

    for season in eval_seasons:
        mask_test = seasons == season
        mask_train = ~mask_test

        X_train, y_train = X[mask_train], y[mask_train]
        X_test, y_test = X[mask_test], y[mask_test]

        print(f"  Season {season}: train={mask_train.sum()}, test={mask_test.sum()}", flush=True)

        t0 = time.time()
        try:
            preds = model_fn(X_train, y_train, X_test)
        except Exception as e:
            print(f"  ERROR on season {season}: {e}", flush=True)
            # Fall back to 0.5 predictions on error
            preds = np.full(mask_test.sum(), 0.5)

        elapsed = time.time() - t0
        preds = np.clip(preds, 0.01, 0.99)

        bs = brier_score(y_test, preds)
        per_season[str(season)] = round(bs, 6)
        print(f"  Season {season}: Brier = {bs:.6f}  ({elapsed:.1f}s)", flush=True)

        all_preds.extend(preds)
        all_true.extend(y_test)

    overall = brier_score(all_true, all_preds)
    print(f"\n  Overall Brier score: {overall:.6f}", flush=True)
    return overall, per_season


# ---------------------------------------------------------------------------
# TabPFN model functions
# ---------------------------------------------------------------------------

def tabpfn_default(X_train, y_train, X_test):
    """TabPFN with default settings."""
    from tabpfn import TabPFNClassifier

    model = TabPFNClassifier(device='cpu')
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    return preds


def tabpfn_scaled(X_train, y_train, X_test):
    """TabPFN with StandardScaler preprocessing."""
    from tabpfn import TabPFNClassifier

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = TabPFNClassifier(device='cpu')
    model.fit(X_train_s, y_train)
    preds = model.predict_proba(X_test_s)[:, 1]
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, flush=True)
    print("  Research: TabPFN for March Madness", flush=True)
    print("=" * 60, flush=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading training data via FeatureBuilder ...", flush=True)
    fb = FeatureBuilder()
    seasons_list = list(range(2003, 2026))
    X, y, seasons = fb.build_training_data(seasons_list)

    print(f"    X shape: {X.shape}", flush=True)
    print(f"    y shape: {y.shape}", flush=True)
    print(f"    Seasons: {sorted(seasons.unique())}", flush=True)

    # ------------------------------------------------------------------
    # 2. Fill NaN
    # ------------------------------------------------------------------
    nan_count = X.isna().sum().sum()
    print(f"    NaN count before fill: {nan_count}", flush=True)
    X = X.fillna(0)

    # Convert to numpy for TabPFN
    feature_names = X.columns.tolist()
    X_np = X.values.astype(np.float32)
    y_np = y.values if hasattr(y, 'values') else np.asarray(y)
    seasons_np = seasons.values if hasattr(seasons, 'values') else np.asarray(seasons)

    print(f"    Features ({len(feature_names)}): {feature_names[:10]} ...", flush=True)

    # ------------------------------------------------------------------
    # 3. Check TabPFN is importable
    # ------------------------------------------------------------------
    print("\n[2] Checking TabPFN installation ...", flush=True)
    try:
        from tabpfn import TabPFNClassifier
        print("    TabPFN imported successfully.", flush=True)
    except ImportError as e:
        print(f"    ERROR: TabPFN not installed: {e}", flush=True)
        print("    Install with: uv pip install tabpfn", flush=True)
        return

    # ------------------------------------------------------------------
    # 4. Run experiments
    # ------------------------------------------------------------------
    configs = [
        ("TabPFN (default)", tabpfn_default),
        ("TabPFN (StandardScaler)", tabpfn_scaled),
    ]

    baseline_brier = 0.023449
    results = []

    for name, model_fn in configs:
        try:
            overall, per_season = run_cv(X_np, y_np, seasons_np, model_fn, name)
            results.append((name, overall, per_season))
        except Exception as e:
            print(f"\n  FAILED: {name} -- {e}", flush=True)
            results.append((name, None, None))

    # ------------------------------------------------------------------
    # 5. Summary and comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 60, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"  {'Config':<30} {'Brier':>10} {'vs Baseline':>12}", flush=True)
    print(f"  {'-'*30} {'-'*10} {'-'*12}", flush=True)
    print(f"  {'Baseline (XGB+LGBM+LR+NN)':<30} {baseline_brier:>10.6f} {'--':>12}", flush=True)

    for name, overall, per_season in results:
        if overall is not None:
            diff = overall - baseline_brier
            sign = "+" if diff > 0 else ""
            print(f"  {name:<30} {overall:>10.6f} {sign}{diff:>11.6f}", flush=True)
        else:
            print(f"  {name:<30} {'FAILED':>10} {'--':>12}", flush=True)

    # ------------------------------------------------------------------
    # 6. Log best result to experiments.tsv
    # ------------------------------------------------------------------
    print("\n[3] Logging results ...", flush=True)

    valid_results = [(n, o, p) for n, o, p in results if o is not None]
    if valid_results:
        best_name, best_brier, best_per_season = min(valid_results, key=lambda x: x[1])
        exp_id = next_experiment_id()
        notes = f"research_tabpfn.py | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        log_experiment(exp_id, best_name, best_brier, best_per_season, notes)

        # Also log any other configs that are different
        for name, overall, per_season in valid_results:
            if name != best_name:
                exp_id2 = f"{int(exp_id) + 1:03d}"
                notes2 = f"research_tabpfn.py | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                log_experiment(exp_id2, name, overall, per_season, notes2)
    else:
        print("  No successful experiments to log.", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
