"""Submission generation and Kaggle API interaction."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def generate_submission(
    test_ids: pd.Series,
    predictions: np.ndarray,
    id_column: str = "id",
    target_column: str = "Churn",
    output_path: str | Path = "submission.csv",
) -> Path:
    """Create a submission CSV file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame({
        id_column: test_ids,
        target_column: predictions,
    })
    submission.to_csv(path, index=False)
    return path


def submit_to_kaggle(
    submission_path: str | Path,
    competition_slug: str,
    message: str = "Automated submission",
) -> str | None:
    """Submit to Kaggle via CLI. Returns LB score if available."""
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", competition_slug,
        "-f", str(submission_path),
        "-m", message,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        return f"ERROR: {result.stderr}"

    return result.stdout.strip()


def blend_predictions(
    predictions_list: list[np.ndarray],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Weighted average of multiple prediction arrays."""
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    weights = np.array(weights) / sum(weights)
    blended = np.zeros_like(predictions_list[0], dtype=np.float64)
    for pred, w in zip(predictions_list, weights):
        blended += w * pred
    return blended


def rank_average(predictions_list: list[np.ndarray]) -> np.ndarray:
    """Rank-average ensemble (more robust than simple averaging)."""
    from scipy.stats import rankdata

    ranked = []
    for preds in predictions_list:
        ranks = rankdata(preds) / len(preds)
        ranked.append(ranks)

    return np.mean(ranked, axis=0)
