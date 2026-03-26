"""Kaggle API interaction tools."""

from __future__ import annotations

import subprocess
from pathlib import Path


def download_competition_data(slug: str, output_dir: str | Path) -> bool:
    """Download competition data via Kaggle CLI."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", slug, "-p", str(output_dir)],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print(f"Error downloading: {result.stderr}")
        return False

    # Unzip if needed
    for zip_file in output_dir.glob("*.zip"):
        subprocess.run(
            ["unzip", "-o", str(zip_file), "-d", str(output_dir)],
            capture_output=True,
        )
    return True


def download_dataset(dataset_slug: str, output_dir: str | Path) -> bool:
    """Download a Kaggle dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(output_dir), "--unzip"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def submit_prediction(
    submission_path: str | Path,
    competition_slug: str,
    message: str,
) -> str:
    """Submit prediction file."""
    result = subprocess.run(
        [
            "kaggle", "competitions", "submit",
            "-c", competition_slug,
            "-f", str(submission_path),
            "-m", message,
        ],
        capture_output=True, text=True,
    )
    return result.stdout + result.stderr


def get_leaderboard(competition_slug: str) -> str:
    """Get competition leaderboard."""
    result = subprocess.run(
        ["kaggle", "competitions", "leaderboard", "-c", competition_slug, "--show"],
        capture_output=True, text=True,
    )
    return result.stdout
