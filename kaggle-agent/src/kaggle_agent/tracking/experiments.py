"""Experiment tracking via append-only TSV log.

Inspired by autoresearch's results.tsv pattern:
- Append-only (never modify past entries)
- Tab-separated (safe for descriptions with commas)
- Git-friendly (easy to diff)
"""

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass, asdict
from pathlib import Path


TSV_FIELDS = [
    "experiment_id",
    "timestamp",
    "model_type",
    "description",
    "cv_score",
    "cv_std",
    "lb_score",
    "status",
    "duration_seconds",
    "n_features",
    "params",
    "notes",
]


@dataclass
class ExperimentResult:
    experiment_id: str
    timestamp: str
    model_type: str
    description: str
    cv_score: float | None
    cv_std: float | None
    lb_score: float | None
    status: str  # "kept", "discarded", "crashed", "submitted"
    duration_seconds: float
    n_features: int
    params: str
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentTracker:
    """Append-only TSV experiment log."""

    def __init__(self, results_path: str | Path):
        self.path = Path(results_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: ExperimentResult) -> None:
        """Append experiment result to TSV."""
        exists = self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=TSV_FIELDS, delimiter="\t", extrasaction="ignore"
            )
            if not exists:
                writer.writeheader()
            writer.writerow(result.to_dict())

    def load_all(self) -> list[ExperimentResult]:
        """Load all experiment results."""
        if not self.path.exists():
            return []
        results = []
        with open(self.path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row["cv_score"] = float(row["cv_score"]) if row.get("cv_score") else None
                row["cv_std"] = float(row["cv_std"]) if row.get("cv_std") else None
                row["lb_score"] = float(row["lb_score"]) if row.get("lb_score") else None
                row["duration_seconds"] = float(row.get("duration_seconds", 0))
                row["n_features"] = int(row.get("n_features", 0))
                results.append(ExperimentResult(**row))
        return results

    def best_score(self, direction: str = "maximize") -> float | None:
        """Get best CV score across all kept experiments."""
        results = self.load_all()
        kept = [r for r in results if r.status in ("kept", "submitted") and r.cv_score is not None]
        if not kept:
            return None
        scores = [r.cv_score for r in kept]
        return max(scores) if direction == "maximize" else min(scores)

    def best_experiment(self, direction: str = "maximize") -> ExperimentResult | None:
        """Get best experiment."""
        results = self.load_all()
        kept = [r for r in results if r.status in ("kept", "submitted") and r.cv_score is not None]
        if not kept:
            return None
        key = max if direction == "maximize" else min
        return key(kept, key=lambda r: r.cv_score)

    def next_id(self) -> str:
        """Generate next experiment ID."""
        results = self.load_all()
        return f"{len(results) + 1:04d}"

    def summary(self) -> str:
        """Human-readable summary of experiments."""
        results = self.load_all()
        if not results:
            return "No experiments recorded yet."

        kept = [r for r in results if r.status in ("kept", "submitted")]
        discarded = [r for r in results if r.status == "discarded"]
        crashed = [r for r in results if r.status == "crashed"]

        lines = [
            f"Total experiments: {len(results)}",
            f"  Kept: {len(kept)}",
            f"  Discarded: {len(discarded)}",
            f"  Crashed: {len(crashed)}",
        ]

        if kept:
            best = max(kept, key=lambda r: r.cv_score or 0)
            lines.append(f"  Best CV: {best.cv_score:.6f} ({best.model_type}: {best.description})")

        return "\n".join(lines)
