"""Configuration system for KaggleAgent.

Inspired by deer-flow's YAML+Pydantic config and autoresearch's simplicity.
"""

from __future__ import annotations

import os
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    """Dataset configuration."""
    train_path: str
    test_path: str
    sample_submission_path: str
    original_data_path: str | None = None  # For blending with original dataset
    target_column: str
    id_column: str = "id"
    feature_columns: list[str] | None = None  # None = auto-detect


class EvalConfig(BaseModel):
    """Evaluation configuration."""
    metric: str  # sklearn metric name or custom
    metric_direction: str = "maximize"  # "maximize" or "minimize"
    cv_strategy: str = "stratified_kfold"
    cv_folds: int = 5
    cv_seed: int = 42


class SubmissionConfig(BaseModel):
    """Submission configuration."""
    competition_slug: str
    submission_columns: list[str]
    prediction_type: str = "probability"  # "probability", "class", "regression"
    max_daily_submissions: int = 5


class HardwareConfig(BaseModel):
    """Hardware configuration."""
    gpu_device: str = "cuda:0"
    gpu_memory_gb: float = 10.0  # 3080 = 10GB
    use_gpu: bool = True
    n_jobs: int = -1  # CPU parallelism


class ExperimentConfig(BaseModel):
    """Experiment loop configuration."""
    max_experiments: int = 100
    time_budget_seconds: int = 300  # per experiment
    results_file: str = "experiments/results.tsv"
    checkpoint_dir: str = "checkpoints"
    auto_submit: bool = False  # Submit when improvement found


class CompetitionConfig(BaseModel):
    """Full competition configuration."""
    name: str
    slug: str
    description: str = ""
    data: DataConfig
    evaluation: EvalConfig
    submission: SubmissionConfig
    hardware: HardwareConfig = HardwareConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    models: dict[str, dict[str, Any]] = {}  # model_name -> default params
    feature_engineering: dict[str, Any] = {}


def load_config(config_path: str | Path) -> CompetitionConfig:
    """Load competition config from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Interpolate environment variables
    raw = _interpolate_env(raw)
    return CompetitionConfig(**raw)


def _interpolate_env(obj: Any) -> Any:
    """Recursively interpolate $ENV_VAR references."""
    if isinstance(obj, str) and obj.startswith("$"):
        return os.environ.get(obj[1:], obj)
    if isinstance(obj, dict):
        return {k: _interpolate_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env(v) for v in obj]
    return obj
