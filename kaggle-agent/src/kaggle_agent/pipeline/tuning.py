"""Hyperparameter tuning via Optuna.

Integrates with the experiment loop for tracked optimization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd

from kaggle_agent.pipeline.models import cross_validate, create_model

optuna.logging.set_verbosity(optuna.logging.WARNING)


def lgbm_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """LightGBM hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def xgb_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """XGBoost hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }


def catboost_search_space(trial: optuna.Trial) -> dict[str, Any]:
    """CatBoost hyperparameter search space."""
    return {
        "iterations": trial.suggest_int("iterations", 500, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
    }


SEARCH_SPACES = {
    "lgbm": lgbm_search_space,
    "xgb": xgb_search_space,
    "catboost": catboost_search_space,
}


def tune_model(
    model_type: str,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str = "roc_auc",
    n_trials: int = 50,
    n_folds: int = 5,
    seed: int = 42,
    timeout: int | None = None,
    base_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Optuna hyperparameter tuning.

    Returns dict with best_params, best_score, study.
    """
    search_space_fn = SEARCH_SPACES.get(model_type)
    if search_space_fn is None:
        raise ValueError(f"No search space for model: {model_type}")

    def objective(trial: optuna.Trial) -> float:
        params = search_space_fn(trial)
        if base_params:
            full_params = {**base_params, **params}
        else:
            full_params = params

        result = cross_validate(
            model_factory=lambda: create_model(model_type, params=full_params),
            X=X,
            y=y,
            metric=metric,
            n_folds=n_folds,
            seed=seed,
        )
        return result["cv_score"]

    study = optuna.create_study(
        direction="maximize" if metric in ("roc_auc", "accuracy") else "minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
        "n_trials": len(study.trials),
    }
