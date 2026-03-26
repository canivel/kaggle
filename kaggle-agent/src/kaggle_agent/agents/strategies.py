"""Pre-defined strategy libraries for common competition patterns.

Inspired by autoresearch's AUTO_STRATEGIES pattern.
Each strategy is a complete experiment definition.
"""

from __future__ import annotations

from kaggle_agent.loop import Strategy


def tabular_binary_strategies() -> list[Strategy]:
    """Strategy library for binary classification tabular competitions.

    Designed for competitions evaluated by AUC-ROC.
    """
    return [
        # === BASELINE MODELS ===
        Strategy(
            name="baseline_lgbm",
            model_type="lgbm",
            params={},
            description="LightGBM baseline with default params",
        ),
        Strategy(
            name="baseline_xgb",
            model_type="xgb",
            params={},
            description="XGBoost baseline with default params",
        ),
        Strategy(
            name="baseline_catboost",
            model_type="catboost",
            params={},
            description="CatBoost baseline with default params",
        ),

        # === TUNED LGBM VARIANTS ===
        Strategy(
            name="lgbm_deep",
            model_type="lgbm",
            params={"num_leaves": 63, "max_depth": 8, "min_child_samples": 50},
            description="LightGBM deeper trees",
        ),
        Strategy(
            name="lgbm_wide",
            model_type="lgbm",
            params={"num_leaves": 127, "max_depth": -1, "min_child_samples": 100},
            description="LightGBM wide trees",
        ),
        Strategy(
            name="lgbm_conservative",
            model_type="lgbm",
            params={
                "num_leaves": 31, "learning_rate": 0.01, "n_estimators": 3000,
                "reg_alpha": 1.0, "reg_lambda": 1.0,
            },
            description="LightGBM low LR high reg",
        ),
        Strategy(
            name="lgbm_dart",
            model_type="lgbm",
            params={
                "boosting_type": "dart", "num_leaves": 63,
                "learning_rate": 0.05, "n_estimators": 500,
            },
            description="LightGBM DART boosting",
        ),
        Strategy(
            name="lgbm_gbdt_deep_low_lr",
            model_type="lgbm",
            params={
                "num_leaves": 95, "max_depth": 10, "learning_rate": 0.02,
                "n_estimators": 2000, "subsample": 0.7, "colsample_bytree": 0.7,
                "min_child_samples": 30, "reg_alpha": 0.5, "reg_lambda": 0.5,
            },
            description="LightGBM deep with low LR",
        ),

        # === TUNED XGB VARIANTS ===
        Strategy(
            name="xgb_deep",
            model_type="xgb",
            params={"max_depth": 8, "min_child_weight": 3, "gamma": 0.1},
            description="XGBoost deeper trees",
        ),
        Strategy(
            name="xgb_conservative",
            model_type="xgb",
            params={
                "max_depth": 5, "learning_rate": 0.01, "n_estimators": 3000,
                "reg_alpha": 1.0, "reg_lambda": 5.0,
            },
            description="XGBoost low LR high reg",
        ),
        Strategy(
            name="xgb_wide",
            model_type="xgb",
            params={
                "max_depth": 4, "n_estimators": 2000, "learning_rate": 0.03,
                "subsample": 0.7, "colsample_bytree": 0.6,
                "min_child_weight": 5, "gamma": 0.2,
            },
            description="XGBoost wide shallow",
        ),

        # === CATBOOST VARIANTS ===
        Strategy(
            name="catboost_deep",
            model_type="catboost",
            params={"depth": 8, "iterations": 2000, "learning_rate": 0.03},
            description="CatBoost deeper trees",
        ),
        Strategy(
            name="catboost_conservative",
            model_type="catboost",
            params={
                "depth": 6, "iterations": 3000, "learning_rate": 0.01,
                "l2_leaf_reg": 10.0,
            },
            description="CatBoost low LR high reg",
        ),

        # === DIVERSITY STRATEGIES (for ensembling) ===
        Strategy(
            name="lgbm_extra_random",
            model_type="lgbm",
            params={
                "num_leaves": 31, "subsample": 0.6, "colsample_bytree": 0.5,
                "extra_trees": True, "random_state": 123,
            },
            description="LightGBM extra randomized",
        ),
        Strategy(
            name="xgb_seed2",
            model_type="xgb",
            params={"random_state": 2024, "max_depth": 7},
            description="XGBoost different seed",
        ),
        Strategy(
            name="lgbm_seed3",
            model_type="lgbm",
            params={"random_state": 2025, "num_leaves": 47},
            description="LightGBM different seed",
        ),
    ]


def get_strategy_library(problem_type: str = "tabular_binary") -> list[Strategy]:
    """Get strategy library by problem type."""
    libraries = {
        "tabular_binary": tabular_binary_strategies,
    }
    factory = libraries.get(problem_type)
    if factory is None:
        raise ValueError(f"Unknown problem type: {problem_type}. Available: {list(libraries.keys())}")
    return factory()
