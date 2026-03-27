"""Model factory and training utilities.

Supports LightGBM, XGBoost, CatBoost, and simple neural networks.
All models implement a common interface for the experiment loop.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error, accuracy_score


METRIC_FUNCTIONS = {
    "roc_auc": roc_auc_score,
    "log_loss": log_loss,
    "rmse": lambda y, p: mean_squared_error(y, p, squared=False),
    "accuracy": accuracy_score,
}


class BaseModel(ABC):
    """Common interface for all models."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return probabilities for classification, values for regression."""
        pass

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        pass

    def feature_importance(self) -> dict[str, float] | None:
        return None


class LGBMModel(BaseModel):
    def __init__(self, params: dict[str, Any] | None = None, **kwargs):
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            self.params.update(params)
        self.params.update(kwargs)
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        import lightgbm as lgb
        fit_params = {}
        if "eval_set" in kwargs:
            fit_params["eval_set"] = kwargs["eval_set"]
            fit_params["callbacks"] = [lgb.early_stopping(50, verbose=False)]
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, **fit_params)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict[str, Any]:
        return self.params

    def feature_importance(self) -> dict[str, float] | None:
        if self.model is None:
            return None
        return dict(zip(self.model.feature_name_, self.model.feature_importances_))


class XGBModel(BaseModel):
    def __init__(self, params: dict[str, Any] | None = None, **kwargs):
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
            "n_jobs": -1,
        }
        if params:
            self.params.update(params)
        self.params.update(kwargs)
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        import xgboost as xgb
        params = {**self.params}
        if "eval_set" in kwargs:
            params["early_stopping_rounds"] = 50
        self.model = xgb.XGBClassifier(**params)
        fit_params = {"verbose": False}
        if "eval_set" in kwargs:
            fit_params["eval_set"] = kwargs["eval_set"]
        self.model.fit(X, y, **fit_params)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict[str, Any]:
        return self.params

    def feature_importance(self) -> dict[str, float] | None:
        if self.model is None:
            return None
        imp = self.model.feature_importances_
        names = self.model.get_booster().feature_names
        return dict(zip(names, imp))


class CatBoostModel(BaseModel):
    def __init__(self, params: dict[str, Any] | None = None, cat_features: list[str] | None = None, **kwargs):
        self.params = {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_seed": 42,
            "verbose": 0,
            "eval_metric": "Logloss",
            "task_type": "GPU",
            "devices": "0",
        }
        if params:
            self.params.update(params)
        self.params.update(kwargs)
        self.cat_features = cat_features
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        from catboost import CatBoostClassifier
        fit_params = {}
        if "eval_set" in kwargs:
            fit_params["eval_set"] = kwargs["eval_set"]
            fit_params["early_stopping_rounds"] = 50
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(X, y, cat_features=self.cat_features, **fit_params)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def get_params(self) -> dict[str, Any]:
        return self.params

    def feature_importance(self) -> dict[str, float] | None:
        if self.model is None:
            return None
        imp = self.model.get_feature_importance()
        names = self.model.feature_names_
        return dict(zip(names, imp))


MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "lgbm": LGBMModel,
    "xgb": XGBModel,
    "catboost": CatBoostModel,
}


def create_model(model_type: str, params: dict[str, Any] | None = None, **kwargs) -> BaseModel:
    """Factory function to create models by name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type](params=params, **kwargs)


def cross_validate(
    model_factory,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str = "roc_auc",
    n_folds: int = 5,
    seed: int = 42,
    stratified: bool = True,
    return_oof: bool = False,
) -> dict[str, Any]:
    """Run cross-validation and return metrics.

    Args:
        model_factory: Callable that returns a BaseModel instance
        X: Features
        y: Target
        metric: Metric name (from METRIC_FUNCTIONS)
        n_folds: Number of CV folds
        seed: Random seed
        stratified: Use StratifiedKFold
        return_oof: Return out-of-fold predictions

    Returns:
        Dict with cv_score, cv_std, fold_scores, duration, and optionally oof_preds
    """
    start_time = time.time()

    if stratified:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = kf.split(X, y)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = kf.split(X)

    metric_fn = METRIC_FUNCTIONS.get(metric)
    if metric_fn is None:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(METRIC_FUNCTIONS.keys())}")

    fold_scores = []
    oof_preds = np.zeros(len(X)) if return_oof else None
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_factory()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

        preds = model.predict(X_val)
        score = metric_fn(y_val, preds)
        fold_scores.append(score)
        models.append(model)

        if return_oof:
            oof_preds[val_idx] = preds

    duration = time.time() - start_time

    result = {
        "cv_score": float(np.mean(fold_scores)),
        "cv_std": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
        "duration": duration,
        "models": models,
    }

    if return_oof:
        result["oof_preds"] = oof_preds
        result["oof_score"] = float(metric_fn(y, oof_preds))

    return result
