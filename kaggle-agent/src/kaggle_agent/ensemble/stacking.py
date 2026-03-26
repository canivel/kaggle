"""Multi-level stacking and ensemble strategies.

Inspired by top Kaggle solutions that blend 30-70+ models.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from kaggle_agent.pipeline.models import BaseModel, cross_validate


class StackedEnsemble:
    """Two-level stacking ensemble.

    Level 0: Base models generate out-of-fold predictions
    Level 1: Meta-learner combines OOF predictions

    Usage:
        ensemble = StackedEnsemble(
            base_models={
                "lgbm": lambda: LGBMModel(params={...}),
                "xgb": lambda: XGBModel(params={...}),
                "catboost": lambda: CatBoostModel(params={...}),
            },
            meta_learner="logistic",
        )
        ensemble.fit(X_train, y_train)
        predictions = ensemble.predict(X_test)
    """

    def __init__(
        self,
        base_model_factories: dict[str, Any],
        meta_learner: str = "logistic",
        n_folds: int = 5,
        seed: int = 42,
        metric: str = "roc_auc",
    ):
        self.base_factories = base_model_factories
        self.meta_learner_type = meta_learner
        self.n_folds = n_folds
        self.seed = seed
        self.metric = metric

        self.base_models: dict[str, list[BaseModel]] = {}
        self.meta_model = None
        self.oof_scores: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Fit the stacked ensemble."""
        oof_matrix = np.zeros((len(X), len(self.base_factories)))
        model_names = list(self.base_factories.keys())

        # Level 0: Generate OOF predictions
        for i, (name, factory) in enumerate(self.base_factories.items()):
            result = cross_validate(
                factory, X, y,
                metric=self.metric,
                n_folds=self.n_folds,
                seed=self.seed,
                return_oof=True,
            )

            oof_matrix[:, i] = result["oof_preds"]
            self.base_models[name] = result["models"]
            self.oof_scores[name] = result["cv_score"]

        # Level 1: Fit meta-learner on OOF predictions
        meta_X = pd.DataFrame(oof_matrix, columns=model_names)

        if self.meta_learner_type == "logistic":
            self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
            self.meta_model.fit(meta_X, y)
        elif self.meta_learner_type == "ridge":
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(meta_X, y)
        elif self.meta_learner_type == "average":
            self.meta_model = None  # Simple average

        # Compute ensemble OOF score
        if self.meta_model is not None:
            if hasattr(self.meta_model, "predict_proba"):
                ensemble_oof = self.meta_model.predict_proba(meta_X)[:, 1]
            else:
                ensemble_oof = self.meta_model.predict(meta_X)
        else:
            ensemble_oof = oof_matrix.mean(axis=1)

        from sklearn.metrics import roc_auc_score
        ensemble_score = roc_auc_score(y, ensemble_oof)

        return {
            "oof_scores": self.oof_scores,
            "ensemble_score": ensemble_score,
            "model_names": model_names,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using all base models and meta-learner."""
        test_preds = np.zeros((len(X), len(self.base_factories)))
        model_names = list(self.base_factories.keys())

        for i, (name, models) in enumerate(self.base_models.items()):
            # Average predictions across folds
            fold_preds = np.zeros(len(X))
            for model in models:
                fold_preds += model.predict(X)
            test_preds[:, i] = fold_preds / len(models)

        meta_X = pd.DataFrame(test_preds, columns=model_names)

        if self.meta_model is not None:
            if hasattr(self.meta_model, "predict_proba"):
                return self.meta_model.predict_proba(meta_X)[:, 1]
            return self.meta_model.predict(meta_X)

        return test_preds.mean(axis=1)


class WeightedEnsemble:
    """Simple weighted ensemble with optimized weights."""

    def __init__(self):
        self.weights: np.ndarray | None = None

    def optimize_weights(
        self,
        oof_predictions: dict[str, np.ndarray],
        y_true: pd.Series,
        metric: str = "roc_auc",
    ) -> dict[str, float]:
        """Find optimal blend weights via grid search."""
        from itertools import product

        names = list(oof_predictions.keys())
        preds = [oof_predictions[n] for n in names]

        from kaggle_agent.pipeline.models import METRIC_FUNCTIONS
        metric_fn = METRIC_FUNCTIONS[metric]

        best_score = -np.inf
        best_weights = None

        # Grid search over weight space (step=0.05)
        steps = np.arange(0, 1.05, 0.05)
        n_models = len(names)

        if n_models == 2:
            for w in steps:
                weights = [w, 1 - w]
                blended = sum(w * p for w, p in zip(weights, preds))
                score = metric_fn(y_true, blended)
                if score > best_score:
                    best_score = score
                    best_weights = weights
        elif n_models == 3:
            for w1 in steps:
                for w2 in np.arange(0, 1.05 - w1, 0.05):
                    w3 = 1 - w1 - w2
                    if w3 < 0:
                        continue
                    weights = [w1, w2, w3]
                    blended = sum(w * p for w, p in zip(weights, preds))
                    score = metric_fn(y_true, blended)
                    if score > best_score:
                        best_score = score
                        best_weights = weights
        else:
            # For many models, use scipy optimize
            from scipy.optimize import minimize

            def neg_score(w):
                w = np.abs(w) / np.abs(w).sum()
                blended = sum(wi * p for wi, p in zip(w, preds))
                return -metric_fn(y_true, blended)

            x0 = np.ones(n_models) / n_models
            result = minimize(neg_score, x0, method="Nelder-Mead")
            best_weights = np.abs(result.x) / np.abs(result.x).sum()
            best_score = -result.fun

        self.weights = np.array(best_weights)
        return {
            "weights": dict(zip(names, best_weights)),
            "best_score": best_score,
        }

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Apply optimized weights to predictions."""
        if self.weights is None:
            return np.mean(predictions, axis=0)
        return sum(w * p for w, p in zip(self.weights, predictions))
