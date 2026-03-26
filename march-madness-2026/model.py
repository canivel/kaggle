"""
Model training and evaluation module for March Madness prediction.

Predicts P(lower_TeamID wins) for every possible matchup.
Evaluation metric: Brier score (mean squared error of probabilities).
"""

import json
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def brier_score(y_true, y_pred):
    """Mean squared error between true labels and predicted probabilities."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return np.mean((y_true - y_pred) ** 2)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_predictions(y_pred, y_true_train=None, y_pred_train=None):
    """Calibrate predicted probabilities via Platt scaling.

    If training predictions and labels are provided, fits a logistic
    regression on the raw predictions and applies it.  Otherwise just
    clips to [0.01, 0.99].
    """
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true_train is not None and y_pred_train is not None:
        y_true_train = np.asarray(y_true_train, dtype=np.float64)
        y_pred_train = np.asarray(y_pred_train, dtype=np.float64).reshape(-1, 1)
        lr = LogisticRegression(C=1e10, max_iter=5000, solver="lbfgs")
        lr.fit(y_pred_train, y_true_train)
        calibrated = lr.predict_proba(y_pred.reshape(-1, 1))[:, 1]
        return np.clip(calibrated, 0.01, 0.99)

    return np.clip(y_pred, 0.01, 0.99)


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

XGB_DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": 0,
}

LGBM_DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbose": -1,
}


# ---------------------------------------------------------------------------
# MarchMadnessModel
# ---------------------------------------------------------------------------

class MarchMadnessModel:
    """Unified interface for XGBoost, LightGBM, Logistic Regression, or an
    ensemble of all three."""

    def __init__(self, model_type="xgb", params=None):
        self.model_type = model_type
        self.params = params or {}
        self.model = None
        self._models = {}  # used by ensemble
        self._scaler = None  # used by logistic
        self._feature_names = None

        if model_type == "xgb":
            merged = {**XGB_DEFAULT_PARAMS, **self.params}
            self.model = xgb.XGBClassifier(**merged)
        elif model_type == "lgbm":
            merged = {**LGBM_DEFAULT_PARAMS, **self.params}
            self.model = lgb.LGBMClassifier(**merged)
        elif model_type == "logistic":
            C = self.params.get("C", 1.0)
            max_iter = self.params.get("max_iter", 1000)
            self.model = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
            self._scaler = StandardScaler()
        elif model_type == "ensemble":
            self._models["xgb"] = MarchMadnessModel("xgb", params)
            self._models["lgbm"] = MarchMadnessModel("lgbm", params)
            self._models["logistic"] = MarchMadnessModel("logistic", params)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X, y):
        """Fit the model on feature matrix X and binary labels y."""
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)

        if self.model_type == "ensemble":
            for m in self._models.values():
                m.train(X, y)
            return

        if self.model_type == "logistic":
            X_scaled = self._scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
        else:
            self.model.fit(X, y)

    def predict(self, X):
        """Return array of predicted probabilities."""
        if self.model_type == "ensemble":
            preds = np.column_stack(
                [m.predict(X) for m in self._models.values()]
            )
            return preds.mean(axis=1)

        if self.model_type == "logistic":
            X_scaled = self._scaler.transform(X)
            return self.model.predict_proba(X_scaled)[:, 1]

        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        """Return dict mapping feature name -> importance score."""
        if self.model_type == "ensemble":
            combined = {}
            for name, m in self._models.items():
                imp = m.get_feature_importance()
                for feat, val in imp.items():
                    combined[feat] = combined.get(feat, 0.0) + val / len(self._models)
            return combined

        if self.model_type == "logistic":
            coefs = np.abs(self.model.coef_[0])
            names = self._feature_names or [f"f{i}" for i in range(len(coefs))]
            return dict(zip(names, coefs))

        if self.model_type == "xgb":
            imp = self.model.feature_importances_
            names = self._feature_names or [f"f{i}" for i in range(len(imp))]
            return dict(zip(names, imp))

        if self.model_type == "lgbm":
            imp = self.model.feature_importances_
            names = self._feature_names or [f"f{i}" for i in range(len(imp))]
            return dict(zip(names, imp))

        return {}


# ---------------------------------------------------------------------------
# Neural network model
# ---------------------------------------------------------------------------

class _TabularNet(nn.Module):
    """Simple feed-forward network for tabular binary classification."""

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralNetModel:
    """PyTorch neural network for tabular March Madness prediction."""

    def __init__(self, lr=0.001, epochs=100, batch_size=256, patience=10):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._scaler = StandardScaler()
        self._feature_names = None

    def train(self, X, y):
        """Train the neural network. X and y are numpy arrays (or DataFrames)."""
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Scale features
        X = self._scaler.fit_transform(X)

        # Train / validation split (90/10)
        n = len(X)
        indices = np.random.RandomState(42).permutation(n)
        val_size = max(1, int(0.1 * n))
        val_idx, train_idx = indices[:val_size], indices[val_size:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32, device=self.device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        n_features = X_train.shape[1]
        self.model = _TabularNet(n_features).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for epoch in range(self.epochs):
            self.model.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        self.model.eval()

    def predict(self, X):
        """Return numpy array of predicted probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.asarray(X, dtype=np.float32)
        X = self._scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            preds = self.model(X_t).cpu().numpy()
        return preds

    def get_feature_importance(self):
        """Approximate feature importance from first-layer weights."""
        if self.model is None:
            return {}
        # Weight of the first Linear layer (after BatchNorm)
        first_linear = self.model.net[1]
        weights = first_linear.weight.detach().cpu().numpy()
        importance = np.abs(weights).mean(axis=0)
        names = self._feature_names or [f"f{i}" for i in range(len(importance))]
        return dict(zip(names, importance.tolist()))


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_by_season(model_class, X, y, seasons_col, n_splits=None):
    """Leave-one-season-out cross-validation.

    Parameters
    ----------
    model_class : callable
        A no-arg callable that returns a fresh model instance with a
        `train(X, y)` method and a `predict(X)` method.
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Binary target (1 = lower TeamID wins).
    seasons_col : array-like
        Season identifier for each row (same length as X).
    n_splits : int or None
        If given, only use the most recent n_splits seasons as held-out
        folds (still trains on all other seasons).

    Returns
    -------
    dict with keys: mean_brier, per_season_brier, all_predictions, all_actuals
    """
    y = np.asarray(y)
    seasons = np.asarray(seasons_col)
    unique_seasons = np.sort(np.unique(seasons))

    if n_splits is not None:
        unique_seasons = unique_seasons[-n_splits:]

    per_season_brier = {}
    all_predictions = np.empty(0)
    all_actuals = np.empty(0)

    for season in unique_seasons:
        test_mask = seasons == season
        train_mask = ~test_mask

        X_train = X.loc[train_mask] if isinstance(X, pd.DataFrame) else X[train_mask]
        y_train = y[train_mask]
        X_test = X.loc[test_mask] if isinstance(X, pd.DataFrame) else X[test_mask]
        y_test = y[test_mask]

        model = model_class()
        model.train(X_train, y_train)
        preds = model.predict(X_test)

        score = brier_score(y_test, preds)
        per_season_brier[int(season)] = score
        all_predictions = np.concatenate([all_predictions, preds])
        all_actuals = np.concatenate([all_actuals, y_test])

        print(f"  Season {season}: Brier={score:.6f}  (n={len(y_test)})")

    mean_b = brier_score(all_actuals, all_predictions)
    print(f"  Overall Brier: {mean_b:.6f}  (n={len(all_actuals)})")

    return {
        "mean_brier": mean_b,
        "per_season_brier": per_season_brier,
        "all_predictions": all_predictions,
        "all_actuals": all_actuals,
    }


# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------

def run_experiment(name, model, X, y, seasons_col, experiments_file="experiments.tsv"):
    """Run leave-one-season-out CV and log results.

    Parameters
    ----------
    name : str
        Human-readable description of this experiment.
    model : callable
        A no-arg callable returning a fresh model with train/predict.
    X, y, seasons_col : passed to cross_validate_by_season.
    experiments_file : str
        Path to the TSV log file.

    Returns
    -------
    dict – results from cross_validate_by_season, plus experiment_id.
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {name}")
    print(f"{'='*60}")

    results = cross_validate_by_season(model, X, y, seasons_col)

    experiment_id = str(uuid.uuid4())[:8]
    row = {
        "experiment_id": experiment_id,
        "description": name,
        "brier_score": f"{results['mean_brier']:.6f}",
        "per_season_scores": json.dumps(
            {str(k): round(v, 6) for k, v in results["per_season_brier"].items()}
        ),
        "notes": "",
        "kept": "",
    }

    write_header = not os.path.exists(experiments_file)
    with open(experiments_file, "a", newline="") as f:
        if write_header:
            f.write("\t".join(row.keys()) + "\n")
        f.write("\t".join(row.values()) + "\n")

    print(f"\nLogged as experiment {experiment_id} -> {experiments_file}")
    results["experiment_id"] = experiment_id
    return results
