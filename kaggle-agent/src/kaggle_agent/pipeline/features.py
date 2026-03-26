"""Feature engineering utilities for tabular competitions."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Extensible feature engineering pipeline.

    Usage:
        fe = FeatureEngineer()
        fe.add_interaction_features(["col1", "col2", "col3"])
        fe.add_ratio_features([("col1", "col2")])
        fe.add_groupby_stats("col1", ["col2"], ["mean", "std"])

        X_train = fe.fit_transform(X_train)
        X_test = fe.transform(X_test)
    """

    def __init__(self):
        self._steps: list[tuple[str, dict[str, Any]]] = []
        self._fitted_params: dict[str, Any] = {}

    def add_interaction_features(
        self,
        columns: list[str],
        max_order: int = 2,
    ) -> FeatureEngineer:
        """Add multiplication interactions between columns."""
        self._steps.append(("interaction", {
            "columns": columns,
            "max_order": max_order,
        }))
        return self

    def add_ratio_features(
        self,
        pairs: list[tuple[str, str]],
    ) -> FeatureEngineer:
        """Add ratio features (col1 / col2)."""
        self._steps.append(("ratio", {"pairs": pairs}))
        return self

    def add_groupby_stats(
        self,
        group_col: str,
        agg_cols: list[str],
        stats: list[str] | None = None,
    ) -> FeatureEngineer:
        """Add group-by aggregate statistics."""
        if stats is None:
            stats = ["mean", "std", "min", "max"]
        self._steps.append(("groupby", {
            "group_col": group_col,
            "agg_cols": agg_cols,
            "stats": stats,
        }))
        return self

    def add_binning(
        self,
        column: str,
        n_bins: int = 10,
        strategy: str = "quantile",
    ) -> FeatureEngineer:
        """Bin a numeric column."""
        self._steps.append(("binning", {
            "column": column,
            "n_bins": n_bins,
            "strategy": strategy,
        }))
        return self

    def add_frequency_encoding(
        self,
        columns: list[str],
    ) -> FeatureEngineer:
        """Replace categorical values with their frequency."""
        self._steps.append(("frequency", {"columns": columns}))
        return self

    def add_target_encoding(
        self,
        columns: list[str],
        smoothing: float = 10.0,
    ) -> FeatureEngineer:
        """Add target encoding (must be fit with target)."""
        self._steps.append(("target_encoding", {
            "columns": columns,
            "smoothing": smoothing,
        }))
        return self

    def add_count_features(
        self,
        binary_columns: list[str],
        name: str = "service_count",
    ) -> FeatureEngineer:
        """Count number of positive values across binary columns."""
        self._steps.append(("count", {
            "columns": binary_columns,
            "name": name,
        }))
        return self

    def add_custom(
        self,
        name: str,
        func: Any,
    ) -> FeatureEngineer:
        """Add a custom feature function: func(df) -> df."""
        self._steps.append(("custom", {"name": name, "func": func}))
        return self

    def fit_transform(self, df: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Fit and transform the dataframe."""
        df = df.copy()
        self._fitted_params = {}

        for step_name, params in self._steps:
            df = self._apply_step(df, step_name, params, fit=True, y=y)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform using previously fitted parameters."""
        df = df.copy()
        for step_name, params in self._steps:
            df = self._apply_step(df, step_name, params, fit=False)
        return df

    def _apply_step(
        self,
        df: pd.DataFrame,
        step_name: str,
        params: dict,
        fit: bool = False,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        if step_name == "interaction":
            return self._interaction(df, params)
        elif step_name == "ratio":
            return self._ratio(df, params)
        elif step_name == "groupby":
            return self._groupby(df, params, fit)
        elif step_name == "binning":
            return self._binning(df, params, fit)
        elif step_name == "frequency":
            return self._frequency(df, params, fit)
        elif step_name == "target_encoding":
            return self._target_encode(df, params, fit, y)
        elif step_name == "count":
            return self._count(df, params)
        elif step_name == "custom":
            return params["func"](df)
        return df

    def _interaction(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        columns = [c for c in params["columns"] if c in df.columns]
        for r in range(2, params["max_order"] + 1):
            for combo in combinations(columns, r):
                name = "_x_".join(combo)
                df[name] = df[list(combo)].prod(axis=1)
        return df

    def _ratio(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        for num, den in params["pairs"]:
            if num in df.columns and den in df.columns:
                name = f"{num}_div_{den}"
                df[name] = df[num] / df[den].replace(0, np.nan)
                df[name] = df[name].fillna(0)
        return df

    def _groupby(self, df: pd.DataFrame, params: dict, fit: bool) -> pd.DataFrame:
        group_col = params["group_col"]
        if group_col not in df.columns:
            return df

        key = f"groupby_{group_col}"
        agg_cols = [c for c in params["agg_cols"] if c in df.columns]

        if fit:
            stats_dict = {}
            for agg_col in agg_cols:
                grouped = df.groupby(group_col)[agg_col]
                for stat in params["stats"]:
                    col_name = f"{group_col}_{agg_col}_{stat}"
                    stats_dict[col_name] = grouped.agg(stat)
            self._fitted_params[key] = stats_dict

        if key in self._fitted_params:
            for col_name, values in self._fitted_params[key].items():
                df[col_name] = df[group_col].map(values)

        return df

    def _binning(self, df: pd.DataFrame, params: dict, fit: bool) -> pd.DataFrame:
        col = params["column"]
        if col not in df.columns:
            return df

        key = f"bins_{col}"
        bin_col = f"{col}_binned"

        if fit:
            if params["strategy"] == "quantile":
                df[bin_col], bins = pd.qcut(
                    df[col], q=params["n_bins"], labels=False, retbins=True, duplicates="drop"
                )
            else:
                df[bin_col], bins = pd.cut(
                    df[col], bins=params["n_bins"], labels=False, retbins=True
                )
            self._fitted_params[key] = bins
        elif key in self._fitted_params:
            df[bin_col] = pd.cut(
                df[col], bins=self._fitted_params[key], labels=False
            )

        return df

    def _frequency(self, df: pd.DataFrame, params: dict, fit: bool) -> pd.DataFrame:
        for col in params["columns"]:
            if col not in df.columns:
                continue
            key = f"freq_{col}"
            if fit:
                freq = df[col].value_counts(normalize=True)
                self._fitted_params[key] = freq
            if key in self._fitted_params:
                df[f"{col}_freq"] = df[col].map(self._fitted_params[key]).fillna(0)
        return df

    def _target_encode(
        self, df: pd.DataFrame, params: dict, fit: bool, y: pd.Series | None
    ) -> pd.DataFrame:
        for col in params["columns"]:
            if col not in df.columns:
                continue
            key = f"te_{col}"
            if fit and y is not None:
                global_mean = y.mean()
                agg = pd.DataFrame({"target": y, "col": df[col]}).groupby("col")["target"]
                counts = agg.count()
                means = agg.mean()
                smooth = params["smoothing"]
                encoded = (counts * means + smooth * global_mean) / (counts + smooth)
                self._fitted_params[key] = encoded
                self._fitted_params[f"{key}_global"] = global_mean
            if key in self._fitted_params:
                df[f"{col}_te"] = df[col].map(self._fitted_params[key]).fillna(
                    self._fitted_params.get(f"{key}_global", 0)
                )
        return df

    def _count(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        cols = [c for c in params["columns"] if c in df.columns]
        if cols:
            df[params["name"]] = df[cols].sum(axis=1)
        return df

    @property
    def feature_names(self) -> list[str]:
        """Description of all feature engineering steps."""
        return [f"{name}: {params}" for name, params in self._steps]
