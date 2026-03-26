"""Data loading and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_competition_data(
    train_path: str | Path,
    test_path: str | Path,
    target_column: str,
    id_column: str = "id",
    original_data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load train/test data, return (X_train, X_test, y_train, test_ids).

    If original_data_path is provided, blends original data with synthetic.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Extract target and IDs
    y_train = train[target_column]
    test_ids = test[id_column]

    # Drop id and target from features
    drop_cols = [c for c in [id_column, target_column] if c in train.columns]
    X_train = train.drop(columns=drop_cols)
    X_test = test.drop(columns=[id_column], errors="ignore")

    # Blend with original dataset if available
    if original_data_path and Path(original_data_path).exists():
        orig = pd.read_csv(original_data_path)
        if target_column in orig.columns:
            y_orig = orig[target_column]
            X_orig = orig.drop(columns=[c for c in drop_cols if c in orig.columns], errors="ignore")
            # Align columns
            common_cols = X_train.columns.intersection(X_orig.columns)
            X_train = pd.concat([X_train[common_cols], X_orig[common_cols]], ignore_index=True)
            y_train = pd.concat([y_train, y_orig], ignore_index=True)

    return X_train, X_test, y_train, test_ids


def preprocess_dataframe(
    df: pd.DataFrame,
    categorical_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Basic preprocessing: encode categoricals, handle missing values.

    Returns (processed_df, encoding_info).
    """
    df = df.copy()
    encoding_info: dict[str, Any] = {}

    # Auto-detect column types
    if categorical_columns is None:
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    # Label encode categoricals
    for col in categorical_columns:
        if col in df.columns:
            categories = sorted(df[col].dropna().unique())
            cat_map = {cat: i for i, cat in enumerate(categories)}
            df[col] = df[col].map(cat_map).astype("float32")
            encoding_info[col] = cat_map

    # Fill numeric NaNs with median
    for col in numeric_columns:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            encoding_info[f"{col}_fill"] = median_val

    return df, encoding_info


def apply_preprocessing(
    df: pd.DataFrame,
    encoding_info: dict[str, Any],
) -> pd.DataFrame:
    """Apply previously fitted preprocessing to new data."""
    df = df.copy()

    for col, mapping in encoding_info.items():
        if col.endswith("_fill"):
            real_col = col[:-5]
            if real_col in df.columns:
                df[real_col] = df[real_col].fillna(mapping)
        elif col in df.columns and isinstance(mapping, dict):
            df[col] = df[col].map(mapping).astype("float32")
            # Handle unseen categories
            if df[col].isna().any():
                df[col] = df[col].fillna(-1)

    return df


def get_data_summary(df: pd.DataFrame, name: str = "dataset") -> str:
    """Generate a human-readable data summary."""
    lines = [
        f"=== {name} ===",
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB",
        "",
        "Column types:",
    ]

    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns
        lines.append(f"  {dtype}: {len(cols)} columns")

    missing = df.isnull().sum()
    if missing.any():
        lines.append(f"\nMissing values: {missing[missing > 0].to_dict()}")
    else:
        lines.append("\nNo missing values.")

    return "\n".join(lines)
