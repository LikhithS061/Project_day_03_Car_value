"""
preprocess.py — Data loading, cleaning, and feature engineering
================================================================
Part of the ML lifecycle for Used Car Price Prediction.

Steps:
  1. Load raw CSV data
  2. Handle missing values
  3. Feature engineering (CarAge, drop Car_Name)
  4. Encode categorical variables (one-hot)
  5. Split into features (X) and target (y)
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    df = pd.read_csv(path)
    print(f"[Preprocess] Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and basic cleaning."""
    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[Preprocess] Dropped {before - after} duplicate rows")

    # Drop rows with any missing values (dataset is small, can't afford imputation noise)
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"[Preprocess] Dropped {before - after} rows with missing values")

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering:
      - CarAge = current_year - Year
      - Drop Car_Name (too many unique values, not useful as-is)
      - Drop Year (replaced by CarAge)
    """
    current_year = datetime.now().year
    df = df.copy()

    # Create CarAge feature
    df["CarAge"] = current_year - df["Year"]

    # Drop columns that won't be used as features
    cols_to_drop = ["Car_Name", "Year"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    print(f"[Preprocess] Engineered features. Columns: {list(df.columns)}")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns (Fuel_Type, Seller_Type, Transmission)."""
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        print(f"[Preprocess] One-hot encoded: {categorical_cols}")

    return df


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline.
    Returns: X (features), y (target), feature_names
    """
    # Step 1: Clean
    df = clean_data(df)

    # Step 2: Feature engineering
    df = engineer_features(df)

    # Step 3: Encode categoricals
    df = encode_categoricals(df)

    # Step 4: Separate features and target
    target_col = "Selling_Price"
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    feature_names = list(X.columns)
    print(f"[Preprocess] Final shape: X={X.shape}, y={y.shape}")
    print(f"[Preprocess] Features: {feature_names}")

    return X, y, feature_names


if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), "..", "archive", "car data.csv")
    df = load_data(data_path)
    X, y, features = preprocess(df)
    print(f"\nSample X:\n{X.head()}")
    print(f"\nTarget stats:\n{y.describe()}")
