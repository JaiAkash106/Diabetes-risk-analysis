"""Data loading and preprocessing utilities for diabetes risk analysis."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age",
    "bmi",
    "blood_pressure",
    "glucose",
]

CATEGORICAL_FEATURES = [
    "physical_activity",
    "diet",
    "family_history",
]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the diabetes dataset from a CSV file."""
    return pd.read_csv(csv_path)


def build_preprocessor() -> ColumnTransformer:
    """Create a preprocessing pipeline for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor
