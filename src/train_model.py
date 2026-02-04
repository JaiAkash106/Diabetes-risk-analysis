"""Train and select the best diabetes prediction model."""

from __future__ import annotations

import joblib
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    load_data,
)

TARGET_COLUMN = "diabetes"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.joblib"


def split_features_labels(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target labels."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    features = data[feature_columns]
    labels = data[TARGET_COLUMN]
    return features, labels


def train_and_select_model(csv_path: str) -> dict:
    """Train models, evaluate accuracy, and save the best-performing pipeline."""
    data = load_data(csv_path)
    features, labels = split_features_labels(data)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    preprocessor = build_preprocessor()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_model_name = None
    best_accuracy = 0.0
    best_pipeline: Pipeline | None = None

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("Model training failed; no pipeline was selected.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    return {
        "best_model": best_model_name,
        "accuracy": best_accuracy,
        "model_path": str(MODEL_PATH),
    }


if __name__ == "__main__":
    results = train_and_select_model("data/diabetes.csv")
    print(
        "Best model: {best_model} | Accuracy: {accuracy:.2f} | Saved to: {model_path}".format(
            **results
        )
    )
