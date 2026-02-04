"""Prediction helpers for the diabetes risk model."""

from __future__ import annotations

import joblib
import pandas as pd
from pathlib import Path

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = Path("models") / "best_model.joblib"


def load_model(model_path: Path = MODEL_PATH):
    """Load a trained model pipeline from disk."""
    if not model_path.exists():
        raise FileNotFoundError(
            "Model file not found. Train the model first by running src/train_model.py"
        )
    return joblib.load(model_path)


def predict_risk(input_data: dict) -> dict:
    """Predict diabetes risk and return label and probability."""
    model = load_model()
    feature_order = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    input_frame = pd.DataFrame([input_data])[feature_order]
    probability = model.predict_proba(input_frame)[0][1]
    prediction = "Diabetic" if probability >= 0.5 else "Not Diabetic"

    return {
        "prediction": prediction,
        "probability": float(probability),
    }
