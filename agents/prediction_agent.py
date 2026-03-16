"""
Prediction Agent
-----------------
Loads the trained ML model and produces an assignment group prediction
along with a raw model probability (later scaled to 1–10 by the
Confidence Scoring Engine).
"""

import logging
import os
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _build_dummy_model() -> LogisticRegression:
    """
    Creates a trivially-trained dummy model so the service can start
    even when no real model file exists yet.  This should be replaced by
    running scripts/train_model.py before going to production.
    """
    model = LogisticRegression(max_iter=500)
    # Three dummy samples – one per default assignment group
    X_dummy = [
        [50, 100, 150, 10, 20, 30, 40, 50, 3, 8, 20],
        [30, 80,  110, 20, 30, 40, 50, 60, 2, 5, 15],
        [70, 120, 190, 30, 40, 50, 60, 70, 4, 12, 25],
    ]
    y_dummy = ["Network Support", "Application Support", "Cloud Operations"]
    model.fit(X_dummy, y_dummy)
    return model


class AssignmentPredictionAgent:
    """
    Wraps a scikit-learn classifier and exposes a predict() method
    that returns (assignment_group, raw_probability).
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logger.info(f"Model loaded from '{self.model_path}'.")
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}. Using dummy model.")
        else:
            logger.warning(
                f"Model file not found at '{self.model_path}'. "
                "Using dummy model. Run scripts/train_model.py to create a real one."
            )
        return _build_dummy_model()

    def reload(self):
        """
        Hot-reloads the model from disk (called after retraining).
        """
        self.model = self._load_model()

    def predict(self, feature_vector: list) -> tuple[str, float]:
        """
        Returns:
            assignment_group  – predicted class label
            raw_probability   – highest class probability (0.0–1.0)
        """
        try:
            probabilities = self.model.predict_proba([feature_vector])[0]
            best_index = int(np.argmax(probabilities))
            assignment_group = self.model.classes_[best_index]
            raw_probability = float(probabilities[best_index])
            return assignment_group, raw_probability
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Unknown", 0.0

    def predict_top_n(self, feature_vector: list, n: int = 3) -> list[tuple[str, float]]:
        """
        Returns the top-n predictions sorted by probability descending.
        Useful for audit logs and human-triage UI.
        """
        try:
            probabilities = self.model.predict_proba([feature_vector])[0]
            indexed = sorted(
                enumerate(probabilities), key=lambda x: x[1], reverse=True
            )[:n]
            return [(self.model.classes_[i], float(p)) for i, p in indexed]
        except Exception as e:
            logger.error(f"Top-N prediction error: {e}")
            return []
