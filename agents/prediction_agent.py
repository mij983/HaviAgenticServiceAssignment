import logging
import os
import sys
from typing import Optional

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def _build_fallback_model():
    # The real model is a sklearn Pipeline (StandardScaler + RandomForestClassifier).
    # If it is missing, exit with a clear message rather than loading a dummy
    # that will silently produce wrong predictions.
    print("")
    print("  [ERROR] Model file not found.")
    print("  The model must be trained before running the agent.")
    print("  Run:  python train_from_tickets.py")
    print("")
    sys.exit(1)


class AssignmentPredictionAgent:

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                model = joblib.load(self.model_path)
                logger.info("Model loaded from '" + self.model_path + "'.")
                return model
            except Exception as e:
                logger.error("Failed to load model: " + str(e))
                _build_fallback_model()
        else:
            _build_fallback_model()

    def reload(self):
        self.model = self._load_model()

    def predict(self, feature_vector: list) -> tuple[str, float]:
        try:
            probabilities = self.model.predict_proba([feature_vector])[0]
            best_index = int(np.argmax(probabilities))
            assignment_group = self.model.classes_[best_index]
            raw_probability = float(probabilities[best_index])
            return assignment_group, raw_probability
        except Exception as e:
            logger.error("Prediction error: " + str(e))
            return "Unknown", 0.0

    def predict_top_n(self, feature_vector: list, n: int = 3) -> list[tuple[str, float]]:
        try:
            probabilities = self.model.predict_proba([feature_vector])[0]
            indexed = sorted(
                enumerate(probabilities), key=lambda x: x[1], reverse=True
            )[:n]
            return [(self.model.classes_[i], float(p)) for i, p in indexed]
        except Exception as e:
            logger.error("Top-N prediction error: " + str(e))
            return []
