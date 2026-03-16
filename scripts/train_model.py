"""
scripts/train_model.py
-----------------------
One-time (and periodic) model training script.

Usage:
    python scripts/train_model.py --csv path/to/historical_tickets.csv

The CSV must have these columns (from a ServiceNow export):
    short_description, description, category, subcategory,
    business_service, priority, assignment_group

Outputs:
    models/assignment_model.pkl
"""

import argparse
import os
import sys

# Make the project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report

from agents.historical_data_agent import HistoricalDataAgent


def main():
    parser = argparse.ArgumentParser(description="Train the ticket assignment model.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the historical tickets CSV exported from ServiceNow.",
    )
    parser.add_argument(
        "--output",
        default="models/assignment_model.pkl",
        help="Where to save the trained model. Default: models/assignment_model.pkl",
    )
    parser.add_argument(
        "--label-column",
        default="assignment_group",
        help="Column name for the target label. Default: assignment_group",
    )
    args = parser.parse_args()

    print(f"[train_model] Loading data from '{args.csv}'…")
    agent = HistoricalDataAgent()
    result = agent.load_historical_csv(args.csv, label_column=args.label_column)

    if result is None:
        print("[train_model] Failed to load data. Exiting.")
        sys.exit(1)

    X, y = result
    print(f"[train_model] Loaded {len(X)} samples across {len(np.unique(y))} classes.")
    print(f"[train_model] Classes: {sorted(np.unique(y))}")

    if len(np.unique(y)) < 2:
        print("[train_model] Need at least 2 distinct assignment groups. Exiting.")
        sys.exit(1)

    # ── Train / test split ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(y) > 10 else None
    )

    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")

    # ── Cross-validation ──────────────────────────────────────────────────
    cv_folds = min(5, len(X_train) // max(len(np.unique(y)), 2))
    if cv_folds >= 2:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)
        print(
            f"[train_model] Cross-val accuracy: "
            f"{cv_scores.mean():.3f} ±{cv_scores.std():.3f} ({cv_folds}-fold)"
        )

    # ── Final training ────────────────────────────────────────────────────
    model.fit(X_train, y_train)

    # ── Evaluation ───────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print("\n[train_model] Test-set classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(model, args.output)
    print(f"[train_model] Model saved to '{args.output}'. ✅")


if __name__ == "__main__":
    main()
