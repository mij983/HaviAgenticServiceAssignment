import os
import sys
import csv
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
from agents.historical_data_agent import HistoricalDataAgent

CSV_PATH = "data/training_tickets.csv"

AUGMENTATIONS = [
    lambda t: t,
    lambda t: {**t, "priority": "1"},
    lambda t: {**t, "priority": "4"},
    lambda t: {**t, "short_description": "URGENT: " + t["short_description"]},
    lambda t: {**t, "description": t["description"] + " Please escalate."},
    lambda t: {**t, "category": "", "subcategory": ""},
]


def main():
    print("")
    print("=" * 60)
    print("  Training Model from Real Team CSV")
    print("=" * 60)
    print("")

    if not os.path.exists(CSV_PATH):
        print("  ERROR: CSV not found at " + CSV_PATH)
        print("  Copy your team CSV there and retry.")
        print("")
        sys.exit(1)

    agent = HistoricalDataAgent()
    X, y = [], []

    with open(CSV_PATH, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            team = row.get("Assignment Team", "").strip()
            if not team:
                continue
            base = {
                "short_description": row.get("Short Description", ""),
                "description":       row.get("Description", ""),
                "category":          row.get("Category", ""),
                "subcategory":       row.get("SubCategory", ""),
                "priority":          row.get("Priority", ""),
            }
            for aug in AUGMENTATIONS:
                t = aug(dict(base))
                X.append(agent.build_features(t))
                y.append(team)

    X = np.array(X)
    y = np.array(y)

    from collections import Counter
    dist = Counter(y)
    print("  Source CSV rows    : " + str(len(y) // len(AUGMENTATIONS)))
    print("  Augmented samples  : " + str(len(X)))
    print("  Feature vector len : " + str(X.shape[1]))
    print("  Teams              : " + str(len(dist)))
    print("")
    for team, cnt in sorted(dist.items()):
        print("    {:<48} {:>3} samples".format(team, cnt))
    print("")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1, random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)
    print("  Cross-validation accuracy : " + str(round(scores.mean() * 100, 1)) + "%  (+/- " + str(round(scores.std() * 100, 1)) + "%)")
    print("")

    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/assignment_model.pkl")

    print("  Model saved -> models/assignment_model.pkl")
    print("")
    print("  Teams the model routes to:")
    for cls in model.classes_:
        print("    + " + cls)
    print("")
    print("  Next steps:")
    print("    python create_sample_tickets.py   <- push tickets to ServiceNow")
    print("    python run_agent.py               <- AI routes them")
    print("")


if __name__ == "__main__":
    main()
