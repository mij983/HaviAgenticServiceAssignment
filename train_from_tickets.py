"""
train_from_tickets.py
----------------------
Trains the routing model from your real CSV file.

  Source  : data/training_tickets.csv  (your Assignment_Ticket CSV)
  Routing : Based on Short Description + Description only
  Ignored : Configuration Item (as requested)

Run once before starting the agent:
    python train_from_tickets.py
"""

import os, sys, csv, joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.dirname(__file__))
from agents.historical_data_agent import HistoricalDataAgent

GREEN = "\033[92m"; CYAN = "\033[96m"; BOLD = "\033[1m"; RESET = "\033[0m"
CSV_PATH = "data/training_tickets.csv"

AUGMENTATIONS = [
    lambda t: t,                                                    # original
    lambda t: {**t, "priority": "1"},                              # critical priority
    lambda t: {**t, "priority": "4"},                              # low priority
    lambda t: {**t, "short_description": "URGENT: " + t["short_description"]},
    lambda t: {**t, "description": t["description"] + " Please escalate."},
    lambda t: {**t, "category": "", "subcategory": ""},            # empty category
]


def main():
    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║   Training Model — Routing by SD + Description Only     ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

    if not os.path.exists(CSV_PATH):
        print(f"  ❌  CSV not found: {CSV_PATH}")
        print(f"      Copy your team CSV there first.\n")
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
    print(f"  Source CSV rows    : {len(y) // len(AUGMENTATIONS)}")
    print(f"  Augmented samples  : {len(X)}")
    print(f"  Feature vector len : {X.shape[1]}")
    print(f"  Teams              : {len(dist)}\n")
    for team, cnt in sorted(dist.items()):
        print(f"    {team:<48} {cnt:>3} samples")
    print()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1, random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)
    print(f"  Cross-validation accuracy : {scores.mean():.1%}  (±{scores.std():.1%})\n")

    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/assignment_model.pkl")

    print(f"  {GREEN}✅  Model saved → models/assignment_model.pkl{RESET}\n")
    print(f"  Teams the model routes to:")
    for cls in model.classes_:
        print(f"    • {cls}")
    print(f"\n  {BOLD}Next steps:{RESET}")
    print(f"    python create_sample_tickets.py   ← push 50 tickets to ServiceNow")
    print(f"    python run_agent.py               ← AI routes them\n")


if __name__ == "__main__":
    main()
