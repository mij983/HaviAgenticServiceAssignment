"""
train_own_model.py
-------------------
Trains a NEW model using your own ServiceNow ticket data.
Replaces the synthetic model with one trained on real patterns.

Features engineered from YOUR data:
  - Short description length, word count
  - Description length, word count
  - Combined text length
  - Category hash, Subcategory hash, Business Service hash
  - First-5-words hash of short description
  - First-5-words hash of description
  - Numeric priority

Usage:
    python train_own_model.py --csv data/my_servicenow_tickets.csv
    python train_own_model.py --csv data/my_servicenow_tickets.csv --analyze
"""

import argparse
import os
import sys
import json

import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

GREEN  = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"


def main():
    parser = argparse.ArgumentParser(description="Train model on your own ServiceNow data")
    parser.add_argument("--csv",     required=True,  help="Path to your exported CSV")
    parser.add_argument("--output",  default="models/assignment_model.pkl")
    parser.add_argument("--analyze", action="store_true", help="Show detailed data analysis before training")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Min tickets per group to include (default: 5)")
    args = parser.parse_args()

    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║   Train Model on Your Own ServiceNow Data                ║
╚══════════════════════════════════════════════════════════╝{RESET}""")

    # ── Load data ─────────────────────────────────────────────────────────────
    from agents.historical_data_agent import HistoricalDataAgent
    agent = HistoricalDataAgent()

    print(f"\n  {BOLD}Loading data from:{RESET} {args.csv}")
    result = agent.load_historical_csv(args.csv, label_column="assignment_group")
    if result is None:
        print(f"  {RED}❌ Could not load CSV. Check the file exists and has correct columns.{RESET}")
        sys.exit(1)

    X_all, y_all = result
    print(f"  {GREEN}✅ Loaded {len(X_all)} records{RESET}")

    # ── Data analysis ─────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv(args.csv)

    unique_groups = df["assignment_group"].value_counts()
    print(f"\n  {BOLD}Assignment Groups Found in Your Data:{RESET}\n")
    print(f"  {'Group':<35} {'Tickets':>8}  {'Bar'}")
    print(f"  {'─'*35} {'─'*8}  {'─'*20}")
    max_count = unique_groups.max()
    for grp, cnt in unique_groups.items():
        bar = "█" * int(cnt / max_count * 20)
        status = ""
        if cnt < args.min_samples:
            status = f"  {YELLOW}⚠ below min ({args.min_samples}) — will be excluded{RESET}"
        elif cnt < 20:
            status = f"  {YELLOW}⚠ low sample count{RESET}"
        else:
            status = f"  {GREEN}✓{RESET}"
        print(f"  {grp:<35} {cnt:>8}  {CYAN}{bar:<20}{RESET}{status}")

    # Filter groups with too few samples
    valid_groups = unique_groups[unique_groups >= args.min_samples].index.tolist()
    excluded     = unique_groups[unique_groups < args.min_samples].index.tolist()

    if excluded:
        print(f"\n  {YELLOW}Excluding {len(excluded)} group(s) with < {args.min_samples} tickets:{RESET}")
        for g in excluded:
            print(f"    • {g} ({unique_groups[g]} tickets)")

    if len(valid_groups) < 2:
        print(f"\n  {RED}❌ Need at least 2 groups with {args.min_samples}+ tickets to train.{RESET}")
        print(f"  Add more resolved tickets to ServiceNow and re-export.")
        sys.exit(1)

    # Filter to valid groups only
    mask = np.isin(y_all, valid_groups)
    X    = X_all[mask]
    y    = y_all[mask]

    print(f"\n  {BOLD}Training on:{RESET} {len(X)} records across {len(valid_groups)} groups")

    if args.analyze:
        _show_analysis(df, valid_groups)

    # ── Train ─────────────────────────────────────────────────────────────────
    from sklearn.linear_model     import LogisticRegression
    from sklearn.model_selection  import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.metrics          import classification_report, confusion_matrix
    from sklearn.preprocessing    import LabelEncoder

    print(f"\n  {BOLD}{'─'*56}{RESET}")
    print(f"  {BOLD}Training Model...{RESET}")
    print(f"  {BOLD}{'─'*56}{RESET}\n")

    # Stratified split — ensures each group is represented in test set
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback if some class has too few samples for stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")

    # Cross-validation
    n_splits = min(5, min(np.bincount(LabelEncoder().fit_transform(y_train))))
    n_splits = max(2, n_splits)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    print(f"  Cross-validation ({n_splits}-fold):")
    for i, s in enumerate(cv_scores, 1):
        bar = "█" * int(s * 20)
        color = GREEN if s > 0.7 else YELLOW if s > 0.5 else RED
        print(f"    Fold {i}:  {color}{bar:<20}{RESET}  {s:.3f}")
    print(f"    {'─'*40}")
    avg = cv_scores.mean()
    color = GREEN if avg > 0.7 else YELLOW if avg > 0.5 else RED
    print(f"    Average: {color}{BOLD}{avg:.3f}{RESET}  (±{cv_scores.std():.3f})\n")

    # Final training
    model.fit(X_train, y_train)

    # Test set evaluation
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"  {BOLD}Test Set Results:{RESET}\n")
    print(f"  {'Group':<35} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print(f"  {'─'*35} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for grp in valid_groups:
        if grp in report:
            r = report[grp]
            f1c = GREEN if r['f1-score'] > 0.7 else YELLOW if r['f1-score'] > 0.5 else RED
            print(f"  {grp:<35} {r['precision']:>10.3f} {r['recall']:>8.3f} "
                  f"{f1c}{r['f1-score']:>8.3f}{RESET} {int(r['support']):>8}")
    print(f"  {'─'*35} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    acc = report['accuracy']
    acc_color = GREEN if acc > 0.7 else YELLOW if acc > 0.5 else RED
    print(f"  {'Overall Accuracy':<35} {acc_color}{BOLD}{acc:>10.3f}{RESET}\n")

    # Interpretation
    if avg > 0.75:
        print(f"  {GREEN}{BOLD}✅ Excellent! Model is well-trained on your data.{RESET}")
    elif avg > 0.60:
        print(f"  {YELLOW}⚠️  Decent accuracy. More resolved tickets will improve this.{RESET}")
    else:
        print(f"  {RED}⚠️  Low accuracy. More training data needed (aim for 100+ per group).{RESET}")

    # ── Save model ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(model, args.output)

    # Save metadata alongside the model
    meta = {
        "classes":        list(model.classes_),
        "n_training":     int(len(X_train)),
        "n_test":         int(len(X_test)),
        "cv_accuracy":    round(float(avg), 4),
        "test_accuracy":  round(float(acc), 4),
        "source_csv":     args.csv,
        "features":       11,
        "feature_names": [
            "short_desc_len", "desc_len", "combined_len",
            "category_hash", "subcategory_hash", "business_service_hash",
            "short_desc_first5_hash", "desc_first5_hash",
            "priority_numeric", "short_desc_word_count", "desc_word_count"
        ]
    }
    meta_path = args.output.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  {BOLD}{'═'*56}{RESET}")
    print(f"  {GREEN}{BOLD}Model saved!{RESET}")
    print(f"  Model file  : {CYAN}{args.output}{RESET}")
    print(f"  Metadata    : {CYAN}{meta_path}{RESET}")
    print(f"  Groups      : {', '.join(model.classes_)}")
    print(f"  CV Accuracy : {BOLD}{avg:.1%}{RESET}")
    print()
    print(f"  {BOLD}Next step — run the agent with your new model:{RESET}")
    print(f"    {CYAN}python run_agent.py{RESET}")
    print(f"  {BOLD}{'═'*56}{RESET}\n")


def _show_analysis(df, valid_groups):
    print(f"\n  {BOLD}{'─'*56}{RESET}")
    print(f"  {BOLD}Data Analysis{RESET}")
    print(f"  {BOLD}{'─'*56}{RESET}\n")

    df_valid = df[df["assignment_group"].isin(valid_groups)]

    print(f"  {BOLD}Category → Assignment Group breakdown:{RESET}\n")
    ct = df_valid.groupby(["category", "assignment_group"]).size().unstack(fill_value=0)
    for cat in ct.index:
        print(f"  {CYAN}{cat}{RESET}")
        for grp in ct.columns:
            cnt = ct.loc[cat, grp]
            if cnt > 0:
                bar = "█" * min(cnt, 20)
                print(f"    {grp:<35} {bar} {cnt}")
    print()

    print(f"  {BOLD}Short description avg length per group:{RESET}")
    for grp in valid_groups:
        sub = df_valid[df_valid["assignment_group"] == grp]
        avg_len = sub["short_description"].str.len().mean()
        print(f"    {grp:<35} {avg_len:.0f} chars avg")
    print()


if __name__ == "__main__":
    main()
