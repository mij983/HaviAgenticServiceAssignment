"""
scripts/accuracy_report.py
----------------------------
Prints a human-readable accuracy report from the feedback database.

Usage:
    python scripts/accuracy_report.py
    python scripts/accuracy_report.py --db data/feedback.db
"""

import argparse
import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.learning_agent import LearningAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/feedback.db")
    parser.add_argument("--model", default="models/assignment_model.pkl")
    args = parser.parse_args()

    agent = LearningAgent(args.db, args.model)
    report = agent.accuracy_report()

    print("\n=== AI Assignment Agent – Accuracy Report ===\n")

    if report["total"] == 0:
        print("No feedback records found. Process some tickets first.\n")
        return

    print(f"Total predictions recorded : {report['total']}")
    print(f"Overall accuracy           : {report['accuracy']:.1%}\n")

    print("Per-group breakdown:")
    print(f"  {'Assignment Group':<30} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*7} {'-'*10}")
    for group, stats in sorted(report["per_group"].items()):
        print(
            f"  {group:<30} {stats['correct']:>8} {stats['total']:>7} "
            f"{stats['accuracy']:>10.1%}"
        )

    print()

    # Also print latest 10 audit log entries
    if os.path.exists(args.db):
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        conn.close()

        if rows:
            print("=== Latest 10 Routing Decisions ===\n")
            for r in rows:
                import datetime
                ts = datetime.datetime.fromtimestamp(r["created_at"]).strftime("%Y-%m-%d %H:%M:%S")
                status = "✅ Auto-assigned" if r["auto_assigned"] else "🔶 Manual triage"
                print(
                    f"  [{ts}] {r['ticket_number'] or 'N/A'} | "
                    f"{status} | Group: {r['predicted_group']} | "
                    f"Confidence: {r['confidence']} | {r['reason']}"
                )
            print()


if __name__ == "__main__":
    main()
