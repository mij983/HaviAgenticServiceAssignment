"""
feedback_loop.py
-----------------
CLI tool to manage the ARIA feedback loop.

The feedback loop collects user corrections after each prediction and
uses them to strengthen the knowledge base over time so that ARIA
becomes more accurate on the same types of tickets it has gotten wrong.

Commands:

  --report
      Print a summary of all feedback collected: total predictions,
      accuracy %, confirmed correct, confirmed wrong, per-group breakdown.

  --apply
      Promote confirmed-correct predictions into ChromaDB as new
      training examples. Safe to run multiple times (upsert).

  --apply-corrections
      Promote confirmed-wrong corrections (where the user typed the
      correct group) into ChromaDB. These become high-signal examples.

  --apply-all
      Apply both --apply and --apply-corrections in one run.

  --export FILE.csv
      Export the full feedback log to a CSV file for external review
      or import into Excel / ServiceNow.

  --clear-pending
      Remove entries with status "pending" (predictions that were never
      rated by a user). Use periodically to keep the file tidy.

Usage:
  python feedback_loop.py --report
  python feedback_loop.py --apply
  python feedback_loop.py --apply-corrections
  python feedback_loop.py --apply-all
  python feedback_loop.py --export data/feedback_export.csv
  python feedback_loop.py --clear-pending
"""

import argparse
import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from agents.embedding_agent      import EmbeddingAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.feedback_agent       import FeedbackAgent


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="ARIA Feedback Loop Manager"
    )
    parser.add_argument("--report",            action="store_true",
                        help="Show feedback summary report")
    parser.add_argument("--apply",             action="store_true",
                        help="Promote confirmed-correct entries into KB")
    parser.add_argument("--apply-corrections", action="store_true",
                        help="Promote confirmed-wrong corrections into KB")
    parser.add_argument("--apply-all",         action="store_true",
                        help="Apply both correct and correction entries")
    parser.add_argument("--export",            type=str, default=None,
                        metavar="FILE.csv",
                        help="Export feedback log to CSV")
    parser.add_argument("--clear-pending",     action="store_true",
                        help="Remove unrated (pending) entries from feedback log")
    args = parser.parse_args()

    config = load_config()

    feedback_path = config.get("feedback", {}).get("path", "data/feedback.jsonl")
    db_path       = config["vector_db"]["path"]
    collection    = config["vector_db"]["collection"]
    embed_model   = config["embedding"]["model"]

    fb_agent = FeedbackAgent(feedback_path=feedback_path)

    print("")
    print("=" * 60)
    print("  ARIA -- Feedback Loop Manager")
    print("=" * 60)
    print("")
    print("  Feedback file  : " + feedback_path)
    print("")

    # ── Report ──────────────────────────────────────────────────────────────
    if args.report:
        fb_agent.report()
        return

    # ── Export ──────────────────────────────────────────────────────────────
    if args.export:
        _export_csv(fb_agent, args.export)
        return

    # ── Clear pending ────────────────────────────────────────────────────────
    if args.clear_pending:
        _clear_pending(fb_agent)
        return

    # ── Apply to KB — needs embed + KB agents ────────────────────────────────
    if args.apply or args.apply_corrections or args.apply_all:
        print("  Loading embedding model: " + embed_model)
        embed_agent = EmbeddingAgent(model_name=embed_model)
        embed_agent.load()

        kb_agent = KnowledgeBaseAgent(db_path=db_path, collection_name=collection)
        kb_agent._connect()

        if args.apply or args.apply_all:
            print("")
            print("  Promoting confirmed-correct entries...")
            fb_agent.apply_to_knowledge_base(kb_agent, embed_agent)

        if args.apply_corrections or args.apply_all:
            print("")
            print("  Promoting user corrections...")
            fb_agent.apply_corrections_to_knowledge_base(kb_agent, embed_agent)

        total = kb_agent.count()
        print("")
        print("  Knowledge base total (all sources): " + str(total) + " entries.")
        print("")
        return

    # No flag — show help
    parser.print_help()
    print("")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _export_csv(fb_agent: FeedbackAgent, output_path: str) -> None:
    entries = fb_agent._read_all()
    if not entries:
        print("  No feedback entries to export.")
        return

    fieldnames = [
        "id", "timestamp", "short_description", "predicted_group",
        "confidence_score", "confidence", "status", "correct_group",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            # Flatten similarity_scores list to a string for CSV
            entry_copy = dict(entry)
            entry_copy.pop("similarity_scores", None)
            writer.writerow(entry_copy)

    print("  Exported " + str(len(entries)) + " entries to: " + output_path)
    print("")


def _clear_pending(fb_agent: FeedbackAgent) -> None:
    entries  = fb_agent._read_all()
    before   = len(entries)
    kept     = [e for e in entries if e.get("status") != "pending"]
    removed  = before - len(kept)
    fb_agent._write_all(kept)
    print("  Removed " + str(removed) + " pending entries.")
    print("  Remaining entries: " + str(len(kept)))
    print("")


if __name__ == "__main__":
    main()
