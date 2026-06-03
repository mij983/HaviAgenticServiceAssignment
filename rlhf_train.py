"""
rlhf_train.py
--------------
CLI tool to manage the ARIA RLHF (Reinforcement Learning from Human Feedback)
training loop.

Commands:

  --apply
      Stage 3A: Upsert all confirmed feedback into ChromaDB.
      Correct predictions -> rlhf_positive entries (positive anchors).
      Wrong predictions   -> rlhf_negative_corrected entries (correction anchors).
      Safe to run multiple times (upsert — no duplicates).

  --compute-rewards
      Stage 2: Recompute per-group reward stats from all feedback and
      save to data/rlhf_rewards.json. Run this after collecting feedback
      to update the caution-group list used in the LLM prompt.

  --report
      Print a full reward summary: overall accuracy, per-group breakdown,
      flagged low-accuracy groups, list of all wrong predictions.

  --export FILE.csv
      Export raw feedback log to CSV for external review.

  --clear-pending
      Remove entries that were never rated (status: pending).
      Use periodically to keep the log file clean.

  --reset-rewards
      Wipe data/rlhf_rewards.json only (keeps raw feedback log).
      Use when you want to recompute rewards from scratch.

Typical weekly workflow:
  1. Run predict.py all week — users give y/n feedback after each prediction
  2. python rlhf_train.py --apply          # push feedback into ChromaDB
  3. python rlhf_train.py --compute-rewards  # update reward stats
  4. python rlhf_train.py --report         # review accuracy
  5. Repeat

Usage:
  python rlhf_train.py --report
  python rlhf_train.py --apply
  python rlhf_train.py --compute-rewards
  python rlhf_train.py --apply --compute-rewards
  python rlhf_train.py --export data/rlhf_export.csv
  python rlhf_train.py --clear-pending
  python rlhf_train.py --reset-rewards
"""

import argparse
import csv
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from agents.embedding_agent      import EmbeddingAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.rlhf_agent           import RLHFAgent


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ARIA — RLHF Training Manager")
    parser.add_argument("--apply",           action="store_true",
                        help="Upsert confirmed feedback into ChromaDB (Stage 3A)")
    parser.add_argument("--compute-rewards", action="store_true",
                        help="Recompute reward stats and save to rlhf_rewards.json (Stage 2)")
    parser.add_argument("--report",          action="store_true",
                        help="Print full RLHF reward report")
    parser.add_argument("--export",          type=str, default=None,
                        metavar="FILE.csv",
                        help="Export feedback log to CSV")
    parser.add_argument("--clear-pending",   action="store_true",
                        help="Remove unrated (pending) entries from feedback log")
    parser.add_argument("--reset-rewards",   action="store_true",
                        help="Wipe rlhf_rewards.json (keeps raw feedback log)")
    args = parser.parse_args()

    config = load_config()

    feedback_path = config.get("rlhf", {}).get("feedback_path", "data/rlhf_feedback.jsonl")
    rewards_path  = config.get("rlhf", {}).get("rewards_path",  "data/rlhf_rewards.json")
    db_path       = config["vector_db"]["path"]
    collection    = config["vector_db"]["collection"]
    embed_model   = config["embedding"]["model"]

    rlhf = RLHFAgent(feedback_path=feedback_path, rewards_path=rewards_path)

    print("")
    print("=" * 60)
    print("  ARIA — RLHF Training Manager")
    print("=" * 60)
    print("")
    print("  Feedback file  : " + feedback_path)
    print("  Rewards file   : " + rewards_path)
    print("")

    # ── Report ──────────────────────────────────────────────────────────────
    if args.report:
        rlhf.report()
        return

    # ── Export ──────────────────────────────────────────────────────────────
    if args.export:
        _export_csv(rlhf, args.export)
        return

    # ── Clear pending ────────────────────────────────────────────────────────
    if args.clear_pending:
        _clear_pending(rlhf)
        return

    # ── Reset rewards ────────────────────────────────────────────────────────
    if args.reset_rewards:
        if os.path.exists(rewards_path):
            os.remove(rewards_path)
            print("  [OK] Reward stats reset: " + rewards_path)
        else:
            print("  [INFO] No rewards file found — nothing to reset.")
        print("")
        return

    # ── Apply to KB ──────────────────────────────────────────────────────────
    if args.apply:
        print("  Loading embedding model: " + embed_model)
        embed_agent = EmbeddingAgent(model_name=embed_model)
        embed_agent.load()

        kb_agent = KnowledgeBaseAgent(db_path=db_path, collection_name=collection)
        kb_agent._connect()

        print("")
        print("  Stage 3A — Applying RLHF feedback to knowledge base...")
        pos, cor = rlhf.apply_to_knowledge_base(kb_agent, embed_agent)
        print("  [OK] Positive anchors upserted : " + str(pos))
        print("  [OK] Correction anchors upserted: " + str(cor))
        print("  [OK] Knowledge base total: " + str(kb_agent.count()) + " entries.")
        print("")

    # ── Compute rewards ──────────────────────────────────────────────────────
    if args.compute_rewards:
        print("  Stage 2 — Computing reward stats...")
        stats = rlhf.compute_rewards()
        rlhf.save_rewards(stats)
        print("  [OK] Reward stats saved to: " + rewards_path)
        print("  [OK] Groups tracked: " + str(len(stats)))

        flagged = rlhf.get_low_reward_groups()
        if flagged:
            print("")
            print("  ⚠  Low-accuracy groups flagged for prompt caution:")
            for g in flagged:
                print("     - " + g)
        print("")

    if not any([args.apply, args.compute_rewards, args.report,
                args.export, args.clear_pending, args.reset_rewards]):
        parser.print_help()
        print("")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _export_csv(rlhf: RLHFAgent, output_path: str) -> None:
    entries = rlhf._read_all()
    if not entries:
        print("  No feedback entries to export.")
        return

    fieldnames = [
        "id", "timestamp", "short_description", "predicted_group",
        "confidence_score", "confidence", "fallback",
        "status", "correct_group", "reward",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in entries:
            row = {k: e.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print("  Exported " + str(len(entries)) + " entries to: " + output_path)
    print("")


def _clear_pending(rlhf: RLHFAgent) -> None:
    entries = rlhf._read_all()
    before  = len(entries)
    kept    = [e for e in entries if e.get("status") != "pending"]
    removed = before - len(kept)
    rlhf._write_all(kept)
    print("  Removed " + str(removed) + " pending entries.")
    print("  Remaining: " + str(len(kept)) + " entries.")
    print("")


if __name__ == "__main__":
    main()
