"""
RLHF Agent — Reinforcement Learning from Human Feedback
---------------------------------------------------------
Implements a lightweight RLHF loop that learns from every human
correction and uses that learning to improve future predictions.

Architecture
------------
Classic RLHF has three stages:

  1. Collect feedback  — human says "correct" or "wrong + right answer"
  2. Train reward model — learn what makes a prediction good or bad
  3. Update policy      — use reward signal to bias future predictions

In this system the "policy" is ChromaDB + the LLM prompt. We cannot
fine-tune the LLM weights locally, so we implement a practical
equivalent:

  Stage 1 — Feedback collection
      Every prediction is logged to data/rlhf_feedback.jsonl.
      After each prediction the user is asked:
        "Was this correct? (y / n / skip)"
      If wrong, the user provides the correct group.
      Status: confirmed_correct | confirmed_wrong | skipped | pending

  Stage 2 — Reward signal computation
      For each confirmed entry a reward score is computed:
        +1.0  confirmed_correct  (model was right)
        -1.0  confirmed_wrong    (model was wrong)
        +0.5  auto_correct       (model matched high-sim ticket, no human needed)
      Per-group running accuracy is maintained in data/rlhf_rewards.json.

  Stage 3 — Policy update (two mechanisms)
      A) Embedding reinforcement
         Confirmed-correct entries are upserted into ChromaDB with
         source_type="rlhf_positive". These act as high-signal anchors
         that pull future similar queries toward the correct group.

         Confirmed-wrong entries are stored with source_type="rlhf_negative"
         and the CORRECT group so the next similar query has a direct
         corrected example.

      B) Prompt bias injection
         Groups with low reward scores (< 0.4 accuracy, >= 5 attempts)
         are flagged. The LLM prompt includes a CAUTION block listing
         these groups with a reminder to only use them when evidence is strong.
         This nudges the LLM away from groups it historically confuses.

Storage
-------
  data/rlhf_feedback.jsonl   — one line per prediction (append-only log)
  data/rlhf_rewards.json     — per-group reward stats (updated on apply)

Usage
-----
  # Collect feedback interactively (called from predict.py automatically)
  rlhf_agent.collect(fb_id, predicted_group, valid_groups)

  # Apply all pending feedback to KB and update reward stats
  python rlhf_train.py --apply

  # View reward report
  python rlhf_train.py --report

  # Reset reward stats (keeps raw feedback log)
  python rlhf_train.py --reset-rewards
"""

import hashlib
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

FEEDBACK_PATH = "data/rlhf_feedback.jsonl"
REWARDS_PATH  = "data/rlhf_rewards.json"


class RLHFAgent:

    def __init__(
        self,
        feedback_path: str = FEEDBACK_PATH,
        rewards_path:  str = REWARDS_PATH,
    ):
        self.feedback_path = feedback_path
        self.rewards_path  = rewards_path
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        os.makedirs(os.path.dirname(rewards_path),  exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1 — Feedback collection
    # ─────────────────────────────────────────────────────────────────────────

    def record_prediction(self, short_description: str, result: dict) -> str:
        """
        Save a prediction to the feedback log with status 'pending'.
        Returns the fb_id so it can be updated after human feedback.
        """
        ts    = datetime.now().isoformat(timespec="seconds")
        raw   = short_description + ts
        fb_id = "rlhf_" + hashlib.sha1(raw.encode()).hexdigest()[:10]

        entry = {
            "id":                fb_id,
            "timestamp":         ts,
            "short_description": short_description,
            "predicted_group":   result.get("assignment_group", ""),
            "confidence_score":  result.get("confidence_score", 0),
            "confidence":        result.get("confidence", ""),
            "similarity_scores": [
                t.get("similarity_score", 0)
                for t in result.get("similar_tickets", [])
            ],
            "fallback":          result.get("fallback", False),
            "status":            "pending",
            "correct_group":     "",
            "reward":            None,
        }
        self._append(entry)
        return fb_id

    def collect_interactive(
        self,
        fb_id:        str,
        predicted:    str,
        valid_groups: list[str],
    ) -> None:
        """
        Ask the user whether the prediction was correct.
        Updates the feedback log entry in-place.
        """
        print("")
        print("  " + "─" * 56)
        print("  RLHF Feedback — help ARIA learn")
        print("  Predicted team: " + predicted)
        print("  Was this correct?  y = yes   n = no   s = skip")

        while True:
            try:
                ans = input("  Your answer [y/n/s]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "s"

            if ans in ("y", "yes"):
                self._update(fb_id, "confirmed_correct", "", reward=+1.0)
                print("  [RLHF] ✓ Correct — reward +1.0 recorded.")
                break

            elif ans in ("n", "no"):
                print("  Enter the correct group name or number:")
                groups_shown = False
                while True:
                    try:
                        correct = input("  Correct group: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        correct = ""

                    if not correct:
                        if not groups_shown:
                            print("")
                            for idx, g in enumerate(valid_groups, 1):
                                print("  " + str(idx).rjust(3) + ". " + g)
                            print("")
                            groups_shown = True
                        continue

                    # Accept number input
                    if correct.isdigit():
                        idx = int(correct) - 1
                        if 0 <= idx < len(valid_groups):
                            correct = valid_groups[idx]
                        else:
                            print("  Invalid number.")
                            continue

                    matched = self._fuzzy_match(correct, valid_groups)
                    if matched:
                        self._update(fb_id, "confirmed_wrong", matched, reward=-1.0)
                        print("  [RLHF] ✗ Wrong — reward -1.0 recorded.")
                        print("  [RLHF]   Correct group saved: " + matched)
                        break
                    else:
                        print("  Group not recognised. Try again or enter a number.")
                break

            elif ans in ("s", "skip", ""):
                self._update(fb_id, "skipped", "", reward=None)
                print("  [RLHF] Skipped.")
                break
            else:
                print("  Please enter y, n, or s.")

        print("  " + "─" * 56)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2 — Reward signal computation
    # ─────────────────────────────────────────────────────────────────────────

    def compute_rewards(self) -> dict[str, dict]:
        """
        Compute per-group reward stats from all confirmed feedback entries.

        Returns dict keyed by group name:
          {
            "attempts":  int,   # total confirmed predictions for this group
            "correct":   int,   # confirmed_correct count
            "wrong":     int,   # confirmed_wrong count
            "accuracy":  float, # correct / (correct + wrong)
            "reward_sum": float # sum of reward values
          }
        """
        entries = self._read_all()
        stats: dict[str, dict] = {}

        for e in entries:
            status = e.get("status", "")
            if status not in ("confirmed_correct", "confirmed_wrong"):
                continue

            group = e.get("predicted_group", "")
            if not group:
                continue

            if group not in stats:
                stats[group] = {
                    "attempts":   0,
                    "correct":    0,
                    "wrong":      0,
                    "accuracy":   0.0,
                    "reward_sum": 0.0,
                }

            stats[group]["attempts"]   += 1
            stats[group]["reward_sum"] += e.get("reward", 0.0) or 0.0

            if status == "confirmed_correct":
                stats[group]["correct"] += 1
            else:
                stats[group]["wrong"] += 1

        # Compute accuracy
        for grp, s in stats.items():
            total = s["correct"] + s["wrong"]
            s["accuracy"] = round(s["correct"] / total, 4) if total > 0 else 0.0

        return stats

    def save_rewards(self, stats: dict) -> None:
        """Persist reward stats to disk."""
        with open(self.rewards_path, "w", encoding="utf-8") as fh:
            json.dump(stats, fh, indent=2)

    def load_rewards(self) -> dict[str, dict]:
        """Load reward stats from disk. Returns empty dict if not found."""
        if not os.path.exists(self.rewards_path):
            return {}
        with open(self.rewards_path, encoding="utf-8") as fh:
            return json.load(fh)

    def get_low_reward_groups(
        self,
        min_attempts: int   = 5,
        max_accuracy: float = 0.4,
    ) -> list[str]:
        """
        Return groups that have low reward scores — i.e. the model
        frequently gets them wrong.

        These are used by the LLM agent to inject a caution hint in
        the prompt (Stage 3B — prompt bias injection).

        Args:
            min_attempts : only flag groups with at least this many attempts
            max_accuracy : flag groups with accuracy below this threshold
        """
        stats = self.load_rewards()
        flagged = []
        for grp, s in stats.items():
            if s["attempts"] >= min_attempts and s["accuracy"] <= max_accuracy:
                flagged.append(grp)
        return sorted(flagged)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3A — Embedding reinforcement (upsert into ChromaDB)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_to_knowledge_base(
        self,
        kb_agent,
        embedding_agent,
    ) -> tuple[int, int]:
        """
        Upsert confirmed feedback entries into ChromaDB.

          confirmed_correct  -> source_type = "rlhf_positive"
                                assignment_group = predicted_group
                                Acts as a positive anchor for future queries.

          confirmed_wrong    -> source_type = "rlhf_negative_corrected"
                                assignment_group = correct_group
                                Acts as a direct correction anchor.

        Safe to run multiple times — uses upsert, no duplicates.

        Returns:
            (positive_count, correction_count)
        """
        entries   = self._read_all()
        positives = [e for e in entries if e.get("status") == "confirmed_correct"
                     and e.get("predicted_group")]
        corrections = [e for e in entries if e.get("status") == "confirmed_wrong"
                       and e.get("correct_group")]

        if kb_agent.collection is None:
            kb_agent._connect()

        pos_count = 0
        for e in positives:
            text  = e["short_description"]
            group = e["predicted_group"]
            emb   = embedding_agent.embed(text)
            kb_agent.collection.upsert(
                ids        = [e["id"] + "_pos"],
                embeddings = [emb],
                metadatas  = [{
                    "short_description": text,
                    "description":       text,
                    "assignment_group":  group,
                    "source_type":       "rlhf_positive",
                    "file_name":         "",
                }],
                documents = [text],
            )
            pos_count += 1

        cor_count = 0
        for e in corrections:
            text  = e["short_description"]
            group = e["correct_group"]
            emb   = embedding_agent.embed(text)
            kb_agent.collection.upsert(
                ids        = [e["id"] + "_cor"],
                embeddings = [emb],
                metadatas  = [{
                    "short_description": text,
                    "description":       text,
                    "assignment_group":  group,
                    "source_type":       "rlhf_negative_corrected",
                    "file_name":         "",
                }],
                documents = [text],
            )
            cor_count += 1

        return pos_count, cor_count

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def report(self) -> None:
        """Print a full RLHF feedback and reward summary."""
        entries = self._read_all()

        if not entries:
            print("  No RLHF feedback entries found in: " + self.feedback_path)
            return

        total   = len(entries)
        correct = sum(1 for e in entries if e.get("status") == "confirmed_correct")
        wrong   = sum(1 for e in entries if e.get("status") == "confirmed_wrong")
        skipped = sum(1 for e in entries if e.get("status") == "skipped")
        pending = sum(1 for e in entries if e.get("status") == "pending")
        acc     = round(correct / (correct + wrong) * 100, 1) if (correct + wrong) > 0 else 0.0

        print("")
        print("  " + "=" * 60)
        print("  ARIA — RLHF Reward Report")
        print("  " + "=" * 60)
        print("")
        print("  Total predictions    : " + str(total))
        print("  Confirmed correct    : " + str(correct)  + "  (reward +1.0)")
        print("  Confirmed wrong      : " + str(wrong)    + "  (reward -1.0)")
        print("  Skipped              : " + str(skipped))
        print("  Pending (no feedback): " + str(pending))
        print("  Overall accuracy     : " + str(acc) + "%")
        print("")

        stats = self.compute_rewards()
        if stats:
            print("  Per-group reward stats:")
            print("")
            print("  {:<42} {:>8} {:>8} {:>8} {:>10}".format(
                "Assignment Group", "Correct", "Wrong", "Total", "Accuracy"))
            print("  " + "-" * 42 + " " + "-" * 8 + " " + "-" * 8
                  + " " + "-" * 8 + " " + "-" * 10)
            for grp, s in sorted(stats.items(), key=lambda x: x[1]["accuracy"]):
                flag = "  ⚠" if s["accuracy"] <= 0.4 and s["attempts"] >= 5 else ""
                print("  {:<42} {:>8} {:>8} {:>8} {:>9}%{}".format(
                    grp[:41],
                    s["correct"],
                    s["wrong"],
                    s["attempts"],
                    round(s["accuracy"] * 100, 1),
                    flag,
                ))
            print("")
            flagged = self.get_low_reward_groups()
            if flagged:
                print("  ⚠  Low-accuracy groups (will receive caution hint in LLM prompt):")
                for g in flagged:
                    print("     - " + g)
                print("")

        if wrong > 0:
            print("  Wrong predictions (user corrections):")
            print("")
            print("  {:<44} {:<35} {}".format(
                "Short Description", "ARIA Predicted", "Correct Group"))
            print("  " + "-" * 44 + " " + "-" * 35 + " " + "-" * 35)
            for e in entries:
                if e.get("status") == "confirmed_wrong":
                    print("  {:<44} {:<35} {}".format(
                        e.get("short_description", "")[:43],
                        e.get("predicted_group", "")[:34],
                        e.get("correct_group", "")[:34],
                    ))
            print("")

        print("  " + "=" * 60)
        print("")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append(self, entry: dict) -> None:
        with open(self.feedback_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _read_all(self) -> list[dict]:
        if not os.path.exists(self.feedback_path):
            return []
        entries = []
        with open(self.feedback_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed RLHF line: %s", line[:80])
        return entries

    def _write_all(self, entries: list[dict]) -> None:
        with open(self.feedback_path, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")

    def _update(self, fb_id: str, status: str, correct_group: str,
                reward: float | None) -> None:
        entries = self._read_all()
        for e in entries:
            if e.get("id") == fb_id:
                e["status"]        = status
                e["correct_group"] = correct_group
                e["reward"]        = reward
                break
        self._write_all(entries)

    def _fuzzy_match(self, raw: str, valid_groups: list[str]) -> str:
        raw_lower = raw.lower().strip()
        if raw in valid_groups:
            return raw
        for g in valid_groups:
            if g.lower() == raw_lower:
                return g
        for g in valid_groups:
            if raw_lower in g.lower() or g.lower() in raw_lower:
                return g
        return ""
