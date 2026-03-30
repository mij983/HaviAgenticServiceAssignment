"""
Feedback Loop Agent
--------------------
Records every prediction made by ARIA, collects corrections from users,
and uses confirmed feedback to improve the knowledge base over time.

How it works:
  1. Every prediction is written to data/feedback.jsonl as a feedback entry
     with status "pending".

  2. The user is optionally prompted right after a prediction:
       "Was this assignment correct? (y/n/skip)"
     - "y"    -> entry marked "confirmed_correct"
     - "n"    -> user types the correct group -> entry marked "confirmed_wrong"
                  + correct_group stored
     - "skip" -> entry marked "skipped" (no action taken)

  3. Confirmed-correct entries are periodically promoted into ChromaDB
     as new training examples (source_type: "feedback") using:
       python feedback_loop.py --apply

  4. Wrong corrections are also stored and can be reviewed with:
       python feedback_loop.py --report

  5. To see a summary of all feedback collected:
       python feedback_loop.py --report

Storage:
  data/feedback.jsonl   — one JSON object per line, appended per prediction

Entry schema:
  {
    "id":                 "fb_<timestamp>_<hash>",
    "timestamp":          "2025-03-26T10:45:00",
    "short_description":  "VPN not connecting from home",
    "predicted_group":    "IT-Network Support",
    "confidence_score":   8,
    "confidence":         "high",
    "similarity_scores":  [9.2, 8.9, 8.4, 8.1, 7.9],
    "status":             "confirmed_correct" | "confirmed_wrong" | "skipped" | "pending",
    "correct_group":      "IT-Access Management"   # only if status=confirmed_wrong
  }
"""

import hashlib
import json
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_FEEDBACK_PATH = "data/feedback.jsonl"


class FeedbackAgent:

    def __init__(self, feedback_path: str = DEFAULT_FEEDBACK_PATH):
        self.feedback_path = feedback_path
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Recording predictions
    # ─────────────────────────────────────────────────────────────────────────

    def record_prediction(self, short_description: str, result: dict) -> str:
        """
        Save a prediction to feedback.jsonl with status "pending".

        Returns the entry ID so the caller can update it after collecting
        user feedback.
        """
        ts      = datetime.now().isoformat(timespec="seconds")
        raw     = short_description + ts
        fb_id   = "fb_" + hashlib.sha1(raw.encode()).hexdigest()[:10]

        sims = [
            t.get("similarity_score", 0)
            for t in result.get("similar_tickets", [])
        ]

        entry = {
            "id":                fb_id,
            "timestamp":         ts,
            "short_description": short_description,
            "predicted_group":   result.get("assignment_group", ""),
            "confidence_score":  result.get("confidence_score", 0),
            "confidence":        result.get("confidence", ""),
            "similarity_scores": sims,
            "status":            "pending",
            "correct_group":     "",
        }

        self._append(entry)
        return fb_id

    def update_feedback(
        self,
        fb_id:         str,
        status:        str,
        correct_group: str = "",
    ) -> None:
        """
        Update an existing feedback entry identified by fb_id.

        status values:
          "confirmed_correct"  — prediction was right
          "confirmed_wrong"    — prediction was wrong; correct_group must be provided
          "skipped"            — user chose not to give feedback
        """
        entries = self._read_all()
        updated = False
        for entry in entries:
            if entry.get("id") == fb_id:
                entry["status"]        = status
                entry["correct_group"] = correct_group
                updated = True
                break

        if updated:
            self._write_all(entries)
        else:
            logger.warning("Feedback entry not found for id: %s", fb_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Interactive feedback collection (called from predict.py)
    # ─────────────────────────────────────────────────────────────────────────

    def collect_interactive(
        self,
        fb_id:        str,
        predicted:    str,
        valid_groups: list[str],
    ) -> None:
        """
        Ask the user interactively whether the prediction was correct.
        Updates the feedback entry in-place.
        """
        print("  " + "-" * 56)
        print("  Feedback  (helps ARIA improve over time)")
        print("  Was '" + predicted + "' the correct team?")
        print("  Enter:  y = yes   n = no   s = skip")

        while True:
            try:
                answer = input("  Your answer [y/n/s]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "s"

            if answer in ("y", "yes"):
                self.update_feedback(fb_id, "confirmed_correct")
                print("  [Feedback saved] Marked as correct. Thank you!")
                break

            elif answer in ("n", "no"):
                print("  Please type the correct assignment group.")
                print("  (Press Enter to see the group list, or type the name directly)")

                while True:
                    try:
                        correct = input("  Correct group: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        correct = ""

                    if not correct:
                        # Print valid groups as a numbered list for reference
                        print("")
                        for idx, g in enumerate(valid_groups, 1):
                            print("  " + str(idx).rjust(3) + ". " + g)
                        print("")
                        continue

                    # Accept either group number or name
                    if correct.isdigit():
                        idx = int(correct) - 1
                        if 0 <= idx < len(valid_groups):
                            correct = valid_groups[idx]
                        else:
                            print("  Invalid number. Please try again.")
                            continue

                    # Fuzzy match
                    matched = self._fuzzy_match_group(correct, valid_groups)
                    if matched:
                        self.update_feedback(fb_id, "confirmed_wrong", matched)
                        print("  [Feedback saved] Recorded correct group: " + matched)
                        break
                    else:
                        print("  Group not recognised. Please try again or type a number.")
                break

            elif answer in ("s", "skip", ""):
                self.update_feedback(fb_id, "skipped")
                print("  [Feedback skipped]")
                break
            else:
                print("  Please enter y, n, or s.")

        print("  " + "-" * 56)

    # ─────────────────────────────────────────────────────────────────────────
    # Applying feedback to the knowledge base (called from feedback_loop.py)
    # ─────────────────────────────────────────────────────────────────────────

    def apply_to_knowledge_base(
        self,
        kb_agent,
        embedding_agent,
        min_confirmations: int = 1,
    ) -> int:
        """
        Promote confirmed-correct feedback entries into ChromaDB as new
        training examples.

        Only entries with status "confirmed_correct" are promoted.
        Each entry is embedded and stored with source_type="feedback" so
        it is searchable alongside CSV tickets and KB articles.

        Entries are upserted (safe to run multiple times).

        Args:
            kb_agent           : KnowledgeBaseAgent instance
            embedding_agent    : EmbeddingAgent instance
            min_confirmations  : minimum times an entry must be confirmed
                                 before being promoted (default: 1)

        Returns:
            int : number of entries promoted in this run
        """
        entries = self._read_all()
        confirmed = [e for e in entries if e.get("status") == "confirmed_correct"]

        if not confirmed:
            print("  No confirmed-correct feedback entries to apply.")
            return 0

        print("  Applying " + str(len(confirmed)) + " confirmed feedback entries to KB...")

        if kb_agent.collection is None:
            kb_agent._connect()

        promoted = 0
        for entry in confirmed:
            fb_id  = entry["id"]
            text   = entry["short_description"]
            group  = entry["predicted_group"]

            if not text or not group:
                continue

            emb = embedding_agent.embed(text)

            kb_agent.collection.upsert(
                ids        = [fb_id],
                embeddings = [emb],
                metadatas  = [{
                    "short_description": text,
                    "description":       text,
                    "assignment_group":  group,
                    "source_type":       "feedback",
                    "file_name":         "",
                }],
                documents = [text],
            )
            promoted += 1

        print("  [OK] Promoted " + str(promoted) + " feedback entries into knowledge base.")
        return promoted

    def apply_corrections_to_knowledge_base(
        self,
        kb_agent,
        embedding_agent,
    ) -> int:
        """
        Promote confirmed-wrong entries where the user provided the
        correct group. These go into ChromaDB as corrected training examples.

        Returns:
            int : number of corrections promoted
        """
        entries = self._read_all()
        corrections = [
            e for e in entries
            if e.get("status") == "confirmed_wrong" and e.get("correct_group")
        ]

        if not corrections:
            print("  No correction entries to apply.")
            return 0

        print("  Applying " + str(len(corrections)) + " correction(s) to KB...")

        if kb_agent.collection is None:
            kb_agent._connect()

        promoted = 0
        for entry in corrections:
            fb_id         = entry["id"] + "_corrected"
            text          = entry["short_description"]
            correct_group = entry["correct_group"]

            if not text or not correct_group:
                continue

            emb = embedding_agent.embed(text)

            kb_agent.collection.upsert(
                ids        = [fb_id],
                embeddings = [emb],
                metadatas  = [{
                    "short_description": text,
                    "description":       text,
                    "assignment_group":  correct_group,
                    "source_type":       "feedback_correction",
                    "file_name":         "",
                }],
                documents = [text],
            )
            promoted += 1

        print("  [OK] Promoted " + str(promoted) + " correction(s) into knowledge base.")
        return promoted

    # ─────────────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────────────

    def report(self) -> None:
        """Print a summary report of all feedback collected."""
        entries = self._read_all()

        if not entries:
            print("  No feedback entries found in: " + self.feedback_path)
            return

        total             = len(entries)
        correct           = sum(1 for e in entries if e.get("status") == "confirmed_correct")
        wrong             = sum(1 for e in entries if e.get("status") == "confirmed_wrong")
        skipped           = sum(1 for e in entries if e.get("status") == "skipped")
        pending           = sum(1 for e in entries if e.get("status") == "pending")
        accuracy          = round(correct / (correct + wrong) * 100, 1) if (correct + wrong) > 0 else 0

        print("")
        print("  " + "=" * 56)
        print("  ARIA Feedback Report")
        print("  " + "=" * 56)
        print("")
        print("  Total predictions recorded : " + str(total))
        print("  Confirmed correct          : " + str(correct))
        print("  Confirmed wrong            : " + str(wrong))
        print("  Skipped                    : " + str(skipped))
        print("  Pending (no feedback yet)  : " + str(pending))
        print("  Accuracy (confirmed only)  : " + str(accuracy) + "%")
        print("")

        if wrong > 0:
            print("  Wrong predictions (user corrections):")
            print("")
            print("  {:<45} {:<35} {}".format(
                "Short Description", "ARIA Predicted", "Correct Group"))
            print("  " + "-" * 45 + " " + "-" * 35 + " " + "-" * 35)
            for e in entries:
                if e.get("status") == "confirmed_wrong":
                    print("  {:<45} {:<35} {}".format(
                        e.get("short_description", "")[:44],
                        e.get("predicted_group", "")[:34],
                        e.get("correct_group", "")[:34],
                    ))
            print("")

        # Group-level accuracy breakdown
        group_stats: dict[str, dict] = {}
        for e in entries:
            if e.get("status") not in ("confirmed_correct", "confirmed_wrong"):
                continue
            grp = e.get("predicted_group", "Unknown")
            if grp not in group_stats:
                group_stats[grp] = {"correct": 0, "wrong": 0}
            if e["status"] == "confirmed_correct":
                group_stats[grp]["correct"] += 1
            else:
                group_stats[grp]["wrong"] += 1

        if group_stats:
            print("  Per-group accuracy:")
            print("")
            print("  {:<40} {:>8} {:>8} {:>10}".format(
                "Assignment Group", "Correct", "Wrong", "Accuracy"))
            print("  " + "-" * 40 + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 10)
            for grp, stats in sorted(group_stats.items()):
                c = stats["correct"]
                w = stats["wrong"]
                acc = round(c / (c + w) * 100, 1) if (c + w) > 0 else 0.0
                print("  {:<40} {:>8} {:>8} {:>9}%".format(grp[:39], c, w, acc))
            print("")

        print("  " + "=" * 56)
        print("")

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append(self, entry: dict) -> None:
        """Append one entry to the JSONL file."""
        with open(self.feedback_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _read_all(self) -> list[dict]:
        """Read all entries from the JSONL file."""
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
                        logger.warning("Skipping malformed feedback line: %s", line[:80])
        return entries

    def _write_all(self, entries: list[dict]) -> None:
        """Overwrite the JSONL file with the full list of entries."""
        with open(self.feedback_path, "w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry) + "\n")

    def _fuzzy_match_group(self, raw: str, valid_groups: list[str]) -> str:
        """Return the best matching valid group for a user-typed string, or ''."""
        raw_lower = raw.lower().strip()
        # Exact
        if raw in valid_groups:
            return raw
        # Case-insensitive exact
        for g in valid_groups:
            if g.lower() == raw_lower:
                return g
        # Substring
        for g in valid_groups:
            if raw_lower in g.lower() or g.lower() in raw_lower:
                return g
        return ""
