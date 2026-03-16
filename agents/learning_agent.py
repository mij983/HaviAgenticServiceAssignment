"""
Learning Agent — Continuous Learning Loop
------------------------------------------
Flow:
  1. Ticket comes in → low confidence → MANUAL TRIAGE
     - Ticket stored in `manual_triage` table with AI prediction
     - Work note added: "AI suggested X, please assign manually"

  2. Human assigns it in ServiceNow (or it gets resolved)
     - `poll_manual_triage_outcomes()` checks ServiceNow for any
       triage tickets that now have an assignment_group set
     - Stores (features, human_assigned_group) as training data

  3. Retrain triggered automatically every N new feedback records
     - Merges original CSV training data + all manual triage feedback
     - Saves new model → next ticket of same type auto-assigns

  4. Confidence threshold per group auto-adjusts
     - Groups with 10+ feedback records and high accuracy get
       a lower threshold so they auto-assign sooner
"""

import json
import logging
import os
import sqlite3
import time
from typing import Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

logger = logging.getLogger(__name__)

# Path to original CSV training data — always included in retrain
CSV_TRAINING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "training_tickets.csv"
)


class LearningAgent:

    SCHEMA = """
        -- Every routing decision ever made (audit trail)
        CREATE TABLE IF NOT EXISTS audit_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_number    TEXT NOT NULL,
            sys_id           TEXT NOT NULL,
            predicted_group  TEXT,
            confidence       REAL,
            auto_assigned    INTEGER,
            reason           TEXT,
            top_predictions  TEXT,
            created_at       REAL
        );

        -- Tickets that went to manual triage — pending human outcome
        CREATE TABLE IF NOT EXISTS manual_triage (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_number    TEXT NOT NULL UNIQUE,
            sys_id           TEXT NOT NULL,
            short_description TEXT,
            description      TEXT,
            features         TEXT,
            ai_predicted     TEXT,
            ai_confidence    REAL,
            human_assigned   TEXT,        -- filled once human acts
            outcome_checked  INTEGER DEFAULT 0,
            created_at       REAL,
            resolved_at      REAL
        );

        -- Ground-truth training feedback (manual triage outcomes)
        CREATE TABLE IF NOT EXISTS feedback (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_number    TEXT NOT NULL,
            short_description TEXT,
            description      TEXT,
            features         TEXT,
            ai_predicted     TEXT,
            human_assigned   TEXT,
            was_correct      INTEGER,
            confidence       REAL,
            created_at       REAL
        );

        -- Per-group confidence threshold adjustments
        CREATE TABLE IF NOT EXISTS group_thresholds (
            group_name       TEXT PRIMARY KEY,
            threshold        REAL DEFAULT 7.0,
            total_feedback   INTEGER DEFAULT 0,
            correct_feedback INTEGER DEFAULT 0,
            last_updated     REAL
        );
    """

    def __init__(self, db_path: str, model_path: str, base_threshold: float = 7.0):
        self.db_path        = db_path
        self.model_path     = model_path
        self.base_threshold = base_threshold
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # ── DB helpers ────────────────────────────────────────────────────────

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Audit log ─────────────────────────────────────────────────────────

    def log_decision(self, ticket_number, sys_id, predicted_group,
                     confidence, auto_assigned, reason, top_predictions):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO audit_log
                  (ticket_number, sys_id, predicted_group, confidence,
                   auto_assigned, reason, top_predictions, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (ticket_number, sys_id, predicted_group, confidence,
                  int(auto_assigned), reason,
                  json.dumps(top_predictions), time.time()))

    # ── Manual triage storage ─────────────────────────────────────────────

    def store_manual_triage(self, ticket_number: str, sys_id: str,
                             short_description: str, description: str,
                             features: list, ai_predicted: str,
                             ai_confidence: float):
        """
        Called when a ticket is sent to manual triage.
        Saves it so we can later check what the human assigned it to.
        """
        with self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO manual_triage
                  (ticket_number, sys_id, short_description, description,
                   features, ai_predicted, ai_confidence, created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (ticket_number, sys_id, short_description, description,
                  json.dumps(features), ai_predicted, ai_confidence,
                  time.time()))
        logger.info(
            f"Manual triage stored | {ticket_number} | "
            f"AI predicted: {ai_predicted} ({ai_confidence:.1f}/10)"
        )

    # ── Poll ServiceNow for manual triage outcomes ────────────────────────

    def poll_manual_triage_outcomes(self, servicenow_agent) -> int:
        """
        Checks ServiceNow for any manually-triaged tickets that now have
        an assignment_group set by a human.

        For each resolved ticket:
          - Saves (features, human_assigned) as training feedback
          - Updates group threshold based on whether AI was right
          - Triggers retraining if enough new data collected

        Returns number of new outcomes collected.
        """
        with self._conn() as conn:
            pending = conn.execute("""
                SELECT ticket_number, sys_id, short_description, description,
                       features, ai_predicted, ai_confidence
                FROM   manual_triage
                WHERE  outcome_checked = 0
                ORDER  BY created_at ASC
            """).fetchall()

        if not pending:
            return 0

        logger.info(f"Checking {len(pending)} manually-triaged ticket(s) for human outcomes...")
        new_outcomes = 0

        for row in pending:
            ticket_number = row["ticket_number"]
            sys_id        = row["sys_id"]

            # Ask ServiceNow what assignment_group the human set
            human_group = servicenow_agent.get_assignment_group(sys_id)

            if not human_group:
                # Human hasn't assigned it yet — skip for now
                continue

            ai_predicted  = row["ai_predicted"]
            ai_confidence = row["ai_confidence"]
            features      = json.loads(row["features"])
            was_correct   = int(ai_predicted == human_group)

            # Mark outcome collected
            with self._conn() as conn:
                conn.execute("""
                    UPDATE manual_triage
                    SET    human_assigned = ?,
                           outcome_checked = 1,
                           resolved_at = ?
                    WHERE  ticket_number = ?
                """, (human_group, time.time(), ticket_number))

                # Store as training feedback
                conn.execute("""
                    INSERT INTO feedback
                      (ticket_number, short_description, description,
                       features, ai_predicted, human_assigned,
                       was_correct, confidence, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (ticket_number,
                      row["short_description"], row["description"],
                      json.dumps(features), ai_predicted, human_group,
                      was_correct, ai_confidence, time.time()))

            # Update per-group threshold
            self._update_group_threshold(human_group, was_correct)

            new_outcomes += 1
            correct_str = "✅ AI was right" if was_correct else f"❌ AI said '{ai_predicted}', human chose '{human_group}'"
            logger.info(
                f"Outcome captured | {ticket_number} | "
                f"Human assigned → {human_group} | {correct_str}"
            )

        if new_outcomes > 0:
            logger.info(
                f"{new_outcomes} new outcome(s) collected from manual triage."
            )
            # Auto-retrain if 5+ new outcomes collected
            pending_count = self._pending_feedback_count()
            if pending_count >= 5:
                logger.info(
                    f"{pending_count} feedback records accumulated → triggering retraining"
                )
                self.retrain_model()

        return new_outcomes

    # ── Per-group threshold management ────────────────────────────────────

    def _update_group_threshold(self, group_name: str, was_correct: int):
        """
        Adjusts the confidence threshold for a group based on feedback.

        Logic:
          - If AI accuracy for this group ≥ 90% over 10+ samples
            → lower threshold to 6.5 (auto-assign more aggressively)
          - If AI accuracy ≥ 95% over 20+ samples
            → lower threshold to 6.0
          - Otherwise keep at base threshold (7.0)
        """
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO group_thresholds (group_name, threshold, total_feedback, correct_feedback, last_updated)
                VALUES (?, ?, 0, 0, ?)
                ON CONFLICT(group_name) DO NOTHING
            """, (group_name, self.base_threshold, time.time()))

            conn.execute("""
                UPDATE group_thresholds
                SET total_feedback   = total_feedback + 1,
                    correct_feedback = correct_feedback + ?,
                    last_updated     = ?
                WHERE group_name = ?
            """, (was_correct, time.time(), group_name))

            row = conn.execute("""
                SELECT total_feedback, correct_feedback FROM group_thresholds
                WHERE group_name = ?
            """, (group_name,)).fetchone()

        if not row:
            return

        total   = row["total_feedback"]
        correct = row["correct_feedback"]
        if total < 10:
            return   # Not enough data yet

        accuracy = correct / total

        if accuracy >= 0.95 and total >= 20:
            new_threshold = 6.0
        elif accuracy >= 0.90 and total >= 10:
            new_threshold = 6.5
        else:
            new_threshold = self.base_threshold

        with self._conn() as conn:
            conn.execute("""
                UPDATE group_thresholds SET threshold = ? WHERE group_name = ?
            """, (new_threshold, group_name))

        logger.info(
            f"Threshold update | {group_name} | "
            f"accuracy={accuracy:.1%} ({correct}/{total}) → threshold={new_threshold}"
        )

    def get_group_threshold(self, group_name: str) -> float:
        """Returns the current confidence threshold for a specific group."""
        with self._conn() as conn:
            row = conn.execute("""
                SELECT threshold FROM group_thresholds WHERE group_name = ?
            """, (group_name,)).fetchone()
        return row["threshold"] if row else self.base_threshold

    def _pending_feedback_count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    # ── Model retraining ──────────────────────────────────────────────────

    def retrain_model(self) -> bool:
        """
        Retrains the model by merging:
          1. Original CSV training data (data/training_tickets.csv)
          2. All manual triage feedback from SQLite

        This means every ticket a human manually assigned becomes
        permanent training data — the model learns from human decisions.
        """
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from agents.historical_data_agent import HistoricalDataAgent

        agent = HistoricalDataAgent()
        X_all, y_all = [], []

        # ── Source 1: Original CSV ─────────────────────────────────────
        csv_rows = 0
        if os.path.exists(CSV_TRAINING_PATH):
            import csv as csv_module
            augmentations = [
                lambda t: t,
                lambda t: {**t, "priority": "1"},
                lambda t: {**t, "priority": "4"},
                lambda t: {**t, "short_description": "URGENT: " + t["short_description"]},
                lambda t: {**t, "description": t["description"] + " Please escalate."},
                lambda t: {**t, "category": "", "subcategory": ""},
            ]
            with open(CSV_TRAINING_PATH, newline="", encoding="utf-8-sig") as fh:
                for row in csv_module.DictReader(fh):
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
                    for aug in augmentations:
                        t = aug(dict(base))
                        X_all.append(agent.build_features(t))
                        y_all.append(team)
                    csv_rows += 1
            logger.info(f"CSV training data: {csv_rows} source rows → {csv_rows * len(augmentations)} samples")
        else:
            logger.warning(f"CSV training file not found: {CSV_TRAINING_PATH}")

        # ── Source 2: Manual triage feedback ──────────────────────────
        feedback_rows = 0
        with self._conn() as conn:
            feedback = conn.execute("""
                SELECT short_description, description, features, human_assigned
                FROM   feedback
                WHERE  human_assigned IS NOT NULL
            """).fetchall()

        for row in feedback:
            human_group = row["human_assigned"]
            if not human_group:
                continue

            # Prefer re-building features from text (more reliable)
            if row["short_description"]:
                ticket = {
                    "short_description": row["short_description"],
                    "description":       row["description"] or "",
                    "category": "", "subcategory": "", "priority": "3",
                }
                features = agent.build_features(ticket)
            else:
                features = json.loads(row["features"])

            # Add 3x — feedback from human decisions gets extra weight
            for _ in range(3):
                X_all.append(features)
                y_all.append(human_group)
            feedback_rows += 1

        if not X_all:
            logger.error("No training data available.")
            return False

        X = np.array(X_all)
        y = np.array(y_all)

        from collections import Counter
        dist = Counter(y)
        logger.info(
            f"Retraining: {len(X)} total samples | "
            f"{csv_rows} CSV rows + {feedback_rows} human feedback records | "
            f"{len(dist)} teams"
        )

        # Need at least 2 classes
        if len(dist) < 2:
            logger.warning("Need at least 2 classes to train.")
            return False

        try:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=1, random_state=42,
                )),
            ])

            # Cross-validate only if enough data
            if len(X) >= 20 and len(dist) >= 2:
                min_class_count = min(dist.values())
                n_splits = min(5, min_class_count)
                if n_splits >= 2:
                    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    scores = cross_val_score(model, X, y, cv=cv)
                    logger.info(
                        f"Cross-validation: {scores.mean():.1%} (±{scores.std():.1%})"
                    )

            model.fit(X, y)
            joblib.dump(model, self.model_path)
            logger.info(
                f"✅ Model retrained and saved → {self.model_path} | "
                f"Teams: {sorted(dist.keys())}"
            )
            return True

        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)
            return False

    # ── Reporting ─────────────────────────────────────────────────────────

    def get_learning_stats(self) -> dict:
        """Returns a summary of the learning loop state."""
        with self._conn() as conn:
            total_decisions = conn.execute(
                "SELECT COUNT(*) FROM audit_log"
            ).fetchone()[0]

            auto_assigned = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE auto_assigned = 1"
            ).fetchone()[0]

            total_triage = conn.execute(
                "SELECT COUNT(*) FROM manual_triage"
            ).fetchone()[0]

            resolved_triage = conn.execute(
                "SELECT COUNT(*) FROM manual_triage WHERE outcome_checked = 1"
            ).fetchone()[0]

            total_feedback = conn.execute(
                "SELECT COUNT(*) FROM feedback"
            ).fetchone()[0]

            correct_feedback = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE was_correct = 1"
            ).fetchone()[0]

            thresholds = conn.execute(
                "SELECT group_name, threshold, total_feedback, correct_feedback FROM group_thresholds"
            ).fetchall()

        accuracy = (
            round(correct_feedback / total_feedback, 3)
            if total_feedback > 0 else None
        )

        return {
            "total_decisions":    total_decisions,
            "auto_assigned":      auto_assigned,
            "manual_triage":      total_triage,
            "outcomes_collected": resolved_triage,
            "pending_outcomes":   total_triage - resolved_triage,
            "total_feedback":     total_feedback,
            "ai_accuracy_on_triage": accuracy,
            "group_thresholds": {
                r["group_name"]: {
                    "threshold":        r["threshold"],
                    "feedback_count":   r["total_feedback"],
                    "accuracy": round(r["correct_feedback"] / r["total_feedback"], 3)
                              if r["total_feedback"] > 0 else None,
                }
                for r in thresholds
            },
        }

    def accuracy_report(self) -> dict:
        """Legacy method kept for compatibility with scripts/accuracy_report.py"""
        return self.get_learning_stats()
