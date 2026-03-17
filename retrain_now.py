import sys
import os
import yaml
import sqlite3
import logging

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.WARNING)

from agents.servicenow_agent import ServiceNowUpdateAgent
from agents.learning_agent   import LearningAgent

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

DB_PATH = config["database"]["feedback_db"]

sn_agent = ServiceNowUpdateAgent(config)
la = LearningAgent(
    DB_PATH,
    config["model"]["path"],
    base_threshold=config["confidence_threshold"],
)

print("")
print("=" * 60)
print("  Learning Loop -- Collect Outcomes + Retrain")
print("=" * 60)
print("")

if not os.path.exists(DB_PATH):
    print("  ERROR: Database not found: " + DB_PATH)
    print("  Run the agent first:  python run_agent.py")
    print("")
    sys.exit(0)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
if "manual_triage" not in tables:
    print("  No manual triage data found yet.")
    print("  The manual_triage table does not exist.")
    print("  Make sure you are using the latest run_agent.py and run it first.")
    print("")
    conn.close()
    sys.exit(0)

all_triage = conn.execute("""
    SELECT ticket_number, short_description, ai_predicted, ai_confidence,
           human_assigned, outcome_checked, created_at
    FROM   manual_triage
    ORDER  BY created_at DESC
""").fetchall()

if not all_triage:
    print("  No manual triage tickets stored yet.")
    print("  Run the agent and let it process some low-confidence tickets first.")
    print("")
    conn.close()
    sys.exit(0)

pending   = [r for r in all_triage if not r["outcome_checked"]]
collected = [r for r in all_triage if r["outcome_checked"]]

print("  Manual Triage Status:")
print("    Total   : " + str(len(all_triage)))
print("    Pending : " + str(len(pending)))
print("    Learned : " + str(len(collected)))
print("")

print("  {:<14}  {:<40}  {:>5}  {}".format("Ticket", "AI Predicted", "Score", "Human Assigned / Status"))
print("  " + "-" * 14 + "  " + "-" * 40 + "  " + "-" * 5 + "  " + "-" * 40)
for r in all_triage:
    if r["outcome_checked"]:
        match = "[OK]" if r["ai_predicted"] == r["human_assigned"] else "[WRONG]"
        human_str = str(r["human_assigned"]) + " " + match
    else:
        human_str = "(not assigned yet in ServiceNow)"
    print("  {:<14}  {:<40}  {:>5.1f}  {}".format(
        r["ticket_number"], r["ai_predicted"][:39], r["ai_confidence"], human_str))

conn.close()

print("")
print("  " + "-" * 58)
if not pending:
    print("  All triage tickets already have outcomes collected.")
    new_outcomes = 0
else:
    print("  Checking ServiceNow for " + str(len(pending)) + " pending ticket(s)...")
    print("")
    new_outcomes = la.poll_manual_triage_outcomes(sn_agent)

    if new_outcomes == 0:
        print("  No new outcomes yet.")
        print("  The " + str(len(pending)) + " pending ticket(s) have not been assigned by a human yet.")
        print("")
        print("  Steps:")
        conn2 = sqlite3.connect(DB_PATH)
        conn2.row_factory = sqlite3.Row
        still_pending = conn2.execute(
            "SELECT ticket_number FROM manual_triage WHERE outcome_checked = 0"
        ).fetchall()
        conn2.close()
        for r in still_pending:
            print("    1. Open " + r["ticket_number"] + " in ServiceNow")
            print("       Set the Assignment Group field to the correct team")
        print("    2. Run this script again: python retrain_now.py")
        print("")
        sys.exit(0)
    else:
        print("  [OK] " + str(new_outcomes) + " new outcome(s) collected from ServiceNow.")
        print("")

        conn3 = sqlite3.connect(DB_PATH)
        conn3.row_factory = sqlite3.Row
        new_fb = conn3.execute("""
            SELECT ticket_number, ai_predicted, human_assigned, was_correct
            FROM   feedback
            ORDER  BY created_at DESC
            LIMIT  ?
        """, (new_outcomes,)).fetchall()
        conn3.close()

        print("  {:<14}  {:<40}  {:<40}  {}".format("Ticket", "AI Predicted", "Human Assigned", "Correct?"))
        print("  " + "-" * 14 + "  " + "-" * 40 + "  " + "-" * 40 + "  " + "-" * 8)
        for r in new_fb:
            mark = "[YES]" if r["was_correct"] else "[NO]"
            print("  {:<14}  {:<40}  {:<40}  {}".format(
                r["ticket_number"], r["ai_predicted"][:39],
                r["human_assigned"][:39], mark))

feedback_count = la._pending_feedback_count()
print("")
print("  " + "-" * 58)
print("  Total feedback records available: " + str(feedback_count))

if feedback_count == 0:
    print("  No feedback to train on yet.")
    print("")
    sys.exit(0)

print("")
print("  Retraining model with original CSV + " + str(feedback_count) + " human feedback record(s)...")
success = la.retrain_model()

if success:
    print("")
    print("  [OK] Model retrained successfully.")
    print("  Saved -> models/assignment_model.pkl")
    print("")
    print("  Run the agent again to see improvements:")
    print("    python run_agent.py")
else:
    print("")
    print("  [ERROR] Retraining failed. Check logs.")
    print("")
    sys.exit(1)

stats = la.get_learning_stats()
base  = config["confidence_threshold"]

print("")
print("  " + "-" * 58)
print("  Learning Loop Summary:")
print("")
print("    Total decisions made    : " + str(stats["total_decisions"]))
print("    Auto-assigned by AI     : " + str(stats["auto_assigned"]))
print("    Sent to manual triage   : " + str(stats["manual_triage"]))
print("    Outcomes captured       : " + str(stats["outcomes_collected"]))
print("    Still pending           : " + str(stats["pending_outcomes"]))
print("    Total feedback records  : " + str(stats["total_feedback"]))
if stats["ai_accuracy_on_triage"] is not None:
    print("    AI accuracy on triage   : " + str(round(stats["ai_accuracy_on_triage"] * 100, 1)) + "%")

if stats["group_thresholds"]:
    print("")
    print("  Per-group Confidence Thresholds (base = " + str(base) + "):")
    print("")
    print("  {:<45}  {:>9}  {:>8}  {:>7}".format("Group", "Threshold", "Accuracy", "Samples"))
    print("  " + "-" * 45 + "  " + "-" * 9 + "  " + "-" * 8 + "  " + "-" * 7)
    for grp, info in sorted(stats["group_thresholds"].items()):
        acc = str(round(info["accuracy"] * 100, 1)) + "%" if info["accuracy"] is not None else "--"
        lowered = "  <- lowered!" if info["threshold"] < base else ""
        print("  {:<45}  {:>9.1f}  {:>8}  {:>7}{}".format(
            grp, info["threshold"], acc, info["feedback_count"], lowered))

print("")
