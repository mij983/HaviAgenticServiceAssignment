"""
retrain_now.py
---------------
Collects outcomes from manually-triaged tickets and retrains the model.

What it does:
  1. Reads all manual triage tickets from SQLite
  2. Checks ServiceNow for each one: has a human assigned it?
  3. Saves human decision as training feedback
  4. Retrains model with CSV data + all human feedback
  5. Shows how the model improved

Run after a human has assigned triaged tickets in ServiceNow:
    python retrain_now.py
"""

import sys, os, yaml, sqlite3, logging
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(level=logging.WARNING)   # quiet for clean output

from agents.servicenow_agent import ServiceNowUpdateAgent
from agents.learning_agent   import LearningAgent
from agents.prediction_agent import AssignmentPredictionAgent

GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"
CYAN="\033[96m";  BOLD="\033[1m";    DIM="\033[2m"; RESET="\033[0m"

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

DB_PATH = config["database"]["feedback_db"]

sn_agent = ServiceNowUpdateAgent(config)
la = LearningAgent(
    DB_PATH,
    config["model"]["path"],
    base_threshold=config["confidence_threshold"],
)

print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║   Learning Loop — Collect Outcomes + Retrain             ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")

# ── Check DB exists and has triage table ─────────────────────────────────────
if not os.path.exists(DB_PATH):
    print(f"  {RED}❌  Database not found: {DB_PATH}{RESET}")
    print(f"  Run the agent first:  python run_agent.py\n")
    sys.exit(0)

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

# Check table exists
tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
if "manual_triage" not in tables:
    print(f"  {YELLOW}⚠  No manual triage data found yet.{RESET}")
    print(f"  The 'manual_triage' table doesn't exist — this means run_agent.py")
    print(f"  hasn't stored any triage tickets yet.")
    print(f"\n  {BOLD}Make sure you're using the latest run_agent.py and run it first.{RESET}\n")
    conn.close()
    sys.exit(0)

# ── Step 1: Show all manual triage tickets ────────────────────────────────────
all_triage = conn.execute("""
    SELECT ticket_number, short_description, ai_predicted, ai_confidence,
           human_assigned, outcome_checked, created_at
    FROM   manual_triage
    ORDER  BY created_at DESC
""").fetchall()

if not all_triage:
    print(f"  {YELLOW}No manual triage tickets stored yet.{RESET}")
    print(f"  Run the agent and let it process some low-confidence tickets first.\n")
    conn.close()
    sys.exit(0)

pending   = [r for r in all_triage if not r["outcome_checked"]]
collected = [r for r in all_triage if r["outcome_checked"]]

print(f"  {'─'*58}")
print(f"  Manual Triage Status:  {len(all_triage)} total  |  "
      f"{YELLOW}{len(pending)} pending{RESET}  |  {GREEN}{len(collected)} learned{RESET}")
print(f"  {'─'*58}\n")

print(f"  {'Ticket':<14}  {'AI Predicted':<40}  {'Score':>5}  {'Human Assigned / Status'}")
print(f"  {'─'*14}  {'─'*40}  {'─'*5}  {'─'*40}")
for r in all_triage:
    if r["outcome_checked"]:
        match = "✅" if r["ai_predicted"] == r["human_assigned"] else "❌"
        human_str = f"{GREEN}{r['human_assigned']}{RESET} {match}"
    else:
        human_str = f"{YELLOW}⏳ Not assigned yet in ServiceNow{RESET}"
    print(f"  {r['ticket_number']:<14}  {r['ai_predicted']:<40}  {r['ai_confidence']:>5.1f}  {human_str}")

conn.close()

# ── Step 2: Poll ServiceNow for new human assignments ─────────────────────────
print(f"\n  {'─'*58}")
if not pending:
    print(f"  {GREEN}✅  All triage tickets already have outcomes collected.{RESET}")
    new_outcomes = 0
else:
    print(f"  Checking ServiceNow for {len(pending)} pending ticket(s)...\n")
    new_outcomes = la.poll_manual_triage_outcomes(sn_agent)

    if new_outcomes == 0:
        print(f"  {YELLOW}⏳  No new outcomes yet.{RESET}")
        print(f"  The {len(pending)} pending ticket(s) haven't been assigned by a human yet.")
        print(f"\n  {BOLD}How to assign in ServiceNow:{RESET}")
        for r in pending:
            print(f"    • Open {r['ticket_number']} → set Assignment Group")
        print(f"\n  Then run this script again: python retrain_now.py\n")
        sys.exit(0)
    else:
        print(f"\n  {GREEN}{BOLD}✅  {new_outcomes} new outcome(s) collected from ServiceNow!{RESET}\n")

        # Show what was learned
        conn2 = sqlite3.connect(DB_PATH)
        conn2.row_factory = sqlite3.Row
        new_fb = conn2.execute("""
            SELECT ticket_number, ai_predicted, human_assigned, was_correct
            FROM   feedback
            ORDER  BY created_at DESC
            LIMIT  ?
        """, (new_outcomes,)).fetchall()
        conn2.close()

        print(f"  {'Ticket':<14}  {'AI Predicted':<40}  {'Human Assigned':<40}  Correct?")
        print(f"  {'─'*14}  {'─'*40}  {'─'*40}  {'─'*8}")
        for r in new_fb:
            mark = f"{GREEN}✅{RESET}" if r["was_correct"] else f"{RED}❌{RESET}"
            print(f"  {r['ticket_number']:<14}  {r['ai_predicted']:<40}  {r['human_assigned']:<40}  {mark}")

# ── Step 3: Retrain ───────────────────────────────────────────────────────────
feedback_count = la._pending_feedback_count()
print(f"\n  {'─'*58}")
print(f"  Total feedback records for retraining: {BOLD}{feedback_count}{RESET}")

if feedback_count == 0:
    print(f"  {YELLOW}No feedback to train on yet.{RESET}\n")
    sys.exit(0)

print(f"\n  Retraining model with original CSV + {feedback_count} human feedback record(s)...")
success = la.retrain_model()

if success:
    print(f"\n  {GREEN}{BOLD}✅  Model retrained successfully!{RESET}")
    print(f"  {GREEN}   Saved → models/assignment_model.pkl{RESET}")
    print(f"\n  {BOLD}What happens next:{RESET}")
    print(f"  When you run run_agent.py and a similar ticket comes in,")
    print(f"  it will be AUTO-ASSIGNED instead of going to manual triage.")
else:
    print(f"\n  {RED}❌  Retraining failed. Check logs.{RESET}\n")
    sys.exit(1)

# ── Step 4: Learning stats ────────────────────────────────────────────────────
stats = learning_stats = la.get_learning_stats()
base  = config["confidence_threshold"]

print(f"\n  {'─'*58}")
print(f"  {BOLD}Learning Loop Summary:{RESET}\n")
print(f"    Total decisions made    : {stats['total_decisions']}")
print(f"    Auto-assigned by AI     : {GREEN}{stats['auto_assigned']}{RESET}")
print(f"    Sent to manual triage   : {YELLOW}{stats['manual_triage']}{RESET}")
print(f"    Outcomes captured       : {GREEN}{stats['outcomes_collected']}{RESET}")
print(f"    Still pending           : {YELLOW}{stats['pending_outcomes']}{RESET}")
print(f"    Total feedback records  : {stats['total_feedback']}")
if stats["ai_accuracy_on_triage"] is not None:
    pct = stats["ai_accuracy_on_triage"]
    col = GREEN if pct >= 0.8 else YELLOW if pct >= 0.6 else RED
    print(f"    AI accuracy on triage   : {col}{pct:.1%}{RESET}")

if stats["group_thresholds"]:
    print(f"\n  {BOLD}Per-group Confidence Thresholds{RESET} (base = {base}):\n")
    print(f"  {'Group':<45}  {'Threshold':>9}  {'Accuracy':>8}  {'Samples':>7}")
    print(f"  {'─'*45}  {'─'*9}  {'─'*8}  {'─'*7}")
    for grp, info in sorted(stats["group_thresholds"].items()):
        acc = f"{info['accuracy']:.1%}" if info["accuracy"] is not None else "—"
        lowered = f"  {GREEN}↓ threshold lowered!{RESET}" if info["threshold"] < base else ""
        print(f"  {grp:<45}  {info['threshold']:>9.1f}  {acc:>8}  {info['feedback_count']:>7}{lowered}")

print(f"\n  {'─'*58}")
print(f"  {BOLD}Run the agent again to see improvements:{RESET}")
print(f"    python run_agent.py\n")
