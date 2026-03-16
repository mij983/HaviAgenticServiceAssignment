"""
Main Orchestrator — Agentic ServiceNow Ticket Assignment
---------------------------------------------------------
Continuous learning loop:

  POLL → PREDICT → DECIDE
    ├── confidence > threshold  → AUTO ASSIGN  (done)
    └── confidence ≤ threshold  → MANUAL TRIAGE
                                    ↓
                              stored in SQLite
                                    ↓
                         human assigns in ServiceNow
                                    ↓
                         poll_manual_triage_outcomes()
                         captures human decision
                                    ↓
                         stored as training feedback
                         (human decision = ground truth)
                                    ↓
                         auto-retrain model
                                    ↓
                    next similar ticket → AUTO ASSIGN ✅
"""

import argparse
import logging
import os
import sys
import time

import yaml

from agents.ingestion_agent    import TicketIngestionAgent
from agents.knowledge_agent    import KnowledgeAgent
from agents.historical_data_agent import HistoricalDataAgent
from agents.prediction_agent   import AssignmentPredictionAgent
from agents.confidence_engine  import ConfidenceScoringEngine
from agents.decision_agent     import DecisionAgent
from agents.servicenow_agent   import ServiceNowUpdateAgent
from agents.learning_agent     import LearningAgent

GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET = "\033[0m"


def setup_logging(config: dict):
    log_level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    log_file  = config.get("logging", {}).get("log_file", "data/audit.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def process_ticket(
    raw_ticket, ingestion_agent, knowledge_agent, historical_agent,
    prediction_agent, confidence_engine, decision_agent,
    servicenow_agent, learning_agent, active_groups, deprecated_mapping,
):
    log = logging.getLogger("orchestrator")

    ticket = ingestion_agent.normalize_ticket(raw_ticket)
    sys_id = ticket["sys_id"]
    number = ticket["number"]
    sd     = ticket.get("short_description", "")
    desc   = ticket.get("description", "")

    # ── Features + Prediction ────────────────────────────────────────────
    features        = historical_agent.build_features(ticket)
    predicted_group, raw_probability = prediction_agent.predict(features)
    top_predictions = prediction_agent.predict_top_n(features, n=3)

    # Resolve deprecated group
    if predicted_group in deprecated_mapping:
        predicted_group = deprecated_mapping[predicted_group]

    # ── Confidence score ─────────────────────────────────────────────────
    confidence = confidence_engine.calculate(
        raw_probability, ticket, predicted_group, active_groups
    )

    # ── Per-group threshold (lowers as AI proves itself for that group) ──
    group_threshold = learning_agent.get_group_threshold(predicted_group)

    # ── Decision ─────────────────────────────────────────────────────────
    is_active = predicted_group in active_groups
    decision  = decision_agent.decide(ticket, predicted_group, confidence, is_active)

    # Override threshold with per-group value if lower
    if is_active and confidence > group_threshold and not decision["auto_assign"]:
        decision["auto_assign"] = True
        decision["reason"] = (
            f"Confidence {confidence} > group threshold {group_threshold} "
            f"(earned through {learning_agent.get_group_threshold.__doc__ or 'feedback'})."
        )

    # ── Audit log ────────────────────────────────────────────────────────
    learning_agent.log_decision(
        ticket_number=number, sys_id=sys_id,
        predicted_group=predicted_group, confidence=confidence,
        auto_assigned=decision["auto_assign"], reason=decision["reason"],
        top_predictions=top_predictions,
    )

    # ── Act ──────────────────────────────────────────────────────────────
    if decision["auto_assign"]:
        success = servicenow_agent.assign_ticket(sys_id, predicted_group)
        if success:
            log.info(f"✅ {number} → '{predicted_group}' (score={confidence}/10)")
    else:
        # ── MANUAL TRIAGE — store for learning loop ───────────────────
        # Save ticket so we can check back after human assigns it
        learning_agent.store_manual_triage(
            ticket_number=number,
            sys_id=sys_id,
            short_description=sd,
            description=desc,
            features=features,
            ai_predicted=predicted_group,
            ai_confidence=confidence,
        )

        # Work note tells the human what the AI predicted
        note = (
            f"🤖 AI Suggestion: '{predicted_group}' "
            f"(confidence {confidence}/10 — below threshold {group_threshold}).\n"
            f"Top predictions: "
            + ", ".join(f"{g} ({p*100:.0f}%)" for g, p in top_predictions)
            + f"\n\nPlease assign to the correct group. "
            f"Your assignment will be used to improve future auto-routing."
        )
        servicenow_agent.add_work_note(sys_id, note)
        log.info(
            f"🔶 {number} → Manual triage | "
            f"AI suggested '{predicted_group}' ({confidence}/10) | "
            f"Stored for learning loop"
        )


def main():
    parser = argparse.ArgumentParser(description="Agentic ServiceNow Assignment")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--once",  action="store_true", help="One poll cycle then exit")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    setup_logging(config)
    log = logging.getLogger("orchestrator")

    print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════╗
║   Agentic AI — ServiceNow Ticket Assignment              ║
║   Continuous Learning Loop Active                        ║
╚══════════════════════════════════════════════════════════╝{RESET}
  Instance  : {BOLD}{config['servicenow']['instance_url']}{RESET}
  User      : {config['servicenow']['username']}
  Threshold : Confidence > {config['confidence_threshold']} → auto-assign
  Learning  : Manual triage outcomes retrain model automatically
""")

    # ── Initialise agents ─────────────────────────────────────────────────
    ingestion_agent   = TicketIngestionAgent(config)
    knowledge_agent   = KnowledgeAgent(config)
    historical_agent  = HistoricalDataAgent()
    prediction_agent  = AssignmentPredictionAgent(config["model"]["path"])
    confidence_engine = ConfidenceScoringEngine()
    decision_agent    = DecisionAgent(config["confidence_threshold"])
    servicenow_agent  = ServiceNowUpdateAgent(config)
    learning_agent    = LearningAgent(
        config["database"]["feedback_db"],
        config["model"]["path"],
        base_threshold=config["confidence_threshold"],
    )

    active_groups, deprecated_mapping = knowledge_agent.load_knowledge()
    log.info(f"Active groups: {active_groups}")

    poll_interval = config["polling"]["interval_seconds"]
    poll_count    = 0

    # ─────────────────────────────────────────────────────────────────────
    # Polling loop
    # ─────────────────────────────────────────────────────────────────────
    while True:
        poll_count += 1
        ts = time.strftime("%H:%M:%S")

        print(f"\n{'═'*58}")
        print(f"  [{ts}]  Poll #{poll_count} — Fetching unassigned tickets...")
        print(f"{'═'*58}\n")

        try:
            # ── Step 1: Process new unassigned tickets ────────────────
            tickets = ingestion_agent.fetch_unassigned_tickets()

            if not tickets:
                print(f"  No unassigned tickets found.")
            else:
                print(f"  Found {len(tickets)} unassigned ticket(s).\n")
                for raw_ticket in tickets:
                    process_ticket(
                        raw_ticket, ingestion_agent, knowledge_agent,
                        historical_agent, prediction_agent, confidence_engine,
                        decision_agent, servicenow_agent, learning_agent,
                        active_groups, deprecated_mapping,
                    )

            # ── Step 2: Check manual triage outcomes ──────────────────
            # Ask ServiceNow: did any human assign a triaged ticket?
            print(f"\n  Checking manual triage outcomes...")
            new_outcomes = learning_agent.poll_manual_triage_outcomes(servicenow_agent)

            if new_outcomes > 0:
                print(f"\n  {GREEN}🎓 {new_outcomes} new outcome(s) learned from human decisions{RESET}")
                print(f"  {GREEN}   Model retrained — future similar tickets will auto-assign{RESET}")
                # Reload model after retraining
                prediction_agent.reload()
            else:
                print(f"  No new outcomes yet (humans haven't assigned triage tickets).")

            # ── Step 3: Print learning stats ──────────────────────────
            stats = learning_agent.get_learning_stats()
            print(f"\n  {'─'*56}")
            print(f"  Learning Loop Stats:")
            print(f"    Total decisions    : {stats['total_decisions']}")
            print(f"    Auto-assigned      : {stats['auto_assigned']}")
            print(f"    Manual triage      : {stats['manual_triage']}")
            print(f"    Outcomes collected : {stats['outcomes_collected']}")
            print(f"    Pending outcomes   : {stats['pending_outcomes']}")
            print(f"    Feedback records   : {stats['total_feedback']}")
            if stats["ai_accuracy_on_triage"] is not None:
                print(f"    AI accuracy (triage): {stats['ai_accuracy_on_triage']:.1%}")
            if stats["group_thresholds"]:
                print(f"\n  Per-group thresholds (auto-lowered as AI proves accuracy):")
                for grp, info in sorted(stats["group_thresholds"].items()):
                    acc_str = f"{info['accuracy']:.1%}" if info["accuracy"] is not None else "—"
                    base_flag = " ← lowered!" if info["threshold"] < config["confidence_threshold"] else ""
                    print(f"    {grp:<45} threshold={info['threshold']}  acc={acc_str}  n={info['feedback_count']}{base_flag}")
            print(f"  {'─'*56}")

            # ── Refresh knowledge ─────────────────────────────────────
            knowledge_agent.refresh()
            active_groups, deprecated_mapping = knowledge_agent.load_knowledge()

        except KeyboardInterrupt:
            print(f"\n  Shutdown requested. Exiting.")
            break
        except Exception as e:
            log.error(f"Error in main loop: {e}", exc_info=True)

        if args.once:
            break

        print(f"\n  Sleeping {poll_interval}s before next poll...\n")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
