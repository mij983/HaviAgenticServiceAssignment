import argparse
import logging
import os
import sys
import time

import yaml

from agents.ingestion_agent       import TicketIngestionAgent
from agents.knowledge_agent       import KnowledgeAgent
from agents.historical_data_agent import HistoricalDataAgent
from agents.prediction_agent      import AssignmentPredictionAgent
from agents.confidence_engine     import ConfidenceScoringEngine
from agents.decision_agent        import DecisionAgent
from agents.servicenow_agent      import ServiceNowUpdateAgent
from agents.learning_agent        import LearningAgent


def setup_logging(config: dict):
    log_level = getattr(logging, config.get("logging", {}).get("level", "INFO"))
    log_file  = config.get("logging", {}).get("log_file", "data/audit.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
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

    features        = historical_agent.build_features(ticket)
    predicted_group, raw_probability = prediction_agent.predict(features)
    top_predictions = prediction_agent.predict_top_n(features, n=3)

    if predicted_group in deprecated_mapping:
        predicted_group = deprecated_mapping[predicted_group]

    confidence = confidence_engine.calculate(raw_probability, ticket, predicted_group, active_groups)
    group_threshold = learning_agent.get_group_threshold(predicted_group)
    is_active = predicted_group in active_groups
    decision  = decision_agent.decide(ticket, predicted_group, confidence, is_active)

    if is_active and confidence > group_threshold and not decision["auto_assign"]:
        decision["auto_assign"] = True
        decision["reason"] = "Confidence " + str(confidence) + " > earned group threshold " + str(group_threshold) + "."

    learning_agent.log_decision(
        ticket_number=number, sys_id=sys_id,
        predicted_group=predicted_group, confidence=confidence,
        auto_assigned=decision["auto_assign"], reason=decision["reason"],
        top_predictions=top_predictions,
    )

    if decision["auto_assign"]:
        success = servicenow_agent.assign_ticket(sys_id, predicted_group)
        if success:
            log.info("[OK] " + number + " -> '" + predicted_group + "' (score=" + str(confidence) + "/10)")
    else:
        learning_agent.store_manual_triage(
            ticket_number=number, sys_id=sys_id,
            short_description=sd, description=desc,
            features=features, ai_predicted=predicted_group,
            ai_confidence=confidence,
        )
        note = (
            "AI Suggestion: '" + predicted_group + "' "
            "(confidence " + str(round(confidence, 1)) + "/10 -- below threshold " + str(group_threshold) + ").\n"
            "Top predictions: "
            + ", ".join(g + " (" + str(round(p * 100)) + "%)" for g, p in top_predictions)
            + "\n\nPlease assign to the correct group. "
            "Your assignment will train the AI to auto-route similar tickets."
        )
        servicenow_agent.add_work_note(sys_id, note)
        log.info("[TRIAGE] " + number + " -> Manual triage | AI suggested '" + predicted_group + "' (" + str(confidence) + "/10)")


def main():
    parser = argparse.ArgumentParser(description="Agentic ServiceNow Assignment")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--once",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    setup_logging(config)
    log = logging.getLogger("orchestrator")

    print("")
    print("=" * 60)
    print("  Agentic AI -- ServiceNow Ticket Assignment")
    print("  Continuous Learning Loop Active")
    print("=" * 60)
    print("  Instance  : " + config["servicenow"]["instance_url"])
    print("  Threshold : Confidence > " + str(config["confidence_threshold"]) + " = auto-assign")
    print("")

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
    log.info("Active groups: " + str(active_groups))

    poll_interval = config["polling"]["interval_seconds"]
    poll_count    = 0

    while True:
        poll_count += 1
        ts = time.strftime("%H:%M:%S")

        print("")
        print("=" * 60)
        print("  [" + ts + "]  Poll #" + str(poll_count) + " -- Fetching unassigned tickets...")
        print("=" * 60)

        try:
            tickets = ingestion_agent.fetch_unassigned_tickets()

            if not tickets:
                print("  No unassigned tickets found.")
            else:
                print("  Found " + str(len(tickets)) + " unassigned ticket(s).")
                for raw_ticket in tickets:
                    process_ticket(
                        raw_ticket, ingestion_agent, knowledge_agent,
                        historical_agent, prediction_agent, confidence_engine,
                        decision_agent, servicenow_agent, learning_agent,
                        active_groups, deprecated_mapping,
                    )

            print("  Checking manual triage outcomes...")
            new_outcomes = learning_agent.poll_manual_triage_outcomes(servicenow_agent)

            if new_outcomes > 0:
                print("  [LEARNED] " + str(new_outcomes) + " human decision(s) captured. Model retrained.")
                prediction_agent.reload()

            stats = learning_agent.get_learning_stats()
            print("  Decisions: " + str(stats["total_decisions"]) +
                  "  |  Auto-assigned: " + str(stats["auto_assigned"]) +
                  "  |  Triage: " + str(stats["manual_triage"]) +
                  "  |  Feedback: " + str(stats["total_feedback"]))

            knowledge_agent.refresh()
            active_groups, deprecated_mapping = knowledge_agent.load_knowledge()

        except KeyboardInterrupt:
            print("  Shutdown requested. Exiting.")
            break
        except Exception as e:
            log.error("Error in main loop: " + str(e), exc_info=True)

        if args.once:
            break

        print("  Sleeping " + str(poll_interval) + "s before next poll...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
