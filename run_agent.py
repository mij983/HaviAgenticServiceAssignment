import argparse
import logging
import os
import sys
import time

import requests
import yaml

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.dirname(__file__))

from agents.ingestion_agent       import TicketIngestionAgent
from agents.knowledge_agent       import KnowledgeAgent
from agents.historical_data_agent import HistoricalDataAgent
from agents.prediction_agent      import AssignmentPredictionAgent
from agents.confidence_engine     import ConfidenceScoringEngine
from agents.decision_agent        import DecisionAgent
from agents.servicenow_agent      import ServiceNowUpdateAgent
from agents.learning_agent        import LearningAgent

with open("config/config.yaml") as f:
    config = yaml.safe_load(f)

INSTANCE = config["servicenow"]["instance_url"].rstrip("/")
AUTH     = (config["servicenow"]["username"], config["servicenow"]["password"])
HEADERS  = {"Content-Type": "application/json", "Accept": "application/json"}


def confidence_bar(score: float) -> str:
    filled = int(score)
    return "[" + "#" * filled + "-" * (10 - filled) + "] " + str(round(score, 1)) + "/10"


def banner():
    print("")
    print("=" * 60)
    print("  Agentic AI -- ServiceNow Ticket Assignment")
    print("  Live Run Against Real Developer Instance")
    print("=" * 60)
    print("  Instance  : " + INSTANCE)
    print("  User      : " + AUTH[0])
    print("  Threshold : Confidence > " + str(config["confidence_threshold"]) + " = auto-assign")
    print("")


def print_ticket_block(ticket, predicted_group, conf_score, decision, top_preds):
    auto = decision["auto_assign"]
    status = "AUTO-ASSIGNED" if auto else "MANUAL TRIAGE"

    print("")
    print("  " + "-" * 56)
    print("  Ticket   : " + ticket["number"])
    print("  Subject  : " + ticket["short_description"])
    print("  Category : " + ticket.get("category", "-") + " / " + ticket.get("subcategory", "-"))
    print("  Priority : " + ticket.get("priority", "-"))
    print("")
    print("  Predicted Group  : " + predicted_group)
    print("  Confidence Score : " + confidence_bar(conf_score))
    print("")
    print("  Top 3 Predictions:")
    for grp, prob in top_preds:
        marker = "  ->" if grp == predicted_group else "    "
        print("  " + marker + "  " + grp + "  (" + str(round(prob * 100, 1)) + "%)")
    print("")
    print("  Decision : [" + status + "]")
    print("    " + decision["reason"])


def print_summary(results):
    if not results:
        return
    auto   = sum(1 for r in results if r["auto_assigned"])
    triage = len(results) - auto
    avg_c  = sum(r["confidence"] for r in results) / len(results)

    print("")
    print("  " + "=" * 56)
    print("  BATCH SUMMARY")
    print("  " + "=" * 56)
    print("  Tickets processed : " + str(len(results)))
    print("  Auto-assigned     : " + str(auto))
    print("  Manual triage     : " + str(triage))
    print("  Avg confidence    : " + str(round(avg_c, 1)) + " / 10")
    print("")
    print("  {:<14} {:<40} {:>6}  {}".format("Ticket", "Assigned To", "Score", "Result"))
    print("  " + "-" * 14 + " " + "-" * 40 + " " + "-" * 6 + "  " + "-" * 14)
    for r in results:
        status = "Auto-assigned" if r["auto_assigned"] else "Manual triage"
        print("  {:<14} {:<40} {:>6.1f}  {}".format(
            r["number"], r["group"][:39], r["confidence"], status))
    print("")
    print("  Audit log -> data/audit.log")
    if triage > 0:
        print("  Next step: Assign the " + str(triage) + " triaged ticket(s) in ServiceNow,")
        print("  then run:  python retrain_now.py")
    print("")


def show_status():
    print("")
    print("Current Ticket States in ServiceNow")
    print("")
    r = requests.get(
        INSTANCE + "/api/now/table/incident",
        auth=AUTH, headers=HEADERS,
        params={
            "sysparm_query": "state=1^ORstate=2",
            "sysparm_limit": 50,
            "sysparm_fields": "number,short_description,assignment_group,state,priority",
            "sysparm_display_value": "true",
        }, timeout=15
    )
    tickets = r.json().get("result", [])
    if not tickets:
        print("  No open tickets found.")
        return

    assigned   = [t for t in tickets if t.get("assignment_group", {}).get("display_value", "").strip()]
    unassigned = [t for t in tickets if not t.get("assignment_group", {}).get("display_value", "").strip()]

    print("  Assigned   : " + str(len(assigned)))
    print("  Unassigned : " + str(len(unassigned)))
    print("")
    print("  {:<14} {:<40} {:<30} {}".format("Number", "Short Description", "Assignment Group", "State"))
    print("  " + "-" * 14 + " " + "-" * 40 + " " + "-" * 30 + " " + "-" * 10)
    for t in tickets:
        grp   = t.get("assignment_group", {}).get("display_value", "") or "(unassigned)"
        state = t.get("state", {}).get("display_value", "?") if isinstance(t.get("state"), dict) else t.get("state", "?")
        desc  = t.get("short_description", "")[:38] + ".." if len(t.get("short_description", "")) > 40 else t.get("short_description", "")
        print("  {:<14} {:<40} {:<30} {}".format(t.get("number", ""), desc, grp, state))
    print("")


def reset_tickets():
    print("")
    print("Resetting assigned tickets back to unassigned...")
    print("")
    r = requests.get(
        INSTANCE + "/api/now/table/incident",
        auth=AUTH, headers=HEADERS,
        params={
            "sysparm_query": "assignment_groupISNOTEMPTY^state=1",
            "sysparm_limit": 50,
            "sysparm_fields": "sys_id,number,short_description",
            "sysparm_display_value": "true",
        }, timeout=15
    )
    tickets = r.json().get("result", [])
    if not tickets:
        print("  No assigned tickets to reset.")
        return

    for t in tickets:
        patch = requests.patch(
            INSTANCE + "/api/now/table/incident/" + t["sys_id"],
            auth=AUTH, headers=HEADERS,
            json={"assignment_group": "", "work_notes": "[AI Agent] Reset for re-testing."},
            timeout=15
        )
        mark = "OK" if patch.status_code == 200 else "FAILED"
        print("  [" + mark + "]  " + t.get("number", "?") + "  cleared")
        time.sleep(0.2)
    print("")
    print("  Done. Tickets are unassigned and ready for re-processing.")
    print("")


def run(watch: bool = False):
    banner()

    ingestion  = TicketIngestionAgent(config)
    knowledge  = KnowledgeAgent(config)
    historical = HistoricalDataAgent()
    prediction = AssignmentPredictionAgent(config["model"]["path"])
    confidence = ConfidenceScoringEngine()
    decision   = DecisionAgent(config["confidence_threshold"])
    sn_agent   = ServiceNowUpdateAgent(config)
    learning   = LearningAgent(
        config["database"]["feedback_db"],
        config["model"]["path"],
        base_threshold=config["confidence_threshold"],
    )

    active_groups, deprecated_mapping = knowledge.load_knowledge()

    print("  Active Assignment Groups:")
    for g in active_groups:
        print("    + " + g)
    print("")

    poll_count = 0

    while True:
        poll_count += 1
        ts = time.strftime("%H:%M:%S")

        print("")
        print("=" * 60)
        print("  [" + ts + "]  Poll #" + str(poll_count) + " -- Fetching unassigned tickets...")
        print("=" * 60)

        tickets = ingestion.fetch_unassigned_tickets(limit=50)

        if not tickets:
            print("")
            print("  All tickets are assigned -- nothing to process.")
        else:
            print("")
            print("  Found " + str(len(tickets)) + " unassigned ticket(s).")
            results = []

            for raw_ticket in tickets:
                ticket   = ingestion.normalize_ticket(raw_ticket)
                sys_id   = ticket["sys_id"]
                number   = ticket["number"]
                sd       = ticket.get("short_description", "")
                desc     = ticket.get("description", "")

                features        = historical.build_features(ticket)
                predicted_group, raw_prob = prediction.predict(features)
                top_preds       = prediction.predict_top_n(features, n=3)

                if predicted_group in deprecated_mapping:
                    predicted_group = deprecated_mapping[predicted_group]

                conf_score = confidence.calculate(raw_prob, ticket, predicted_group, active_groups)

                group_threshold = learning.get_group_threshold(predicted_group)
                is_active = predicted_group in active_groups
                dec       = decision.decide(ticket, predicted_group, conf_score, is_active)

                if is_active and not dec["auto_assign"] and conf_score > group_threshold:
                    dec["auto_assign"] = True
                    dec["reason"] = (
                        "Confidence " + str(conf_score) + " > earned group threshold "
                        + str(group_threshold) + " for '" + predicted_group + "'."
                    )

                print_ticket_block(ticket, predicted_group, conf_score, dec, top_preds)

                if dec["auto_assign"]:
                    sn_agent.assign_ticket(sys_id, predicted_group)
                else:
                    learning.store_manual_triage(
                        ticket_number=number,
                        sys_id=sys_id,
                        short_description=sd,
                        description=desc,
                        features=features,
                        ai_predicted=predicted_group,
                        ai_confidence=conf_score,
                    )
                    note = (
                        "AI Suggestion: '" + predicted_group + "' "
                        "(confidence " + str(round(conf_score, 1)) + "/10 -- below threshold "
                        + str(group_threshold) + ").\n"
                        "Top predictions: "
                        + ", ".join(g + " (" + str(round(p * 100)) + "%)" for g, p in top_preds)
                        + "\n\nPlease assign to the correct group. "
                        "Your assignment will train the AI to auto-route similar tickets."
                    )
                    sn_agent.add_work_note(sys_id, note)

                learning.log_decision(
                    ticket_number=number,
                    sys_id=sys_id,
                    predicted_group=predicted_group,
                    confidence=conf_score,
                    auto_assigned=dec["auto_assign"],
                    reason=dec["reason"],
                    top_predictions=top_preds,
                )

                results.append({
                    "number":        number,
                    "group":         predicted_group,
                    "confidence":    conf_score,
                    "auto_assigned": dec["auto_assign"],
                })

                time.sleep(0.5)

            print_summary(results)

        print("  Checking manual triage outcomes...")
        new_outcomes = learning.poll_manual_triage_outcomes(sn_agent)

        if new_outcomes > 0:
            print("")
            print("  [LEARNED] " + str(new_outcomes) + " human decision(s) captured.")
            print("  Model retrained -- similar tickets will now auto-assign.")
            prediction.reload()
            print("")

        if not watch:
            break

        interval = config["polling"]["interval_seconds"]
        print("  Sleeping " + str(interval) + "s... (Ctrl+C to stop)")
        print("")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("")
            print("  Stopped.")
            print("")
            break

        knowledge.refresh()
        active_groups, deprecated_mapping = knowledge.load_knowledge()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI ticket assignment agent")
    parser.add_argument("--watch",  action="store_true", help="Keep polling every N seconds")
    parser.add_argument("--reset",  action="store_true", help="Clear all assignment groups for re-testing")
    parser.add_argument("--status", action="store_true", help="Show current ticket states only")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.reset:
        reset_tickets()
    else:
        run(watch=args.watch)
