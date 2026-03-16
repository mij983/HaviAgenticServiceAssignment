"""
run_agent.py
-------------
Runs the AI assignment agent against your real ServiceNow instance.
Includes the continuous learning loop:
  - Manual triage tickets are stored in SQLite
  - When humans assign them, retrain_now.py or --watch captures outcomes
  - Model improves over time from human decisions

Usage:
    python run_agent.py              # process all unassigned tickets once
    python run_agent.py --watch      # keep polling (captures outcomes too)
    python run_agent.py --reset      # clear all assignments for re-testing
    python run_agent.py --status     # show current ticket states only
"""

import argparse, logging, os, sys, time
import requests, yaml

GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
CYAN   = "\033[96m"; BLUE   = "\033[94m"; BOLD   = "\033[1m"
DIM    = "\033[2m";  RESET  = "\033[0m"

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


# ── Visual helpers ────────────────────────────────────────────────────────────

def confidence_bar(score: float) -> str:
    filled = int(score)
    color  = GREEN if score > 7 else YELLOW if score > 5 else RED
    return f"{color}{'█' * filled}{'░' * (10 - filled)}{RESET} {score:.1f}/10"


def banner():
    print(f"""
{BOLD}{BLUE}╔══════════════════════════════════════════════════════════╗
║   Agentic AI — ServiceNow Ticket Assignment              ║
║   Live Run Against Real Developer Instance               ║
╚══════════════════════════════════════════════════════════╝{RESET}
  Instance  : {BOLD}{INSTANCE}{RESET}
  User      : {BOLD}{AUTH[0]}{RESET}
  Threshold : Confidence > {BOLD}{config['confidence_threshold']}{RESET} → auto-assign
""")


def print_ticket_block(ticket, predicted_group, conf_score, decision, top_preds):
    auto   = decision["auto_assign"]
    status = f"{GREEN}{BOLD}✅  AUTO-ASSIGNED{RESET}" if auto else f"{YELLOW}{BOLD}🔶  MANUAL TRIAGE{RESET}"

    print(f"\n  {'─'*56}")
    print(f"  {BOLD}Ticket   :{RESET}  {CYAN}{ticket['number']}{RESET}")
    print(f"  {BOLD}Subject  :{RESET}  {ticket['short_description']}")
    print(f"  {BOLD}Category :{RESET}  {ticket.get('category','—')} / {ticket.get('subcategory','—')}")
    print(f"  {BOLD}Priority :{RESET}  {ticket.get('priority','—')}")
    print()
    print(f"  {BOLD}Predicted Group  :{RESET}  {CYAN}{predicted_group}{RESET}")
    print(f"  {BOLD}Confidence Score :{RESET}  {confidence_bar(conf_score)}")
    print()
    print(f"  {BOLD}Top 3 Predictions:{RESET}")
    for grp, prob in top_preds:
        arrow = f"{CYAN}→{RESET}" if grp == predicted_group else " "
        print(f"    {arrow}  {grp:<40}  {prob*100:5.1f}%")
    print()
    print(f"  {BOLD}Decision :{RESET}  {status}")
    print(f"  {DIM}  {decision['reason']}{RESET}")


def print_summary(results):
    if not results:
        return
    auto   = sum(1 for r in results if r["auto_assigned"])
    triage = len(results) - auto
    avg_c  = sum(r["confidence"] for r in results) / len(results)

    print(f"\n\n  {'═'*56}")
    print(f"  {BOLD}BATCH SUMMARY{RESET}")
    print(f"  {'═'*56}")
    print(f"  Tickets processed : {BOLD}{len(results)}{RESET}")
    print(f"  Auto-assigned     : {GREEN}{BOLD}{auto}{RESET}")
    print(f"  Manual triage     : {YELLOW}{BOLD}{triage}{RESET}")
    print(f"  Avg confidence    : {BOLD}{avg_c:.1f} / 10{RESET}")
    print()
    print(f"  {'Ticket':<14} {'Assigned To':<40} {'Score':>6}  Result")
    print(f"  {'─'*14} {'─'*40} {'─'*6}  {'─'*14}")
    for r in results:
        status = f"{GREEN}Auto-assigned{RESET}" if r["auto_assigned"] else f"{YELLOW}Manual triage{RESET}"
        print(f"  {r['number']:<14} {r['group'][:39]:<40} {r['confidence']:>6.1f}  {status}")
    print()
    print(f"  {DIM}Full audit log → data/audit.log{RESET}")
    if triage > 0:
        print(f"  {YELLOW}{BOLD}Next step:{RESET} Assign the {triage} triaged ticket(s) in ServiceNow,")
        print(f"  {YELLOW}then run:  python retrain_now.py{RESET}")
    print()


# ── Status display ────────────────────────────────────────────────────────────

def show_status():
    print(f"\n{BOLD}Current Ticket States in ServiceNow{RESET}\n")
    r = requests.get(
        f"{INSTANCE}/api/now/table/incident", auth=AUTH, headers=HEADERS,
        params={
            "sysparm_query": "state=1^ORstate=2",
            "sysparm_limit": 50,
            "sysparm_fields": "number,short_description,assignment_group,state,priority",
            "sysparm_display_value": "true",
        }, timeout=15
    )
    tickets = r.json().get("result", [])
    if not tickets:
        print(f"  {YELLOW}No open tickets found.{RESET}\n")
        return

    assigned   = [t for t in tickets if t.get("assignment_group", {}).get("display_value", "").strip()]
    unassigned = [t for t in tickets if not t.get("assignment_group", {}).get("display_value", "").strip()]

    print(f"  {GREEN}Assigned   : {len(assigned)}{RESET}")
    print(f"  {YELLOW}Unassigned : {len(unassigned)}{RESET}\n")
    print(f"  {'Number':<14} {'Short Description':<40} {'Assignment Group':<30} State")
    print(f"  {'─'*14} {'─'*40} {'─'*30} {'─'*10}")
    for t in tickets:
        grp   = t.get("assignment_group", {}).get("display_value", "") or f"{YELLOW}(unassigned){RESET}"
        state = t.get("state", {}).get("display_value", "?") if isinstance(t.get("state"), dict) else t.get("state", "?")
        desc  = t.get("short_description", "")[:38] + ".." if len(t.get("short_description", "")) > 40 else t.get("short_description", "")
        print(f"  {t.get('number', ''):<14} {desc:<40} {grp:<30} {state}")
    print()


# ── Reset tickets for re-testing ─────────────────────────────────────────────

def reset_tickets():
    print(f"\n{BOLD}{YELLOW}Resetting assigned tickets back to unassigned...{RESET}\n")
    r = requests.get(
        f"{INSTANCE}/api/now/table/incident", auth=AUTH, headers=HEADERS,
        params={
            "sysparm_query": "assignment_groupISNOTEMPTY^state=1",
            "sysparm_limit": 50,
            "sysparm_fields": "sys_id,number,short_description",
            "sysparm_display_value": "true",
        }, timeout=15
    )
    tickets = r.json().get("result", [])
    if not tickets:
        print(f"  {YELLOW}No assigned tickets to reset.{RESET}\n")
        return

    for t in tickets:
        patch = requests.patch(
            f"{INSTANCE}/api/now/table/incident/{t['sys_id']}",
            auth=AUTH, headers=HEADERS,
            json={"assignment_group": "", "work_notes": "[AI Agent] Reset for re-testing."},
            timeout=15
        )
        mark = f"{GREEN}✅{RESET}" if patch.status_code == 200 else f"{RED}❌{RESET}"
        print(f"  {mark}  {t.get('number','?')}  cleared")
        time.sleep(0.2)
    print(f"\n  {GREEN}Done.{RESET}\n")


# ── Core processing loop ──────────────────────────────────────────────────────

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

    print(f"  {BOLD}Active Assignment Groups:{RESET}")
    for g in active_groups:
        print(f"    {GREEN}•{RESET}  {g}")
    print()

    poll_count = 0

    while True:
        poll_count += 1
        ts = time.strftime("%H:%M:%S")

        print(f"\n{BOLD}{'═'*58}{RESET}")
        print(f"{BOLD}  [{ts}]  Poll #{poll_count} — Fetching unassigned tickets...{RESET}")
        print(f"{BOLD}{'═'*58}{RESET}")

        tickets = ingestion.fetch_unassigned_tickets(limit=20)

        if not tickets:
            print(f"\n  {GREEN}✅  All tickets are assigned — nothing to process.{RESET}")
        else:
            print(f"\n  Found {BOLD}{len(tickets)}{RESET} unassigned ticket(s).\n")
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

                # Use per-group threshold if lower than base
                group_threshold = learning.get_group_threshold(predicted_group)

                is_active = predicted_group in active_groups
                dec       = decision.decide(ticket, predicted_group, conf_score, is_active)

                # Override if per-group threshold earned (lower than base)
                if is_active and not dec["auto_assign"] and conf_score > group_threshold:
                    dec["auto_assign"] = True
                    dec["reason"] = (
                        f"Confidence {conf_score} > earned group threshold "
                        f"{group_threshold} for '{predicted_group}'."
                    )

                print_ticket_block(ticket, predicted_group, conf_score, dec, top_preds)

                # ── ACT ──────────────────────────────────────────────────
                if dec["auto_assign"]:
                    sn_agent.assign_ticket(sys_id, predicted_group)
                else:
                    # ── STORE FOR LEARNING LOOP ───────────────────────────
                    learning.store_manual_triage(
                        ticket_number     = number,
                        sys_id            = sys_id,
                        short_description = sd,
                        description       = desc,
                        features          = features,
                        ai_predicted      = predicted_group,
                        ai_confidence     = conf_score,
                    )
                    note = (
                        f"🤖 AI Suggestion: '{predicted_group}' "
                        f"(confidence {conf_score:.1f}/10 — below threshold {group_threshold}).\n"
                        f"Top predictions: "
                        + ", ".join(f"{g} ({p*100:.0f}%)" for g, p in top_preds)
                        + "\n\nPlease assign to the correct group. "
                        "Your assignment will train the AI to auto-route similar tickets."
                    )
                    sn_agent.add_work_note(sys_id, note)

                # ── AUDIT LOG ─────────────────────────────────────────────
                learning.log_decision(
                    ticket_number   = number,
                    sys_id          = sys_id,
                    predicted_group = predicted_group,
                    confidence      = conf_score,
                    auto_assigned   = dec["auto_assign"],
                    reason          = dec["reason"],
                    top_predictions = top_preds,
                )

                results.append({
                    "number":       number,
                    "group":        predicted_group,
                    "confidence":   conf_score,
                    "auto_assigned": dec["auto_assign"],
                })

                time.sleep(0.5)

            print_summary(results)

        # ── CHECK MANUAL TRIAGE OUTCOMES (every poll) ─────────────────────
        print(f"  {DIM}Checking manual triage outcomes...{RESET}")
        new_outcomes = learning.poll_manual_triage_outcomes(sn_agent)

        if new_outcomes > 0:
            print(f"\n  {GREEN}{BOLD}🎓 {new_outcomes} human decision(s) learned!{RESET}")
            print(f"  {GREEN}   Model retrained — similar tickets will now auto-assign.{RESET}\n")
            prediction.reload()   # hot-reload the improved model

            # Show which groups got better
            stats = learning.get_learning_stats()
            base  = config["confidence_threshold"]
            improved = {
                g: info for g, info in stats["group_thresholds"].items()
                if info["threshold"] < base
            }
            if improved:
                print(f"  {GREEN}Groups with lowered threshold (AI proved accurate):{RESET}")
                for grp, info in improved.items():
                    print(f"    {grp}: threshold {info['threshold']} (accuracy {info['accuracy']:.1%})")
                print()

        if not watch:
            break

        interval = config["polling"]["interval_seconds"]
        print(f"  {DIM}Sleeping {interval}s... (Ctrl+C to stop){RESET}\n")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n  {YELLOW}Stopped.{RESET}\n")
            break

        knowledge.refresh()
        active_groups, deprecated_mapping = knowledge.load_knowledge()


# ── Entry point ───────────────────────────────────────────────────────────────

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
