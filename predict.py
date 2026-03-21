"""
predict.py
-----------
Interactive prompt — type a ticket short description,
get back the predicted assignment group.

Changes:
  - Non-IT input is caught and rejected with a clear message
  - Similarity threshold: if best similarity < 7.0, user is asked to
    confirm the predicted group (YES / NO). If NO, a numbered list of
    all assignment groups is shown and the user picks a number.

No ServiceNow connection needed. Everything runs locally.

Usage:
    python predict.py
    python predict.py --once "VPN not connecting from home office"
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from agents.preprocessing_agent  import PreprocessingAgent
from agents.embedding_agent       import EmbeddingAgent
from agents.knowledge_base_agent  import KnowledgeBaseAgent
from agents.llm_agent             import LLMAgent

# Similarity score threshold (1-10 scale).
# If the best match among similar tickets is below this, the prediction
# is considered low-confidence and the user is asked to confirm.
SIMILARITY_THRESHOLD = 7.0


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def print_result(result: dict, short_description: str):
    group            = result["assignment_group"]
    confidence       = result["confidence"].upper()
    confidence_score = result.get("confidence_score", "N/A")
    matches          = result["match_count"]
    top_k            = result["top_k"]
    fallback         = result.get("fallback", False)

    print("")
    print("  " + "=" * 56)
    print("  PREDICTION RESULT")
    print("  " + "=" * 56)
    print("")
    print("  Ticket       : " + short_description)
    print("  Assignment   : " + group)
    print("  Confidence   : " + confidence + "  |  Score: " + str(confidence_score) + "/10"
          + "  (" + str(matches) + " of " + str(top_k) + " similar tickets matched)")
    if fallback:
        print("  Note         : LLM unavailable - used weighted similarity vote")
    print("")
    print("  Similar historical tickets used:")
    print("")
    print("  {:<5} {:<50} {:<35} {}".format(
        "Rank", "Short Description", "Assignment Group", "Similarity (1-10)"))
    print("  " + "-" * 5 + " " + "-" * 50 + " " + "-" * 35 + " " + "-" * 17)
    for i, t in enumerate(result["similar_tickets"], 1):
        match_marker = " <--" if t["assignment_group"] == group else ""
        sim_display  = "{:.1f}".format(t["similarity_score"])
        print("  {:<5} {:<50} {:<35} {}{}".format(
            str(i) + ".",
            t["short_description"][:49],
            t["assignment_group"][:34],
            sim_display,
            match_marker,
        ))
    print("")
    print("  " + "=" * 56)
    print("")


def best_similarity(result: dict) -> float:
    """Return the highest similarity score among the retrieved similar tickets."""
    tickets = result.get("similar_tickets", [])
    if not tickets:
        return 0.0
    return max(t["similarity_score"] for t in tickets)


def prompt_manual_group(valid_groups: list[str]) -> str:
    """
    Show a numbered list of all assignment groups and ask the user
    to pick one by number. Loops until a valid number is entered.
    """
    print("")
    print("  Please select the correct assignment group by number:")
    print("")
    for i, group in enumerate(valid_groups, 1):
        print("  {:>3}.  {}".format(i, group))
    print("")

    while True:
        try:
            raw = input("  Enter number (1-" + str(len(valid_groups)) + "): ").strip()
            num = int(raw)
            if 1 <= num <= len(valid_groups):
                chosen = valid_groups[num - 1]
                print("")
                print("  ✔  Manually assigned to: " + chosen)
                print("")
                return chosen
            else:
                print("  [ERROR] Please enter a number between 1 and " + str(len(valid_groups)))
        except ValueError:
            print("  [ERROR] Invalid input. Please enter a number.")


def handle_low_similarity(result: dict, valid_groups: list[str]) -> str:
    """
    When best similarity < SIMILARITY_THRESHOLD:
      1. Show the predicted group with a low-confidence warning
      2. Ask: is this the correct assignment group? (yes / no)
      3. If yes  -> return the predicted group
      4. If no   -> show numbered list, user picks manually
    Returns the final assignment group name.
    """
    predicted = result["assignment_group"]

    print("")
    print("  ⚠  LOW SIMILARITY WARNING")
    print("  The best matching ticket has similarity below " + str(SIMILARITY_THRESHOLD) + "/10.")
    print("  The predicted group may not be accurate.")
    print("")
    print("  Predicted assignment group: " + predicted)
    print("")

    while True:
        answer = input("  Is this the correct assignment group? (yes / no): ").strip().lower()
        if answer in ("yes", "y"):
            print("")
            print("  ✔  Confirmed: " + predicted)
            print("")
            return predicted
        elif answer in ("no", "n"):
            return prompt_manual_group(valid_groups)
        else:
            print("  [ERROR] Please type yes or no.")


def run_pipeline(short_description: str, config: dict,
                 embed_agent: EmbeddingAgent,
                 kb_agent: KnowledgeBaseAgent,
                 llm_agent: LLMAgent,
                 preprocessor: PreprocessingAgent) -> dict:
    """Run the full prediction pipeline for one ticket description."""

    # Step 1 — Preprocess
    clean_text = preprocessor.process(short_description)

    # Step 2 — Embed
    query_vector = embed_agent.embed(clean_text)

    # Step 3 — Search knowledge base
    top_k           = config["vector_db"]["top_k"]
    similar_tickets = kb_agent.search(query_vector, top_k=top_k)

    # Step 4 — LLM reasoning
    valid_groups = config["assignment_groups"]
    result       = llm_agent.predict(
        short_description = clean_text,
        similar_tickets   = similar_tickets,
        valid_groups      = valid_groups,
    )

    return result


def startup_checks(config: dict, kb_agent: KnowledgeBaseAgent,
                   llm_agent: LLMAgent) -> bool:
    """Verify knowledge base and LLM are ready before starting."""
    ok = True

    count = kb_agent.count()
    if count == 0:
        print("")
        print("  [ERROR] Knowledge base is empty.")
        print("  Run this first:  python build_knowledge_base.py --start 0 --end 10000")
        print("")
        ok = False
    else:
        print("  Knowledge base   : " + str(count) + " tickets loaded")

    if llm_agent.is_available():
        print("  LLM              : " + config["llm"]["model"] + " (Ollama running)")
    else:
        print("  LLM              : [WARNING] Ollama not running or model not found")
        print("                     Predictions will use weighted similarity vote fallback")
        print("                     To enable LLM: ollama pull " + config["llm"]["model"])

    return ok


def process_one(user_input: str, config: dict,
                embed_agent, kb_agent, llm_agent, preprocessor):
    """
    Run the pipeline for one input and handle all output logic:
      - Non-IT input rejection
      - Normal high-similarity result
      - Low-similarity confirmation flow
    """
    valid_groups = config["assignment_groups"]

    result = run_pipeline(user_input, config, embed_agent, kb_agent, llm_agent, preprocessor)

    # ── Non-IT input ──────────────────────────────────────────────────────
    if not result.get("is_valid_ticket", True):
        print("")
        print("  " + "=" * 56)
        print("  NOT AN IT TICKET")
        print("  " + "=" * 56)
        print("")
        print("  The input does not appear to be an IT support ticket.")
        print("  Please describe a technical issue, access problem,")
        print("  software/hardware fault, or IT service request.")
        print("")
        print("  Examples:")
        print("    - 'Cannot log in to SAP'")
        print("    - 'HaviConnect website not loading'")
        print("    - 'Laptop not connecting to VPN'")
        print("")
        print("  " + "=" * 56)
        print("")
        return

    # ── Low similarity — ask user to confirm ─────────────────────────────
    top_sim = best_similarity(result)
    if top_sim < SIMILARITY_THRESHOLD:
        print_result(result, user_input)
        final_group = handle_low_similarity(result, valid_groups)
        # Update result with confirmed/manual group for display
        result["assignment_group"] = final_group
        print("  Final assignment: " + final_group)
        print("")
        return

    # ── Normal high-confidence result ─────────────────────────────────────
    print_result(result, user_input)


def main():
    parser = argparse.ArgumentParser(description="ARIA - Ticket Assignment Predictor")
    parser.add_argument("--once", type=str, default=None,
                        help="Predict for a single description and exit")
    args = parser.parse_args()

    config = load_config()

    print("")
    print("=" * 60)
    print("  ARIA -- Automated Routing and Intelligent Assignment")
    print("=" * 60)
    print("")
    print("  Embedding model  : " + config["embedding"]["model"])
    print("  LLM model        : " + config["llm"]["model"] + " via Ollama")
    print("  Assignment groups: " + str(len(config["assignment_groups"])))
    print("  Similarity threshold : " + str(SIMILARITY_THRESHOLD) + "/10")
    print("")

    # Initialise agents
    preprocessor = PreprocessingAgent()

    embed_agent = EmbeddingAgent(model_name=config["embedding"]["model"])
    embed_agent.load()

    kb_agent = KnowledgeBaseAgent(
        db_path         = config["vector_db"]["path"],
        collection_name = config["vector_db"]["collection"],
    )

    llm_agent = LLMAgent(
        model       = config["llm"]["model"],
        temperature = config["llm"]["temperature"],
    )

    # Startup checks
    startup_ok = startup_checks(config, kb_agent, llm_agent)
    if not startup_ok:
        sys.exit(1)

    print("")
    print("  " + "-" * 56)

    # Single prediction mode
    if args.once:
        valid, err = preprocessor.is_valid(args.once)
        if not valid:
            print("  [ERROR] " + err)
            sys.exit(1)
        process_one(args.once, config, embed_agent, kb_agent, llm_agent, preprocessor)
        return

    # Interactive loop
    print("")
    print("  Type a ticket short description to get the assignment group.")
    print("  Type 'exit' or press Ctrl+C to quit.")
    print("")

    while True:
        try:
            print("  " + "-" * 56)
            user_input = input("  Ticket description: ").strip()

            if user_input.lower() in ("exit", "quit", "q"):
                print("")
                print("  Goodbye.")
                print("")
                break

            valid, err = preprocessor.is_valid(user_input)
            if not valid:
                print("  [ERROR] " + err)
                continue

            process_one(user_input, config, embed_agent, kb_agent, llm_agent, preprocessor)

        except KeyboardInterrupt:
            print("")
            print("")
            print("  Goodbye.")
            print("")
            break
        except Exception as e:
            print("  [ERROR] " + str(e))
            continue


if __name__ == "__main__":
    main()
