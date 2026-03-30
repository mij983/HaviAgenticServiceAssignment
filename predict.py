"""
predict.py
-----------
Interactive prompt - type a ticket short description,
get back the predicted assignment group.

Now supports:
  - Azure OpenAI as LLM provider (set in config/config.yaml + .env)
  - Ollama local LLM (original behaviour, still supported)
  - Feedback loop: after each prediction, user is asked if it was correct
    Feedback is saved to data/feedback.jsonl for later KB improvement

Usage:
    python predict.py
    python predict.py --once "VPN not connecting from home office"
    python predict.py --no-feedback      (skip feedback prompts)
"""

import argparse
import os
import sys

import yaml

# Load .env file if present (needed for Azure OpenAI credentials)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — env vars can be set directly in the shell

sys.path.insert(0, os.path.dirname(__file__))

from agents.preprocessing_agent  import PreprocessingAgent
from agents.embedding_agent       import EmbeddingAgent
from agents.knowledge_base_agent  import KnowledgeBaseAgent
from agents.llm_agent             import LLMAgent
from agents.feedback_agent        import FeedbackAgent

SIMILARITY_THRESHOLD = 7.0
DEFAULT_GROUP        = "IT-Service Desk"


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
    print("  Similar historical tickets and KB articles used:")
    print("")
    print("  {:<5} {:<48} {:<35} {} {}".format(
        "Rank", "Short Description", "Assignment Group", "Similarity", "Source"))
    print("  " + "-" * 5 + " " + "-" * 48 + " " + "-" * 35 + " " + "-" * 10 + " " + "-" * 8)
    for i, t in enumerate(result["similar_tickets"], 1):
        match_marker = " <--" if t["assignment_group"] == group else ""
        sim_display  = "{:.1f}".format(t["similarity_score"])
        src_label    = "[doc]"      if t.get("source_type") == "document" \
                       else "[fb]"  if t.get("source_type") in ("feedback", "feedback_correction") \
                       else "[csv]"
        print("  {:<5} {:<48} {:<35} {:<10} {}{}".format(
            str(i) + ".",
            t["short_description"][:47],
            t["assignment_group"][:34],
            sim_display,
            src_label,
            match_marker,
        ))
    print("")
    print("  " + "=" * 56)
    print("")


def best_similarity(result: dict) -> float:
    tickets = result.get("similar_tickets", [])
    if not tickets:
        return 0.0
    return max(t["similarity_score"] for t in tickets)


def run_pipeline(short_description: str, config: dict,
                 embed_agent: EmbeddingAgent,
                 kb_agent: KnowledgeBaseAgent,
                 llm_agent: LLMAgent,
                 preprocessor: PreprocessingAgent) -> dict:
    clean_text      = preprocessor.process(short_description)
    query_vector    = embed_agent.embed(clean_text)
    top_k           = config["vector_db"]["top_k"]
    similar_tickets = kb_agent.search(query_vector, top_k=top_k)
    valid_groups    = config["assignment_groups"]
    result          = llm_agent.predict(
        short_description = clean_text,
        similar_tickets   = similar_tickets,
        valid_groups      = valid_groups,
    )
    return result


def startup_checks(config: dict, kb_agent: KnowledgeBaseAgent,
                   llm_agent: LLMAgent) -> bool:
    ok    = True
    count = kb_agent.count()

    if count == 0:
        print("")
        print("  [ERROR] Knowledge base is empty.")
        print("  Run:  python build_knowledge_base.py --start 0 --end 10000")
        print("")
        ok = False
    else:
        print("  Knowledge base   : " + str(count) + " entries loaded")

    provider = config["llm"].get("provider", "ollama")
    if llm_agent.is_available():
        print("  LLM              : " + llm_agent.provider_label() + "  [OK]")
    else:
        if provider == "azure_openai":
            print("  LLM              : [ERROR] Azure OpenAI not reachable")
            print("                     Check .env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,")
            print("                                 AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION")
        else:
            print("  LLM              : [WARNING] Ollama not running or model not found")
            print("                     Predictions will use weighted similarity vote fallback")
            print("                     To enable LLM: ollama pull " + config["llm"]["model"])

    return ok


def process_one(user_input: str, config: dict,
                embed_agent, kb_agent, llm_agent, preprocessor,
                fb_agent=None, enable_feedback=True):
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

    # ── Low similarity — auto-assign to default group ─────────────────────
    top_sim = best_similarity(result)
    if top_sim < SIMILARITY_THRESHOLD:
        result["assignment_group"] = DEFAULT_GROUP
        result["confidence"]       = "low"
        result["confidence_score"] = 1
        print_result(result, user_input)
        print("  ⚠  Similarity below " + str(SIMILARITY_THRESHOLD) + "/10 — auto-assigned to: " + DEFAULT_GROUP)
        print("")

    else:
        # ── Normal result ─────────────────────────────────────────────────
        print_result(result, user_input)

    # ── Feedback loop ─────────────────────────────────────────────────────
    if fb_agent and enable_feedback:
        fb_id = fb_agent.record_prediction(user_input, result)
        fb_agent.collect_interactive(
            fb_id        = fb_id,
            predicted    = result["assignment_group"],
            valid_groups = config["assignment_groups"],
        )


def main():
    parser = argparse.ArgumentParser(description="ARIA - Ticket Assignment Predictor")
    parser.add_argument("--once",        type=str,  default=None,
                        help="Predict for a single description and exit")
    parser.add_argument("--no-feedback", action="store_true",
                        help="Disable the feedback prompt after each prediction")
    args = parser.parse_args()

    config = load_config()

    llm_cfg  = config["llm"]
    provider = llm_cfg.get("provider", "ollama")

    print("")
    print("=" * 60)
    print("  ARIA -- Automated Routing and Intelligent Assignment")
    print("=" * 60)
    print("")
    print("  Embedding model  : " + config["embedding"]["model"])
    print("  LLM provider     : " + provider)
    print("  Assignment groups: " + str(len(config["assignment_groups"])))
    print("  Similarity threshold : " + str(SIMILARITY_THRESHOLD) + "/10")
    feedback_enabled = not args.no_feedback
    print("  Feedback loop    : " + ("enabled" if feedback_enabled else "disabled"))
    print("")

    preprocessor = PreprocessingAgent()

    embed_agent = EmbeddingAgent(model_name=config["embedding"]["model"])
    embed_agent.load()

    kb_agent = KnowledgeBaseAgent(
        db_path         = config["vector_db"]["path"],
        collection_name = config["vector_db"]["collection"],
    )

    llm_agent = LLMAgent(
        provider    = provider,
        model       = llm_cfg.get("model", "gemma:2b"),
        temperature = llm_cfg.get("temperature", 0.1),
    )

    # Feedback agent
    feedback_path = config.get("feedback", {}).get("path", "data/feedback.jsonl")
    fb_agent      = FeedbackAgent(feedback_path=feedback_path) if feedback_enabled else None

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
        process_one(args.once, config, embed_agent, kb_agent, llm_agent, preprocessor,
                    fb_agent=fb_agent, enable_feedback=feedback_enabled)
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

            process_one(user_input, config, embed_agent, kb_agent, llm_agent, preprocessor,
                        fb_agent=fb_agent, enable_feedback=feedback_enabled)

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
