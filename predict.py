"""
predict.py
-----------
Interactive prompt — type a ticket short description,
get back the predicted assignment group.

No ServiceNow connection needed.
Everything runs locally.

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


def run_pipeline(short_description: str, config: dict,
                 embed_agent: EmbeddingAgent,
                 kb_agent: KnowledgeBaseAgent,
                 llm_agent: LLMAgent,
                 preprocessor: PreprocessingAgent) -> dict:
    """Run the full pipeline for one ticket description."""

    # Step 1 - Preprocess
    clean_text = preprocessor.process(short_description)

    # Step 2 - Embed
    query_vector = embed_agent.embed(clean_text)

    # Step 3 - Search knowledge base
    top_k          = config["vector_db"]["top_k"]
    similar_tickets = kb_agent.search(query_vector, top_k=top_k)

    # Step 4 - LLM reasoning
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

    # Check knowledge base
    count = kb_agent.count()
    if count == 0:
        print("")
        print("  [ERROR] Knowledge base is empty.")
        print("  Run this first:  python build_knowledge_base.py")
        print("")
        ok = False
    else:
        print("  Knowledge base   : " + str(count) + " tickets loaded")

    # Check LLM
    if llm_agent.is_available():
        print("  LLM              : " + config["llm"]["model"] + " (Ollama running)")
    else:
        print("  LLM              : [WARNING] Ollama not running or model not found")
        print("                     Predictions will use majority vote fallback")
        print("                     To enable LLM: ollama pull " + config["llm"]["model"])

    return ok


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
        result = run_pipeline(args.once, config, embed_agent, kb_agent, llm_agent, preprocessor)
        print_result(result, args.once)
        return

    # Interactive prompt loop
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

            result = run_pipeline(
                user_input, config,
                embed_agent, kb_agent, llm_agent, preprocessor,
            )
            print_result(result, user_input)

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
