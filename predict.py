"""
predict.py
-----------
Interactive prompt — type a ticket short description,
get back the predicted assignment group.

RLHF integration:
  - After each prediction, user is asked: "Was this correct? (y/n/skip)"
  - Feedback is saved to data/rlhf_feedback.jsonl
  - RLHF reward stats (data/rlhf_rewards.json) are loaded at startup
    and low-accuracy groups are passed to the LLM as caution hints
    (Stage 3B — prompt bias injection)
  - Run python rlhf_train.py --apply --compute-rewards weekly to push
    feedback into ChromaDB and update reward stats

Usage:
    python predict.py
    python predict.py --once "VPN not connecting from home office"
    python predict.py --no-rlhf     (disable feedback prompts)
"""

import argparse
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from agents.preprocessing_agent  import PreprocessingAgent
from agents.embedding_agent       import EmbeddingAgent
from agents.knowledge_base_agent  import KnowledgeBaseAgent
from agents.llm_agent             import LLMAgent
from agents.rlhf_agent            import RLHFAgent

SIMILARITY_THRESHOLD = 7.0
DEFAULT_GROUP        = "IT-Service Desk"


def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


def print_result(result: dict, short_description: str, elapsed: float = 0.0):
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
    print("  Time taken   : {:.2f}s".format(elapsed))
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
        src = t.get("source_type", "ticket")
        if src == "document":
            src_label = "[doc] "
        elif src == "rlhf_positive":
            src_label = "[rlhf+]"
        elif src == "rlhf_negative_corrected":
            src_label = "[rlhf~]"
        else:
            src_label = "[csv] "
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
                 preprocessor: PreprocessingAgent,
                 caution_groups: list[str] = None) -> dict:
    """Run the full prediction pipeline for one ticket description."""
    clean_text      = preprocessor.process(short_description)
    query_vector    = embed_agent.embed(clean_text)
    top_k           = config["vector_db"]["top_k"]
    similar_tickets = kb_agent.search(query_vector, top_k=top_k)
    valid_groups    = config["assignment_groups"]
    result          = llm_agent.predict(
        short_description = clean_text,
        similar_tickets   = similar_tickets,
        valid_groups      = valid_groups,
        caution_groups    = caution_groups or [],
    )
    return result


def startup_checks(config: dict, kb_agent: KnowledgeBaseAgent,
                   llm_agent: LLMAgent) -> bool:
    ok    = True
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
        print("                     Recommended: ollama pull gemma3:4b")
        print("                     Or try:      ollama pull " + config["llm"]["model"])

    return ok


def process_one(user_input: str, config: dict,
                embed_agent, kb_agent, llm_agent, preprocessor,
                rlhf_agent=None, caution_groups=None, enable_rlhf=True):
    """Run the pipeline and handle output + RLHF feedback collection."""
    start   = time.time()
    result  = run_pipeline(
        user_input, config, embed_agent, kb_agent, llm_agent, preprocessor,
        caution_groups=caution_groups
    )
    elapsed = time.time() - start

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
        print("  Time taken   : {:.2f}s".format(elapsed))
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
        print_result(result, user_input, elapsed)
        print("  ⚠  Similarity below " + str(SIMILARITY_THRESHOLD) + "/10 — auto-assigned to: " + DEFAULT_GROUP)
        print("")
    else:
        print_result(result, user_input, elapsed)

    # ── RLHF feedback collection ──────────────────────────────────────────
    if rlhf_agent and enable_rlhf:
        fb_id = rlhf_agent.record_prediction(user_input, result)
        rlhf_agent.collect_interactive(
            fb_id        = fb_id,
            predicted    = result["assignment_group"],
            valid_groups = config["assignment_groups"],
        )


def main():
    parser = argparse.ArgumentParser(description="ARIA - Ticket Assignment Predictor")
    parser.add_argument("--once",     type=str,  default=None,
                        help="Predict for a single description and exit")
    parser.add_argument("--no-rlhf",  action="store_true",
                        help="Disable RLHF feedback prompts")
    args = parser.parse_args()

    config = load_config()

    rlhf_cfg      = config.get("rlhf", {})
    feedback_path = rlhf_cfg.get("feedback_path", "data/rlhf_feedback.jsonl")
    rewards_path  = rlhf_cfg.get("rewards_path",  "data/rlhf_rewards.json")
    enable_rlhf   = not args.no_rlhf

    print("")
    print("=" * 60)
    print("  ARIA -- Automated Routing and Intelligent Assignment")
    print("=" * 60)
    print("")
    print("  Embedding model      : " + config["embedding"]["model"])
    print("  LLM model            : " + config["llm"]["model"] + " via Ollama")
    print("  Temperature          : " + str(config["llm"]["temperature"]) + " (0.0 = fully deterministic)")
    print("  Top-K results        : " + str(config["vector_db"]["top_k"]))
    print("  Assignment groups    : " + str(len(config["assignment_groups"])))
    print("  Similarity threshold : " + str(SIMILARITY_THRESHOLD) + "/10")
    print("  RLHF feedback        : " + ("enabled" if enable_rlhf else "disabled"))
    print("")

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

    # Load RLHF reward stats for Stage 3B prompt bias
    rlhf_agent     = RLHFAgent(feedback_path=feedback_path, rewards_path=rewards_path)
    caution_groups = rlhf_agent.get_low_reward_groups() if enable_rlhf else []

    if caution_groups:
        print("  RLHF caution groups  : " + str(len(caution_groups)) +
              " group(s) flagged (low accuracy from past feedback)")
        for g in caution_groups:
            print("    - " + g)
        print("")

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
                    rlhf_agent=rlhf_agent if enable_rlhf else None,
                    caution_groups=caution_groups,
                    enable_rlhf=enable_rlhf)
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
                        rlhf_agent=rlhf_agent if enable_rlhf else None,
                        caution_groups=caution_groups,
                        enable_rlhf=enable_rlhf)

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
