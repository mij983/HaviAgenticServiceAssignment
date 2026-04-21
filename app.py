"""
app.py
-------
Flask web interface for ARIA — Automated Routing and Intelligent Assignment.

Wraps the exact same pipeline as predict.py.
No existing agent or config code is modified.

Usage:
    pip install flask
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import time

import yaml
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))

from agents.embedding_agent      import EmbeddingAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.llm_agent            import LLMAgent
from agents.preprocessing_agent  import PreprocessingAgent
from agents.rlhf_agent           import RLHFAgent
from agents.decision_agent       import DecisionAgent
from agents.explanation_agent    import ExplanationAgent

# ── Constants (same as predict.py) ───────────────────────────────────────────
SIMILARITY_THRESHOLD = 7.0
DEFAULT_GROUP        = "IT-Service Desk"

app = Flask(__name__)

# ── Load everything once at startup ──────────────────────────────────────────
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

config       = load_config()
rlhf_cfg     = config.get("rlhf", {})

preprocessor = PreprocessingAgent()

embed_agent  = EmbeddingAgent(model_name=config["embedding"]["model"])
embed_agent.load()

kb_agent     = KnowledgeBaseAgent(
    db_path         = config["vector_db"]["path"],
    collection_name = config["vector_db"]["collection"],
)

llm_agent    = LLMAgent(
    model       = config["llm"]["model"],
    temperature = config["llm"]["temperature"],
)
decision_agent = DecisionAgent(llm_agent=llm_agent)
explanation_agent = ExplanationAgent()

rlhf_agent   = RLHFAgent(
    feedback_path = rlhf_cfg.get("feedback_path", "data/rlhf_feedback.jsonl"),
    rewards_path  = rlhf_cfg.get("rewards_path",  "data/rlhf_rewards.json"),
)
caution_groups = rlhf_agent.get_low_reward_groups()

# ── Pipeline (identical to predict.py run_pipeline) ───────────────────────
def run_pipeline(short_description: str) -> dict:
    clean_text      = preprocessor.process(short_description)
    query_vector    = embed_agent.embed(clean_text)
    top_k           = config["vector_db"].get("top_k", 30)
    similar_tickets = kb_agent.search(query_vector, top_k=top_k)
    valid_groups    = config["assignment_groups"]
    result          = decision_agent.predict(
        short_description = clean_text,
        similar_tickets   = similar_tickets,
        valid_groups      = valid_groups,
        caution_groups    = caution_groups or [],
    )
    result["clean_text"] = clean_text
    result["explanation"] = explanation_agent.explain(result)
    return result


def best_similarity(result: dict) -> float:
    tickets = result.get("similar_tickets", [])
    if not tickets:
        return 0.0
    return max(t["similarity_score"] for t in tickets)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        embedding_model = config["embedding"]["model"],
        llm_model       = config["llm"]["model"],
        kb_count        = kb_agent.count(),
        llm_status      = "Online" if llm_agent.is_available() else "Offline (fallback mode)",
        group_count     = len(config["assignment_groups"]),
    )


@app.route("/predict", methods=["POST"])
def predict():
    data        = request.get_json()
    user_input  = (data or {}).get("description", "").strip()

    if not user_input:
        return jsonify({"error": "Please enter a ticket description."}), 400

    valid, err = preprocessor.is_valid(user_input)
    if not valid:
        return jsonify({"error": err}), 400

    start   = time.time()
    result  = run_pipeline(user_input)
    elapsed = round(time.time() - start, 2)

    # Non-IT ticket
    if not result.get("is_valid_ticket", True):
        return jsonify({
            "valid_ticket": False,
            "elapsed":      elapsed,
        })

    # Low similarity — auto-assign to default
    top_sim = best_similarity(result)
    if top_sim < SIMILARITY_THRESHOLD:
        result["assignment_group"] = DEFAULT_GROUP
        result["confidence"]       = "low"
        result["confidence_score"] = 1
        low_similarity_warning     = True
    else:
        low_similarity_warning = False

    # Build similar tickets list for the UI
    similar = []
    for t in result.get("similar_tickets", []):
        src = t.get("source_type", "ticket")
        if src == "document":
            src_label = "doc"
        elif src == "rlhf_positive":
            src_label = "rlhf+"
        elif src == "rlhf_negative_corrected":
            src_label = "rlhf~"
        else:
            src_label = "csv"
        similar.append({
            "rank":             len(similar) + 1,
            "short_description": t["short_description"][:60],
            "assignment_group":  t["assignment_group"],
            "similarity":        t["similarity_score"],
            "source":            src_label,
            "matched":           t["assignment_group"] == result["assignment_group"],
        })

    # Save RLHF prediction (no interactive prompt in web mode)
    rlhf_agent.record_prediction(user_input, result)

    return jsonify({
        "valid_ticket":           True,
        "assignment_group":       result["assignment_group"],
        "confidence":             result["confidence"].upper(),
        "confidence_score":       result.get("confidence_score", "N/A"),
        "match_count":            result["match_count"],
        "top_k":                  result["top_k"],
        "fallback":               result.get("fallback", False),
        "retrieval_only":         result.get("retrieval_only", False),
        "candidate_groups":       result.get("candidate_groups", []),
        "top_alternative_groups": result.get("top_alternative_groups", []),
        "explanation":            result.get("explanation", ""),
        "low_similarity_warning": low_similarity_warning,
        "elapsed":                elapsed,
        "similar_tickets":        similar,
    })


@app.route("/status")
def status():
    return jsonify({
        "kb_count":   kb_agent.count(),
        "llm_online": llm_agent.is_available(),
        "llm_model":  config["llm"]["model"],
    })


if __name__ == "__main__":
    print("")
    print("=" * 55)
    print("  ARIA Web — starting Flask server")
    print("=" * 55)
    print("  Embedding : " + config["embedding"]["model"])
    print("  LLM       : " + config["llm"]["model"])
    print("  KB count  : " + str(kb_agent.count()) + " tickets")
    print("  Open      : http://localhost:5000")
    print("=" * 55)
    print("")
    app.run(host="0.0.0.0", port=5000, debug=False)