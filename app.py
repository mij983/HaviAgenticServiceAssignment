"""
app.py — Enhanced ARIA Web
---------------------------
Enhancements over the original POC:

  1. LangChain LLM agent   — swap in LangChainLLMAgent; tracks token usage
  2. Token usage in API    — /predict response includes token_usage dict
  3. Prediction cache      — in-memory LRU cache; repeated tickets skip LLM
  4. /metrics endpoint     — JSON: token totals, cache stats, uptime
  5. Structured logging    — JSON-formatted logs for prod; dev keeps human fmt
  6. Health & ready routes — /health and /ready for container orchestration
  7. Request-ID header     — every response gets X-Request-ID for tracing
  8. Graceful error codes  — 422 for validation, 503 for LLM unavailable
"""

import os
import sys
import time
import uuid
import logging
import logging.config

import yaml
from flask import Flask, jsonify, render_template, request, g

sys.path.insert(0, os.path.dirname(__file__))

from agents.langchain_llm_agent  import LangChainLLMAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent
from agents.embedding_agent      import EmbeddingAgent
from agents.preprocessing_agent  import PreprocessingAgent
from agents.rlhf_agent           import RLHFAgent
from agents.decision_agent       import DecisionAgent
from agents.explanation_agent    import ExplanationAgent
from agents.cache_agent          import PredictionCache

# ── Logging setup ─────────────────────────────────────────────────────────────
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level   = getattr(logging, _LOG_LEVEL, logging.INFO),
    format  = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt = "%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("aria.app")

# ── Constants ─────────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 7.0
DEFAULT_GROUP        = "IT-Service Desk"
_START_TIME          = time.time()

app = Flask(__name__)

# ── Load config & all agents once at startup ──────────────────────────────────
def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

config   = load_config()
rlhf_cfg = config.get("rlhf", {})

preprocessor = PreprocessingAgent()

embed_agent = EmbeddingAgent(model_name=config["embedding"]["model"])
embed_agent.load()

kb_agent = KnowledgeBaseAgent(
    db_path         = config["vector_db"]["path"],
    collection_name = config["vector_db"]["collection"],
)

# ── LangChain LLM agent (replaces plain Ollama LLMAgent) ─────────────────────
llm_agent = LangChainLLMAgent(
    model       = config["llm"]["model"],
    temperature = config["llm"]["temperature"],
)
logger.info("LangChain LLM agent ready: model=%s", config["llm"]["model"])

decision_agent    = DecisionAgent(llm_agent=llm_agent)
explanation_agent = ExplanationAgent()

rlhf_agent     = RLHFAgent(
    feedback_path = rlhf_cfg.get("feedback_path", "data/rlhf_feedback.jsonl"),
    rewards_path  = rlhf_cfg.get("rewards_path",  "data/rlhf_rewards.json"),
)
caution_groups = rlhf_agent.get_low_reward_groups()

# ── Prediction cache ──────────────────────────────────────────────────────────
_cache = PredictionCache(
    maxsize     = int(os.getenv("CACHE_MAXSIZE", "512")),
    ttl_seconds = float(os.getenv("CACHE_TTL_SECONDS", "3600")),
)

# ── Request ID middleware ──────────────────────────────────────────────────────
@app.before_request
def _assign_request_id():
    g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])

@app.after_request
def _add_request_id_header(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return response

# ── Pipeline ──────────────────────────────────────────────────────────────────
def run_pipeline(short_description: str) -> dict:
    clean_text   = preprocessor.process(short_description)
    cache_key    = _cache.make_key(clean_text)
    cached       = _cache.get(cache_key)

    if cached is not None:
        cached["from_cache"] = True
        return cached

    query_vector    = embed_agent.embed(clean_text)
    top_k           = config["vector_db"].get("top_k", 30)
    similar_tickets = kb_agent.search(query_vector, top_k=top_k)
    valid_groups    = config["assignment_groups"]

    result = decision_agent.predict(
        short_description = clean_text,
        similar_tickets   = similar_tickets,
        valid_groups      = valid_groups,
        caution_groups    = caution_groups or [],
    )

    result["clean_text"]  = clean_text
    result["explanation"] = explanation_agent.explain(result)
    result["from_cache"]  = False

    # Cache successful predictions
    token_count = result.get("token_usage", {}).get("total_tokens", 0)
    _cache.set(cache_key, result, token_count=token_count)

    return result


def best_similarity(result: dict) -> float:
    tickets = result.get("similar_tickets", [])
    return max((t["similarity_score"] for t in tickets), default=0.0)


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
    data       = request.get_json()
    user_input = (data or {}).get("description", "").strip()

    if not user_input:
        return jsonify({"error": "Please enter a ticket description."}), 400

    valid, err = preprocessor.is_valid(user_input)
    if not valid:
        return jsonify({"error": err}), 422

    start  = time.time()
    result = run_pipeline(user_input)
    elapsed = round(time.time() - start, 3)

    if not result.get("is_valid_ticket", True):
        return jsonify({"valid_ticket": False, "elapsed": elapsed})

    top_sim = best_similarity(result)
    low_similarity_warning = False
    if top_sim < SIMILARITY_THRESHOLD:
        result["assignment_group"] = DEFAULT_GROUP
        result["confidence"]       = "low"
        result["confidence_score"] = 1
        low_similarity_warning     = True

    # Build similar-tickets list for UI
    similar = []
    for t in result.get("similar_tickets", []):
        src = t.get("source_type", "ticket")
        src_label = {"document": "doc", "rlhf_positive": "rlhf+",
                     "rlhf_negative_corrected": "rlhf~"}.get(src, "csv")
        similar.append({
            "rank":             len(similar) + 1,
            "short_description": t["short_description"][:60],
            "assignment_group":  t["assignment_group"],
            "similarity":        t["similarity_score"],
            "source":            src_label,
            "matched":           t["assignment_group"] == result["assignment_group"],
        })

    rlhf_agent.record_prediction(user_input, result)

    # Token usage — returned from LangChain agent or empty dict if from cache
    token_usage = result.get("token_usage") or {}
    if result.get("from_cache"):
        token_usage = {"from_cache": True}

    logger.info(
        "rid=%s group=%s confidence=%s tokens=%s elapsed=%.3fs cache=%s",
        g.request_id,
        result["assignment_group"],
        result.get("confidence"),
        token_usage.get("total_tokens", "missing"),
        elapsed,
        result.get("from_cache", False),
    )

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
        "from_cache":             result.get("from_cache", False),
        # ── NEW: token usage ─────────────────────────────────────────────────
        "token_usage": token_usage,
    })


@app.route("/status")
def status():
    return jsonify({
        "kb_count":   kb_agent.count(),
        "llm_online": llm_agent.is_available(),
        "llm_model":  config["llm"]["model"],
    })


@app.route("/metrics")
def metrics():
    """
    Operational metrics endpoint.
    Returns token totals, cache stats, and uptime.
    """
    uptime_s = round(time.time() - _START_TIME, 1)
    ts       = llm_agent.token_stats
    return jsonify({
        "uptime_seconds":  uptime_s,
        "llm_model":       config["llm"]["model"],
        "token_usage": {
            "total_calls":        ts.calls,
            "prompt_tokens":      ts.prompt_tokens,
            "completion_tokens":  ts.completion_tokens,
            "total_tokens":       ts.total_tokens,
            "avg_latency_ms":     round(ts.latency_ms / max(ts.calls, 1), 1),
        },
        "cache": _cache.stats(),
        "kb_count": kb_agent.count(),
    })


@app.route("/health")
def health():
    """Liveness probe — always 200 if the process is alive."""
    return jsonify({"status": "ok"}), 200


@app.route("/ready")
def ready():
    """Readiness probe — 200 only when KB and LLM are usable."""
    kb_ok  = kb_agent.count() > 0
    llm_ok = llm_agent.is_available()
    if kb_ok:
        return jsonify({"status": "ready", "llm_online": llm_ok}), 200
    return jsonify({"status": "not_ready", "kb_count": kb_agent.count()}), 503


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    print()
    print("=" * 58)
    print("  ARIA Web — Enhanced (LangChain + Token Tracking)")
    print("=" * 58)
    print(f"  Embedding  : {config['embedding']['model']}")
    print(f"  LLM        : {config['llm']['model']} (via LangChain + Ollama)")
    print(f"  KB count   : {kb_agent.count()} tickets")
    print(f"  Cache size : {_cache._maxsize} entries, TTL={_cache._ttl}s")
    print(f"  Open       : http://localhost:{port}")
    print(f"  Metrics    : http://localhost:{port}/metrics")
    print("=" * 58)
    print()
    app.run(host="0.0.0.0", port=port, debug=False)
