# ARIA — Enhancement Guide
## POC → Industrial-Grade with LangChain + Token Tracking

---

## What Changed & Why

### 1. LangChain LLM Integration (`agents/langchain_llm_agent.py`)

**Replace** `LLMAgent` with `LangChainLLMAgent` — a drop-in swap:

```python
# Before (POC)
from agents.llm_agent import LLMAgent
llm_agent = LLMAgent(model="gemma3:4b", temperature=0.0)

# After (enhanced)
from agents.langchain_llm_agent import LangChainLLMAgent
llm_agent = LangChainLLMAgent(model="gemma3:4b", temperature=0.0)
```

Everything downstream (`DecisionAgent`, `app.py`, `predict.py`) is **unchanged**.

**What LangChain adds:**

| Feature | POC | Enhanced |
|---|---|---|
| Token tracking | ❌ | ✅ Per-call + session totals |
| Async calls | ❌ | ✅ `predict_async()` |
| LCEL chain | ❌ | ✅ Composable, testable |
| Output parser | ❌ | ✅ Strips hallucinated punctuation |
| Retry logic | ❌ | ✅ Built into LangChain runnable |
| Callback hooks | ❌ | ✅ Plug in LangSmith/Langfuse |

---

### 2. Token Usage Tracking

Every `/predict` response now includes a `token_usage` field:

```json
{
  "assignment_group": "IT-Intune Support",
  "confidence": "HIGH",
  "token_usage": {
    "prompt_tokens": 487,
    "completion_tokens": 12,
    "total_tokens": 499,
    "calls": 1,
    "latency_ms": 623.4
  }
}
```

**Session totals** are available via `/metrics`:

```json
{
  "token_usage": {
    "total_calls": 34,
    "prompt_tokens": 16512,
    "completion_tokens": 408,
    "total_tokens": 16920,
    "avg_latency_ms": 587.2
  }
}
```

**UI**: A green token panel appears below each result showing prompt + completion + total tokens and LLM latency.

---

### 3. Prediction Cache (`agents/cache_agent.py`)

LRU in-memory cache keyed on preprocessed ticket text (SHA-256, 16 chars).

- Repeated/near-identical tickets skip the LLM entirely → **0 tokens, ~5ms**
- Default: 512 entries, 1 hour TTL
- Configure via env vars: `CACHE_MAXSIZE=1024 CACHE_TTL_SECONDS=7200`
- Cache stats visible at `/metrics`

When a result is served from cache, the UI shows:
> ⚡ **Served from cache** — 0 tokens consumed, 0 ms LLM latency.

---

### 4. New API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /metrics` | Token totals, cache stats, uptime |
| `GET /health` | Liveness probe (always 200) |
| `GET /ready` | Readiness probe (503 until KB is loaded) |

---

### 5. Structured Logging

Every request logs a one-line summary with request ID, group, confidence, token count, and latency:

```
2025-06-02T09:41:22 | INFO     | aria.app | rid=a3f1c2 group=IT-Intune Support confidence=high tokens=499 elapsed=0.631s cache=False
```

Set `LOG_LEVEL=DEBUG` for verbose LangChain chain tracing.

---

### 6. Request Tracing

Every response includes `X-Request-ID` header (or echoes the one the client sends). Use this to correlate logs with specific API calls.

---

## Installation

```bash
# 1. Install Python dependencies
pip install langchain langchain-core langchain-ollama

# Or use the updated requirements.txt:
pip install -r requirements.txt

# 2. Ollama must be running with your model
ollama serve
ollama pull gemma3:4b

# 3. Start the app (same as before)
python app.py
```

---

## Files Added / Changed

```
agents/
  langchain_llm_agent.py   ← NEW: LangChain LLM with token tracking
  token_tracker.py         ← NEW: standalone token counter utility
  cache_agent.py           ← NEW: LRU prediction cache
  __init__.py              ← UPDATED: exports new agents

app.py                     ← UPDATED: uses LangChainLLMAgent + cache + metrics
requirements.txt           ← UPDATED: adds langchain-ollama deps
templates/index.html       ← UPDATED: token usage panel in UI
ENHANCEMENTS.md            ← NEW: this file
```

Files **not changed**: all original agents remain intact. `LLMAgent` still exists for backward compatibility.

---

## Token Usage Notes for Ollama

Ollama's token counting depends on the model version:

- **gemma3:4b, gemma3:1b**: token counts returned in `llm_output`
- **Older models (gemma:2b)**: may return 0 for prompt_tokens; completion tokens are usually available
- If counts show 0, upgrade to `gemma3:4b` — it has full token reporting

To verify token reporting is working:
```bash
curl -s http://localhost:5000/metrics | python -m json.tool | grep -A5 token_usage
```

---

## Future Enhancements (Roadmap)

1. **LangSmith tracing** — add `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true` to env; zero code changes needed
2. **Async batch endpoint** — `POST /predict/batch` using `predict_async()` for bulk ticket routing
3. **Token cost alerting** — extend `SessionTokenTracker` to emit an alert when daily token budget is exceeded
4. **Redis cache** — swap `PredictionCache` for a Redis-backed version for multi-instance deployments
5. **Model switching** — `/config/model` endpoint to hot-swap between `gemma3:1b` and `gemma3:4b` at runtime
