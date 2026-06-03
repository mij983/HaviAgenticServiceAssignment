"""
Token Tracker
-------------
Standalone token usage tracker that can be used independently of the
LangChain agent. Useful for logging, dashboards, and alerting.

Import and attach to any agent:

    from agents.token_tracker import SessionTokenTracker
    tracker = SessionTokenTracker()
    # pass tracker.callback to LangChain callbacks=[...]
    # or call tracker.record(prompt_tokens, completion_tokens, latency_ms)

Provides:
  - Per-call stats
  - Session totals
  - Cost estimation (based on configurable pricing)
  - Rich console report
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    call_id:           int
    prompt_tokens:     int
    completion_tokens: int
    total_tokens:      int
    latency_ms:        float
    timestamp:         float = field(default_factory=time.time)
    model:             str   = ""
    notes:             str   = ""


class SessionTokenTracker:
    """
    Tracks token usage across multiple LLM calls in a session.

    Usage with LangChain callback (already wired inside LangChainLLMAgent):
        # The _TokenTracker callback calls record() automatically.

    Manual usage:
        tracker = SessionTokenTracker()
        tracker.record(prompt_tokens=120, completion_tokens=15, latency_ms=430.0)
        print(tracker.report())
    """

    # Rough cost estimates (USD per 1 000 tokens) — update to match your provider
    COST_PER_1K = {
        "prompt":     0.0,   # Ollama = local, so zero API cost
        "completion": 0.0,
    }

    def __init__(self, model: str = ""):
        self.model    = model
        self._calls:  list[CallRecord] = []
        self._call_id = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        prompt_tokens:     int,
        completion_tokens: int,
        latency_ms:        float,
        notes:             str = "",
    ) -> CallRecord:
        self._call_id += 1
        rec = CallRecord(
            call_id           = self._call_id,
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            total_tokens      = prompt_tokens + completion_tokens,
            latency_ms        = latency_ms,
            model             = self.model,
            notes             = notes,
        )
        self._calls.append(rec)
        logger.debug(
            "Token record #%d: prompt=%d completion=%d total=%d latency=%.1fms",
            rec.call_id, rec.prompt_tokens, rec.completion_tokens,
            rec.total_tokens, rec.latency_ms,
        )
        return rec

    @property
    def total_prompt_tokens(self)     -> int:   return sum(c.prompt_tokens     for c in self._calls)
    @property
    def total_completion_tokens(self) -> int:   return sum(c.completion_tokens for c in self._calls)
    @property
    def total_tokens(self)            -> int:   return sum(c.total_tokens      for c in self._calls)
    @property
    def total_calls(self)             -> int:   return len(self._calls)
    @property
    def total_latency_ms(self)        -> float: return sum(c.latency_ms        for c in self._calls)
    @property
    def avg_latency_ms(self)          -> float:
        return self.total_latency_ms / max(self.total_calls, 1)

    def estimated_cost_usd(self) -> float:
        prompt_cost     = (self.total_prompt_tokens     / 1000) * self.COST_PER_1K["prompt"]
        completion_cost = (self.total_completion_tokens / 1000) * self.COST_PER_1K["completion"]
        return prompt_cost + completion_cost

    def as_dict(self) -> dict:
        return {
            "calls":             self.total_calls,
            "prompt_tokens":     self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens":      self.total_tokens,
            "latency_ms":        round(self.total_latency_ms, 1),
            "avg_latency_ms":    round(self.avg_latency_ms, 1),
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
        }

    def report(self) -> str:
        """Return a formatted multi-line report string."""
        s = self.as_dict()
        lines = [
            "",
            "╔══════════════════════════════════════════╗",
            "║         TOKEN USAGE REPORT               ║",
            "╠══════════════════════════════════════════╣",
            f"║  Model              : {self.model:<20} ║",
            f"║  Total calls        : {s['calls']:<20} ║",
            "╠══════════════════════════════════════════╣",
            f"║  Prompt tokens      : {s['prompt_tokens']:<20} ║",
            f"║  Completion tokens  : {s['completion_tokens']:<20} ║",
            f"║  Total tokens       : {s['total_tokens']:<20} ║",
            "╠══════════════════════════════════════════╣",
            f"║  Total latency      : {s['latency_ms']:>10.1f} ms          ║",
            f"║  Avg latency/call   : {s['avg_latency_ms']:>10.1f} ms          ║",
            "╠══════════════════════════════════════════╣",
            f"║  Est. cost (USD)    : ${s['estimated_cost_usd']:<19.6f} ║",
            "╚══════════════════════════════════════════╝",
            "",
        ]
        return "\n".join(lines)

    def reset(self):
        """Clear all records. Call between test runs."""
        self._calls   = []
        self._call_id = 0
