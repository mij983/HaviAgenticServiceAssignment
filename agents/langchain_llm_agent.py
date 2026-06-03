"""
LangChain LLM Agent
--------------------
Drop-in replacement for llm_agent.py that uses LangChain for:

  1. Token usage tracking  — reads usage_metadata directly from ChatOllama
     response object (the only reliable source for Ollama models).
  2. Structured output    — strips hallucinated punctuation from group names.
  3. Fallback             — weighted-vote fallback if LLM is unavailable.
  4. Tracing hooks        — LangChain callbacks for LangSmith / Langfuse.
  5. Async support        — predict_async() for high-throughput scenarios.

Usage (identical to LLMAgent):

    from agents.langchain_llm_agent import LangChainLLMAgent
    llm_agent = LangChainLLMAgent(model="gemma3:1b", temperature=0.0)
    result = llm_agent.predict(short_description, similar_tickets, valid_groups)

Token stats are in result["token_usage"] and accumulated in llm_agent.token_stats.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Token usage dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    calls:             int   = 0
    latency_ms:        float = 0.0

    def as_dict(self) -> dict:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
            "calls":             self.calls,
            "latency_ms":        round(self.latency_ms, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Latency-only callback (token counts read directly from response instead)
# ─────────────────────────────────────────────────────────────────────────────

class _LatencyTracker(BaseCallbackHandler):
    """Tracks only call latency via LangChain callbacks.
    Token counts are read from usage_metadata on the response object directly
    because ChatOllama does not reliably populate llm_output/generation_info."""

    def __init__(self):
        super().__init__()
        self._t0: float = 0.0
        self.last_latency_ms: float = 0.0

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._t0 = time.monotonic()

    def on_llm_end(self, response, **kwargs):
        self.last_latency_ms = (time.monotonic() - self._t0) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Session-level token accumulator
# ─────────────────────────────────────────────────────────────────────────────

class _SessionTracker:
    def __init__(self):
        self.last    = TokenUsage()
        self.session = TokenUsage()

    def record(self, pt: int, ct: int, latency_ms: float):
        tt = pt + ct
        self.last = TokenUsage(
            prompt_tokens     = pt,
            completion_tokens = ct,
            total_tokens      = tt,
            calls             = 1,
            latency_ms        = latency_ms,
        )
        self.session.prompt_tokens     += pt
        self.session.completion_tokens += ct
        self.session.total_tokens      += tt
        self.session.calls             += 1
        self.session.latency_ms        += latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert IT service desk ticket routing assistant at HAVI.

Your ONLY job is to read a support ticket and output the correct assignment group name.

CRITICAL INSTRUCTIONS:

1. OUTPUT FORMAT
   Output EXACTLY ONE line: the assignment group name, nothing else.
   No explanation. No "I think". No punctuation after the name. Just the name.

2. NOT AN IT TICKET
   Only output NOT_IT_TICKET if the input is completely unrelated to IT —
   for example: random words, greetings, personal matters, or gibberish.

   The following ARE valid IT tickets — do NOT reject them:
   - Spam email, phishing email, suspicious email, junk mail
   - Virus, malware, ransomware, suspicious attachment
   - Email not working, mailbox full, Outlook issues
   - Password reset, account locked, access denied
   - VPN issues, network connectivity, Wi-Fi problems
   - Software installation, hardware fault, printer issues
   - SAP, HaviConnect, or any business application issues
   - Any security incident or cyber threat report

   When in doubt, route the ticket — do NOT reject it.

3. USE THE EVIDENCE
   You will see similar historical tickets with their correct assignment groups.
   The ticket marked [TOP-VOTE] is the statistical best match — trust it unless
   another group has significantly stronger similarity scores.

4. EXACT GROUP NAME
   Copy the group name character-for-character from the VALID ASSIGNMENT GROUPS list.
   Never invent, shorten, or paraphrase a group name.

5. WHEN UNSURE
   Pick the group with the most and highest-similarity matching tickets.
   Default to IT-Service Desk only if NO evidence points to any other group.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main agent
# ─────────────────────────────────────────────────────────────────────────────

class LangChainLLMAgent:
    """
    LangChain-powered replacement for LLMAgent.
    Drop-in: same predict() signature, same return dict shape.

    Token counts are read from response.usage_metadata which ChatOllama
    populates reliably for gemma3:1b and gemma3:4b.
    """

    def __init__(
        self,
        model:       str   = "gemma3:1b",
        temperature: float = 0.0,
        base_url:    str   = "http://localhost:11434",
        num_predict: int   = 60,
    ):
        self.model       = model
        self.temperature = temperature
        self.base_url    = base_url
        self.num_predict = num_predict

        self._latency    = _LatencyTracker()
        self._tracker    = _SessionTracker()

        self._llm = ChatOllama(
            model       = model,
            temperature = temperature,
            base_url    = base_url,
            num_predict = num_predict,
            callbacks   = [self._latency],
        )

        logger.info("LangChainLLMAgent initialised: model=%s temperature=%s", model, temperature)

    # ── Public accessors ──────────────────────────────────────────────────────

    @property
    def token_stats(self) -> TokenUsage:
        return self._tracker.session

    @property
    def last_tokens(self) -> TokenUsage:
        return self._tracker.last

    # ── Sync predict ──────────────────────────────────────────────────────────

    def predict(
        self,
        short_description: str,
        similar_tickets:   list[dict],
        valid_groups:      list[str],
        caution_groups:    list[str] = None,
    ) -> dict:
        prompt_text = self._build_prompt(
            short_description, similar_tickets, valid_groups,
            caution_groups=caution_groups or []
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]

        try:
            response   = self._llm.invoke(messages)
            raw_answer = response.content.strip().strip(".,;:\"'")

            # Read token counts directly from usage_metadata
            meta = getattr(response, "usage_metadata", None) or {}
            pt   = int(meta.get("input_tokens",  0))
            ct   = int(meta.get("output_tokens", 0))
            self._tracker.record(pt, ct, self._latency.last_latency_ms)

            logger.info("Tokens — prompt=%d completion=%d total=%d latency=%.0fms",
                        pt, ct, pt + ct, self._latency.last_latency_ms)

            result = self._process_answer(raw_answer, similar_tickets, valid_groups)
            result["token_usage"] = self._tracker.last.as_dict()
            print(result)
            return result

        except Exception as e:
            logger.error("LangChain LLM error: %s", e)
            result = self._weighted_vote_result(similar_tickets, valid_groups, error=str(e))
            result["token_usage"] = TokenUsage().as_dict()
            return result

    # ── Async predict ─────────────────────────────────────────────────────────

    async def predict_async(
        self,
        short_description: str,
        similar_tickets:   list[dict],
        valid_groups:      list[str],
        caution_groups:    list[str] = None,
    ) -> dict:
        prompt_text = self._build_prompt(
            short_description, similar_tickets, valid_groups,
            caution_groups=caution_groups or []
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ]

        try:
            response   = await self._llm.ainvoke(messages)
            raw_answer = response.content.strip().strip(".,;:\"'")

            meta = getattr(response, "usage_metadata", None) or {}
            pt   = int(meta.get("input_tokens",  0))
            ct   = int(meta.get("output_tokens", 0))
            self._tracker.record(pt, ct, self._latency.last_latency_ms)

            result = self._process_answer(raw_answer, similar_tickets, valid_groups)
            result["token_usage"] = self._tracker.last.as_dict()
            return result

        except Exception as e:
            logger.error("LangChain async LLM error: %s", e)
            result = self._weighted_vote_result(similar_tickets, valid_groups, error=str(e))
            result["token_usage"] = TokenUsage().as_dict()
            return result

    # ── Availability check ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            import ollama
            response  = ollama.list()
            available = [m.model for m in response.models]
            return any(self.model in m for m in available)
        except Exception:
            return False

    # ── Token report ──────────────────────────────────────────────────────────

    def token_report(self) -> str:
        s = self._tracker.session
        lines = [
            "┌─── Token Usage Report ─────────────────┐",
            f"│  Calls              : {s.calls:<6}           │",
            f"│  Prompt tokens      : {s.prompt_tokens:<6}           │",
            f"│  Completion tokens  : {s.completion_tokens:<6}           │",
            f"│  Total tokens       : {s.total_tokens:<6}           │",
            f"│  Total latency      : {s.latency_ms:>8.1f} ms      │",
            f"│  Avg latency/call   : {(s.latency_ms/max(s.calls,1)):>8.1f} ms      │",
            "└────────────────────────────────────────┘",
        ]
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _process_answer(self, raw_answer: str, similar_tickets: list[dict], valid_groups: list[str]) -> dict:
        token_info = self._tracker.last.as_dict()

        if raw_answer.upper() == "NOT_IT_TICKET":
            result = self._invalid_ticket_result(similar_tickets, raw_answer)
            result["token_usage"] = token_info
            return result

        if not self._looks_like_group(raw_answer, valid_groups):
            result = self._weighted_vote_result(similar_tickets, valid_groups, llm_raw=raw_answer)
            result["token_usage"] = token_info
            return result

        predicted = self._validate(raw_answer, valid_groups)
        if predicted is None:
            logger.warning("LLM returned unrecognised group '%s'. Using weighted fallback.", raw_answer)
            result = self._weighted_vote_result(similar_tickets, valid_groups, llm_raw=raw_answer)
            result["token_usage"] = token_info
            return result

        match_count, confidence_score, confidence_label = self._score(predicted, similar_tickets)

        return {
            "is_valid_ticket":  True,
            "assignment_group": predicted,
            "confidence":       confidence_label,
            "confidence_score": confidence_score,
            "match_count":      match_count,
            "top_k":            len(similar_tickets),
            "raw_llm_response": raw_answer,
            "similar_tickets":  similar_tickets,
            "token_usage":      token_info,
        }

    def _looks_like_group(self, raw: str, valid_groups: list[str]) -> bool:
        raw_lower = raw.lower()
        if "not_it_ticket" in raw_lower:
            return True
        return any(group.lower() in raw_lower or raw_lower in group.lower() for group in valid_groups)

    def _validate(self, raw: str, valid_groups: list[str]) -> str | None:
        raw = raw.strip().strip(".,;:\"'")
        if raw in valid_groups:
            return raw
        raw_lower = raw.lower()
        for group in valid_groups:
            if group.lower() == raw_lower:
                return group
        for group in valid_groups:
            if group.lower() in raw_lower or raw_lower in group.lower():
                return group
        return None

    def _score(self, predicted: str, similar_tickets: list[dict]):
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = t.get("similarity_raw", t["similarity_score"])
            weighted_votes[t["assignment_group"]] += weight

        total_weight  = sum(weighted_votes.values()) or 1.0
        winning_share = weighted_votes.get(predicted, 0.0) / total_weight

        if   winning_share >= 0.90: score = 10
        elif winning_share >= 0.80: score = 9
        elif winning_share >= 0.70: score = 8
        elif winning_share >= 0.60: score = 7
        elif winning_share >= 0.50: score = 6
        elif winning_share >= 0.42: score = 5
        elif winning_share >= 0.34: score = 4
        elif winning_share >= 0.25: score = 3
        elif winning_share >= 0.15: score = 2
        else:                       score = 1

        label       = "high" if score >= 7 else "medium" if score >= 4 else "low"
        match_count = sum(1 for t in similar_tickets if t["assignment_group"] == predicted)
        return match_count, score, label

    def _invalid_ticket_result(self, similar_tickets: list[dict], raw: str) -> dict:
        return {
            "is_valid_ticket":  False,
            "assignment_group": None,
            "confidence":       "low",
            "confidence_score": 0,
            "match_count":      0,
            "top_k":            len(similar_tickets),
            "raw_llm_response": raw,
            "similar_tickets":  similar_tickets,
        }

    def _weighted_vote_result(self, similar_tickets: list[dict], valid_groups: list[str],
                               llm_raw: str = "", error: str = "") -> dict:
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = t.get("similarity_raw", t["similarity_score"])
            weighted_votes[t["assignment_group"]] += weight

        predicted = (
            max(weighted_votes, key=weighted_votes.__getitem__)
            if weighted_votes else valid_groups[0]
        )
        match_count, confidence_score, confidence_label = self._score(predicted, similar_tickets)

        return {
            "is_valid_ticket":  True,
            "assignment_group": predicted,
            "confidence":       confidence_label,
            "confidence_score": confidence_score,
            "match_count":      match_count,
            "top_k":            len(similar_tickets),
            "raw_llm_response": llm_raw or ("LLM unavailable: " + error),
            "similar_tickets":  similar_tickets,
            "fallback":         True,
        }

    def _build_prompt(self, short_description: str, similar_tickets: list[dict],
                      valid_groups: list[str], caution_groups: list[str] = None) -> str:
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = t.get("similarity_raw", t["similarity_score"])
            weighted_votes[t["assignment_group"]] += weight
        top_vote = max(weighted_votes, key=weighted_votes.__getitem__) if weighted_votes else ""

        prompt  = "TICKET TO ROUTE:\n" + short_description + "\n\n"

        hist_tickets = [t for t in similar_tickets
                        if t.get("source_type", "ticket") in ("ticket", "rlhf_positive", "rlhf_negative_corrected")]
        doc_chunks   = [t for t in similar_tickets if t.get("source_type") == "document"]

        if doc_chunks:
            prompt += "RELEVANT KB ARTICLES (read these first):\n"
            for i, chunk in enumerate(doc_chunks, 1):
                team_hint = (" => " + chunk["assignment_group"]) if chunk["assignment_group"] else ""
                prompt += (str(i) + ". [" + chunk["short_description"] + "]" + team_hint
                           + " sim=" + str(chunk["similarity_score"]) + "/10\n"
                           + "   " + chunk["description"][:300] + "\n")
            prompt += "\n"

        if hist_tickets:
            prompt += "SIMILAR HISTORICAL TICKETS (cross-verify with KB above):\n"
            for i, ticket in enumerate(hist_tickets, 1):
                src = ticket.get("source_type", "ticket")
                tag = ""
                if src == "rlhf_positive":
                    tag = " [confirmed-correct]"
                elif src == "rlhf_negative_corrected":
                    tag = " [human-corrected]"
                elif ticket["assignment_group"] == top_vote and i == 1:
                    tag = " [TOP-VOTE]"

                prompt += (str(i) + ". [" + ticket["assignment_group"] + "]"
                           + tag + " sim=" + str(ticket["similarity_score"]) + "/10\n"
                           + "   Title: " + ticket["short_description"] + "\n")
                if i <= 3 and ticket.get("description", "").strip():
                    prompt += "   Detail: " + ticket["description"][:200].strip() + "\n"
            prompt += "\n"

        if caution_groups:
            prompt += "CAUTION — FREQUENTLY WRONG GROUPS (from human feedback):\n"
            prompt += "Only use these if similarity is >= 8/10:\n"
            for g in caution_groups:
                prompt += "  - " + g + "\n"
            prompt += "\n"

        if top_vote:
            prompt += "STATISTICAL SUGGESTION (weighted similarity vote): " + top_vote + "\n\n"

        prompt += "VALID ASSIGNMENT GROUPS:\n"
        for group in valid_groups:
            prompt += "- " + group + "\n"
        prompt += "\nOutput ONLY the assignment group name (or NOT_IT_TICKET):"
        return prompt