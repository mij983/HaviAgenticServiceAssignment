"""
LLM Agent
----------
Uses a locally running LLM via Ollama to reason over the retrieved
similar tickets and predict the correct assignment group.

The LLM receives:
  - The user's ticket short description
  - The top-K most similar historical tickets with their assignment groups
  - The full list of valid assignment groups

It must return ONLY the assignment group name from the valid list.

Model options (set in config/config.yaml):
  mistral     - Best accuracy, ~4GB RAM
  gemma:2b    - Lighter, ~2GB RAM, slightly less accurate
  llama3.2    - Good balance, ~2GB RAM

Install Ollama: https://ollama.com
Then run: ollama pull gemma:2b

Changes from original:
  - System prompt updated with 6 anti-hallucination rules
  - Non-IT / irrelevant input detection added via RULE 1 (NOT_IT_TICKET signal)
  - Secondary focused validation call (_is_non_it_input) as double-check
  - If input is not a valid IT ticket, returns is_valid_ticket=False
    so predict.py can handle it gracefully before showing the table

Confidence Score (1-10):
  The score is computed from weighted similarity votes (each similar ticket
  contributes its similarity score as a vote weight for its group).

  score = (weighted_share_of_winning_group) mapped onto 1-10:
    >= 0.90  -> 10    (near-unanimous, all top matches agree)
    >= 0.80  -> 9     (very strong majority)
    >= 0.70  -> 8     (strong majority)
    >= 0.60  -> 7     (good majority)
    >= 0.50  -> 6     (moderate majority)
    >= 0.42  -> 5     (weak majority)
    >= 0.34  -> 4     (marginal majority)
    >= 0.25  -> 3     (split evidence)
    >= 0.15  -> 2     (very split)
    <  0.15  -> 1     (near-random)

  HIGH   : score 7-10
  MEDIUM : score 4-6
  LOW    : score 1-3

System prompt — anti-hallucination rules summary:
  RULE 1 : If not an IT ticket -> respond NOT_IT_TICKET
  RULE 2 : Respond with ONLY the group name, no extra text
  RULE 3 : Group name must be copied exactly from the valid list
  RULE 4 : Base decisions only on evidence from similar tickets
  RULE 5 : When split, pick the group with highest combined similarity
  RULE 6 : Never hallucinate — never output anything except a valid group or NOT_IT_TICKET
"""

import logging
from collections import defaultdict

import ollama

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — anti-hallucination rules
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an IT service desk routing assistant.

Your job is to read a support ticket and decide which IT team should handle it.

You will be given:
1. The ticket short description submitted by the user
2. Similar historical tickets that were successfully resolved, each showing which team handled it
3. A list of all valid assignment groups

STRICT RULES — follow every rule without exception:

RULE 1 — VALID IT TICKET CHECK:
  Before doing anything else, decide: is this a genuine IT support ticket?
  A valid IT ticket describes a technical problem, system issue, access issue,
  software/hardware fault, network problem, or any IT-related service request.

  If the input is NOT a valid IT ticket — for example: personal matters,
  medical issues, greetings, random words, nonsense sentences, or anything
  clearly unrelated to IT support — respond with exactly:
    NOT_IT_TICKET
  and nothing else.

RULE 2 — RESPOND ONLY WITH THE GROUP NAME:
  If it IS a valid IT ticket, respond with ONLY the assignment group name.
  No explanation. No punctuation. No bullet points. No extra text whatsoever.

RULE 3 — ONLY USE GROUPS FROM THE VALID LIST:
  The assignment group MUST be exactly one from the valid list provided.
  Do NOT invent or create group names that are not in the list.
  Do NOT combine or shorten group names.
  Copy the group name exactly — character for character.

RULE 4 — BASE DECISIONS ON EVIDENCE ONLY:
  Only predict a group if the similar historical tickets support that decision.
  Weight higher-similarity tickets more heavily.
  Do NOT assume a group based on general knowledge if the evidence does not support it.

RULE 5 — WHEN EVIDENCE IS SPLIT:
  If the similar tickets point to multiple groups with similar weight,
  pick the single group that has the highest combined similarity score.
  Still respond with ONLY that one group name.

RULE 6 — NO HALLUCINATION:
  Never make up ticket details, group names, or reasoning.
  Never output anything except a valid group name from the list OR NOT_IT_TICKET.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Focused non-IT validation prompt (used as secondary double-check)
# ─────────────────────────────────────────────────────────────────────────────

VALIDATION_PROMPT = """You are an IT ticket validator.

Decide if the following text is a genuine IT support ticket.

A genuine IT ticket describes: a technical problem, system error, login or access
issue, software or hardware fault, network issue, or any IT-related service request.

NOT a genuine IT ticket: personal problems, medical issues, greetings, random words,
nonsense text, questions about non-IT topics, or anything unrelated to IT support.

Respond with exactly one word — YES or NO:
  YES  — if it is a genuine IT support ticket
  NO   — if it is not a genuine IT support ticket

Text:
"""


class LLMAgent:

    def __init__(self, model: str = "gemma:2b", temperature: float = 0.1):
        self.model       = model
        self.temperature = temperature

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        short_description: str,
        similar_tickets:   list[dict],
        valid_groups:      list[str],
    ) -> dict:
        """
        Ask the LLM to predict the assignment group.

        Returns:
            {
                "is_valid_ticket":  True / False,
                "assignment_group": "IT-SC-EPAM-SAP Workflow" or None,
                "confidence":       "high" / "medium" / "low",
                "confidence_score": 8,          # 1-10  (0 if not a valid ticket)
                "match_count":      3,
                "top_k":            5,
                "raw_llm_response": "...",
                "similar_tickets":  [...],
                "fallback":         True / False,
            }
        """
        prompt = self._build_prompt(short_description, similar_tickets, valid_groups)

        try:
            response = ollama.chat(
                model    = self.model,
                options  = {"temperature": self.temperature},
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )

            raw_answer = response["message"]["content"].strip()

            # ── Non-IT ticket signal from LLM (RULE 1) ───────────────────
            if raw_answer.upper() == "NOT_IT_TICKET":
                return self._invalid_ticket_result(similar_tickets, raw_answer)

            # ── Response doesn't match any group — secondary check ────────
            if not self._looks_like_group(raw_answer, valid_groups):
                if self._is_non_it_input(short_description):
                    return self._invalid_ticket_result(similar_tickets, raw_answer)

            predicted = self._validate(raw_answer, valid_groups)

            # If LLM returns something unrecognised, fall back to weighted vote
            if predicted is None:
                logger.warning("LLM returned unrecognised group '%s'. Using weighted fallback.", raw_answer)
                return self._weighted_vote_result(similar_tickets, valid_groups,
                                                  llm_raw=raw_answer)

            match_count, confidence_score, confidence_label = self._score(
                predicted, similar_tickets
            )

            return {
                "is_valid_ticket":  True,
                "assignment_group": predicted,
                "confidence":       confidence_label,
                "confidence_score": confidence_score,
                "match_count":      match_count,
                "top_k":            len(similar_tickets),
                "raw_llm_response": raw_answer,
                "similar_tickets":  similar_tickets,
            }

        except Exception as e:
            logger.error("LLM error: %s", e)
            # Fallback: use weighted similarity vote from similar tickets
            return self._weighted_vote_result(similar_tickets, valid_groups,
                                              error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Non-IT input detection
    # ─────────────────────────────────────────────────────────────────────────

    def _is_non_it_input(self, text: str) -> bool:
        """
        Secondary focused yes/no LLM call to verify whether the text is
        a genuine IT support ticket.
        Returns True if it is NOT an IT ticket.
        If this call fails for any reason, returns False (do not block).
        """
        try:
            response = ollama.chat(
                model    = self.model,
                options  = {"temperature": 0.0},
                messages = [
                    {"role": "user", "content": VALIDATION_PROMPT + text},
                ],
            )
            answer = response["message"]["content"].strip().upper()
            return answer.startswith("NO")
        except Exception:
            return False  # If check fails, assume valid — do not block

    def _looks_like_group(self, raw: str, valid_groups: list[str]) -> bool:
        """
        Quick check: does the raw LLM response resemble any valid group name?
        Used to decide whether a secondary validation call is needed.
        """
        raw_lower = raw.lower()
        return any(
            group.lower() in raw_lower or raw_lower in group.lower()
            for group in valid_groups
        )

    def _invalid_ticket_result(self, similar_tickets: list[dict], raw: str) -> dict:
        """Return a result dict signalling that the input is not an IT ticket."""
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

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence scoring (1-10)
    # ─────────────────────────────────────────────────────────────────────────

    def _score(self, predicted: str, similar_tickets: list[dict]):
        """
        Compute a 1-10 confidence score using weighted similarity votes.

        Uses similarity_raw (0-1 cosine similarity) for mathematically
        correct weighting. similarity_score is the 1-10 display value.
        """
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = t.get("similarity_raw", t["similarity_score"])
            weighted_votes[t["assignment_group"]] += weight

        total_weight  = sum(weighted_votes.values()) or 1.0
        winning_share = weighted_votes.get(predicted, 0.0) / total_weight

        # Map share to 1-10
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

        label = (
            "high"   if score >= 7 else
            "medium" if score >= 4 else
            "low"
        )

        match_count = sum(
            1 for t in similar_tickets
            if t["assignment_group"] == predicted
        )

        return match_count, score, label

    # ─────────────────────────────────────────────────────────────────────────
    # Weighted-vote fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _weighted_vote_result(
        self,
        similar_tickets: list[dict],
        valid_groups:    list[str],
        llm_raw:  str = "",
        error:    str = "",
    ) -> dict:
        """
        When the LLM is unavailable or returns garbage, use weighted
        similarity voting: each similar ticket votes for its group with
        weight = similarity_raw (raw cosine 0-1). The group with the
        highest total weight wins.
        """
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = t.get("similarity_raw", t["similarity_score"])
            weighted_votes[t["assignment_group"]] += weight

        # Debug: show weighted vote breakdown in console
        if weighted_votes:
            logger.debug("Weighted vote breakdown:")
            total = sum(weighted_votes.values())
            for grp, w in sorted(weighted_votes.items(), key=lambda x: -x[1]):
                logger.debug("  %-40s %.3f  (%.1f%%)", grp, w, 100 * w / total)

        if not weighted_votes:
            predicted = valid_groups[0]
        else:
            predicted = max(weighted_votes, key=weighted_votes.__getitem__)

        match_count, confidence_score, confidence_label = self._score(
            predicted, similar_tickets
        )

        result = {
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
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        short_description: str,
        similar_tickets:   list[dict],
        valid_groups:      list[str],
    ) -> str:
        prompt  = "NEW TICKET:\n"
        prompt += short_description + "\n\n"

        # Split results into historical tickets vs KB document chunks
        hist_tickets = [t for t in similar_tickets if t.get("source_type", "ticket") == "ticket"]
        doc_chunks   = [t for t in similar_tickets if t.get("source_type") == "document"]

        if hist_tickets:
            prompt += "SIMILAR HISTORICAL TICKETS (ranked by similarity, highest first):\n"
            for i, ticket in enumerate(hist_tickets, 1):
                prompt += (
                    str(i) + ". [" + ticket["assignment_group"] + "] "
                    + ticket["short_description"]
                    + " (similarity: " + str(ticket["similarity_score"]) + "/10)\n"
                )
            prompt += "\n"

        if doc_chunks:
            prompt += "RELEVANT KB ARTICLES / DOCUMENTS:\n"
            for i, chunk in enumerate(doc_chunks, 1):
                team_hint = (" -> " + chunk["assignment_group"]) if chunk["assignment_group"] else ""
                prompt += (
                    str(i) + ". [" + chunk["short_description"] + "]" + team_hint + "\n"
                    + "   " + chunk["description"][:300] + "\n"
                    + "   (similarity: " + str(chunk["similarity_score"]) + "/10)\n"
                )
            prompt += "\n"

        prompt += "VALID ASSIGNMENT GROUPS:\n"
        for group in valid_groups:
            prompt += "- " + group + "\n"

        prompt += "\nRespond with ONLY the assignment group name (or NOT_IT_TICKET if not an IT issue):"
        return prompt

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self, raw: str, valid_groups: list[str]) -> str | None:
        """
        Check if the LLM response exactly matches a valid group.
        Returns None if no match found (caller handles fallback).
        """
        # Exact match
        if raw in valid_groups:
            return raw

        # Case-insensitive match
        raw_lower = raw.lower()
        for group in valid_groups:
            if group.lower() == raw_lower:
                return group

        # Partial match — find which valid group the response contains
        for group in valid_groups:
            if group.lower() in raw_lower or raw_lower in group.lower():
                return group

        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Health check
    # ─────────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            models    = ollama.list()
            available = [m.model for m in models.models]
            return any(self.model in m for m in available)
        except Exception:
            return False
