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

Confidence Score (1-10):
  The score is computed from weighted similarity votes (each similar ticket
  contributes its similarity score as a vote weight for its group).

  score = (weighted_share_of_winning_group) mapped onto 1-10:
    >= 0.90  -> 10    (near-unanimous, all top matches agree)
    >= 0.75  -> 8-9   (strong majority)
    >= 0.55  -> 6-7   (moderate majority)
    >= 0.40  -> 4-5   (weak majority / split evidence)
    <  0.40  -> 1-3   (very split / low similarity)

  HIGH   : score 7-10
  MEDIUM : score 4-6
  LOW    : score 1-3
"""

import logging
from collections import defaultdict

import ollama

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an IT service desk routing assistant.

Your job is to read a support ticket and decide which IT team should handle it.

You will be given:
1. The ticket short description submitted by the user
2. Similar historical tickets that were successfully resolved, each showing which team handled it
3. A list of all valid assignment groups

Rules:
- You MUST respond with ONLY the assignment group name
- The assignment group MUST be exactly one from the valid list provided
- Do not add any explanation, punctuation, or extra text
- Base your decision on the patterns from the similar historical tickets
- Weight higher-similarity tickets more heavily in your decision
- If the ticket is ambiguous, pick the most likely group based on the weighted evidence
"""


class LLMAgent:

    def __init__(self, model: str = "gemma", temperature: float = 0.1):
        self.model       = model
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        short_description: str,
        similar_tickets: list[dict],
        valid_groups: list[str],
    ) -> dict:
        """
        Ask the LLM to predict the assignment group.

        Returns:
            {
                "assignment_group": "IT-SC-EPAM-SAP Workflow",
                "confidence":       "high",
                "confidence_score": 8,          # 1-10
                "match_count":      3,
                "top_k":            5,
                "similar_tickets":  [...],
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
            predicted  = self._validate(raw_answer, valid_groups)

            # If LLM returns something unrecognised, fall back to weighted vote
            if predicted is None:
                logger.warning("LLM returned unrecognised group '%s'. Using weighted fallback.", raw_answer)
                return self._weighted_vote_result(similar_tickets, valid_groups,
                                                  llm_raw=raw_answer)

            match_count, confidence_score, confidence_label = self._score(
                predicted, similar_tickets
            )

            return {
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
            return self._weighted_vote_result(similar_tickets, valid_groups,
                                              error=str(e))

    # ------------------------------------------------------------------
    # Confidence scoring (1-10)
    # ------------------------------------------------------------------

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

        total_weight = sum(weighted_votes.values()) or 1.0
        winning_share = weighted_votes.get(predicted, 0.0) / total_weight

        # Map share to 1-10
        if   winning_share >= 0.90:
            score = 10
        elif winning_share >= 0.80:
            score = 9
        elif winning_share >= 0.70:
            score = 8
        elif winning_share >= 0.60:
            score = 7
        elif winning_share >= 0.50:
            score = 6
        elif winning_share >= 0.42:
            score = 5
        elif winning_share >= 0.34:
            score = 4
        elif winning_share >= 0.25:
            score = 3
        elif winning_share >= 0.15:
            score = 2
        else:
            score = 1

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

    # ------------------------------------------------------------------
    # Weighted-vote fallback
    # ------------------------------------------------------------------

    def _weighted_vote_result(
        self,
        similar_tickets: list[dict],
        valid_groups: list[str],
        llm_raw: str = "",
        error: str = "",
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

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        short_description: str,
        similar_tickets: list[dict],
        valid_groups: list[str],
    ) -> str:
        prompt = "NEW TICKET:\n"
        prompt += short_description + "\n\n"

        prompt += "SIMILAR HISTORICAL TICKETS (ranked by similarity, highest first):\n"
        for i, ticket in enumerate(similar_tickets, 1):
            prompt += (
                str(i) + ". [" + ticket["assignment_group"] + "] "
                + ticket["short_description"]
                + " (similarity: " + str(ticket["similarity_score"]) + ")\n"
            )

        prompt += "\nVALID ASSIGNMENT GROUPS:\n"
        for group in valid_groups:
            prompt += "- " + group + "\n"

        prompt += "\nRespond with ONLY the assignment group name:"
        return prompt

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, raw: str, valid_groups: list[str]) -> str | None:
        """
        Check if the LLM response exactly matches a valid group.
        Returns None if no match found (caller handles fallback).
        """
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

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            models    = ollama.list()
            available = [m.model for m in models.models]
            return any(self.model in m for m in available)
        except Exception:
            return False
