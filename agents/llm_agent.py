"""
LLM Agent
----------
Uses a locally running LLM via Ollama to reason over retrieved similar
tickets and predict the correct assignment group.

Accuracy improvements in this version:
  1. Stronger system prompt — explicit few-shot style instructions that
     work better with smaller models (gemma3:4b, gemma:2b).
  2. Description context in prompt — the top-3 most similar tickets now
     include their description snippet (not just short_description).
     This gives the LLM more evidence to distinguish ambiguous tickets.
  3. Top-group hint — the weighted-vote winner is shown to the LLM as a
     suggested answer. The LLM can override it but has a strong anchor.
  4. gemma4 support — works with any Ollama model including gemma4.
  5. RLHF Stage 3B prompt bias injection (caution_groups) preserved.
  6. Fixed for ollama >= 0.6.1 (Pydantic response objects).
  7. temperature=0.0 by default for fully deterministic output.
"""

import logging
from collections import defaultdict

import ollama

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# System prompt — rewritten for better accuracy with small models
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert IT service desk ticket routing assistant at HAVI.

Your ONLY job is to read a support ticket and output the correct assignment group name.

CRITICAL INSTRUCTIONS:

1. OUTPUT FORMAT
   Output EXACTLY ONE line: the assignment group name, nothing else.
   No explanation. No "I think". No punctuation after the name. Just the name.

2. NOT AN IT TICKET
   If the input is not a real IT support ticket (e.g. random words, greetings,
   personal matters), output exactly: NOT_IT_TICKET

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

VALIDATION_PROMPT = """Is the following text a genuine IT support ticket?
A genuine IT ticket describes a technical problem, system issue, access issue,
software/hardware fault, or IT service request.
Reply with YES or NO only.
Text: """


class LLMAgent:

    def __init__(self, model: str = "gemma3:4b", temperature: float = 0.0):
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
        caution_groups:    list[str] = None,
    ) -> dict:
        prompt = self._build_prompt(
            short_description, similar_tickets, valid_groups,
            caution_groups=caution_groups or []
        )

        try:
            response = ollama.chat(
                model    = self.model,
                options  = {"temperature": self.temperature},
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )

            # ollama >= 0.6.1 returns Pydantic ChatResponse
            raw_answer = response.message.content.strip()

            if raw_answer.upper() == "NOT_IT_TICKET":
                return self._invalid_ticket_result(similar_tickets, raw_answer)

            if not self._looks_like_group(raw_answer, valid_groups):
                if self._is_non_it_input(short_description):
                    return self._invalid_ticket_result(similar_tickets, raw_answer)

            predicted = self._validate(raw_answer, valid_groups)

            if predicted is None:
                logger.warning(
                    "LLM returned unrecognised group '%s'. Using weighted fallback.", raw_answer
                )
                return self._weighted_vote_result(
                    similar_tickets, valid_groups, llm_raw=raw_answer
                )

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
            return self._weighted_vote_result(
                similar_tickets, valid_groups, error=str(e)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Non-IT input detection
    # ─────────────────────────────────────────────────────────────────────────

    def _is_non_it_input(self, text: str) -> bool:
        try:
            response = ollama.chat(
                model    = self.model,
                options  = {"temperature": 0.0},
                messages = [{"role": "user", "content": VALIDATION_PROMPT + text}],
            )
            answer = response.message.content.strip().upper()
            return answer.startswith("NO")
        except Exception:
            return False

    def _looks_like_group(self, raw: str, valid_groups: list[str]) -> bool:
        raw_lower = raw.lower()
        return any(
            group.lower() in raw_lower or raw_lower in group.lower()
            for group in valid_groups
        )

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

    # ─────────────────────────────────────────────────────────────────────────
    # Confidence scoring (1-10)
    # ─────────────────────────────────────────────────────────────────────────

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

        label = "high" if score >= 7 else "medium" if score >= 4 else "low"
        match_count = sum(
            1 for t in similar_tickets if t["assignment_group"] == predicted
        )
        return match_count, score, label

    # ─────────────────────────────────────────────────────────────────────────
    # Weighted-vote fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _weighted_vote_result(
        self,
        similar_tickets: list[dict],
        valid_groups:    list[str],
        llm_raw: str = "",
        error:   str = "",
    ) -> dict:
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = t.get("similarity_raw", t["similarity_score"])
            weighted_votes[t["assignment_group"]] += weight

        predicted = (
            max(weighted_votes, key=weighted_votes.__getitem__)
            if weighted_votes else valid_groups[0]
        )

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
            "raw_llm_response": llm_raw or ("LLM unavailable: " + error),
            "similar_tickets":  similar_tickets,
            "fallback":         True,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt builder
    # Accuracy improvement: includes description snippets for top matches,
    # shows weighted-vote top group as a [TOP-VOTE] anchor hint to the LLM.
    # ─────────────────────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        short_description: str,
        similar_tickets:   list[dict],
        valid_groups:      list[str],
        caution_groups:    list[str] = None,
    ) -> str:

        # Compute weighted-vote winner to show as anchor hint
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = t.get("similarity_raw", t["similarity_score"])
            weighted_votes[t["assignment_group"]] += weight
        top_vote = (
            max(weighted_votes, key=weighted_votes.__getitem__)
            if weighted_votes else ""
        )

        prompt  = "TICKET TO ROUTE:\n"
        prompt += short_description + "\n\n"

        hist_tickets = [t for t in similar_tickets
                        if t.get("source_type", "ticket") in (
                            "ticket", "rlhf_positive", "rlhf_negative_corrected")]
        doc_chunks   = [t for t in similar_tickets if t.get("source_type") == "document"]

        if hist_tickets:
            prompt += "SIMILAR HISTORICAL TICKETS (highest similarity first):\n"
            for i, ticket in enumerate(hist_tickets, 1):
                src = ticket.get("source_type", "ticket")
                tag = ""
                if src == "rlhf_positive":
                    tag = " [confirmed-correct]"
                elif src == "rlhf_negative_corrected":
                    tag = " [human-corrected]"
                elif ticket["assignment_group"] == top_vote and i == 1:
                    tag = " [TOP-VOTE]"

                prompt += (
                    str(i) + ". [" + ticket["assignment_group"] + "]"
                    + tag + " sim=" + str(ticket["similarity_score"]) + "/10\n"
                    + "   Title: " + ticket["short_description"] + "\n"
                )
                # Include description for top 3 tickets — gives LLM richer evidence
                if i <= 3 and ticket.get("description", "").strip():
                    desc_snippet = ticket["description"][:200].strip()
                    prompt += "   Detail: " + desc_snippet + "\n"
            prompt += "\n"

        if doc_chunks:
            prompt += "RELEVANT KB ARTICLES:\n"
            for i, chunk in enumerate(doc_chunks, 1):
                team_hint = (" => " + chunk["assignment_group"]) if chunk["assignment_group"] else ""
                prompt += (
                    str(i) + ". [" + chunk["short_description"] + "]" + team_hint
                    + " sim=" + str(chunk["similarity_score"]) + "/10\n"
                    + "   " + chunk["description"][:300] + "\n"
                )
            prompt += "\n"

        # Stage 3B RLHF caution hint
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

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate(self, raw: str, valid_groups: list[str]) -> str | None:
        # Strip any accidental punctuation the model may add
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

    # ─────────────────────────────────────────────────────────────────────────
    # Health check — supports all Ollama models including gemma4
    # ─────────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if Ollama is running and the configured model is available."""
        try:
            # ollama >= 0.6.1 returns ListResponse Pydantic object
            response  = ollama.list()
            available = [m.model for m in response.models]
            return any(self.model in m for m in available)
        except Exception:
            return False
