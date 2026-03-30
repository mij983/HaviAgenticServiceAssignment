"""
LLM Agent
----------
Supports two LLM providers, selected by config/config.yaml:

  provider: "azure_openai"   — Azure OpenAI (GPT-4o, GPT-4, GPT-35-turbo)
  provider: "ollama"         — Local Ollama (Mistral, Gemma, LLaMA — no API key)

Azure OpenAI configuration (set in .env):
  AZURE_OPENAI_ENDPOINT      e.g. https://your-resource.openai.azure.com/
  AZURE_OPENAI_API_KEY       your Azure OpenAI API key
  AZURE_OPENAI_API_VERSION   e.g. 2024-02-01
  AZURE_OPENAI_DEPLOYMENT    your deployment name e.g. gpt-4o

Ollama configuration (no .env needed):
  Install Ollama from https://ollama.com
  Then: ollama pull mistral

Confidence Score (1-10):
  >= 0.90 -> 10  >= 0.80 -> 9  >= 0.70 -> 8  >= 0.60 -> 7
  >= 0.50 -> 6   >= 0.42 -> 5  >= 0.34 -> 4  >= 0.25 -> 3
  >= 0.15 -> 2   <  0.15 -> 1

  HIGH: 7-10  MEDIUM: 4-6  LOW: 1-3

System prompt anti-hallucination rules:
  RULE 1: Not an IT ticket -> NOT_IT_TICKET
  RULE 2: Respond ONLY with group name
  RULE 3: Group must be from valid list exactly
  RULE 4: Base decisions on evidence only
  RULE 5: Split evidence -> highest combined similarity wins
  RULE 6: No hallucination
"""

import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an IT service desk routing assistant.

Your job is to read a support ticket and decide which IT team should handle it.

You will be given:
1. The ticket short description submitted by the user
2. Similar historical tickets that were successfully resolved, each showing which team handled it
3. Relevant KB articles / documents that may contain routing guidance
4. A list of all valid assignment groups

STRICT RULES - follow every rule without exception:

RULE 1 - VALID IT TICKET CHECK:
  Before doing anything else, decide: is this a genuine IT support ticket?
  A valid IT ticket describes a technical problem, system issue, access issue,
  software/hardware fault, network problem, or any IT-related service request.
  If NOT a valid IT ticket, respond with exactly: NOT_IT_TICKET

RULE 2 - RESPOND ONLY WITH THE GROUP NAME:
  If it IS a valid IT ticket, respond with ONLY the assignment group name.
  No explanation. No punctuation. No bullet points. No extra text whatsoever.

RULE 3 - ONLY USE GROUPS FROM THE VALID LIST:
  The assignment group MUST be exactly one from the valid list provided.
  Copy the group name exactly - character for character.

RULE 4 - BASE DECISIONS ON EVIDENCE ONLY:
  Only predict a group if the similar historical tickets or KB articles support it.
  Weight higher-similarity tickets more heavily.

RULE 5 - WHEN EVIDENCE IS SPLIT:
  Pick the single group with the highest combined similarity score.

RULE 6 - NO HALLUCINATION:
  Never output anything except a valid group name from the list OR NOT_IT_TICKET.
"""

VALIDATION_PROMPT = """You are an IT ticket validator.

Decide if the following text is a genuine IT support ticket.

A genuine IT ticket describes: a technical problem, system error, login or access
issue, software or hardware fault, network issue, or any IT-related service request.

NOT a genuine IT ticket: personal problems, medical issues, greetings, random words,
nonsense text, questions about non-IT topics, or anything unrelated to IT support.

Respond with exactly one word - YES or NO.

Text:
"""


class LLMAgent:

    def __init__(
        self,
        provider:          str   = "ollama",
        model:             str   = "gemma:2b",
        temperature:       float = 0.1,
        azure_endpoint:    str   = "",
        azure_api_key:     str   = "",
        azure_api_version: str   = "",
        azure_deployment:  str   = "",
    ):
        self.provider    = provider.lower()
        self.model       = model
        self.temperature = temperature

        self.azure_endpoint    = azure_endpoint    or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.azure_api_key     = azure_api_key     or os.getenv("AZURE_OPENAI_API_KEY", "")
        self.azure_api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.azure_deployment  = azure_deployment  or os.getenv("AZURE_OPENAI_DEPLOYMENT", model)

        self._azure_client = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        short_description: str,
        similar_tickets:   list[dict],
        valid_groups:      list[str],
    ) -> dict:
        prompt = self._build_prompt(short_description, similar_tickets, valid_groups)

        try:
            if self.provider == "azure_openai":
                raw_answer = self._call_azure(SYSTEM_PROMPT, prompt)
            else:
                raw_answer = self._call_ollama(SYSTEM_PROMPT, prompt)

            if raw_answer.upper() == "NOT_IT_TICKET":
                return self._invalid_ticket_result(similar_tickets, raw_answer)

            if not self._looks_like_group(raw_answer, valid_groups):
                if self._is_non_it_input(short_description):
                    return self._invalid_ticket_result(similar_tickets, raw_answer)

            predicted = self._validate(raw_answer, valid_groups)

            if predicted is None:
                logger.warning("LLM returned unrecognised group '%s'. Using weighted fallback.", raw_answer)
                return self._weighted_vote_result(similar_tickets, valid_groups, llm_raw=raw_answer)

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
            }

        except Exception as e:
            logger.error("LLM error: %s", e)
            return self._weighted_vote_result(similar_tickets, valid_groups, error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Provider calls
    # ─────────────────────────────────────────────────────────────────────────

    def _call_azure(self, system_prompt: str, user_prompt: str) -> str:
        client = self._get_azure_client()
        response = client.chat.completions.create(
            model       = self.azure_deployment,
            temperature = self.temperature,
            max_tokens  = 100,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        import ollama
        response = ollama.chat(
            model    = self.model,
            options  = {"temperature": self.temperature},
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        return response["message"]["content"].strip()

    def _get_azure_client(self):
        if self._azure_client is not None:
            return self._azure_client
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for Azure OpenAI.\n"
                "Install it with:  pip install openai"
            )
        if not self.azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set. Add it to your .env file.")
        if not self.azure_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set. Add it to your .env file.")
        from openai import AzureOpenAI
        self._azure_client = AzureOpenAI(
            azure_endpoint = self.azure_endpoint,
            api_key        = self.azure_api_key,
            api_version    = self.azure_api_version,
        )
        return self._azure_client

    # ─────────────────────────────────────────────────────────────────────────
    # Non-IT input detection
    # ─────────────────────────────────────────────────────────────────────────

    def _is_non_it_input(self, text: str) -> bool:
        try:
            if self.provider == "azure_openai":
                answer = self._call_azure("", VALIDATION_PROMPT + text)
            else:
                import ollama
                response = ollama.chat(
                    model    = self.model,
                    options  = {"temperature": 0.0},
                    messages = [{"role": "user", "content": VALIDATION_PROMPT + text}],
                )
                answer = response["message"]["content"].strip()
            return answer.strip().upper().startswith("NO")
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
    # Confidence scoring
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
        match_count = sum(1 for t in similar_tickets if t["assignment_group"] == predicted)
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
    # Health check
    # ─────────────────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self.provider == "azure_openai":
            try:
                client = self._get_azure_client()
                client.chat.completions.create(
                    model      = self.azure_deployment,
                    max_tokens = 1,
                    messages   = [{"role": "user", "content": "ping"}],
                )
                return True
            except Exception as e:
                logger.warning("Azure OpenAI health check failed: %s", e)
                return False
        else:
            try:
                import ollama
                models    = ollama.list()
                available = [m["name"] for m in models.get("models", [])]
                return any(self.model in m for m in available)
            except Exception:
                return False

    def provider_label(self) -> str:
        if self.provider == "azure_openai":
            return self.azure_deployment + " (Azure OpenAI)"
        return self.model + " via Ollama"
