from __future__ import annotations

from collections import defaultdict
from typing import Any

from agents.candidate_group_agent import CandidateGroupAgent
from agents.confidence_agent import ConfidenceAgent


class DecisionAgent:
    """LLM-first final decision engine.

    CHANGE FROM ORIGINAL:
    ---------------------
    The original code had three conditions that silently bypassed the LLM:
      1. candidates["dominant"]         -- top group is 20%+ stronger than second
      2. candidates["top_gap"] >= 0.12  -- score gap between top two candidates
      3. not llm_agent.is_available()   -- Ollama offline (this one kept as safe fallback)

    This meant the LLM was skipped for most real-world tickets where retrieval
    evidence leans one way, producing retrieval-only predictions with no
    language-model reasoning.

    NEW BEHAVIOUR:
    --------------
    - The LLM is ALWAYS called as long as Ollama is available.
    - "dominant" and "top_gap" are kept as DIAGNOSTIC SIGNALS only -- they no
      longer skip the LLM.
    - The retrieval weighted-vote winner is still shown to the LLM as [TOP-VOTE]
      so strong retrieval evidence still anchors the LLM -- it just doesn't
      replace it.
    - If Ollama is genuinely offline, falls back to weighted vote (safe fallback).

    Flow:
      1. Rank candidate groups from retrieval evidence        (CandidateGroupAgent)
      2. Compute retrieval weighted-vote winner               (always)
      3. Call LLM with candidates + full evidence prompt      (ALWAYS if online)
      4. Accept LLM answer if it is within the candidates
      5. Compute confidence                                   (ConfidenceAgent)
      6. Return full result dict
    """

    def __init__(
        self,
        llm_agent,
        candidate_agent: CandidateGroupAgent | None = None,
        confidence_agent: ConfidenceAgent | None = None,
        llm_candidate_count: int = 3,
    ):
        self.llm_agent = llm_agent
        self.candidate_agent = candidate_agent or CandidateGroupAgent()
        self.confidence_agent = confidence_agent or ConfidenceAgent()
        self.llm_candidate_count = llm_candidate_count
        # NOTE: llm_min_gap_for_skip intentionally REMOVED -- it caused LLM bypass.

    def predict(
        self,
        short_description: str,
        similar_tickets: list[dict[str, Any]],
        valid_groups: list[str],
        caution_groups: list[str] | None = None,
    ) -> dict[str, Any]:

        # Step 1 -- Rank candidate groups from retrieval evidence
        candidates = self.candidate_agent.rank_candidates(
            similar_tickets,
            max_candidates=min(self.llm_candidate_count, max(len(valid_groups), 1)),
        )
        candidate_groups = candidates["candidate_groups"] or valid_groups[: self.llm_candidate_count]
        filtered_tickets  = candidates["filtered_tickets"] or similar_tickets
        group_scores      = candidates["group_scores"]

        # Step 2 -- Compute retrieval weighted-vote (used as LLM anchor hint)
        retrieval_choice = self._weighted_vote(filtered_tickets, valid_groups)
        retrieval_winner = retrieval_choice["assignment_group"]

        used_llm        = False
        fallback        = False
        invalid_ticket  = False
        raw_llm_response = ""
        final_group     = retrieval_winner   # safe default if LLM is offline

        # Step 3 -- ALWAYS call the LLM (if Ollama is available)
        llm_available = getattr(self.llm_agent, "is_available", lambda: False)()

        if llm_available:
            llm_result = self.llm_agent.predict(
                short_description = short_description,
                similar_tickets   = filtered_tickets,
                valid_groups      = candidate_groups,
                caution_groups    = caution_groups or [],
            )
            raw_llm_response = llm_result.get("raw_llm_response", "")
            used_llm         = not llm_result.get("fallback", False)
            fallback         = llm_result.get("fallback", False)
            invalid_ticket   = not llm_result.get("is_valid_ticket", True)

            if not invalid_ticket:
                llm_group = llm_result.get("assignment_group")
                if llm_group in candidate_groups:
                    # LLM gave a valid candidate -- use it
                    final_group = llm_group
                else:
                    # LLM returned something outside the candidates -- trust retrieval
                    final_group = retrieval_winner
                    fallback    = True
                    raw_llm_response = (
                        f"LLM returned '{llm_group}' (outside candidates) "
                        f"-- falling back to retrieval winner '{retrieval_winner}'"
                    )
        else:
            # Ollama is offline -- fall back to retrieval weighted vote
            final_group      = retrieval_winner
            fallback         = True
            raw_llm_response = "LLM unavailable -- using retrieval weighted vote"

        # Step 4 -- Compute confidence
        confidence = self.confidence_agent.score(
            predicted_group              = final_group,
            similar_tickets              = filtered_tickets,
            group_scores                 = group_scores,
            llm_used                     = used_llm,
            llm_agrees_with_top_retrieval= (final_group == retrieval_winner),
        )

        top_alternatives = [g["group"] for g in group_scores[1:3]]

        return {
            "is_valid_ticket":         not invalid_ticket,
            "assignment_group":        final_group,
            "confidence":              confidence["confidence"],
            "confidence_score":        confidence["confidence_score"],
            "confidence_signals":      confidence["signals"],
            "match_count":             confidence["match_count"],
            "top_k":                   len(similar_tickets),
            "similar_tickets":         similar_tickets,
            "candidate_groups":        candidate_groups,
            "group_scores":            group_scores,
            "top_alternative_groups":  top_alternatives,
            "retrieval_winner":        retrieval_winner,
            # retrieval_only is now only True when Ollama is offline
            "retrieval_only":          not llm_available,
            "fallback":                fallback,
            "raw_llm_response":        raw_llm_response,
            # Diagnostic signals (kept for transparency, no longer affect routing)
            "retrieval_dominant":      candidates["dominant"],
            "retrieval_top_gap":       candidates["top_gap"],
        }

    # ── Internal weighted vote ────────────────────────────────────────────────

    def _weighted_vote(
        self,
        similar_tickets: list[dict[str, Any]],
        valid_groups: list[str],
    ) -> dict[str, Any]:
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = float(t.get("similarity_raw", t.get("similarity_score", 0.0)))
            weighted_votes[t.get("assignment_group", "")] += weight

        predicted = (
            max(weighted_votes, key=weighted_votes.__getitem__)
            if weighted_votes else valid_groups[0]
        )
        return {
            "is_valid_ticket":  True,
            "assignment_group": predicted,
            "fallback":         True,
            "raw_llm_response": "retrieval_weighted_vote",
        }
