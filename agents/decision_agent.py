from __future__ import annotations

from collections import defaultdict
from typing import Any

from agents.candidate_group_agent import CandidateGroupAgent
from agents.confidence_agent import ConfidenceAgent


class DecisionAgent:
    """Retrieval-first final decision engine.

    Flow:
      - rank candidate groups from retrieval evidence
      - auto-decide on strong dominance
      - otherwise use the LLM only across shortlisted groups
      - always compute confidence from measurable signals
    """

    def __init__(
        self,
        llm_agent,
        candidate_agent: CandidateGroupAgent | None = None,
        confidence_agent: ConfidenceAgent | None = None,
        llm_min_gap_for_skip: float = 0.12,
        llm_candidate_count: int = 3,
    ):
        self.llm_agent = llm_agent
        self.candidate_agent = candidate_agent or CandidateGroupAgent()
        self.confidence_agent = confidence_agent or ConfidenceAgent()
        self.llm_min_gap_for_skip = llm_min_gap_for_skip
        self.llm_candidate_count = llm_candidate_count

    def predict(
        self,
        short_description: str,
        similar_tickets: list[dict[str, Any]],
        valid_groups: list[str],
        caution_groups: list[str] | None = None,
    ) -> dict[str, Any]:
        candidates = self.candidate_agent.rank_candidates(
            similar_tickets,
            max_candidates=min(self.llm_candidate_count, max(len(valid_groups), 1)),
        )
        candidate_groups = candidates["candidate_groups"] or valid_groups[: self.llm_candidate_count]
        filtered_tickets = candidates["filtered_tickets"] or similar_tickets
        group_scores = candidates["group_scores"]

        retrieval_choice = self._weighted_vote(filtered_tickets, valid_groups)
        retrieval_winner = retrieval_choice["assignment_group"]
        used_llm = False
        fallback = retrieval_choice.get("fallback", False)
        invalid_ticket = False
        raw_llm_response = retrieval_choice.get("raw_llm_response", "")

        should_skip_llm = (
            candidates["dominant"]
            or candidates["top_gap"] >= self.llm_min_gap_for_skip
            or not getattr(self.llm_agent, "is_available")()
        )

        final_group = retrieval_winner
        if not should_skip_llm:
            llm_result = self.llm_agent.predict(
                short_description=short_description,
                similar_tickets=filtered_tickets,
                valid_groups=candidate_groups,
                caution_groups=caution_groups or [],
            )
            raw_llm_response = llm_result.get("raw_llm_response", "")
            used_llm = not llm_result.get("fallback", False)
            fallback = llm_result.get("fallback", False)
            invalid_ticket = not llm_result.get("is_valid_ticket", True)
            if not invalid_ticket and llm_result.get("assignment_group") in candidate_groups:
                final_group = llm_result["assignment_group"]

        confidence = self.confidence_agent.score(
            predicted_group=final_group,
            similar_tickets=filtered_tickets,
            group_scores=group_scores,
            llm_used=used_llm,
            llm_agrees_with_top_retrieval=(final_group == retrieval_winner),
        )

        top_alternatives = [g["group"] for g in group_scores[1:3]]
        result = {
            "is_valid_ticket": not invalid_ticket,
            "assignment_group": final_group,
            "confidence": confidence["confidence"],
            "confidence_score": confidence["confidence_score"],
            "confidence_signals": confidence["signals"],
            "match_count": confidence["match_count"],
            "top_k": len(similar_tickets),
            "similar_tickets": similar_tickets,
            "candidate_groups": candidate_groups,
            "group_scores": group_scores,
            "top_alternative_groups": top_alternatives,
            "retrieval_winner": retrieval_winner,
            "retrieval_only": should_skip_llm,
            "fallback": fallback,
            "raw_llm_response": raw_llm_response,
        }
        return result

    def _weighted_vote(
        self,
        similar_tickets: list[dict[str, Any]],
        valid_groups: list[str],
    ) -> dict[str, Any]:
        weighted_votes = defaultdict(float)
        for t in similar_tickets:
            weight = float(t.get("similarity_raw", t.get("similarity_score", 0.0)))
            weighted_votes[t.get("assignment_group", "")] += weight

        predicted = max(weighted_votes, key=weighted_votes.__getitem__) if weighted_votes else valid_groups[0]
        return {
            "is_valid_ticket": True,
            "assignment_group": predicted,
            "fallback": True,
            "raw_llm_response": "retrieval_weighted_vote",
        }
