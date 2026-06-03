from __future__ import annotations

from collections import defaultdict
from typing import Any


class CandidateGroupAgent:
    """Builds assignment-group candidates from retrieved tickets.

    Retrieval-first logic:
      1. Filter weak matches
      2. Aggregate evidence per assignment group
      3. Shortlist top candidate groups
    """

    def __init__(self, similarity_floor_raw: float = 0.20, min_candidates: int = 3):
        self.similarity_floor_raw = similarity_floor_raw
        self.min_candidates = min_candidates

    def rank_candidates(
        self,
        similar_tickets: list[dict[str, Any]],
        max_candidates: int = 3,
    ) -> dict[str, Any]:
        if not similar_tickets:
            return {
                "filtered_tickets": [],
                "candidate_groups": [],
                "group_scores": [],
                "dominant": False,
                "top_gap": 0.0,
            }

        filtered = [
            t for t in similar_tickets
            if t.get("similarity_raw", 0.0) >= self.similarity_floor_raw
        ]
        if len(filtered) < self.min_candidates:
            filtered = similar_tickets[: max(self.min_candidates, max_candidates)]

        grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "group": "",
            "ticket_count": 0,
            "weighted_score": 0.0,
            "avg_similarity_raw": 0.0,
            "best_similarity_raw": 0.0,
            "examples": [],
        })

        for ticket in filtered:
            group = ticket.get("assignment_group", "")
            raw = float(ticket.get("similarity_raw", 0.0))
            bucket = grouped[group]
            bucket["group"] = group
            bucket["ticket_count"] += 1
            bucket["weighted_score"] += raw
            bucket["best_similarity_raw"] = max(bucket["best_similarity_raw"], raw)
            if len(bucket["examples"]) < 3:
                bucket["examples"].append(ticket)

        group_scores = sorted(
            grouped.values(),
            key=lambda x: (
                x["weighted_score"],
                x["ticket_count"],
                x["best_similarity_raw"],
            ),
            reverse=True,
        )

        for item in group_scores:
            count = item["ticket_count"] or 1
            item["avg_similarity_raw"] = round(item["weighted_score"] / count, 4)
            item["weighted_score"] = round(item["weighted_score"], 4)
            item["best_similarity_raw"] = round(item["best_similarity_raw"], 4)

        candidate_groups = [g["group"] for g in group_scores[:max_candidates]]

        top_gap = 0.0
        dominant = False
        if len(group_scores) >= 2:
            top_gap = round(group_scores[0]["weighted_score"] - group_scores[1]["weighted_score"], 4)
            dominant = (
                group_scores[0]["weighted_score"] >= group_scores[1]["weighted_score"] * 1.20
                and group_scores[0]["ticket_count"] >= group_scores[1]["ticket_count"]
            )
        elif len(group_scores) == 1:
            dominant = True
            top_gap = group_scores[0]["weighted_score"]

        return {
            "filtered_tickets": filtered,
            "candidate_groups": candidate_groups,
            "group_scores": group_scores,
            "dominant": dominant,
            "top_gap": top_gap,
        }
