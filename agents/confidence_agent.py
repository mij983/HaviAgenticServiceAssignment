from __future__ import annotations

from typing import Any


class ConfidenceAgent:
    """Computes a more grounded confidence score from retrieval evidence."""

    def score(
        self,
        predicted_group: str,
        similar_tickets: list[dict[str, Any]],
        group_scores: list[dict[str, Any]],
        llm_used: bool = False,
        llm_agrees_with_top_retrieval: bool = True,
    ) -> dict[str, Any]:
        if not similar_tickets or not predicted_group:
            return {
                "confidence_score": 1,
                "confidence": "low",
                "match_count": 0,
                "signals": {},
            }

        top_sim = max(float(t.get("similarity_raw", 0.0)) for t in similar_tickets)
        match_count = sum(1 for t in similar_tickets if t.get("assignment_group") == predicted_group)
        total = max(len(similar_tickets), 1)
        agreement = match_count / total

        best_group = group_scores[0] if group_scores else None
        second_group = group_scores[1] if len(group_scores) > 1 else None
        top_group_name = best_group["group"] if best_group else predicted_group
        top_group_score = best_group["weighted_score"] if best_group else 0.0
        second_group_score = second_group["weighted_score"] if second_group else 0.0
        margin = max(top_group_score - second_group_score, 0.0)

        if predicted_group == top_group_name:
            predicted_score = top_group_score
        else:
            predicted_score = next(
                (g["weighted_score"] for g in group_scores if g["group"] == predicted_group),
                0.0,
            )

        total_weight = sum(float(g.get("weighted_score", 0.0)) for g in group_scores) or 1.0
        winning_share = predicted_score / total_weight

        raw_score = 1.0
        raw_score += min(top_sim * 4.0, 3.0)
        raw_score += min(agreement * 2.5, 2.0)
        raw_score += min(winning_share * 3.0, 2.0)
        raw_score += min(margin * 3.0, 1.5)
        if llm_used and llm_agrees_with_top_retrieval:
            raw_score += 0.5
        elif llm_used and not llm_agrees_with_top_retrieval:
            raw_score -= 0.5

        score = max(1, min(10, int(round(raw_score))))
        label = "high" if score >= 8 else "medium" if score >= 5 else "low"

        return {
            "confidence_score": score,
            "confidence": label,
            "match_count": match_count,
            "signals": {
                "top_similarity_raw": round(top_sim, 4),
                "agreement": round(agreement, 4),
                "winning_share": round(winning_share, 4),
                "margin": round(margin, 4),
                "llm_used": llm_used,
                "llm_agrees_with_top_retrieval": llm_agrees_with_top_retrieval,
            },
        }
