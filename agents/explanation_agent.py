from __future__ import annotations

from typing import Any


class ExplanationAgent:
    """Creates concise user-facing reasoning from retrieval evidence."""

    def explain(self, result: dict[str, Any]) -> str:
        group = result.get("assignment_group")
        group_scores = result.get("group_scores", [])
        tickets = result.get("similar_tickets", [])
        top_alts = result.get("top_alternative_groups", [])
        retrieval_only = result.get("retrieval_only", False)

        snippets = []
        same_group_examples = [
            t for t in tickets if t.get("assignment_group") == group
        ][:2]
        for item in same_group_examples:
            desc = item.get("short_description") or item.get("description") or ""
            if desc:
                snippets.append(desc[:70])

        mode = "retrieval evidence" if retrieval_only else "retrieval evidence plus shortlisted LLM review"
        reason = f"Predicted {group} using {mode}."

        if group_scores:
            top = group_scores[0]
            reason += (
                f" Top group weight: {top.get('weighted_score', 0.0):.2f}"
                f" from {top.get('ticket_count', 0)} similar tickets."
            )
        if top_alts:
            reason += f" Next candidates: {', '.join(top_alts)}."
        if snippets:
            reason += f" Closest evidence: {' | '.join(snippets)}."
        return reason
