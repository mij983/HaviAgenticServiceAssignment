import logging
from agents.historical_data_agent import TEAM_KEYWORDS, kw_score

logger = logging.getLogger(__name__)


def _text_confidence(ticket: dict, predicted_group: str) -> float:
    sd   = (ticket.get("short_description", "") or "").lower()
    desc = (ticket.get("description", "")       or "").lower()

    keywords = TEAM_KEYWORDS.get(predicted_group, set())
    if not keywords:
        return 0.5

    SAT = 3
    sd_hits   = min(kw_score(sd,   keywords), SAT)
    desc_hits = min(kw_score(desc, keywords), SAT)
    score = (2 * sd_hits + desc_hits) / (3 * SAT)
    return min(score, 1.0)


def _knowledge_score(predicted_group: str, active_groups: list) -> float:
    return 1.0 if predicted_group in active_groups else 0.0


class ConfidenceScoringEngine:

    def __init__(self, historical_weight=0.60, text_weight=0.30, knowledge_weight=0.10):
        assert abs(historical_weight + text_weight + knowledge_weight - 1.0) < 1e-9
        self.hw = historical_weight
        self.tw = text_weight
        self.kw = knowledge_weight

    def calculate(self, raw_probability, ticket, predicted_group, active_groups) -> float:
        text_score = _text_confidence(ticket, predicted_group)
        know_score = _knowledge_score(predicted_group, active_groups)
        composite  = self.hw * raw_probability + self.tw * text_score + self.kw * know_score
        scaled = round(1.0 + composite * 9.0, 1)
        return max(1.0, min(10.0, scaled))
