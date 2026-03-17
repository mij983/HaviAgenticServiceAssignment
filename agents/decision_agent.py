import logging

logger = logging.getLogger(__name__)


class DecisionAgent:

    def __init__(self, threshold: float = 7.0):
        self.threshold = threshold

    def should_auto_assign(self, confidence: float) -> bool:
        return confidence > self.threshold

    def decide(self, ticket: dict, predicted_group: str, confidence: float, is_active: bool) -> dict:
        if not is_active:
            return {
                "auto_assign": False,
                "reason": "Predicted group '" + predicted_group + "' is not active.",
                "group": predicted_group,
                "confidence": confidence,
            }

        if not self.should_auto_assign(confidence):
            return {
                "auto_assign": False,
                "reason": "Confidence " + str(confidence) + " <= threshold " + str(self.threshold) + ". Routing to manual triage.",
                "group": predicted_group,
                "confidence": confidence,
            }

        return {
            "auto_assign": True,
            "reason": "Confidence " + str(confidence) + " > threshold " + str(self.threshold) + ". Auto-assigning to '" + predicted_group + "'.",
            "group": predicted_group,
            "confidence": confidence,
        }
