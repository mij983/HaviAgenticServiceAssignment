"""
Decision Agent
--------------
Applies the confidence threshold to determine whether a ticket should
be auto-assigned or routed to manual triage.
"""

import logging

logger = logging.getLogger(__name__)


class DecisionAgent:
    """
    Makes the final routing decision based on confidence score and
    whether the predicted assignment group is currently active.
    """

    def __init__(self, threshold: float = 7.0):
        self.threshold = threshold
        logger.info(f"DecisionAgent initialized with threshold={self.threshold}")

    def should_auto_assign(self, confidence: float) -> bool:
        """
        Returns True only when the confidence score exceeds the threshold.
        """
        result = confidence > self.threshold
        logger.debug(
            f"Decision: confidence={confidence}, threshold={self.threshold}, "
            f"auto_assign={result}"
        )
        return result

    def decide(
        self,
        ticket: dict,
        predicted_group: str,
        confidence: float,
        is_active: bool,
    ) -> dict:
        """
        Returns a decision dict with all context needed for the Update Agent
        and the audit log.

        Keys:
            auto_assign   – bool
            reason        – human-readable explanation
            group         – final recommended assignment group
            confidence    – score (1–10)
        """
        if not is_active:
            return {
                "auto_assign": False,
                "reason": f"Predicted group '{predicted_group}' is not active.",
                "group": predicted_group,
                "confidence": confidence,
            }

        if not self.should_auto_assign(confidence):
            return {
                "auto_assign": False,
                "reason": (
                    f"Confidence {confidence} ≤ threshold {self.threshold}. "
                    "Routing to manual triage."
                ),
                "group": predicted_group,
                "confidence": confidence,
            }

        return {
            "auto_assign": True,
            "reason": (
                f"Confidence {confidence} > threshold {self.threshold}. "
                f"Auto-assigning to '{predicted_group}'."
            ),
            "group": predicted_group,
            "confidence": confidence,
        }
