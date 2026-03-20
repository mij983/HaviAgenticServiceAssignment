"""
Preprocessing Agent
--------------------
Cleans and normalises the user input before embedding.

Handles:
  - Whitespace normalisation
  - Basic noise removal (ticket IDs, numbers)
  - Text lowercasing for consistent embedding
"""

import re
import logging

logger = logging.getLogger(__name__)


class PreprocessingAgent:

    def process(self, text: str) -> str:
        """Clean and normalise input text."""
        if not text:
            return ""

        # Strip leading/trailing whitespace
        text = text.strip()

        # Remove ticket number patterns like INC0001234, REQ123456, CHG000001
        text = re.sub(r"\b(INC|REQ|CHG|PRB|TASK)\d+\b", "", text, flags=re.IGNORECASE)

        # Remove excessive punctuation but keep meaningful ones
        text = re.sub(r"[^\w\s\-/\.\,]", " ", text)

        # Normalise whitespace
        text = " ".join(text.split())

        return text.strip()

    def is_valid(self, text: str) -> tuple[bool, str]:
        """
        Check if the input is usable.
        Returns (is_valid, error_message).
        """
        if not text or not text.strip():
            return False, "Please enter a ticket short description."

        if len(text.strip()) < 5:
            return False, "Description too short. Please provide more detail."

        if len(text.strip()) > 500:
            return False, "Description too long. Please keep it under 500 characters."

        return True, ""
