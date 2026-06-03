"""
Preprocessing Agent
--------------------
Cleans and normalises input text before embedding.

Accuracy improvements in this version:
  - Expanded IT synonym normalisation: common abbreviations and informal
    phrases are mapped to their canonical terms before embedding.
    e.g. "cant login" -> "cannot login"
         "pw reset"   -> "password reset"
         "ms teams"   -> "microsoft teams"
    This reduces embedding distance between tickets that mean the same
    thing but use different words.
  - Stopword removal: generic words that carry no routing signal
    (e.g. "please", "urgent", "dear team") are stripped so the
    embedding vector focuses on the actual IT issue.
  - Expanded ticket ID pattern removal.
"""

import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# IT synonym map — normalise common abbreviations and informal variants
# ─────────────────────────────────────────────────────────────────────────────
_SYNONYM_MAP = [
    # Authentication / access
    (r"\bpw\b",                          "password"),
    (r"\bpasswd\b",                      "password"),
    (r"\bpwd\b",                         "password"),
    (r"\bcant login\b",                  "cannot login"),
    (r"\bcannot log in\b",               "cannot login"),
    (r"\bcan't log in\b",                "cannot login"),
    (r"\bunable to login\b",             "cannot login"),
    (r"\bno access\b",                   "access denied"),
    (r"\baccess issue\b",                "access problem"),
    (r"\bmfa\b",                         "multi-factor authentication"),
    (r"\b2fa\b",                         "two-factor authentication"),
    (r"\bsso\b",                         "single sign-on"),
    (r"\bad\b",                          "active directory"),
    # Microsoft / Office
    (r"\bms teams\b",                    "microsoft teams"),
    (r"\bteams app\b",                   "microsoft teams"),
    (r"\bo365\b",                        "office 365"),
    (r"\bm365\b",                        "microsoft 365"),
    (r"\bms office\b",                   "microsoft office"),
    (r"\boutlook mail\b",                "outlook email"),
    (r"\bonedrive\b",                    "one drive"),
    (r"\bsharepoint\b",                  "sharepoint"),
    # Network
    (r"\bvpn\b",                         "vpn network"),
    (r"\bno internet\b",                 "no internet connection"),
    (r"\bno connectivity\b",             "no network connection"),
    (r"\bwifi\b",                        "wireless network"),
    (r"\bwi-fi\b",                       "wireless network"),
    # Hardware
    (r"\blaptop crash\b",                "laptop not working"),
    (r"\bpc crash\b",                    "computer not working"),
    (r"\bbsod\b",                        "blue screen of death windows error"),
    (r"\bblue screen\b",                 "blue screen of death windows error"),
    # SAP
    (r"\bsap basis\b",                   "sap basis support"),
    (r"\bsap workflow\b",                "sap workflow approval"),
    (r"\bsap login\b",                   "sap cannot login"),
    (r"\bsap access\b",                  "sap access problem"),
    # General IT
    (r"\berror msg\b",                   "error message"),
    (r"\bapp crash\b",                   "application crash"),
    (r"\bapp not working\b",             "application not working"),
    (r"\bapplication down\b",            "application not working"),
    (r"\bsystem down\b",                 "system not working"),
    (r"\bserver down\b",                 "server not working"),
    (r"\bservice down\b",                "service not working"),
    (r"\bintune\b",                      "intune device management"),
    (r"\bbitlocker\b",                   "bitlocker encryption"),
    (r"\bmake me admin\b",               "make me admin local admin"),
    (r"\bmma\b",                         "make me admin"),
    (r"\bcompany portal\b",              "company portal intune"),
    (r"\bwin update\b",                  "windows update"),
    (r"\bwindows update fail\b",         "windows update failing"),
    (r"\bslow laptop\b",                 "laptop running slow"),
    (r"\bslow pc\b",                     "computer running slow"),
    (r"\bprinter not working\b",         "printer issue"),
    (r"\bprint issue\b",                 "printer issue"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Noise words — strip these from ticket text before embedding
# They carry no routing signal and dilute the embedding
# ─────────────────────────────────────────────────────────────────────────────
_NOISE_WORDS = {
    "please", "kindly", "urgent", "urgently", "asap", "hi", "hello",
    "dear", "team", "greetings", "thanks", "thank", "you", "regards",
    "sincerely", "help", "need", "assistance", "issue", "problem",
    "request", "ticket", "noted", "attached", "screenshot",
    "following", "below", "above", "morning", "afternoon", "evening",
}


class PreprocessingAgent:

    def process(self, text: str) -> str:
        """Clean and normalise input text for embedding."""
        if not text:
            return ""

        text = text.strip()

        # Lowercase early so all regex matches are case-insensitive
        text = text.lower()

        # Remove ticket number patterns (INC, REQ, CHG, PRB, TASK, TKT)
        text = re.sub(
            r"\b(inc|req|chg|prb|task|tkt|sr|sd|cs)\d+\b", "", text
        )

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Apply IT synonym normalisation
        for pattern, replacement in _SYNONYM_MAP:
            text = re.sub(pattern, replacement, text)

        # Remove excessive punctuation but keep meaningful ones
        text = re.sub(r"[^\w\s\-/\.,]", " ", text)

        # Remove noise words
        words   = text.split()
        cleaned = [w for w in words if w not in _NOISE_WORDS and len(w) > 1]
        text    = " ".join(cleaned)

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
