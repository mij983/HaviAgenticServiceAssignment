"""
Historical Data Agent
----------------------
Converts ServiceNow ticket fields into numeric feature vectors.

HOW ROUTING WORKS:
  Assignment group is decided by keywords found in SHORT DESCRIPTION
  and DESCRIPTION only. Category/subcategory are identical across all 50
  tickets ("Applications and Software / Business Application") so they add
  zero routing value. Configuration Item is ignored as requested.

  Short description keywords are counted TWICE (2x weight) because:
    - It is always filled — description can be vague or empty
    - It is the engineer's primary intent in one line
    - Real tickets like "DIMS", "ILOS", "VIM" have the full signal there

KEYWORD SETS:
  Derived by reading every SD + Description in the CSV and identifying
  words that uniquely appear in each team's tickets.
"""

import csv
import hashlib
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def stable_hash(value: str, mod: int = 100) -> int:
    if not value:
        return 0
    return int(hashlib.md5(value.lower().encode()).hexdigest(), 16) % mod


def kw_score(text: str, keywords: set) -> int:
    """Count how many keywords from the set appear in text. Cap at 5."""
    t = text.lower()
    return min(sum(1 for kw in keywords if kw in t), 5)


# ─────────────────────────────────────────────────────────────────────────────
# Keywords derived directly by reading SD + Description of all 50 CSV tickets.
# Each set contains words that ONLY appear in that team's tickets.
# ─────────────────────────────────────────────────────────────────────────────
TEAM_KEYWORDS: dict[str, set] = {

    # Tickets 1-10: INFOR WMS + Kisoft issues
    # Unique words from SD+DESC: infor, kisoft, knapp, systore, wms,
    #   route, routes, shipment, load, crossdock, picking, allocat,
    #   sscc, panel weight, blocking, drucken, p-loc, xml message
    "IT-SC Operations Application Support": {
        "infor", "kisoft", "knapp", "systore", "wms",
        "route", "routes", "shipment", "load closed", "crossdock",
        "picking", "sscc", "panel weight", "blocking 3216",
        "drucken", "p-loc", "xml message", "replenish",
        "rocio", "scale fusion", "hpt-500",
    },

    # Tickets 11-20: DIMS + ILOS issues
    # Unique words: dims, ilos, plaridel, cabuyao, philippines,
    #   wareneingang, goods receipt, boomi, startload, employee signature,
    #   loadid, sales order, client 170
    # NOTE: tickets 18,20 mention "infor" and "ilos" — but their primary
    # signal is "ilos" in sd or desc, which scores here too.
    "IT-SC GBS App Support": {
        "dims", "ilos",
        "plaridel", "cabuyao", "philippines",
        "wareneingang", "goods receipt", "employee signature",
        "boomi", "startload", "loadid",
        "client 170", "client 1700", "tour 1542",
    },

    # Tickets 21-25: SOA application issues
    # Unique words: soa, dn number, tio, ti2-, mcddn, item store delivery,
    #   daily sales report, delivery note, print dn
    "IT-Asia-SOA-Central-App-Support": {
        "soa",
        "dn number", "tio", "ti2-", "mcddn",
        "item store delivery", "daily sales report",
        "delivery note", "print dn", "download pdf",
        "exclude item", "reference number",
    },

    # Tickets 26-30: VIM invoice approval workflow
    # Unique words: vim, workflow, approver, ic4s, ap team,
    #   invoices blocked, sehr langsam, id 3314851
    "IT-SC-EPAM-SAP Workflow": {
        "vim",
        "workflow", "approver", "approve",
        "ic4s", "ap team",
        "invoices blocked", "sehr langsam",
        "id 3314851", "invoices are not coming",
    },

    # Tickets 31-35: SAP ZEDI/AMS issues
    # Unique words: sap, zedi, archiva, b2bi, vf31, sap pc4,
    #   mass invoice, mass upload, shell price, debtor
    # NOTE: ticket 35 "Bitte Ticket in SAP C4C" — SAP keyword routes to AMS
    # which matches CSV (team = IT-SC-EPAM-SAP-AMS-Support)
    "IT-SC-EPAM-SAP-AMS-Support": {
        "sap",
        "zedi", "archiva", "b2bi", "vf31", "sap pc4",
        "mass invoice", "mass invoices",
        "mass upload", "shell price", "debtor",
        "no zedi", "edi output",
    },

    # Tickets 36-40: C4C / Hybris CRM sensitive data + CRM issues
    # Unique words: c4c, c4h, entpersonalisieren, sensitive data,
    #   delete, remove, active user report, last logon, errands, nordic
    # NOTE: ticket 36 SD = "Bitte Ticket in SAP C4C - Ticket Nr. 7813759
    # entpersonalisieren" — "entpersonalisieren" + "c4c" route here (not AMS)
    "IT-SC-EPAM-SAP Hybris CRM Support": {
        "c4c", "c4h", "crm", "hybris",
        "entpersonalisieren", "sensitive data",
        "delete", "completely",
        "active user report", "last logon",
        "errands", "nordic", "nordics",
    },

    # Tickets 41-45: HaviConnect portal issues
    # Unique words: haviconnect, havi connect, restaurant, claims, claim+,
    #   order proposal, zabok, portal, mcd slo, password havi
    "IT-Portal-Central": {
        "haviconnect", "havi connect",
        "restaurant", "claims", "claim +", "claims+",
        "order proposal", "zabok", "mcd slo",
        "password havi", "portal",
    },

    # Tickets 46-50: PMIX/JDA/PIR forecasting data issues
    # Unique words: pmix, jda, pir, hist, smi, bmi, cfm,
    #   missing pmix, pl-gc, hlpl, store 248, portugal market
    "IT-SCM-EPAM L2": {
        "pmix", "jda", "pir", "hist",
        "smi", "bmi", "cfm",
        "missing pmix", "pl-gc", "hlpl",
        "store 248", "portugal market",
    },
}

ALL_TEAMS = list(TEAM_KEYWORDS.keys())


class HistoricalDataAgent:

    PRIORITY_MAP = {
        "1 - Critical": 5, "1": 5,
        "2 - High": 4,     "2": 4,
        "3 - Moderate": 3, "3": 3,
        "4 - Low": 2,      "4": 2,
        "5 - Planning": 1, "5": 1,
        "": 0,
    }

    def build_features(self, ticket: dict) -> list:
        """
        Feature vector (length = 7 + 8*3 = 31):

          [0]    short_description character length
          [1]    description character length
          [2]    combined length
          [3]    stable_hash(category)
          [4]    stable_hash(subcategory)
          [5]    stable_hash(business_service)
          [6]    numeric priority

          [7-14]   keyword hits per team in SHORT DESC  (8 scores, weight x2)
          [15-22]  same scores AGAIN                    (2x weight via dupe)
          [23-30]  keyword hits per team in DESCRIPTION (8 scores, weight x1)
        """
        sd   = ticket.get("short_description", "") or ""
        desc = ticket.get("description", "")        or ""
        cat  = ticket.get("category", "")           or ""
        sub  = ticket.get("subcategory", "")        or ""
        biz  = ticket.get("business_service", "")   or ""
        pri  = ticket.get("priority", "")           or ""

        sd_scores   = [kw_score(sd,   kws) for kws in TEAM_KEYWORDS.values()]
        desc_scores = [kw_score(desc, kws) for kws in TEAM_KEYWORDS.values()]

        return [
            len(sd),
            len(desc),
            len(sd) + len(desc),
            stable_hash(cat),
            stable_hash(sub),
            stable_hash(biz),
            self.PRIORITY_MAP.get(pri, 0),
            # SD scores x2 (higher weight — SD is primary signal)
            *sd_scores,
            *sd_scores,
            # Description scores x1
            *desc_scores,
        ]

    def load_historical_csv(self, csv_path, label_column="Assignment Team"):
        """Load training data from CSV export. Configuration Item ignored."""
        X, y = [], []
        try:
            with open(csv_path, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    label = row.get(label_column, "").strip()
                    if not label:
                        continue
                    ticket = {
                        "short_description": row.get("Short Description", ""),
                        "description":       row.get("Description", ""),
                        "category":          row.get("Category", ""),
                        "subcategory":       row.get("SubCategory", ""),
                        "priority":          row.get("Priority", ""),
                        # Configuration Item intentionally excluded
                    }
                    X.append(self.build_features(ticket))
                    y.append(label)
            if not X:
                logger.error("No valid rows in CSV.")
                return None
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"CSV load error: {e}")
            return None
