"""
Ticket Ingestion Agent
----------------------
Polls ServiceNow for new, unassigned tickets and normalizes the payload
for downstream agents.
"""

import logging
import requests
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)


class TicketIngestionAgent:
    """
    Polls ServiceNow REST API for unassigned incidents.
    """

    TICKET_FIELDS = [
        "sys_id",
        "number",
        "short_description",
        "description",
        "category",
        "subcategory",
        "business_service",
        "priority",
        "state",
        "assignment_group",
        "opened_at",
        "caller_id",
    ]

    def __init__(self, config: dict):
        self.base_url = config["servicenow"]["instance_url"].rstrip("/")
        self.auth = (
            config["servicenow"]["username"],
            config["servicenow"]["password"],
        )
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def fetch_unassigned_tickets(self, limit: int = 10) -> list[dict]:
        """
        Returns a list of unassigned incident records from ServiceNow.
        """
        url = f"{self.base_url}/api/now/table/incident"
        params = {
            "sysparm_query": "assignment_groupISEMPTY^state=1",  # state=1 = New
            "sysparm_limit": limit,
            "sysparm_fields": ",".join(self.TICKET_FIELDS),
            "sysparm_display_value": "true",
        }

        try:
            response = requests.get(
                url, auth=self.auth, headers=self.headers, params=params, timeout=30
            )
            response.raise_for_status()
            tickets = response.json().get("result", [])
            logger.info(f"Fetched {len(tickets)} unassigned ticket(s).")
            return tickets

        except RequestException as e:
            logger.error(f"Failed to fetch tickets from ServiceNow: {e}")
            return []

    def normalize_ticket(self, ticket: dict) -> dict:
        """
        Normalizes a raw ServiceNow ticket into a clean dict.
        """
        return {
            "sys_id": ticket.get("sys_id", ""),
            "number": ticket.get("number", ""),
            "short_description": ticket.get("short_description", "") or "",
            "description": ticket.get("description", "") or "",
            "category": ticket.get("category", "") or "",
            "subcategory": ticket.get("subcategory", "") or "",
            "business_service": ticket.get("business_service", {}).get("display_value", "") if isinstance(ticket.get("business_service"), dict) else ticket.get("business_service", "") or "",
            "priority": ticket.get("priority", "") or "",
        }
