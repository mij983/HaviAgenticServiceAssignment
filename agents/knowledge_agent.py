import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

FALLBACK_KNOWLEDGE = {
    "active_assignment_groups": [
        "IT-SC Operations Application Support",
        "IT-SC GBS App Support",
        "IT-Asia-SOA-Central-App-Support",
        "IT-SC-EPAM-SAP Workflow",
        "IT-SC-EPAM-SAP-AMS-Support",
        "IT-SC-EPAM-SAP Hybris CRM Support",
        "IT-Portal-Central",
        "IT-SCM-EPAM L2",
    ],
    "deprecated_mapping": {},
}

LOCAL_JSON_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "assignment_groups.json"
)


class KnowledgeAgent:

    def __init__(self, config: dict):
        self.config = config.get("azure_blob", {})
        self._active_groups: Optional[list] = None
        self._deprecated_mapping: Optional[dict] = None

    def load_knowledge(self) -> tuple[list, dict]:
        if self._active_groups is not None:
            return self._active_groups, self._deprecated_mapping

        content = self._fetch()
        self._active_groups = (
            content.get("active_assignment_groups")
            or content.get("active_groups")
            or []
        )
        self._deprecated_mapping = content.get("deprecated_mapping", {})
        logger.info("Loaded " + str(len(self._active_groups)) + " active assignment group(s).")
        return self._active_groups, self._deprecated_mapping

    def refresh(self):
        self._active_groups = None
        self._deprecated_mapping = None

    def _fetch(self) -> dict:
        connection_string = self.config.get("connection_string", "")
        if AZURE_AVAILABLE and connection_string and connection_string not in ("", "AZURE_BLOB_CONNECTION_STRING"):
            try:
                blob_service = BlobServiceClient.from_connection_string(connection_string)
                container_client = blob_service.get_container_client(self.config["container_name"])
                blob_client = container_client.get_blob_client(self.config["blob_name"])
                raw_data = blob_client.download_blob().readall()
                logger.info("Loaded knowledge from Azure Blob Storage.")
                return json.loads(raw_data)
            except Exception as e:
                logger.error("Azure Blob failed: " + str(e) + ". Trying local file.")

        if os.path.exists(LOCAL_JSON_PATH):
            try:
                with open(LOCAL_JSON_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                logger.info("Loaded knowledge from local file: " + LOCAL_JSON_PATH)
                return data
            except Exception as e:
                logger.error("Local JSON failed: " + str(e) + ". Using hardcoded fallback.")

        logger.warning("Using hardcoded fallback knowledge.")
        return FALLBACK_KNOWLEDGE

    def resolve_deprecated(self, group_name: str) -> str:
        _, deprecated_mapping = self.load_knowledge()
        return deprecated_mapping.get(group_name, group_name)

    def is_active(self, group_name: str) -> bool:
        active_groups, _ = self.load_knowledge()
        return group_name in active_groups
