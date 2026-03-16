"""
tests/test_agents.py
---------------------
Unit tests for each agent.  Run with:  pytest tests/
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.historical_data_agent import HistoricalDataAgent, stable_hash
from agents.confidence_engine import ConfidenceScoringEngine
from agents.decision_agent import DecisionAgent
from agents.learning_agent import LearningAgent
from agents.ingestion_agent import TicketIngestionAgent
from agents.knowledge_agent import KnowledgeAgent


# ===========================================================================
# HistoricalDataAgent
# ===========================================================================

class TestHistoricalDataAgent:
    def setup_method(self):
        self.agent = HistoricalDataAgent()

    def test_build_features_returns_list_of_11(self):
        ticket = {
            "short_description": "VPN not working",
            "description": "Cannot connect to corporate VPN",
            "category": "Network",
            "subcategory": "VPN",
            "business_service": "IT Services",
            "priority": "2 - High",
        }
        features = self.agent.build_features(ticket)
        assert isinstance(features, list)
        assert len(features) == 11

    def test_build_features_empty_ticket(self):
        features = self.agent.build_features({})
        assert len(features) == 11
        assert all(isinstance(f, (int, float)) for f in features)

    def test_stable_hash_deterministic(self):
        assert stable_hash("Network") == stable_hash("Network")
        assert stable_hash("VPN") != stable_hash("Application")

    def test_load_historical_csv(self, tmp_path):
        csv_file = tmp_path / "tickets.csv"
        csv_file.write_text(
            "short_description,description,category,subcategory,"
            "business_service,priority,assignment_group\n"
            "VPN issue,Cannot connect,Network,VPN,IT,2 - High,Network Support\n"
            "App crash,App not starting,Software,Error,Apps,3 - Moderate,Application Support\n"
        )
        result = self.agent.load_historical_csv(str(csv_file))
        assert result is not None
        X, y = result
        assert X.shape == (2, 11)
        assert list(y) == ["Network Support", "Application Support"]

    def test_load_csv_missing_file(self):
        result = self.agent.load_historical_csv("nonexistent.csv")
        assert result is None


# ===========================================================================
# ConfidenceScoringEngine
# ===========================================================================

class TestConfidenceScoringEngine:
    def setup_method(self):
        self.engine = ConfidenceScoringEngine()
        self.active_groups = ["Network Support", "Application Support", "Cloud Operations"]

    def test_high_confidence_network(self):
        ticket = {"short_description": "VPN connectivity issue", "description": "Cannot connect to VPN", "category": "Network", "subcategory": ""}
        score = self.engine.calculate(0.95, ticket, "Network Support", self.active_groups)
        assert score > 7.0

    def test_low_confidence_inactive_group(self):
        ticket = {"short_description": "Something", "description": "", "category": "", "subcategory": ""}
        score = self.engine.calculate(0.1, ticket, "Unknown Group", self.active_groups)
        assert score < 5.0

    def test_score_range(self):
        ticket = {"short_description": "test", "description": "test", "category": "", "subcategory": ""}
        for prob in [0.0, 0.5, 1.0]:
            score = self.engine.calculate(prob, ticket, "Network Support", self.active_groups)
            assert 1.0 <= score <= 10.0

    def test_weight_sum_validation(self):
        with pytest.raises(AssertionError):
            ConfidenceScoringEngine(historical_weight=0.5, text_weight=0.5, knowledge_weight=0.5)


# ===========================================================================
# DecisionAgent
# ===========================================================================

class TestDecisionAgent:
    def setup_method(self):
        self.agent = DecisionAgent(threshold=7.0)

    def test_auto_assign_above_threshold(self):
        assert self.agent.should_auto_assign(7.5) is True

    def test_no_auto_assign_at_threshold(self):
        assert self.agent.should_auto_assign(7.0) is False

    def test_no_auto_assign_below_threshold(self):
        assert self.agent.should_auto_assign(5.0) is False

    def test_decide_auto_assign(self):
        ticket = {"sys_id": "abc", "number": "INC001"}
        result = self.agent.decide(ticket, "Network Support", 8.0, True)
        assert result["auto_assign"] is True
        assert result["group"] == "Network Support"

    def test_decide_inactive_group(self):
        ticket = {}
        result = self.agent.decide(ticket, "Old Group", 9.5, False)
        assert result["auto_assign"] is False
        assert "not active" in result["reason"]

    def test_decide_low_confidence(self):
        ticket = {}
        result = self.agent.decide(ticket, "Network Support", 4.0, True)
        assert result["auto_assign"] is False
        assert "manual triage" in result["reason"].lower()


# ===========================================================================
# LearningAgent
# ===========================================================================

class TestLearningAgent:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_feedback.db")
        self.model_path = os.path.join(self.tmpdir, "model.pkl")
        self.agent = LearningAgent(self.db_path, self.model_path)

    def test_store_feedback(self):
        self.agent.store_feedback(
            ticket_number="INC001",
            features=[1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 5],
            predicted_group="Network Support",
            final_group="Network Support",
            confidence=8.5,
        )
        report = self.agent.accuracy_report()
        assert report["total"] == 1
        assert report["accuracy"] == 1.0

    def test_accuracy_report_wrong_prediction(self):
        self.agent.store_feedback("INC002", [1]*11, "Network Support", "Application Support", 6.0)
        report = self.agent.accuracy_report()
        assert report["accuracy"] == 0.0

    def test_retrain_requires_minimum_samples(self):
        # Less than 10 rows → should not retrain
        for i in range(5):
            self.agent.store_feedback(f"INC{i}", [i]*11, "Network Support", "Network Support", 8.0)
        result = self.agent.retrain_model()
        assert result is False

    def test_log_decision(self):
        self.agent.log_decision("INC001", "abc123", "Network Support", 8.5, True, "High confidence", [])
        # Should not raise


# ===========================================================================
# KnowledgeAgent (fallback mode)
# ===========================================================================

class TestKnowledgeAgentFallback:
    def test_fallback_returns_defaults(self):
        # Deliberately set a blank connection string to trigger fallback
        config = {
            "azure_blob": {
                "connection_string": "",
                "container_name": "test",
                "blob_name": "test.json",
            }
        }
        agent = KnowledgeAgent(config)
        active, deprecated = agent.load_knowledge()
        assert "Network Support" in active
        assert "Application Support" in active
        assert isinstance(deprecated, dict)

    def test_resolve_deprecated(self):
        config = {"azure_blob": {"connection_string": "", "container_name": "", "blob_name": ""}}
        agent = KnowledgeAgent(config)
        agent.load_knowledge()
        assert agent.resolve_deprecated("Legacy Network Team") == "Network Support"
        assert agent.resolve_deprecated("Network Support") == "Network Support"

    def test_is_active(self):
        config = {"azure_blob": {"connection_string": "", "container_name": "", "blob_name": ""}}
        agent = KnowledgeAgent(config)
        assert agent.is_active("Network Support") is True
        assert agent.is_active("Ghost Group") is False
