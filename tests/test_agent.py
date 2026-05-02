"""Tests for the classifier and reasoning engine."""

import pytest

from app.agents.classifier import QueryClassifier, QueryType


class TestQueryClassifier:
    def setup_method(self):
        self.classifier = QueryClassifier()

    def test_factual_query(self):
        assert self.classifier.classify("Who owns the payment gateway?") == QueryType.FACTUAL
        assert self.classifier.classify("Who is on call for platform team?") == QueryType.FACTUAL

    def test_time_based_query(self):
        assert self.classifier.classify("What incidents happened last week?") == QueryType.TIME_BASED
        assert self.classifier.classify("Show incidents between April 10 and April 15") == QueryType.TIME_BASED
        assert self.classifier.classify("What happened yesterday?") == QueryType.TIME_BASED

    def test_causal_query(self):
        assert self.classifier.classify("What caused the payment gateway outage?") == QueryType.CAUSAL
        assert self.classifier.classify("Why were hosts down?") == QueryType.CAUSAL

    def test_exploratory_query(self):
        assert self.classifier.classify("Show me recent critical incidents") == QueryType.EXPLORATORY
        assert self.classifier.classify("List all major incidents") == QueryType.EXPLORATORY

    def test_relational_query(self):
        assert self.classifier.classify("What services does the platform team own?") == QueryType.RELATIONAL
        assert self.classifier.classify("What hosts belong to payment-gateway?") == QueryType.RELATIONAL

    def test_multi_hop_detection(self):
        assert self.classifier.needs_multi_hop("Why were hosts down between X and Y?") is True
        assert self.classifier.needs_multi_hop("What caused the outage and who was on call?") is True
        assert self.classifier.needs_multi_hop("Who owns service A?") is False

    def test_time_reference_extraction(self):
        refs = self.classifier.extract_time_references("What happened yesterday?")
        assert refs.get("relative") == "24h"

        refs = self.classifier.extract_time_references("Incidents last week")
        assert refs.get("relative") == "7d"

        refs = self.classifier.extract_time_references("From 2026-04-01 to 2026-04-15")
        assert "2026-04-01" in refs.get("dates", [])

    def test_entity_extraction(self):
        entities = self.classifier.extract_entities("What caused the payment-gateway outage?")
        # Proper nouns
        assert len(entities.get("proper_nouns", [])) > 0

        entities = self.classifier.extract_entities("Is web-01.prod healthy?")
        assert "web-01.prod" in str(entities.get("hostnames", []))


class TestQueryTypeValues:
    def test_all_types_defined(self):
        expected = {"factual", "relational", "time_based", "causal", "exploratory", "comparative", "host_status", "multi_hop"}
        actual = {t.value for t in QueryType}
        assert expected == actual
