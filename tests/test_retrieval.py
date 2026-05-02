"""Tests for retrieval components: BM25, hybrid fusion, and vector store helpers."""

import pytest

from app.retrieval.keyword_store import BM25Index
from app.retrieval.hybrid import reciprocal_rank_fusion


class TestBM25Index:
    def setup_method(self):
        self.bm25 = BM25Index()

    def test_empty_index(self):
        assert self.bm25.search("test") == []
        assert self.bm25.is_ready is False

    def test_index_and_search(self):
        docs = [
            {"id": "1", "content": "Payment gateway database connection pool exhaustion"},
            {"id": "2", "content": "Redis cluster failover caused high latency"},
            {"id": "3", "content": "Search service Elasticsearch disk space full"},
            {"id": "4", "content": "Notification SES rate limit exceeded"},
            {"id": "5", "content": "Database connection limits and pooling strategies"},
        ]
        self.bm25.index(docs)
        assert self.bm25.is_ready is True
        assert self.bm25.doc_count == 5

    def test_relevant_results(self):
        docs = [
            {"id": "1", "content": "Payment gateway database connection pool exhaustion"},
            {"id": "2", "content": "Redis cluster failover caused high latency"},
            {"id": "3", "content": "Database query optimization for PostgreSQL"},
        ]
        self.bm25.index(docs)
        results = self.bm25.search("database connection pool")
        assert len(results) > 0
        # Doc 1 should be most relevant (contains "database", "connection", "pool")
        assert results[0]["id"] == "1"

    def test_no_match(self):
        docs = [{"id": "1", "content": "Payment processing and fraud detection"}]
        self.bm25.index(docs)
        results = self.bm25.search("kubernetes deployment strategy")
        assert len(results) == 0


class TestHybridFusion:
    def test_rrf_basic(self):
        vector_results = [
            {"id": "A", "content": "Database pool", "score": 0.9, "source": "vector"},
            {"id": "B", "content": "Redis failover", "score": 0.7, "source": "vector"},
        ]
        keyword_results = [
            {"id": "C", "content": "Connection pool", "score": 5.2, "source": "bm25"},
            {"id": "B", "content": "Redis failover", "score": 3.1, "source": "bm25"},
        ]
        fused = reciprocal_rank_fusion(vector_results, keyword_results, k=60, top_k=3)
        assert len(fused) == 3
        # B appears in both, should rank highest
        assert fused[0]["id"] == "B"

    def test_rrf_empty_lists(self):
        assert reciprocal_rank_fusion([], [], top_k=5) == []

    def test_rrf_single_source(self):
        vector = [{"id": "A", "content": "test"}]
        fused = reciprocal_rank_fusion(vector, [], top_k=5)
        assert len(fused) == 1
        assert fused[0]["id"] == "A"
