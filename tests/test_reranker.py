"""Tests for the cross-encoder reranker."""

import pytest

from app.retrieval.reranker import CrossEncoderReranker


class TestCrossEncoderReranker:
    def test_rerank_preserves_order_when_no_model(self):
        """Without a loaded model, reranker should return results as-is."""
        reranker = CrossEncoderReranker()
        candidates = [
            {"id": "1", "content": "Database connection pool exhaustion", "score": 0.9},
            {"id": "2", "content": "Redis cluster failover", "score": 0.7},
            {"id": "3", "content": "Some unrelated text", "score": 0.3},
        ]
        result = reranker.rerank("database issues", candidates, top_k=2)
        assert len(result) == 2
        # Without model, identity order (first N)
        assert result[0]["id"] == "1"

    def test_rerank_empty_candidates(self):
        reranker = CrossEncoderReranker()
        assert reranker.rerank("test", []) == []
        assert reranker.rerank("test", [], top_k=5) == []

    def test_rerank_respects_top_k(self):
        reranker = CrossEncoderReranker()
        candidates = [{"id": str(i), "content": f"Content {i}"} for i in range(20)]
        result = reranker.rerank("test", candidates, top_k=5)
        assert len(result) == 5

    def test_rerank_adds_score_key(self):
        reranker = CrossEncoderReranker()
        candidates = [{"id": "1", "content": "Test content"}]
        result = reranker.rerank("test", candidates, top_k=1)
        assert "rerank_score" in result[0]
