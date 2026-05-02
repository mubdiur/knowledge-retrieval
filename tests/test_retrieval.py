"""Tests for retrieval components: BM25, merge+dedup, and reranker."""

import pytest

from app.retrieval.keyword_store import BM25Index
from app.retrieval.reranker import merge_and_dedup


class TestBM25Index:
    def test_empty_index(self):
        bm25 = BM25Index()
        assert bm25.search("test") == []
        assert bm25.is_ready is False

    def test_index_and_search(self):
        bm25 = BM25Index()
        docs = [
            {"id": "1", "content": "Payment gateway database connection pool exhaustion"},
            {"id": "2", "content": "Redis cluster failover caused high latency"},
            {"id": "3", "content": "Search service Elasticsearch disk space full"},
            {"id": "4", "content": "Notification SES rate limit exceeded"},
            {"id": "5", "content": "Database connection limits and pooling strategies"},
        ]
        bm25.index(docs)
        assert bm25.is_ready is True
        assert bm25.doc_count == 5

    def test_relevant_results(self):
        bm25 = BM25Index()
        docs = [
            {"id": "1", "content": "Payment gateway database connection pool exhaustion"},
            {"id": "2", "content": "Redis cluster failover caused high latency"},
            {"id": "3", "content": "Database query optimization for PostgreSQL"},
        ]
        bm25.index(docs)
        results = bm25.search("database connection pool")
        assert len(results) > 0
        assert results[0]["id"] == "1"

    def test_no_match(self):
        bm25 = BM25Index()
        docs = [{"id": "1", "content": "Payment processing and fraud detection"}]
        bm25.index(docs)
        results = bm25.search("kubernetes deployment strategy")
        assert len(results) == 0


class TestMergeAndDedup:
    def test_merge_basic(self):
        vector = [
            {"id": "A", "content": "Database pool", "score": 0.9},
            {"id": "B", "content": "Redis failover", "score": 0.7},
        ]
        keyword = [
            {"id": "C", "content": "Connection pool", "score": 5.2},
            {"id": "D", "content": "New item", "score": 3.1},
        ]
        merged = merge_and_dedup(vector, keyword)
        assert len(merged) == 4

    def test_dedup_by_content(self):
        vector = [
            {"id": "A", "content": "Same content here"},
        ]
        keyword = [
            {"id": "B", "content": "Same content here"},  # Same content, different ID
        ]
        merged = merge_and_dedup(vector, keyword)
        assert len(merged) == 1  # Deduplicated
        assert merged[0]["source_rank"] == "vector"  # Vector wins

    def test_dedup_by_id(self):
        vector = [{"id": "same_id", "content": "Content A"}]
        keyword = [{"id": "same_id", "content": "Content B"}]
        merged = merge_and_dedup(vector, keyword)
        assert len(merged) == 1

    def test_empty_lists(self):
        assert merge_and_dedup([], []) == []
        assert len(merge_and_dedup([{"id": "a", "content": "x"}], [])) == 1
        assert len(merge_and_dedup([], [{"id": "a", "content": "x"}])) == 1

    def test_all_duplicates(self):
        vector = [{"id": "A", "content": "Only item"}]
        keyword = [{"id": "B", "content": "Only item"}]
        assert len(merge_and_dedup(vector, keyword)) == 1
