"""Hybrid search — merges vector + BM25, deduplicates, then reranks.

Pipeline:
    1. Retrieve top-N from vector store + BM25
    2. Merge and deduplicate (content-hash based)
    3. Optional: cross-encoder reranker scores all candidates
    4. Return full ranked list with scores (caller picks top-k)
"""

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines vector (semantic) and BM25 (keyword) search with optional reranking.

    The pipeline is always:
        retrieve → merge + dedup → (optionally) rerank → return ranked
    """

    def __init__(self, vector_store, bm25_index, reranker: "CrossEncoderReranker | None" = None):
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.reranker = reranker

    def set_reranker(self, reranker: "CrossEncoderReranker") -> None:
        self.reranker = reranker

    def search(
        self,
        query: str,
        collection: str = "knowledge_docs",
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
        top_k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Hybrid search.

        1. Retrieve generously from both vector and BM25
        2. Merge + deduplicate (content-hash based)
        3. If reranker available: score all candidates with cross-encoder
        4. Return top_k

        Each result dict includes:
            - score / weighted_score (original retrieval score)
            - rerank_score (cross-encoder score, if reranked)
            - rerank_rank (position after reranking)
            - source / source_rank (where it came from)
        """
        # Retrieve generously — more candidates means better reranking
        fetch_k = max(top_k * 5, 50) if (rerank and self.reranker is not None) else top_k * 3

        vector_results = self.vector_store.search(
            collection=collection,
            query=query,
            filters=filters,
            time_range=time_range,
            top_k=fetch_k,
        )
        keyword_results = self.bm25.search(query, top_k=fetch_k)

        # Tag weighted scores
        for r in vector_results:
            r["weighted_score"] = r.get("score", 0) * vector_weight
        for r in keyword_results:
            r["weighted_score"] = r.get("score", 0) * keyword_weight

        # Step 1: Merge + deduplicate
        from app.retrieval.reranker import merge_and_dedup
        merged = merge_and_dedup(vector_results, keyword_results)

        if not merged:
            return []

        # Step 2: Cross-encoder reranker
        if rerank and self.reranker is not None:
            merged = self.reranker.score(query, merged)

        # Step 3: Return top_k
        return merged[:top_k]
