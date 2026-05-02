"""Hybrid search — fuses vector + BM25 results with RFF score fusion, then optionally reranks."""

import logging
from collections import defaultdict
from typing import Any, TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:
    from app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)
settings = get_settings()


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Fuse two ranked result lists using RRF.

    RRF score = sum(1 / (k + rank)) for each item across result sets.
    Higher k = less influence of ranking position.
    """
    fusion_scores: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(vector_results):
        key = item.get("id", item.get("content", "")[:100])
        fusion_scores.setdefault(key, {**item, "rrf_score": 0.0})
        fusion_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, item in enumerate(keyword_results):
        key = item.get("id", item.get("content", "")[:100])
        fusion_scores.setdefault(key, {**item, "rrf_score": 0.0})
        fusion_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(fusion_scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_items[:top_k]


class HybridRetriever:
    """Combines vector (semantic) and BM25 (keyword) search with optional reranking."""

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
        """Hybrid search with optional cross-encoder reranking.

        When rerank=True and a reranker is available:
            1. Retrieve top_k*5 from vector + top_k*5 from BM25
            2. Fuse with RRF → keep top_k*3
            3. Cross-encoder rerank → keep top_k
        Otherwise, standard RRF fusion.
        """
        # Retrieve generously for reranking
        fetch_k = top_k * 5 if (rerank and self.reranker is not None) else top_k * 2

        vector_results = self.vector_store.search(
            collection=collection,
            query=query,
            filters=filters,
            time_range=time_range,
            top_k=fetch_k,
        )
        keyword_results = self.bm25.search(query, top_k=fetch_k)

        for r in vector_results:
            r["weighted_score"] = r.get("score", 0) * vector_weight
        for r in keyword_results:
            r["weighted_score"] = r.get("score", 0) * keyword_weight

        # RRF fusion — keep more candidates when reranking
        fused_top_k = top_k * 3 if (rerank and self.reranker is not None) else top_k
        fused = reciprocal_rank_fusion(vector_results, keyword_results, top_k=fused_top_k)

        # Cross-encoder rerank
        if rerank and self.reranker is not None:
            fused = self.reranker.rerank(query, fused, top_k=top_k)

        return fused
