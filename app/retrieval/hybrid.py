"""Hybrid search — merges vector + BM25 via Reciprocal Rank Fusion, deduplicates, then reranks.

Pipeline:
    1. Retrieve top-N from vector store + BM25
    2. Reciprocal Rank Fusion (RRF) to merge ranked lists
    3. Deduplicate (content-hash based)
    4. Optional: cross-encoder reranker scores candidates (top 50 → top 5)
    5. Return full ranked list with scores (caller picks top-k)
"""

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
    k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Merge two ranked result lists using Reciprocal Rank Fusion.

    RRF formula: score(d) = Σ(1 / (k + rank_i(d)))

    where rank_i(d) is the 1-based rank of document d in list i.
    Documents appearing in only one list still get a score from that list.
    Documents appearing in both lists get a higher fused score.

    Args:
        vector_results: List of result dicts from vector search, already ranked by relevance.
        keyword_results: List of result dicts from BM25 search, already ranked by relevance.
        k: RRF constant (default 60). Prevents dominance by top-ranked items.

    Returns:
        List of unique result dicts sorted by RRF score descending,
        each with 'rrf_score' and 'rrf_rank' added.
    """
    # Build a map from doc identifier -> {doc, ranks}
    docs: dict[str, dict] = {}

    def _key(item: dict[str, Any]) -> str:
        # Prefer stable ID, fallback to content digest
        doc_id = item.get("id", "")
        content = item.get("content", "")[:300].strip()
        return doc_id if doc_id else content

    for rank, item in enumerate(vector_results, start=1):
        key = _key(item)
        if key not in docs:
            docs[key] = {"item": dict(item), "ranks": {}}
        docs[key]["ranks"]["vector"] = rank

    for rank, item in enumerate(keyword_results, start=1):
        key = _key(item)
        if key not in docs:
            docs[key] = {"item": dict(item), "ranks": {}}
        docs[key]["ranks"]["bm25"] = rank

    # Compute RRF scores
    scored = []
    for key, entry in docs.items():
        score = 0.0
        for source, rank in entry["ranks"].items():
            score += 1.0 / (k + rank)
        doc = entry["item"]
        doc["rrf_score"] = round(score, 6)
        doc["rrf_sources"] = list(entry["ranks"].keys())
        scored.append(doc)

    # Sort by RRF score descending
    scored.sort(key=lambda x: x["rrf_score"], reverse=True)

    # Assign RRF ranks
    for i, doc in enumerate(scored, start=1):
        doc["rrf_rank"] = i

    logger.debug(
        "RRF merged %d vector + %d bm25 → %d unique",
        len(vector_results), len(keyword_results), len(scored),
    )
    return scored


class HybridRetriever:
    """Combines vector (semantic) and BM25 (keyword) search with optional reranking.

    The pipeline is always:
        retrieve → RRF merge + dedup → (optionally) rerank → return ranked
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
        rerank: bool = True,
    ) -> list[dict[str, Any]]:
        """Hybrid search.

        1. Retrieve generously from both vector and BM25
        2. Merge via Reciprocal Rank Fusion (RRF)
        3. If reranker available: score top candidates with cross-encoder
        4. Return top_k

        Each result dict includes:
            - rrf_score / rrf_rank (from RRF merging)
            - rerank_score (cross-encoder score, if reranked)
            - rerank_rank (position after reranking)
            - rrf_sources (which retrievers found this doc)
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

        # Step 1: RRF merge + deduplicate
        merged = reciprocal_rank_fusion(vector_results, keyword_results)

        if not merged:
            return []

        # Step 2: Cross-encoder reranker (score top 50, return top_k)
        if rerank and self.reranker is not None:
            # Only rerank the top candidates to stay fast
            candidates = merged[:50]
            reranked = self.reranker.score(query, candidates)
            return reranked[:top_k]

        return merged[:top_k]
