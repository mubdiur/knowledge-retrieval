"""Cross-encoder reranker — re-ranks retrieval candidates for precision.

Uses a cross-encoder model (e.g. BAAI/bge-reranker-v2-m3 or
cross-encoder/ms-marco-MiniLM-L-6-v2) to score query-document pairs jointly,
producing more accurate relevance than embedding cosine similarity.

The pipeline is:

    vector results + BM25 results
        → merge + deduplicate
        → cross-encoder reranker
        → all results with scores (caller picks top-k)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def merge_and_dedup(
    vector_results: list[dict[str, Any]],
    keyword_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge two ranked lists, deduplicating by content digest.

    Uses content hash for dedup so identical docs from vector + BM25
    are not double-counted. The higher-ranked entry (from vector, which
    is typically more relevant) wins ties.
    """
    seen_content: set[str] = set()
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []

    def is_duplicate(item: dict[str, Any]) -> bool:
        content = item.get("content", "")[:300].strip()
        doc_id = item.get("id", "")
        if content and content in seen_content:
            return True
        if doc_id and doc_id in seen_ids:
            return True
        if content:
            seen_content.add(content)
        if doc_id:
            seen_ids.add(doc_id)
        return False

    # Vector results first (higher confidence)
    for item in vector_results:
        if not is_duplicate(item):
            item["source_rank"] = "vector"
            merged.append(item)

    # Then BM25 results
    for item in keyword_results:
        if not is_duplicate(item):
            item["source_rank"] = "bm25"
            merged.append(item)

    logger.debug("Merged %d vector + %d bm25 → %d unique", len(vector_results), len(keyword_results), len(merged))
    return merged


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        merged = merge_and_dedup(vector_results, keyword_results)
        all_scored = reranker.score(query, merged)       # all with rerank_score
        top_k = all_scored[:top_k]                        # caller picks k
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or "BAAI/bge-reranker-v2-m3"
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Cross-encoder loaded")
        except Exception as e:
            logger.warning(
                "Failed to load cross-encoder '%s': %s. "
                "Falling back to identity scoring (no reranking).",
                self.model_name, e,
            )
            self._model = None

    def score(
        self,
        query: str,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Score all candidates by cross-encoder relevance.

        Returns the FULL list of candidates with 'rerank_score' attached,
        sorted by score descending. Caller decides top-k.

        Args:
            query: The search query.
            candidates: List of dicts, each with at least {'content': str, ...}.

        Returns:
            All candidates sorted by relevance (highest first), each with
            'rerank_score' (float) and 'rerank_rank' (int) added.
        """
        if not candidates:
            return []

        self._load_model()

        if self._model is None:
            # No model — assign identity scores based on position
            for i, c in enumerate(candidates):
                c["rerank_score"] = 1.0 / (i + 1)
                c["rerank_rank"] = i + 1
            return candidates

        # Build query-document pairs with truncation
        pairs = []
        for c in candidates:
            content = c.get("content", "")
            if len(content) > 2000:
                content = content[:2000]
            pairs.append((query, content))

        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.error("Reranker prediction failed: %s", e)
            for i, c in enumerate(candidates):
                c["rerank_score"] = 1.0 / (i + 1)
                c["rerank_rank"] = i + 1
            return candidates

        # Attach scores
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        # Sort by score descending
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Assign ranks
        for i, c in enumerate(candidates):
            c["rerank_rank"] = i + 1

        return candidates

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Convenience: score then return top-k.

        Prefer score() when you want the full ranked list.
        """
        all_scored = self.score(query, candidates)
        if min_score is not None:
            all_scored = [c for c in all_scored if c["rerank_score"] >= min_score]
        return all_scored[:top_k]


# Singleton for app-wide use
_shared_reranker: CrossEncoderReranker | None = None


def get_reranker(model_name: str | None = None) -> CrossEncoderReranker:
    global _shared_reranker
    if _shared_reranker is None:
        _shared_reranker = CrossEncoderReranker(model_name)
    return _shared_reranker
