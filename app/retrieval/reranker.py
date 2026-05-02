"""Cross-encoder reranker — re-ranks retrieval candidates for precision.

Uses a cross-encoder model (e.g. BAAI/bge-reranker-v2-m3 or cross-encoder/ms-marco-MiniLM-L-6-v2)
to score query-document pairs jointly, producing more accurate relevance than embedding cosine similarity.

This is the single highest-leverage quality improvement for any retrieval system.
"""

import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """Reranks retrieval results using a cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        results = retriever.search(query, top_k=50)
        reranked = reranker.rerank(query, results, top_k=5)
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
                "Falling back to identity rerank (no reranking). "
                "Install with: pip install sentence-transformers",
                self.model_name, e,
            )
            self._model = None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank candidate documents by cross-encoder relevance.

        Args:
            query: The search query.
            candidates: List of dicts, each with at least {'content': str, ...}.
            top_k: Number of results to keep after reranking.
            min_score: Optional minimum score threshold (0-1 range, model-specific).

        Returns:
            Candidates sorted by relevance (highest first), with 'rerank_score' added.
        """
        if not candidates:
            return []

        self._load_model()

        if self._model is None:
            # No model available — return as-is with identity score
            for i, c in enumerate(candidates):
                c["rerank_score"] = c.get("score", 1.0 / (i + 1))
            return candidates[:top_k]

        # Build query-document pairs
        pairs = []
        for c in candidates:
            content = c.get("content", "")
            # Truncate to avoid blowing the model's max length
            if len(content) > 2000:
                content = content[:2000]
            pairs.append((query, content))

        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.error("Reranker prediction failed: %s", e)
            for i, c in enumerate(candidates):
                c["rerank_score"] = c.get("score", 1.0 / (i + 1))
            return candidates[:top_k]

        # Attach scores and sort
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        if min_score is not None:
            candidates = [c for c in candidates if c["rerank_score"] >= min_score]

        return candidates[:top_k]


# Singleton for app-wide use
_shared_reranker: CrossEncoderReranker | None = None


def get_reranker(model_name: str | None = None) -> CrossEncoderReranker:
    global _shared_reranker
    if _shared_reranker is None:
        _shared_reranker = CrossEncoderReranker(model_name)
    return _shared_reranker
