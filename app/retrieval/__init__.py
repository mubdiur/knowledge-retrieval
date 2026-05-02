"""Retrieval package — lazy imports for faster startup."""

def VectorStore():
    from .vector_store import VectorStore as _cls
    return _cls

def BM25Index():
    from .keyword_store import BM25Index as _cls
    return _cls

def HybridRetriever(*a, **kw):
    from .hybrid import HybridRetriever as _cls
    return _cls(*a, **kw)

def reciprocal_rank_fusion(*a, **kw):
    from .hybrid import reciprocal_rank_fusion as _fn
    return _fn(*a, **kw)

def CrossEncoderReranker(*a, **kw):
    from .reranker import CrossEncoderReranker as _cls
    return _cls(*a, **kw)

def get_reranker(*a, **kw):
    from .reranker import get_reranker as _fn
    return _fn(*a, **kw)

__all__ = [
    "VectorStore", "BM25Index", "HybridRetriever",
    "reciprocal_rank_fusion", "CrossEncoderReranker", "get_reranker",
]
