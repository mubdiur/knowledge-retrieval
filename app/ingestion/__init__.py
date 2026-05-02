"""Ingestion package — lazy imports."""

def FileParser():
    from .parser import FileParser as _cls
    return _cls()

def AdaptiveChunker(*a, **kw):
    from .chunker import AdaptiveChunker as _cls
    return _cls(*a, **kw)

def EntityExtractor():
    from .extractor import EntityExtractor as _cls
    return _cls()

def EmbeddingProcessor(*a, **kw):
    from .embedder import EmbeddingProcessor as _cls
    return _cls(*a, **kw)

def IngestionPipeline(*a, **kw):
    from .pipeline import IngestionPipeline as _cls
    return _cls(*a, **kw)

__all__ = ["FileParser", "AdaptiveChunker", "EntityExtractor", "EmbeddingProcessor", "IngestionPipeline"]
