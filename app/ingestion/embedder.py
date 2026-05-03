"""Embedding batch processor — manages efficient embedding generation."""

import logging
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.retrieval import VectorStore

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingProcessor:
    """Handles embedding generation and Qdrant upsert in batches."""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    async def embed_and_store(
        self,
        chunks: list[dict[str, Any]],
        collection: str = "knowledge_docs",
        batch_size: int = 64,
    ) -> list[str]:
        """Embed a list of chunks and store in Qdrant. Returns vector IDs."""
        if not chunks:
            return []

        texts = [c["content"] for c in chunks]
        vector_ids = [str(uuid4()) for _ in chunks]

        # Batch embed — offload to thread pool so we don't block the event loop
        embeddings = await self.vs.embed_batch_async(texts)

        # Build points with metadata
        points = []
        for chunk, vec, vid in zip(chunks, embeddings, vector_ids):
            payload = {
                "content": chunk["content"],
                "source": chunk.get("metadata", {}).get("filename", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "token_count": chunk.get("token_count", 0),
                "doc_type": chunk.get("metadata", {}).get("doc_type", "document"),
                "metadata": {
                    "entities": chunk.get("entities", {}),
                    "summary": chunk.get("summary", ""),
                    **chunk.get("metadata", {}),
                },
            }
            # Add timestamp if available in entities
            timestamps = chunk.get("entities", {}).get("timestamps", [])
            if timestamps:
                payload["metadata"]["timestamp"] = timestamps[0]

            points.append((vid, vec, payload))

        # Upsert in batches
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.vs.upsert(collection, batch)

        logger.info(
            "Stored %d vectors in collection '%s' (batches of %d)",
            len(points),
            collection,
            batch_size,
        )
        return vector_ids
