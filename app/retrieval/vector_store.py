"""Qdrant vector store wrapper — handles collections, embeddings, search."""

import uuid
import logging
from typing import Any

import numpy as np
from qdrant_client import QdrantClient, models as qm
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class VectorStore:
    """Manages vector embeddings and Qdrant interactions."""

    def __init__(self):
        self.client = QdrantClient(url=settings.qdrant_url)
        self.model = SentenceTransformer(settings.embedding_model)
        self._dim = settings.embedding_dim
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        for col in (settings.vector_collection, settings.log_collection):
            try:
                self.client.get_collection(col)
                logger.info("Collection %s exists", col)
            except Exception:
                self.client.create_collection(
                    collection_name=col,
                    vectors_config=qm.VectorParams(
                        size=self._dim,
                        distance=qm.Distance.COSINE,
                    ),
                )
                # Add payload indexes for filtered search
                self.client.create_payload_index(
                    collection_name=col,
                    field_name="metadata.timestamp",
                    field_schema=qm.PayloadSchemaType.DATETIME,
                )
                self.client.create_payload_index(
                    collection_name=col,
                    field_name="metadata.service",
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=col,
                    field_name="metadata.team",
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=col,
                    field_name="metadata.severity",
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
                logger.info("Created collection %s with indexes", col)

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    async def embed_async(self, text: str) -> list[float]:
        import asyncio
        return await asyncio.to_thread(self.embed, text)

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        return await asyncio.to_thread(self.embed_batch, texts)

    def upsert(
        self,
        collection: str,
        points: list[tuple[str, list[float], dict[str, Any]]],
    ) -> int:
        """Upsert (vector_id, vector, payload) tuples. Returns count."""
        if not points:
            return 0
        qdrant_points = [
            qm.PointStruct(id=pid, vector=vec, payload=payload)
            for pid, vec, payload in points
        ]
        result = self.client.upsert(
            collection_name=collection,
            points=qdrant_points,
            wait=True,
        )
        return len(qdrant_points)

    def search(
        self,
        collection: str,
        query: str,
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
        top_k: int = 10,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Hybrid vector + filtered search."""
        vector = self.embed(query)

        # Build Qdrant filter
        must_conditions: list[qm.FieldCondition | qm.IsEmptyCondition] = []

        if filters:
            for key, value in filters.items():
                if isinstance(value, list):
                    must_conditions.append(
                        qm.FieldCondition(
                            key=f"metadata.{key}",
                            match=qm.MatchAny(any=value),
                        )
                    )
                else:
                    must_conditions.append(
                        qm.FieldCondition(
                            key=f"metadata.{key}",
                            match=qm.MatchValue(value=value),
                        )
                    )

        if time_range:
            start, end = time_range
            must_conditions.append(
                qm.FieldCondition(
                    key="metadata.timestamp",
                    range=qm.Range(
                        gte=int(start.timestamp()),
                        lte=int(end.timestamp()),
                    ),
                )
            )

        filter_obj = qm.Filter(must=must_conditions) if must_conditions else None

        # qdrant-client >= 1.17 uses query_points instead of search
        try:
            response = self.client.query_points(
                collection_name=collection,
                query=vector,
                query_filter=filter_obj,
                limit=top_k,
                score_threshold=score_threshold,
            )
            # New API returns QueryResponse with .points
            hits = response.points
        except TypeError:
            # Fallback for older qdrant-client versions
            hits = self.client.search(
                collection_name=collection,
                query_vector=vector,
                query_filter=filter_obj,
                limit=top_k,
                score_threshold=score_threshold,
            )

        results = []
        for hit in hits:
            payload_source = hit.payload.get("source", "")
            # Tag results with the retrieval source type for downstream filtering
            retrieval_source = "vector_store"
            if "log" in collection:
                retrieval_source = "vector_store_logs"
            results.append({
                "id": str(hit.id),
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "source": payload_source,
                "retrieval_source": retrieval_source,
                "metadata": hit.payload.get("metadata", {}),
            })
        return results

    def delete_collection(self, collection: str) -> None:
        try:
            self.client.delete_collection(collection)
        except Exception:
            pass
