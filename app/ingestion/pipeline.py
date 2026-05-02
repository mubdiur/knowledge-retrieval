"""Ingestion pipeline — coordinates parsing, chunking, extraction, embedding, and storage."""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.ingestion.parser import FileParser
from app.ingestion.chunker import AdaptiveChunker
from app.ingestion.extractor import EntityExtractor
from app.ingestion.embedder import EmbeddingProcessor
from app.models.db import Document, Chunk
from app.retrieval import VectorStore, BM25Index

logger = logging.getLogger(__name__)
settings = get_settings()


class IngestionPipeline:
    """End-to-end ingestion: file → parse → chunk → extract → embed → store."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        session_factory,
        data_dir: str = "data/ingested",
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_index
        self.session_factory = session_factory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.parser = FileParser()
        self.chunker = AdaptiveChunker(
            max_size=settings.max_chunk_size,
            overlap=settings.chunk_overlap,
        )
        self.extractor = EntityExtractor()
        self.embedder = EmbeddingProcessor(vector_store)

    async def ingest_file(self, file_path: str | Path) -> dict[str, Any]:
        """Ingest a single file through the full pipeline."""
        file_path = Path(file_path)
        logger.info("Ingesting: %s", file_path)

        # 1. Parse
        parsed = self.parser.parse(file_path)
        content = parsed["content"]
        doc_metadata = parsed["metadata"]
        doc_type = parsed["doc_type"]

        # 2. Copy raw to data dir
        raw_path = self._store_raw(file_path, content)

        # 3. Chunk (adaptive)
        chunks = self.chunker.chunk(content, metadata=doc_metadata)
        logger.info("  → %d chunks from %s", len(chunks), file_path.name)

        # 4. Extract entities + summarize per chunk
        enriched_chunks = []
        all_chunk_texts = []
        for chunk in chunks:
            entities = self.extractor.extract(chunk["content"])
            summary = self.extractor.summarize(chunk["content"])
            chunk["entities"] = entities
            chunk["summary"] = summary
            enriched_chunks.append(chunk)
            all_chunk_texts.append(chunk["content"])

        # 5. Build document summary from first chunk
        doc_summary = self.extractor.summarize(content, max_sentences=4)

        # 6. Embed and store in Qdrant
        vector_ids = self.embedder.embed_and_store(
            enriched_chunks,
            collection="knowledge_logs" if doc_type == "log" else "knowledge_docs",
        )

        # 7. Store in PostgreSQL
        db_result = await self._store_in_db(
            file_path.name, doc_type, raw_path, doc_summary,
            enriched_chunks, vector_ids,
        )

        # 8. Rebuild BM25 index with new chunks
        await self._rebuild_bm25()

        return {
            "filename": file_path.name,
            "doc_type": doc_type,
            "chunk_count": len(chunks),
            "vector_ids_count": len(vector_ids),
            "db_id": db_result,
        }

    async def ingest_directory(self, dir_path: str | Path) -> list[dict[str, Any]]:
        """Ingest all supported files in a directory."""
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"{dir_path} is not a directory")

        results = []
        for f in sorted(dir_path.iterdir()):
            if f.suffix.lower() in FileParser.SUPPORTED:
                try:
                    result = await self.ingest_file(f)
                    results.append(result)
                except Exception as e:
                    logger.error("Failed to ingest %s: %s", f.name, e)
                    results.append({"filename": f.name, "error": str(e)})
        return results

    def _store_raw(self, source_path: Path, content: str) -> str:
        """Copy raw content to centralized storage."""
        dest = self.data_dir / f"{source_path.stem}_{uuid4().hex[:8]}{source_path.suffix}"
        dest.write_text(content, encoding="utf-8")
        return str(dest)

    async def _store_in_db(
        self,
        filename: str,
        doc_type: str,
        raw_path: str,
        summary: str,
        chunks: list[dict],
        vector_ids: list[str],
    ) -> int:
        """Persist document and chunk metadata in PostgreSQL."""
        async with self.session_factory() as session:
            doc = Document(
                filename=filename,
                doc_type=doc_type,
                source_path=raw_path,
                summary=summary,
                chunk_count=len(chunks),
            )
            session.add(doc)
            await session.flush()

            db_chunks = []
            for idx, (chunk, vid) in enumerate(zip(chunks, vector_ids)):
                db_chunks.append(Chunk(
                    document_id=doc.id,
                    chunk_index=idx,
                    content=chunk["content"],
                    vector_id=vid,
                    token_count=chunk.get("token_count", 0),
                    entities=chunk.get("entities", {}),
                    metadata_json=chunk.get("metadata", {}),
                ))
            session.add_all(db_chunks)
            await session.commit()
            logger.info("  → DB record: document_id=%d, chunks=%d", doc.id, len(db_chunks))
            return doc.id

    async def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all chunks in the database."""
        try:
            async with self.session_factory() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(Chunk).limit(5000)
                )
                rows = result.scalars().all()
                docs = [
                    {"id": c.vector_id, "content": c.content, "metadata": c.metadata_json}
                    for c in rows
                ]
                if docs:
                    self.bm25.index(docs)
                    logger.info("BM25 index rebuilt: %d docs", len(docs))
        except Exception as e:
            logger.warning("BM25 rebuild skipped: %s", e)

    async def reset_all(self) -> None:
        """Reset all data — collections, tables, indexes."""
        self.vector_store.delete_collection(settings.vector_collection)
        self.vector_store.delete_collection(settings.log_collection)
        self.vector_store._ensure_collections()
        self.bm25 = BM25Index()

        async with self.session_factory() as session:
            from sqlalchemy import text
            for table in ["chunks", "documents"]:
                await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            await session.commit()
        logger.warning("All data reset complete")
