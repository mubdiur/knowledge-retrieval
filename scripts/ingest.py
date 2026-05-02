#!/usr/bin/env python3
"""
Ingest arbitrary files into the knowledge retrieval system.

Usage:
    python scripts/ingest.py path/to/file.txt
    python scripts/ingest.py path/to/directory/
    python scripts/ingest.py path/to/file.log --type logs
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.config import get_settings
from app.retrieval import VectorStore, BM25Index
from app.ingestion.pipeline import IngestionPipeline
from app.models.db import Chunk
from sqlalchemy import select

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


async def ingest(path: str, doc_type: str | None = None):
    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    vector_store = VectorStore()
    bm25 = BM25Index()

    # Load existing BM25 from DB
    async with session_factory() as session:
        result = await session.execute(select(Chunk).limit(5000))
        rows = result.scalars().all()
        docs = [{"id": c.vector_id, "content": c.content, "metadata": c.metadata_json} for c in rows]
        if docs:
            bm25.index(docs)

    pipeline = IngestionPipeline(vector_store, bm25, session_factory)

    path = os.path.abspath(path)
    if os.path.isfile(path):
        result = await pipeline.ingest_file(path)
        logger.info("Ingested: %s → %s", result["filename"], result)
    elif os.path.isdir(path):
        results = await pipeline.ingest_directory(path)
        for r in results:
            status = "✅" if "error" not in r else "❌"
            logger.info("%s %s: %s", status, r["filename"], r.get("error", f"{r.get('chunk_count', 0)} chunks"))
    else:
        logger.error("Path not found: %s", path)

    await engine.dispose()
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest files into the knowledge system")
    parser.add_argument("path", help="File or directory to ingest")
    args = parser.parse_args()
    asyncio.run(ingest(args.path))
