"""Application entry point — wires together all components and starts the server."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.config import get_settings
from app.models.db import Base
from app.retrieval import VectorStore, BM25Index, HybridRetriever
from app.retrieval.reranker import get_reranker
from app.tools import (
    ToolRegistry,
    register_vector_tools,
    register_keyword_tool,
    register_sql_tool,
    register_incident_tool,
    register_entity_tool,
)
from app.agents.orchestrator import AgentOrchestrator
from app.api.routes import router, init_routes

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — init and teardown."""
    logger.info("Starting Knowledge Retrieval System...")

    # ── Database ──────────────────────────────────────────────────────────
    engine = create_async_engine(settings.database_url, echo=settings.debug)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ensured")

    # ── Vector Store ──────────────────────────────────────────────────────
    vector_store = VectorStore()

    # ── BM25 ──────────────────────────────────────────────────────────────
    bm25 = BM25Index()

    # Rebuild BM25 from DB on startup
    try:
        from app.models.db import Chunk
        from sqlalchemy import select
        async with session_factory() as session:
            result = await session.execute(select(Chunk).limit(5000))
            rows = result.scalars().all()
            docs = [
                {"id": c.vector_id, "content": c.content, "metadata": c.metadata_json}
                for c in rows
            ]
            if docs:
                bm25.index(docs)
                logger.info("BM25 index loaded from DB: %d docs", len(docs))
    except Exception as e:
        logger.warning("Could not load BM25 from DB (first run?): %s", e)

    hybrid = HybridRetriever(vector_store, bm25)

    # ── Reranker ─────────────────────────────────────────────────────────
    try:
        reranker = get_reranker()
        hybrid.set_reranker(reranker)
        logger.info("Cross-encoder reranker loaded and attached to hybrid retriever")
    except Exception as e:
        logger.warning("Reranker not available (non-fatal): %s", e)

    # ── Register Tools ────────────────────────────────────────────────────
    register_vector_tools(vector_store)
    register_keyword_tool(bm25)
    register_sql_tool(session_factory)
    register_incident_tool(session_factory)
    register_entity_tool(session_factory)
    logger.info("Tools registered:\n%s", ToolRegistry.describe())

    # ── Agent ─────────────────────────────────────────────────────────────
    orchestrator = AgentOrchestrator()
    init_routes(orchestrator)

    # Store references in app state for other components
    app.state.vector_store = vector_store
    app.state.bm25 = bm25
    app.state.hybrid = hybrid
    app.state.session_factory = session_factory
    app.state.orchestrator = orchestrator

    logger.info("Knowledge Retrieval System ready")
    yield

    # Teardown
    await engine.dispose()
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Factory to create the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
