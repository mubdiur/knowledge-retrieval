"""FastAPI routes for the knowledge retrieval system."""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse, HealthResponse
from app.agents.orchestrator import AgentOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

# Populated at app startup
_orchestrator: AgentOrchestrator | None = None


def init_routes(orchestrator: AgentOrchestrator) -> None:
    global _orchestrator
    _orchestrator = orchestrator


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Submit a natural language query to the knowledge retrieval agent."""
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    logger.info("Query: %s (filters=%s, time_range=%s, conv=%s)",
                request.query[:100], request.filters, request.time_range, request.conversation_id)

    try:
        response = await _orchestrator.answer(
            query=request.query,
            filters=request.filters,
            time_range=request.time_range,
            top_k=request.top_k,
            enable_refinement=request.enable_refinement,
            conversation_id=request.conversation_id,
        )
        return response
    except Exception as e:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        services={
            "agent": "ready" if _orchestrator is not None else "not_initialized",
            "tools": f"{len(app.tools.base.ToolRegistry._tools) if _orchestrator else 0} registered",
        },
    )


# Import needed for health endpoint
import app.tools.base  # noqa: E402
