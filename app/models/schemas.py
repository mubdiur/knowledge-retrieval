"""Pydantic schemas for API and internal use."""

from datetime import datetime
from pydantic import BaseModel, Field
from typing import Any


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    filters: dict[str, Any] | None = Field(default=None, description="Optional metadata filters")
    time_range: tuple[datetime, datetime] | None = Field(default=None, description="(start, end) UTC")
    top_k: int = Field(default=10, ge=1, le=100)
    enable_refinement: bool = Field(default=True, description="Allow iterative retrieval")
    conversation_id: str | None = Field(default=None, description="Continue an existing conversation")


class SourceRef(BaseModel):
    title: str
    source: str
    score: float | None = None
    snippet: str
    metadata: dict[str, Any] = {}


class ReasoningStep(BaseModel):
    step: int
    action: str
    tool: str | None = None
    input_summary: str
    output_summary: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = []
    reasoning_trace: list[ReasoningStep] = []
    query_type: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]


# ── Internal schemas ──────────────────────────────────────────────────────────

class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str | None = None
    source: str | None = None


class AgentContext(BaseModel):
    query: str
    query_type: str | None = None
    filters: dict[str, Any] = {}
    time_range: tuple[datetime, datetime] | None = None
    top_k: int = 10
    conversation_history: list[dict] = []
    intermediate_results: list[dict] = []
    refinement_count: int = 0
    max_refinements: int = 3


class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    query_type: str | None = None


class Conversation(BaseModel):
    id: str
    turns: list[ConversationTurn] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
