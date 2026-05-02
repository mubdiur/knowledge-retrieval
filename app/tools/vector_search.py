"""Vector search tool — semantic retrieval over ingested knowledge."""

import logging
from typing import Any

from app.tools.base import BaseTool, ToolSpec, ToolRegistry
from app.retrieval import VectorStore

logger = logging.getLogger(__name__)


class VectorSearchTool(BaseTool):
    """Search over vector embeddings for semantically similar content."""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_vector",
            description="Search for semantically similar content using vector embeddings. Best for conceptual questions, "
                        "understanding 'what' and 'why', finding related incidents, documents, or logs by meaning.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query (natural language)",
                },
                "filters": {
                    "type": "dict",
                    "description": "Optional metadata filters: {service, team, severity, source}",
                },
                "time_range": {
                    "type": "tuple",
                    "description": "Optional (start, end) datetime tuple to scope results",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 10)",
                },
            },
            required_params=["query"],
        )

    async def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
        top_k: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            results = self.vs.search(
                collection="knowledge_docs",
                query=query,
                filters=filters,
                time_range=time_range,
                top_k=top_k,
            )
            return {
                "success": True,
                "data": results,
                "result_count": len(results),
                "source": "vector_store",
            }
        except Exception as e:
            logger.exception("Vector search failed")
            return {"success": False, "data": [], "error": str(e)}


class VectorLogSearchTool(BaseTool):
    """Vector search over indexed logs (separate collection)."""

    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_vector_logs",
            description="Search over log entries using vector embeddings. Best for finding log patterns, error messages, "
                        "and understanding system behavior from log data.",
            parameters={
                "query": {"type": "string", "description": "The search query"},
                "filters": {"type": "dict", "description": "Optional metadata filters"},
                "time_range": {"type": "tuple", "description": "Optional (start, end) datetime tuple"},
                "top_k": {"type": "integer", "description": "Number of results (default 10)"},
            },
            required_params=["query"],
        )

    async def run(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
        top_k: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            results = self.vs.search(
                collection="knowledge_logs",
                query=query,
                filters=filters,
                time_range=time_range,
                top_k=top_k,
            )
            return {
                "success": True,
                "data": results,
                "result_count": len(results),
                "source": "vector_store_logs",
            }
        except Exception as e:
            logger.exception("Log vector search failed")
            return {"success": False, "data": [], "error": str(e)}


def register_vector_tools(vector_store: VectorStore) -> None:
    ToolRegistry.register(VectorSearchTool(vector_store))
    ToolRegistry.register(VectorLogSearchTool(vector_store))
