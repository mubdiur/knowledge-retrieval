"""Keyword search tool — BM25-based exact-term retrieval."""

import logging
from typing import Any

from app.tools.base import BaseTool, ToolSpec, ToolRegistry
from app.retrieval import BM25Index

logger = logging.getLogger(__name__)


class KeywordSearchTool(BaseTool):
    """Search for exact terms using BM25 keyword matching."""

    def __init__(self, bm25: BM25Index):
        self.bm25 = bm25

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="search_keyword",
            description="Exact-term keyword search using BM25. Best for finding specific terms, names, error codes, "
                        "hostnames, IP addresses, and technical identifiers. Complements vector search for precision.",
            parameters={
                "query": {"type": "string", "description": "The keyword query"},
                "top_k": {"type": "integer", "description": "Number of results (default 10)"},
            },
            required_params=["query"],
        )

    async def run(self, query: str, top_k: int = 10, **kwargs) -> dict[str, Any]:
        try:
            if not self.bm25.is_ready:
                return {"success": False, "data": [], "error": "BM25 index not built"}
            results = self.bm25.search(query, top_k=top_k)
            return {
                "success": True,
                "data": results,
                "result_count": len(results),
                "source": "bm25",
            }
        except Exception as e:
            logger.exception("Keyword search failed")
            return {"success": False, "data": [], "error": str(e)}


def register_keyword_tool(bm25: BM25Index) -> None:
    ToolRegistry.register(KeywordSearchTool(bm25))
