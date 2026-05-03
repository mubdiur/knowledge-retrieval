"""Keyword search tool — BM25-based exact-term retrieval."""

import logging
from typing import Any

from app.tools.base import BaseTool, ToolSpec, ToolRegistry
from app.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class KeywordSearchTool(BaseTool):
    """Search for exact terms using BM25 keyword matching."""

    def __init__(self, hybrid_retriever: HybridRetriever):
        self.retriever = hybrid_retriever

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
            results = self.retriever.search(
                query=query,
                collection="knowledge_docs",
                top_k=top_k,
                rerank=True,
            )
            return {
                "success": True,
                "data": results,
                "result_count": len(results),
                "source": "bm25",
            }
        except Exception as e:
            logger.exception("Keyword search failed")
            return {"success": False, "data": [], "error": str(e)}


def register_keyword_tool(hybrid_retriever: HybridRetriever) -> None:
    ToolRegistry.register(KeywordSearchTool(hybrid_retriever))
