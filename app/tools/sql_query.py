"""SQL query tool — structured data retrieval from PostgreSQL."""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.base import BaseTool, ToolSpec, ToolRegistry

logger = logging.getLogger(__name__)

# Whitelist of safe SQL patterns — the tool only allows SELECT queries
SAFE_PATTERNS = ["select", "with", "explain"]


class SQLQueryTool(BaseTool):
    """Execute safe SQL queries against the structured knowledge layer."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="query_sql",
            description="Execute SQL queries against the relational database (users, teams, services, hosts, incidents). "
                        "Only SELECT queries are allowed. Use for precise structured lookups and aggregations.",
            parameters={
                "query": {"type": "string", "description": "SQL SELECT query to execute"},
                "params": {"type": "dict", "description": "Optional query parameters"},
            },
            required_params=["query"],
        )

    async def run(self, query: str, params: dict[str, Any] | None = None, **kwargs) -> dict[str, Any]:
        # Safety check
        query_stripped = query.strip().lower()
        if not any(query_stripped.startswith(prefix) for prefix in SAFE_PATTERNS):
            return {
                "success": False,
                "data": [],
                "error": "Only SELECT and WITH queries are allowed",
            }

        try:
            async with self.session_factory() as session:
                result = await session.execute(text(query), params or {})
                rows = result.mappings().all()
                data = [dict(row) for row in rows]
                return {
                    "success": True,
                    "data": data,
                    "row_count": len(data),
                    "source": "postgresql",
                }
        except Exception as e:
            logger.exception("SQL query failed: %s", query[:200])
            return {"success": False, "data": [], "error": str(e)}


def register_sql_tool(session_factory):
    ToolRegistry.register(SQLQueryTool(session_factory))
