"""Entity lookup tool — find people, services, teams, hosts by name."""

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.base import BaseTool, ToolSpec, ToolRegistry

logger = logging.getLogger(__name__)


class EntityLookupTool(BaseTool):
    """Look up entities (users, services, teams, hosts) by name or partial name."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="entity_lookup",
            description="Look up entities by name or partial match across users, services, teams, and hosts. "
                        "Use for: 'Who owns service X?', 'What team runs Y?', 'Find host Z'.",
            parameters={
                "name": {"type": "string", "description": "Name or partial name to search for"},
                "entity_type": {
                    "type": "string",
                    "description": "Optional type filter: user, service, team, host (default: all)",
                },
            },
            required_params=["name"],
        )

    async def run(self, name: str, entity_type: str | None = None, **kwargs) -> dict[str, Any]:
        try:
            results = {}
            pattern = f"%{name}%"

            async with self.session_factory() as session:
                # Users
                if entity_type in (None, "user"):
                    stmt = text("""
                        SELECT id, name, email, role
                        FROM users
                        WHERE name ILIKE :pattern OR email ILIKE :pattern
                        LIMIT 5
                    """).bindparams(pattern=pattern)
                    rows = (await session.execute(stmt)).mappings().all()
                    results["users"] = [
                        {"id": r["id"], "name": r["name"], "email": r["email"], "role": r["role"]}
                        for r in rows
                    ]

                # Services with owner and team names
                if entity_type in (None, "service"):
                    stmt = text("""
                        SELECT s.id, s.name, s.environment,
                               u.name as owner_name, u.email as owner_email,
                               t.name as team_name
                        FROM services s
                        LEFT JOIN users u ON s.owner_id = u.id
                        LEFT JOIN teams t ON s.team_id = t.id
                        WHERE s.name ILIKE :pattern
                        LIMIT 5
                    """).bindparams(pattern=pattern)
                    rows = (await session.execute(stmt)).mappings().all()
                    results["services"] = [
                        {
                            "id": r["id"],
                            "name": r["name"],
                            "environment": r["environment"],
                            "owner": r["owner_name"],
                            "owner_email": r["owner_email"],
                            "team": r["team_name"],
                        }
                        for r in rows
                    ]

                # Teams
                if entity_type in (None, "team"):
                    stmt = text("""
                        SELECT id, name, channel
                        FROM teams
                        WHERE name ILIKE :pattern
                        LIMIT 5
                    """).bindparams(pattern=pattern)
                    rows = (await session.execute(stmt)).mappings().all()
                    results["teams"] = [
                        {"id": r["id"], "name": r["name"], "channel": r["channel"]}
                        for r in rows
                    ]

                # Hosts
                if entity_type in (None, "host"):
                    stmt = text("""
                        SELECT id, hostname, ip_address, environment
                        FROM hosts
                        WHERE hostname ILIKE :pattern
                        LIMIT 5
                    """).bindparams(pattern=pattern)
                    rows = (await session.execute(stmt)).mappings().all()
                    results["hosts"] = [
                        {"id": r["id"], "hostname": r["hostname"], "ip_address": r["ip_address"], "environment": r["environment"]}
                        for r in rows
                    ]

            return {
                "success": True,
                "data": results,
                "source": "entity_db",
            }
        except Exception as e:
            logger.exception("Entity lookup failed")
            return {"success": False, "data": {}, "error": str(e)}


def register_entity_tool(session_factory):
    ToolRegistry.register(EntityLookupTool(session_factory))
