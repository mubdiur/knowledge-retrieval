"""Entity lookup tool — find people, services, teams, hosts by name."""

import logging
from typing import Any

from sqlalchemy import select, or_
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
            from app.models.db import User, Service, Team, Host

            results = {}
            async with self.session_factory() as session:
                # Users
                if entity_type in (None, "user"):
                    stmt = select(User).where(
                        or_(
                            User.name.ilike(f"%{name}%"),
                            User.email.ilike(f"%{name}%"),
                        )
                    ).limit(5)
                    rows = (await session.execute(stmt)).scalars().all()
                    results["users"] = [
                        {"id": u.id, "name": u.name, "email": u.email, "role": u.role}
                        for u in rows
                    ]

                # Services
                if entity_type in (None, "service"):
                    stmt = select(Service).where(Service.name.ilike(f"%{name}%")).limit(5)
                    rows = (await session.execute(stmt)).scalars().all()
                    results["services"] = [
                        {"id": s.id, "name": s.name, "environment": s.environment.value if hasattr(s.environment, 'value') else s.environment}
                        for s in rows
                    ]

                # Teams
                if entity_type in (None, "team"):
                    stmt = select(Team).where(Team.name.ilike(f"%{name}%")).limit(5)
                    rows = (await session.execute(stmt)).scalars().all()
                    results["teams"] = [
                        {"id": t.id, "name": t.name, "channel": t.channel}
                        for t in rows
                    ]

                # Hosts
                if entity_type in (None, "host"):
                    stmt = select(Host).where(Host.hostname.ilike(f"%{name}%")).limit(5)
                    rows = (await session.execute(stmt)).scalars().all()
                    results["hosts"] = [
                        {"id": h.id, "hostname": h.hostname, "ip_address": h.ip_address, "environment": h.environment.value if hasattr(h.environment, 'value') else h.environment}
                        for h in rows
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
