"""Incident query tool — time-range and relationship queries on incidents."""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.tools.base import BaseTool, ToolSpec, ToolRegistry

logger = logging.getLogger(__name__)


class IncidentQueryTool(BaseTool):
    """Query incidents by time range, severity, service, and team."""

    def __init__(self, session_factory):
        self.session_factory = session_factory

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="get_incidents",
            description="Retrieve incidents with filters for time range, severity, service, team, and status. "
                        "Essential for incident analysis, root cause investigation, and trend spotting.",
            parameters={
                "time_range": {
                    "type": "tuple",
                    "description": "Optional (start, end) ISO datetime tuple to filter incidents",
                },
                "severity": {
                    "type": "string",
                    "description": "Filter by severity: critical, major, minor, trivial",
                },
                "service": {
                    "type": "string",
                    "description": "Filter by service name",
                },
                "team": {
                    "type": "string",
                    "description": "Filter by team name",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by status: open, investigating, mitigated, resolved, closed",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum incidents to return (default 20)",
                },
            },
            required_params=[],
        )

    async def run(
        self,
        time_range: tuple | None = None,
        severity: str | None = None,
        service: str | None = None,
        team: str | None = None,
        status: str | None = None,
        limit: int = 20,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            from app.models.db import Incident, Service, Team
            from sqlalchemy.orm import selectinload
            from sqlalchemy import func

            conditions = []

            if time_range:
                start, end = time_range
                if isinstance(start, str):
                    start = datetime.fromisoformat(start)
                    end = datetime.fromisoformat(end)
                conditions.append(Incident.started_at >= start)
                conditions.append(Incident.started_at <= end)

            if severity:
                conditions.append(Incident.severity == severity)

            if status:
                conditions.append(Incident.status == status)

            if service:
                conditions.append(Service.name == service)

            if team:
                conditions.append(Team.name == team)

            stmt = (
                select(Incident)
                .options(selectinload(Incident.service), selectinload(Incident.team))
            )

            if service:
                stmt = stmt.join(Service, Incident.service_id == Service.id)
            if team:
                stmt = stmt.join(Team, Incident.team_id == Team.id)

            if conditions:
                stmt = stmt.where(and_(*conditions))

            stmt = stmt.order_by(Incident.started_at.desc()).limit(limit)

            async with self.session_factory() as session:
                result = await session.execute(stmt)
                incidents = result.scalars().all()
                data = []
                for inc in incidents:
                    data.append({
                        "id": inc.id,
                        "title": inc.title,
                        "severity": inc.severity.value if hasattr(inc.severity, 'value') else inc.severity,
                        "status": inc.status.value if hasattr(inc.status, 'value') else inc.status,
                        "service": inc.service.name if inc.service else None,
                        "team": inc.team.name if inc.team else None,
                        "root_cause": inc.root_cause,
                        "resolution": inc.resolution,
                        "started_at": inc.started_at.isoformat() if inc.started_at else None,
                        "resolved_at": inc.resolved_at.isoformat() if inc.resolved_at else None,
                        "impacted_hosts": inc.impacted_hosts,
                        "tags": inc.tags,
                    })
                return {
                    "success": True,
                    "data": data,
                    "result_count": len(data),
                    "source": "incidents_db",
                }
        except Exception as e:
            logger.exception("Incident query failed")
            return {"success": False, "data": [], "error": str(e)}


def register_incident_tool(session_factory):
    ToolRegistry.register(IncidentQueryTool(session_factory))
