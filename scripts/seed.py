#!/usr/bin/env python3
"""
Seed script — populates PostgreSQL + Qdrant with sample organizational data.

Usage:
    python scripts/seed.py                    # Full seed
    python scripts/seed.py --db-only          # Skip Qdrant/BM25
    python scripts/seed.py --reset            # Reset all first
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

# Ensure app module is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from app.config import get_settings
from app.models.db import Base, User, Team, TeamMember, Service, Host, Incident, IncidentTimeline
from app.retrieval import VectorStore, BM25Index
from app.ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "sample")


async def seed_database(session_factory, reset: bool = False):
    """Populate PostgreSQL with sample data."""
    async with session_factory() as session:
        if reset:
            logger.info("Resetting all tables...")
            for table in ["incident_timelines", "incidents", "hosts", "services", "team_members", "teams", "users", "chunks", "documents"]:
                await session.execute(text(f"TRUNCATE TABLE {table} CASCADE"))
            await session.commit()

    with open(os.path.join(DATA_DIR, "services.json")) as f:
        data = json.load(f)

    # ── Users ───────────────────────────────────────────────────────────────
    user_map = {}
    async with session_factory() as session:
        for u in data["users"]:
            user = User(name=u["name"], email=u["email"], role=u["role"], slack_handle=u["slack"])
            session.add(user)
            await session.flush()
            user_map[u["email"]] = user
        logger.info("Seeded %d users", len(data["users"]))

        # ── Teams ───────────────────────────────────────────────────────────
        team_map = {}
        for t in data["teams"]:
            team = Team(name=t["name"], description=t.get("description"), channel=t.get("channel"))
            session.add(team)
            await session.flush()
            team_map[t["name"]] = team
        logger.info("Seeded %d teams", len(data["teams"]))

        # ── Team Memberships ────────────────────────────────────────────────
        team_assignments = {
            "Payments Squad": ["alice@company.com"],
            "Platform Team": ["bob@company.com", "carol@company.com"],
            "Commerce Team": ["dave@company.com", "eve@company.com"],
            "SRE Team": ["frank@company.com"],
            "Security Team": ["frank@company.com"],
        }
        for team_name, emails in team_assignments.items():
            for email in emails:
                if email in user_map and team_name in team_map:
                    session.add(TeamMember(user_id=user_map[email].id, team_id=team_map[team_name].id))
        logger.info("Seeded team memberships")

        # ── Services + Hosts ────────────────────────────────────────────────
        for svc in data["services"]:
            owner_email = svc.get("owner")
            team_name = svc.get("team")
            service = Service(
                name=svc["name"],
                description=svc.get("description"),
                owner_id=user_map.get(owner_email).id if owner_email in user_map else None,
                team_id=team_map.get(team_name).id if team_name in team_map else None,
                environment=svc.get("environment", "production"),
                repository=svc.get("repository"),
            )
            session.add(service)
            await session.flush()

            for h in svc.get("hosts", []):
                host = Host(
                    hostname=h["hostname"],
                    ip_address=h.get("ip"),
                    service_id=service.id,
                    environment=svc.get("environment", "production"),
                    region=h.get("region"),
                    provider=h.get("provider"),
                )
                session.add(host)

        await session.commit()
    logger.info("Seeded services and hosts")

    # ── Incidents ───────────────────────────────────────────────────────────
    with open(os.path.join(DATA_DIR, "incidents.json")) as f:
        incidents_data = json.load(f)

    async with session_factory() as session:
        for inc_data in incidents_data:
            service_name = inc_data.get("service")
            team_name = inc_data.get("team")
            reporter_email = inc_data.get("reported_by")

            # Get service and team IDs
            svc_result = await session.execute(
                text("SELECT id FROM services WHERE name = :name"),
                {"name": service_name},
            )
            svc_row = svc_result.scalar_one_or_none()

            team_result = await session.execute(
                text("SELECT id FROM teams WHERE name = :name"),
                {"name": team_name},
            )
            team_row = team_result.scalar_one_or_none()

            incident = Incident(
                title=inc_data["title"],
                severity=inc_data["severity"],
                status=inc_data.get("status", "resolved"),
                service_id=svc_row if svc_row else None,
                team_id=team_row if team_row else None,
                reported_by=user_map.get(reporter_email).id if reporter_email in user_map else None,
                root_cause=inc_data.get("root_cause"),
                resolution=inc_data.get("resolution"),
                started_at=datetime.fromisoformat(inc_data["started_at"].replace("Z", "+00:00")),
                detected_at=datetime.fromisoformat(inc_data["detected_at"].replace("Z", "+00:00")) if inc_data.get("detected_at") else None,
                mitigated_at=datetime.fromisoformat(inc_data["mitigated_at"].replace("Z", "+00:00")) if inc_data.get("mitigated_at") else None,
                resolved_at=datetime.fromisoformat(inc_data["resolved_at"].replace("Z", "+00:00")) if inc_data.get("resolved_at") else None,
                impacted_hosts=inc_data.get("impacted_hosts", []),
                tags=inc_data.get("tags", []),
            )
            session.add(incident)
            await session.flush()

            for tl in inc_data.get("timeline", []):
                timeline_entry = IncidentTimeline(
                    incident_id=incident.id,
                    timestamp=datetime.fromisoformat(tl["timestamp"].replace("Z", "+00:00")),
                    entry_type=tl["type"],
                    content=tl["content"],
                    actor=tl.get("actor"),
                )
                session.add(timeline_entry)

        await session.commit()
    logger.info("Seeded %d incidents with timelines", len(incidents_data))


async def seed_vector_and_bm25(vector_store, session_factory):
    """Ingest sample docs into Qdrant and rebuild BM25."""
    pipeline = IngestionPipeline(vector_store, BM25Index(), session_factory)

    sample_dir = os.path.join(DATA_DIR)

    # Ingest runbook
    runbook_path = os.path.join(sample_dir, "runbook.md")
    if os.path.exists(runbook_path):
        await pipeline.ingest_file(runbook_path)
        logger.info("Ingested runbook")

    # Ingest logs
    logs_path = os.path.join(sample_dir, "logs.txt")
    if os.path.exists(logs_path):
        await pipeline.ingest_file(logs_path)
        logger.info("Ingested logs")

    # Ingest incidents JSON as a document
    incidents_path = os.path.join(sample_dir, "incidents.json")
    if os.path.exists(incidents_path):
        await pipeline.ingest_file(incidents_path)
        logger.info("Ingested incidents JSON")

    logger.info("Vector store and BM25 seeded")


async def main():
    parser = argparse.ArgumentParser(description="Seed the knowledge retrieval system")
    parser.add_argument("--reset", action="store_true", help="Reset all data before seeding")
    parser.add_argument("--db-only", action="store_true", help="Only seed database, skip vector/BM25")
    args = parser.parse_args()

    engine = create_async_engine(settings.database_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    # Ensure tables exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed database
    await seed_database(session_factory, reset=args.reset)

    if not args.db_only:
        vector_store = VectorStore()
        await seed_vector_and_bm25(vector_store, session_factory)

    await engine.dispose()
    logger.info("Seeding complete! 🚀")


if __name__ == "__main__":
    asyncio.run(main())
