"""SQLAlchemy ORM models for structured knowledge layer."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Enum as SAEnum,
    Float, Boolean, Index, JSON, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship
import enum

Base = declarative_base()


# ── Enums ─────────────────────────────────────────────────────────────────────

class IncidentSeverity(str, enum.Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    TRIVIAL = "trivial"


class IncidentStatus(str, enum.Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    MITIGATED = "mitigated"
    RESOLVED = "resolved"
    CLOSED = "closed"


class EnvironmentType(str, enum.Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    DR = "dr"


# ── Core Tables ───────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    slack_handle = Column(String(100))
    role = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    teams = relationship("TeamMember", back_populates="user")
    owned_services = relationship("Service", back_populates="owner")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    channel = Column(String(100))  # Slack channel
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    members = relationship("TeamMember", back_populates="team")
    services = relationship("Service", back_populates="team")
    incidents = relationship("Incident", back_populates="team")


class TeamMember(Base):
    __tablename__ = "team_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="CASCADE"), nullable=False)
    is_lead = Column(Boolean, default=False)
    joined_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="teams")
    team = relationship("Team", back_populates="members")

    __table_args__ = (UniqueConstraint("user_id", "team_id"),)


class Service(Base):
    __tablename__ = "services"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"))
    environment = Column(SAEnum(EnvironmentType), default=EnvironmentType.PRODUCTION)
    repository = Column(String(500))
    documentation_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    owner = relationship("User", back_populates="owned_services")
    team = relationship("Team", back_populates="services")
    hosts = relationship("Host", back_populates="service")
    incidents = relationship("Incident", back_populates="service")


class Host(Base):
    __tablename__ = "hosts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    hostname = Column(String(255), unique=True, nullable=False, index=True)
    ip_address = Column(String(45))
    service_id = Column(Integer, ForeignKey("services.id", ondelete="SET NULL"))
    environment = Column(SAEnum(EnvironmentType), default=EnvironmentType.PRODUCTION)
    provider = Column(String(100))
    region = Column(String(100))
    os_version = Column(String(100))
    tags = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    service = relationship("Service", back_populates="hosts")
    timeline_entries = relationship("IncidentTimeline", back_populates="host")

    __table_args__ = (
        Index("ix_hosts_service_env", "service_id", "environment"),
    )


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    severity = Column(SAEnum(IncidentSeverity), nullable=False)
    status = Column(SAEnum(IncidentStatus), default=IncidentStatus.OPEN, nullable=False)
    service_id = Column(Integer, ForeignKey("services.id", ondelete="SET NULL"))
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"))
    reported_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    root_cause = Column(Text)
    resolution = Column(Text)
    started_at = Column(DateTime, nullable=False)
    detected_at = Column(DateTime)
    mitigated_at = Column(DateTime)
    resolved_at = Column(DateTime)
    impacted_hosts = Column(JSON, default=list)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    service = relationship("Service", back_populates="incidents")
    team = relationship("Team", back_populates="incidents")
    reporter = relationship("User")
    timeline = relationship("IncidentTimeline", back_populates="incident", order_by="IncidentTimeline.timestamp")

    __table_args__ = (
        Index("ix_incidents_severity_status", "severity", "status"),
        Index("ix_incidents_time_range", "started_at", "resolved_at"),
    )


class IncidentTimeline(Base):
    __tablename__ = "incident_timelines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    incident_id = Column(Integer, ForeignKey("incidents.id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    entry_type = Column(String(50), nullable=False)  # detection, mitigation, escalation, note
    content = Column(Text, nullable=False)
    host_id = Column(Integer, ForeignKey("hosts.id", ondelete="SET NULL"))
    actor = Column(String(255))
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    incident = relationship("Incident", back_populates="timeline")
    host = relationship("Host", back_populates="timeline_entries")


class Document(Base):
    """Tracks raw documents stored in the file system / object store."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(500), nullable=False)
    doc_type = Column(String(50), nullable=False, index=True)  # runbook, log, note, report
    source_path = Column(String(1000), nullable=False)
    summary = Column(Text)
    checksum = Column(String(64))
    metadata_json = Column(JSON, default=dict)
    ingested_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    chunk_count = Column(Integer, default=0)


class Chunk(Base):
    """Tracks individual chunks and their vector IDs."""
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    vector_id = Column(String(64), unique=True, index=True)
    token_count = Column(Integer)
    entities = Column(JSON, default=list)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
