"""Entity extractor — pulls key entities from text chunks."""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extract named entities from text: services, hosts, teams, timestamps, error codes.

    Uses pattern matching (no external NLP dependency for reliability/speed).
    """

    # Patterns
    HOSTNAME_RE = re.compile(r"[\w-]+\.(?:com|org|net|io|app|local|prod|staging|dev|corp|internal)")
    IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    ERROR_CODE_RE = re.compile(r"\b(?:ERR|ERROR|WARN|FATAL)[-_]?\d{3,6}\b", re.IGNORECASE)
    TIMESTAMP_RE = re.compile(
        r"\b\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
    )
    SERVICE_RE = re.compile(
        r"\b(?:service|api|backend|frontend|db|cache|queue|worker)[-_:]?([a-z0-9-]+)\b",
        re.IGNORECASE,
    )
    SEVERITY_RE = re.compile(r"\b(critical|major|minor|trivial|p0|p1|p2|p3|severe)\b", re.IGNORECASE)

    @classmethod
    def extract(cls, text: str) -> dict[str, list[str]]:
        """Extract entities from text."""
        entities: dict[str, list[str]] = {
            "hostnames": [],
            "ip_addresses": [],
            "error_codes": [],
            "timestamps": [],
            "services": [],
            "severities": [],
            "mentions": [],
        }

        # Dedup sets
        hostnames = set()
        ips = set()
        errors = set()
        timestamps = set()
        services = set()
        severities = set()

        for match in cls.HOSTNAME_RE.finditer(text):
            hostnames.add(match.group(0).lower())
        for match in cls.IP_RE.finditer(text):
            ips.add(match.group(0))
        for match in cls.ERROR_CODE_RE.finditer(text):
            errors.add(match.group(0).upper())
        for match in cls.TIMESTAMP_RE.finditer(text):
            timestamps.add(match.group(0))
        for match in cls.SERVICE_RE.finditer(text):
            services.add(match.group(0).lower())
        for match in cls.SEVERITY_RE.finditer(text):
            severities.add(match.group(0).lower())

        entities["hostnames"] = sorted(hostnames)
        entities["ip_addresses"] = sorted(ips)
        entities["error_codes"] = sorted(errors)
        entities["timestamps"] = sorted(timestamps)[:20]  # cap for large logs
        entities["services"] = sorted(services)
        entities["severities"] = sorted(severities)

        # Team mentions (@team or #team patterns)
        team_mentions = re.findall(r"@(\w+)|#(\w+)-(team| squad)", text)
        for t in team_mentions:
            name = t[0] or t[1]
            if name:
                entities["mentions"].append(name)

        return entities

    @classmethod
    def summarize(cls, text: str, max_sentences: int = 3) -> str:
        """Generate a simple extractive summary: first meaningful sentences."""
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", text)
        meaningful = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not meaningful:
            return text[:200]

        summary = " ".join(meaningful[:max_sentences])
        if len(summary) > 500:
            summary = summary[:497] + "..."
        return summary
