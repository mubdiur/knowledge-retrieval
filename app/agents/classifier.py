"""Query classifier — categorizes incoming questions by type and intent."""

import re
from enum import Enum


class QueryType(str, Enum):
    FACTUAL = "factual"               # "Who owns service X?"
    RELATIONAL = "relational"         # "What services does team Y own?"
    TIME_BASED = "time_based"         # "What incidents happened last week?"
    CAUSAL = "causal"                 # "What caused incident Z?"
    EXPLORATORY = "exploratory"       # "Show me recent critical incidents"
    COMPARATIVE = "comparative"       # "Compare incidents in prod vs staging"
    HOST_STATUS = "host_status"       # "Are the hosts healthy?"
    HOP = "multi_hop"                 # "Why were hosts down between X and Y?" (links incidents→hosts)


# Pattern-based classification rules
CLASSIFIER_RULES = [
    (r"who\s+owns|who\s+is\s+(responsible|oncall\s*|the\s+owner|the\s+contact|on\s+call)", QueryType.FACTUAL),
    (r"what\s+(services|teams|hosts|systems).*(team|belong|owned)", QueryType.RELATIONAL),
    (r"(between|during|from|since|last|past|yesterday|today|this\s+week|this\s+month)", QueryType.TIME_BASED),
    (r"(why|root\s*cause|caused|triggered|led\s+to)", QueryType.CAUSAL),
    (r"(show|list|find|get|all|recent|latest)\s+(incident|critical|major)", QueryType.EXPLORATORY),
    (r"(compare|versus|vs|difference|more\s+than|less\s+than)", QueryType.COMPARATIVE),
    (r"(down|healthy|unreachable|offline|status|up\s+and\s+running)", QueryType.HOST_STATUS),
    (r"(why\s+were|how\s+did|what\s+caused|what\s+led\s+to).*(down|outage|failure|incident)", QueryType.HOP),
]

# Common service-name patterns in enterprise infra
_SERVICE_SUFFIXES = [
    r"-api", r"-svc", r"-service", r"-gateway", r"-worker",
    r"-db", r"-cache", r"-queue", r"-frontend", r"-backend",
    r"_api", r"_svc", r"_service",
]

_ERROR_CODE_PATTERNS = [
    r"\b[45]\d{2}\b",                    # HTTP status codes
    r"\bERR_[A-Z_]+\b",                  # Error constants
    r"\b[A-Z]+-\d{4,6}\b",               # Vendor error codes
    r"\b0x[0-9a-fA-F]{8}\b",             # Hex error codes
]

_IP_PATTERNS = [
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",      # IPv4
    r"\b[0-9a-fA-F:]{4,39}\b",           # IPv6 (loose)
]

_HOSTNAME_PATTERNS = [
    r"[\w-]+\.(?:com|org|net|io|app|local|prod|staging|dev|internal)",
    r"[\w-]+-\d+\.[\w.-]+",              # pod-123.namespace.svc
]


class QueryClassifier:
    """Classifies a natural language query into a QueryType."""

    @staticmethod
    def classify(query: str) -> QueryType:
        """Determine query type based on linguistic patterns."""
        query_lower = query.lower().strip()

        # Check patterns
        for pattern, qtype in CLASSIFIER_RULES:
            if re.search(pattern, query_lower):
                return qtype

        # Default heuristics
        if len(query_lower.split()) <= 5 and any(
            kw in query_lower for kw in ("who", "what", "when", "where")
        ):
            return QueryType.FACTUAL

        return QueryType.EXPLORATORY

    @staticmethod
    def needs_multi_hop(query: str) -> bool:
        """Detect if a query requires multi-hop reasoning (linking >1 entities)."""
        query_lower = query.lower()
        indicators = [
            "why were", "how did", "what caused", "what led to",
            "correlation", "related to", "associated with",
            "both", "and also", "as well as",
        ]
        return any(ind in query_lower for ind in indicators)

    @staticmethod
    def extract_time_references(query: str) -> dict:
        """Extract time references from query."""
        query_lower = query.lower()
        refs = {}

        # ISO dates
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", query)
        if dates:
            refs["dates"] = dates

        # Relative time
        if any(w in query_lower for w in ("yesterday", "last 24h", "past day", "24 hours")):
            refs["relative"] = "24h"
        elif any(w in query_lower for w in ("last week", "past week", "this week", "7 days")):
            refs["relative"] = "7d"
        elif any(w in query_lower for w in ("last month", "past month", "this month", "30 days")):
            refs["relative"] = "30d"

        return refs

    @staticmethod
    def extract_entities(query: str) -> dict[str, list[str]]:
        """Extract entities from query using multiple pattern types.

        Returns:
            dict with keys: services, hostnames, ips, error_codes, proper_nouns, teams
        """
        entities: dict[str, list[str]] = {
            "services": [],
            "hostnames": [],
            "ips": [],
            "error_codes": [],
            "proper_nouns": [],
            "teams": [],
        }

        # 1. Quoted strings (highest confidence)
        quoted = re.findall(r'"([^"]+)"', query)
        for q in quoted:
            if any(s in q.lower() for s in _SERVICE_SUFFIXES) or "-" in q:
                entities["services"].append(q)
            else:
                entities["proper_nouns"].append(q)

        # 2. Service-name patterns (also proper nouns)
        words = query.split()
        for w in words:
            w_clean = w.strip(",.;:?!")
            if any(re.search(s + r"\b", w_clean, re.I) for s in _SERVICE_SUFFIXES):
                if w_clean not in entities["services"]:
                    entities["services"].append(w_clean)
                if w_clean not in entities["proper_nouns"]:
                    entities["proper_nouns"].append(w_clean)

        # 3. Hostnames
        for pattern in _HOSTNAME_PATTERNS:
            matches = re.findall(pattern, query)
            for m in matches:
                if m not in entities["hostnames"]:
                    entities["hostnames"].append(m)

        # 4. IP addresses
        for pattern in _IP_PATTERNS:
            matches = re.findall(pattern, query)
            for m in matches:
                if m not in entities["ips"]:
                    entities["ips"].append(m)

        # 5. Error codes
        for pattern in _ERROR_CODE_PATTERNS:
            matches = re.findall(pattern, query)
            for m in matches:
                if m not in entities["error_codes"]:
                    entities["error_codes"].append(m)

        # 6. Proper nouns (uppercase words, excluding sentence-start)
        for i, w in enumerate(words):
            w_clean = w.strip(",.;:?!")
            if len(w_clean) <= 1:
                continue
            # Skip first word if it starts the sentence (could just be capitalized)
            if i == 0 and w_clean[0].isupper() and w_clean[1:].islower():
                continue
            # Skip common words
            if w_clean.lower() in {"the", "a", "an", "this", "that", "it", "is", "are", "was", "were"}:
                continue
            if w_clean[0].isupper() and not any(w_clean in v for v in entities.values()):
                entities["proper_nouns"].append(w_clean)

        # 7. Team names ("team X" or "X team")
        team_refs = re.findall(r"(?:team|squad)\s+([A-Z][\w-]+)", query, re.I)
        team_refs += re.findall(r"([A-Z][\w-]+)\s+(?:team|squad)", query, re.I)
        for t in team_refs:
            if t not in entities["teams"]:
                entities["teams"].append(t)

        # Deduplicate within each list
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        return entities
