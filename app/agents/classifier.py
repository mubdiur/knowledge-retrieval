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
        # Multi-hop indicators
        indicators = [
            "why were", "how did", "what caused", "what led to",
            "correlation", "related to", "associated with",
            "both", "and also", "as well as",
        ]
        return any(ind in query_lower for ind in indicators)

    @staticmethod
    def extract_time_references(query: str) -> dict:
        """Rudimentary time reference extraction."""
        query_lower = query.lower()
        refs = {}

        # Very basic ISO date detection
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", query)
        if dates:
            refs["dates"] = dates

        # Relative time
        if any(w in query_lower for w in ("yesterday", "last 24h", "past day")):
            refs["relative"] = "24h"
        elif any(w in query_lower for w in ("last week", "past week", "this week")):
            refs["relative"] = "7d"
        elif any(w in query_lower for w in ("last month", "past month", "this month")):
            refs["relative"] = "30d"

        return refs

    @staticmethod
    def extract_entities(query: str) -> dict[str, str]:
        """Naive entity extraction — services, hostnames, team names from query."""
        entities = {}
        # Uppercase words often signal proper names / service names
        words = query.split()
        proper = [w.strip("?.,!:;") for w in words if w[0].isupper() and len(w) > 1]
        if proper:
            entities["proper_nouns"] = proper

        # Hostname-like patterns
        hosts = re.findall(r"[\w-]+\.(?:com|org|net|io|app|local|prod|staging)", query)
        if hosts:
            entities["hostnames"] = hosts

        return entities
