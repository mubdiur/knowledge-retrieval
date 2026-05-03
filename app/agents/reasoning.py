"""Multi-hop reasoning engine — iterative planning with observe → refine loops."""

import hashlib
import logging
from typing import Any

from app.agents.planner import QueryPlanner
from app.models.schemas import ReasoningStep

logger = logging.getLogger(__name__)


def _content_hash(item: dict) -> str:
    """Stable content hash for deduplication."""
    content = item.get("content", "") or str(item.get("metadata", ""))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _normalize_value(v: Any) -> str:
    """Normalize a value for conflict detection."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    return str(v).strip().lower()


class Evidence:
    """A single piece of evidence with dedup and conflict tracking."""

    def __init__(self, item: dict, source: str, score: float = 0.0):
        self.item = item
        self.source = source
        self.score = score
        self.hash = _content_hash(item)

    def __repr__(self):
        return f"Evidence(source={self.source}, score={self.score:.3f})"


class Synthesizer:
    """Collates, deduplicates, and synthesizes evidence into coherent answers."""

    def __init__(self):
        self.evidence: list[Evidence] = []
        self.seen_hashes: set[str] = set()
        self.conflicts: list[dict] = []

    def add(self, item: dict, source: str, score: float = 0.0) -> None:
        """Add evidence, skipping exact duplicates."""
        ev = Evidence(item, source, score)
        if ev.hash in self.seen_hashes:
            return
        self.seen_hashes.add(ev.hash)
        self.evidence.append(ev)

    def collect(self, results: dict[int, Any]) -> None:
        """Extract evidence from all execution results."""
        for step_id, result in results.items():
            if not isinstance(step_id, int):
                continue
            if not result.get("success"):
                continue

            data = result.get("data", [])
            source = result.get("source", "unknown")
            score = result.get("score", 0.0)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        self.add(item, source, score)
            elif isinstance(data, dict):
                # Entity lookup returns {users: [...], services: [...], ...}
                for category, items in data.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                item["_entity_type"] = category
                                self.add(item, source, score)

    def group_by_source(self) -> dict[str, list[Evidence]]:
        """Group evidence by source type."""
        groups: dict[str, list[Evidence]] = {}
        for ev in self.evidence:
            groups.setdefault(ev.source, []).append(ev)
        return groups

    def find_conflicts(self) -> list[dict]:
        """Detect conflicting facts for the same entity."""
        # Group by entity name
        by_name: dict[str, list[Evidence]] = {}
        for ev in self.evidence:
            name = ev.item.get("name") or ev.item.get("hostname") or ev.item.get("title")
            if name:
                by_name.setdefault(name, []).append(ev)

        conflicts = []
        for name, evs in by_name.items():
            if len(evs) < 2:
                continue
            # Check for conflicting values on common keys
            key_values: dict[str, set[str]] = {}
            for ev in evs:
                for key in ("owner", "team", "environment", "status", "severity", "root_cause", "resolution"):
                    val = ev.item.get(key)
                    if val is not None:
                        key_values.setdefault(key, set()).add(_normalize_value(val))
            for key, vals in key_values.items():
                if len(vals) > 1:
                    conflicts.append({
                        "entity": name,
                        "field": key,
                        "values": list(vals)[:5],
                        "sources": list(set(ev.source for ev in evs)),
                    })
        return conflicts

    def incidents(self) -> list[Evidence]:
        return [e for e in self.evidence if e.source == "incidents_db"]

    def docs(self) -> list[Evidence]:
        return [e for e in self.evidence if e.source in ("vector_store", "bm25", "vector_store_logs")]

    def entities(self) -> list[Evidence]:
        return [e for e in self.evidence if e.source == "entity_db"]

    def sql(self) -> list[Evidence]:
        return [e for e in self.evidence if e.source == "postgresql"]

    def synthesize(self, query: str, query_type: str) -> str:
        """Generate a coherent answer from collected evidence."""
        if not self.evidence:
            return "I couldn't find sufficient information to answer this query. Try rephrasing or broadening the search."

        conflicts = self.find_conflicts()
        parts: list[str] = []

        # ── Causal / Multi-hop / Time-based ──────────────────────────────
        if query_type in ("causal", "multi_hop", "time_based"):
            parts.append(self._synthesize_causal(conflicts))

        # ── Factual / Relational ─────────────────────────────────────────
        elif query_type in ("factual", "relational"):
            parts.append(self._synthesize_factual(conflicts))

        # ── Exploratory ──────────────────────────────────────────────────
        elif query_type == "exploratory":
            parts.append(self._synthesize_exploratory())

        # ── Fallback ─────────────────────────────────────────────────────
        else:
            parts.append(self._synthesize_fallback())

        return "\n\n".join(parts)

    def _synthesize_causal(self, conflicts: list[dict]) -> str:
        incidents = sorted(self.incidents(), key=lambda e: e.score, reverse=True)
        docs = sorted(self.docs(), key=lambda e: e.score, reverse=True)
        parts = []

        if incidents:
            # Group incidents by service for coherence
            by_service: dict[str, list[Evidence]] = {}
            for ev in incidents[:8]:
                svc = ev.item.get("service", "Unknown service")
                by_service.setdefault(svc, []).append(ev)

            for svc, evs in by_service.items():
                parts.append(f"**{svc}**")
                for ev in evs[:3]:
                    inc = ev.item
                    title = inc.get("title", "Unnamed incident")
                    severity = inc.get("severity", "")
                    status = inc.get("status", "")
                    cause = inc.get("root_cause", "")
                    resolution = inc.get("resolution", "")

                    detail_parts = []
                    if severity:
                        detail_parts.append(f"severity: {severity}")
                    if status:
                        detail_parts.append(f"status: {status}")
                    detail = f" ({', '.join(detail_parts)})" if detail_parts else ""

                    parts.append(f"  • {title}{detail}")
                    if cause:
                        parts.append(f"    Root cause: {cause}")
                    if resolution:
                        parts.append(f"    Resolution: {resolution}")

        if docs:
            parts.append("\n**Supporting documentation:**")
            seen_snippets: set[str] = set()
            for ev in docs[:5]:
                content = ev.item.get("content", "")[:250].strip()
                if content and content not in seen_snippets:
                    seen_snippets.add(content)
                    parts.append(f"  {content}...")

        if conflicts:
            parts.append("\n**Note:** Some sources report conflicting information about " +
                         ", ".join(f"{c['entity']} ({c['field']})" for c in conflicts[:3]) + ".")

        return "\n".join(parts) if parts else "No incident data found."

    def _synthesize_factual(self, conflicts: list[dict]) -> str:
        entities = self.entities()
        sql_data = self.sql()
        parts = []

        # Build entity index for quick lookup
        entity_index: dict[str, list[Evidence]] = {}
        for ev in entities:
            name = ev.item.get("name", "")
            if name:
                entity_index.setdefault(name, []).append(ev)

        # SQL results are authoritative
        if sql_data:
            parts.append("**Database records:**")
            for ev in sql_data[:6]:
                row = ev.item
                # Build natural sentence from row
                kv_pairs = [(k, v) for k, v in row.items()
                            if not k.startswith("_") and v is not None]
                if kv_pairs:
                    # Try to form a sentence
                    name = row.get("name", "")
                    if name:
                        attrs = ", ".join(f"{k}: {v}" for k, v in kv_pairs if k != "name")
                        parts.append(f"  • **{name}** — {attrs}")
                    else:
                        attrs = ", ".join(f"{k}: {v}" for k, v in kv_pairs)
                        parts.append(f"  • {attrs}")

        # Entity results supplement
        if entities:
            services = [e for e in entities if e.item.get("_entity_type") == "services"]
            users = [e for e in entities if e.item.get("_entity_type") == "users"]
            teams = [e for e in entities if e.item.get("_entity_type") == "teams"]
            hosts = [e for e in entities if e.item.get("_entity_type") == "hosts"]

            if services and not sql_data:
                parts.append("**Services:**")
                for ev in services[:5]:
                    s = ev.item
                    name = s.get("name", "Unknown")
                    env = s.get("environment", "")
                    owner = s.get("owner", "")
                    team = s.get("team", "")
                    details = [d for d in [env, team, owner] if d]
                    parts.append(f"  • **{name}**" + (f" — {', '.join(details)}" if details else ""))

            if users and not sql_data:
                parts.append("**People:**")
                for ev in users[:5]:
                    u = ev.item
                    parts.append(f"  • **{u.get('name', '')}** — {u.get('role', '')} ({u.get('email', '')})")

            if hosts and not sql_data:
                parts.append("**Hosts:**")
                for ev in hosts[:5]:
                    h = ev.item
                    parts.append(f"  • {h.get('hostname', '')} ({h.get('ip_address', '')}) — {h.get('environment', '')}")

        if conflicts:
            parts.append("\n**Warning:** Conflicting data detected for " +
                         ", ".join(f"{c['entity']} ({c['field']})" for c in conflicts[:3]) + ".")

        return "\n".join(parts) if parts else "No matching entities found."

    def _synthesize_exploratory(self) -> str:
        groups = self.group_by_source()
        parts = []

        for source, evs in groups.items():
            if len(evs) == 0:
                continue
            label = {
                "incidents_db": "Incidents",
                "vector_store": "Documentation",
                "vector_store_logs": "Log entries",
                "bm25": "Keyword matches",
                "entity_db": "Entities",
                "postgresql": "Database records",
            }.get(source, source)

            parts.append(f"**{label}** ({len(evs)} found)")
            seen: set[str] = set()
            for ev in evs[:4]:
                content = ev.item.get("content", "")[:200].strip()
                if not content:
                    content = str(ev.item.get("title", ev.item.get("name", "")))[:200]
                if content and content not in seen:
                    seen.add(content)
                    parts.append(f"  {content}...")

        return "\n".join(parts) if parts else "No results found."

    def _synthesize_fallback(self) -> str:
        parts = []
        seen: set[str] = set()
        for ev in self.evidence[:8]:
            content = ev.item.get("content", "")[:250].strip()
            if not content:
                content = str(ev.item)[:250]
            if content not in seen:
                seen.add(content)
                parts.append(f"• {content}...")
        return "\n".join(parts) if parts else "No results found."


class ReasoningEngine:
    """Multi-hop, iterative reasoning with observe → refine → repeat loops."""

    def __init__(self, max_iterations: int = 6, confidence_threshold: float = 0.7, llm_client=None):
        self.planner = QueryPlanner(llm_client=llm_client)
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self._llm = llm_client
        self.steps: list[ReasoningStep] = []
        self._trace: list[dict] = []

    async def reason(
        self,
        query: str,
        query_type: str,
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
        top_k: int = 10,
        enable_refinement: bool = True,
    ) -> tuple[str, list[ReasoningStep]]:
        """Run the full reasoning pipeline.

        Returns (answer_text, reasoning_trace).
        """
        self.steps = []
        self._trace = []

        # ── Execute iterative planner ────────────────────────────────────
        max_iter = self.max_iterations if enable_refinement else 3
        execution_results = await self.planner.execute_iterative(
            query=query,
            query_type=query_type,
            base_filters=filters,
            base_time_range=time_range,
            max_iterations=max_iter,
            confidence_threshold=self.confidence_threshold,
            trace_log=self._trace,
        )

        # ── Convert trace to ReasoningStep models ─────────────────────────
        for entry in self._trace:
            self.steps.append(ReasoningStep(
                step=entry.get("step", 0),
                action=entry.get("action", "unknown"),
                tool=entry.get("tool"),
                input_summary=entry.get("input_summary", ""),
                output_summary=entry.get("output_summary", ""),
            ))

        # ── Synthesize answer ─────────────────────────────────────────────
        synthesizer = Synthesizer()
        synthesizer.collect(execution_results)

        # Try LLM synthesis first, fall back to rule-based
        answer = await self._synthesize_with_llm(query, query_type, synthesizer)
        if answer is None:
            answer = synthesizer.synthesize(query, query_type)

        return answer, self.steps

    async def _synthesize_with_llm(
        self, query: str, query_type: str, synthesizer: Synthesizer
    ) -> str | None:
        """Use LLM to generate a natural language answer from evidence.

        Returns None if LLM is unavailable or fails.
        """
        if self._llm is None or not synthesizer.evidence:
            return None

        # Build a compact evidence summary for the prompt
        evidence_lines = []

        incidents = synthesizer.incidents()
        if incidents:
            evidence_lines.append("INCIDENTS:")
            for ev in incidents[:5]:
                inc = ev.item
                title = inc.get("title", "")
                svc = inc.get("service", "")
                cause = inc.get("root_cause", "")
                res = inc.get("resolution", "")
                evidence_lines.append(f"- {title} (service: {svc}) cause: {cause} resolution: {res}")

        entities = synthesizer.entities()
        if entities:
            evidence_lines.append("ENTITIES:")
            for ev in entities[:5]:
                item = ev.item
                name = item.get("name", item.get("hostname", ""))
                etype = item.get("_entity_type", "")
                evidence_lines.append(f"- {name} ({etype})")

        docs = synthesizer.docs()
        if docs:
            evidence_lines.append("DOCUMENTATION:")
            for ev in docs[:3]:
                content = ev.item.get("content", "")[:200].strip()
                if content:
                    evidence_lines.append(f"- {content}")

        conflicts = synthesizer.find_conflicts()
        if conflicts:
            evidence_lines.append("CONFLICTS:")
            for c in conflicts[:2]:
                evidence_lines.append(f"- {c['entity']}: {c['field']} has values {c['values']}")

        evidence_text = "\n".join(evidence_lines)

        prompt = f"""Answer the user's question based on the evidence below. Be concise and direct.

User question: {query}
Question type: {query_type}

Evidence:
{evidence_text}

Answer in 2-4 sentences."""

        response = await self._llm.generate(prompt, max_tokens=120, temperature=0.3)
        if response:
            logger.info("LLM synthesis produced %d chars", len(response))
            return response
        return None
