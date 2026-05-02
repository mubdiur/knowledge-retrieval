"""Multi-hop reasoning engine — iterative planning with observe → refine loops.

Instead of static decomposition, it:
    1. Plans initial steps based on query type
    2. Executes each step, observes the output
    3. If gaps exist, dynamically generates refinement steps
    4. Repeats until confidence is high or max iterations reached
    5. Synthesizes final answer with full reasoning trace

Each trace entry includes: step number, action type, tool used,
rationale (why this step exists), and result summary.
"""

import logging
from typing import Any

from app.agents.planner import QueryPlanner
from app.models.schemas import ReasoningStep

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Multi-hop, iterative reasoning with observe → refine → repeat loops."""

    def __init__(self, max_iterations: int = 6, confidence_threshold: float = 0.7):
        self.planner = QueryPlanner()
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
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
        answer = self._synthesize(query, query_type, execution_results)

        return answer, self.steps

    def _synthesize(self, query: str, query_type: str, results: dict[int, Any]) -> str:
        """Synthesize a coherent answer from all execution results."""
        parts = []

        # Collect all evidence
        all_evidence: list[dict] = []
        sql_data: list | None = None

        for step_id, result in results.items():
            if not isinstance(step_id, int):
                continue
            if not result.get("success"):
                continue
            data = result.get("data", [])
            if result.get("source") == "postgresql":
                sql_data = data
            elif isinstance(data, list):
                all_evidence.extend(data)
            elif isinstance(data, dict):
                for category, items in data.items():
                    if isinstance(items, list):
                        all_evidence.extend(items)

        if not all_evidence and not sql_data:
            # Check refinement results
            ref_data = []
            for k, v in results.items():
                if isinstance(k, str) and k.startswith("refine"):
                    ref_data.extend(v.get("data", []))
            if ref_data:
                all_evidence = ref_data
            else:
                return "I couldn't find sufficient information to answer this query. Try rephrasing or broadening the search."

        # Type-specific synthesis
        if query_type in ("causal", "multi_hop", "time_based"):
            incidents = [e for e in all_evidence if e.get("source") == "incidents_db"]
            docs = [e for e in all_evidence if e.get("source") in ("vector_store", "bm25", "vector_store_logs")]

            if incidents:
                for inc in incidents[:5]:
                    title = inc.get("title", "Unknown incident")
                    severity = inc.get("severity", "")
                    service = inc.get("service", "")
                    cause = inc.get("root_cause", "")
                    resolution = inc.get("resolution", "")
                    entry = f"• **{title}** (severity: {severity}, service: {service})"
                    if cause:
                        entry += f"\n  Root cause: {cause}"
                    if resolution:
                        entry += f"\n  Resolution: {resolution}"
                    parts.append(entry)

            if docs:
                parts.append("\n**Supporting documentation:**")
                for doc in docs[:3]:
                    content = doc.get("content", "")[:300]
                    parts.append(f"• {content}...")

            if sql_data:
                parts.append(f"\n**Structured data:** {len(sql_data)} rows from database.")

        elif query_type in ("factual", "relational"):
            if sql_data:
                for row in sql_data[:5]:
                    parts.append(f"• {', '.join(f'{k}: {v}' for k, v in row.items())}")
            if all_evidence:
                for item in all_evidence[:5]:
                    if item.get("source") == "entity_db":
                        for etype, entities in item.get("data", {}).items():
                            if entities:
                                for e in entities[:3]:
                                    parts.append(f"• {etype.title()}: {', '.join(f'{k}={v}' for k, v in e.items())}")
                    else:
                        parts.append(f"• {item.get('content', '')[:200]}")

        elif query_type == "exploratory":
            sources: dict[str, list] = {}
            for item in all_evidence:
                src = item.get("source", "unknown")
                if src not in sources:
                    sources[src] = []
                if len(sources[src]) < 4:
                    snippet = item.get("content", "")[:200]
                    sources[src].append(snippet)

            for src, snippets in sources.items():
                parts.append(f"\n**From {src}:**")
                for snip in snippets:
                    parts.append(f"• {snip}")

        else:
            for item in all_evidence[:5]:
                content = item.get("content", str(item.get("metadata", {})))[:300]
                parts.append(f"• {content}")

        if not parts:
            return "No relevant information found for the query."

        return "\n".join(parts)
