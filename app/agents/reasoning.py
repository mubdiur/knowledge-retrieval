"""Multi-hop reasoning engine — now powered by the query planner.

Instead of static decomposition, the engine uses a dynamic planner that generates
a dependency-aware execution graph. Each step can inspect prior results and
formulate new queries based on what was found.

This enables true multi-hop: find incidents → extract service names → 
look up owners → search docs about those owners — all driven by
intermediate results, not hardcoded paths.
"""

import logging
from typing import Any

from app.agents.planner import QueryPlanner, ExecutionPlan
from app.tools.base import ToolRegistry
from app.models.schemas import ReasoningStep

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """Multi-hop, iterative reasoning over tool results using dynamic planning."""

    def __init__(self):
        self.planner = QueryPlanner()
        self.steps: list[ReasoningStep] = []
        self._execution_results: dict[int, Any] = {}

    async def reason(
        self,
        query: str,
        query_type: str,
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
        top_k: int = 10,
        enable_refinement: bool = True,
    ) -> tuple[str, list[ReasoningStep]]:
        """Run the full reasoning pipeline: plan → execute → synthesize.

        Returns (answer_text, reasoning_trace).
        """
        self.steps = []
        self._execution_results = {}

        # ── Step 1: Generate execution plan ────────────────────────────────
        self._log_step(1, "plan", None,
                       f"Generate execution plan for type={query_type}",
                       f"Query: {query[:100]}...")

        plan = self.planner.plan(query, query_type, filters, time_range)
        self._log_step(2, "plan_details", None,
                       f"Plan has {len(plan.steps)} steps",
                       str(plan)[:200])

        # ── Step 2: Execute plan (steps dispatch with dependency resolution) ─
        self._execution_results = await self.planner.execute(
            plan, base_filters=filters, base_time_range=time_range,
        )

        # Log each step's execution
        for step in plan.topological_order():
            result = self._execution_results.get(step.step_id, {})
            status = "✅" if result.get("success") else "❌"
            if result.get("skipped"):
                status = "⏭️"
            self._log_step(
                3 + step.step_id, "execute", step.tool,
                f"[{status}] {step.description}",
                f"result_count={result.get('result_count', 0)}",
            )

        # ── Step 3: Refinement loop ────────────────────────────────────────
        if enable_refinement:
            gaps = self._detect_gaps(self._execution_results)
            for r_idx, gap in enumerate(gaps[:2]):  # at most 2 refinements
                self._log_step(
                    20 + r_idx, "refine", "search_keyword",
                    f"Refinement: {gap}",
                    "Searching with refined terms",
                )
                kw_tool = ToolRegistry.get("search_keyword")
                if kw_tool:
                    ref_result = await kw_tool.run(query=gap, top_k=5)
                    if ref_result.get("success") and ref_result.get("data"):
                        self._execution_results[f"refine_{r_idx}"] = ref_result

        # ── Step 4: Synthesize answer ──────────────────────────────────────
        answer = self._synthesize(query, query_type, self._execution_results)
        self._log_step(
            30, "synthesize", None,
            "Synthesize final answer from all execution results",
            f"Generated answer ({len(answer)} chars)",
        )

        return answer, self.steps

    # ── Internal ─────────────────────────────────────────────────────────────

    def _detect_gaps(self, results: dict[int, Any]) -> list[str]:
        """Identify knowledge gaps from execution results."""
        gaps = []
        for step_id, result in results.items():
            if isinstance(step_id, int) and not result.get("success") and not result.get("skipped"):
                gaps.append(f"Step {step_id} failed: {result.get('error', 'unknown error')}")
        # If we got incidents but no docs, that's a gap
        has_incidents = any(
            r.get("result_count", 0) > 0
            for s, r in results.items()
            if isinstance(s, int) and r.get("source") == "incidents_db"
        )
        has_docs = any(
            r.get("result_count", 0) > 0
            for s, r in results.items()
            if isinstance(s, int) and r.get("source") in ("vector_store", "bm25")
        )
        if has_incidents and not has_docs:
            gaps.append("Found incidents but need documentation context")
        return gaps

    def _synthesize(self, query: str, query_type: str, results: dict[int, Any]) -> str:
        """Synthesize a coherent answer from all execution results."""
        parts = []
        seen_content = set()

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
            # Try refinement results
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
            # Group by source
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
            sources = defaultdict(list)
            for item in all_evidence:
                src = item.get("source", "unknown")
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

    def _log_step(self, step: int, action: str, tool: str | None, inp: str, out: str) -> None:
        self.steps.append(ReasoningStep(
            step=step,
            action=action,
            tool=tool,
            input_summary=inp,
            output_summary=out,
        ))


from collections import defaultdict  # noqa: E402 — needed by _synthesize
