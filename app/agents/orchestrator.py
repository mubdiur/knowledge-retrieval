"""Agent orchestrator — coordinates classification, tool execution, and reasoning."""

import time
import logging
from datetime import datetime, timedelta
from typing import Any

from app.agents.classifier import QueryClassifier
from app.agents.reasoning import ReasoningEngine
from app.tools.base import ToolRegistry
from app.models.schemas import QueryResponse, SourceRef

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Top-level orchestrator that runs the full agent pipeline."""

    def __init__(self):
        self.classifier = QueryClassifier()
        self.reasoning = ReasoningEngine()

    async def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
        top_k: int = 10,
        enable_refinement: bool = True,
    ) -> QueryResponse:
        """Process a query end-to-end and return a structured response."""
        start_time = time.monotonic()

        # 1. Classify
        query_type = self.classifier.classify(query)
        needs_multi = self.classifier.needs_multi_hop(query)
        logger.info("Query classified: type=%s multi_hop=%s", query_type, needs_multi)

        # Override type if multi-hop detected
        if needs_multi and query_type not in ("causal", "time_based"):
            query_type = "multi_hop"

        # 2. Time range from query if not provided
        if time_range is None:
            time_refs = self.classifier.extract_time_references(query)
            time_range = self._resolve_time_range(time_refs)

        # 3. Run reasoning
        answer_text, reasoning_steps = await self.reasoning.reason(
            query=query,
            query_type=query_type,
            filters=filters,
            time_range=time_range,
            top_k=top_k,
            enable_refinement=enable_refinement,
        )

        # 4. Collect sources from reasoning steps
        sources = self._extract_sources(reasoning_steps, answer_text)

        elapsed = (time.monotonic() - start_time) * 1000

        return QueryResponse(
            answer=answer_text,
            sources=sources,
            reasoning_trace=reasoning_steps,
            query_type=query_type,
            latency_ms=round(elapsed, 1),
        )

    def _resolve_time_range(self, refs: dict) -> tuple | None:
        """Convert time references to concrete (start, end) datetimes."""
        now = datetime.utcnow()
        relative = refs.get("relative")
        if relative == "24h":
            return (now - timedelta(hours=24), now)
        elif relative == "7d":
            return (now - timedelta(days=7), now)
        elif relative == "30d":
            return (now - timedelta(days=30), now)
        return None

    def _extract_sources(self, steps: list, answer_text: str) -> list[SourceRef]:
        """Build source references from reasoning trace."""
        sources = []
        seen = set()
        for step in steps:
            if step.tool and step.tool not in seen:
                seen.add(step.tool)
                sources.append(SourceRef(
                    title=f"Tool: {step.tool}",
                    source=f"Step {step.step}: {step.action}",
                    snippet=step.output_summary[:200],
                    metadata={"step": step.step, "action": step.action},
                ))
        return sources
