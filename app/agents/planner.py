"""Query planner — generates and executes dynamic multi-hop execution plans.

This is the core difference between naive RAG and a real agentic system.

Instead of:
    classify → call tools → merge → answer

It does:
    plan (generate step-wise execution graph)
    → for each step: call tool → inspect output → inject into next input
    → replan if gaps found
    → synthesize final answer

Each step's input can depend on the output of previous steps,
enabling true multi-hop: "Find incidents → extract service names →
look up owners → search docs about those owners' past incidents."
"""

import logging
from collections import defaultdict
from typing import Any, Callable

from app.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


class ExecutionStep:
    """A single step in an execution plan."""

    def __init__(
        self,
        step_id: int,
        tool: str,
        description: str,
        params_builder: Callable[[dict[int, Any]], dict[str, Any]] | None = None,
        params: dict[str, Any] | None = None,
        depends_on: list[int] | None = None,
        result_key: str | None = None,
        condition: Callable[[dict[int, Any]], bool] | None = None,
    ):
        self.step_id = step_id
        self.tool = tool
        self.description = description
        # params_builder receives all prior results and returns params dict
        self.params_builder = params_builder
        self.params = params or {}
        self.depends_on = depends_on or []
        self.result_key = result_key or f"step_{step_id}"
        # Optional: only run this step if condition(prior_results) is True
        self.condition = condition

    def build_params(self, prior_results: dict[int, Any]) -> dict[str, Any]:
        if self.params_builder:
            return {**self.params, **self.params_builder(prior_results)}
        return dict(self.params)


class ExecutionPlan:
    """A directed acyclic graph of execution steps."""

    def __init__(self, steps: list[ExecutionStep]):
        self.steps = steps
        self._validate()

    def _validate(self):
        step_ids = {s.step_id for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                assert dep in step_ids, f"Step {s.step_id} depends on missing step {dep}"

    def topological_order(self) -> list[ExecutionStep]:
        """Return steps in execution order (dependencies first)."""
        ordered = []
        visited = set()

        def visit(step: ExecutionStep):
            if step.step_id in visited:
                return
            for dep_id in step.depends_on:
                dep = next(s for s in self.steps if s.step_id == dep_id)
                visit(dep)
            visited.add(step.step_id)
            ordered.append(step)

        for step in self.steps:
            visit(step)
        return ordered

    def __repr__(self) -> str:
        lines = ["ExecutionPlan:"]
        for s in self.topological_order():
            deps = f" [after step {s.depends_on}]" if s.depends_on else ""
            lines.append(f"  {s.step_id}. {s.tool}: {s.description}{deps}")
        return "\n".join(lines)


class QueryPlanner:
    """Generates execution plans for queries based on type and content.

    Rule-based by default, with an extension point for LLM-based planning.
    The planner inspects the query, classifies it, then produces a DAG of steps
    where each step can depend on prior step outputs.
    """

    def __init__(self):
        self._plan_hooks: list[Callable] = []

    def register_planner_hook(self, hook: Callable) -> None:
        """Register an external planner (e.g. LLM-based) that can override planning."""
        self._plan_hooks.append(hook)

    # ── Plan Generation ──────────────────────────────────────────────────────

    def plan(
        self,
        query: str,
        query_type: str,
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
    ) -> ExecutionPlan:
        """Generate an execution plan for the given query.

        Tries external hooks first, falls back to built-in rules.
        """
        # Try external planners first
        for hook in self._plan_hooks:
            try:
                plan = hook(query, query_type, filters, time_range)
                if plan is not None:
                    logger.info("Used external planner hook")
                    return plan
            except Exception as e:
                logger.warning("External planner hook failed: %s", e)

        # Built-in planning
        plan_fn = getattr(self, f"_plan_{query_type}", self._plan_exploratory)
        return plan_fn(query, filters, time_range)

    # ── Built-in Plans ───────────────────────────────────────────────────────

    def _plan_factual(self, query: str, filters: dict | None, time_range: tuple | None) -> ExecutionPlan:
        """Single-step: entity lookup."""
        # Extract potential entity name from query
        name = self._extract_entity_name(query)
        return ExecutionPlan([
            ExecutionStep(
                step_id=1,
                tool="entity_lookup",
                description=f"Look up entity '{name}'",
                params={"name": name, "entity_type": None},
                result_key="entities",
            ),
        ])

    def _plan_relational(self, query: str, filters: dict | None, time_range: tuple | None) -> ExecutionPlan:
        """Multi-step: find entity → query related data."""
        name = self._extract_entity_name(query)
        steps = [
            ExecutionStep(
                step_id=1,
                tool="entity_lookup",
                description=f"Find '{name}' to identify type",
                params={"name": name},
                result_key="entity_info",
            ),
        ]

        # Step 2 depends on step 1 — use the result to choose query
        def build_sql(params_builder):
            def _build(prior):
                entity = prior.get(1, {}).get("data", {})
                # If we found a service, query its team/owner
                services = entity.get("services", [])
                if services:
                    svc_name = services[0].get("name", "")
                    return {"query": f"""
                        SELECT s.name, s.environment, u.name as owner, t.name as team
                        FROM services s
                        LEFT JOIN users u ON s.owner_id = u.id
                        LEFT JOIN teams t ON s.team_id = t.id
                        WHERE s.name ILIKE '%{svc_name}%'
                    """}
                # If we found a team, query its services
                teams = entity.get("teams", [])
                if teams:
                    team_name = teams[0].get("name", "")
                    return {"query": f"""
                        SELECT s.name, s.environment, u.name as owner
                        FROM services s
                        JOIN teams t ON s.team_id = t.id
                        JOIN users u ON s.owner_id = u.id
                        WHERE t.name ILIKE '%{team_name}%'
                    """}
                return {"query": "SELECT name, environment FROM services LIMIT 10"}
            return _build

        steps.append(
            ExecutionStep(
                step_id=2,
                tool="query_sql",
                description="Query relational data based on entity found",
                params_builder=build_sql(None),
                depends_on=[1],
                result_key="relationships",
            )
        )
        return ExecutionPlan(steps)

    def _plan_time_based(self, query: str, filters: dict | None, time_range: tuple | None) -> ExecutionPlan:
        """Two-step: get incidents → entity enrichment for each."""
        steps = [
            ExecutionStep(
                step_id=1,
                tool="get_incidents",
                description="Fetch incidents in time range",
                params={"time_range": time_range, "limit": 20},
                result_key="incidents",
            ),
        ]

        def build_enrich(params_builder):
            def _build(prior):
                incidents = prior.get(1, {}).get("data", [])
                # Extract a service name from the first incident
                if incidents:
                    svc = incidents[0].get("service")
                    if svc:
                        return {"name": svc, "entity_type": "service"}
                return {"name": "", "entity_type": None}
            return _build

        steps.append(
            ExecutionStep(
                step_id=2,
                tool="entity_lookup",
                description="Enrich with entity details about the service",
                params_builder=build_enrich(None),
                depends_on=[1],
                result_key="entity_enrichment",
                # Only run if we have incidents
                condition=lambda prior: bool(prior.get(1, {}).get("data")),
            )
        )
        return ExecutionPlan(steps)

    def _plan_causal(self, query: str, filters: dict | None, time_range: tuple | None) -> ExecutionPlan:
        """Three-step: find incidents → search docs about root cause → entity lookup."""
        svc_name = self._extract_entity_name(query)

        steps = [
            ExecutionStep(
                step_id=1,
                tool="get_incidents",
                description="Find incidents related to the query",
                params={"service": svc_name if svc_name else None, "limit": 10},
                result_key="incidents",
            ),
        ]

        def build_doc_search(params_builder):
            def _build(prior):
                incidents = prior.get(1, {}).get("data", [])
                title = ""
                if incidents:
                    title = incidents[0].get("title", "")
                return {
                    "query": f"{query} {title}",
                    "filters": filters,
                    "time_range": time_range,
                    "top_k": 15,
                }
            return _build

        steps.append(
            ExecutionStep(
                step_id=2,
                tool="search_vector",
                description="Search documentation about the incident",
                params_builder=build_doc_search(None),
                depends_on=[1],
                result_key="docs",
            )
        )

        def build_entity_lookup(params_builder):
            def _build(prior):
                incidents = prior.get(1, {}).get("data", [])
                if incidents:
                    svc = incidents[0].get("service", "")
                    if svc:
                        return {"name": svc}
                return {"name": query}
            return _build

        steps.append(
            ExecutionStep(
                step_id=3,
                tool="entity_lookup",
                description="Look up entity for ownership context",
                params_builder=build_entity_lookup(None),
                depends_on=[1],
                result_key="entity_owner",
            )
        )
        return ExecutionPlan(steps)

    def _plan_multi_hop(self, query: str, filters: dict | None, time_range: tuple | None) -> ExecutionPlan:
        """Full multi-hop: incidents → extract hosts/services → enrichment → docs.

        This is the most complex plan and closest to real multi-hop.
        """
        steps = [
            ExecutionStep(
                step_id=1,
                tool="get_incidents",
                description="Fetch incidents in time range",
                params={"time_range": time_range, "limit": 20},
                result_key="incidents",
            ),
        ]

        # Step 2: vector search for context (parallel to step 1)
        steps.append(
            ExecutionStep(
                step_id=2,
                tool="search_vector",
                description="Search for relevant documentation",
                params={"query": query, "filters": filters, "time_range": time_range, "top_k": 15},
                result_key="vector_docs",
            )
        )

        # Step 3: enrich with entity details (depends on step 1)
        def build_enrichment(params_builder):
            def _build(prior):
                incidents = prior.get(1, {}).get("data", [])
                if incidents:
                    services = list(set(
                        inc.get("service") for inc in incidents if inc.get("service")
                    ))
                    if services:
                        return {"name": services[0]}
                return {"name": self._extract_entity_name(query)}
            return _build

        steps.append(
            ExecutionStep(
                step_id=3,
                tool="entity_lookup",
                description="Look up entity from incidents",
                params_builder=build_enrichment(None),
                depends_on=[1],
                result_key="entity_detail",
            )
        )

        # Step 4: keyword search for precision terms (depends on step 1)
        def build_keyword(params_builder):
            def _build(prior):
                incidents = prior.get(1, {}).get("data", [])
                terms = []
                for inc in incidents[:3]:
                    if inc.get("root_cause"):
                        terms.append(inc["root_cause"])
                keyword_q = " ".join(terms[:3]) if terms else query
                return {"query": keyword_q, "top_k": 10}
            return _build

        steps.append(
            ExecutionStep(
                step_id=4,
                tool="search_keyword",
                description="Keyword search for precise root cause terms",
                params_builder=build_keyword(None),
                depends_on=[1],
                result_key="keyword_results",
            )
        )

        return ExecutionPlan(steps)

    def _plan_host_status(self, query: str, filters: dict | None, time_range: tuple | None) -> ExecutionPlan:
        """SQL query for hosts, optional incident enrichment."""
        steps = [
            ExecutionStep(
                step_id=1,
                tool="query_sql",
                description="Query hosts and services",
                params={"query": """
                    SELECT h.hostname, h.ip_address, h.environment, h.region, h.is_active,
                           s.name as service_name
                    FROM hosts h
                    LEFT JOIN services s ON h.service_id = s.id
                    ORDER BY s.name, h.hostname
                    LIMIT 50
                """},
                result_key="hosts",
            ),
        ]
        return ExecutionPlan(steps)

    def _plan_exploratory(self, query: str, filters: dict | None, time_range: tuple | None) -> ExecutionPlan:
        """Broad exploration: vector search + recent incidents."""
        return ExecutionPlan([
            ExecutionStep(
                step_id=1,
                tool="search_vector",
                description="Semantic search for relevant content",
                params={"query": query, "filters": filters, "time_range": time_range, "top_k": 15},
                result_key="vector_docs",
            ),
            ExecutionStep(
                step_id=2,
                tool="get_incidents",
                description="Fetch recent incidents",
                params={"time_range": time_range, "limit": 10, "severity": filters.get("severity") if filters else None},
                result_key="incidents",
            ),
        ])

    def _plan_comparative(self, query: str, filters: dict | None, time_range: tuple | None) -> ExecutionPlan:
        """Compare incidents across dimensions — requires SQL + vector."""
        return ExecutionPlan([
            ExecutionStep(
                step_id=1,
                tool="query_sql",
                description="Query incidents grouped by service",
                params={"query": """
                    SELECT s.name as service, i.severity, i.status, COUNT(*) as count
                    FROM incidents i
                    JOIN services s ON i.service_id = s.id
                    GROUP BY s.name, i.severity, i.status
                    ORDER BY s.name, i.severity
                    LIMIT 50
                """},
                result_key="sql_comparison",
            ),
            ExecutionStep(
                step_id=2,
                tool="search_vector",
                description="Search docs for comparison context",
                params={"query": query, "top_k": 10},
                result_key="vector_context",
            ),
        ])

    # ── Execution ────────────────────────────────────────────────────────────

    async def execute(
        self,
        plan: ExecutionPlan,
        base_filters: dict[str, Any] | None = None,
        base_time_range: tuple | None = None,
    ) -> dict[int, Any]:
        """Execute a plan step-by-step, injecting prior results into dependent steps.

        Returns: dict[step_id, tool_result]
        """
        results: dict[int, Any] = {}
        ordered = plan.topological_order()

        logger.info("Executing plan with %d steps", len(ordered))
        for step in ordered:
            logger.info("  Step %d: %s → %s", step.step_id, step.tool, step.description)

            # Check condition
            if step.condition and not step.condition(results):
                logger.info("  Step %d skipped (condition not met)", step.step_id)
                results[step.step_id] = {"success": True, "data": [], "skipped": True}
                continue

            # Get tool
            tool = ToolRegistry.get(step.tool)
            if not tool:
                logger.warning("  Tool '%s' not found, skipping step %d", step.tool, step.step_id)
                results[step.step_id] = {"success": False, "data": [], "error": f"Tool '{step.tool}' not found"}
                continue

            # Build params — dynamic injection from prior results
            params = step.build_params(results)
            # Inject base filters/time_range if not explicitly overridden
            if base_filters and "filters" not in params:
                params.setdefault("filters", base_filters)
            if base_time_range and "time_range" not in params:
                params.setdefault("time_range", base_time_range)

            # Execute
            try:
                result = await tool.run(**params)
                results[step.step_id] = result
                logger.info("  Step %d result: %s", step.step_id, {
                    "success": result.get("success"),
                    "count": result.get("result_count", len(result.get("data", []))),
                })
            except Exception as e:
                logger.exception("  Step %d failed: %s", step.step_id, e)
                results[step.step_id] = {"success": False, "data": [], "error": str(e)}

        return results

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_entity_name(query: str) -> str:
        """Naive entity name extraction from query text."""
        # Strip common prefixes
        q = query.strip().rstrip("?.")
        for prefix in ["who owns ", "who is ", "what is ", "find ", "about ",
                        "what services does ", "what hosts does ", "what incidents "]:
            if q.lower().startswith(prefix):
                q = q[len(prefix):]

        # Remove common stop words at the start
        for stop in ["the ", "a ", "an "]:
            if q.lower().startswith(stop):
                q = q[len(stop):]

        # If there's a quoted term, use it
        import re
        quoted = re.findall(r'"([^"]+)"', q)
        if quoted:
            return quoted[0]

        words = q.split()
        # Try uppercase words first (proper names)
        proper = [w for w in words if w[0].isupper()]
        if proper:
            return " ".join(proper[:3])

        # Fallback: words adjacent to entity-indicating keywords
        entity_keywords = {"team", "service", "host", "user", "incident", "system",
                           "gateway", "database", "server", "cluster", "api"}
        for i, w in enumerate(words):
            stripped = w.strip(",;:").lower()
            if stripped in entity_keywords:
                # Check word before the keyword (e.g. "platform team" → "platform")
                if i > 0:
                    candidate = words[i - 1].strip(",;:")
                    if len(candidate) > 2 and not candidate.lower() in {"the", "a", "an", "this", "that", "our", "your"}:
                        return candidate
                # Check word after the keyword (e.g. "service payment-gateway" → "payment-gateway")
                if i + 1 < len(words):
                    return words[i + 1].strip(",;:")

        # Look for hyphenated compound names (common in service names)
        compounds = [w.strip("?.,!:;") for w in words if "-" in w and len(w) > 3]
        if compounds:
            return compounds[0]

        # Last resort: first 2-3 meaningful alpha words
        meaningful = [w for w in words if len(w) > 2 and w.isalpha()]
        if meaningful:
            return " ".join(meaningful[:2])

        return q[:50]
