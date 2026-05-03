"""Query planner — generates and executes iterative multi-hop execution plans.

This is the core difference between naive RAG and a real agentic system.

Instead of static decomposition, it runs a loop:

    plan → execute → OBSERVE → refine → repeat → early-stop

Each tool call's output is inspected. If the result contains entities that
could unlock more information, a refinement step is dynamically created.
The loop stops when confidence is high or max iterations reached.
"""

import logging
import math
import re
from typing import Any, Callable

from app.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# Lazy import to avoid loading LLM client unless needed
def _get_llm_client():
    from app.llm.client import OllamaClient
    return OllamaClient()


# ── Data Classes ──────────────────────────────────────────────────────────────

class ExecutionStep:
    """A single step in an execution plan."""

    def __init__(
        self,
        step_id: int,
        tool: str,
        description: str,
        rationale: str = "",
        params_builder: Callable[[dict[int, Any]], dict[str, Any]] | None = None,
        params: dict[str, Any] | None = None,
        depends_on: list[int] | None = None,
        result_key: str | None = None,
        condition: Callable[[dict[int, Any]], bool] | None = None,
    ):
        self.step_id = step_id
        self.tool = tool
        self.description = description
        self.rationale = rationale  # WHY this step exists — logged in trace
        self.params_builder = params_builder
        self.params = params or {}
        self.depends_on = depends_on or []
        self.result_key = result_key or f"step_{step_id}"
        self.condition = condition

    def build_params(self, prior_results: dict[int, Any]) -> dict[str, Any]:
        if self.params_builder:
            return {**self.params, **self.params_builder(prior_results)}
        return dict(self.params)


class Observation:
    """What the planner observes after executing a step."""

    def __init__(
        self,
        step_id: int,
        tool: str,
        success: bool,
        result_count: int,
        extracted_entities: list[str],
        has_data: bool,
        confidence: float,
        gap: str | None = None,
    ):
        self.step_id = step_id
        self.tool = tool
        self.success = success
        self.result_count = result_count
        self.extracted_entities = extracted_entities
        self.has_data = has_data
        self.confidence = confidence  # 0.0 - 1.0
        self.gap = gap  # Description of what's missing, if anything


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

    def append(self, step: ExecutionStep) -> None:
        self.steps.append(step)

    def __repr__(self) -> str:
        lines = ["ExecutionPlan:"]
        for s in self.topological_order():
            deps = f" [after step {s.depends_on}]" if s.depends_on else ""
            lines.append(f"  {s.step_id}. {s.tool}: {s.description}{deps}")
        return "\n".join(lines)


# ── Planner ───────────────────────────────────────────────────────────────────

class QueryPlanner:
    """Generates and iteratively executes multi-hop plans.

    The planner is callable externally but also powers the iterative loop:
        plan() → execute_step() → observe() → refine() → execute_step() → ...

    Key design:
        - Each step can inject prior results into its params
        - After each step, observe() checks data quality and extracts entities
        - If confidence is low, refine() generates the next step dynamically
        - The loop stops when confidence >= threshold or max iterations reached
        - Early stopping prevents wasted tool calls when we have enough
    """

    def __init__(self, llm_client=None):
        self._plan_hooks: list[Callable] = []
        self._llm = llm_client

    def register_planner_hook(self, hook: Callable) -> None:
        """Register an external planner (e.g. LLM-based) that can override planning."""
        self._plan_hooks.append(hook)

    def set_llm_client(self, llm_client) -> None:
        """Attach an LLM client for intelligent refinement."""
        self._llm = llm_client

    # ── Iterative Loop ───────────────────────────────────────────────────────

    async def execute_iterative(
        self,
        query: str,
        query_type: str,
        base_filters: dict[str, Any] | None = None,
        base_time_range: tuple | None = None,
        max_iterations: int = 5,
        confidence_threshold: float = 0.7,
        trace_log: list[dict] | None = None,
    ) -> dict[int, Any]:
        """Run the iterative plan → execute → observe → refine loop.

        Args:
            query: Original user query.
            query_type: Classified query type.
            base_filters: Optional global filters.
            base_time_range: Optional global time range.
            max_iterations: Maximum steps before forced stop.
            confidence_threshold: Stop when overall confidence >= this.
            trace_log: Optional list to append trace dicts to (for reasoning trace).

        Returns:
            dict[step_id, tool_result] for all executed steps.
        """
        _trace = trace_log if trace_log is not None else []

        # ── Phase 1: Generate initial plan ───────────────────────────────
        _trace.append({
            "step": len(_trace) + 1,
            "action": "plan",
            "tool": None,
            "input_summary": f"Query type={query_type}, filters={base_filters}",
            "output_summary": "Generating initial execution plan",
        })

        initial_plan = self.plan(query, query_type, base_filters, base_time_range)
        steps_executed: dict[int, Any] = {}
        step_counter = 0
        entities_found: list[str] = []

        # ── Phase 2: Iterative execution loop ────────────────────────────
        for step in initial_plan.topological_order():
            if step_counter >= max_iterations:
                break
            step_counter += 1

            result = await self._execute_single(
                step, steps_executed, base_filters, base_time_range, _trace,
            )
            if result is None:
                continue

            # Observe
            observation = self._observe(step, result, entities_found, query)
            _trace.append({
                "step": len(_trace) + 1,
                "action": "observe",
                "tool": step.tool,
                "input_summary": f"Inspecting {observation.result_count} results from step {step.step_id}",
                "output_summary": (
                    f"confidence={observation.confidence:.2f}, "
                    f"entities={observation.extracted_entities}, "
                    f"gap={observation.gap or 'none'}"
                ),
            })

            entities_found.extend(observation.extracted_entities)

            # Refinement loop: keep refining until confidence high or maxed
            while (
                observation.confidence < confidence_threshold
                and observation.gap is not None
                and step_counter < max_iterations
            ):
                step_counter += 1
                refine_step = await self._generate_refinement(
                    step_counter, observation, entities_found, query,
                )
                if refine_step is None:
                    break

                _trace.append({
                    "step": len(_trace) + 1,
                    "action": "refine",
                    "tool": refine_step.tool,
                    "input_summary": refine_step.rationale,
                    "output_summary": f"Generated refinement step using {refine_step.tool}",
                })

                result = await self._execute_single(
                    refine_step, steps_executed, base_filters, base_time_range, _trace,
                )
                if result is None:
                    break

                observation = self._observe(refine_step, result, entities_found, query)
                _trace.append({
                    "step": len(_trace) + 1,
                    "action": "observe",
                    "tool": refine_step.tool,
                    "input_summary": f"Refinement step {refine_step.step_id} produced {observation.result_count} results",
                    "output_summary": (
                        f"confidence={observation.confidence:.2f}, "
                        f"gap={observation.gap or 'none'}"
                    ),
                })
                entities_found.extend(observation.extracted_entities)

        # ── Phase 3: Final assessment ────────────────────────────────────
        overall_confidence = self._assess_confidence(steps_executed)
        _trace.append({
            "step": len(_trace) + 1,
            "action": "assess",
            "tool": None,
            "input_summary": f"After {len(steps_executed)} steps",
            "output_summary": f"Overall confidence={overall_confidence:.2f}, "
                              f"entities={list(set(entities_found))}, "
                              f"stopped={'threshold met' if overall_confidence >= confidence_threshold else 'max iterations' if step_counter >= max_iterations else 'plan complete'}",
        })

        return steps_executed

    # ── Single Step Execution ─────────────────────────────────────────────

    async def _execute_single(
        self,
        step: ExecutionStep,
        results_so_far: dict[int, Any],
        base_filters: dict[str, Any] | None,
        base_time_range: tuple | None,
        trace_log: list[dict],
    ) -> dict[str, Any] | None:
        """Execute one step, log to trace, return result."""
        # Check condition
        if step.condition and not step.condition(results_so_far):
            logger.info("  Step %d skipped (condition not met)", step.step_id)
            trace_log.append({
                "step": len(trace_log) + 1,
                "action": "skip",
                "tool": step.tool,
                "input_summary": step.rationale or step.description,
                "output_summary": "Condition not met — skipped",
            })
            results_so_far[step.step_id] = {"success": True, "data": [], "skipped": True}
            return None

        # Get tool
        tool = ToolRegistry.get(step.tool)
        if not tool:
            logger.warning("  Tool '%s' not found", step.tool)
            results_so_far[step.step_id] = {"success": False, "data": [], "error": f"Tool '{step.tool}' not found"}
            return None

        # Build params from prior results
        params = step.build_params(results_so_far)
        if base_filters and "filters" not in params:
            params.setdefault("filters", base_filters)
        if base_time_range and "time_range" not in params:
            params.setdefault("time_range", base_time_range)

        # Execute
        logger.info("  Step %d: %s → %s", step.step_id, step.tool, step.description)
        try:
            result = await tool.run(**params)
            results_so_far[step.step_id] = result
            count = result.get("result_count", len(result.get("data", [])))
            trace_log.append({
                "step": len(trace_log) + 1,
                "action": "execute",
                "tool": step.tool,
                "input_summary": step.rationale or step.description,
                "output_summary": f"Got {count} results (success={result.get('success')})",
            })
            return result
        except Exception as e:
            logger.exception("  Step %d failed", step.step_id)
            results_so_far[step.step_id] = {"success": False, "data": [], "error": str(e)}
            trace_log.append({
                "step": len(trace_log) + 1,
                "action": "execute",
                "tool": step.tool,
                "input_summary": step.rationale or step.description,
                "output_summary": f"Failed: {str(e)[:200]}",
            })
            return None

    # ── Observation ───────────────────────────────────────────────────────

    def _observe(
        self,
        step: ExecutionStep,
        result: dict[str, Any],
        existing_entities: list[str],
        original_query: str,
    ) -> Observation:
        """Inspect tool output: extract entities, assess confidence, detect gaps."""
        if not result.get("success"):
            return Observation(
                step_id=step.step_id,
                tool=step.tool,
                success=False,
                result_count=0,
                extracted_entities=[],
                has_data=False,
                confidence=0.0,
                gap="Tool call failed",
            )

        data = result.get("data", [])
        count = result.get("result_count", len(data))
        has_data = count > 0

        # Extract entities from this result
        entities = self._extract_entities_from_result(result)

        # Confidence scoring
        confidence = self._score_confidence(step, result, count)

        # Gap detection
        gap = self._detect_gap(step, result, count, existing_entities, original_query)

        return Observation(
            step_id=step.step_id,
            tool=step.tool,
            success=True,
            result_count=count,
            extracted_entities=entities,
            has_data=has_data,
            confidence=confidence,
            gap=gap,
        )

    def _score_confidence(self, step: ExecutionStep, result: dict[str, Any], count: int) -> float:
        """Score confidence in this result on 0.0-1.0.

        Factors:
            - Has data at all
            - Result quality (log-saturating: 1 result = 0.5, 3 = 0.7, 10 = 0.85)
            - Source reliability (SQL structured > incidents > vector > BM25 > entity)
        """
        if not result.get("success"):
            return 0.0

        if count == 0:
            return 0.05  # Empty but successful

        # Quality score: log-saturating so 3 great results > 10 weak ones
        count_score = min(math.log1p(count) / math.log1p(10), 0.85)

        # Source reliability tier (smaller, additive)
        source = result.get("source", "")
        source_bonus = {
            "postgresql": 0.10,
            "incidents_db": 0.08,
            "vector_store": 0.06,
            "vector_store_logs": 0.06,
            "bm25": 0.04,
            "entity_db": 0.02,
        }.get(source, 0.0)

        return round(min(count_score + source_bonus, 1.0), 3)

    def _detect_gap(
        self,
        step: ExecutionStep,
        result: dict[str, Any],
        count: int,
        existing_entities: list[str],
        original_query: str,
    ) -> str | None:
        """Detect what information is still missing.

        Returns a description of the gap, or None if no gap.
        """
        if count == 0:
            return "No results found"

        data = result.get("data", [])

        # If we got incidents but no root cause, that's a gap
        if result.get("source") == "incidents_db":
            has_root_causes = any(
                inc.get("root_cause") for inc in (data if isinstance(data, list) else [])
            )
            if not has_root_causes:
                return "Incidents found but root cause information missing"

        # If we got doc results but entities are sparse
        if result.get("source") in ("vector_store", "bm25") and count > 0:
            if not existing_entities:
                return "Need to identify entities (services, teams) mentioned in these results"

        # If original query asks for ownership and we haven't found it
        if any(w in original_query.lower() for w in ("who owns", "owner", "responsible")):
            if not any(e for e in existing_entities if e in str(data).lower()):
                return "Need to identify owner/team for entity"

        return None

    @staticmethod
    def _extract_entities_from_result(result: dict[str, Any]) -> list[str]:
        """Extract entity names from a tool result."""
        entities = []
        data = result.get("data", [])

        if isinstance(data, list):
            for item in data:
                # Check various entity fields
                for key in ("name", "service", "team", "hostname", "service_name", "title"):
                    val = item.get(key)
                    if val and isinstance(val, str) and len(val) > 1:
                        entities.append(val)
        elif isinstance(data, dict):
            for category, items in data.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            for key in ("name", "hostname", "service"):
                                val = item.get(key)
                                if val and isinstance(val, str):
                                    entities.append(val)

        return list(set(entities))

    # ── Refinement Generation ─────────────────────────────────────────────

    async def _generate_refinement_llm(
        self,
        step_id: int,
        observation: Observation,
        entities_found: list[str],
        original_query: str,
    ) -> ExecutionStep | None:
        """Use LLM to intelligently decide the next refinement step."""
        if self._llm is None:
            return None

        gap = observation.gap or "No gap detected"
        tools_available = [
            "search_vector — semantic search over documents/logs",
            "search_keyword — exact BM25 keyword search",
            "entity_lookup — find people, services, teams, hosts by name",
            "query_sql — execute SELECT queries on structured data",
            "get_incidents — query incidents by time, severity, service, team",
        ]

        prompt = f"""You are a query planner for an incident knowledge system.

Original query: {original_query}
Gap detected: {gap}
Entities found so far: {entities_found[-5:] if entities_found else 'none'}
Last tool used: {observation.tool}
Results count: {observation.result_count}

Available tools:
{chr(10).join(tools_available)}

Respond with EXACTLY one line in this format:
TOOL: <tool_name> | QUERY: <search query or entity name> | RATIONALE: <brief reason>

Choose the tool and query that best addresses the gap. Be specific."""

        response = await self._llm.generate(prompt, max_tokens=60, temperature=0.2)
        if not response:
            return None

        # Parse the response
        try:
            # Extract tool, query, rationale from the response line
            tool_match = re.search(r"TOOL:\s*(\w+)", response, re.I)
            query_match = re.search(r"QUERY:\s*([^|]+)", response, re.I)
            rationale_match = re.search(r"RATIONALE:\s*(.+)", response, re.I)

            tool_name = tool_match.group(1).strip().lower() if tool_match else "search_vector"
            query_str = query_match.group(1).strip() if query_match else original_query
            rationale = rationale_match.group(1).strip() if rationale_match else f"LLM refinement for: {gap}"

            # Map tool names
            tool_map = {
                "search_vector": "search_vector",
                "search_keyword": "search_keyword",
                "entity_lookup": "entity_lookup",
                "query_sql": "query_sql",
                "get_incidents": "get_incidents",
            }
            tool_name = tool_map.get(tool_name, "search_vector")

            # Build params based on tool
            params = {"query": query_str}
            if tool_name in ("search_vector", "search_keyword"):
                params["top_k"] = 10
            elif tool_name == "entity_lookup":
                params = {"name": query_str}
            elif tool_name == "get_incidents":
                params = {"limit": 10}

            return ExecutionStep(
                step_id=step_id,
                tool=tool_name,
                description=f"LLM refinement: {query_str[:60]}",
                rationale=rationale[:200],
                params=params,
                result_key=f"refine_{step_id}",
            )
        except Exception as e:
            logger.warning("LLM refinement parse failed: %s", e)
            return None

    async def _generate_refinement(
        self,
        step_id: int,
        observation: Observation,
        entities_found: list[str],
        original_query: str,
    ) -> ExecutionStep | None:
        """Dynamically generate a refinement step based on observation.

        Tries LLM first for intelligent refinement, falls back to rule-based.
        """
        gap = observation.gap
        if gap is None:
            return None

        # Try LLM-driven refinement first
        if self._llm is not None:
            llm_step = await self._generate_refinement_llm(step_id, observation, entities_found, original_query)
            if llm_step is not None:
                logger.info("Using LLM-driven refinement: %s", llm_step.tool)
                return llm_step

        # Gap: incidents without root cause → search for related docs
        if "root cause" in (gap or "").lower():
            def build_rc_search(ents):
                def _build(prior):
                    query_str = original_query
                    if ents:
                        query_str = f"{' '.join(ents[-3:])} root cause"
                    return {"query": query_str, "top_k": 10}
                return _build

            return ExecutionStep(
                step_id=step_id,
                tool="search_vector",
                description="Search for root cause documentation",
                rationale=f"Incidents found but missing root cause. Searching docs with entities: {entities_found[-3:] if entities_found else original_query}",
                params_builder=build_rc_search(entities_found),
                result_key=f"refine_{step_id}",
            )

        # Gap: need entity identification → entity lookup
        if "entities" in (gap or "").lower() or "identify" in (gap or "").lower():
            # Extract a candidate name from the query
            name = self._extract_entity_name(original_query)

            return ExecutionStep(
                step_id=step_id,
                tool="entity_lookup",
                description=f"Look up entity: {name}",
                rationale=f"Results mention unknown entities. Looking up '{name}' to identify service/team context",
                params={"name": name},
                result_key=f"refine_{step_id}",
            )

        # Gap: need owner → SQL query for ownership info
        if "owner" in (gap or "").lower():
            name = self._extract_entity_name(original_query)
            # Build a SQL query to find the owner
            def build_owner_sql(ent_name):
                def _build(prior):
                    # Check if entity_lookup already ran and found the entity
                    for k, v in prior.items():
                        if isinstance(k, int):
                            data = v.get("data", {})
                            if isinstance(data, dict):
                                services = data.get("services", [])
                                if services:
                                    svc = services[0].get("name", ent_name)
                                    return {
                                        "query": """
                                            SELECT s.name, u.name as owner_name, u.email as owner_email,
                                                   t.name as team_name
                                            FROM services s
                                            LEFT JOIN users u ON s.owner_id = u.id
                                            LEFT JOIN teams t ON s.team_id = t.id
                                            WHERE s.name ILIKE :pattern
                                            LIMIT 5
                                        """,
                                        "params": {"pattern": f"%{svc}%"},
                                    }
                    # Fallback: use extracted name
                    return {
                        "query": """
                            SELECT s.name, u.name as owner_name, u.email as owner_email, t.name as team_name
                            FROM services s
                            LEFT JOIN users u ON s.owner_id = u.id
                            LEFT JOIN teams t ON s.team_id = t.id
                            WHERE s.name ILIKE :pattern
                            LIMIT 5
                        """,
                        "params": {"pattern": f"%{ent_name}%"},
                    }
                return _build
            return ExecutionStep(
                step_id=step_id, tool="query_sql",
                description=f"Find owner for: {name}",
                rationale=f"Entity found but missing owner details. Querying database for ownership relationships.",
                params_builder=build_owner_sql(name),
                result_key=f"refine_{step_id}",
            )

        # Generic: search_keyword for the gap terms
        return ExecutionStep(
            step_id=step_id,
            tool="search_keyword",
            description=f"Keyword search: {gap[:80]}",
            rationale=f"Gap detected: '{gap}'. Using keyword search for precision.",
            params={"query": gap, "top_k": 10},
            result_key=f"refine_{step_id}",
        )

    # ── Final Confidence Assessment ───────────────────────────────────────

    def _assess_confidence(self, results: dict[int, Any]) -> float:
        """Overall confidence across all executed steps (0.0-1.0).

        Uses the maximum step confidence weighted by source reliability,
        not an average of raw counts.
        """
        if not results:
            return 0.0

        scores = []
        for step_id, result in results.items():
            if isinstance(step_id, int) and result.get("success"):
                count = result.get("result_count", len(result.get("data", [])))
                score = self._score_confidence(
                    ExecutionStep(step_id, result.get("source", "unknown"), ""),
                    result, count,
                )
                scores.append(score)

        if not scores:
            return 0.0

        # Use max confidence, not average — one strong result is enough
        return round(max(scores), 3)

    # ── Plan Generation ──────────────────────────────────────────────────────

    def plan(
        self,
        query: str,
        query_type: str,
        filters: dict[str, Any] | None = None,
        time_range: tuple | None = None,
    ) -> "ExecutionPlan":
        """Generate an execution plan for the given query.

        Tries external hooks first, falls back to built-in rules.
        """
        

        for hook in self._plan_hooks:
            try:
                plan = hook(query, query_type, filters, time_range)
                if plan is not None:
                    logger.info("Used external planner hook")
                    return plan
            except Exception as e:
                logger.warning("External planner hook failed: %s", e)

        plan_fn = getattr(self, f"_plan_{query_type}", self._plan_exploratory)
        # If plan_fn is still the fallback, try extracting enum value
        if plan_fn == self._plan_exploratory and hasattr(query_type, "value"):
            plan_fn = getattr(self, f"_plan_{query_type.value}", self._plan_exploratory)
        return plan_fn(query, filters, time_range)

    # ── Built-in Plans ───────────────────────────────────────────────────────

    def _plan_factual(self, query: str, filters: dict | None, time_range: tuple | None):
        
        name = self._extract_entity_name(query)
        return ExecutionPlan([
            ExecutionStep(
                step_id=1, tool="entity_lookup",
                description=f"Look up entity '{name}'",
                rationale=f"Query asks about '{name}'. Direct entity lookup is the fastest path.",
                params={"name": name, "entity_type": None},
                result_key="entities",
            ),
        ])

    def _plan_relational(self, query: str, filters: dict | None, time_range: tuple | None):
        
        name = self._extract_entity_name(query)
        steps = [
            ExecutionStep(
                step_id=1, tool="entity_lookup",
                description=f"Find '{name}' to identify type",
                rationale="Need to know if this is a service, team, or user before querying relationships.",
                params={"name": name},
                result_key="entity_info",
            ),
        ]

        def build_sql(prior):
            entity = prior.get(1, {}).get("data", {})
            services = entity.get("services", [])
            if services:
                svc_name = services[0].get("name", "")
                return {
                    "query": """
                        SELECT s.name, s.environment, u.name as owner, t.name as team
                        FROM services s
                        LEFT JOIN users u ON s.owner_id = u.id
                        LEFT JOIN teams t ON s.team_id = t.id
                        WHERE s.name ILIKE :pattern
                        LIMIT 20
                    """,
                    "params": {"pattern": f"%{svc_name}%"},
                }
            teams = entity.get("teams", [])
            if teams:
                team_name = teams[0].get("name", "")
                return {
                    "query": """
                        SELECT s.name, s.environment, u.name as owner
                        FROM services s
                        JOIN teams t ON s.team_id = t.id
                        JOIN users u ON s.owner_id = u.id
                        WHERE t.name ILIKE :pattern
                        LIMIT 20
                    """,
                    "params": {"pattern": f"%{team_name}%"},
                }
            return {"query": "SELECT name, environment FROM services LIMIT 10", "params": {}}

        steps.append(
            ExecutionStep(
                step_id=2, tool="query_sql",
                description="Query relational data based on entity found",
                rationale="Entity identified. Now querying the database for relationships.",
                params_builder=build_sql,
                depends_on=[1],
                result_key="relationships",
            )
        )
        return ExecutionPlan(steps)

    def _plan_time_based(self, query: str, filters: dict | None, time_range: tuple | None):
        
        steps = [
            ExecutionStep(
                step_id=1, tool="get_incidents",
                description="Fetch incidents in time range",
                rationale="Time-based query. First get all incidents in the window.",
                params={"time_range": time_range, "limit": 20},
                result_key="incidents",
            ),
        ]

        def build_enrich(prior):
            incidents = prior.get(1, {}).get("data", [])
            if incidents:
                svc = incidents[0].get("service")
                if svc:
                    return {"name": svc, "entity_type": "service"}
            return {"name": "", "entity_type": None}

        steps.append(
            ExecutionStep(
                step_id=2, tool="entity_lookup",
                description="Enrich with entity details about the service",
                rationale="Incidents found. Enriching with service ownership for context.",
                params_builder=build_enrich,
                depends_on=[1],
                result_key="entity_enrichment",
                condition=lambda prior: bool(prior.get(1, {}).get("data")),
            )
        )
        return ExecutionPlan(steps)

    def _plan_causal(self, query: str, filters: dict | None, time_range: tuple | None):
        
        svc_name = self._extract_entity_name(query)
        steps = [
            ExecutionStep(
                step_id=1, tool="get_incidents",
                description="Find incidents related to the query",
                rationale="Causal analysis starts with finding the incidents.",
                params={"service": svc_name if svc_name else None, "limit": 10},
                result_key="incidents",
            ),
        ]

        def build_doc_search(prior):
            incidents = prior.get(1, {}).get("data", [])
            title = incidents[0].get("title", "") if incidents else ""
            return {"query": f"{query} {title}", "filters": filters, "time_range": time_range, "top_k": 15}

        steps.append(
            ExecutionStep(
                step_id=2, tool="search_vector",
                description="Search documentation about the incident",
                rationale="Incidents identified. Searching docs for root cause and context.",
                params_builder=build_doc_search,
                depends_on=[1],
                result_key="docs",
            )
        )

        def build_entity_lookup(prior):
            incidents = prior.get(1, {}).get("data", [])
            if incidents:
                svc = incidents[0].get("service", "")
                if svc:
                    return {"name": svc}
            return {"name": query}

        steps.append(
            ExecutionStep(
                step_id=3, tool="entity_lookup",
                description="Look up entity for ownership context",
                rationale="Documentation found. Now identifying who owns the affected service.",
                params_builder=build_entity_lookup,
                depends_on=[1],
                result_key="entity_owner",
            )
        )
        return ExecutionPlan(steps)

    def _plan_multi_hop(self, query: str, filters: dict | None, time_range: tuple | None):
        
        steps = [
            ExecutionStep(
                step_id=1, tool="get_incidents",
                description="Fetch incidents in time range",
                rationale="Multi-hop: start with incidents to establish the event timeline.",
                params={"time_range": time_range, "limit": 20},
                result_key="incidents",
            ),
            ExecutionStep(
                step_id=2, tool="search_vector",
                description="Search for relevant documentation",
                rationale="Running doc search in parallel with incident fetch (no dependency).",
                params={"query": query, "filters": filters, "time_range": time_range, "top_k": 15},
                result_key="vector_docs",
            ),
        ]

        def build_enrichment(prior):
            incidents = prior.get(1, {}).get("data", [])
            if incidents:
                services = list(set(
                    inc.get("service") for inc in incidents if inc.get("service")
                ))
                if services:
                    return {"name": services[0]}
            return {"name": self._extract_entity_name(query)}

        steps.append(
            ExecutionStep(
                step_id=3, tool="entity_lookup",
                description="Look up entity from incidents",
                rationale="Incidents mention services. Enriching with ownership details.",
                params_builder=build_enrichment,
                depends_on=[1],
                result_key="entity_detail",
            )
        )

        def build_keyword(prior):
            incidents = prior.get(1, {}).get("data", [])
            terms = []
            for inc in incidents[:3]:
                if inc.get("root_cause"):
                    terms.append(inc["root_cause"])
            return {"query": " ".join(terms[:3]) if terms else query, "top_k": 10}

        steps.append(
            ExecutionStep(
                step_id=4, tool="search_keyword",
                description="Keyword search for precise root cause terms",
                rationale="Root cause terms extracted from incidents. Using keyword search for precision matching.",
                params_builder=build_keyword,
                depends_on=[1],
                result_key="keyword_results",
            )
        )
        return ExecutionPlan(steps)

    def _plan_host_status(self, query: str, filters: dict | None, time_range: tuple | None):
        
        return ExecutionPlan([
            ExecutionStep(
                step_id=1, tool="query_sql",
                description="Query hosts and services",
                rationale="Direct SQL query for host inventory — fastest path to host data.",
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
        ])

    def _plan_exploratory(self, query: str, filters: dict | None, time_range: tuple | None):
        
        return ExecutionPlan([
            ExecutionStep(
                step_id=1, tool="search_vector",
                description="Semantic search for relevant content",
                rationale="Exploratory query. Semantic search casts the widest net.",
                params={"query": query, "filters": filters, "time_range": time_range, "top_k": 15},
                result_key="vector_docs",
            ),
            ExecutionStep(
                step_id=2, tool="get_incidents",
                description="Fetch recent incidents",
                rationale="Also fetching recent incidents for complete picture.",
                params={"time_range": time_range, "limit": 10, "severity": filters.get("severity") if filters else None},
                result_key="incidents",
            ),
        ])

    def _plan_comparative(self, query: str, filters: dict | None, time_range: tuple | None):
        
        return ExecutionPlan([
            ExecutionStep(
                step_id=1, tool="query_sql",
                description="Query incidents grouped by service",
                rationale="Comparison requires structured aggregation. SQL is the right tool.",
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
                step_id=2, tool="search_vector",
                description="Search docs for comparison context",
                rationale="Structured data retrieved. Adding doc context for qualitative comparison.",
                params={"query": query, "top_k": 10},
                result_key="vector_context",
            ),
        ])

    # ── Entity Extraction ─────────────────────────────────────────────────

    @staticmethod
    def _extract_entity_name(query: str) -> str:
        """Extract entity name from a query string."""
        q = query.strip().rstrip("?.")
        # Strip question/query prefixes — they're not entity names
        for prefix in [
            "who owns ", "who is ", "what is ", "find ", "about ",
            "what services does ", "what hosts does ", "what incidents ",
            "tell me about ", "show me ",
            "what caused the ", "what caused ", "why did ", "why was ",
            "how did ", "how was ",
        ]:
            if q.lower().startswith(prefix):
                q = q[len(prefix):]

        for stop in ["the ", "a ", "an ", "this ", "that "]:
            if q.lower().startswith(stop):
                q = q[len(stop):]

        # If there's a quoted term, use it
        quoted = re.findall(r'"([^"]+)"', q)
        if quoted:
            return quoted[0]

        words = q.split()
        # Skip interrogative words at the start
        interrogatives = {"what", "who", "why", "how", "when", "where", "which", "did", "does", "is", "are", "was", "were"}
        while words and words[0].lower().strip("?.,!:;") in interrogatives:
            words = words[1:]

        if not words:
            return ""

        # Try uppercase words first (proper names)
        proper = [w for w in words if w[0].isupper()]
        if proper:
            return " ".join(proper[:3])

        # Look for hyphenated compound names (common in service names)
        compounds = [w.strip("?.,!:;") for w in words if "-" in w and len(w) > 3]
        if compounds:
            return compounds[0]

        # Fallback: words adjacent to entity-indicating keywords
        entity_kw = {"team", "service", "host", "user", "incident", "system",
                      "gateway", "database", "server", "cluster", "api"}
        for i, w in enumerate(words):
            stripped = w.strip(",;:").lower()
            if stripped in entity_kw:
                if i > 0:
                    candidate = words[i - 1].strip(",;:")
                    if len(candidate) > 2 and candidate.lower() not in {"the","a","an","this","that","our","your"}:
                        if "-" in candidate:
                            return candidate
                        return f"{candidate}-{stripped}"  # reconstruct compound name
                if i + 1 < len(words):
                    return words[i + 1].strip(",;:")

        # Last resort: first 2 meaningful alpha words
        meaningful = [w for w in words if len(w) > 2 and w.isalpha()]
        if meaningful:
            return " ".join(meaningful[:2])

        return q[:50]
