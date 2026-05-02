"""Tests for the query planner — the core multi-hop engine."""

import pytest

from app.agents.planner import QueryPlanner, ExecutionPlan, ExecutionStep


class TestExecutionPlan:
    def test_topological_order_simple(self):
        plan = ExecutionPlan([
            ExecutionStep(1, "tool_a", "first", params={"p": 1}),
            ExecutionStep(2, "tool_b", "second", params={"p": 2}, depends_on=[1]),
        ])
        order = plan.topological_order()
        assert order[0].step_id == 1
        assert order[1].step_id == 2

    def test_topological_order_complex_dag(self):
        plan = ExecutionPlan([
            ExecutionStep(1, "tool_a", "step 1"),
            ExecutionStep(2, "tool_b", "step 2", depends_on=[1]),
            ExecutionStep(3, "tool_c", "step 3", depends_on=[1]),
            ExecutionStep(4, "tool_d", "step 4", depends_on=[2, 3]),
        ])
        order = plan.topological_order()
        order_ids = [s.step_id for s in order]
        # 1 must come before 2 and 3; 2 and 3 must come before 4
        assert order_ids.index(1) < order_ids.index(2)
        assert order_ids.index(1) < order_ids.index(3)
        assert order_ids.index(2) < order_ids.index(4)
        assert order_ids.index(3) < order_ids.index(4)

    def test_validation_missing_dependency(self):
        with pytest.raises(AssertionError):
            ExecutionPlan([
                ExecutionStep(1, "tool_a", "first"),
                ExecutionStep(2, "tool_b", "second", depends_on=[99]),
            ])

    def test_params_builder_injection(self):
        def builder(prior):
            return {"query": f"found_{len(prior)}_results"}
        step = ExecutionStep(1, "tool", "test", params_builder=builder)
        params = step.build_params({0: {"data": ["x", "y"]}})
        assert params["query"] == "found_1_results"

    def test_condition_skip(self):
        step = ExecutionStep(1, "tool", "test", params={"x": 1},
                              condition=lambda prior: False)
        assert step.condition({}) is False


class TestQueryPlanner:
    def setup_method(self):
        self.planner = QueryPlanner()

    def test_plan_factual(self):
        plan = self.planner.plan("Who owns payment-gateway?", "factual")
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "entity_lookup"

    def test_plan_causal(self):
        plan = self.planner.plan("What caused the payment outage?", "causal")
        assert len(plan.steps) >= 2
        tools = [s.tool for s in plan.steps]
        assert "get_incidents" in tools
        assert "search_vector" in tools

    def test_plan_multi_hop(self):
        plan = self.planner.plan("Why were hosts down between X and Y?", "multi_hop")
        assert len(plan.steps) >= 3
        tools = [s.tool for s in plan.steps]
        assert "get_incidents" in tools
        assert "search_vector" in tools
        assert "entity_lookup" in tools

    def test_plan_exploratory(self):
        plan = self.planner.plan("Show me recent incidents", "exploratory")
        assert len(plan.steps) == 2
        tools = [s.tool for s in plan.steps]
        assert "search_vector" in tools
        assert "get_incidents" in tools

    def test_plan_host_status(self):
        plan = self.planner.plan("Are the hosts healthy?", "host_status")
        assert len(plan.steps) == 1
        assert plan.steps[0].tool == "query_sql"

    def test_entity_name_extraction(self):
        assert "payment" in self.planner._extract_entity_name("Who owns payment-gateway?")
        assert "Alice" in self.planner._extract_entity_name("Find Alice Wang")
        assert "platform" in self.planner._extract_entity_name("What services does the platform team own?")

    def test_plan_type_routing(self):
        """Test that each query type hits the correct plan method."""
        plan = self.planner.plan("Compare incidents by severity", "comparative")
        assert len(plan.steps) == 2
        assert "query_sql" in [s.tool for s in plan.steps]

        plan = self.planner.plan("What happened yesterday?", "time_based")
        assert "get_incidents" in [s.tool for s in plan.steps]
