"""Orchestrator 单元测试 — 计划生成 + 执行引擎"""

import pytest
from app.services.orchestrator import (
    QueryPlan, StepContext, build_plan, execute_plan,
    execute_plan_stream, PLAN_TEMPLATES, STEP_REGISTRY,
    _reorder_for_streaming,
)
import app.services.steps  # noqa: F401 — trigger registration


class TestBuildPlan:
    def test_all_intents_produce_valid_plans(self):
        """每种意图都应生成非空步骤序列。"""
        for intent in ["market_data", "market_reasoning", "knowledge_rag", "compare", "general"]:
            plan = build_plan(intent, ["AAPL"])
            assert plan.question_type == intent
            assert len(plan.execution_steps) > 0

    def test_all_steps_in_registry(self):
        """PLAN_TEMPLATES 中的所有步骤名必须在 STEP_REGISTRY 中注册。"""
        for intent, steps in PLAN_TEMPLATES.items():
            for step_name in steps:
                assert step_name in STEP_REGISTRY, f"{step_name} not registered (intent={intent})"

    def test_unknown_intent_falls_back_to_general(self):
        plan = build_plan("unknown_intent", [])
        assert plan.execution_steps == PLAN_TEMPLATES["general"]

    def test_assets_passed_through(self):
        plan = build_plan("compare", ["AAPL", "MSFT"])
        assert plan.assets == ["AAPL", "MSFT"]

    def test_market_data_step_count(self):
        plan = build_plan("market_data", ["TSLA"])
        assert len(plan.execution_steps) == 6

    def test_market_reasoning_step_count(self):
        plan = build_plan("market_reasoning", ["TSLA"])
        assert len(plan.execution_steps) == 8

    def test_compare_step_count(self):
        plan = build_plan("compare", ["AAPL", "MSFT"])
        assert len(plan.execution_steps) == 5


class TestReorderForStreaming:
    def test_assemble_moved_before_generate(self):
        """实际 PLAN_TEMPLATES 中 assemble 在 generate 之后，reorder 应将其移到 generate 之前。"""
        # 使用实际的 market_data 模板
        from app.services.orchestrator import PLAN_TEMPLATES
        steps = list(PLAN_TEMPLATES["market_data"])
        reordered = _reorder_for_streaming(steps)
        gen_idx = reordered.index("generate_answer")
        asm_idx = reordered.index("assemble_market_response")
        assert asm_idx < gen_idx

    def test_non_assemble_steps_unchanged(self):
        steps = ["fetch_price", "validate_data", "generate_answer"]
        reordered = _reorder_for_streaming(steps)
        assert reordered == steps

    def test_multiple_assemble_steps(self):
        steps = ["fetch_price", "generate_answer", "assemble_a", "assemble_b"]
        # No generate_answer after assembles, they stay at end
        reordered = _reorder_for_streaming(steps)
        assert "fetch_price" in reordered
        assert "generate_answer" in reordered


class TestExecutePlan:
    def test_early_exit(self):
        """当步骤设置 early_exit 时，后续步骤不应执行。"""
        call_log = []

        def step_a(ctx, **kw):
            call_log.append("a")
            ctx.early_exit = True
            ctx.result = {"text_response": "stopped early"}

        def step_b(ctx, **kw):
            call_log.append("b")

        # Temporarily register
        STEP_REGISTRY["_test_a"] = step_a
        STEP_REGISTRY["_test_b"] = step_b
        try:
            plan = QueryPlan(execution_steps=["_test_a", "_test_b"])
            ctx = StepContext(plan=plan)
            result = execute_plan(ctx)
            assert result == {"text_response": "stopped early"}
            assert call_log == ["a"]
        finally:
            del STEP_REGISTRY["_test_a"]
            del STEP_REGISTRY["_test_b"]

    def test_unknown_step_skipped(self):
        """未注册的步骤应被跳过，不报错。"""
        plan = QueryPlan(execution_steps=["_nonexistent_step_xyz"])
        ctx = StepContext(plan=plan)
        ctx.result = {"text_response": "default"}
        result = execute_plan(ctx)
        assert result == {"text_response": "default"}


class TestExecutePlanStream:
    def test_stream_yields_done_event(self):
        """流式执行应以 done 事件结束。"""
        def step_noop(ctx, emit=None, **kw):
            ctx.result = {"text_response": "ok"}

        STEP_REGISTRY["_test_noop"] = step_noop
        try:
            plan = QueryPlan(execution_steps=["_test_noop"])
            ctx = StepContext(plan=plan)
            events = list(execute_plan_stream(ctx))
            assert any(e["event"] == "done" for e in events)
        finally:
            del STEP_REGISTRY["_test_noop"]

    def test_stream_early_exit_yields_done(self):
        """early_exit 时也应 yield done 事件。"""
        def step_exit(ctx, emit=None, **kw):
            ctx.early_exit = True
            ctx.result = {"text_response": "exit"}

        STEP_REGISTRY["_test_exit"] = step_exit
        try:
            plan = QueryPlan(execution_steps=["_test_exit"])
            ctx = StepContext(plan=plan)
            events = list(execute_plan_stream(ctx))
            done_events = [e for e in events if e["event"] == "done"]
            assert len(done_events) == 1
        finally:
            del STEP_REGISTRY["_test_exit"]


class TestStepContext:
    def test_default_values(self):
        ctx = StepContext()
        assert ctx.question == ""
        assert ctx.ticker is None
        assert ctx.early_exit is False
        assert ctx.result is None
        assert ctx.steps == []
        assert ctx.rag_sources == []

    def test_fields_assignable(self):
        ctx = StepContext(question="test", ticker="AAPL")
        assert ctx.question == "test"
        assert ctx.ticker == "AAPL"
