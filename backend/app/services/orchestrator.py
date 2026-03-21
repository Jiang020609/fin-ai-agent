"""Query Orchestrator — 将用户意图映射为执行计划，统一调度步骤执行。

三阶段架构：
1. Query Understanding — 由 agent.py 完成（意图分类、ticker 提取）
2. Plan Generation — build_plan() 根据意图生成步骤列表
3. Step Execution — execute_plan() / execute_plan_stream() 逐步执行
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ========== 核心数据结构 ==========

@dataclass
class QueryPlan:
    """意图 → 执行步骤的映射。"""
    assets: list[str] = field(default_factory=list)
    question_type: str = "general"
    time_range: str | None = None
    needs_news: bool = False
    needs_rag: bool = False
    execution_steps: list[str] = field(default_factory=list)


@dataclass
class StepContext:
    """步骤间共享状态，所有步骤通过读写 ctx 通信。"""
    # 输入
    question: str = ""
    history: list[dict] | None = None
    plan: QueryPlan | None = None

    # 步骤累积结果
    ticker: str | None = None
    tickers: list[str] | None = None
    market_data: dict | None = None
    summaries: dict | None = None
    chart_data: list | None = None
    market_meta: dict | None = None
    rag_context: str = ""
    rag_used: bool = False
    rag_sources: list[dict] = field(default_factory=list)
    web_sources: list[dict] = field(default_factory=list)
    evidence_raw: str = ""
    evidence_analysis: dict | None = None
    comparison: dict | None = None
    date_ref: Any = None
    data_validation: dict | None = None
    is_stale: bool = False

    # LLM 生成
    system_prompt: str = ""
    user_message: str = ""
    text_response: str = ""

    # 思考链
    steps: list = field(default_factory=list)

    # 控制流
    early_exit: bool = False
    result: dict | None = None


# ========== 步骤注册表 ==========

STEP_REGISTRY: dict[str, Callable] = {}


def register_step(name: str):
    """装饰器：注册步骤函数到全局注册表。"""
    def decorator(func):
        STEP_REGISTRY[name] = func
        return func
    return decorator


# ========== 计划模板 ==========

PLAN_TEMPLATES: dict[str, list[str]] = {
    "market_data": [
        "fetch_price", "validate_data", "build_market_prompt",
        "generate_answer", "validate_response_numbers",
        "assemble_market_response",
    ],
    "market_reasoning": [
        "extract_date", "fetch_price", "validate_data",
        "search_news", "classify_evidence",
        "build_reasoning_prompt", "generate_answer",
        "assemble_reasoning_response",
    ],
    "knowledge_rag": [
        "rag_search", "web_search_fallback",
        "build_knowledge_prompt", "generate_answer",
        "fact_check", "assemble_knowledge_response",
    ],
    "compare": [
        "fetch_prices_multi", "compute_comparison",
        "build_compare_prompt", "generate_answer",
        "assemble_compare_response",
    ],
    "general": [
        "build_general_prompt", "generate_answer",
        "assemble_general_response",
    ],
}


def build_plan(intent: str, assets: list[str]) -> QueryPlan:
    """根据意图和资产列表生成执行计划。"""
    steps = list(PLAN_TEMPLATES.get(intent, PLAN_TEMPLATES["general"]))
    return QueryPlan(
        assets=assets,
        question_type=intent,
        execution_steps=steps,
    )


# ========== Sync 执行引擎 ==========

def execute_plan(ctx: StepContext) -> dict:
    """顺序执行计划中的所有步骤，返回最终结果。"""
    for step_name in ctx.plan.execution_steps:
        step_fn = STEP_REGISTRY.get(step_name)
        if step_fn is None:
            logger.warning("Unknown step: %s, skipping", step_name)
            continue
        step_fn(ctx)
        if ctx.early_exit:
            return ctx.result
    return ctx.result


# ========== Streaming 执行引擎 ==========

def _reorder_for_streaming(execution_steps: list[str]) -> list[str]:
    """将 assemble_* 步骤移到 generate_answer 之前。

    Streaming 模式下，meta/chart 事件必须在 token 流之前发送，
    因此 assemble 步骤（负责生成 meta/chart 事件）需要提前执行。
    """
    # 分离为三组：assemble 步骤、generate_answer、其他步骤
    other_steps = []
    assemble_steps = []
    has_generate = False

    for step in execution_steps:
        if step.startswith("assemble_"):
            assemble_steps.append(step)
        elif step == "generate_answer":
            has_generate = True
        else:
            other_steps.append(step)

    # 重组：其他步骤 → assemble 步骤 → generate_answer
    reordered = other_steps + assemble_steps
    if has_generate:
        reordered.append("generate_answer")
    return reordered


def execute_plan_stream(ctx: StepContext):
    """流式执行计划，yield SSE 事件字典。

    事件类型：thought / token / chart / meta / done / error
    """
    reordered = _reorder_for_streaming(ctx.plan.execution_steps)

    for step_name in reordered:
        step_fn = STEP_REGISTRY.get(step_name)
        if step_fn is None:
            logger.warning("Unknown step: %s, skipping", step_name)
            continue

        if step_name == "generate_answer":
            # generate_answer 在 streaming 模式下直接 yield token 事件
            yield from step_fn(ctx, stream=True)
        else:
            # 其他步骤通过 emit 回调收集事件
            events = []
            step_fn(ctx, emit=events.append)
            yield from events

        if ctx.early_exit:
            if ctx.result and ctx.result.get("text_response"):
                yield {"event": "token", "data": ctx.result["text_response"]}
            yield {"event": "done", "data": {"steps": [asdict(s) for s in ctx.steps]}}
            return

    yield {"event": "done", "data": {"steps": [asdict(s) for s in ctx.steps]}}
