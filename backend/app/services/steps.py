"""步骤函数 — 从 agent.py handler 中提取的独立执行单元。

每个步骤函数签名统一：
    def step_xxx(ctx: StepContext, emit=None, stream=False) -> None
        - ctx: 共享上下文，步骤通过读写 ctx 通信
        - emit: 可选回调，streaming 模式下用于发送 SSE 事件
        - stream: generate_answer 专用，True 时 yield token 事件

步骤通过 @register_step 装饰器注册到 STEP_REGISTRY。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from app.services.orchestrator import StepContext, register_step
from app.services.llm import chat_completion, chat_completion_stream
from app.services.market import get_stock_summary
from app.services.rag import get_relevant_context
from app.services.web_search import web_search
from app.services.grounding import (
    fact_check,
    validate_market_data,
    validate_market_response_numbers,
    build_missing_field_notice,
)
from app.services.compare import compute_comparison
from app.services.news_classifier import classify_evidence
from app.prompts.templates import (
    MARKET_SYSTEM_PROMPT,
    MARKET_USER_TEMPLATE,
    MARKET_REASONING_SYSTEM_PROMPT,
    MARKET_REASONING_USER_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    FACT_CHECK_PROMPT,
    COMPARE_SYSTEM_PROMPT,
    COMPARE_USER_TEMPLATE,
)
from app.utils.date_extract import extract_date_ref, build_search_query

logger = logging.getLogger(__name__)


# ========== ThoughtStep（从 agent.py 移入） ==========

@dataclass
class ThoughtStep:
    """记录 Agent 单步推理过程，构成可审计的思考链。"""
    step: str
    result: str
    detail: dict = field(default_factory=dict)


# ========== 免责声明常量 ==========

_DISCLAIMER_MARKET = "以上数据来自第三方行情接口，可能存在延迟。分析内容仅供参考，不构成投资建议。"
_DISCLAIMER_REASONING = "新闻归因存在不确定性，不同信息源可能有不同解读。以上分析不构成投资建议。"
_DISCLAIMER_COMPARE = "以上对比基于历史数据，不构成投资建议。不同时间窗口下结论可能不同。"


# ========== 来源解析工具 ==========

def parse_web_sources(raw_text: str) -> list[dict]:
    """从 web_search 返回的格式化文本中解析来源信息。"""
    sources: list[dict] = []
    for block in raw_text.split("\n\n"):
        if not block.startswith("[网络搜索结果"):
            continue
        parts = block.split("\n")
        title = parts[0].split("] ", 1)[-1] if "] " in parts[0] else parts[0]
        url = ""
        for p in parts:
            if p.startswith("来源: "):
                url = p.replace("来源: ", "").strip()
        sources.append({"title": title, "source": "Web Search", "url": url, "published_at": None})
    return sources


def parse_rag_sources(context: str) -> list[dict]:
    """从 RAG 格式化上下文中解析来源信息（含页码和相关度）。"""
    sources: list[dict] = []
    for block in context.split("\n\n---\n\n"):
        if not block.startswith("[参考资料"):
            continue
        m = re.search(r'来源:\s*([^,)]+),\s*页码:\s*([^,)]+),\s*相关度:\s*([\d.]+)', block)
        if m:
            page_raw = m.group(2).strip()
            page = page_raw if page_raw != "-" else None
            try:
                score = float(m.group(3))
            except ValueError:
                score = None
            sources.append({
                "title": m.group(1).strip(),
                "source": "知识库",
                "url": None,
                "published_at": None,
                "page": page,
                "relevance_score": score,
            })
        else:
            m2 = re.search(r'来源:\s*([^,\s)]+)', block)
            if m2:
                sources.append({"title": m2.group(1), "source": "知识库", "url": None,
                                "published_at": None, "page": None, "relevance_score": None})
    return sources


# ========== 结构化回答组装工具 ==========

def build_no_data_message(ticker: str) -> str:
    """构建数据不可用时的用户友好提示。"""
    return (
        f"抱歉，yfinance 目前无法获取 **{ticker}** 的实时数据。\n\n"
        "**可能原因：**\n\n"
        f"- 该代码 ({ticker}) 可能不存在、已退市，或属于 yfinance 未覆盖的小众资产\n\n"
        "- Yahoo Finance API 暂时不可用或被限流\n\n"
        "- 网络连接异常\n\n"
        "**建议：**\n\n"
        "- 检查 Ticker 是否正确（如苹果应为 AAPL、腾讯应为 0700.HK）\n\n"
        "- 尝试使用主流交易所的标准代码查询\n\n"
        "- 稍后重试\n\n"
        "> 系统严格基于 API 返回的数据回答，不会从训练数据中猜测价格。"
    )


def build_data_summary(market_data: dict) -> dict:
    """从 market_data 程序化组装 DataSummary 字段。"""
    price_info = market_data.get("price", {})
    change_7d = market_data.get("change_7d", {})
    change_30d = market_data.get("change_30d", {})
    return {
        "price": price_info.get("current_price"),
        "currency": price_info.get("currency", "USD"),
        "change_7d": change_7d.get("change_pct"),
        "change_30d": change_30d.get("change_pct"),
        "high_7d": change_7d.get("high"),
        "low_7d": change_7d.get("low"),
        "high_30d": change_30d.get("high"),
        "low_30d": change_30d.get("low"),
        "timestamp": market_data.get("query_time"),
        "data_source": market_data.get("data_source", "unknown"),
    }


def build_trend_summary(market_data: dict) -> dict:
    """从 market_data 组装 TrendSummary。"""
    from app.services.trend import classify_trend_from_market_data
    trend_result = classify_trend_from_market_data(market_data)
    return trend_result.to_dict()


def build_explainability_meta(
    data_sources: list[str],
    evidence_count: int = 0,
    has_api_data: bool = False,
) -> dict:
    """构建可解释性元数据。"""
    confidence_data = "high" if has_api_data else "low"
    if evidence_count >= 3:
        confidence_reasoning = "high"
    elif evidence_count >= 1:
        confidence_reasoning = "medium"
    else:
        confidence_reasoning = "low"
    return {
        "data_sources": data_sources,
        "evidence_count": evidence_count,
        "confidence_data": confidence_data,
        "confidence_reasoning": confidence_reasoning,
    }


def _emit_thought(ctx: StepContext, emit, step: str, result: str, detail: dict | None = None):
    """向思考链追加步骤，若 emit 回调存在则发送 thought 事件。"""
    ts = ThoughtStep(step=step, result=result, detail=detail or {})
    ctx.steps.append(ts)
    if emit:
        emit({"event": "thought", "data": asdict(ts)})
    return ts


def _extract_chart_data(market_data: dict) -> list | None:
    """从 market_data 提取图表数据。"""
    change_30d = market_data.get("change_30d", {})
    change_7d = market_data.get("change_7d", {})
    if "history" in change_30d:
        return change_30d["history"]
    elif "history" in change_7d:
        return change_7d["history"]
    return None


def _extract_market_meta(market_data: dict, ticker: str) -> dict | None:
    """从 market_data 提取 KPI 卡片所需的 market_meta。"""
    price_info = market_data.get("price", {})
    change_7d = market_data.get("change_7d", {})
    raw_meta = {
        "current_price": price_info.get("current_price"),
        "previous_close": price_info.get("previous_close"),
        "change_pct": change_7d.get("change_pct"),
        "pe_ratio": price_info.get("pe_ratio"),
        "market_cap": price_info.get("market_cap"),
        "name": price_info.get("name", ticker),
        "currency": price_info.get("currency", "USD"),
    }
    return raw_meta if raw_meta.get("current_price") is not None else None


# ========== 步骤函数 ==========

@register_step("fetch_price")
def step_fetch_price(ctx: StepContext, emit=None, **kw):
    """获取单个 ticker 的行情数据。"""
    ts = _emit_thought(ctx, emit, "调用行情API", f"获取 {ctx.ticker} 数据中...")
    ctx.market_data = get_stock_summary(ctx.ticker)
    data_available = ctx.market_data.get("data_available", False)
    ts.result = f"ticker={ctx.ticker}, data_available={data_available}"
    ts.detail = {
        "current_price": ctx.market_data.get("price", {}).get("current_price"),
        "7d_trend": ctx.market_data.get("change_7d", {}).get("trend"),
        "30d_trend": ctx.market_data.get("change_30d", {}).get("trend"),
    }
    if emit:
        emit({"event": "thought", "data": asdict(ts)})


@register_step("fetch_prices_multi")
def step_fetch_prices_multi(ctx: StepContext, emit=None, **kw):
    """逐 ticker 获取行情数据（对比场景）。"""
    ctx.summaries = {}
    for ticker in ctx.tickers:
        ts = _emit_thought(ctx, emit, f"获取 {ticker} 数据", "获取中...")
        summary = get_stock_summary(ticker)
        ctx.summaries[ticker] = summary
        available = summary.get("data_available", False)
        ts.result = f"data_available={available}"
        ts.detail = {"current_price": summary.get("price", {}).get("current_price")}
        if emit:
            emit({"event": "thought", "data": asdict(ts)})


@register_step("validate_data")
def step_validate_data(ctx: StepContext, emit=None, **kw):
    """校验行情数据可用性 + 降级检测。无数据时设 early_exit。"""
    market_data = ctx.market_data
    data_available = market_data.get("data_available", False)
    validation = validate_market_data(market_data)
    ctx.data_validation = validation
    ctx.is_stale = validation["is_stale"]

    if not data_available and not ctx.is_stale:
        _emit_thought(ctx, emit, "数据校验", "数据完全不可用，无缓存可降级",
                      {"grounding_issues": validation["issues"]})
        _emit_thought(ctx, emit, "生成降级回复", "返回用户友好提示")
        ctx.early_exit = True
        ctx.text_response = build_no_data_message(ctx.ticker)
        ctx.result = {
            "text_response": ctx.text_response,
            "chart_data": None,
            "intent": ctx.plan.question_type,
            "ticker": ctx.ticker,
            "steps": [asdict(s) for s in ctx.steps],
        }
        return

    if ctx.is_stale:
        stale_ages = [
            market_data.get(k, {}).get("stale_age_seconds", 0)
            for k in ("price", "change_7d", "change_30d")
            if market_data.get(k, {}).get("stale")
        ]
        max_age = max(stale_ages) if stale_ages else 0
        if max_age < 120:
            age_text = f"{max_age} 秒前"
        elif max_age < 7200:
            age_text = f"{max_age // 60} 分钟前"
        else:
            age_text = f"{max_age // 3600} 小时前"
        _emit_thought(ctx, emit, "数据校验",
                      f"API 不可用，降级使用缓存数据（{age_text}）",
                      {"stale_age_seconds": max_age})
    else:
        _emit_thought(ctx, emit, "数据校验", "数据校验通过（实时）")


@register_step("extract_date")
def step_extract_date(ctx: StepContext, emit=None, **kw):
    """提取用户问题中的日期引用。"""
    ctx.date_ref = extract_date_ref(ctx.question)
    if ctx.date_ref:
        _emit_thought(ctx, emit, "日期识别",
                      f"识别到时间引用: {ctx.date_ref.date_str}",
                      {"date_str": ctx.date_ref.date_str, "search_hint": ctx.date_ref.search_hint})
    else:
        _emit_thought(ctx, emit, "日期识别", "未指定具体日期，分析近期走势")


@register_step("search_news")
def step_search_news(ctx: StepContext, emit=None, **kw):
    """搜索相关新闻/事件证据。"""
    market_data = ctx.market_data
    if not market_data:
        _emit_thought(ctx, emit, "搜索新闻证据", "无行情数据，跳过搜索")
        return

    ts = _emit_thought(ctx, emit, "搜索新闻证据", "搜索中...")
    price_info = market_data.get("price", {})
    asset_name = price_info.get("name", ctx.ticker)
    trend_dir = market_data.get("change_7d", {}).get("trend_detail", {}).get("label")
    search_query = build_search_query(
        asset_name=asset_name,
        ticker=ctx.ticker,
        question=ctx.question,
        date_ref=ctx.date_ref,
        trend_direction=trend_dir,
    )
    ts.detail = {"search_query": search_query}

    evidence = web_search(search_query, max_results=5)
    if evidence:
        ts.result = f"搜索命中，证据长度={len(evidence)}"
        ctx.evidence_raw = evidence
        ctx.web_sources = parse_web_sources(evidence)
        if emit:
            emit({"event": "thought", "data": asdict(ts)})
    else:
        ts.result = "未配置搜索 API 或搜索无结果"
        ctx.evidence_raw = "（未检索到相关新闻或事件证据）"
        if emit:
            emit({"event": "thought", "data": asdict(ts)})


@register_step("classify_evidence")
def step_classify_evidence(ctx: StepContext, emit=None, **kw):
    """对搜索到的证据进行分类。"""
    if not ctx.web_sources:
        return

    ts = _emit_thought(ctx, emit, "证据分类", "分类中...")
    price_info = ctx.market_data.get("price", {}) if ctx.market_data else {}
    asset_name = price_info.get("name", ctx.ticker)
    ea = classify_evidence(ctx.evidence_raw, asset_name, ctx.ticker)
    ctx.evidence_analysis = {
        "main_drivers": ea.main_drivers,
        "secondary_drivers": ea.secondary_drivers,
        "evidence_strength": ea.evidence_strength,
        "summary": ea.summary,
    }
    ts.result = ea.summary
    ts.detail = {"main_drivers": ea.main_drivers, "strength": ea.evidence_strength}
    if emit:
        emit({"event": "thought", "data": asdict(ts)})


@register_step("rag_search")
def step_rag_search(ctx: StepContext, emit=None, **kw):
    """RAG 向量检索（不可用时优雅降级到 web search）。"""
    from app.services.rag import is_rag_available
    if not is_rag_available():
        ts = _emit_thought(ctx, emit, "RAG向量检索", "知识库未加载，将使用 Web 搜索")
        if emit:
            emit({"event": "thought", "data": asdict(ts)})
        return

    ts = _emit_thought(ctx, emit, "RAG向量检索", "检索中...")
    try:
        context = get_relevant_context(ctx.question, top_k=4)
        if context:
            ctx.rag_used = True
            ctx.rag_context = context
            ctx.rag_sources = parse_rag_sources(context)
            ts.result = f"命中，上下文长度={len(context)}"
        else:
            ts.result = "未命中相关文档"
    except Exception as e:
        ts.result = f"检索失败: {e}"
        logger.warning("RAG retrieval failed: %s", e)
    if emit:
        emit({"event": "thought", "data": asdict(ts)})


@register_step("web_search_fallback")
def step_web_search_fallback(ctx: StepContext, emit=None, **kw):
    """Web 搜索兜底（仅当 RAG 未命中时执行）。"""
    if ctx.rag_used:
        return

    ts = _emit_thought(ctx, emit, "Web搜索兜底", "搜索中...")
    web_context = web_search(ctx.question)
    if web_context:
        ts.result = f"搜索命中，长度={len(web_context)}"
        ctx.rag_context = web_context
        ctx.rag_used = True
        ctx.web_sources = parse_web_sources(web_context)
    else:
        ts.result = "未配置搜索 API 或搜索无结果"
    if emit:
        emit({"event": "thought", "data": asdict(ts)})


@register_step("build_market_prompt")
def step_build_market_prompt(ctx: StepContext, emit=None, **kw):
    """组装 market_data 类型的 LLM prompt。"""
    market_data = ctx.market_data
    market_data_str = json.dumps(market_data, ensure_ascii=False, indent=2)

    if ctx.is_stale:
        stale_notice = (
            "\n\n⚠️ 注意：行情 API 暂时不可用，以下数据来自缓存，"
            "可能非最新价格。请在回答开头明确提醒用户这是参考数据。"
        )
        ctx.user_message = MARKET_USER_TEMPLATE.format(
            question=ctx.question, market_data=market_data_str,
        ) + stale_notice
    else:
        ctx.user_message = MARKET_USER_TEMPLATE.format(
            question=ctx.question, market_data=market_data_str,
        )

    # 注入缺失字段提示
    missing_notice = build_missing_field_notice(market_data)
    if missing_notice:
        ctx.user_message += f"\n\n⚠️ {missing_notice}请在回答中明确告知用户哪些数据无法获取，不要编造。"

    ctx.system_prompt = MARKET_SYSTEM_PROMPT


@register_step("build_reasoning_prompt")
def step_build_reasoning_prompt(ctx: StepContext, emit=None, **kw):
    """组装 market_reasoning 类型的 LLM prompt。"""
    market_data_str = json.dumps(ctx.market_data, ensure_ascii=False, indent=2)
    ctx.user_message = MARKET_REASONING_USER_TEMPLATE.format(
        question=ctx.question,
        market_data=market_data_str,
        evidence=ctx.evidence_raw,
    )
    if ctx.evidence_analysis and ctx.evidence_analysis.get("summary"):
        ctx.user_message += f"\n\n证据分类摘要：{ctx.evidence_analysis['summary']}"
    ctx.system_prompt = MARKET_REASONING_SYSTEM_PROMPT


@register_step("build_knowledge_prompt")
def step_build_knowledge_prompt(ctx: StepContext, emit=None, **kw):
    """组装 knowledge_rag 类型的 LLM prompt。"""
    if ctx.rag_used:
        ctx.system_prompt = RAG_SYSTEM_PROMPT
        ctx.user_message = RAG_USER_TEMPLATE.format(
            question=ctx.question, context=ctx.rag_context,
        )
    else:
        ctx.system_prompt = (
            RAG_SYSTEM_PROMPT
            + "\n\n注意：知识库中未检索到相关资料，请基于你的通用金融知识回答，"
            '并在回答末尾注明"本回答基于模型通用知识，未引用知识库资料"。'
        )
        ctx.user_message = RAG_USER_TEMPLATE.format(
            question=ctx.question, context="（未检索到相关参考资料）",
        )


@register_step("build_compare_prompt")
def step_build_compare_prompt(ctx: StepContext, emit=None, **kw):
    """组装 compare 类型的 LLM prompt。"""
    comparison_data_str = json.dumps(ctx.comparison["assets"], ensure_ascii=False, indent=2)
    winners_str = json.dumps(ctx.comparison["winners"], ensure_ascii=False, indent=2)
    ctx.user_message = COMPARE_USER_TEMPLATE.format(
        question=ctx.question,
        comparison_data=comparison_data_str,
        winners=winners_str,
    )
    ctx.system_prompt = COMPARE_SYSTEM_PROMPT


@register_step("build_general_prompt")
def step_build_general_prompt(ctx: StepContext, emit=None, **kw):
    """组装通用问答的 LLM prompt。"""
    ctx.system_prompt = "你是一个友好的金融助手。简洁回答用户问题，使用用户的语言。回答要分点清晰，要点之间空一行，关键内容加粗。"
    ctx.user_message = ctx.question


@register_step("generate_answer")
def step_generate_answer(ctx: StepContext, emit=None, stream=False, **kw):
    """调用 LLM 生成回答。

    sync 模式：直接调用 chat_completion，写入 ctx.text_response。
    stream 模式：yield token 事件（作为 generator）。
    """
    if stream:
        # Streaming 模式：返回 generator
        return _stream_generate(ctx, emit)
    else:
        # Sync 模式
        ts = _emit_thought(ctx, emit, "LLM生成回答", "调用中...")
        ctx.text_response = chat_completion(
            ctx.system_prompt, ctx.user_message, history=ctx.history,
        )
        ts.result = f"生成完毕，长度={len(ctx.text_response)}"


def _stream_generate(ctx: StepContext, emit=None):
    """Streaming generate_answer：yield token 事件。"""
    ts = ThoughtStep(step="LLM生成回答", result="流式生成中...")
    ctx.steps.append(ts)
    if emit:
        yield {"event": "thought", "data": asdict(ts)}
    else:
        yield {"event": "thought", "data": asdict(ts)}

    collected = []
    for token in chat_completion_stream(ctx.system_prompt, ctx.user_message, history=ctx.history):
        collected.append(token)
        yield {"event": "token", "data": token}

    ctx.text_response = "".join(collected)
    ts.result = "生成完毕"


@register_step("validate_response_numbers")
def step_validate_response_numbers(ctx: StepContext, emit=None, **kw):
    """交叉验证 LLM 回答中的数字是否匹配原始数据。"""
    if not ctx.market_data or not ctx.text_response:
        return
    nums_ok, suspicious = validate_market_response_numbers(ctx.text_response, ctx.market_data)
    if not nums_ok:
        _emit_thought(ctx, emit, "数字交叉验证",
                      f"警告: 回答中出现可疑数字 {suspicious}",
                      {"suspicious_numbers": suspicious})
        logger.warning("[Grounding] Suspicious numbers in response: %s", suspicious)
    else:
        _emit_thought(ctx, emit, "数字交叉验证", "通过")


@register_step("fact_check")
def step_fact_check(ctx: StepContext, emit=None, **kw):
    """事实核查（仅当 rag_used 时执行）。"""
    if not ctx.rag_used or not ctx.rag_context:
        return

    ts = _emit_thought(ctx, emit, "事实核查", "校验中...")

    passed, issues = fact_check(
        text_response=ctx.text_response,
        context=ctx.rag_context,
        llm_call=chat_completion,
        fact_check_prompt_template=FACT_CHECK_PROMPT,
    )

    if passed:
        ts.result = "通过"
        if emit:
            emit({"event": "thought", "data": asdict(ts)})
    else:
        ts.result = f"未通过: {issues}"
        if emit:
            emit({"event": "thought", "data": asdict(ts)})

        # 重试一次
        retry_ts = _emit_thought(ctx, emit, "LLM重新生成",
                                 f"事实核查未通过（{issues}），重新生成...")
        retry_system = (
            ctx.system_prompt
            + f"\n\n【重要】上一次回答存在事实偏差：{issues}。"
            "请严格依据参考资料修正，不要编造数据。"
        )
        ctx.text_response = chat_completion(retry_system, ctx.user_message, history=ctx.history)
        retry_ts.result = f"重新生成完毕，长度={len(ctx.text_response)}"


@register_step("compute_comparison")
def step_compute_comparison(ctx: StepContext, emit=None, **kw):
    """计算多资产对比指标。"""
    ts = _emit_thought(ctx, emit, "计算对比指标", "计算中...")
    ctx.comparison = compute_comparison(ctx.summaries)
    ts.result = f"对比完成，{len(ctx.comparison['assets'])} 个资产"
    if emit:
        emit({"event": "thought", "data": asdict(ts)})


# ========== assemble 步骤 ==========

@register_step("assemble_market_response")
def step_assemble_market_response(ctx: StepContext, emit=None, **kw):
    """组装 market_data 最终响应。"""
    market_data = ctx.market_data
    data_available = market_data.get("data_available", False)
    actual_source = market_data.get("data_source", "yahoo")

    ctx.chart_data = _extract_chart_data(market_data)
    ctx.market_meta = _extract_market_meta(market_data, ctx.ticker)

    meta = build_explainability_meta(
        data_sources=[actual_source],
        has_api_data=data_available,
    )
    structured_response = {
        "response_type": "market_data",
        "data_summary": build_data_summary(market_data),
        "trend_summary": build_trend_summary(market_data),
        "analysis": [{"title": "分析与解读", "content": ctx.text_response}],
        "sources": [{"title": "行情数据", "source": market_data.get("data_source", "yahoo"), "url": None}],
        "disclaimer": _DISCLAIMER_MARKET,
        "meta": meta,
    }

    ctx.result = {
        "text_response": ctx.text_response,
        "chart_data": ctx.chart_data,
        "intent": "market_data",
        "ticker": ctx.ticker,
        "steps": [asdict(s) for s in ctx.steps],
        "market_meta": ctx.market_meta,
        "structured_response": structured_response,
    }

    # Streaming 模式：发送 meta + chart 事件
    if emit:
        # 在 streaming 模式，analysis 内容由 token 流填充，此处留空
        streaming_sr = dict(structured_response)
        streaming_sr["analysis"] = []
        emit({"event": "meta", "data": {
            "intent": "market_data", "ticker": ctx.ticker, "rag_used": None,
            "market_meta": ctx.market_meta,
            "structured_response": streaming_sr,
        }})
        if ctx.chart_data:
            emit({"event": "chart", "data": ctx.chart_data})


@register_step("assemble_reasoning_response")
def step_assemble_reasoning_response(ctx: StepContext, emit=None, **kw):
    """组装 market_reasoning 最终响应。"""
    market_data = ctx.market_data
    data_available = market_data.get("data_available", False) if market_data else False

    # 无数据时 early exit
    if not data_available and market_data:
        _emit_thought(ctx, emit, "数据校验", "数据完全不可用")
        ctx.early_exit = True
        ctx.text_response = build_no_data_message(ctx.ticker)
        ctx.result = {
            "text_response": ctx.text_response,
            "chart_data": None,
            "intent": "market_reasoning",
            "ticker": ctx.ticker,
            "steps": [asdict(s) for s in ctx.steps],
        }
        return

    ctx.chart_data = _extract_chart_data(market_data) if market_data else None
    ctx.market_meta = _extract_market_meta(market_data, ctx.ticker) if market_data else None

    actual_source = market_data.get("data_source", "yahoo")
    evidence_sources = ctx.web_sources
    all_sources = [{"title": "行情数据", "source": actual_source, "url": None}]
    all_sources.extend(evidence_sources)

    data_src_list = [actual_source]
    if evidence_sources:
        data_src_list.append("web_search")
    meta = build_explainability_meta(
        data_sources=data_src_list,
        evidence_count=len(evidence_sources),
        has_api_data=data_available,
    )
    structured_response = {
        "response_type": "market_reasoning",
        "data_summary": build_data_summary(market_data),
        "trend_summary": build_trend_summary(market_data),
        "analysis": [{"title": "可能原因分析", "content": ctx.text_response}],
        "sources": all_sources,
        "disclaimer": _DISCLAIMER_REASONING,
        "evidence_analysis": ctx.evidence_analysis,
        "meta": meta,
    }

    ctx.result = {
        "text_response": ctx.text_response,
        "chart_data": ctx.chart_data,
        "intent": "market_reasoning",
        "ticker": ctx.ticker,
        "rag_used": bool(evidence_sources),
        "steps": [asdict(s) for s in ctx.steps],
        "market_meta": ctx.market_meta,
        "structured_response": structured_response,
    }

    if emit:
        streaming_sr = dict(structured_response)
        streaming_sr["analysis"] = []
        emit({"event": "meta", "data": {
            "intent": "market_reasoning", "ticker": ctx.ticker,
            "rag_used": bool(evidence_sources),
            "market_meta": ctx.market_meta,
            "structured_response": streaming_sr,
        }})
        if ctx.chart_data:
            emit({"event": "chart", "data": ctx.chart_data})


@register_step("assemble_knowledge_response")
def step_assemble_knowledge_response(ctx: StepContext, emit=None, **kw):
    """组装 knowledge_rag 最终响应。"""
    all_sources = ctx.rag_sources + ctx.web_sources
    data_src = []
    if ctx.rag_sources:
        data_src.append("knowledge_base")
    if ctx.web_sources:
        data_src.append("web_search")
    meta = build_explainability_meta(
        data_sources=data_src,
        evidence_count=len(all_sources),
        has_api_data=bool(all_sources),
    )
    structured_response = {
        "response_type": "knowledge_rag",
        "data_summary": None,
        "trend_summary": None,
        "analysis": [{"title": "知识问答", "content": ctx.text_response}],
        "sources": all_sources,
        "disclaimer": None,
        "meta": meta,
    }

    ctx.result = {
        "text_response": ctx.text_response,
        "chart_data": None,
        "intent": "knowledge_rag",
        "rag_used": ctx.rag_used,
        "steps": [asdict(s) for s in ctx.steps],
        "structured_response": structured_response,
    }

    if emit:
        emit({"event": "meta", "data": {
            "intent": "knowledge_rag", "ticker": None, "rag_used": ctx.rag_used,
        }})


@register_step("assemble_compare_response")
def step_assemble_compare_response(ctx: StepContext, emit=None, **kw):
    """组装 compare 最终响应。"""
    tickers = ctx.tickers
    actual_sources = list({s.get("data_source", "yahoo") for s in ctx.summaries.values()})
    sources = [{"title": f"{t} 行情数据", "source": ctx.summaries[t].get("data_source", "yahoo"), "url": None} for t in tickers]
    meta = build_explainability_meta(
        data_sources=actual_sources,
        evidence_count=0,
        has_api_data=any(s.get("data_available") for s in ctx.summaries.values()),
    )
    structured_response = {
        "response_type": "compare",
        "data_summary": None,
        "trend_summary": None,
        "analysis": [{"title": "对比分析", "content": ctx.text_response}],
        "sources": sources,
        "disclaimer": _DISCLAIMER_COMPARE,
        "comparison": ctx.comparison,
        "meta": meta,
    }

    ctx.result = {
        "text_response": ctx.text_response,
        "chart_data": None,
        "intent": "compare",
        "ticker": tickers[0],
        "tickers": tickers,
        "steps": [asdict(s) for s in ctx.steps],
        "structured_response": structured_response,
    }

    if emit:
        streaming_sr = dict(structured_response)
        streaming_sr["analysis"] = []
        emit({"event": "meta", "data": {
            "intent": "compare", "ticker": tickers[0], "tickers": tickers,
            "rag_used": None,
            "structured_response": streaming_sr,
        }})


@register_step("assemble_general_response")
def step_assemble_general_response(ctx: StepContext, emit=None, **kw):
    """组装通用问答最终响应。"""
    ctx.result = {
        "text_response": ctx.text_response,
        "chart_data": None,
        "intent": "general",
        "steps": [asdict(s) for s in ctx.steps],
    }

    if emit:
        emit({"event": "meta", "data": {"intent": "general", "ticker": None, "rag_used": None}})
