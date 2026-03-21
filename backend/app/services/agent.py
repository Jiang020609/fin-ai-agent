"""Agent 路由 — 根据用户意图分发到行情服务或 RAG 知识库

架构加固：
- ThoughtStep: 每次请求记录完整的思考链（意图识别 → Tool 调用 → 结果摘要）
- audit_log: 装饰器，记录每次请求的 Input/Output/耗时，模拟生产级监控
- 数据不可用时，向 LLM 发送明确的"数据缺失"信号，防止幻觉

路由分类（4 类）：
- market_data: 价格、涨跌幅、走势等纯行情查询
- market_reasoning: 涨跌原因分析，需要行情 + 新闻证据 + LLM 归因
- knowledge_rag: 金融知识问答，基于 RAG / web search
- general: 兜底通用问答
"""

import json
import time
import logging
import functools
from dataclasses import dataclass, field, asdict
from typing import Callable

import re as _re

from app.services.llm import chat_completion, chat_completion_stream, classify_intent, classify_intent_with_tools
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
from app.services.session import SessionState
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
    QUERY_TRANSFORM_PROMPT,
)
from app.utils.ticker_map import resolve_ticker, resolve_tickers_multi
from app.utils.date_extract import extract_date_ref, build_search_query

logger = logging.getLogger(__name__)


# ========== 快速意图分类（正则，免 LLM 调用） ==========
# 注意顺序：compare > reasoning > data，compare 优先级最高
_COMPARE_PATTERNS = [
    _re.compile(r"(对比|比较|对照|pk|PK)", _re.IGNORECASE),
    _re.compile(r"\bvs\.?\b", _re.IGNORECASE),
    _re.compile(r"(哪个更|谁更强|谁更好|哪只更|哪支更|哪个好)", _re.IGNORECASE),
    _re.compile(r"(和|与|跟).{1,10}(比|对比|相比|比较)", _re.IGNORECASE),
]
_REASONING_PATTERNS = [
    _re.compile(r"(为什么|为何|原因|怎么回事|因为什么|受什么影响|什么导致|什么原因)", _re.IGNORECASE),
    _re.compile(r"(大涨|大跌|暴涨|暴跌|飙升|跳水|急涨|急跌).*(原因|为什么|为何|怎么)", _re.IGNORECASE),
    _re.compile(r"(why|reason|what\s+caused|what\s+happened|due\s+to|because)", _re.IGNORECASE),
]
_MARKET_PATTERNS = [
    _re.compile(r"(股价|股票|行情|涨跌|走势|市值|k线|K线|大盘|指数|收盘|开盘|成交量)", _re.IGNORECASE),
    _re.compile(r"(stock|price|market|trading|shares?|ticker|bull|bear)", _re.IGNORECASE),
    _re.compile(r"(多少钱|怎么样|涨了|跌了|最新)", _re.IGNORECASE),
]
_KNOWLEDGE_PATTERNS = [
    _re.compile(r"(什么是|解释|定义|概念|区别|原理|如何计算|怎么理解)", _re.IGNORECASE),
    _re.compile(r"(what\s+is|explain|define|difference\s+between|how\s+to\s+calculate)", _re.IGNORECASE),
]


def fast_classify_intent(question: str) -> str | None:
    """正则快速分类，命中直接返回，未命中返回 None（需 LLM fallback）。

    优先级：compare > reasoning > market_data > knowledge_rag
    compare 检测需要 2+ ticker 才成立。
    """
    # compare 优先级最高，但需要 2+ ticker
    for p in _COMPARE_PATTERNS:
        if p.search(question):
            tickers = resolve_tickers_multi(question)
            if len(tickers) >= 2:
                return "compare"
            break  # 有对比意图但不够 2 个 ticker，继续其他分类

    for p in _REASONING_PATTERNS:
        if p.search(question):
            return "market_reasoning"
    for p in _MARKET_PATTERNS:
        if p.search(question):
            return "market_data"
    for p in _KNOWLEDGE_PATTERNS:
        if p.search(question):
            return "knowledge_rag"
    return None


def _normalize_intent(raw: str) -> str:
    """将 LLM 返回的意图标签归一化到 5 类。

    兼容旧的 'market' / 'knowledge' 标签。
    """
    r = raw.strip().lower()
    if "compare" in r:
        return "compare"
    if "market_reasoning" in r or "reasoning" in r:
        return "market_reasoning"
    if "market_data" in r or "market" in r:
        return "market_data"
    if "knowledge" in r:
        return "knowledge_rag"
    return "general"


# ========== 思考链数据结构 ==========
@dataclass
class ThoughtStep:
    """记录 Agent 单步推理过程，构成可审计的思考链。"""
    step: str          # 步骤名称
    result: str        # 步骤结果摘要
    detail: dict = field(default_factory=dict)  # 可选的详细数据


# ========== 审计日志装饰器 ==========
def audit_log(func: Callable) -> Callable:
    """生产级审计日志装饰器。

    记录每次请求的：
    - 输入参数（用户原始问题）
    - 输出摘要（意图 + ticker + 是否使用 RAG + 步骤数）
    - 耗时（ms），用于性能监控和告警
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        question = args[0] if args else kwargs.get("question", "")

        logger.info(
            "[AUDIT] >>> Request | question=%s",
            question[:80],
        )

        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start) * 1000

            logger.info(
                "[AUDIT] <<< Response | intent=%s | ticker=%s | rag_used=%s | "
                "steps=%d | time=%.0fms",
                result.get("intent", "?"),
                result.get("ticker", "-"),
                result.get("rag_used", "-"),
                len(result.get("steps", [])),
                elapsed_ms,
            )

            # 慢请求告警（生产环境可对接 Prometheus / Datadog）
            if elapsed_ms > 10000:
                logger.warning(
                    "[AUDIT] SLOW REQUEST | question=%s | time=%.0fms",
                    question[:50],
                    elapsed_ms,
                )

            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error(
                "[AUDIT] !!! Error | question=%s | error=%s | time=%.0fms",
                question[:50],
                str(e),
                elapsed_ms,
            )
            raise

    return wrapper


# ========== 来源解析工具 ==========
def _parse_web_sources(raw_text: str) -> list[dict]:
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


def _parse_rag_sources(context: str) -> list[dict]:
    """从 RAG 格式化上下文中解析来源信息（含页码和相关度）。"""
    import re
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
            # Fallback: 兼容旧格式
            m2 = re.search(r'来源:\s*([^,\s)]+)', block)
            if m2:
                sources.append({"title": m2.group(1), "source": "知识库", "url": None, "published_at": None, "page": None, "relevance_score": None})
    return sources


# ========== 结构化回答组装工具 ==========
_DISCLAIMER_MARKET = "以上数据来自第三方行情接口，可能存在延迟。分析内容仅供参考，不构成投资建议。"
_DISCLAIMER_REASONING = "新闻归因存在不确定性，不同信息源可能有不同解读。以上分析不构成投资建议。"


def _build_data_summary(market_data: dict) -> dict:
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


def _build_trend_summary(market_data: dict) -> dict:
    """从 market_data 组装 TrendSummary，委托 trend.py 规则模块。"""
    from app.services.trend import classify_trend_from_market_data

    trend_result = classify_trend_from_market_data(market_data)
    return trend_result.to_dict()


# ========== 可解释性 Meta ==========
def _build_explainability_meta(
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


# ========== Session 指代消解 ==========
def _resolve_with_session(question: str, session: SessionState | None) -> tuple[str, list[str] | None, str | None]:
    """利用 session 上下文消解指代。

    返回 (resolved_question, session_tickers, inherited_intent)
    """
    if session is None:
        return question, None, None

    resolved_question = question
    session_tickers = None
    inherited_intent = None

    # "它为什么涨" → 继承 session 中的资产
    pronoun_pattern = _re.compile(
        r"^(它|这只|这个|该股|这家|那个)(为什么|怎么|最近|现在)",
        _re.IGNORECASE,
    )
    if pronoun_pattern.search(question) and session.current_assets:
        session_tickers = session.current_assets

    # "那腾讯呢" → 继承 query 类型
    continuation_pattern = _re.compile(r"^那.{1,6}呢[？?]?$")
    if continuation_pattern.search(question) and session.last_intent:
        inherited_intent = session.last_intent

    # "再比一下" → compare 模式
    recompare_pattern = _re.compile(r"(再比|再对比|继续比|再比较)")
    if recompare_pattern.search(question) and session.last_intent == "compare":
        inherited_intent = "compare"
        if session.current_assets:
            session_tickers = session.current_assets

    return resolved_question, session_tickers, inherited_intent


# ========== 对比处理 ==========
_DISCLAIMER_COMPARE = "以上对比基于历史数据，不构成投资建议。不同时间窗口下结论可能不同。"


def handle_compare_question(
    question: str, tickers: list[str], steps: list[ThoughtStep], history: list[dict] | None = None
) -> dict:
    """处理多资产对比问题：逐个获取数据 → 计算对比指标 → LLM 分析。"""
    # Step: 逐个获取行情数据
    summaries = {}
    for ticker in tickers:
        steps.append(ThoughtStep(step=f"获取 {ticker} 数据", result="获取中..."))
        summary = get_stock_summary(ticker)
        summaries[ticker] = summary
        available = summary.get("data_available", False)
        steps[-1].result = f"data_available={available}"
        steps[-1].detail = {
            "current_price": summary.get("price", {}).get("current_price"),
        }

    # Step: 计算对比指标
    steps.append(ThoughtStep(step="计算对比指标", result="计算中..."))
    comparison = compute_comparison(summaries)
    steps[-1].result = f"对比完成，{len(comparison['assets'])} 个资产"

    # Step: LLM 生成对比分析
    steps.append(ThoughtStep(step="LLM对比分析", result="调用中..."))
    comparison_data_str = json.dumps(comparison["assets"], ensure_ascii=False, indent=2)
    winners_str = json.dumps(comparison["winners"], ensure_ascii=False, indent=2)
    user_message = COMPARE_USER_TEMPLATE.format(
        question=question,
        comparison_data=comparison_data_str,
        winners=winners_str,
    )
    text_response = chat_completion(COMPARE_SYSTEM_PROMPT, user_message, history=history)
    steps[-1].result = f"生成完毕，长度={len(text_response)}"

    # 组装结构化回答
    sources = [{"title": f"{t} 行情数据", "source": "yahoo", "url": None} for t in tickers]
    meta = _build_explainability_meta(
        data_sources=["yahoo_finance"],
        evidence_count=0,
        has_api_data=any(s.get("data_available") for s in summaries.values()),
    )
    structured_response = {
        "response_type": "compare",
        "data_summary": None,
        "trend_summary": None,
        "analysis": [{"title": "对比分析", "content": text_response}],
        "sources": sources,
        "disclaimer": _DISCLAIMER_COMPARE,
        "comparison": comparison,
        "meta": meta,
    }

    return {
        "text_response": text_response,
        "chart_data": None,
        "intent": "compare",
        "ticker": tickers[0],
        "tickers": tickers,
        "steps": [asdict(s) for s in steps],
        "structured_response": structured_response,
    }


# ========== 行情处理 ==========
def handle_market_question(question: str, ticker: str, steps: list[ThoughtStep], history: list[dict] | None = None) -> dict:
    """处理行情类问题：获取数据 → 数据可用性检查 → LLM 结构化回答。"""
    # Step: 调用行情 API
    steps.append(ThoughtStep(step="调用行情API", result=f"获取 {ticker} 数据中..."))
    market_data = get_stock_summary(ticker)

    data_available = market_data.get("data_available", False)
    steps[-1].result = f"ticker={ticker}, data_available={data_available}"
    steps[-1].detail = {
        "current_price": market_data.get("price", {}).get("current_price"),
        "7d_trend": market_data.get("change_7d", {}).get("trend"),
        "30d_trend": market_data.get("change_30d", {}).get("trend"),
    }

    # Step: 数据可用性判定 + 降级检测（grounding 层校验）
    validation = validate_market_data(market_data)
    is_stale = validation["is_stale"]

    if not data_available and not is_stale:
        # 完全无数据：API 失败 + 无历史缓存
        steps.append(ThoughtStep(
            step="数据校验",
            result="数据完全不可用，无缓存可降级",
            detail={"grounding_issues": validation["issues"]},
        ))
        text_response = (
            f"抱歉，当前无法获取 **{ticker}** 的行情数据。\n\n"
            "**可能原因：**\n"
            "- Yahoo Finance API 暂时不可用或被限流\n"
            "- 网络连接异常\n"
            "- 该 Ticker 可能不存在或已退市\n\n"
            "**建议：**\n"
            "- 请稍后重试\n"
            "- 或尝试使用标准代码查询（如 BABA、TSLA、AAPL）\n\n"
            "> 系统正在自动重试，如持续失败请检查网络连接。"
        )
        steps.append(ThoughtStep(step="生成降级回复", result="返回用户友好提示"))
        return {
            "text_response": text_response,
            "chart_data": None,
            "intent": "market_data",
            "ticker": ticker,
            "steps": [asdict(s) for s in steps],
        }

    if is_stale:
        # 降级模式：API 失败但有过期缓存
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

        steps.append(ThoughtStep(
            step="数据校验",
            result=f"API 不可用，降级使用缓存数据（{age_text}）",
            detail={"stale_age_seconds": max_age},
        ))
        market_data_str = json.dumps(market_data, ensure_ascii=False, indent=2)

        # 在 prompt 中注入降级提示，让 LLM 告知用户数据非实时
        stale_notice = (
            f"\n\n⚠️ 注意：行情 API 暂时不可用，以下数据来自 {age_text} 的缓存，"
            "可能非最新价格。请在回答开头明确提醒用户这是参考数据。"
        )
        user_message = MARKET_USER_TEMPLATE.format(
            question=question,
            market_data=market_data_str,
        ) + stale_notice
    else:
        steps.append(ThoughtStep(step="数据校验", result="数据校验通过（实时）"))
        market_data_str = json.dumps(market_data, ensure_ascii=False, indent=2)
        user_message = MARKET_USER_TEMPLATE.format(
            question=question,
            market_data=market_data_str,
        )

    # 注入缺失字段提示（grounding 层：防止 LLM 编造缺失数据）
    missing_notice = build_missing_field_notice(market_data)
    if missing_notice:
        user_message += f"\n\n⚠️ {missing_notice}请在回答中明确告知用户哪些数据无法获取，不要编造。"

    # Step: LLM 生成回答
    steps.append(ThoughtStep(step="LLM生成回答", result="调用中..."))
    text_response = chat_completion(MARKET_SYSTEM_PROMPT, user_message, history=history)
    steps[-1].result = f"生成完毕，长度={len(text_response)}"

    # Step: 数字交叉验证（grounding 层：检查 LLM 回答中的数字是否匹配原始数据）
    nums_ok, suspicious = validate_market_response_numbers(text_response, market_data)
    if not nums_ok:
        steps.append(ThoughtStep(
            step="数字交叉验证",
            result=f"警告: 回答中出现可疑数字 {suspicious}",
            detail={"suspicious_numbers": suspicious},
        ))
        logger.warning("[Grounding] Suspicious numbers in response: %s", suspicious)
    else:
        steps.append(ThoughtStep(step="数字交叉验证", result="通过"))

    # 提取图表数据（降级数据也可出图）
    chart_data = None
    change_30d = market_data.get("change_30d", {})
    change_7d = market_data.get("change_7d", {})
    if "history" in change_30d:
        chart_data = change_30d["history"]
    elif "history" in change_7d:
        chart_data = change_7d["history"]

    # 提取 market_meta 用于 KPI 卡片（至少有 current_price 才发送）
    price_info = market_data.get("price", {})
    raw_meta = {
        "current_price": price_info.get("current_price"),
        "previous_close": price_info.get("previous_close"),
        "change_pct": change_7d.get("change_pct"),
        "pe_ratio": price_info.get("pe_ratio"),
        "market_cap": price_info.get("market_cap"),
        "name": price_info.get("name", ticker),
        "currency": price_info.get("currency", "USD"),
    }
    market_meta = raw_meta if raw_meta.get("current_price") is not None else None

    # 组装结构化回答（程序端组装，保证准确性）
    meta = _build_explainability_meta(
        data_sources=["yahoo_finance"],
        has_api_data=data_available,
    )
    structured_response = {
        "response_type": "market_data",
        "data_summary": _build_data_summary(market_data),
        "trend_summary": _build_trend_summary(market_data),
        "analysis": [{"title": "分析与解读", "content": text_response}],
        "sources": [{"title": "行情数据", "source": market_data.get("data_source", "yahoo"), "url": None}],
        "disclaimer": _DISCLAIMER_MARKET,
        "meta": meta,
    }

    return {
        "text_response": text_response,
        "chart_data": chart_data,
        "intent": "market_data",
        "ticker": ticker,
        "steps": [asdict(s) for s in steps],
        "market_meta": market_meta,
        "structured_response": structured_response,
    }


# ========== 事实核查（委托 grounding 模块） ==========
def _fact_check(text_response: str, context: str, steps: list[ThoughtStep]) -> tuple[bool, str]:
    """调用 grounding.fact_check 检查生成内容中的关键数值是否与检索上下文一致。

    返回 (passed, issues)。
    仅在有 RAG context 时执行 — 无参考资料则无从核查。
    """
    steps.append(ThoughtStep(step="事实核查", result="校验中..."))

    passed, issues = fact_check(
        text_response=text_response,
        context=context,
        llm_call=chat_completion,
        fact_check_prompt_template=FACT_CHECK_PROMPT,
    )

    if passed:
        steps[-1].result = "通过"
    else:
        steps[-1].result = f"未通过: {issues}"

    return passed, issues


# ========== 行情原因分析处理 ==========
def handle_market_reasoning_question(
    question: str, ticker: str, steps: list[ThoughtStep], history: list[dict] | None = None
) -> dict:
    """处理行情原因分析类问题：行情数据 + Web 搜索证据 + LLM 归因。

    完整链路：
    1. 提取日期引用（如果用户指定了特定日期）
    2. 获取行情数据（复用现有 get_stock_summary）
    3. 构造优化搜索 query → 搜索相关新闻/事件
    4. LLM 基于数据+证据生成可能原因分析
    5. 组装 structured_response
    """
    # Step: 日期提取
    date_ref = extract_date_ref(question)
    if date_ref:
        steps.append(ThoughtStep(
            step="日期识别",
            result=f"识别到时间引用: {date_ref.date_str}",
            detail={"date_str": date_ref.date_str, "search_hint": date_ref.search_hint},
        ))
    else:
        steps.append(ThoughtStep(step="日期识别", result="未指定具体日期，分析近期走势"))

    # Step: 调用行情 API（复用）
    steps.append(ThoughtStep(step="调用行情API", result=f"获取 {ticker} 数据中..."))
    market_data = get_stock_summary(ticker)
    data_available = market_data.get("data_available", False)
    steps[-1].result = f"ticker={ticker}, data_available={data_available}"
    steps[-1].detail = {
        "current_price": market_data.get("price", {}).get("current_price"),
        "7d_trend": market_data.get("change_7d", {}).get("trend"),
    }

    # Step: Web 搜索相关新闻/事件（使用优化搜索 query）
    steps.append(ThoughtStep(step="搜索新闻证据", result="搜索中..."))
    price_info = market_data.get("price", {})
    asset_name = price_info.get("name", ticker)

    # 获取趋势方向用于优化搜索词
    trend_dir = market_data.get("change_7d", {}).get("trend_detail", {}).get("label")
    search_query = build_search_query(
        asset_name=asset_name,
        ticker=ticker,
        question=question,
        date_ref=date_ref,
        trend_direction=trend_dir,
    )
    steps[-1].detail = {"search_query": search_query}

    evidence = web_search(search_query, max_results=5)
    evidence_sources: list[dict] = []
    evidence_analysis_result = None

    if evidence:
        steps[-1].result = f"搜索命中，证据长度={len(evidence)}"
        evidence_sources = _parse_web_sources(evidence)

        # Step: 证据分类
        steps.append(ThoughtStep(step="证据分类", result="分类中..."))
        ea = classify_evidence(evidence, asset_name, ticker)
        evidence_analysis_result = {
            "main_drivers": ea.main_drivers,
            "secondary_drivers": ea.secondary_drivers,
            "evidence_strength": ea.evidence_strength,
            "summary": ea.summary,
        }
        steps[-1].result = ea.summary
        steps[-1].detail = {"main_drivers": ea.main_drivers, "strength": ea.evidence_strength}
    else:
        steps[-1].result = "未配置搜索 API 或搜索无结果"
        evidence = "（未检索到相关新闻或事件证据）"

    # 构建 prompt（注入分类摘要）
    market_data_str = json.dumps(market_data, ensure_ascii=False, indent=2)
    user_message = MARKET_REASONING_USER_TEMPLATE.format(
        question=question,
        market_data=market_data_str,
        evidence=evidence,
    )
    if evidence_analysis_result and evidence_analysis_result.get("summary"):
        user_message += f"\n\n证据分类摘要：{evidence_analysis_result['summary']}"

    # Step: LLM 生成原因分析
    steps.append(ThoughtStep(step="LLM归因分析", result="调用中..."))
    text_response = chat_completion(MARKET_REASONING_SYSTEM_PROMPT, user_message, history=history)
    steps[-1].result = f"生成完毕，长度={len(text_response)}"

    # 提取图表数据
    chart_data = None
    change_30d = market_data.get("change_30d", {})
    change_7d = market_data.get("change_7d", {})
    if "history" in change_30d:
        chart_data = change_30d["history"]
    elif "history" in change_7d:
        chart_data = change_7d["history"]

    # 提取 market_meta
    raw_meta = {
        "current_price": price_info.get("current_price"),
        "previous_close": price_info.get("previous_close"),
        "change_pct": change_7d.get("change_pct"),
        "pe_ratio": price_info.get("pe_ratio"),
        "market_cap": price_info.get("market_cap"),
        "name": price_info.get("name", ticker),
        "currency": price_info.get("currency", "USD"),
    }
    market_meta = raw_meta if raw_meta.get("current_price") is not None else None

    # 组装结构化回答
    all_sources = [{"title": "行情数据", "source": market_data.get("data_source", "yahoo"), "url": None}]
    all_sources.extend(evidence_sources)

    meta = _build_explainability_meta(
        data_sources=["yahoo_finance", "web_search"],
        evidence_count=len(evidence_sources),
        has_api_data=data_available,
    )
    structured_response = {
        "response_type": "market_reasoning",
        "data_summary": _build_data_summary(market_data),
        "trend_summary": _build_trend_summary(market_data),
        "analysis": [{"title": "可能原因分析", "content": text_response}],
        "sources": all_sources,
        "disclaimer": _DISCLAIMER_REASONING,
        "evidence_analysis": evidence_analysis_result,
        "meta": meta,
    }

    return {
        "text_response": text_response,
        "chart_data": chart_data,
        "intent": "market_reasoning",
        "ticker": ticker,
        "rag_used": bool(evidence_sources),
        "steps": [asdict(s) for s in steps],
        "market_meta": market_meta,
        "structured_response": structured_response,
    }


# ========== 知识库处理 ==========
def handle_knowledge_question(question: str, steps: list[ThoughtStep], history: list[dict] | None = None) -> dict:
    """处理知识类问题：RAG 检索 → LLM 回答 → 事实核查（有 RAG 命中时）。"""
    # Step: RAG 检索
    steps.append(ThoughtStep(step="RAG向量检索", result="检索中..."))
    context = ""
    rag_used = False
    rag_sources: list[dict] = []
    try:
        context = get_relevant_context(question, top_k=4)
        if context:
            rag_used = True
            steps[-1].result = f"命中，上下文长度={len(context)}"
            rag_sources = _parse_rag_sources(context)
        else:
            steps[-1].result = "未命中相关文档"
    except Exception as e:
        steps[-1].result = f"检索失败: {e}"
        logger.warning("RAG retrieval failed: %s", e)

    # Web search 兜底（RAG 未命中时）
    web_context = ""
    web_sources: list[dict] = []
    if not rag_used:
        steps.append(ThoughtStep(step="Web搜索兜底", result="搜索中..."))
        web_context = web_search(question)
        if web_context:
            steps[-1].result = f"搜索命中，长度={len(web_context)}"
            context = web_context
            rag_used = True
            web_sources = _parse_web_sources(web_context)
        else:
            steps[-1].result = "未配置搜索 API 或搜索无结果"

    # Step: 构建 Prompt
    if rag_used:
        system_prompt = RAG_SYSTEM_PROMPT
        user_message = RAG_USER_TEMPLATE.format(question=question, context=context)
        steps.append(ThoughtStep(step="Prompt构建", result="使用参考资料"))
    else:
        system_prompt = (
            RAG_SYSTEM_PROMPT
            + "\n\n注意：知识库中未检索到相关资料，请基于你的通用金融知识回答，"
            '并在回答末尾注明"本回答基于模型通用知识，未引用知识库资料"。'
        )
        user_message = RAG_USER_TEMPLATE.format(
            question=question, context="（未检索到相关参考资料）"
        )
        steps.append(ThoughtStep(step="Prompt构建", result="回退至模型通用知识"))

    # Step: LLM 生成（第一次）
    steps.append(ThoughtStep(step="LLM生成回答", result="调用中..."))
    text_response = chat_completion(system_prompt, user_message, history=history)
    steps[-1].result = f"生成完毕，长度={len(text_response)}"

    # Step: 事实核查 — 仅在有 RAG context 时执行
    if rag_used and context:
        passed, issues = _fact_check(text_response, context, steps)

        if not passed:
            # 重试一次：在 system prompt 中注入核查反馈
            steps.append(ThoughtStep(
                step="LLM重新生成",
                result=f"事实核查未通过（{issues}），重新生成...",
            ))
            retry_system = (
                system_prompt
                + f"\n\n【重要】上一次回答存在事实偏差：{issues}。"
                "请严格依据参考资料修正，不要编造数据。"
            )
            text_response = chat_completion(retry_system, user_message, history=history)
            steps[-1].result = f"重新生成完毕，长度={len(text_response)}"

    # 组装结构化回答
    all_sources = rag_sources + web_sources
    data_src = []
    if rag_sources:
        data_src.append("knowledge_base")
    if web_sources:
        data_src.append("web_search")
    meta = _build_explainability_meta(
        data_sources=data_src,
        evidence_count=len(all_sources),
        has_api_data=bool(all_sources),
    )
    structured_response = {
        "response_type": "knowledge_rag",
        "data_summary": None,
        "trend_summary": None,
        "analysis": [{"title": "知识问答", "content": text_response}],
        "sources": all_sources,
        "disclaimer": None,
        "meta": meta,
    }

    return {
        "text_response": text_response,
        "chart_data": None,
        "intent": "knowledge_rag",
        "rag_used": rag_used,
        "steps": [asdict(s) for s in steps],
        "structured_response": structured_response,
    }


# ========== 通用处理 ==========
def handle_general_question(question: str, steps: list[ThoughtStep], history: list[dict] | None = None) -> dict:
    """处理通用问题。"""
    steps.append(ThoughtStep(step="LLM生成回答", result="通用问答"))
    system = "你是一个友好的金融助手。简洁回答用户问题，使用用户的语言。回答要分点清晰，要点之间空一行，关键内容加粗。"
    text_response = chat_completion(system, question, history=history)
    steps[-1].result = f"生成完毕，长度={len(text_response)}"

    return {
        "text_response": text_response,
        "chart_data": None,
        "intent": "general",
        "steps": [asdict(s) for s in steps],
    }


# ========== 上下文 Ticker 回溯 ==========
def _resolve_ticker_with_context(question: str, history: list[dict] | None) -> tuple[str | None, str]:
    """从当前问题 + 历史对话中识别 Ticker。

    返回 (ticker, source):
      - source: "current" 当前问题直接匹配
      - source: "history" 从历史消息中回溯
      - source: None 未匹配
    """
    # 优先从当前问题识别
    ticker = resolve_ticker(question)
    if ticker:
        return ticker, "current"

    # 当前问题含有指代词/后续追问模式时，从历史中回溯
    follow_up_patterns = _re.compile(
        r"(它|这只|这个|该股|这家|那个|上面|刚才|之前|前面|继续|还有|另外|"
        r"其|the stock|this|that|same)",
        _re.IGNORECASE,
    )
    if not history or not follow_up_patterns.search(question):
        # 没有指代词也没有 history，可能是独立的 market 问题但没给 ticker
        # 也尝试从历史回溯（例如 "最近财报如何？" 紧跟在 "阿里巴巴股价" 之后）
        if not history:
            return None, ""

    # 从最近的历史消息中查找 ticker（倒序）
    for msg in reversed(history):
        if msg.get("role") == "user":
            t = resolve_ticker(msg.get("content", ""))
            if t:
                return t, "history"

    return None, ""


# ========== 查询改写 ==========
# 快速跳过改写的信号：问题已包含明确的动作词 + ticker 名称
_CLEAR_QUERY_PATTERN = _re.compile(
    r"(股价|对比|比较|什么是|涨跌|走势|市值|行情|vs|price|compare|what\s+is)",
    _re.IGNORECASE,
)
# 需要改写的信号：代词或模糊词
_VAGUE_QUERY_PATTERN = _re.compile(
    r"(它|这只|这个|该股|这家|那个|怎么样|最近|情况|如何$)",
    _re.IGNORECASE,
)


def _transform_query(
    question: str,
    history: list[dict] | None,
    session: SessionState | None,
    steps: list[ThoughtStep],
) -> str:
    """将模糊/指代性查询改写为明确查询。

    快速跳过已经明确的问题，避免额外 LLM 调用。
    """
    # 快速跳过：已经足够明确
    if _CLEAR_QUERY_PATTERN.search(question) and not _VAGUE_QUERY_PATTERN.search(question):
        return question

    # 无改写信号也跳过
    if not _VAGUE_QUERY_PATTERN.search(question):
        return question

    # 构建上下文
    history_text = ""
    if history:
        recent = history[-4:]  # 最近 4 条
        history_text = "\n".join(
            f"{m.get('role', '?')}: {m.get('content', '')[:100]}" for m in recent
        )

    session_assets = ""
    if session and session.current_assets:
        session_assets = ", ".join(session.current_assets)

    prompt = QUERY_TRANSFORM_PROMPT.format(
        history=history_text or "（无历史）",
        session_assets=session_assets or "（无）",
        question=question,
    )

    try:
        rewritten = chat_completion("你是一个查询改写助手。", prompt).strip()
        if rewritten and rewritten != question:
            steps.append(ThoughtStep(
                step="查询改写",
                result=f"'{question}' → '{rewritten}'",
            ))
            return rewritten
    except Exception as e:
        logger.warning("[QueryTransform] LLM call failed: %s", e)

    return question


# ========== 路由决策核心逻辑 ==========
def _route_intent(question: str, ticker: str | None, steps: list[ThoughtStep]) -> str:
    """确定最终意图路由。

    优先级：
    1. 快速正则分类（节省 API 调用）
    2. Function Calling 分类（替代纯文本 classify_intent）
    3. 如果有 ticker 但分类为 general，强制升级为 market_data
    """
    # 1. 正则快速分类
    fast_intent = fast_classify_intent(question)
    if fast_intent:
        steps.append(ThoughtStep(step="快速意图分类", result=f"正则命中: {fast_intent}"))
        return fast_intent

    # 2. Function Calling 分类
    steps.append(ThoughtStep(step="Tool-based意图分类", result="分类中..."))
    intent, tool_args = classify_intent_with_tools(question)
    steps[-1].result = f"Tool 选择: {intent}"
    steps[-1].detail = {"tool_args": tool_args}

    # 3. Ticker 强制升级逻辑
    if ticker and intent not in ("market_data", "market_reasoning"):
        steps.append(ThoughtStep(step="意图修正", result=f"有 Ticker({ticker})，从 {intent} 升级为 market_data"))
        intent = "market_data"

    return intent


# ========== 主入口（支持多轮对话） ==========
@audit_log
def process_question(question: str, history: list[dict] | None = None, session: SessionState | None = None) -> dict:
    """主入口：接收用户问题，执行分步思考链，路由到对应处理逻辑。

    思考链（steps）完整记录了 Agent 的决策过程：
    1. Session 消解 → 2. Ticker 识别 → 3. 意图分类 → 4. Tool 调用 → 5. 数据校验 → 6. LLM 生成
    每一步都有 step/result/detail，便于调试和审计。
    """
    steps: list[ThoughtStep] = []

    # Step 0: Session 指代消解
    session_tickers = None
    inherited_intent = None
    if session:
        question, session_tickers, inherited_intent = _resolve_with_session(question, session)
        if session_tickers:
            steps.append(ThoughtStep(
                step="Session消解",
                result=f"从会话继承资产: {session_tickers}",
                detail={"session_tickers": session_tickers, "inherited_intent": inherited_intent},
            ))

    # Step 0.5: 查询改写（模糊/指代性查询 → 明确查询）
    question = _transform_query(question, history, session, steps)

    # Step 1: 多 ticker 提取（对比场景）
    multi_tickers = resolve_tickers_multi(question)
    if session_tickers and len(multi_tickers) < 2:
        multi_tickers = session_tickers

    # Step 1b: 单 Ticker 识别（含上下文回溯）
    ticker, ticker_source = _resolve_ticker_with_context(question, history)
    if ticker and ticker_source == "history":
        steps.append(ThoughtStep(
            step="Ticker识别",
            result=f"当前问题未匹配，从历史对话回溯到 {ticker}",
            detail={"ticker": ticker, "source": "history"},
        ))
    else:
        steps.append(ThoughtStep(
            step="Ticker识别",
            result=f"匹配到 {ticker}" if ticker else "未匹配到股票代码",
            detail={"ticker": ticker, "multi_tickers": multi_tickers},
        ))

    # Step 2: 路由决策
    # compare 路由优先：2+ ticker + compare 意图
    intent = inherited_intent or _route_intent(question, ticker, steps)

    if intent == "compare" and len(multi_tickers) >= 2:
        steps.append(ThoughtStep(step="意图路由", result=f"compare（{len(multi_tickers)} 个资产）"))
        result = handle_compare_question(question, multi_tickers, steps, history=history)
        if session:
            session.update(assets=multi_tickers, intent="compare")
        return result

    if ticker:
        if intent == "market_reasoning":
            steps.append(ThoughtStep(step="意图路由", result="market_reasoning（原因分析）"))
            result = handle_market_reasoning_question(question, ticker, steps, history=history)
            if session:
                session.update(assets=[ticker], intent="market_reasoning")
            return result
        else:
            steps.append(ThoughtStep(step="意图路由", result="market_data（基于 Ticker 命中）"))
            result = handle_market_question(question, ticker, steps, history=history)
            if session:
                session.update(assets=[ticker], intent="market_data")
            return result

    # Step 3: 无 ticker 时的路由
    if intent not in ("compare",):
        intent = _route_intent(question, ticker, steps)

    if intent in ("market_data", "market_reasoning"):
        # 最后一次尝试：从历史中找 ticker
        if history:
            for msg in reversed(history):
                if msg.get("role") == "user":
                    t = resolve_ticker(msg.get("content", ""))
                    if t:
                        ticker = t
                        steps.append(ThoughtStep(step="上下文回溯", result=f"从历史中找到 {ticker}"))
                        if intent == "market_reasoning":
                            return handle_market_reasoning_question(question, ticker, steps, history=history)
                        return handle_market_question(question, ticker, steps, history=history)

        steps.append(ThoughtStep(
            step="意图路由",
            result=f"{intent}（但未识别 Ticker，要求用户补充）",
        ))
        return {
            "text_response": (
                "抱歉，我无法从您的问题中识别出具体的股票或资产。"
                "请提供股票名称或代码（如：阿里巴巴、TSLA、特斯拉）。"
            ),
            "chart_data": None,
            "intent": intent,
            "steps": [asdict(s) for s in steps],
        }
    elif intent == "knowledge_rag":
        steps.append(ThoughtStep(step="意图路由", result="knowledge_rag"))
        return handle_knowledge_question(question, steps, history=history)
    else:
        steps.append(ThoughtStep(step="意图路由", result="general"))
        return handle_general_question(question, steps, history=history)


# ========== 流式主入口 ==========
def process_question_stream(question: str, history: list[dict] | None = None, session: SessionState | None = None):
    """流式生成器：yield SSE 事件字典。

    事件类型：thought / token / chart / meta / done / error
    """
    import time as _time

    start = _time.time()
    steps: list[ThoughtStep] = []

    try:
        # Step 0: Session 指代消解
        session_tickers = None
        inherited_intent = None
        if session:
            question, session_tickers, inherited_intent = _resolve_with_session(question, session)
            if session_tickers:
                steps.append(ThoughtStep(
                    step="Session消解",
                    result=f"从会话继承资产: {session_tickers}",
                ))
                yield {"event": "thought", "data": asdict(steps[-1])}

        # Step 0.5: 查询改写（模糊/指代性查询 → 明确查询）
        question = _transform_query(question, history, session, steps)
        if steps and steps[-1].step == "查询改写":
            yield {"event": "thought", "data": asdict(steps[-1])}

        # Step 1: 多 ticker 提取
        multi_tickers = resolve_tickers_multi(question)
        if session_tickers and len(multi_tickers) < 2:
            multi_tickers = session_tickers

        # Step 1b: Ticker 识别（含上下文回溯）
        ticker, ticker_source = _resolve_ticker_with_context(question, history)
        if ticker and ticker_source == "history":
            steps.append(ThoughtStep(
                step="Ticker识别",
                result=f"当前问题未匹配，从历史对话回溯到 {ticker}",
                detail={"ticker": ticker, "source": "history"},
            ))
        else:
            steps.append(ThoughtStep(
                step="Ticker识别",
                result=f"匹配到 {ticker}" if ticker else "未匹配到股票代码",
                detail={"ticker": ticker, "multi_tickers": multi_tickers},
            ))
        yield {"event": "thought", "data": asdict(steps[-1])}

        # Step 2: 路由决策
        intent = inherited_intent or _route_intent(question, ticker, steps)

        # compare 路由优先
        if intent == "compare" and len(multi_tickers) >= 2:
            steps.append(ThoughtStep(step="意图路由", result=f"compare（{len(multi_tickers)} 个资产）"))
            yield {"event": "thought", "data": asdict(steps[-1])}
            yield from _stream_compare(question, multi_tickers, steps, history)
            if session:
                session.update(assets=multi_tickers, intent="compare")
            return

        if ticker:
            for s in steps[-2:]:
                yield {"event": "thought", "data": asdict(s)}

            if intent == "market_reasoning":
                steps.append(ThoughtStep(step="意图路由", result="market_reasoning（原因分析）"))
                yield {"event": "thought", "data": asdict(steps[-1])}
                yield from _stream_market_reasoning(question, ticker, steps, history)
                if session:
                    session.update(assets=[ticker], intent="market_reasoning")
            else:
                steps.append(ThoughtStep(step="意图路由", result="market_data（基于 Ticker 命中）"))
                yield {"event": "thought", "data": asdict(steps[-1])}
                yield from _stream_market(question, ticker, steps, history)
                if session:
                    session.update(assets=[ticker], intent="market_data")
            return

        # Step 3: 无 ticker 的路由
        if not inherited_intent:
            intent = _route_intent(question, ticker, steps)
        for s in steps[-2:]:
            yield {"event": "thought", "data": asdict(s)}

        if intent in ("market_data", "market_reasoning"):
            # 最后一次尝试：从历史中找 ticker
            if history:
                for msg in reversed(history):
                    if msg.get("role") == "user":
                        t = resolve_ticker(msg.get("content", ""))
                        if t:
                            ticker = t
                            steps.append(ThoughtStep(step="上下文回溯", result=f"从历史中找到 {ticker}"))
                            yield {"event": "thought", "data": asdict(steps[-1])}
                            if intent == "market_reasoning":
                                yield from _stream_market_reasoning(question, ticker, steps, history)
                            else:
                                yield from _stream_market(question, ticker, steps, history)
                            return

            steps.append(ThoughtStep(step="意图路由", result=f"{intent}（未识别 Ticker）"))
            yield {"event": "thought", "data": asdict(steps[-1])}
            yield {"event": "meta", "data": {"intent": intent, "ticker": None, "rag_used": None}}
            text = (
                "抱歉，我无法从您的问题中识别出具体的股票或资产。"
                "请提供股票名称或代码（如：阿里巴巴、TSLA、特斯拉）。"
            )
            yield {"event": "token", "data": text}
            yield {"event": "done", "data": {"steps": [asdict(s) for s in steps]}}
            return

        if intent == "knowledge_rag":
            steps.append(ThoughtStep(step="意图路由", result="knowledge_rag"))
            yield {"event": "thought", "data": asdict(steps[-1])}
            yield from _stream_knowledge(question, steps, history)
        else:
            steps.append(ThoughtStep(step="意图路由", result="general"))
            yield {"event": "thought", "data": asdict(steps[-1])}
            yield from _stream_general(question, steps, history)

    except Exception as e:
        logger.error("[STREAM] Error: %s", e)
        yield {"event": "error", "data": {"message": str(e)}}


def _stream_compare(question: str, tickers: list[str], steps: list[ThoughtStep], history: list[dict] | None = None):
    """流式多资产对比。"""
    # 逐个获取数据，每个 ticker 一个 thought 事件
    summaries = {}
    for ticker in tickers:
        steps.append(ThoughtStep(step=f"获取 {ticker} 数据", result="获取中..."))
        yield {"event": "thought", "data": asdict(steps[-1])}

        summary = get_stock_summary(ticker)
        summaries[ticker] = summary
        available = summary.get("data_available", False)
        steps[-1].result = f"data_available={available}"
        steps[-1].detail = {"current_price": summary.get("price", {}).get("current_price")}
        yield {"event": "thought", "data": asdict(steps[-1])}

    # 计算对比指标
    steps.append(ThoughtStep(step="计算对比指标", result="计算中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}
    comparison = compute_comparison(summaries)
    steps[-1].result = f"对比完成，{len(comparison['assets'])} 个资产"
    yield {"event": "thought", "data": asdict(steps[-1])}

    # Meta 事件
    meta = _build_explainability_meta(
        data_sources=["yahoo_finance"],
        has_api_data=any(s.get("data_available") for s in summaries.values()),
    )
    structured_response = {
        "response_type": "compare",
        "data_summary": None,
        "trend_summary": None,
        "analysis": [],
        "sources": [{"title": f"{t} 行情数据", "source": "yahoo", "url": None} for t in tickers],
        "disclaimer": _DISCLAIMER_COMPARE,
        "comparison": comparison,
        "meta": meta,
    }
    yield {"event": "meta", "data": {
        "intent": "compare", "ticker": tickers[0], "tickers": tickers,
        "rag_used": None,
        "structured_response": structured_response,
    }}

    # 流式 LLM
    comparison_data_str = json.dumps(comparison["assets"], ensure_ascii=False, indent=2)
    winners_str = json.dumps(comparison["winners"], ensure_ascii=False, indent=2)
    user_message = COMPARE_USER_TEMPLATE.format(
        question=question,
        comparison_data=comparison_data_str,
        winners=winners_str,
    )

    steps.append(ThoughtStep(step="LLM对比分析", result="流式生成中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}
    for token in chat_completion_stream(COMPARE_SYSTEM_PROMPT, user_message, history=history):
        yield {"event": "token", "data": token}

    steps[-1].result = "生成完毕"
    yield {"event": "done", "data": {"steps": [asdict(s) for s in steps]}}


def _stream_market(question: str, ticker: str, steps: list[ThoughtStep], history: list[dict] | None = None):
    """流式行情回答。"""
    steps.append(ThoughtStep(step="调用行情API", result=f"获取 {ticker} 数据中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}

    market_data = get_stock_summary(ticker)
    data_available = market_data.get("data_available", False)
    steps[-1].result = f"ticker={ticker}, data_available={data_available}"
    steps[-1].detail = {
        "current_price": market_data.get("price", {}).get("current_price"),
        "7d_trend": market_data.get("change_7d", {}).get("trend"),
        "30d_trend": market_data.get("change_30d", {}).get("trend"),
    }
    yield {"event": "thought", "data": asdict(steps[-1])}

    # 提取图表数据
    chart_data = None
    change_30d = market_data.get("change_30d", {})
    change_7d = market_data.get("change_7d", {})
    if "history" in change_30d:
        chart_data = change_30d["history"]
    elif "history" in change_7d:
        chart_data = change_7d["history"]

    # 提取 market_meta（至少有 current_price 才发送）
    price_info = market_data.get("price", {})
    raw_meta = {
        "current_price": price_info.get("current_price"),
        "previous_close": price_info.get("previous_close"),
        "change_pct": change_7d.get("change_pct"),
        "pe_ratio": price_info.get("pe_ratio"),
        "market_cap": price_info.get("market_cap"),
        "name": price_info.get("name", ticker),
        "currency": price_info.get("currency", "USD"),
    }
    market_meta = raw_meta if raw_meta.get("current_price") is not None else None

    # 组装结构化回答
    structured_response = {
        "response_type": "market_data",
        "data_summary": _build_data_summary(market_data),
        "trend_summary": _build_trend_summary(market_data),
        "analysis": [],
        "sources": [{"title": "行情数据", "source": market_data.get("data_source", "yahoo"), "url": None}],
        "disclaimer": _DISCLAIMER_MARKET,
    }

    yield {"event": "meta", "data": {
        "intent": "market_data", "ticker": ticker, "rag_used": None,
        "market_meta": market_meta,
        "structured_response": structured_response,
    }}

    if chart_data:
        yield {"event": "chart", "data": chart_data}

    is_stale = any(
        market_data.get(k, {}).get("stale", False)
        for k in ("price", "change_7d", "change_30d")
    )

    if not data_available and not is_stale:
        steps.append(ThoughtStep(step="数据校验", result="数据完全不可用"))
        yield {"event": "thought", "data": asdict(steps[-1])}
        text = (
            f"抱歉，当前无法获取 **{ticker}** 的行情数据。\n\n"
            "**可能原因：**\n- Yahoo Finance API 暂时不可用或被限流\n- 网络连接异常\n\n"
            "**建议：** 请稍后重试"
        )
        yield {"event": "token", "data": text}
        yield {"event": "done", "data": {"steps": [asdict(s) for s in steps]}}
        return

    # 构建 prompt
    market_data_str = json.dumps(market_data, ensure_ascii=False, indent=2)
    if is_stale:
        steps.append(ThoughtStep(step="数据校验", result="降级使用缓存数据"))
        stale_notice = "\n\n⚠️ 注意：行情 API 暂时不可用，以下数据来自缓存，可能非最新价格。"
        user_message = MARKET_USER_TEMPLATE.format(question=question, market_data=market_data_str) + stale_notice
    else:
        steps.append(ThoughtStep(step="数据校验", result="数据校验通过（实时）"))
        user_message = MARKET_USER_TEMPLATE.format(question=question, market_data=market_data_str)
    yield {"event": "thought", "data": asdict(steps[-1])}

    # 流式 LLM
    steps.append(ThoughtStep(step="LLM生成回答", result="流式生成中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}
    for token in chat_completion_stream(MARKET_SYSTEM_PROMPT, user_message, history=history):
        yield {"event": "token", "data": token}

    steps[-1].result = "生成完毕"
    yield {"event": "done", "data": {"steps": [asdict(s) for s in steps]}}


def _stream_market_reasoning(question: str, ticker: str, steps: list[ThoughtStep], history: list[dict] | None = None):
    """流式行情原因分析。"""
    # Step: 日期提取
    date_ref = extract_date_ref(question)
    if date_ref:
        steps.append(ThoughtStep(
            step="日期识别",
            result=f"识别到时间引用: {date_ref.date_str}",
            detail={"date_str": date_ref.date_str, "search_hint": date_ref.search_hint},
        ))
    else:
        steps.append(ThoughtStep(step="日期识别", result="未指定具体日期，分析近期走势"))
    yield {"event": "thought", "data": asdict(steps[-1])}

    # Step: 调用行情 API
    steps.append(ThoughtStep(step="调用行情API", result=f"获取 {ticker} 数据中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}

    market_data = get_stock_summary(ticker)
    data_available = market_data.get("data_available", False)
    steps[-1].result = f"ticker={ticker}, data_available={data_available}"
    yield {"event": "thought", "data": asdict(steps[-1])}

    # 提取图表和 meta
    chart_data = None
    change_30d = market_data.get("change_30d", {})
    change_7d = market_data.get("change_7d", {})
    if "history" in change_30d:
        chart_data = change_30d["history"]
    elif "history" in change_7d:
        chart_data = change_7d["history"]

    price_info = market_data.get("price", {})
    raw_meta = {
        "current_price": price_info.get("current_price"),
        "previous_close": price_info.get("previous_close"),
        "change_pct": change_7d.get("change_pct"),
        "pe_ratio": price_info.get("pe_ratio"),
        "market_cap": price_info.get("market_cap"),
        "name": price_info.get("name", ticker),
        "currency": price_info.get("currency", "USD"),
    }
    market_meta = raw_meta if raw_meta.get("current_price") is not None else None

    # Step: 搜索新闻证据（优化搜索 query）
    steps.append(ThoughtStep(step="搜索新闻证据", result="搜索中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}

    asset_name = price_info.get("name", ticker)
    trend_dir = market_data.get("change_7d", {}).get("trend_detail", {}).get("label")
    search_query = build_search_query(
        asset_name=asset_name, ticker=ticker, question=question,
        date_ref=date_ref, trend_direction=trend_dir,
    )
    steps[-1].detail = {"search_query": search_query}

    evidence = web_search(search_query, max_results=5)
    evidence_sources: list[dict] = []
    evidence_analysis_result = None

    if evidence:
        steps[-1].result = f"搜索命中，证据长度={len(evidence)}"
        evidence_sources = _parse_web_sources(evidence)
        yield {"event": "thought", "data": asdict(steps[-1])}

        # 证据分类
        steps.append(ThoughtStep(step="证据分类", result="分类中..."))
        yield {"event": "thought", "data": asdict(steps[-1])}
        ea = classify_evidence(evidence, asset_name, ticker)
        evidence_analysis_result = {
            "main_drivers": ea.main_drivers,
            "secondary_drivers": ea.secondary_drivers,
            "evidence_strength": ea.evidence_strength,
            "summary": ea.summary,
        }
        steps[-1].result = ea.summary
        yield {"event": "thought", "data": asdict(steps[-1])}
    else:
        steps[-1].result = "未配置搜索 API 或搜索无结果"
        evidence = "（未检索到相关新闻或事件证据）"
        yield {"event": "thought", "data": asdict(steps[-1])}

    # 组装结构化回答
    all_sources = [{"title": "行情数据", "source": market_data.get("data_source", "yahoo"), "url": None}]
    all_sources.extend(evidence_sources)

    meta = _build_explainability_meta(
        data_sources=["yahoo_finance", "web_search"],
        evidence_count=len(evidence_sources),
        has_api_data=data_available,
    )
    structured_response = {
        "response_type": "market_reasoning",
        "data_summary": _build_data_summary(market_data),
        "trend_summary": _build_trend_summary(market_data),
        "analysis": [],
        "sources": all_sources,
        "disclaimer": _DISCLAIMER_REASONING,
        "evidence_analysis": evidence_analysis_result,
        "meta": meta,
    }

    yield {"event": "meta", "data": {
        "intent": "market_reasoning", "ticker": ticker,
        "rag_used": bool(evidence_sources),
        "market_meta": market_meta,
        "structured_response": structured_response,
    }}

    if chart_data:
        yield {"event": "chart", "data": chart_data}

    if not data_available:
        steps.append(ThoughtStep(step="数据校验", result="数据完全不可用"))
        yield {"event": "thought", "data": asdict(steps[-1])}
        text = f"抱歉，当前无法获取 **{ticker}** 的行情数据，无法进行原因分析。请稍后重试。"
        yield {"event": "token", "data": text}
        yield {"event": "done", "data": {"steps": [asdict(s) for s in steps]}}
        return

    # 构建 prompt（注入分类摘要）
    market_data_str = json.dumps(market_data, ensure_ascii=False, indent=2)
    user_message = MARKET_REASONING_USER_TEMPLATE.format(
        question=question,
        market_data=market_data_str,
        evidence=evidence,
    )
    if evidence_analysis_result and evidence_analysis_result.get("summary"):
        user_message += f"\n\n证据分类摘要：{evidence_analysis_result['summary']}"

    # 流式 LLM
    steps.append(ThoughtStep(step="LLM归因分析", result="流式生成中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}
    for token in chat_completion_stream(MARKET_REASONING_SYSTEM_PROMPT, user_message, history=history):
        yield {"event": "token", "data": token}

    steps[-1].result = "生成完毕"
    yield {"event": "done", "data": {"steps": [asdict(s) for s in steps]}}


def _stream_knowledge(question: str, steps: list[ThoughtStep], history: list[dict] | None = None):
    """流式知识问答。"""
    steps.append(ThoughtStep(step="RAG向量检索", result="检索中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}

    context = ""
    rag_used = False
    try:
        context = get_relevant_context(question, top_k=4)
        if context:
            rag_used = True
            steps[-1].result = f"命中，上下文长度={len(context)}"
        else:
            steps[-1].result = "未命中相关文档"
    except Exception as e:
        steps[-1].result = f"检索失败: {e}"
    yield {"event": "thought", "data": asdict(steps[-1])}

    # Web search 兜底
    if not rag_used:
        steps.append(ThoughtStep(step="Web搜索兜底", result="搜索中..."))
        yield {"event": "thought", "data": asdict(steps[-1])}
        web_context = web_search(question)
        if web_context:
            steps[-1].result = f"搜索命中，长度={len(web_context)}"
            context = web_context
            rag_used = True
        else:
            steps[-1].result = "未配置搜索 API 或搜索无结果"
        yield {"event": "thought", "data": asdict(steps[-1])}

    yield {"event": "meta", "data": {"intent": "knowledge_rag", "ticker": None, "rag_used": rag_used}}

    if rag_used:
        system_prompt = RAG_SYSTEM_PROMPT
        user_message = RAG_USER_TEMPLATE.format(question=question, context=context)
    else:
        system_prompt = (
            RAG_SYSTEM_PROMPT
            + "\n\n注意：知识库中未检索到相关资料，请基于你的通用金融知识回答，"
            '并在回答末尾注明"本回答基于模型通用知识，未引用知识库资料"。'
        )
        user_message = RAG_USER_TEMPLATE.format(question=question, context="（未检索到相关参考资料）")

    steps.append(ThoughtStep(step="LLM生成回答", result="流式生成中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}
    for token in chat_completion_stream(system_prompt, user_message, history=history):
        yield {"event": "token", "data": token}

    steps[-1].result = "生成完毕"
    yield {"event": "done", "data": {"steps": [asdict(s) for s in steps]}}


def _stream_general(question: str, steps: list[ThoughtStep], history: list[dict] | None = None):
    """流式通用问答。"""
    yield {"event": "meta", "data": {"intent": "general", "ticker": None, "rag_used": None}}

    system = "你是一个友好的金融助手。简洁回答用户问题，使用用户的语言。回答要分点清晰，要点之间空一行，关键内容加粗。"
    steps.append(ThoughtStep(step="LLM生成回答", result="流式生成中..."))
    yield {"event": "thought", "data": asdict(steps[-1])}
    for token in chat_completion_stream(system, question, history=history):
        yield {"event": "token", "data": token}

    steps[-1].result = "生成完毕"
    yield {"event": "done", "data": {"steps": [asdict(s) for s in steps]}}
