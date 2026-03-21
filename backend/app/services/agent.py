"""Agent 路由 — 根据用户意图分发到 Query Orchestrator 执行。

架构（三阶段）：
1. Query Understanding — Session 消解 + 查询改写 + Ticker 提取 + 意图分类
2. Plan Generation — orchestrator.build_plan() 生成步骤列表
3. Step Execution — orchestrator.execute_plan() / execute_plan_stream()

保留：公共 API、Session 消解、意图分类、Ticker 提取、审计日志。
handler/streaming 逻辑已迁移至 steps.py，由 orchestrator 统一调度。
"""

import time
import logging
import functools
from dataclasses import asdict
from typing import Callable

import re as _re

from app.services.llm import chat_completion, classify_intent_with_tools
from app.services.session import SessionState
from app.services.orchestrator import StepContext, build_plan, execute_plan, execute_plan_stream
# 导入 steps 模块以触发 @register_step 注册
import app.services.steps as _steps  # noqa: F401
from app.services.steps import ThoughtStep, build_no_data_message
from app.prompts.templates import QUERY_TRANSFORM_PROMPT
from app.utils.ticker_map import resolve_ticker, resolve_tickers_multi

logger = logging.getLogger(__name__)


# ========== 快速意图分类（正则，免 LLM 调用） ==========
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

    优先级：compare > knowledge > reasoning > market
    knowledge 必须在 market 之前，否则"什么是市盈率"会因含"市"被误分类为 market。
    """
    # 1. compare 最高优先级（需 2+ ticker）
    for p in _COMPARE_PATTERNS:
        if p.search(question):
            tickers = resolve_tickers_multi(question)
            if len(tickers) >= 2:
                return "compare"
            break
    # 2. knowledge 优先于 market（"什么是市盈率"含"市"字但应走知识路由）
    for p in _KNOWLEDGE_PATTERNS:
        if p.search(question):
            return "knowledge_rag"
    # 3. reasoning
    for p in _REASONING_PATTERNS:
        if p.search(question):
            return "market_reasoning"
    # 4. market_data
    for p in _MARKET_PATTERNS:
        if p.search(question):
            return "market_data"
    return None


def _normalize_intent(raw: str) -> str:
    """将 LLM 返回的意图标签归一化到 5 类。"""
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


# ========== 审计日志装饰器 ==========
def audit_log(func: Callable) -> Callable:
    """生产级审计日志装饰器。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        question = args[0] if args else kwargs.get("question", "")
        logger.info("[AUDIT] >>> Request | question=%s", question[:80])
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.time() - start) * 1000
            logger.info(
                "[AUDIT] <<< Response | intent=%s | ticker=%s | rag_used=%s | steps=%d | time=%.0fms",
                result.get("intent", "?"),
                result.get("ticker", "-"),
                result.get("rag_used", "-"),
                len(result.get("steps", [])),
                elapsed_ms,
            )
            if elapsed_ms > 10000:
                logger.warning("[AUDIT] SLOW REQUEST | question=%s | time=%.0fms", question[:50], elapsed_ms)
            return result
        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            logger.error("[AUDIT] !!! Error | question=%s | error=%s | time=%.0fms", question[:50], str(e), elapsed_ms)
            raise
    return wrapper


# ========== Session 指代消解 ==========
def _resolve_with_session(question: str, session: SessionState | None) -> tuple[str, list[str] | None, str | None]:
    """利用 session 上下文消解指代。返回 (resolved_question, session_tickers, inherited_intent)。"""
    if session is None:
        return question, None, None

    resolved_question = question
    session_tickers = None
    inherited_intent = None

    pronoun_pattern = _re.compile(r"^(它|这只|这个|该股|这家|那个)(为什么|怎么|最近|现在)", _re.IGNORECASE)
    if pronoun_pattern.search(question) and session.current_assets:
        session_tickers = session.current_assets

    continuation_pattern = _re.compile(r"^那.{1,6}呢[？?]?$")
    if continuation_pattern.search(question) and session.last_intent:
        inherited_intent = session.last_intent

    recompare_pattern = _re.compile(r"(再比|再对比|继续比|再比较)")
    if recompare_pattern.search(question) and session.last_intent == "compare":
        inherited_intent = "compare"
        if session.current_assets:
            session_tickers = session.current_assets

    return resolved_question, session_tickers, inherited_intent


# ========== 上下文 Ticker 回溯 ==========
def _resolve_ticker_with_context(question: str, history: list[dict] | None) -> tuple[str | None, str]:
    """从当前问题 + 历史对话中识别 Ticker。返回 (ticker, source)。"""
    ticker = resolve_ticker(question)
    if ticker:
        return ticker, "current"

    follow_up_patterns = _re.compile(
        r"(它|这只|这个|该股|这家|那个|上面|刚才|之前|前面|继续|还有|另外|其|the stock|this|that|same)",
        _re.IGNORECASE,
    )
    if not history or not follow_up_patterns.search(question):
        if not history:
            return None, ""

    for msg in reversed(history):
        if msg.get("role") == "user":
            t = resolve_ticker(msg.get("content", ""))
            if t:
                return t, "history"

    return None, ""


# ========== 查询改写 ==========
_CLEAR_QUERY_PATTERN = _re.compile(
    r"(股价|对比|比较|什么是|涨跌|走势|市值|行情|vs|price|compare|what\s+is)",
    _re.IGNORECASE,
)
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
    """将模糊/指代性查询改写为明确查询。"""
    if _CLEAR_QUERY_PATTERN.search(question) and not _VAGUE_QUERY_PATTERN.search(question):
        return question
    if not _VAGUE_QUERY_PATTERN.search(question):
        return question

    history_text = ""
    if history:
        recent = history[-4:]
        history_text = "\n".join(f"{m.get('role', '?')}: {m.get('content', '')[:100]}" for m in recent)

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
            steps.append(ThoughtStep(step="查询改写", result=f"'{question}' → '{rewritten}'"))
            return rewritten
    except Exception as e:
        logger.warning("[QueryTransform] LLM call failed: %s", e)

    return question


# ========== 路由决策核心逻辑 ==========
def _route_intent(question: str, ticker: str | None, steps: list[ThoughtStep]) -> str:
    """确定最终意图路由。"""
    fast_intent = fast_classify_intent(question)
    if fast_intent:
        steps.append(ThoughtStep(step="快速意图分类", result=f"正则命中: {fast_intent}"))
        return fast_intent

    steps.append(ThoughtStep(step="Tool-based意图分类", result="分类中..."))
    intent, tool_args = classify_intent_with_tools(question)
    steps[-1].result = f"Tool 选择: {intent}"
    steps[-1].detail = {"tool_args": tool_args}

    if ticker and intent not in ("market_data", "market_reasoning"):
        steps.append(ThoughtStep(step="意图修正", result=f"有 Ticker({ticker})，从 {intent} 升级为 market_data"))
        intent = "market_data"

    return intent


# ========== 主入口（Sync） ==========
@audit_log
def process_question(question: str, history: list[dict] | None = None, session: SessionState | None = None) -> dict:
    """主入口：接收用户问题，路由到 orchestrator 执行。"""
    steps: list[ThoughtStep] = []

    # Phase 0: Session 消解 + 查询改写
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

    question = _transform_query(question, history, session, steps)

    # Phase 1: Ticker + Intent
    multi_tickers = resolve_tickers_multi(question)
    if session_tickers and len(multi_tickers) < 2:
        multi_tickers = session_tickers

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

    intent = inherited_intent or _route_intent(question, ticker, steps)

    # Compare 路由
    if intent == "compare" and len(multi_tickers) >= 2:
        steps.append(ThoughtStep(step="意图路由", result=f"compare（{len(multi_tickers)} 个资产）"))
        plan = build_plan("compare", multi_tickers)
        ctx = StepContext(question=question, history=history, plan=plan,
                          ticker=multi_tickers[0], tickers=multi_tickers, steps=steps)
        result = execute_plan(ctx)
        if session:
            session.update(assets=multi_tickers, intent="compare")
        return result

    # 有 Ticker 的路由
    if ticker:
        if intent == "market_reasoning":
            steps.append(ThoughtStep(step="意图路由", result="market_reasoning（原因分析）"))
        else:
            steps.append(ThoughtStep(step="意图路由", result="market_data（基于 Ticker 命中）"))
            intent = "market_data"

        plan = build_plan(intent, [ticker])
        ctx = StepContext(question=question, history=history, plan=plan,
                          ticker=ticker, tickers=[ticker], steps=steps)
        result = execute_plan(ctx)
        if session:
            session.update(assets=[ticker], intent=intent)
        return result

    # 无 ticker 时：如果首次路由结果是 market 类但没有 ticker，降级为 general
    # 不重新调用 _route_intent 避免重复 thought steps
    if intent in ("market_data", "market_reasoning"):
        pass  # 下面会尝试从 history 找 ticker，或要求用户补充
    elif intent == "compare":
        pass  # compare 但 ticker 不够，已在上面处理
    # knowledge_rag / general 直接走下面的分支

    if intent in ("market_data", "market_reasoning"):
        # 最后一次尝试：从历史中找 ticker
        if history:
            for msg in reversed(history):
                if msg.get("role") == "user":
                    t = resolve_ticker(msg.get("content", ""))
                    if t:
                        ticker = t
                        steps.append(ThoughtStep(step="上下文回溯", result=f"从历史中找到 {ticker}"))
                        plan = build_plan(intent, [ticker])
                        ctx = StepContext(question=question, history=history, plan=plan,
                                          ticker=ticker, tickers=[ticker], steps=steps)
                        result = execute_plan(ctx)
                        return result

        steps.append(ThoughtStep(step="意图路由", result=f"{intent}（但未识别 Ticker，要求用户补充）"))
        return {
            "text_response": (
                "抱歉，我无法从您的问题中识别出具体的股票或资产。"
                "请提供股票名称或代码（如：阿里巴巴、TSLA、特斯拉）。"
            ),
            "chart_data": None,
            "intent": intent,
            "steps": [asdict(s) for s in steps],
        }

    if intent == "knowledge_rag":
        steps.append(ThoughtStep(step="意图路由", result="knowledge_rag"))
        plan = build_plan("knowledge_rag", [])
        ctx = StepContext(question=question, history=history, plan=plan, steps=steps)
        result = execute_plan(ctx)
        return result

    # general
    steps.append(ThoughtStep(step="意图路由", result="general"))
    plan = build_plan("general", [])
    ctx = StepContext(question=question, history=history, plan=plan, steps=steps)
    result = execute_plan(ctx)
    return result


# ========== 流式主入口 ==========
def process_question_stream(question: str, history: list[dict] | None = None, session: SessionState | None = None):
    """流式生成器：yield SSE 事件字典。事件类型：thought / token / chart / meta / done / error。"""
    steps: list[ThoughtStep] = []

    try:
        # Phase 0: Session 消解 + 查询改写
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

        question = _transform_query(question, history, session, steps)
        if steps and steps[-1].step == "查询改写":
            yield {"event": "thought", "data": asdict(steps[-1])}

        # Phase 1: Ticker + Intent
        multi_tickers = resolve_tickers_multi(question)
        if session_tickers and len(multi_tickers) < 2:
            multi_tickers = session_tickers

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

        intent = inherited_intent or _route_intent(question, ticker, steps)

        # Compare 路由
        if intent == "compare" and len(multi_tickers) >= 2:
            steps.append(ThoughtStep(step="意图路由", result=f"compare（{len(multi_tickers)} 个资产）"))
            yield {"event": "thought", "data": asdict(steps[-1])}
            plan = build_plan("compare", multi_tickers)
            ctx = StepContext(question=question, history=history, plan=plan,
                              ticker=multi_tickers[0], tickers=multi_tickers, steps=steps)
            yield from execute_plan_stream(ctx)
            if session:
                session.update(assets=multi_tickers, intent="compare")
            return

        # 有 Ticker 的路由
        if ticker:
            for s in steps[-2:]:
                yield {"event": "thought", "data": asdict(s)}

            if intent == "market_reasoning":
                steps.append(ThoughtStep(step="意图路由", result="market_reasoning（原因分析）"))
            else:
                steps.append(ThoughtStep(step="意图路由", result="market_data（基于 Ticker 命中）"))
                intent = "market_data"
            yield {"event": "thought", "data": asdict(steps[-1])}

            plan = build_plan(intent, [ticker])
            ctx = StepContext(question=question, history=history, plan=plan,
                              ticker=ticker, tickers=[ticker], steps=steps)
            yield from execute_plan_stream(ctx)
            if session:
                session.update(assets=[ticker], intent=intent)
            return

        # 无 ticker 的路由 — intent 已在上面确定，不重复调用 _route_intent
        for s in steps[-2:]:
            yield {"event": "thought", "data": asdict(s)}

        if intent in ("market_data", "market_reasoning"):
            if history:
                for msg in reversed(history):
                    if msg.get("role") == "user":
                        t = resolve_ticker(msg.get("content", ""))
                        if t:
                            ticker = t
                            steps.append(ThoughtStep(step="上下文回溯", result=f"从历史中找到 {ticker}"))
                            yield {"event": "thought", "data": asdict(steps[-1])}
                            plan = build_plan(intent, [ticker])
                            ctx = StepContext(question=question, history=history, plan=plan,
                                              ticker=ticker, tickers=[ticker], steps=steps)
                            yield from execute_plan_stream(ctx)
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
        else:
            steps.append(ThoughtStep(step="意图路由", result="general"))
            yield {"event": "thought", "data": asdict(steps[-1])}

        plan = build_plan(intent, [])
        ctx = StepContext(question=question, history=history, plan=plan, steps=steps)
        yield from execute_plan_stream(ctx)

    except Exception as e:
        logger.error("[STREAM] Error: %s", e)
        yield {"event": "error", "data": {"message": str(e)}}
