"""Agent 路由 — 根据用户意图分发到行情服务或 RAG 知识库

架构加固：
- ThoughtStep: 每次请求记录完整的思考链（意图识别 → Tool 调用 → 结果摘要）
- audit_log: 装饰器，记录每次请求的 Input/Output/耗时，模拟生产级监控
- 数据不可用时，向 LLM 发送明确的"数据缺失"信号，防止幻觉
"""

import json
import time
import logging
import functools
from dataclasses import dataclass, field, asdict
from typing import Callable

from app.services.llm import chat_completion, classify_intent
from app.services.market import get_stock_summary
from app.services.rag import get_relevant_context
from app.prompts.templates import (
    MARKET_SYSTEM_PROMPT,
    MARKET_USER_TEMPLATE,
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    FACT_CHECK_PROMPT,
)
from app.utils.ticker_map import resolve_ticker

logger = logging.getLogger(__name__)


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


# ========== 行情处理 ==========
def handle_market_question(question: str, ticker: str, steps: list[ThoughtStep]) -> dict:
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

    # Step: 数据可用性判定 — 向 LLM 发送明确信号，防止幻觉
    if not data_available:
        steps.append(ThoughtStep(
            step="数据校验",
            result="数据不可用，向 LLM 发送缺失信号",
        ))
        market_data_str = json.dumps(
            {"error": f"无法获取 {ticker} 的行情数据，API 暂时不可用", "data_available": False},
            ensure_ascii=False,
        )
    else:
        steps.append(ThoughtStep(step="数据校验", result="数据校验通过"))
        market_data_str = json.dumps(market_data, ensure_ascii=False, indent=2)

    # Step: LLM 生成回答
    steps.append(ThoughtStep(step="LLM生成回答", result="调用中..."))
    user_message = MARKET_USER_TEMPLATE.format(
        question=question,
        market_data=market_data_str,
    )
    text_response = chat_completion(MARKET_SYSTEM_PROMPT, user_message)
    steps[-1].result = f"生成完毕，长度={len(text_response)}"

    # 提取图表数据
    chart_data = None
    if data_available:
        change_30d = market_data.get("change_30d", {})
        change_7d = market_data.get("change_7d", {})
        if "history" in change_30d:
            chart_data = change_30d["history"]
        elif "history" in change_7d:
            chart_data = change_7d["history"]

    return {
        "text_response": text_response,
        "chart_data": chart_data,
        "intent": "market",
        "ticker": ticker,
        "steps": [asdict(s) for s in steps],
    }


# ========== 事实核查 ==========
def _fact_check(text_response: str, context: str, steps: list[ThoughtStep]) -> tuple[bool, str]:
    """调用 LLM 检查生成内容中的关键数值是否与检索上下文一致。

    返回 (passed, issues)。
    仅在有 RAG context 时执行 — 无参考资料则无从核查。
    """
    steps.append(ThoughtStep(step="事实核查", result="校验中..."))

    try:
        check_prompt = FACT_CHECK_PROMPT.format(context=context, response=text_response)
        raw = chat_completion(
            "你是一个严格的事实核查助手，只输出 JSON。",
            check_prompt,
        )

        # 解析 JSON 结果
        import re
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            passed = result.get("passed", True)
            issues = result.get("issues", "")
        else:
            passed, issues = True, ""

        if passed:
            steps[-1].result = "通过"
        else:
            steps[-1].result = f"未通过: {issues}"

        return passed, issues

    except Exception as e:
        # 核查失败不应阻断主流程，降级放行
        logger.warning("Fact check error: %s", e)
        steps[-1].result = f"核查异常（降级放行）: {e}"
        return True, ""


# ========== 知识库处理 ==========
def handle_knowledge_question(question: str, steps: list[ThoughtStep]) -> dict:
    """处理知识类问题：RAG 检索 → LLM 回答 → 事实核查（有 RAG 命中时）。"""
    # Step: RAG 检索
    steps.append(ThoughtStep(step="RAG向量检索", result="检索中..."))
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
        logger.warning("RAG retrieval failed: %s", e)

    # Step: 构建 Prompt
    if rag_used:
        system_prompt = RAG_SYSTEM_PROMPT
        user_message = RAG_USER_TEMPLATE.format(question=question, context=context)
        steps.append(ThoughtStep(step="Prompt构建", result="使用知识库参考资料"))
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
    text_response = chat_completion(system_prompt, user_message)
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
            text_response = chat_completion(retry_system, user_message)
            steps[-1].result = f"重新生成完毕，长度={len(text_response)}"

    return {
        "text_response": text_response,
        "chart_data": None,
        "intent": "knowledge",
        "rag_used": rag_used,
        "steps": [asdict(s) for s in steps],
    }


# ========== 通用处理 ==========
def handle_general_question(question: str, steps: list[ThoughtStep]) -> dict:
    """处理通用问题。"""
    steps.append(ThoughtStep(step="LLM生成回答", result="通用问答"))
    system = "你是一个友好的金融助手。简洁回答用户问题，使用用户的语言。"
    text_response = chat_completion(system, question)
    steps[-1].result = f"生成完毕，长度={len(text_response)}"

    return {
        "text_response": text_response,
        "chart_data": None,
        "intent": "general",
        "steps": [asdict(s) for s in steps],
    }


# ========== 主入口 ==========
@audit_log
def process_question(question: str) -> dict:
    """主入口：接收用户问题，执行分步思考链，路由到对应处理逻辑。

    思考链（steps）完整记录了 Agent 的决策过程：
    1. Ticker 识别 → 2. 意图分类 → 3. Tool 调用 → 4. 数据校验 → 5. LLM 生成
    每一步都有 step/result/detail，便于调试和审计。
    """
    steps: list[ThoughtStep] = []

    # Step 1: Ticker 识别
    ticker = resolve_ticker(question)
    steps.append(ThoughtStep(
        step="Ticker识别",
        result=f"匹配到 {ticker}" if ticker else "未匹配到股票代码",
        detail={"ticker": ticker},
    ))

    # Step 2: 路由决策
    if ticker:
        steps.append(ThoughtStep(step="意图路由", result="market（基于 Ticker 命中）"))
        return handle_market_question(question, ticker, steps)

    # Step 3: LLM 意图分类
    steps.append(ThoughtStep(step="LLM意图分类", result="分类中..."))
    intent = classify_intent(question)
    steps[-1].result = f"分类结果: {intent}"

    if intent == "market":
        steps.append(ThoughtStep(
            step="意图路由",
            result="market（但未识别 Ticker，要求用户补充）",
        ))
        return {
            "text_response": (
                "抱歉，我无法从您的问题中识别出具体的股票或资产。"
                "请提供股票名称或代码（如：阿里巴巴、TSLA、特斯拉）。"
            ),
            "chart_data": None,
            "intent": "market",
            "steps": [asdict(s) for s in steps],
        }
    elif intent == "knowledge":
        steps.append(ThoughtStep(step="意图路由", result="knowledge"))
        return handle_knowledge_question(question, steps)
    else:
        steps.append(ThoughtStep(step="意图路由", result="general"))
        return handle_general_question(question, steps)
