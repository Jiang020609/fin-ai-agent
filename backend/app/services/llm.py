"""LLM 调用封装 — 基于 OpenAI API"""

import os
from typing import Generator

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    return _client


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _build_messages(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
) -> list[dict]:
    """构建 messages 数组，支持多轮对话历史。"""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        msgs.extend(history)
    msgs.append({"role": "user", "content": user_message})
    return msgs


def chat_completion(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
) -> str:
    """调用 LLM 获取回答。"""
    client = _get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=_build_messages(system_prompt, user_message, history),
        temperature=0.3,
        max_tokens=2000,
    )
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""


def chat_completion_stream(
    system_prompt: str,
    user_message: str,
    history: list[dict] | None = None,
) -> Generator[str, None, None]:
    """流式调用 LLM，逐 chunk yield 文本片段。"""
    client = _get_client()
    stream = client.chat.completions.create(
        model=get_model(),
        messages=_build_messages(system_prompt, user_message, history),
        temperature=0.3,
        max_tokens=2000,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "query_market_data",
            "description": "查询股票/资产的实时价格、涨跌幅、走势等行情数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "股票代码或名称"},
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要查询的指标，如 price, change_7d, change_30d, volume",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_market_movement",
            "description": "分析股票/资产价格涨跌的原因，需要行情数据和新闻证据",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "股票代码或名称"},
                    "time_range": {"type": "string", "description": "关注的时间范围，如 today, this_week, specific date"},
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_assets",
            "description": "对比多个股票/资产的表现、指标差异",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的股票代码或名称列表（至少2个）",
                    },
                },
                "required": ["tickers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_financial_knowledge",
            "description": "查询金融概念、术语定义、财务指标解释等知识性问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "要查询的金融概念或术语"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_chat",
            "description": "处理与金融无关的闲聊或无法归类的问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "用户消息"},
                },
                "required": ["message"],
            },
        },
    },
]

_TOOL_TO_INTENT = {
    "query_market_data": "market_data",
    "analyze_market_movement": "market_reasoning",
    "compare_assets": "compare",
    "query_financial_knowledge": "knowledge_rag",
    "general_chat": "general",
}


def classify_intent_with_tools(question: str) -> tuple[str, dict]:
    """使用 OpenAI Function Calling (tools) 进行意图分类。

    返回 (intent, tool_args)：
    - intent: 归一化后的意图字符串
    - tool_args: LLM 解析出的工具参数（含 ticker 等信息）
    """
    import json as _json

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=get_model(),
            messages=[
                {"role": "system", "content": "根据用户问题选择最合适的工具。"},
                {"role": "user", "content": question},
            ],
            tools=TOOL_SCHEMAS,
            tool_choice="required",
            temperature=0,
            max_tokens=200,
        )
        msg = response.choices[0].message
        if msg.tool_calls:
            tool_call = msg.tool_calls[0]
            fn_name = tool_call.function.name
            try:
                tool_args = _json.loads(tool_call.function.arguments)
            except (_json.JSONDecodeError, TypeError):
                tool_args = {}
            intent = _TOOL_TO_INTENT.get(fn_name, "general")
            return intent, tool_args
    except Exception:
        pass

    # Fallback to text-based classification
    fallback = classify_intent(question)
    return fallback, {}


def classify_intent(question: str) -> str:
    """使用 LLM 判断用户意图：market_data / market_reasoning / knowledge / general。

    返回原始分类结果，由 agent.py 的 _normalize_intent 做最终归一化。
    """
    from app.prompts.templates import INTENT_CLASSIFICATION_PROMPT

    prompt = INTENT_CLASSIFICATION_PROMPT.format(question=question)
    client = _get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    if not response.choices:
        return "general"
    result = (response.choices[0].message.content or "").strip().lower()
    return result
