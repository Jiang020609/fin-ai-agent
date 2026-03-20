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
