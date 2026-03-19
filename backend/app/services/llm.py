"""LLM 调用封装 — 基于 OpenAI API"""

import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def get_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def chat_completion(system_prompt: str, user_message: str) -> str:
    """调用 LLM 获取回答。"""
    client = _get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    return response.choices[0].message.content or ""


def classify_intent(question: str) -> str:
    """使用 LLM 判断用户意图：market / knowledge / general。"""
    from app.prompts.templates import INTENT_CLASSIFICATION_PROMPT

    prompt = INTENT_CLASSIFICATION_PROMPT.format(question=question)
    client = _get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10,
    )
    result = (response.choices[0].message.content or "").strip().lower()

    if "market" in result:
        return "market"
    if "knowledge" in result:
        return "knowledge"
    return "general"
