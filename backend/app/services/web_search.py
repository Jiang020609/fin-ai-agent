"""Web Search 兜底 — RAG 未命中时调用搜索 API 获取补充上下文

支持 Tavily / SerpAPI，通过 SEARCH_API_KEY 配置。
"""

import os
import logging

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SEARCH_API_KEY = os.getenv("SEARCH_API_KEY", "")
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "tavily")  # tavily | serpapi


def web_search(query: str, max_results: int = 3) -> str:
    """执行 web search，返回格式化的 RAG context 字符串。

    未配置 API key 或搜索失败时返回空字符串。
    """
    if not SEARCH_API_KEY:
        return ""

    try:
        if SEARCH_PROVIDER == "tavily":
            return _tavily_search(query, max_results)
        elif SEARCH_PROVIDER == "serpapi":
            return _serpapi_search(query, max_results)
        else:
            logger.warning("[WebSearch] Unknown provider: %s", SEARCH_PROVIDER)
            return ""
    except Exception as e:
        logger.warning("[WebSearch] Failed: %s", e)
        return ""


def _tavily_search(query: str, max_results: int) -> str:
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={
            "api_key": SEARCH_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return ""

    results = resp.json().get("results", [])
    return _format_results(results)


def _serpapi_search(query: str, max_results: int) -> str:
    resp = httpx.get(
        "https://serpapi.com/search.json",
        params={
            "api_key": SEARCH_API_KEY,
            "q": query,
            "num": max_results,
        },
        timeout=10,
    )
    if resp.status_code != 200:
        return ""

    organic = resp.json().get("organic_results", [])
    results = [{"title": r.get("title", ""), "content": r.get("snippet", ""), "url": r.get("link", "")} for r in organic[:max_results]]
    return _format_results(results)


def _format_results(results: list[dict]) -> str:
    """格式化搜索结果为 RAG context 格式。"""
    if not results:
        return ""

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        parts.append(f"[网络搜索结果 {i}] {title}\n来源: {url}\n{content}")

    return "\n\n".join(parts)
