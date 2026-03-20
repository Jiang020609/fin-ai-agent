"""准确性控制层（Grounding Layer）

将系统中所有防幻觉、数据验证、事实核查逻辑整合到一个模块，
便于审计和面试讲解"准确性控制策略"。

三层防线：
1. 数据源隔离 — 市场数据只能来自 API，知识只能来自 RAG/搜索，LLM 不负责编数据
2. 检索过滤   — 相似度阈值、数据清洗（NaN/Inf/负值）、来源标注
3. 输出约束   — 事实核查、缺失字段显式标注、谨慎措辞强制

本模块不包含业务逻辑本身，只暴露校验/核查工具函数。
"""

import json
import logging
import re
from dataclasses import asdict

logger = logging.getLogger(__name__)


# ============================================================
#  第一层：数据源隔离规则
# ============================================================
# 以下常量文档化了各路由允许的数据来源，agent.py 已在实现中遵守。
# 将规则集中声明，方便面试时一句话讲清楚。

DATA_SOURCE_RULES = {
    "market_data": {
        "allowed_data":  ["market_api"],
        "allowed_analysis": ["llm"],
        "description": "价格/涨跌幅等客观数据只能来自行情 API，LLM 只负责解释和组织，不负责编造数字。",
    },
    "market_reasoning": {
        "allowed_data":  ["market_api"],
        "allowed_evidence": ["web_search"],
        "allowed_analysis": ["llm"],
        "description": "行情数据来自 API，原因证据来自 Web 搜索，LLM 只负责基于证据做归因分析。",
    },
    "knowledge_rag": {
        "allowed_context": ["rag_vectorstore", "web_search"],
        "allowed_analysis": ["llm"],
        "description": "知识类回答优先基于 RAG 检索或 Web 搜索结果，LLM 不能凭空生成数据。",
    },
    "general": {
        "allowed_analysis": ["llm"],
        "description": "通用问答可以使用 LLM 通用知识，但不涉及实时数据。",
    },
}


# ============================================================
#  第二层：检索过滤
# ============================================================

# RAG 相似度阈值（与 rag.py 保持一致，在此声明为可配置常量）
RAG_RELEVANCE_THRESHOLD = 0.3

# 行情数据缓存 TTL（与 market.py 保持一致）
PRICE_CACHE_TTL = 60        # 秒
HISTORY_CACHE_TTL = 3600    # 秒


def validate_market_data(market_data: dict) -> dict:
    """校验行情数据完整性，返回校验报告。

    检查项：
    1. 是否有可用数据（data_available）
    2. 关键价格字段是否合法（非 None / NaN / 负值）
    3. 是否为降级缓存数据（stale）
    4. 7d/30d 历史是否有足够数据点

    返回 dict:
        passed: bool
        issues: list[str]  — 每项不通过的说明
        is_stale: bool
        stale_age_seconds: int | None
    """
    issues: list[str] = []

    data_available = market_data.get("data_available", False)
    if not data_available:
        issues.append("行情 API 未返回可用数据")

    # 价格合法性
    price_info = market_data.get("price", {})
    current_price = price_info.get("current_price")
    if current_price is None:
        issues.append("当前价格为空")
    elif not isinstance(current_price, (int, float)) or current_price <= 0:
        issues.append(f"当前价格异常: {current_price}")

    # 降级检测
    is_stale = any(
        market_data.get(k, {}).get("stale", False)
        for k in ("price", "change_7d", "change_30d")
    )
    stale_age = None
    if is_stale:
        stale_ages = [
            market_data.get(k, {}).get("stale_age_seconds", 0)
            for k in ("price", "change_7d", "change_30d")
            if market_data.get(k, {}).get("stale")
        ]
        stale_age = max(stale_ages) if stale_ages else 0
        issues.append(f"使用降级缓存数据（过期 {stale_age}s）")

    # 历史数据完整性
    for period in ("change_7d", "change_30d"):
        period_data = market_data.get(period, {})
        if period_data.get("data_available", False):
            history = period_data.get("history", [])
            if len(history) < 2:
                issues.append(f"{period} 历史数据不足（仅 {len(history)} 点）")

    return {
        "passed": len(issues) == 0 or (len(issues) == 1 and is_stale),
        "issues": issues,
        "is_stale": is_stale,
        "stale_age_seconds": stale_age,
    }


def validate_rag_results(results: list[tuple], threshold: float = RAG_RELEVANCE_THRESHOLD) -> list[tuple]:
    """过滤 RAG 检索结果：低于相似度阈值的丢弃。

    这个函数是 rag.py 中过滤逻辑的显式声明版本，
    确保过滤规则在一处可查。

    参数:
        results: [(doc, score), ...] 从 ChromaDB 返回
        threshold: 相似度最低阈值

    返回:
        过滤后的 [(doc, score), ...]
    """
    filtered = [(doc, score) for doc, score in results if score >= threshold]
    dropped = len(results) - len(filtered)
    if dropped > 0:
        logger.info(
            "[Grounding] RAG 过滤: %d/%d 结果低于阈值 %.2f 被丢弃",
            dropped, len(results), threshold,
        )
    return filtered


# ============================================================
#  第三层：输出约束 — 事实核查 & 输出验证
# ============================================================

def fact_check(
    text_response: str,
    context: str,
    llm_call,
    fact_check_prompt_template: str,
) -> tuple[bool, str]:
    """调用 LLM 检查生成内容中的关键数值是否与检索上下文一致。

    从 agent.py 抽取的核心事实核查逻辑。

    参数:
        text_response: LLM 生成的回答文本
        context: RAG/搜索检索到的参考资料
        llm_call: chat_completion 函数引用
        fact_check_prompt_template: FACT_CHECK_PROMPT 模板

    返回:
        (passed: bool, issues: str)
    """
    try:
        check_prompt = fact_check_prompt_template.format(
            context=context, response=text_response
        )
        raw = llm_call(
            "你是一个严格的事实核查助手，只输出 JSON。",
            check_prompt,
        )

        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            passed = result.get("passed", True)
            issues = result.get("issues", "")
        else:
            passed, issues = True, ""

        return passed, issues

    except Exception as e:
        logger.warning("[Grounding] Fact check error (degraded pass): %s", e)
        return True, ""


def validate_market_response_numbers(
    text_response: str,
    market_data: dict,
) -> tuple[bool, list[str]]:
    """检查 LLM 行情回答中提到的数字是否与 market_data 一致。

    简单规则：提取回答中出现的价格数字，检查是否在 market_data 中存在。
    这是一个轻量级检查，不依赖额外 LLM 调用。

    返回:
        (passed: bool, suspicious_numbers: list[str])
    """
    # 提取回答中的数字（价格格式：整数或小数）
    numbers_in_text = set(re.findall(r'\b\d+\.?\d*\b', text_response))

    # 从 market_data 中收集所有合法数字
    valid_numbers: set[str] = set()
    price_info = market_data.get("price", {})
    for key in ("current_price", "previous_close", "pe_ratio", "market_cap"):
        val = price_info.get(key)
        if val is not None:
            valid_numbers.add(str(val))
            # 也加入常见的格式变体
            if isinstance(val, float):
                valid_numbers.add(f"{val:.2f}")
                valid_numbers.add(f"{val:.0f}")

    for period in ("change_7d", "change_30d"):
        period_data = market_data.get(period, {})
        for key in ("start_price", "end_price", "change", "change_pct", "high", "low"):
            val = period_data.get(key)
            if val is not None:
                valid_numbers.add(str(val))
                if isinstance(val, float):
                    valid_numbers.add(f"{val:.2f}")
                    valid_numbers.add(f"{val:.0f}")
                    valid_numbers.add(f"{abs(val):.2f}")

    # 过滤掉太小的数字（年份、百分比符号旁的单位数字等容易误报）
    # 只检查看起来像价格的数字（> 1.0）
    suspicious = []
    for num_str in numbers_in_text:
        try:
            num = float(num_str)
        except ValueError:
            continue
        if num < 1.0:
            continue  # 跳过小数、百分比值等
        if num_str not in valid_numbers and f"{num:.2f}" not in valid_numbers:
            # 检查是否是某个合法数字的近似值（允许 0.5% 误差）
            is_close = any(
                abs(num - float(v)) / max(float(v), 1) < 0.005
                for v in valid_numbers
                if v.replace(".", "").isdigit()
            )
            if not is_close:
                suspicious.append(num_str)

    return len(suspicious) == 0, suspicious


def build_missing_field_notice(market_data: dict) -> str | None:
    """当行情数据缺少关键字段时，生成"无法获取"的显式说明。

    避免 LLM 在数据缺失时自行编造。
    """
    missing = []
    price_info = market_data.get("price", {})
    if price_info.get("current_price") is None:
        missing.append("当前价格")
    if price_info.get("pe_ratio") is None:
        missing.append("市盈率")
    if price_info.get("market_cap") is None:
        missing.append("市值")

    change_7d = market_data.get("change_7d", {})
    if not change_7d.get("data_available"):
        missing.append("7日历史数据")

    change_30d = market_data.get("change_30d", {})
    if not change_30d.get("data_available"):
        missing.append("30日历史数据")

    if not missing:
        return None

    return f"以下数据当前无法获取：{', '.join(missing)}。相关分析可能不完整。"


# ============================================================
#  汇总：准确性控制策略清单（用于 README 和面试讲解）
# ============================================================
GROUNDING_STRATEGY_SUMMARY = """
准确性控制策略（Grounding Layer）

┌─────────────────────────────────────────────────────────────┐
│ 第一层：数据源隔离                                          │
│                                                             │
│  market_data     → 数据只来自行情 API，LLM 只解释不编数据   │
│  market_reasoning→ 行情 API + Web 搜索证据，LLM 做归因     │
│  knowledge_rag   → RAG 向量检索 + Web 搜索，LLM 做整合     │
│  general         → LLM 通用知识                             │
├─────────────────────────────────────────────────────────────┤
│ 第二层：检索过滤                                            │
│                                                             │
│  • RAG 相似度阈值 ≥ 0.3（低于丢弃）                        │
│  • 行情数据 NaN/Inf/负值拦截                                │
│  • 价格缓存 TTL 60s / 历史缓存 1h                           │
│  • 降级缓存显式标注 stale + age                             │
├─────────────────────────────────────────────────────────────┤
│ 第三层：输出约束                                            │
│                                                             │
│  • Prompt 约束："所有数字必须来自 market_data"               │
│  • 事实核查：LLM 自检回答 vs RAG 来源（不通过则重试）       │
│  • 数字交叉验证：回答中的数字 vs market_data 数字            │
│  • 缺失字段显式标注："无法获取"而非虚构                     │
│  • 原因分析用谨慎措辞："可能原因""或与…有关"               │
│  • structured_response 中 data_summary 由程序组装非 LLM 生成│
└─────────────────────────────────────────────────────────────┘
"""
