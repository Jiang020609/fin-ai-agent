"""新闻/证据分类模块 — 关键词规则分类，不用 LLM。"""

import re
from dataclasses import dataclass, field


# 分类关键词规则
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "earnings": [
        "财报", "营收", "净利润", "利润", "收入", "业绩", "季报", "年报", "盈利",
        "EPS", "earnings", "revenue", "profit", "quarterly", "annual report",
        "财务", "毛利", "营业收入", "净收入", "Q1", "Q2", "Q3", "Q4",
    ],
    "macro": [
        "美联储", "加息", "降息", "利率", "通胀", "CPI", "GDP", "就业", "非农",
        "Fed", "interest rate", "inflation", "monetary", "fiscal",
        "央行", "货币政策", "经济数据", "PMI", "失业率",
    ],
    "policy": [
        "监管", "政策", "法规", "制裁", "关税", "审查", "合规", "反垄断",
        "regulation", "policy", "sanction", "tariff", "antitrust", "ban",
        "禁令", "罚款", "调查", "出口管制",
    ],
    "industry": [
        "行业", "赛道", "产业链", "供应链", "竞争", "市场份额",
        "industry", "sector", "supply chain", "competition", "market share",
        "产能", "芯片", "AI", "人工智能", "新能源", "电动车",
    ],
    "company_news": [
        "收购", "并购", "上市", "退市", "回购", "分红", "拆股", "增发",
        "CEO", "管理层", "高管", "辞职", "任命", "合作", "战略",
        "acquisition", "merger", "IPO", "buyback", "dividend", "partnership",
        "产品", "发布", "新品", "升级",
    ],
}


@dataclass
class EvidenceAnalysis:
    """证据分析结果。"""
    main_drivers: list[str] = field(default_factory=list)
    secondary_drivers: list[str] = field(default_factory=list)
    evidence_strength: str = "low"  # high / medium / low
    classified_items: list[dict] = field(default_factory=list)
    summary: str = ""


def _classify_text(text: str) -> list[str]:
    """对单条文本做分类，返回匹配的类别列表。"""
    categories = []
    text_lower = text.lower()
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                categories.append(category)
                break
    return categories


def _judge_relevance(text: str, asset_name: str, ticker: str) -> str:
    """判断相关性：high / medium / low。"""
    # 标题一般在第一行
    lines = text.strip().split("\n")
    title = lines[0] if lines else ""
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    title_lower = title.lower()
    body_lower = body.lower()
    name_lower = asset_name.lower()
    ticker_lower = ticker.lower()

    # 标题包含公司名或 ticker = high
    if name_lower in title_lower or ticker_lower in title_lower:
        return "high"
    # 正文包含 = medium
    if name_lower in body_lower or ticker_lower in body_lower:
        return "medium"
    return "low"


def classify_evidence(raw_text: str, asset_name: str, ticker: str) -> EvidenceAnalysis:
    """解析搜索结果并分类。

    Args:
        raw_text: web_search 返回的格式化文本（包含 [网络搜索结果 N] 块）
        asset_name: 资产名称（如"阿里巴巴"）
        ticker: 股票代码（如"BABA"）

    Returns:
        EvidenceAnalysis: 分类结果
    """
    # 解析搜索结果块
    blocks = re.split(r'\[网络搜索结果\s*\d+\]', raw_text)
    blocks = [b.strip() for b in blocks if b.strip()]

    if not blocks:
        return EvidenceAnalysis(
            evidence_strength="low",
            summary="未检索到相关证据",
        )

    classified_items = []
    category_counts: dict[str, int] = {}
    relevant_count = 0

    for block in blocks:
        categories = _classify_text(block)
        relevance = _judge_relevance(block, asset_name, ticker)

        item = {
            "text_preview": block[:100],
            "categories": categories,
            "relevance": relevance,
        }
        classified_items.append(item)

        if relevance in ("high", "medium"):
            relevant_count += 1
            for cat in categories:
                category_counts[cat] = category_counts.get(cat, 0) + 1

    # 按频率排序
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    main_drivers = [cat for cat, _ in sorted_categories[:2]]
    secondary_drivers = [cat for cat, _ in sorted_categories[2:4]]

    # 证据强度
    if relevant_count >= 3:
        strength = "high"
    elif relevant_count >= 1:
        strength = "medium"
    else:
        strength = "low"

    # 生成摘要
    summary_parts = []
    if main_drivers:
        driver_names = {"earnings": "财报/业绩", "macro": "宏观经济", "policy": "政策/监管",
                        "industry": "行业动态", "company_news": "公司事件"}
        main_names = [driver_names.get(d, d) for d in main_drivers]
        summary_parts.append(f"主要驱动因素: {', '.join(main_names)}")
    if secondary_drivers:
        driver_names = {"earnings": "财报/业绩", "macro": "宏观经济", "policy": "政策/监管",
                        "industry": "行业动态", "company_news": "公司事件"}
        sec_names = [driver_names.get(d, d) for d in secondary_drivers]
        summary_parts.append(f"次要因素: {', '.join(sec_names)}")
    summary_parts.append(f"证据强度: {strength} (相关证据 {relevant_count} 条)")

    return EvidenceAnalysis(
        main_drivers=main_drivers,
        secondary_drivers=secondary_drivers,
        evidence_strength=strength,
        classified_items=classified_items,
        summary="; ".join(summary_parts),
    )
