"""从用户问题中提取日期/时间窗口

用于 market_reasoning 链路中精准定位事件时间，构造更好的搜索 query。
纯正则实现，不引入外部依赖。

示例：
  "阿里巴巴1月15日为什么大涨" → DateRef(date_str="1月15日", search_hint="01-15")
  "特斯拉上周为什么跌"        → DateRef(date_str="上周", search_hint="上周")
  "英伟达最近走势"            → DateRef(date_str="最近", search_hint="近期")
  "苹果2024年Q3财报"          → DateRef(date_str="2024年Q3", search_hint="2024 Q3")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class DateRef:
    """从问题中提取的日期引用。"""
    date_str: str       # 用户原文中的日期表达
    search_hint: str    # 用于搜索 query 的日期提示


# 具体日期：X月X日、X月X号、YYYY-MM-DD、YYYY/MM/DD、MM.DD
_DATE_SPECIFIC = re.compile(
    r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日号]?'  # 2024-01-15, 2024年1月15日
    r'|\d{1,2}月\d{1,2}[日号]'                     # 1月15日
    r'|\d{1,2}\.\d{1,2})'                          # 1.15
)

# 相对时间：上周、本周、昨天、前天、近几天、最近X天
_DATE_RELATIVE = re.compile(
    r'(昨天|前天|今天|上周|本周|这周|上个月|本月|这个月'
    r'|近\d+[天日周月]|最近\d+[天日周月]'
    r'|过去\d+[天日周月]|前\d+[天日周月]'
    r'|last\s+week|this\s+week|yesterday|today'
    r'|recent(?:ly)?|past\s+\d+\s+days?)',
    re.IGNORECASE,
)

# 季度/年份：2024Q3、2024年第三季度、Q3
_DATE_QUARTER = re.compile(
    r'(\d{4}\s*[年]?\s*Q[1-4]'
    r'|\d{4}\s*年?\s*第?[一二三四1-4]\s*季度?'
    r'|Q[1-4]\s*\d{4})',
    re.IGNORECASE,
)

# 模糊时间：最近、近期、近来
_DATE_VAGUE = re.compile(
    r'(最近|近期|近来|近日|recently|lately)',
    re.IGNORECASE,
)


def extract_date_ref(question: str) -> DateRef | None:
    """从用户问题中提取日期引用。

    优先级：具体日期 > 季度 > 相对时间 > 模糊时间
    未匹配到任何日期表达时返回 None。
    """
    # 1. 具体日期
    m = _DATE_SPECIFIC.search(question)
    if m:
        raw = m.group(1)
        # 转换为搜索友好的格式
        hint = raw
        # "1月15日" → "01-15"
        month_day = re.match(r'(\d{1,2})月(\d{1,2})', raw)
        if month_day:
            hint = f"{int(month_day.group(1)):02d}-{int(month_day.group(2)):02d}"
        return DateRef(date_str=raw, search_hint=hint)

    # 2. 季度
    m = _DATE_QUARTER.search(question)
    if m:
        raw = m.group(1)
        # 提取年份和季度号用于搜索
        hint = re.sub(r'[年第季度]', ' ', raw).strip()
        return DateRef(date_str=raw, search_hint=hint)

    # 3. 相对时间
    m = _DATE_RELATIVE.search(question)
    if m:
        raw = m.group(1)
        return DateRef(date_str=raw, search_hint=raw)

    # 4. 模糊时间
    m = _DATE_VAGUE.search(question)
    if m:
        return DateRef(date_str=m.group(1), search_hint="近期")

    return None


def build_search_query(
    asset_name: str,
    ticker: str,
    question: str,
    date_ref: DateRef | None = None,
    trend_direction: str | None = None,
) -> str:
    """构造优化后的搜索 query。

    策略：
    - 资产名 + ticker 确保搜到相关内容
    - 如果有具体日期，加入日期提示
    - 如果知道涨跌方向，加入方向关键词
    - 避免过长的 query（搜索引擎对长 query 效果差）
    """
    parts = [asset_name]
    if ticker != asset_name:
        parts.append(ticker)

    # 方向关键词
    if trend_direction:
        direction_map = {
            "uptrend": "大涨 上涨",
            "downtrend": "大跌 下跌",
            "sideways": "波动 震荡",
        }
        parts.append(direction_map.get(trend_direction, "股价"))
    else:
        parts.append("股价 涨跌")

    # 日期提示
    if date_ref:
        parts.append(date_ref.search_hint)

    parts.append("原因 新闻")

    return " ".join(parts)
