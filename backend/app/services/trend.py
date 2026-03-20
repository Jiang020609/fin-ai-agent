"""趋势分析规则模块 — 纯规则计算，不依赖 LLM

基于以下指标综合判断趋势：
1. 区间涨跌幅（change_pct）
2. 区间振幅（amplitude）= (high - low) / low * 100
3. 简单线性斜率（slope）— 收盘价序列的最小二乘斜率方向

分类输出：
- uptrend   / 上涨
- downtrend / 下跌
- sideways  / 震荡

设计原则：
- 规则透明、可解释（面试时可以直接讲清楚阈值逻辑）
- 不依赖外部服务
- 返回结构化 TrendResult，包含分类标签和判断依据
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ========== 阈值配置 ==========
# 涨跌幅阈值（%）
CHANGE_PCT_UP_THRESHOLD = 2.0     # 涨幅超过此值视为上涨
CHANGE_PCT_DOWN_THRESHOLD = -2.0  # 跌幅超过此值视为下跌

# 振幅阈值（%）：高振幅 + 低涨跌幅 = 震荡
AMPLITUDE_HIGH_THRESHOLD = 8.0    # 振幅超过此值视为高波动

# 斜率阈值（归一化后）：用于辅助判断
SLOPE_THRESHOLD = 0.001  # 斜率绝对值低于此视为无方向


# ========== 输出数据结构 ==========
@dataclass
class TrendResult:
    """趋势分析结果。"""
    label: str           # uptrend / downtrend / sideways
    label_cn: str        # 上涨 / 下跌 / 震荡
    confidence: str      # high / medium / low
    rationale: str       # 人类可读的判断依据

    # 原始指标
    change_pct: float | None = None
    amplitude: float | None = None
    slope_direction: str | None = None  # positive / negative / flat

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "label_cn": self.label_cn,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "change_pct": self.change_pct,
            "amplitude": self.amplitude,
            "slope_direction": self.slope_direction,
        }


# ========== 指标计算 ==========
def _calc_amplitude(high: float | None, low: float | None) -> float | None:
    """计算区间振幅 = (high - low) / low * 100。"""
    if high is None or low is None or low <= 0:
        return None
    return round((high - low) / low * 100, 2)


def _calc_slope_direction(prices: list[float]) -> str:
    """用最小二乘法计算收盘价序列的线性斜率方向。

    返回 'positive' / 'negative' / 'flat'。
    只关心方向，不关心具体值。
    """
    n = len(prices)
    if n < 3:
        return "flat"

    # 归一化：除以首个价格，消除量纲
    p0 = prices[0]
    if p0 <= 0:
        return "flat"
    normalized = [p / p0 for p in prices]

    # 最小二乘斜率: slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
    sum_x = sum(range(n))
    sum_y = sum(normalized)
    sum_xy = sum(i * y for i, y in enumerate(normalized))
    sum_x2 = sum(i * i for i in range(n))

    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return "flat"

    slope = (n * sum_xy - sum_x * sum_y) / denominator

    if slope > SLOPE_THRESHOLD:
        return "positive"
    elif slope < -SLOPE_THRESHOLD:
        return "negative"
    return "flat"


# ========== 核心分类函数 ==========
def classify_trend(
    change_pct: float | None = None,
    high: float | None = None,
    low: float | None = None,
    prices: list[float] | None = None,
    period_label: str = "7日",
) -> TrendResult:
    """基于规则的趋势分类。

    参数:
        change_pct: 区间涨跌幅（%），如 +3.5 或 -2.1
        high: 区间最高价
        low: 区间最低价
        prices: 收盘价序列（可选，用于斜率计算）
        period_label: 区间描述，用于生成 rationale

    返回:
        TrendResult

    分类规则（三级优先）：
    1. 涨跌幅 > +2%  → uptrend（高置信）
    2. 涨跌幅 < -2%  → downtrend（高置信）
    3. 涨跌幅在 ±2% 之间：
       a. 振幅 > 8%   → sideways（中置信，高波动震荡）
       b. 斜率 positive → uptrend（低置信，微涨）
       c. 斜率 negative → downtrend（低置信，微跌）
       d. 都无 → sideways（中置信）
    """
    amplitude = _calc_amplitude(high, low)
    slope_dir = _calc_slope_direction(prices) if prices else None

    reasons: list[str] = []

    # 无涨跌幅数据时返回 unknown
    if change_pct is None:
        return TrendResult(
            label="unknown",
            label_cn="未知",
            confidence="low",
            rationale=f"{period_label}涨跌幅数据缺失",
            change_pct=None,
            amplitude=amplitude,
            slope_direction=slope_dir,
        )

    # 规则 1 & 2：涨跌幅主导
    if change_pct > CHANGE_PCT_UP_THRESHOLD:
        reasons.append(f"{period_label}涨幅 {change_pct:+.2f}%")
        if slope_dir == "positive":
            reasons.append("价格斜率向上")
        return TrendResult(
            label="uptrend",
            label_cn="上涨",
            confidence="high",
            rationale="，".join(reasons),
            change_pct=change_pct,
            amplitude=amplitude,
            slope_direction=slope_dir,
        )

    if change_pct < CHANGE_PCT_DOWN_THRESHOLD:
        reasons.append(f"{period_label}跌幅 {change_pct:+.2f}%")
        if slope_dir == "negative":
            reasons.append("价格斜率向下")
        return TrendResult(
            label="downtrend",
            label_cn="下跌",
            confidence="high",
            rationale="，".join(reasons),
            change_pct=change_pct,
            amplitude=amplitude,
            slope_direction=slope_dir,
        )

    # 规则 3：涨跌幅在 ±2% 之间
    reasons.append(f"{period_label}涨跌幅 {change_pct:+.2f}%（窄幅）")

    # 3a: 高振幅震荡
    if amplitude is not None and amplitude > AMPLITUDE_HIGH_THRESHOLD:
        reasons.append(f"振幅 {amplitude:.1f}%（高波动）")
        return TrendResult(
            label="sideways",
            label_cn="震荡",
            confidence="medium",
            rationale="，".join(reasons),
            change_pct=change_pct,
            amplitude=amplitude,
            slope_direction=slope_dir,
        )

    # 3b/3c: 用斜率辅助判断微趋势
    if slope_dir == "positive":
        reasons.append("价格斜率向上")
        return TrendResult(
            label="uptrend",
            label_cn="上涨",
            confidence="low",
            rationale="，".join(reasons),
            change_pct=change_pct,
            amplitude=amplitude,
            slope_direction=slope_dir,
        )

    if slope_dir == "negative":
        reasons.append("价格斜率向下")
        return TrendResult(
            label="downtrend",
            label_cn="下跌",
            confidence="low",
            rationale="，".join(reasons),
            change_pct=change_pct,
            amplitude=amplitude,
            slope_direction=slope_dir,
        )

    # 3d: 默认震荡
    return TrendResult(
        label="sideways",
        label_cn="震荡",
        confidence="medium",
        rationale="，".join(reasons),
        change_pct=change_pct,
        amplitude=amplitude,
        slope_direction=slope_dir,
    )


def classify_trend_from_market_data(market_data: dict) -> TrendResult:
    """从 get_stock_summary 返回的 market_data 中提取指标并分类。

    优先用 7 日数据，不可用时降级到 30 日。
    """
    change_7d = market_data.get("change_7d", {})
    change_30d = market_data.get("change_30d", {})

    # 优先 7 日
    if change_7d.get("data_available"):
        prices = [p["close"] for p in change_7d.get("history", []) if "close" in p]
        return classify_trend(
            change_pct=change_7d.get("change_pct"),
            high=change_7d.get("high"),
            low=change_7d.get("low"),
            prices=prices or None,
            period_label="7日",
        )

    # 降级 30 日
    if change_30d.get("data_available"):
        prices = [p["close"] for p in change_30d.get("history", []) if "close" in p]
        return classify_trend(
            change_pct=change_30d.get("change_pct"),
            high=change_30d.get("high"),
            low=change_30d.get("low"),
            prices=prices or None,
            period_label="30日",
        )

    return TrendResult(
        label="unknown",
        label_cn="未知",
        confidence="low",
        rationale="无可用历史数据",
    )
