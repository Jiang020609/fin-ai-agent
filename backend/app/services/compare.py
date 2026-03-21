"""多资产对比指标计算模块 — 纯函数，无 LLM 调用。"""

import math


def _daily_returns(history: list[dict]) -> list[float] | None:
    """从 history (daily closes) 提取日收益率序列，复用于多个指标。"""
    if not history or len(history) < 2:
        return None
    closes = [h.get("close") for h in history if h.get("close") is not None]
    if len(closes) < 2:
        return None
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] != 0:
            returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
    return returns if returns else None


def _calc_volatility(history: list[dict]) -> float | None:
    """计算年化波动率：日收益率标准差 × √252。

    history: [{"close": float}, ...]
    """
    returns = _daily_returns(history)
    if not returns:
        return None

    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    daily_vol = math.sqrt(variance)
    annualized = daily_vol * math.sqrt(252)
    return round(annualized, 4)


RISK_FREE_RATE = 0.04  # 年化无风险利率假设（约等于 10Y US Treasury）


def _calc_sharpe_ratio(history: list[dict], annualized_risk_free: float = RISK_FREE_RATE) -> float | None:
    """计算年化夏普比率。

    Sharpe = (mean_daily_return - daily_rf) / daily_std × √252
    """
    returns = _daily_returns(history)
    if not returns:
        return None

    daily_rf = annualized_risk_free / 252
    mean_r = sum(returns) / len(returns)
    variance = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    daily_std = math.sqrt(variance)

    if daily_std == 0:
        return None

    sharpe = (mean_r - daily_rf) / daily_std * math.sqrt(252)
    return round(sharpe, 4)


def _calc_max_drawdown(history: list[dict]) -> float | None:
    """计算最大回撤（负百分比，如 -0.0823 表示 -8.23%）。"""
    if not history or len(history) < 2:
        return None

    closes = [h.get("close") for h in history if h.get("close") is not None]
    if len(closes) < 2:
        return None

    peak = closes[0]
    max_dd = 0.0
    for close in closes[1:]:
        if close > peak:
            peak = close
        dd = (peak - close) / peak if peak != 0 else 0.0
        if dd > max_dd:
            max_dd = dd

    return round(-max_dd, 4) if max_dd > 0 else None


def _extract_metrics(ticker: str, summary: dict) -> dict:
    """从 get_stock_summary() 结果提取对比指标。"""
    price_info = summary.get("price", {})
    change_7d = summary.get("change_7d", {})
    change_30d = summary.get("change_30d", {})

    # 取历史数据算波动率
    history_30d = change_30d.get("history", [])
    history_7d = change_7d.get("history", [])

    vol_30d = _calc_volatility(history_30d)
    vol_7d = _calc_volatility(history_7d)

    return {
        "ticker": ticker,
        "name": price_info.get("name", ticker),
        "current_price": price_info.get("current_price"),
        "currency": price_info.get("currency", "USD"),
        "return_7d": change_7d.get("change_pct"),
        "return_30d": change_30d.get("change_pct"),
        "volatility_7d": vol_7d,
        "volatility_30d": vol_30d,
        "high_7d": change_7d.get("high"),
        "low_7d": change_7d.get("low"),
        "high_30d": change_30d.get("high"),
        "low_30d": change_30d.get("low"),
        "sharpe_30d": _calc_sharpe_ratio(history_30d),
        "max_drawdown_30d": _calc_max_drawdown(history_30d),
        "data_available": summary.get("data_available", False),
    }


def _determine_winners(assets: list[dict]) -> dict:
    """各维度的优胜者判断。"""
    winners = {}

    # 7日收益最高
    valid = [(a["ticker"], a["return_7d"]) for a in assets if a["return_7d"] is not None]
    if valid:
        best = max(valid, key=lambda x: x[1])
        winners["return_7d"] = {"ticker": best[0], "value": best[1]}

    # 30日收益最高
    valid = [(a["ticker"], a["return_30d"]) for a in assets if a["return_30d"] is not None]
    if valid:
        best = max(valid, key=lambda x: x[1])
        winners["return_30d"] = {"ticker": best[0], "value": best[1]}

    # 波动率最低（更稳定）
    valid = [(a["ticker"], a["volatility_30d"]) for a in assets if a["volatility_30d"] is not None]
    if valid:
        best = min(valid, key=lambda x: x[1])
        winners["volatility_lower"] = {"ticker": best[0], "value": best[1]}

    # 动量（7d return > 0 且最大者）
    valid = [(a["ticker"], a["return_7d"]) for a in assets if a["return_7d"] is not None and a["return_7d"] > 0]
    if valid:
        best = max(valid, key=lambda x: x[1])
        winners["momentum"] = {"ticker": best[0], "value": best[1]}

    # 夏普比率最高（风险调整后收益最优）
    valid = [(a["ticker"], a["sharpe_30d"]) for a in assets if a["sharpe_30d"] is not None]
    if valid:
        best = max(valid, key=lambda x: x[1])
        winners["sharpe_best"] = {"ticker": best[0], "value": best[1]}

    # 最大回撤最小（绝对值最小 = 下行风险最低）
    valid = [(a["ticker"], a["max_drawdown_30d"]) for a in assets if a["max_drawdown_30d"] is not None]
    if valid:
        best = max(valid, key=lambda x: x[1])  # 负值中最大 = 绝对值最小
        winners["max_drawdown_least"] = {"ticker": best[0], "value": best[1]}

    return winners


def compute_comparison(summaries: dict[str, dict]) -> dict:
    """从多个 get_stock_summary() 结果计算对比指标。

    Args:
        summaries: {ticker: get_stock_summary() 结果}

    Returns:
        {
            "assets": [...],    # 各资产指标
            "winners": {...},   # 各维度优胜者
            "tickers": [...]    # ticker 列表
        }
    """
    assets = []
    for ticker, summary in summaries.items():
        metrics = _extract_metrics(ticker, summary)
        assets.append(metrics)

    winners = _determine_winners(assets)

    return {
        "assets": assets,
        "winners": winners,
        "tickers": list(summaries.keys()),
        "assumptions": {
            "risk_free_rate": RISK_FREE_RATE,
            "risk_free_rate_note": f"夏普比率基于年化 {RISK_FREE_RATE*100:.0f}% 无风险利率计算",
            "max_drawdown_note": "最大回撤为负数，越接近 0 表示下行风险越小",
        },
    }
