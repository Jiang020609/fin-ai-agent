"""行情数据服务 — 基于 yfinance 获取股票价格与涨跌幅

生产级加固：
- MarketDataCache: 热数据缓存层，实时价格 60s TTL，历史数据 1h TTL
- retry_with_backoff: 指数退避重试，应对 yfinance 瞬时连接失败 / 429 限流
- _validate_price: 数据清洗层，拦截 NaN / Inf / 负价格等异常值
- 所有对外返回的 dict 均包含 data_available 字段，供下游 LLM 判断是否有可靠数据
"""

import logging
import math
import time
import functools
import threading
from datetime import datetime
from typing import TypeVar, Callable

import yfinance as yf

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ========== 缓存管理层 ==========
class MarketDataCache:
    """线程安全的行情数据缓存。

    设计思路：
    - 实时价格缓存 60s：价格每秒都在变，但对问答场景来说分钟级新鲜度足够，
      且能大幅减少对 Yahoo Finance 的请求量，避免 429 限流。
    - 历史趋势缓存 1h：7d/30d 趋势变化缓慢，1 小时内多次相同查询复用数据。
    - 日志记录 Hit/Miss，用于监控缓存命中率，指导 TTL 调优。
    """

    # TTL 配置（秒）
    PRICE_TTL = 60       # 实时价格：60s
    HISTORY_TTL = 3600   # 历史趋势：1h

    def __init__(self):
        self._store: dict[str, tuple[float, dict]] = {}  # key → (expire_at, data)
        self._lock = threading.Lock()

    def _make_key(self, prefix: str, ticker: str, *args) -> str:
        return f"{prefix}:{ticker}:{':'.join(str(a) for a in args)}"

    def get(self, key: str) -> dict | None:
        """获取缓存，返回 None 表示 miss 或已过期。"""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                logger.info("[Cache MISS] %s", key)
                return None
            expire_at, data = entry
            if time.time() > expire_at:
                del self._store[key]
                logger.info("[Cache EXPIRED] %s", key)
                return None
            ttl_remaining = expire_at - time.time()
            logger.info("[Cache HIT] %s (TTL remaining: %.0fs)", key, ttl_remaining)
            return data

    def set(self, key: str, data: dict, ttl: float):
        """写入缓存。"""
        with self._lock:
            self._store[key] = (time.time() + ttl, data)

    def invalidate(self, ticker: str):
        """手动失效某 ticker 的全部缓存（用于管理接口）。"""
        with self._lock:
            keys_to_delete = [k for k in self._store if f":{ticker}:" in k]
            for k in keys_to_delete:
                del self._store[k]
            if keys_to_delete:
                logger.info("[Cache INVALIDATE] %s (%d keys)", ticker, len(keys_to_delete))

    @property
    def stats(self) -> dict:
        """缓存统计（调试用）。"""
        with self._lock:
            now = time.time()
            total = len(self._store)
            alive = sum(1 for _, (exp, _) in self._store.items() if exp > now)
            return {"total_keys": total, "alive_keys": alive}


# 全局单例
_cache = MarketDataCache()


# ---------- 重试装饰器 ----------
def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    """指数退避重试装饰器。

    生产级可靠性：网络 API 调用天然不可靠，重试是最基本的容错手段。
    使用指数退避避免在限流场景下加剧问题。
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(
                            "[Retry %d/%d] %s failed: %s — retrying in %.1fs",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            e,
                            delay,
                        )
                        time.sleep(delay)
            logger.error(
                "[Retry exhausted] %s failed after %d attempts",
                func.__name__,
                max_retries,
            )
            raise last_exception  # type: ignore
        return wrapper
    return decorator


# ---------- 数据清洗 ----------
def _validate_price(value: float | None, field_name: str = "price") -> float | None:
    """验证价格值的合法性。

    拦截 NaN / Inf / 负数等异常值，防止脏数据进入 LLM prompt 产生幻觉。
    返回 None 表示"数据不可用"，下游会收到明确信号。
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s value: %s (not numeric)", field_name, value)
        return None
    if math.isnan(v) or math.isinf(v):
        logger.warning("Invalid %s value: %s (NaN/Inf)", field_name, v)
        return None
    if v < 0:
        logger.warning("Invalid %s value: %s (negative)", field_name, v)
        return None
    return round(v, 2)


def _safe_round(value: float | None, decimals: int = 2) -> float | None:
    """安全四舍五入，处理 None / NaN。"""
    if value is None:
        return None
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, decimals)
    except (TypeError, ValueError):
        return None


# ---------- 行情接口 ----------
@retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(Exception,))
def get_current_price(ticker: str) -> dict:
    """获取股票最新价格信息。

    三级容错：fast_info → info → history 兜底。
    所有价格字段经过 _validate_price 清洗。
    """
    stock = yf.Ticker(ticker)
    result: dict = {
        "ticker": ticker,
        "name": ticker,
        "currency": "USD",
        "data_available": False,  # 明确标记数据可用性
    }

    # 层级 1: fast_info（轻量，不触发重量级 API 调用）
    try:
        fi = stock.fast_info
        price = _validate_price(fi.last_price, "last_price")
        if price is not None:
            result.update({
                "current_price": price,
                "previous_close": _validate_price(fi.previous_close, "previous_close"),
                "market_cap": int(fi.market_cap) if fi.market_cap else None,
                "data_available": True,
            })
    except Exception as e:
        logger.warning("fast_info failed for %s: %s", ticker, e)

    # 层级 2: info（可能被限流）
    try:
        info = stock.info
        result["name"] = info.get("shortName") or info.get("longName", ticker)
        result["currency"] = info.get("currency", result["currency"])
        result["pe_ratio"] = _safe_round(info.get("trailingPE"))
        result["fifty_two_week_high"] = _validate_price(info.get("fiftyTwoWeekHigh"), "52w_high")
        result["fifty_two_week_low"] = _validate_price(info.get("fiftyTwoWeekLow"), "52w_low")
        if not result.get("current_price"):
            price = _validate_price(
                info.get("currentPrice") or info.get("regularMarketPrice"),
                "current_price",
            )
            if price is not None:
                result["current_price"] = price
                result["data_available"] = True
    except Exception as e:
        logger.warning("info failed for %s (rate limited?): %s", ticker, e)

    # 层级 3: history 兜底
    if not result.get("current_price"):
        try:
            hist = stock.history(period="5d")
            if not hist.empty:
                price = _validate_price(float(hist["Close"].iloc[-1]), "close_fallback")
                if price is not None:
                    result["current_price"] = price
                    result["data_available"] = True
        except Exception as e:
            logger.warning("history fallback failed for %s: %s", ticker, e)

    return result


@retry_with_backoff(max_retries=3, base_delay=1.0, exceptions=(Exception,))
def get_history_and_change(ticker: str, days: int = 7) -> dict:
    """获取过去 N 天的历史价格并计算涨跌幅。

    数据清洗：跳过含 NaN 的行，确保所有输出值合法。
    """
    stock = yf.Ticker(ticker)

    period = f"{days}d" if days <= 30 else "3mo"
    try:
        hist = stock.history(period=period)
    except Exception as e:
        return {"ticker": ticker, "error": str(e), "data_available": False}

    if hist.empty:
        return {
            "ticker": ticker,
            "error": f"过去 {days} 天无可用数据",
            "data_available": False,
        }

    hist = hist.tail(days)

    # 数据清洗：丢弃 Close 为 NaN 的行
    hist = hist.dropna(subset=["Close"])
    if len(hist) < 2:
        return {
            "ticker": ticker,
            "error": f"有效数据不足（仅 {len(hist)} 条）",
            "data_available": False,
        }

    start_price = _validate_price(float(hist["Close"].iloc[0]), "start_price")
    end_price = _validate_price(float(hist["Close"].iloc[-1]), "end_price")

    if start_price is None or end_price is None or start_price == 0:
        return {
            "ticker": ticker,
            "error": "价格数据异常，无法计算涨跌幅",
            "data_available": False,
        }

    change = end_price - start_price
    change_pct = (change / start_price) * 100
    high = _validate_price(float(hist["High"].max()), "high")
    low = _validate_price(float(hist["Low"].min()), "low")

    if change_pct > 2:
        trend = "上涨"
    elif change_pct < -2:
        trend = "下跌"
    else:
        trend = "震荡"

    # 构建图表数据，跳过异常行
    history = []
    for idx, row in hist.iterrows():
        close = _validate_price(float(row["Close"]), "chart_close")
        if close is None:
            continue
        volume = int(row["Volume"]) if not math.isnan(row["Volume"]) else 0
        history.append({
            "date": idx.strftime("%Y-%m-%d"),
            "close": close,
            "volume": volume,
        })

    return {
        "ticker": ticker,
        "period_days": days,
        "start_price": start_price,
        "end_price": end_price,
        "change": _safe_round(change),
        "change_pct": _safe_round(change_pct),
        "high": high,
        "low": low,
        "trend": trend,
        "history": history,
        "data_available": True,
    }


def _cached_current_price(ticker: str) -> dict:
    """带缓存的实时价格查询（TTL 60s）。"""
    cache_key = _cache._make_key("price", ticker)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    data = get_current_price(ticker)
    if data.get("data_available"):
        _cache.set(cache_key, data, MarketDataCache.PRICE_TTL)
    return data


def _cached_history(ticker: str, days: int) -> dict:
    """带缓存的历史趋势查询（TTL 1h）。"""
    cache_key = _cache._make_key("history", ticker, days)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    data = get_history_and_change(ticker, days)
    if data.get("data_available"):
        _cache.set(cache_key, data, MarketDataCache.HISTORY_TTL)
    return data


def get_stock_summary(ticker: str) -> dict:
    """获取完整的股票摘要：当前价格 + 7日/30日涨跌幅。

    通过缓存层减少 yfinance 调用，汇总 data_available 标记。
    """
    price_info = _cached_current_price(ticker)
    change_7d = _cached_history(ticker, days=7)
    change_30d = _cached_history(ticker, days=30)

    any_available = (
        price_info.get("data_available", False)
        or change_7d.get("data_available", False)
        or change_30d.get("data_available", False)
    )

    return {
        "price": price_info,
        "change_7d": change_7d,
        "change_30d": change_30d,
        "data_available": any_available,
        "query_time": datetime.now().isoformat(),
        "cache_stats": _cache.stats,
    }
