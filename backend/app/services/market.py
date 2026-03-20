"""行情数据服务 — 多数据源级联 + 缓存 + 重试 + 数据清洗

数据源优先级（瀑布式降级）：
  1. yfinance（Yahoo Finance，免费，无 key）
  2. Finnhub（免费 tier，60 次/分钟，需 API key）
  3. Alpha Vantage（免费 tier，25 次/天，需 API key）
  4. 过期缓存（stale cache fallback）

任何一个源成功就立即返回，不再调用后续源。
"""

import logging
import math
import os
import time
import functools
import threading
from datetime import datetime
from typing import TypeVar, Callable

import httpx
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ========== 环境变量 ==========
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")


# ========== 缓存管理层 ==========
class MarketDataCache:
    PRICE_TTL = 60
    HISTORY_TTL = 3600

    def __init__(self):
        self._store: dict[str, tuple[float, dict]] = {}
        self._stale_store: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _make_key(self, prefix: str, ticker: str, *args) -> str:
        return f"{prefix}:{ticker}:{':'.join(str(a) for a in args)}"

    def get(self, key: str) -> dict | None:
        from app.services.metrics import metrics
        metrics.incr("cache_get")
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                logger.info("[Cache MISS] %s", key)
                return None
            expire_at, data = entry
            if time.time() > expire_at:
                self._stale_store[key] = (expire_at, data)
                del self._store[key]
                logger.info("[Cache EXPIRED] %s", key)
                return None
            ttl_remaining = expire_at - time.time()
            logger.info("[Cache HIT] %s (TTL remaining: %.0fs)", key, ttl_remaining)
            metrics.incr("cache_hit")
            return data

    def set(self, key: str, data: dict, ttl: float):
        with self._lock:
            self._store[key] = (time.time() + ttl, data)

    def get_stale(self, key: str) -> tuple[dict | None, float]:
        with self._lock:
            stale = self._stale_store.get(key)
            if stale is None:
                return None, 0
            expired_at, data = stale
            age = time.time() - expired_at
            logger.info("[Cache STALE] %s (expired %.0fs ago)", key, age)
            return data, age

    def invalidate(self, ticker: str):
        with self._lock:
            keys_to_delete = [k for k in self._store if f":{ticker}:" in k]
            for k in keys_to_delete:
                del self._store[k]

    @property
    def stats(self) -> dict:
        with self._lock:
            now = time.time()
            total = len(self._store)
            alive = sum(1 for _, (exp, _) in self._store.items() if exp > now)
            return {"total_keys": total, "alive_keys": alive}


_cache = MarketDataCache()


# ========== 重试装饰器 ==========
def retry_with_backoff(
    max_retries: int = 2,
    base_delay: float = 0.5,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable:
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
                            "[Retry %d/%d] %s: %s — %.1fs",
                            attempt + 1, max_retries, func.__name__, e, delay,
                        )
                        time.sleep(delay)
            raise last_exception  # type: ignore
        return wrapper
    return decorator


# ========== 数据清洗 ==========
def _validate_price(value: float | None, field_name: str = "price") -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v) or v < 0:
        logger.warning("Invalid %s: %s", field_name, v)
        return None
    return round(v, 2)


def _safe_round(value: float | None, decimals: int = 2) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        return None if (math.isnan(v) or math.isinf(v)) else round(v, decimals)
    except (TypeError, ValueError):
        return None


# ================================================================
#  数据源 1：yfinance（Yahoo Finance）
# ================================================================
@retry_with_backoff(max_retries=2, base_delay=0.5)
def _yfinance_current_price(ticker: str) -> dict | None:
    """yfinance 获取当前价格。返回 None 表示失败。"""
    stock = yf.Ticker(ticker)
    result: dict = {"ticker": ticker, "source": "yahoo"}

    # fast_info
    try:
        fi = stock.fast_info
        price = _validate_price(fi.last_price, "yf_last_price")
        if price is not None:
            result.update({
                "current_price": price,
                "previous_close": _validate_price(fi.previous_close),
                "market_cap": int(fi.market_cap) if fi.market_cap else None,
                "name": ticker,
                "currency": "USD",
                "data_available": True,
            })
            # 尝试拿公司名（可能限流，不阻断）
            try:
                info = stock.info
                result["name"] = info.get("shortName") or info.get("longName", ticker)
                result["currency"] = info.get("currency", "USD")
                result["pe_ratio"] = _safe_round(info.get("trailingPE"))
            except Exception:
                pass
            return result
    except Exception as e:
        logger.warning("[yfinance] fast_info failed for %s: %s", ticker, e)

    # history 兜底
    try:
        hist = stock.history(period="5d")
        if not hist.empty:
            price = _validate_price(float(hist["Close"].iloc[-1]))
            if price is not None:
                return {
                    "ticker": ticker, "source": "yahoo",
                    "current_price": price, "name": ticker,
                    "currency": "USD", "data_available": True,
                }
    except Exception as e:
        logger.warning("[yfinance] history failed for %s: %s", ticker, e)

    return None


@retry_with_backoff(max_retries=2, base_delay=0.5)
def _yfinance_history(ticker: str, days: int) -> dict | None:
    """yfinance 获取历史数据。返回 None 表示失败。"""
    stock = yf.Ticker(ticker)
    period = f"{days}d" if days <= 30 else "3mo"
    try:
        hist = stock.history(period=period)
    except Exception:
        return None

    if hist.empty:
        return None

    hist = hist.tail(days).dropna(subset=["Close"])
    if len(hist) < 2:
        return None

    return _build_history_result(ticker, hist, days, source="yahoo")


# ================================================================
#  数据源 2：Finnhub（免费 60 次/分钟）
# ================================================================
@retry_with_backoff(max_retries=2, base_delay=0.3)
def _finnhub_current_price(ticker: str) -> dict | None:
    """Finnhub REST API 获取实时报价。"""
    if not FINNHUB_API_KEY:
        return None
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
        resp = httpx.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        data = resp.json()
        price = _validate_price(data.get("c"), "finnhub_price")  # c = current
        if price is None or price == 0:
            return None
        prev_close = _validate_price(data.get("pc"))
        logger.info("[Finnhub] Got price for %s: %s", ticker, price)
        return {
            "ticker": ticker, "source": "finnhub",
            "current_price": price,
            "previous_close": prev_close,
            "name": ticker, "currency": "USD",
            "data_available": True,
        }
    except Exception as e:
        logger.warning("[Finnhub] Failed for %s: %s", ticker, e)
        return None


# ================================================================
#  数据源 3：Alpha Vantage（免费 25 次/天）
# ================================================================
@retry_with_backoff(max_retries=2, base_delay=0.5)
def _alphavantage_current_price(ticker: str) -> dict | None:
    """Alpha Vantage GLOBAL_QUOTE 获取价格。"""
    if not ALPHA_VANTAGE_API_KEY:
        return None
    try:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        resp = httpx.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        data = resp.json().get("Global Quote", {})
        price = _validate_price(data.get("05. price"), "av_price")
        if price is None:
            return None
        prev_close = _validate_price(data.get("08. previous close"))
        logger.info("[AlphaVantage] Got price for %s: %s", ticker, price)
        return {
            "ticker": ticker, "source": "alphavantage",
            "current_price": price,
            "previous_close": prev_close,
            "name": ticker, "currency": "USD",
            "data_available": True,
        }
    except Exception as e:
        logger.warning("[AlphaVantage] Failed for %s: %s", ticker, e)
        return None


# ================================================================
#  数据源 4：Stooq（免费 CSV 历史数据，无 key 限制）
# ================================================================
def _ticker_to_stooq(ticker: str) -> str | None:
    """将 Yahoo Finance ticker 转为 Stooq 格式。"""
    if ticker.startswith("^"):
        # 指数映射
        idx_map = {"^GSPC": "^SPX", "^IXIC": "^NDQ", "^DJI": "^DJI"}
        return idx_map.get(ticker)
    if ticker.endswith(".HK"):
        # 港股：去前导零
        code = ticker.replace(".HK", "").lstrip("0")
        return f"{code}.HK"
    if ticker.isalpha() and ticker.isupper():
        return f"{ticker}.US"
    return None


@retry_with_backoff(max_retries=2, base_delay=0.3)
def _stooq_history(ticker: str, days: int) -> dict | None:
    """Stooq CSV API 获取历史数据。"""
    stooq_sym = _ticker_to_stooq(ticker)
    if not stooq_sym:
        return None

    try:
        url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        if resp.status_code != 200:
            return None

        lines = resp.text.strip().split("\n")
        if len(lines) < 3:  # header + at least 2 rows
            return None

        # 解析 CSV: Date,Open,High,Low,Close,Volume
        import io
        import pandas as pd
        df = pd.read_csv(io.StringIO(resp.text))
        if df.empty or "Close" not in df.columns:
            return None

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index().tail(days)
        df = df.dropna(subset=["Close"])

        if len(df) < 2:
            return None

        logger.info("[Stooq] Got %d rows for %s (%s)", len(df), ticker, stooq_sym)
        return _build_history_result(ticker, df, days, source="stooq")

    except Exception as e:
        logger.warning("[Stooq] Failed for %s: %s", ticker, e)
        return None


def _stooq_current_price(ticker: str) -> dict | None:
    """从 Stooq 历史数据的最后一行提取当前价格。"""
    stooq_sym = _ticker_to_stooq(ticker)
    if not stooq_sym:
        return None

    try:
        url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
        resp = httpx.get(url, timeout=10, follow_redirects=True)
        if resp.status_code != 200:
            return None

        lines = resp.text.strip().split("\n")
        if len(lines) < 3:
            return None

        # 最后一行: Date,Open,High,Low,Close,Volume
        last_line = lines[-1].split(",")
        if len(last_line) < 5:
            return None

        price = _validate_price(float(last_line[4]), "stooq_close")
        if price is None:
            return None

        # 前一行作为 previous close
        prev_close = None
        if len(lines) >= 4:
            prev_line = lines[-2].split(",")
            if len(prev_line) >= 5:
                prev_close = _validate_price(float(prev_line[4]))

        logger.info("[Stooq] Got price for %s: %s", ticker, price)
        return {
            "ticker": ticker, "source": "stooq",
            "current_price": price,
            "previous_close": prev_close,
            "name": ticker, "currency": "USD",
            "data_available": True,
        }
    except Exception as e:
        logger.warning("[Stooq] Price failed for %s: %s", ticker, e)
        return None


# ================================================================
#  级联调度器
# ================================================================
def _build_history_result(ticker: str, hist, days: int, source: str) -> dict:
    """从 pandas DataFrame 构建统一的历史数据 dict。"""
    start_price = _validate_price(float(hist["Close"].iloc[0]))
    end_price = _validate_price(float(hist["Close"].iloc[-1]))

    if start_price is None or end_price is None or start_price == 0:
        return {"ticker": ticker, "error": "价格异常", "data_available": False}

    change = end_price - start_price
    change_pct = (change / start_price) * 100

    history = []
    for idx, row in hist.iterrows():
        close = _validate_price(float(row["Close"]))
        if close is None:
            continue
        volume = int(row["Volume"]) if not math.isnan(row.get("Volume", 0)) else 0
        history.append({"date": idx.strftime("%Y-%m-%d"), "close": close, "volume": volume})

    high = _validate_price(float(hist["High"].max()))
    low = _validate_price(float(hist["Low"].min()))

    # 趋势分类：委托 trend.py 规则模块（含涨跌幅 + 振幅 + 斜率）
    from app.services.trend import classify_trend
    prices = [p["close"] for p in history]
    period_label = f"{days}日"
    trend_result = classify_trend(
        change_pct=_safe_round(change_pct),
        high=high,
        low=low,
        prices=prices or None,
        period_label=period_label,
    )

    return {
        "ticker": ticker, "source": source,
        "period_days": days,
        "start_price": start_price, "end_price": end_price,
        "change": _safe_round(change), "change_pct": _safe_round(change_pct),
        "high": high, "low": low,
        "trend": trend_result.label_cn, "history": history,
        "trend_detail": trend_result.to_dict(),
        "data_available": True,
    }


def get_current_price(ticker: str) -> dict:
    """级联获取当前价格：yfinance → Finnhub → Alpha Vantage。

    任何一个源成功就立即返回。全部失败返回 data_available=False。
    """
    sources = [
        ("yfinance", lambda: _yfinance_current_price(ticker)),
        ("finnhub", lambda: _finnhub_current_price(ticker)),
        ("alphavantage", lambda: _alphavantage_current_price(ticker)),
        ("stooq", lambda: _stooq_current_price(ticker)),
    ]

    for name, fetcher in sources:
        try:
            result = fetcher()
            if result and result.get("data_available"):
                logger.info("[Price] %s succeeded for %s", name, ticker)
                return result
        except Exception as e:
            logger.warning("[Price] %s failed for %s: %s", name, ticker, e)

    logger.error("[Price] All sources failed for %s", ticker)
    return {
        "ticker": ticker, "name": ticker, "currency": "USD",
        "data_available": False, "source": "none",
    }


def get_history_and_change(ticker: str, days: int = 7) -> dict:
    """获取历史数据：yfinance → Stooq 级联。"""
    sources = [
        ("yfinance", lambda: _yfinance_history(ticker, days)),
        ("stooq", lambda: _stooq_history(ticker, days)),
    ]

    for name, fetcher in sources:
        try:
            result = fetcher()
            if result and result.get("data_available"):
                logger.info("[History] %s succeeded for %s/%dd", name, ticker, days)
                return result
        except Exception as e:
            logger.warning("[History] %s failed for %s: %s", name, ticker, e)

    return {
        "ticker": ticker,
        "error": f"过去 {days} 天无可用历史数据",
        "data_available": False,
    }


# ========== 缓存包装 ==========
def _cached_current_price(ticker: str) -> dict:
    cache_key = _cache._make_key("price", ticker)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    data = get_current_price(ticker)
    if data.get("data_available"):
        _cache.set(cache_key, data, MarketDataCache.PRICE_TTL)
        return data

    # 降级：过期缓存
    stale_data, age = _cache.get_stale(cache_key)
    if stale_data is not None:
        stale_data = {**stale_data, "stale": True, "stale_age_seconds": round(age)}
        logger.warning("[FALLBACK] Stale cache for %s (age: %.0fs)", ticker, age)
        return stale_data

    return data


def _cached_history(ticker: str, days: int) -> dict:
    cache_key = _cache._make_key("history", ticker, days)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    data = get_history_and_change(ticker, days)
    if data.get("data_available"):
        _cache.set(cache_key, data, MarketDataCache.HISTORY_TTL)
        return data

    stale_data, age = _cache.get_stale(cache_key)
    if stale_data is not None:
        stale_data = {**stale_data, "stale": True, "stale_age_seconds": round(age)}
        logger.warning("[FALLBACK] Stale history for %s/%dd (age: %.0fs)", ticker, days, age)
        return stale_data

    return data


def get_stock_summary(ticker: str) -> dict:
    """获取完整摘要：级联数据源 + 缓存 + 降级。"""
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
        "data_source": price_info.get("source", "none"),
        "query_time": datetime.now().isoformat(),
        "cache_stats": _cache.stats,
    }
