"""行情服务单元测试 — _validate_price、MarketDataCache、_build_history_result"""

import math
import time
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from app.services.market import (
    _validate_price,
    _safe_round,
    MarketDataCache,
    _build_history_result,
)


class TestValidatePrice:
    def test_valid_price(self):
        assert _validate_price(123.456) == 123.46

    def test_none(self):
        assert _validate_price(None) is None

    def test_nan(self):
        assert _validate_price(float("nan")) is None

    def test_inf(self):
        assert _validate_price(float("inf")) is None

    def test_negative(self):
        assert _validate_price(-10.0) is None

    def test_zero(self):
        assert _validate_price(0.0) == 0.0

    def test_string_number(self):
        assert _validate_price(100.5) == 100.5

    def test_invalid_type(self):
        assert _validate_price("abc") is None


class TestSafeRound:
    def test_normal(self):
        assert _safe_round(3.14159, 2) == 3.14

    def test_none(self):
        assert _safe_round(None) is None

    def test_nan(self):
        assert _safe_round(float("nan")) is None


class TestMarketDataCache:
    def test_set_and_get(self):
        cache = MarketDataCache()
        cache.set("test_key", {"price": 100}, ttl=10)
        assert cache.get("test_key") == {"price": 100}

    def test_miss(self):
        cache = MarketDataCache()
        assert cache.get("nonexistent") is None

    def test_expired(self):
        cache = MarketDataCache()
        cache.set("test_key", {"price": 100}, ttl=0.01)
        time.sleep(0.02)
        assert cache.get("test_key") is None

    def test_stale_fallback(self):
        cache = MarketDataCache()
        cache.set("test_key", {"price": 100}, ttl=0.01)
        time.sleep(0.02)
        # 触发 expire → stale
        cache.get("test_key")
        stale_data, age = cache.get_stale("test_key")
        assert stale_data == {"price": 100}
        assert age > 0

    def test_invalidate(self):
        cache = MarketDataCache()
        cache.set("price:TSLA:", {"price": 100}, ttl=60)
        cache.set("history:TSLA:7", {"trend": "up"}, ttl=60)
        cache.invalidate("TSLA")
        assert cache.get("price:TSLA:") is None
        assert cache.get("history:TSLA:7") is None

    def test_stats(self):
        cache = MarketDataCache()
        cache.set("k1", {}, ttl=60)
        cache.set("k2", {}, ttl=60)
        stats = cache.stats
        assert stats["total_keys"] == 2
        assert stats["alive_keys"] == 2


class TestBuildHistoryResult:
    def _make_df(self, prices, volumes=None):
        dates = pd.date_range("2024-01-01", periods=len(prices))
        data = {
            "Close": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Volume": volumes or [1000000] * len(prices),
        }
        return pd.DataFrame(data, index=dates)

    def test_uptrend(self):
        df = self._make_df([100, 102, 105, 108, 110])
        result = _build_history_result("TSLA", df, 5, "yahoo")
        assert result["trend"] == "上涨"
        assert result["data_available"] is True
        assert len(result["history"]) == 5

    def test_downtrend(self):
        df = self._make_df([100, 98, 95, 93, 90])
        result = _build_history_result("TSLA", df, 5, "yahoo")
        assert result["trend"] == "下跌"

    def test_sideways(self):
        df = self._make_df([100, 100.5, 99.8, 100.2, 100.1])
        result = _build_history_result("TSLA", df, 5, "yahoo")
        assert result["trend"] == "震荡"

    def test_change_calculation(self):
        df = self._make_df([100, 110])
        result = _build_history_result("TEST", df, 2, "yahoo")
        assert result["change_pct"] == 10.0
        assert result["change"] == 10.0
