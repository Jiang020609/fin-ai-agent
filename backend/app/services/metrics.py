"""Metrics 监控 — 单例收集器，记录请求数/耗时/缓存命中率"""

import time
import threading
from collections import defaultdict


class MetricsCollector:
    """线程安全的指标收集器（单例）。"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._data_lock = threading.Lock()
        self._start_time = time.time()

    def incr(self, key: str, amount: int = 1):
        with self._data_lock:
            self._counters[key] += amount

    def timing(self, key: str, duration_ms: float):
        with self._data_lock:
            self._timings[key].append(duration_ms)
            # 保留最近 1000 条
            if len(self._timings[key]) > 1000:
                self._timings[key] = self._timings[key][-500:]

    def snapshot(self) -> dict:
        with self._data_lock:
            timing_stats = {}
            for key, values in self._timings.items():
                if values:
                    sorted_v = sorted(values)
                    timing_stats[key] = {
                        "count": len(sorted_v),
                        "avg_ms": round(sum(sorted_v) / len(sorted_v), 1),
                        "p50_ms": round(sorted_v[len(sorted_v) // 2], 1),
                        "p95_ms": round(sorted_v[int(len(sorted_v) * 0.95)], 1),
                        "max_ms": round(sorted_v[-1], 1),
                    }

            cache_gets = self._counters.get("cache_get", 0)
            cache_hits = self._counters.get("cache_hit", 0)
            hit_rate = round(cache_hits / cache_gets * 100, 1) if cache_gets > 0 else 0

            return {
                "uptime_seconds": round(time.time() - self._start_time),
                "counters": dict(self._counters),
                "timings": timing_stats,
                "cache_hit_rate_pct": hit_rate,
            }


# 全局单例
metrics = MetricsCollector()
