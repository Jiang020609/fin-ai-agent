"""聊天 API 端到端测试 — mock LLM + market"""

from unittest.mock import patch, MagicMock
import pytest


# Mock paths — 步骤函数从 steps.py 导入外部服务
_STEPS = "app.services.steps"
_AGENT = "app.services.agent"
_LLM = "app.services.llm"


def _make_market_data(**overrides):
    """构造标准的 market_data mock 数据。"""
    base = {
        "data_available": True,
        "data_source": "yahoo",
        "price": {
            "current_price": 250.0,
            "previous_close": 248.0,
            "source": "yahoo",
            "data_available": True,
            "name": "Tesla",
            "currency": "USD",
            "pe_ratio": 50.0,
            "market_cap": 800000000000,
        },
        "change_7d": {
            "data_available": True,
            "trend": "上涨",
            "change_pct": 3.5,
        },
        "change_30d": {
            "data_available": True,
            "trend": "震荡",
            "history": [
                {"date": "2024-01-01", "close": 240.0, "volume": 1000000},
                {"date": "2024-01-02", "close": 250.0, "volume": 1200000},
            ],
        },
        "query_time": "2024-01-02T12:00:00",
        "cache_stats": {"total_keys": 0, "alive_keys": 0},
    }
    base.update(overrides)
    return base


class TestChatEndpoint:
    def test_empty_question_returns_400(self, client):
        res = client.post("/api/chat", json={"question": ""})
        assert res.status_code == 400

    def test_whitespace_question_returns_400(self, client):
        res = client.post("/api/chat", json={"question": "   "})
        assert res.status_code == 400

    @patch(f"{_LLM}.classify_intent_with_tools", return_value=("general", {}))
    @patch(f"{_STEPS}.chat_completion", return_value="你好！我是金融助手。")
    def test_general_question_success(self, mock_llm, mock_intent, client):
        res = client.post("/api/chat", json={"question": "你好"})
        assert res.status_code == 200
        data = res.json()
        assert "text_response" in data
        assert data["intent"] == "general"
        assert isinstance(data["steps"], list)

    @patch(f"{_STEPS}.get_stock_summary")
    @patch(f"{_STEPS}.chat_completion", return_value="特斯拉当前股价 $250。")
    @patch(f"{_STEPS}.validate_market_response_numbers", return_value=(True, []))
    @patch(f"{_STEPS}.validate_market_data", return_value={"is_stale": False, "issues": []})
    def test_market_question_success(self, mock_validate, mock_nums, mock_llm, mock_market, client):
        mock_market.return_value = _make_market_data()
        res = client.post("/api/chat", json={"question": "特斯拉股价"})
        assert res.status_code == 200
        data = res.json()
        assert data["intent"] == "market_data"
        assert data["ticker"] == "TSLA"
        assert data["chart_data"] is not None
        assert data.get("market_meta") is not None

    @patch("app.routers.chat.process_question", side_effect=RuntimeError("LLM down"))
    def test_internal_error_returns_500(self, mock_process, client):
        res = client.post("/api/chat", json={"question": "test error"})
        assert res.status_code == 500
        assert "处理失败" in res.json()["detail"]

    @patch(f"{_LLM}.classify_intent_with_tools", return_value=("general", {}))
    @patch(f"{_STEPS}.chat_completion", return_value="回答")
    def test_response_schema_fields(self, mock_llm, mock_intent, client):
        """验证响应 schema 包含所有必需字段。"""
        res = client.post("/api/chat", json={"question": "hello"})
        data = res.json()
        assert "text_response" in data
        assert "chart_data" in data
        assert "intent" in data
        assert "steps" in data

    @patch(f"{_LLM}.classify_intent_with_tools", return_value=("general", {}))
    @patch(f"{_STEPS}.chat_completion", return_value="回答")
    def test_history_accepted(self, mock_llm, mock_intent, client):
        """验证 history 参数被接受。"""
        res = client.post("/api/chat", json={
            "question": "它最近怎么样",
            "history": [
                {"role": "user", "content": "特斯拉"},
                {"role": "assistant", "content": "特斯拉是一家电动车公司"},
            ]
        })
        assert res.status_code == 200

    @patch(f"{_STEPS}.get_stock_summary")
    @patch(f"{_STEPS}.validate_market_data", return_value={"is_stale": False, "issues": []})
    def test_no_data_returns_friendly_message(self, mock_validate, mock_market, client):
        """API 无数据时返回用户友好提示，不调用 LLM。"""
        mock_market.return_value = {"data_available": False, "price": {}, "change_7d": {}, "change_30d": {}}
        # 使用已知中文名以确保进入 market_data 路由
        res = client.post("/api/chat", json={"question": "特斯拉股价"})
        assert res.status_code == 200
        data = res.json()
        assert "无法获取" in data["text_response"]

    @patch(f"{_STEPS}.get_stock_summary")
    @patch(f"{_STEPS}.chat_completion", return_value="对比分析结果")
    @patch(f"{_STEPS}.compute_comparison")
    def test_compare_question(self, mock_compare, mock_llm, mock_market, client):
        """多资产对比问题应返回 compare 意图。"""
        mock_market.return_value = _make_market_data()
        mock_compare.return_value = {
            "assets": [{"ticker": "AAPL"}, {"ticker": "MSFT"}],
            "winners": {"return_7d": "AAPL"},
        }
        res = client.post("/api/chat", json={"question": "苹果和微软对比"})
        assert res.status_code == 200
        data = res.json()
        assert data["intent"] == "compare"


class TestStreamEndpoint:
    @patch(f"{_LLM}.classify_intent_with_tools", return_value=("general", {}))
    @patch(f"{_STEPS}.chat_completion_stream")
    def test_stream_returns_sse(self, mock_stream, mock_intent, client):
        mock_stream.return_value = iter(["你", "好", "！"])
        res = client.post("/api/chat/stream", json={"question": "你好"})
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

    def test_stream_empty_question_400(self, client):
        res = client.post("/api/chat/stream", json={"question": ""})
        assert res.status_code == 400

    @patch(f"{_LLM}.classify_intent_with_tools", return_value=("general", {}))
    @patch(f"{_STEPS}.chat_completion_stream")
    def test_stream_events_contain_done(self, mock_stream, mock_intent, client):
        """流式响应应包含 done 事件。"""
        mock_stream.return_value = iter(["回答"])
        res = client.post("/api/chat/stream", json={"question": "你好"})
        body = res.text
        assert "event: done" in body
