"""步骤函数单元测试"""

from unittest.mock import patch, MagicMock
import pytest
from dataclasses import asdict

from app.services.orchestrator import StepContext, QueryPlan
from app.services.steps import (
    ThoughtStep, parse_web_sources, parse_rag_sources,
    build_no_data_message, build_data_summary, build_explainability_meta,
    step_fetch_price, step_validate_data, step_extract_date,
    step_rag_search, step_web_search_fallback, step_build_general_prompt,
    step_generate_answer, step_assemble_general_response,
    step_assemble_market_response, step_compute_comparison,
    step_build_market_prompt,
)

_STEPS = "app.services.steps"


class TestParseWebSources:
    def test_parse_valid_sources(self):
        text = (
            "[网络搜索结果 1] 特斯拉大涨5%\n来源: https://example.com\n\n"
            "[网络搜索结果 2] AI芯片需求激增\n来源: https://example2.com"
        )
        sources = parse_web_sources(text)
        assert len(sources) == 2
        assert sources[0]["url"] == "https://example.com"
        assert sources[0]["source"] == "Web Search"

    def test_parse_empty(self):
        assert parse_web_sources("") == []

    def test_parse_no_match(self):
        assert parse_web_sources("普通文本，没有来源") == []


class TestParseRagSources:
    def test_parse_with_page_and_score(self):
        context = (
            "[参考资料 1] 市盈率说明\n"
            "来源: financial_basics.md, 页码: 3, 相关度: 0.85\n\n---\n\n"
            "其他内容"
        )
        sources = parse_rag_sources(context)
        assert len(sources) == 1
        assert sources[0]["title"] == "financial_basics.md"
        assert sources[0]["page"] == "3"
        assert sources[0]["relevance_score"] == 0.85

    def test_parse_fallback_format(self):
        context = "[参考资料 1] 内容\n来源: doc.md"
        sources = parse_rag_sources(context)
        assert len(sources) == 1
        assert sources[0]["title"] == "doc.md"


class TestBuildNoDataMessage:
    def test_contains_ticker(self):
        msg = build_no_data_message("XYZXYZ")
        assert "XYZXYZ" in msg
        assert "无法获取" in msg

    def test_no_hallucination_hint(self):
        msg = build_no_data_message("AAPL")
        assert "不会从训练数据中猜测" in msg


class TestBuildDataSummary:
    def test_extracts_fields(self):
        market_data = {
            "price": {"current_price": 100, "currency": "USD"},
            "change_7d": {"change_pct": 2.5, "high": 105, "low": 95},
            "change_30d": {"change_pct": -1.0, "high": 110, "low": 90},
            "query_time": "2024-01-01T00:00:00",
            "data_source": "yahoo",
        }
        summary = build_data_summary(market_data)
        assert summary["price"] == 100
        assert summary["change_7d"] == 2.5
        assert summary["change_30d"] == -1.0


class TestBuildExplainabilityMeta:
    def test_high_confidence_with_api_data(self):
        meta = build_explainability_meta(["yahoo"], has_api_data=True, evidence_count=3)
        assert meta["confidence_data"] == "high"
        assert meta["confidence_reasoning"] == "high"

    def test_low_confidence_without_data(self):
        meta = build_explainability_meta([], has_api_data=False, evidence_count=0)
        assert meta["confidence_data"] == "low"
        assert meta["confidence_reasoning"] == "low"


class TestStepFetchPrice:
    @patch(f"{_STEPS}.get_stock_summary")
    def test_writes_market_data_to_ctx(self, mock_market):
        mock_market.return_value = {"data_available": True, "price": {"current_price": 100}, "change_7d": {}, "change_30d": {}}
        ctx = StepContext(ticker="AAPL")
        step_fetch_price(ctx)
        assert ctx.market_data is not None
        assert ctx.market_data["data_available"] is True

    @patch(f"{_STEPS}.get_stock_summary")
    def test_emits_thought_events(self, mock_market):
        mock_market.return_value = {"data_available": True, "price": {"current_price": 100}, "change_7d": {}, "change_30d": {}}
        ctx = StepContext(ticker="AAPL")
        events = []
        step_fetch_price(ctx, emit=events.append)
        thought_events = [e for e in events if e["event"] == "thought"]
        assert len(thought_events) >= 1


class TestStepValidateData:
    @patch(f"{_STEPS}.validate_market_data")
    def test_early_exit_on_no_data(self, mock_validate):
        mock_validate.return_value = {"is_stale": False, "issues": []}
        ctx = StepContext(
            ticker="XYZXYZ",
            plan=QueryPlan(question_type="market_data"),
        )
        ctx.market_data = {"data_available": False, "price": {}, "change_7d": {}, "change_30d": {}}
        step_validate_data(ctx)
        assert ctx.early_exit is True
        assert "无法获取" in ctx.result["text_response"]

    @patch(f"{_STEPS}.validate_market_data")
    def test_stale_data_continues(self, mock_validate):
        mock_validate.return_value = {"is_stale": True, "issues": ["stale"]}
        ctx = StepContext(ticker="AAPL", plan=QueryPlan(question_type="market_data"))
        ctx.market_data = {"data_available": False, "price": {"stale": True, "stale_age_seconds": 300}, "change_7d": {}, "change_30d": {}}
        step_validate_data(ctx)
        assert ctx.early_exit is False
        assert ctx.is_stale is True


class TestStepRagSearch:
    @patch("app.services.rag.is_rag_available", return_value=True)
    @patch(f"{_STEPS}.get_relevant_context", return_value="[参考资料 1] 内容\n来源: doc.md, 页码: 1, 相关度: 0.9")
    def test_rag_hit(self, mock_rag, mock_avail):
        ctx = StepContext(question="什么是市盈率")
        step_rag_search(ctx)
        assert ctx.rag_used is True
        assert len(ctx.rag_context) > 0

    @patch("app.services.rag.is_rag_available", return_value=True)
    @patch(f"{_STEPS}.get_relevant_context", return_value="")
    def test_rag_miss(self, mock_rag, mock_avail):
        ctx = StepContext(question="什么是量子计算")
        step_rag_search(ctx)
        assert ctx.rag_used is False

    def test_rag_unavailable_skips(self):
        """RAG 不可用时应跳过，不报错。"""
        with patch("app.services.rag.is_rag_available", return_value=False):
            ctx = StepContext(question="什么是市盈率")
            step_rag_search(ctx)
            assert ctx.rag_used is False
            assert len(ctx.steps) == 1
            assert "未加载" in ctx.steps[0].result


class TestStepWebSearchFallback:
    @patch(f"{_STEPS}.web_search", return_value="[网络搜索结果 1] 结果\n来源: https://a.com")
    def test_fallback_when_rag_missed(self, mock_ws):
        ctx = StepContext(question="什么是量子计算", rag_used=False)
        step_web_search_fallback(ctx)
        assert ctx.rag_used is True
        assert len(ctx.web_sources) == 1

    def test_skip_when_rag_hit(self):
        ctx = StepContext(question="什么是市盈率", rag_used=True)
        step_web_search_fallback(ctx)
        assert ctx.web_sources == []


class TestStepGenerateAnswer:
    @patch(f"{_STEPS}.chat_completion", return_value="LLM回答")
    def test_sync_mode(self, mock_llm):
        ctx = StepContext(system_prompt="system", user_message="user")
        step_generate_answer(ctx)
        assert ctx.text_response == "LLM回答"

    @patch(f"{_STEPS}.chat_completion_stream", return_value=iter(["你", "好"]))
    def test_stream_mode(self, mock_stream):
        ctx = StepContext(system_prompt="system", user_message="user")
        events = list(step_generate_answer(ctx, stream=True))
        token_events = [e for e in events if e["event"] == "token"]
        assert len(token_events) == 2
        assert ctx.text_response == "你好"


class TestStepAssembleGeneralResponse:
    def test_assembles_result(self):
        ctx = StepContext(text_response="回答内容")
        step_assemble_general_response(ctx)
        assert ctx.result is not None
        assert ctx.result["intent"] == "general"
        assert ctx.result["text_response"] == "回答内容"

    def test_emit_meta_event(self):
        ctx = StepContext(text_response="回答")
        events = []
        step_assemble_general_response(ctx, emit=events.append)
        meta_events = [e for e in events if e["event"] == "meta"]
        assert len(meta_events) == 1
        assert meta_events[0]["data"]["intent"] == "general"
