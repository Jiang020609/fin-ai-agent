"""resolve_ticker 单元测试 — 正向/反向/边界用例"""

from app.utils.ticker_map import resolve_ticker


class TestResolveTickerDirect:
    """直接 ticker 格式识别。"""

    def test_uppercase_ticker(self):
        assert resolve_ticker("TSLA") == "TSLA"

    def test_uppercase_short(self):
        assert resolve_ticker("AAPL") == "AAPL"

    def test_hk_ticker(self):
        assert resolve_ticker("0700.HK") == "0700.HK"

    def test_index_ticker(self):
        assert resolve_ticker("^GSPC") == "^GSPC"


class TestResolveTickerMap:
    """映射表精确匹配。"""

    def test_chinese_name(self):
        assert resolve_ticker("阿里巴巴") == "BABA"

    def test_english_name(self):
        assert resolve_ticker("tesla") == "TSLA"

    def test_case_insensitive(self):
        assert resolve_ticker("Tesla") == "TSLA"

    def test_alias(self):
        assert resolve_ticker("阿里") == "BABA"


class TestResolveTickerFuzzy:
    """子串模糊匹配。"""

    def test_chinese_with_suffix(self):
        assert resolve_ticker("阿里巴巴的股价") == "BABA"

    def test_mixed_sentence(self):
        assert resolve_ticker("特斯拉近期走势如何") == "TSLA"


class TestResolveTickerNone:
    """无法识别的输入。"""

    def test_empty(self):
        assert resolve_ticker("") is None

    def test_random_text(self):
        assert resolve_ticker("今天天气怎么样") is None

    def test_whitespace(self):
        assert resolve_ticker("   ") is None
