"""公司名称 / 别名 → 股票代码 映射模块"""

# 常见公司名（中文 / 英文 / 别名）→ Yahoo Finance Ticker
_TICKER_MAP: dict[str, str] = {
    # 中概股
    "阿里巴巴": "BABA",
    "阿里": "BABA",
    "alibaba": "BABA",
    "百度": "BIDU",
    "baidu": "BIDU",
    "京东": "JD",
    "jd": "JD",
    "拼多多": "PDD",
    "pinduoduo": "PDD",
    "网易": "NTES",
    "netease": "NTES",
    "腾讯": "0700.HK",
    "tencent": "0700.HK",
    "比亚迪": "1211.HK",
    "byd": "1211.HK",
    "小米": "1810.HK",
    "xiaomi": "1810.HK",
    # 美股科技
    "特斯拉": "TSLA",
    "tesla": "TSLA",
    "苹果": "AAPL",
    "apple": "AAPL",
    "谷歌": "GOOGL",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "微软": "MSFT",
    "microsoft": "MSFT",
    "亚马逊": "AMZN",
    "amazon": "AMZN",
    "英伟达": "NVDA",
    "nvidia": "NVDA",
    "meta": "META",
    "脸书": "META",
    "facebook": "META",
    "奈飞": "NFLX",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "英特尔": "INTC",
    # 金融
    "摩根大通": "JPM",
    "jpmorgan": "JPM",
    "高盛": "GS",
    "goldman": "GS",
    # 指数 & ETF
    "标普500": "^GSPC",
    "s&p500": "^GSPC",
    "纳斯达克": "^IXIC",
    "nasdaq": "^IXIC",
    "道琼斯": "^DJI",
    "dow jones": "^DJI",
}


def resolve_ticker(query: str) -> str | None:
    """尝试从用户输入中识别股票代码。

    优先级：
    1. 输入本身就是合法 ticker（全大写字母 / 含 . 或 ^）
    2. 在映射表中匹配（忽略大小写）
    """
    cleaned = query.strip()

    # 如果输入看起来就是 ticker（如 BABA, 0700.HK, ^GSPC）
    upper = cleaned.upper()
    if upper.isascii() and upper.isalpha() and len(upper) <= 5:
        return upper
    if ("." in cleaned or cleaned.startswith("^")) and upper.isascii():
        return upper

    # 在映射表中查找
    key = cleaned.lower()
    if key in _TICKER_MAP:
        return _TICKER_MAP[key]

    # 子串模糊匹配（用户说 "阿里巴巴的股价" → 提取 "阿里巴巴"）
    for name, ticker in _TICKER_MAP.items():
        if name in key:
            return ticker

    return None
