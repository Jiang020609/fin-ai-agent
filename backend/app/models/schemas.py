"""Pydantic 数据模型"""

from pydantic import BaseModel


class HistoryMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []
    session_id: str | None = None


class ChartPoint(BaseModel):
    date: str
    close: float
    volume: int


class ThoughtStepSchema(BaseModel):
    """Agent 思考链单步记录 — 前端可据此展示推理过程。"""
    step: str
    result: str
    detail: dict = {}


class MarketMeta(BaseModel):
    """行情关键指标 — 前端 KPI 卡片数据源。"""
    current_price: float | None = None
    previous_close: float | None = None
    change_pct: float | None = None
    pe_ratio: float | None = None
    market_cap: int | None = None
    name: str | None = None
    currency: str = "USD"


# ========== 结构化回答模型 ==========

class DataSummary(BaseModel):
    """客观行情数据摘要 — 所有字段均来自市场 API，非 LLM 生成。"""
    price: float | None = None
    currency: str = "USD"
    change_7d: float | None = None       # 7日涨跌幅 (%)
    change_30d: float | None = None      # 30日涨跌幅 (%)
    high_7d: float | None = None
    low_7d: float | None = None
    high_30d: float | None = None
    low_30d: float | None = None
    timestamp: str | None = None         # 数据查询时间
    data_source: str | None = None       # yahoo / finnhub / stooq 等


class TrendSummary(BaseModel):
    """趋势分类 — 规则计算，非 LLM 判断。"""
    label: str | None = None             # uptrend / downtrend / sideways
    label_cn: str | None = None          # 上涨 / 下跌 / 震荡
    rationale: str | None = None         # 简短依据，如"7日涨幅 +5.2%"


class SourceItem(BaseModel):
    """来源引用条目。"""
    title: str = ""
    source: str = ""                     # 来源名称（如 Tavily、知识库文档名）
    url: str | None = None
    published_at: str | None = None
    page: str | None = None              # 页码（PDF），如 "3"
    relevance_score: float | None = None  # 相关度 0-1


class AnalysisSection(BaseModel):
    """分析段落 — LLM 生成的解读内容。"""
    title: str = ""
    content: str = ""


class ExplainabilityMeta(BaseModel):
    """可解释性元数据 — 记录数据来源和置信度信息。"""
    data_sources: list[str] = []          # 使用了哪些数据源
    evidence_count: int = 0               # 证据条数
    confidence_data: str = "low"          # 数据置信度: high / medium / low
    confidence_reasoning: str = "low"     # 推理置信度: high / medium / low


class ComparisonResult(BaseModel):
    """多资产对比结果。"""
    assets: list[dict] = []
    winners: dict = {}
    tickers: list[str] = []


class EvidenceAnalysisResult(BaseModel):
    """新闻分类结果。"""
    main_drivers: list[str] = []
    secondary_drivers: list[str] = []
    evidence_strength: str = "low"
    summary: str = ""


class StructuredResponse(BaseModel):
    """结构化回答 — 区分客观数据、趋势判断、分析解读、来源引用。

    设计原则：
    - data_summary / trend_summary 由后端程序从 API 数据组装，保证准确
    - analysis 由 LLM 生成，但限定在"解释和组织"角色
    - sources 记录数据和证据的来源
    - disclaimer 固定风险提示
    """
    response_type: str = "general"       # market_data / market_reasoning / knowledge_rag / compare / general
    data_summary: DataSummary | None = None
    trend_summary: TrendSummary | None = None
    analysis: list[AnalysisSection] = []
    sources: list[SourceItem] = []
    disclaimer: str | None = None
    comparison: ComparisonResult | None = None
    evidence_analysis: EvidenceAnalysisResult | None = None
    meta: ExplainabilityMeta | None = None


class ChatResponse(BaseModel):
    text_response: str
    chart_data: list[ChartPoint] | None = None
    intent: str
    ticker: str | None = None
    tickers: list[str] | None = None
    rag_used: bool | None = None
    steps: list[ThoughtStepSchema] = []
    market_meta: MarketMeta | None = None
    structured_response: StructuredResponse | None = None
