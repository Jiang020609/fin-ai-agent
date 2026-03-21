export interface ChartPoint {
  date: string;
  close: number;
  volume: number;
}

export interface ThoughtStep {
  step: string;
  result: string;
  detail?: Record<string, unknown>;
}

export interface MarketMeta {
  current_price: number | null;
  previous_close: number | null;
  change_pct: number | null;
  pe_ratio: number | null;
  market_cap: number | null;
  name: string | null;
  currency: string;
}

// ========== 结构化回答类型 ==========

export interface DataSummary {
  price: number | null;
  currency: string;
  change_7d: number | null;
  change_30d: number | null;
  high_7d: number | null;
  low_7d: number | null;
  high_30d: number | null;
  low_30d: number | null;
  timestamp: string | null;
  data_source: string | null;
}

export interface TrendSummary {
  label: string | null;        // uptrend / downtrend / sideways
  label_cn: string | null;
  confidence: string | null;   // high / medium / low
  rationale: string | null;
  change_pct: number | null;
  amplitude: number | null;
  slope_direction: string | null;
}

export interface SourceItem {
  title: string;
  source: string;
  url: string | null;
  published_at: string | null;
  page: string | null;
  relevance_score: number | null;
}

export interface AnalysisSection {
  title: string;
  content: string;
}

// ========== 对比分析类型 ==========

export interface ComparisonAsset {
  ticker: string;
  name: string;
  current_price: number | null;
  currency: string;
  return_7d: number | null;
  return_30d: number | null;
  volatility_30d: number | null;
  sharpe_30d: number | null;
  max_drawdown_30d: number | null;
  data_available: boolean;
}

export interface ComparisonWinner {
  ticker: string;
  value: number;
}

export interface ComparisonResult {
  assets: ComparisonAsset[];
  winners: Record<string, ComparisonWinner>;
  tickers: string[];
  assumptions?: {
    risk_free_rate: number;
    risk_free_rate_note: string;
    max_drawdown_note: string;
  };
}

// ========== 证据分析类型 ==========

export interface EvidenceAnalysis {
  main_drivers: string[];
  secondary_drivers: string[];
  evidence_strength: string;       // strong / moderate / weak
  summary: string;
}

export interface StructuredResponse {
  response_type: string;       // market_data / market_reasoning / knowledge_rag / compare / general
  data_summary: DataSummary | null;
  trend_summary: TrendSummary | null;
  analysis: AnalysisSection[];
  sources: SourceItem[];
  disclaimer: string | null;
  comparison?: ComparisonResult | null;
  evidence_analysis?: EvidenceAnalysis | null;
  meta?: Record<string, unknown> | null;
}

// ========== API 响应 & 消息 ==========

export interface ChatResponse {
  text_response: string;
  chart_data: ChartPoint[] | null;
  intent: string;
  ticker: string | null;
  rag_used: boolean | null;
  steps: ThoughtStep[];
  market_meta: MarketMeta | null;
  structured_response: StructuredResponse | null;
}

export interface HistoryMessage {
  role: "user" | "assistant";
  content: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  chartData?: ChartPoint[] | null;
  ticker?: string | null;
  intent?: string;
  ragUsed?: boolean | null;
  steps?: ThoughtStep[];
  loading?: boolean;
  marketMeta?: MarketMeta | null;
  structuredResponse?: StructuredResponse | null;
}

/**
 * 聊天状态机 — 严格约束 UI 状态转换：
 *
 *   idle ──(用户发送)──> sending ──(API 响应开始)──> thinking
 *     │                                                │
 *     │                                           ┌────┴────┐
 *     │                                        success    error
 *     │                                           │         │
 *     └───────────────────────────────────────────┘─────────┘
 *                        (回到 idle)
 */
export type ChatState = "idle" | "sending" | "thinking" | "error" | "success";
