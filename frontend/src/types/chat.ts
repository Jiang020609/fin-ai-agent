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

export interface ChatResponse {
  text_response: string;
  chart_data: ChartPoint[] | null;
  intent: string;
  ticker: string | null;
  rag_used: boolean | null;
  steps: ThoughtStep[];
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
