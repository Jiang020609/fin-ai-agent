/**
 * API 客户端 — 带请求拦截器、统一错误处理、SSE 流式支持
 */

import { ChatResponse, HistoryMessage, ThoughtStep, ChartPoint, MarketMeta, StructuredResponse } from "@/types/chat";

const API_BASE = "/api";
const REQUEST_TIMEOUT_MS = 30000;

// ---------- 请求拦截器 ----------
interface RequestContext {
  requestId: string;
  startTime: number;
}

function beforeRequest(): RequestContext {
  const requestId = `req_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  console.log(`[API] >>> ${requestId} | 请求发送`);
  return { requestId, startTime: Date.now() };
}

function afterResponse(ctx: RequestContext, ok: boolean) {
  const elapsed = Date.now() - ctx.startTime;
  const status = ok ? "成功" : "失败";
  console.log(`[API] <<< ${ctx.requestId} | ${status} | ${elapsed}ms`);

  if (elapsed > 15000) {
    console.warn(`[API] 慢请求告警 | ${ctx.requestId} | ${elapsed}ms`);
  }
}

// ---------- 统一错误处理 ----------
class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public requestId: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function parseErrorResponse(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return body.detail || body.message || `HTTP ${res.status}`;
  } catch {
    return `服务暂不可用 (HTTP ${res.status})`;
  }
}

// ---------- 非流式请求 ----------
export async function sendMessage(
  question: string,
  history?: HistoryMessage[]
): Promise<ChatResponse> {
  const ctx = beforeRequest();

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Request-Id": ctx.requestId,
      },
      body: JSON.stringify({ question, history: history || [] }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const errorMsg = await parseErrorResponse(res);
      afterResponse(ctx, false);
      throw new ApiError(errorMsg, res.status, ctx.requestId);
    }

    const data: ChatResponse = await res.json();
    afterResponse(ctx, true);
    return data;
  } catch (err) {
    if (err instanceof ApiError) throw err;

    afterResponse(ctx, false);
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ApiError("请求超时，请稍后重试", 408, ctx.requestId);
    }
    throw new ApiError(
      "网络连接失败，请检查后端服务是否运行",
      0,
      ctx.requestId
    );
  } finally {
    clearTimeout(timeout);
  }
}

// ---------- SSE 流式请求 ----------
export interface StreamCallbacks {
  onToken: (text: string) => void;
  onThought: (step: ThoughtStep) => void;
  onChart: (data: ChartPoint[]) => void;
  onMeta: (meta: { intent: string; ticker: string | null; rag_used: boolean | null; market_meta?: MarketMeta | null; structured_response?: StructuredResponse | null }) => void;
  onDone: (data: { steps: ThoughtStep[] }) => void;
  onError: (error: string) => void;
}

export async function sendMessageStream(
  question: string,
  callbacks: StreamCallbacks,
  history?: HistoryMessage[]
): Promise<void> {
  const ctx = beforeRequest();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000); // 流式给更长超时

  try {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Request-Id": ctx.requestId,
      },
      body: JSON.stringify({ question, history: history || [] }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const errorMsg = await parseErrorResponse(res);
      afterResponse(ctx, false);
      callbacks.onError(errorMsg);
      return;
    }

    const reader = res.body?.getReader();
    if (!reader) {
      callbacks.onError("浏览器不支持流式响应");
      return;
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let lastDataTime = Date.now();
    const STREAM_IDLE_TIMEOUT = 60000; // 60s 无数据视为超时

    while (true) {
      // 为每次 read 加超时保护，防止后端挂住时前端永久阻塞
      const readResult = await Promise.race([
        reader.read(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("stream_idle_timeout")), STREAM_IDLE_TIMEOUT)
        ),
      ]);
      const { done, value } = readResult;
      if (done) break;
      lastDataTime = Date.now();

      buffer += decoder.decode(value, { stream: true });

      // 解析 SSE 格式
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // 保留未完成行

      let currentEvent = "token";
      for (const line of lines) {
        if (line.startsWith("event: ")) {
          currentEvent = line.slice(7).trim();
        } else if (line.startsWith("data: ")) {
          const rawData = line.slice(6);
          try {
            switch (currentEvent) {
              case "token":
                // token data 可以是纯字符串或 JSON 字符串
                if (rawData.startsWith('"')) {
                  callbacks.onToken(JSON.parse(rawData));
                } else {
                  callbacks.onToken(rawData);
                }
                break;
              case "thought":
                callbacks.onThought(JSON.parse(rawData));
                break;
              case "chart":
                callbacks.onChart(JSON.parse(rawData));
                break;
              case "meta":
                callbacks.onMeta(JSON.parse(rawData));
                break;
              case "done":
                callbacks.onDone(JSON.parse(rawData));
                break;
              case "error":
                callbacks.onError(JSON.parse(rawData).message || "未知错误");
                break;
            }
          } catch {
            // 非 JSON token — 直接当文本
            if (currentEvent === "token") {
              callbacks.onToken(rawData);
            }
          }
        }
      }
    }

    afterResponse(ctx, true);
  } catch (err) {
    afterResponse(ctx, false);
    if (err instanceof DOMException && err.name === "AbortError") {
      callbacks.onError("请求超时，请稍后重试");
    } else if (err instanceof Error && err.message === "stream_idle_timeout") {
      callbacks.onError("服务响应超时，请稍后重试");
    } else {
      callbacks.onError("网络连接失败，请检查后端服务是否运行");
    }
  } finally {
    clearTimeout(timeout);
  }
}
