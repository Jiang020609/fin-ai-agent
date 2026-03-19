/**
 * API 客户端 — 带请求拦截器和统一错误处理
 *
 * 生产级可靠性：
 * - 请求拦截器：统一注入 headers、添加请求 ID、记录耗时
 * - 响应拦截器：统一解析错误格式、区分网络错误 vs 业务错误
 * - 超时控制：AbortController 防止请求无限挂起
 */

import { ChatResponse } from "@/types/chat";

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

  // 慢请求告警（前端侧）
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
    // 非 JSON 响应（如 502 网关错误）
    return `服务暂不可用 (HTTP ${res.status})`;
  }
}

// ---------- 核心请求函数 ----------
export async function sendMessage(question: string): Promise<ChatResponse> {
  const ctx = beforeRequest();

  // 超时控制 — 防止请求无限挂起
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Request-Id": ctx.requestId,
      },
      body: JSON.stringify({ question }),
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

    // 网络错误 / 超时
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
