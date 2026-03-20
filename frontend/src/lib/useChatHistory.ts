/**
 * useChatHistory — 基于状态机的聊天状态管理 Hook
 *
 * 支持：SSE 流式响应、多轮对话上下文、错误重试
 *
 *   idle ──(send)──> sending ──> streaming ──> idle
 *                       │            │
 *                       └────> error ─┘──(retry)──> idle
 */

"use client";

import { useCallback, useReducer, useEffect, useRef } from "react";
import { Message, ChatState, ThoughtStep, ChartPoint, MarketMeta, StructuredResponse, HistoryMessage } from "@/types/chat";
import { sendMessageStream } from "./api";

// ---------- localStorage 持久化 ----------
const STORAGE_KEY = "finai_chat_history";

function loadFromStorage(): Message[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Message[];
    return parsed.filter((m) => !m.loading);
  } catch {
    return [];
  }
}

function saveToStorage(messages: Message[]) {
  if (typeof window === "undefined") return;
  try {
    const toSave = messages.filter((m) => !m.loading);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
  } catch {
    // 静默失败
  }
}

function clearStorage() {
  if (typeof window === "undefined") return;
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // 静默失败
  }
}

// ---------- State ----------
interface State {
  messages: Message[];
  chatState: ChatState;
  error: string | null;
}

// ---------- Actions ----------
type Action =
  | { type: "INIT"; messages: Message[] }
  | { type: "SEND_START"; userMsg: Message; loadingMsg: Message }
  | { type: "THINKING" }
  | { type: "STREAM_TOKEN"; loadingId: string; token: string }
  | { type: "STREAM_THOUGHT"; loadingId: string; step: ThoughtStep }
  | { type: "STREAM_CHART"; loadingId: string; chart: ChartPoint[] }
  | { type: "STREAM_META"; loadingId: string; meta: { intent: string; ticker: string | null; rag_used: boolean | null; market_meta?: MarketMeta | null; structured_response?: StructuredResponse | null } }
  | { type: "STREAM_DONE"; loadingId: string; steps: ThoughtStep[] }
  | { type: "ERROR"; loadingId: string; error: string }
  | { type: "REMOVE_LAST_PAIR"; keepQuestion?: string }
  | { type: "CLEAR" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "INIT":
      return { ...state, messages: action.messages };

    case "SEND_START":
      return {
        ...state,
        messages: [...state.messages, action.userMsg, action.loadingMsg],
        chatState: "sending",
        error: null,
      };

    case "THINKING":
      return { ...state, chatState: "thinking" };

    case "STREAM_TOKEN":
      return {
        ...state,
        chatState: "thinking",
        messages: state.messages.map((m) =>
          m.id === action.loadingId
            ? { ...m, content: m.content + action.token, loading: false }
            : m
        ),
      };

    case "STREAM_THOUGHT":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.loadingId
            ? { ...m, steps: [...(m.steps || []), action.step] }
            : m
        ),
      };

    case "STREAM_CHART":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.loadingId
            ? { ...m, chartData: action.chart }
            : m
        ),
      };

    case "STREAM_META":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.loadingId
            ? {
                ...m,
                intent: action.meta.intent,
                ticker: action.meta.ticker,
                ragUsed: action.meta.rag_used,
                marketMeta: action.meta.market_meta || null,
                structuredResponse: action.meta.structured_response || null,
              }
            : m
        ),
      };

    case "STREAM_DONE":
      return {
        ...state,
        chatState: "idle",
        error: null,
        messages: state.messages.map((m) =>
          m.id === action.loadingId
            ? { ...m, loading: false, steps: action.steps }
            : m
        ),
      };

    case "ERROR":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.loadingId
            ? {
                id: m.id,
                role: "assistant" as const,
                content: `请求失败：${action.error}`,
                intent: "error",
                loading: false,
              }
            : m
        ),
        chatState: "error",
        error: action.error,
      };

    case "REMOVE_LAST_PAIR": {
      // 移除最后一组 user+assistant 消息对
      const msgs = [...state.messages];
      if (msgs.length >= 2) {
        msgs.splice(-2, 2);
      }
      return { ...state, messages: msgs, chatState: "idle", error: null };
    }

    case "CLEAR":
      return { messages: [], chatState: "idle", error: null };

    default:
      return state;
  }
}

// ---------- Hook ----------
export function useChatHistory() {
  const [state, dispatch] = useReducer(reducer, {
    messages: [],
    chatState: "idle" as ChatState,
    error: null,
  });

  const initialized = useRef(false);

  // 初始化：从 localStorage 加载历史
  useEffect(() => {
    if (!initialized.current) {
      initialized.current = true;
      const saved = loadFromStorage();
      if (saved.length > 0) {
        dispatch({ type: "INIT", messages: saved });
      }
    }
  }, []);

  // 每次 messages 变化时同步到 localStorage
  useEffect(() => {
    if (initialized.current) {
      saveToStorage(state.messages);
    }
  }, [state.messages]);

  // 从 messages 提取最近 6 条作为 history
  const getHistory = useCallback((): HistoryMessage[] => {
    const completed = state.messages.filter((m) => !m.loading && m.content && m.intent !== "error");
    const recent = completed.slice(-6);
    return recent.map((m) => ({ role: m.role, content: m.content }));
  }, [state.messages]);

  const send = useCallback(
    async (question: string) => {
      if (state.chatState === "sending" || state.chatState === "thinking") {
        return;
      }

      const ts = Date.now();
      const userMsg: Message = {
        id: `user-${ts}`,
        role: "user",
        content: question,
      };
      const loadingMsg: Message = {
        id: `assistant-${ts}`,
        role: "assistant",
        content: "",
        loading: true,
      };

      dispatch({ type: "SEND_START", userMsg, loadingMsg });
      await Promise.resolve();
      dispatch({ type: "THINKING" });

      const history = getHistory();

      await sendMessageStream(
        question,
        {
          onToken: (text) => {
            dispatch({ type: "STREAM_TOKEN", loadingId: loadingMsg.id, token: text });
          },
          onThought: (step) => {
            dispatch({ type: "STREAM_THOUGHT", loadingId: loadingMsg.id, step });
          },
          onChart: (data) => {
            dispatch({ type: "STREAM_CHART", loadingId: loadingMsg.id, chart: data });
          },
          onMeta: (meta) => {
            dispatch({ type: "STREAM_META", loadingId: loadingMsg.id, meta });
          },
          onDone: (data) => {
            dispatch({ type: "STREAM_DONE", loadingId: loadingMsg.id, steps: data.steps || [] });
          },
          onError: (error) => {
            dispatch({ type: "ERROR", loadingId: loadingMsg.id, error });
          },
        },
        history
      );
    },
    [state.chatState, getHistory]
  );

  // 用 ref 保存待重试问题，避免闭包竞态
  const pendingRetryRef = useRef<string | null>(null);

  // 监听 state 变化，当 REMOVE_LAST_PAIR 完成且 chatState=idle 时触发重发
  useEffect(() => {
    if (pendingRetryRef.current && state.chatState === "idle") {
      const question = pendingRetryRef.current;
      pendingRetryRef.current = null;
      send(question);
    }
  }, [state.chatState, state.messages, send]);

  const retry = useCallback(
    (userMsgId: string) => {
      // 先捕获问题内容，再 dispatch
      const userMsg = state.messages.find((m) => m.id === userMsgId);
      if (!userMsg) return;
      pendingRetryRef.current = userMsg.content;
      dispatch({ type: "REMOVE_LAST_PAIR" });
    },
    [state.messages]
  );

  const clearHistory = useCallback(() => {
    clearStorage();
    dispatch({ type: "CLEAR" });
  }, []);

  return {
    messages: state.messages,
    chatState: state.chatState,
    error: state.error,
    send,
    retry,
    clearHistory,
    isProcessing: state.chatState === "sending" || state.chatState === "thinking",
  };
}
