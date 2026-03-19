/**
 * useChatHistory — 基于状态机的聊天状态管理 Hook
 *
 * 深度加固：
 * - localStorage 持久化：刷新页面不丢失对话历史
 * - clearHistory：清空历史记录（同时清除 localStorage）
 * - 状态机确保 UI 状态转换严丝合缝
 *
 *   idle ──(send)──> sending ──> thinking ──> success ──> idle
 *                       │            │
 *                       └────> error ─┘──(重试)──> idle
 */

"use client";

import { useCallback, useReducer, useEffect, useRef } from "react";
import { Message, ChatState, ChatResponse } from "@/types/chat";
import { sendMessage } from "./api";

// ---------- localStorage 持久化 ----------
const STORAGE_KEY = "finai_chat_history";

function loadFromStorage(): Message[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Message[];
    // 过滤掉残留的 loading 状态消息（上次未完成的请求）
    return parsed.filter((m) => !m.loading);
  } catch {
    return [];
  }
}

function saveToStorage(messages: Message[]) {
  if (typeof window === "undefined") return;
  try {
    // 只持久化已完成的消息，跳过 loading 占位
    const toSave = messages.filter((m) => !m.loading);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
  } catch {
    // localStorage 满或不可用，静默失败
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
  | { type: "SUCCESS"; loadingId: string; response: ChatResponse }
  | { type: "ERROR"; loadingId: string; error: string }
  | { type: "CLEAR" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    // 初始化：从 localStorage 恢复
    case "INIT":
      return { ...state, messages: action.messages };

    // idle → sending
    case "SEND_START":
      return {
        ...state,
        messages: [...state.messages, action.userMsg, action.loadingMsg],
        chatState: "sending",
        error: null,
      };

    // sending → thinking
    case "THINKING":
      return { ...state, chatState: "thinking" };

    // thinking → idle
    case "SUCCESS":
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.loadingId
            ? {
                id: m.id,
                role: "assistant" as const,
                content: action.response.text_response,
                chartData: action.response.chart_data,
                ticker: action.response.ticker,
                intent: action.response.intent,
                ragUsed: action.response.rag_used,
                steps: action.response.steps,
              }
            : m
        ),
        chatState: "idle",
        error: null,
      };

    // → error
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
              }
            : m
        ),
        chatState: "error",
        error: action.error,
      };

    // 清空历史
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

  // 初始化：从 localStorage 加载历史（仅客户端，仅一次）
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

      try {
        const response = await sendMessage(question);
        dispatch({ type: "SUCCESS", loadingId: loadingMsg.id, response });
      } catch (err) {
        dispatch({
          type: "ERROR",
          loadingId: loadingMsg.id,
          error: err instanceof Error ? err.message : "未知错误",
        });
      }
    },
    [state.chatState]
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
    clearHistory,
    isProcessing: state.chatState === "sending" || state.chatState === "thinking",
  };
}
