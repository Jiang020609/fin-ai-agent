"use client";

import { Message } from "@/types/chat";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import FinancialChart from "./FinancialChart";
import ThoughtChain from "./ThoughtChain";
import KPICards from "./KPICards";
import SourceList from "./SourceList";
import DisclaimerBar from "./DisclaimerBar";
import TrendBadge from "./TrendBadge";
import { Bot, User, BookOpen, RotateCcw, Search } from "lucide-react";
import { ColorConfig } from "@/lib/useColorScheme";

interface Props {
  message: Message;
  onRetry?: (userMsgId: string) => void;
  pairedUserMsgId?: string;
  colorConfig?: ColorConfig;
}

function LoadingDots() {
  return (
    <div className="flex items-center gap-1.5 py-2 px-1">
      <div className="loading-dot w-2 h-2 rounded-full bg-brand-500" />
      <div className="loading-dot w-2 h-2 rounded-full bg-brand-500" />
      <div className="loading-dot w-2 h-2 rounded-full bg-brand-500" />
    </div>
  );
}

function BlinkingCursor() {
  return (
    <span className="inline-block w-0.5 h-4 bg-brand-500 streaming-cursor ml-0.5 align-middle" />
  );
}

/** 意图标签配置 */
function intentTagConfig(intent: string) {
  switch (intent) {
    case "market_data":
      return { label: "行情数据", className: "bg-yellow-500/10 text-yellow-500" };
    case "market_reasoning":
      return { label: "原因分析", className: "bg-orange-500/10 text-orange-400" };
    case "knowledge_rag":
      return { label: "知识问答", className: "bg-purple-500/10 text-purple-400" };
    // 兼容旧 intent 名
    case "market":
      return { label: "行情分析", className: "bg-yellow-500/10 text-yellow-500" };
    case "knowledge":
      return { label: "知识问答", className: "bg-purple-500/10 text-purple-400" };
    default:
      return { label: "通用", className: "bg-gray-700/50 text-gray-400" };
  }
}

export default function MessageBubble({ message, onRetry, pairedUserMsgId, colorConfig }: Props) {
  const isUser = message.role === "user";
  const isStreaming = !isUser && message.loading && message.content.length > 0;
  const isWaiting = !isUser && message.loading && message.content.length === 0 && !(message.steps && message.steps.length > 0);
  const isError = message.intent === "error";

  const sr = message.structuredResponse;
  const hasSources = sr && sr.sources && sr.sources.length > 0;
  const hasDisclaimer = sr && sr.disclaimer;
  const hasTrend = sr && sr.trend_summary && sr.trend_summary.label && sr.trend_summary.label !== "unknown";

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      {/* Avatar */}
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
          isUser
            ? "bg-brand-600"
            : "bg-gradient-to-br from-gray-700 to-gray-800 ring-1 ring-gray-600"
        }`}
      >
        {isUser ? (
          <User size={16} className="text-white" />
        ) : (
          <Bot size={16} className="text-brand-500" />
        )}
      </div>

      {/* Content */}
      <div className={`flex-1 max-w-[85%] ${isUser ? "flex flex-col items-end" : ""}`}>
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? "bg-brand-600 text-white rounded-tr-sm"
              : "bg-gray-800/80 border border-gray-700/50 rounded-tl-sm"
          }`}
        >
          {isWaiting ? (
            <LoadingDots />
          ) : message.content ? (
            <div className={isUser ? "" : "prose"}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
              {isStreaming && <BlinkingCursor />}
            </div>
          ) : message.loading && message.steps && message.steps.length > 0 ? (
            <div className="text-xs text-gray-500">正在分析...</div>
          ) : null}
        </div>

        {/* KPI Cards */}
        {!isUser && message.marketMeta && message.marketMeta.current_price && (
          <div className="w-full">
            <KPICards meta={message.marketMeta} />
          </div>
        )}

        {/* Trend Badge */}
        {!isUser && !message.loading && hasTrend && (
          <div className="mt-2">
            <TrendBadge trend={sr!.trend_summary!} />
          </div>
        )}

        {/* ThoughtChain */}
        {!isUser && message.steps && message.steps.length > 0 && (
          <ThoughtChain steps={message.steps} streaming={!!message.loading} />
        )}

        {/* Chart */}
        {!isUser && message.chartData && message.ticker && (
          <div className="w-full mt-1">
            <FinancialChart data={message.chartData} ticker={message.ticker} colorConfig={colorConfig} />
          </div>
        )}

        {/* Source List */}
        {!isUser && !message.loading && hasSources && (
          <SourceList sources={sr!.sources} />
        )}

        {/* Disclaimer */}
        {!isUser && !message.loading && hasDisclaimer && (
          <DisclaimerBar text={sr!.disclaimer!} />
        )}

        {/* RAG badge — 仅在没有 SourceList 时显示旧样式 */}
        {!isUser && message.ragUsed && !hasSources && (
          <div className="flex items-center gap-1.5 mt-2 text-xs text-gray-500">
            <BookOpen size={12} />
            <span>基于知识库检索生成</span>
          </div>
        )}

        {/* Evidence badge for market_reasoning */}
        {!isUser && !message.loading && message.intent === "market_reasoning" && message.ragUsed && (
          <div className="flex items-center gap-1.5 mt-1 text-xs text-gray-500">
            <Search size={12} />
            <span>基于新闻证据分析</span>
          </div>
        )}

        {/* Error retry button */}
        {!isUser && isError && onRetry && pairedUserMsgId && (
          <button
            onClick={() => onRetry(pairedUserMsgId)}
            className="flex items-center gap-1.5 mt-2 text-xs text-brand-500 hover:text-brand-400 transition-colors"
          >
            <RotateCcw size={12} />
            <span>重试</span>
          </button>
        )}

        {/* Intent tag */}
        {!isUser && message.intent && !message.loading && message.intent !== "error" && (
          <div className="mt-1.5">
            <span
              className={`inline-block text-[10px] px-2 py-0.5 rounded-full font-medium ${
                intentTagConfig(message.intent).className
              }`}
            >
              {intentTagConfig(message.intent).label}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
