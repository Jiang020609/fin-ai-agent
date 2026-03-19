"use client";

import { Message } from "@/types/chat";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import FinancialChart from "./FinancialChart";
import { Bot, User, BookOpen } from "lucide-react";

interface Props {
  message: Message;
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

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";

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
          {message.loading ? (
            <LoadingDots />
          ) : (
            <div className={isUser ? "" : "prose"}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Chart card */}
        {!isUser && message.chartData && message.ticker && (
          <div className="w-full mt-1">
            <FinancialChart data={message.chartData} ticker={message.ticker} />
          </div>
        )}

        {/* RAG badge */}
        {!isUser && message.ragUsed && (
          <div className="flex items-center gap-1.5 mt-2 text-xs text-gray-500">
            <BookOpen size={12} />
            <span>基于知识库检索生成</span>
          </div>
        )}

        {/* Intent tag */}
        {!isUser && message.intent && !message.loading && (
          <div className="mt-1.5">
            <span
              className={`inline-block text-[10px] px-2 py-0.5 rounded-full font-medium ${
                message.intent === "market"
                  ? "bg-yellow-500/10 text-yellow-500"
                  : message.intent === "knowledge"
                  ? "bg-purple-500/10 text-purple-400"
                  : "bg-gray-700/50 text-gray-400"
              }`}
            >
              {message.intent === "market"
                ? "行情分析"
                : message.intent === "knowledge"
                ? "知识问答"
                : "通用"}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
