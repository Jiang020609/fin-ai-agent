"use client";

import { useRef, useEffect } from "react";
import { useChatHistory } from "@/lib/useChatHistory";
import { useServerStatus } from "@/lib/useServerStatus";
import { useColorScheme } from "@/lib/useColorScheme";
import MessageBubble from "@/components/MessageBubble";
import InputBar from "@/components/InputBar";
import ExampleQuestions from "@/components/ExampleQuestions";
import { Palette } from "lucide-react";

export default function Home() {
  const { messages, isProcessing, send, retry, clearHistory } = useChatHistory();
  const serverStatus = useServerStatus();
  const { colors, toggle, label } = useColorScheme();
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }
  }, [messages]);

  function findPairedUserMsgId(assistantIndex: number): string | undefined {
    for (let i = assistantIndex - 1; i >= 0; i--) {
      if (messages[i].role === "user") return messages[i].id;
    }
    return undefined;
  }

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="border-b border-gray-800 bg-brand-950/90 backdrop-blur-sm px-4 py-3">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-blue-600 flex items-center justify-center">
              <span className="text-white font-bold text-sm">F</span>
            </div>
            <div>
              <h1 className="text-sm font-semibold text-gray-100">FinAI</h1>
              <p className="text-[11px] text-gray-500">智能金融问答系统</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* 涨跌配色切换 */}
            <button
              onClick={toggle}
              className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
              title="切换涨跌配色"
            >
              <Palette size={12} />
              <span>{label}</span>
            </button>

            {messages.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                清空记录
              </button>
            )}
            <div className="flex items-center gap-1.5">
              <span
                className={`w-2 h-2 rounded-full ${
                  serverStatus === "online"
                    ? "bg-green-500 animate-pulse"
                    : serverStatus === "offline"
                    ? "bg-red-500"
                    : "bg-yellow-500 animate-pulse"
                }`}
              />
              <span className="text-xs text-gray-500">
                {serverStatus === "online"
                  ? "在线"
                  : serverStatus === "offline"
                  ? "离线"
                  : "检测中"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto scrollbar-thin"
      >
        {messages.length === 0 ? (
          <ExampleQuestions onSelect={send} />
        ) : (
          <div className="max-w-3xl mx-auto py-6 px-4 space-y-6">
            {messages.map((msg, idx) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onRetry={retry}
                pairedUserMsgId={msg.role === "assistant" ? findPairedUserMsgId(idx) : undefined}
                colorConfig={colors}
              />
            ))}
          </div>
        )}
      </div>

      {/* Input */}
      <InputBar onSend={send} disabled={isProcessing} />
    </div>
  );
}
