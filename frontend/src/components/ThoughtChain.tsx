"use client";

import { useState, useEffect } from "react";
import { ThoughtStep } from "@/types/chat";
import { ChevronDown, ChevronRight, Brain } from "lucide-react";

interface Props {
  steps: ThoughtStep[];
  streaming?: boolean;
}

export default function ThoughtChain({ steps, streaming = false }: Props) {
  const [expanded, setExpanded] = useState(streaming);

  // 流式模式下默认展开，完成后延迟 1s 自动收起
  useEffect(() => {
    if (streaming) {
      setExpanded(true);
    } else if (steps.length > 0) {
      const timer = setTimeout(() => setExpanded(false), 1000);
      return () => clearTimeout(timer);
    }
  }, [streaming, steps.length]);

  if (!steps || steps.length === 0) return null;

  return (
    <div className="mt-2">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
      >
        <Brain size={12} />
        <span>推理过程 ({steps.length} 步)</span>
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>

      {expanded && (
        <div className="mt-2 ml-1 border-l-2 border-gray-700/50 pl-3 space-y-2">
          {steps.map((step, i) => (
            <div key={i} className="relative">
              {/* 连接点 */}
              <div className="absolute -left-[17px] top-1.5 w-2 h-2 rounded-full bg-gray-600" />

              <div className="text-xs">
                <span className="text-brand-500 font-medium">{step.step}</span>
                <span className="text-gray-500 ml-2">{step.result}</span>
              </div>
            </div>
          ))}

          {streaming && (
            <div className="relative">
              <div className="absolute -left-[17px] top-1.5 w-2 h-2 rounded-full bg-brand-500 animate-pulse" />
              <div className="text-xs text-gray-500">思考中...</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
