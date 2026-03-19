"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { Send } from "lucide-react";

interface Props {
  onSend: (message: string) => void;
  disabled: boolean;
}

export default function InputBar({ onSend, disabled }: Props) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = Math.min(el.scrollHeight, 150) + "px";
    }
  };

  return (
    <div className="border-t border-gray-800 bg-brand-950/95 backdrop-blur-sm p-4">
      <div className="max-w-3xl mx-auto flex items-end gap-3">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              handleInput();
            }}
            onKeyDown={handleKeyDown}
            placeholder="输入金融问题，如：阿里巴巴当前股价？什么是市盈率？"
            disabled={disabled}
            rows={1}
            className="w-full bg-gray-800/60 border border-gray-700 rounded-xl px-4 py-3 pr-12
                       text-gray-100 placeholder:text-gray-500 resize-none
                       focus:outline-none focus:ring-2 focus:ring-brand-500/50 focus:border-brand-500
                       disabled:opacity-50 transition-all"
          />
          <button
            onClick={handleSend}
            disabled={disabled || !input.trim()}
            className="absolute right-2 bottom-2 p-2 rounded-lg
                       bg-brand-600 hover:bg-brand-700
                       disabled:bg-gray-700 disabled:cursor-not-allowed
                       transition-colors"
          >
            <Send size={16} className="text-white" />
          </button>
        </div>
      </div>
      <p className="text-center text-[11px] text-gray-600 mt-2">
        数据来源：Yahoo Finance | AI 分析仅供参考，不构成投资建议
      </p>
    </div>
  );
}
