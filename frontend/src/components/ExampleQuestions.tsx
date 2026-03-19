"use client";

import { TrendingUp, BookOpen, BarChart3 } from "lucide-react";

const EXAMPLES = [
  { icon: TrendingUp, text: "特斯拉近期走势如何？", color: "text-yellow-500" },
  { icon: BarChart3, text: "阿里巴巴当前股价是多少？", color: "text-blue-400" },
  { icon: BookOpen, text: "什么是市盈率？", color: "text-purple-400" },
  { icon: BookOpen, text: "收入和净利润的区别是什么？", color: "text-green-400" },
];

interface Props {
  onSelect: (question: string) => void;
}

export default function ExampleQuestions({ onSelect }: Props) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 px-4">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold bg-gradient-to-r from-brand-500 to-blue-400 bg-clip-text text-transparent">
          FinAI
        </h1>
        <p className="text-gray-500 mt-2 text-sm">
          智能金融问答 — 实时行情 + 知识检索
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.text}
            onClick={() => onSelect(ex.text)}
            className="flex items-center gap-3 px-4 py-3 rounded-xl
                       bg-gray-800/40 border border-gray-700/50
                       hover:bg-gray-800/80 hover:border-gray-600
                       transition-all text-left group"
          >
            <ex.icon
              size={18}
              className={`${ex.color} flex-shrink-0 group-hover:scale-110 transition-transform`}
            />
            <span className="text-sm text-gray-300 group-hover:text-gray-100">
              {ex.text}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
