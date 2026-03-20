"use client";

import { TrendSummary } from "@/types/chat";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface Props {
  trend: TrendSummary;
}

export default function TrendBadge({ trend }: Props) {
  if (!trend.label || trend.label === "unknown") return null;

  const config = {
    uptrend: {
      icon: TrendingUp,
      bg: "bg-red-500/10",
      text: "text-red-400",
      border: "border-red-500/20",
    },
    downtrend: {
      icon: TrendingDown,
      bg: "bg-green-500/10",
      text: "text-green-400",
      border: "border-green-500/20",
    },
    sideways: {
      icon: Minus,
      bg: "bg-yellow-500/10",
      text: "text-yellow-400",
      border: "border-yellow-500/20",
    },
  }[trend.label] || {
    icon: Minus,
    bg: "bg-gray-500/10",
    text: "text-gray-400",
    border: "border-gray-500/20",
  };

  const Icon = config.icon;

  return (
    <div className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs ${config.bg} ${config.text} border ${config.border}`}>
      <Icon size={12} />
      <span className="font-medium">{trend.label_cn}</span>
      {trend.confidence && trend.confidence !== "high" && (
        <span className="text-[10px] opacity-60">
          ({trend.confidence === "low" ? "弱" : "中"})
        </span>
      )}
      {trend.rationale && (
        <span className="text-[10px] opacity-70 ml-0.5">{trend.rationale}</span>
      )}
    </div>
  );
}
