"use client";

import { ComparisonResult } from "@/types/chat";
import { Trophy, TrendingUp, TrendingDown, Activity, Shield, BarChart3 } from "lucide-react";

interface Props {
  comparison: ComparisonResult;
}

function fmt(val: number | null, suffix = "", decimals = 2): string {
  if (val === null || val === undefined) return "--";
  const sign = val > 0 ? "+" : "";
  return `${sign}${val.toFixed(decimals)}${suffix}`;
}

function fmtPrice(val: number | null, currency: string): string {
  if (val === null || val === undefined) return "--";
  const sym = currency === "USD" ? "$" : currency === "HKD" ? "HK$" : "";
  return `${sym}${val.toFixed(2)}`;
}

const METRIC_ROWS: {
  key: string;
  label: string;
  icon: typeof TrendingUp;
  format: (a: Record<string, unknown>) => string;
  winnerKey?: string;
  better?: "higher" | "lower";
}[] = [
  {
    key: "current_price",
    label: "当前价格",
    icon: BarChart3,
    format: (a) => fmtPrice(a.current_price as number | null, (a.currency as string) || "USD"),
  },
  {
    key: "return_7d",
    label: "7日收益",
    icon: TrendingUp,
    format: (a) => fmt(a.return_7d as number | null, "%"),
    winnerKey: "return_7d",
    better: "higher",
  },
  {
    key: "return_30d",
    label: "30日收益",
    icon: TrendingUp,
    format: (a) => fmt(a.return_30d as number | null, "%"),
    winnerKey: "return_30d",
    better: "higher",
  },
  {
    key: "volatility_30d",
    label: "波动率(年化)",
    icon: Activity,
    format: (a) => fmt(a.volatility_30d as number | null, "%", 2),
    winnerKey: "volatility_lower",
    better: "lower",
  },
  {
    key: "sharpe_30d",
    label: "夏普比率",
    icon: Shield,
    format: (a) => {
      const v = a.sharpe_30d as number | null;
      return v !== null && v !== undefined ? v.toFixed(2) : "--";
    },
    winnerKey: "sharpe_best",
    better: "higher",
  },
  {
    key: "max_drawdown_30d",
    label: "最大回撤",
    icon: TrendingDown,
    format: (a) => {
      const v = a.max_drawdown_30d as number | null;
      return v !== null && v !== undefined ? `${(v * 100).toFixed(2)}%` : "--";
    },
    winnerKey: "max_drawdown_least",
    better: "higher", // Less negative = better, max() picks the winner
  },
];

export default function ComparisonTable({ comparison }: Props) {
  const { assets, winners, assumptions } = comparison;
  if (!assets || assets.length === 0) return null;

  return (
    <div className="bg-gray-900/80 border border-gray-700/50 rounded-xl p-4 mt-3 overflow-x-auto">
      <div className="flex items-center gap-2 mb-3">
        <Trophy size={14} className="text-yellow-500" />
        <span className="text-sm font-semibold text-gray-200">多资产对比</span>
      </div>

      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="text-left text-gray-500 text-xs py-2 pr-3">指标</th>
            {assets.map((a) => (
              <th key={a.ticker} className="text-right text-xs py-2 px-2">
                <span className="bg-brand-600/20 text-brand-500 font-mono font-bold px-1.5 py-0.5 rounded text-[10px]">
                  {a.ticker}
                </span>
                <div className="text-gray-500 font-normal mt-0.5">{a.name}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRIC_ROWS.map((row) => {
            const winnerTicker = row.winnerKey && winners[row.winnerKey]?.ticker;
            return (
              <tr key={row.key} className="border-b border-gray-800/50">
                <td className="py-2 pr-3">
                  <div className="flex items-center gap-1.5 text-gray-400">
                    <row.icon size={12} />
                    <span className="text-xs">{row.label}</span>
                  </div>
                </td>
                {assets.map((a) => {
                  const isWinner = winnerTicker === a.ticker;
                  const val = a[row.key as keyof typeof a] as number | null;
                  const isPositive = typeof val === "number" && val > 0;
                  const isNegative = typeof val === "number" && val < 0;
                  const colorClass =
                    row.key.includes("return") || row.key === "sharpe_30d"
                      ? isPositive
                        ? "text-up"
                        : isNegative
                        ? "text-down"
                        : "text-gray-300"
                      : row.key === "max_drawdown_30d"
                      ? "text-orange-400"
                      : "text-gray-200";

                  return (
                    <td key={a.ticker} className="text-right py-2 px-2">
                      <span className={`text-xs font-medium ${colorClass}`}>
                        {row.format(a as unknown as Record<string, unknown>)}
                      </span>
                      {isWinner && (
                        <Trophy size={10} className="inline ml-1 text-yellow-500" />
                      )}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>

      {assumptions && (
        <p className="text-[10px] text-gray-600 mt-2">
          {assumptions.risk_free_rate_note} | {assumptions.max_drawdown_note}
        </p>
      )}
    </div>
  );
}
