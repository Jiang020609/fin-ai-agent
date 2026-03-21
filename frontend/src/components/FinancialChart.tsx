"use client";

import { ChartPoint } from "@/types/chat";
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
  CartesianGrid,
} from "recharts";
import { TrendingUp, TrendingDown } from "lucide-react";
import { ColorConfig } from "@/lib/useColorScheme";

interface Props {
  data: ChartPoint[];
  ticker: string;
  colorConfig?: ColorConfig;
}

const DEFAULT_COLORS: ColorConfig = {
  up: "#ef4444",
  down: "#22c55e",
  upBg: "rgba(239,68,68,0.1)",
  downBg: "rgba(34,197,94,0.1)",
};

export default function FinancialChart({ data, ticker, colorConfig }: Props) {
  if (!data || data.length === 0) return null;

  const colors = colorConfig || DEFAULT_COLORS;

  const first = data[0].close;
  const last = data[data.length - 1].close;
  const change = last - first;
  const changePct = ((change / first) * 100).toFixed(2);
  const isUp = change >= 0;

  const color = isUp ? colors.up : colors.down;

  const minClose = Math.min(...data.map((d) => d.close));
  const maxClose = Math.max(...data.map((d) => d.close));
  const padding = (maxClose - minClose) * 0.1 || 1;

  return (
    <div className="bg-gray-900/80 border border-gray-700/50 rounded-xl p-4 mt-3">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="bg-brand-600/20 text-brand-500 text-xs font-mono font-bold px-2 py-1 rounded">
            {ticker}
          </span>
          <span className="text-white font-semibold text-lg">
            ${last.toFixed(2)}
          </span>
        </div>
        <div
          className="flex items-center gap-1 text-sm font-medium"
          style={{ color }}
        >
          {isUp ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
          <span>
            {isUp ? "+" : ""}
            {change.toFixed(2)} ({isUp ? "+" : ""}
            {changePct}%)
          </span>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={200}>
        <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
          <defs>
            <linearGradient id={`grad-${ticker}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.3} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="rgba(255,255,255,0.05)"
          />
          <XAxis
            dataKey="date"
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickFormatter={(d: string) => d.slice(5)}
            axisLine={{ stroke: "#374151" }}
            tickLine={false}
          />
          <YAxis
            domain={[minClose - padding, maxClose + padding]}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            axisLine={false}
            tickLine={false}
            width={55}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
              color: "#f3f4f6",
              fontSize: "13px",
            }}
            formatter={(value) => [`$${Number(value).toFixed(2)}`, "收盘价"]}
            labelFormatter={(label) => `日期: ${label}`}
          />
          <Area
            type="monotone"
            dataKey="close"
            stroke={color}
            strokeWidth={2}
            fill={`url(#grad-${ticker})`}
          />
        </AreaChart>
      </ResponsiveContainer>

      {/* Footer stats */}
      <div className="flex flex-col sm:flex-row justify-between text-xs text-gray-500 mt-2 px-1 gap-1">
        <span>
          区间: {data[0].date} ~ {data[data.length - 1].date}
        </span>
        <span>
          最高 ${maxClose.toFixed(2)} / 最低 ${minClose.toFixed(2)}
        </span>
      </div>
    </div>
  );
}
