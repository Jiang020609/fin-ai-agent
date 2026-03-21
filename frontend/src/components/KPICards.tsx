"use client";

import { MarketMeta } from "@/types/chat";
import { DollarSign, TrendingUp, BarChart3, Building2 } from "lucide-react";

interface Props {
  meta: MarketMeta;
}

function formatMarketCap(cap: number | null): string {
  if (!cap) return "--";
  if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`;
  if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`;
  if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`;
  return `$${cap.toLocaleString()}`;
}

function formatPrice(price: number | null, currency: string): string {
  if (price === null || price === undefined) return "--";
  const symbol = currency === "USD" ? "$" : currency === "HKD" ? "HK$" : "";
  return `${symbol}${price.toFixed(2)}`;
}

export default function KPICards({ meta }: Props) {
  const cards = [
    {
      label: "当前价格",
      value: formatPrice(meta.current_price, meta.currency),
      icon: DollarSign,
      color: "text-brand-500",
    },
    {
      label: "7日涨跌",
      value: meta.change_pct !== null && meta.change_pct !== undefined
        ? `${meta.change_pct > 0 ? "+" : ""}${meta.change_pct.toFixed(2)}%`
        : "--",
      icon: TrendingUp,
      color: meta.change_pct !== null
        ? meta.change_pct > 0
          ? "text-up"
          : meta.change_pct < 0
          ? "text-down"
          : "text-gray-400"
        : "text-gray-400",
    },
    {
      label: "市盈率",
      value: meta.pe_ratio !== null && meta.pe_ratio !== undefined ? meta.pe_ratio.toFixed(2) : "--",
      icon: BarChart3,
      color: "text-purple-400",
    },
    {
      label: "市值",
      value: formatMarketCap(meta.market_cap),
      icon: Building2,
      color: "text-yellow-500",
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
      {cards.map(({ label, value, icon: Icon, color }) => (
        <div
          key={label}
          className="bg-gray-800/60 border border-gray-700/30 rounded-lg px-3 py-2"
        >
          <div className="flex items-center gap-1.5 mb-1">
            <Icon size={12} className={color} />
            <span className="text-[10px] text-gray-500">{label}</span>
          </div>
          <div className={`text-sm font-semibold ${color}`}>{value}</div>
        </div>
      ))}
    </div>
  );
}
