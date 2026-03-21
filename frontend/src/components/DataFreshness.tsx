"use client";

import { Clock } from "lucide-react";

interface Props {
  timestamp: string | null | undefined;
}

/**
 * 将 ISO 时间戳格式化为用户友好的 "YYYY-MM-DD HH:mm (UTC+8)" 格式。
 * 若输入为 null/undefined/非法，返回 null 不渲染。
 */
function formatTimestamp(raw: string | null | undefined): string | null {
  if (!raw) return null;
  try {
    const date = new Date(raw);
    if (isNaN(date.getTime())) return null;

    // 转为 UTC+8
    const utc8 = new Date(date.getTime() + 8 * 60 * 60 * 1000);
    const y = utc8.getUTCFullYear();
    const m = String(utc8.getUTCMonth() + 1).padStart(2, "0");
    const d = String(utc8.getUTCDate()).padStart(2, "0");
    const hh = String(utc8.getUTCHours()).padStart(2, "0");
    const mm = String(utc8.getUTCMinutes()).padStart(2, "0");
    return `${y}-${m}-${d} ${hh}:${mm} (UTC+8)`;
  } catch {
    return null;
  }
}

export default function DataFreshness({ timestamp }: Props) {
  const formatted = formatTimestamp(timestamp);
  if (!formatted) return null;

  return (
    <div className="flex items-center gap-1.5 mt-1.5 text-[10px] text-gray-500">
      <Clock size={10} className="flex-shrink-0" />
      <span>数据更新时间：{formatted}</span>
    </div>
  );
}
