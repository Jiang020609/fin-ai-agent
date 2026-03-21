"use client";

import { useState, useEffect, useCallback } from "react";

export type ColorScheme = "cn" | "us";

export interface ColorConfig {
  up: string;
  down: string;
  upBg: string;
  downBg: string;
}

const STORAGE_KEY = "finai_color_scheme";

const COLOR_CONFIGS: Record<ColorScheme, ColorConfig> = {
  cn: {
    up: "#ef4444",    // 红涨
    down: "#22c55e",  // 绿跌
    upBg: "rgba(239,68,68,0.1)",
    downBg: "rgba(34,197,94,0.1)",
  },
  us: {
    up: "#22c55e",    // 绿涨
    down: "#ef4444",  // 红跌
    upBg: "rgba(34,197,94,0.1)",
    downBg: "rgba(239,68,68,0.1)",
  },
};

export function useColorScheme() {
  const [scheme, setScheme] = useState<ColorScheme>("cn");

  useEffect(() => {
    if (typeof window === "undefined") return;
    const saved = localStorage.getItem(STORAGE_KEY) as ColorScheme | null;
    if (saved === "cn" || saved === "us") {
      setScheme(saved);
    }
  }, []);

  const toggle = useCallback(() => {
    setScheme((prev) => {
      const next = prev === "cn" ? "us" : "cn";
      localStorage.setItem(STORAGE_KEY, next);
      return next;
    });
  }, []);

  return {
    scheme,
    colors: COLOR_CONFIGS[scheme],
    toggle,
    label: scheme === "cn" ? "红涨绿跌" : "绿涨红跌",
  };
}
