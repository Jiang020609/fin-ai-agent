"use client";

import { useState, useEffect, useCallback, useRef } from "react";

type Status = "online" | "offline" | "checking";

const HEALTH_URL = "/api/../health";  // 后端 /health 端点
const POLL_INTERVAL = 15000;          // 15s 轮询
const TIMEOUT_MS = 5000;              // 5s 超时判定离线

/**
 * 轮询后端 /health 端点，返回真实的服务可用状态。
 */
export function useServerStatus(): Status {
  const [status, setStatus] = useState<Status>("checking");
  const timer = useRef<ReturnType<typeof setInterval>>(undefined);

  const check = useCallback(async () => {
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

      const res = await fetch(HEALTH_URL, { signal: controller.signal });
      clearTimeout(timeout);

      setStatus(res.ok ? "online" : "offline");
    } catch {
      setStatus("offline");
    }
  }, []);

  useEffect(() => {
    check(); // 立即检查一次
    timer.current = setInterval(check, POLL_INTERVAL);
    return () => clearInterval(timer.current);
  }, [check]);

  return status;
}
