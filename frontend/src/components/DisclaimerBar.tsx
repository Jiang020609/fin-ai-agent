"use client";

import { AlertTriangle } from "lucide-react";

interface Props {
  text: string;
}

export default function DisclaimerBar({ text }: Props) {
  return (
    <div className="flex items-start gap-1.5 mt-2 px-2.5 py-1.5 rounded-lg bg-yellow-500/5 border border-yellow-500/10 text-[11px] text-yellow-600/80">
      <AlertTriangle size={12} className="mt-0.5 flex-shrink-0" />
      <span>{text}</span>
    </div>
  );
}
