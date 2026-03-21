"use client";

import { EvidenceAnalysis as EvidenceAnalysisType } from "@/types/chat";
import { Zap, ChevronRight, AlertTriangle } from "lucide-react";

interface Props {
  evidence: EvidenceAnalysisType;
}

const STRENGTH_CONFIG: Record<string, { label: string; color: string; bg: string }> = {
  strong: { label: "证据充分", color: "text-green-400", bg: "bg-green-500/10" },
  moderate: { label: "证据中等", color: "text-yellow-400", bg: "bg-yellow-500/10" },
  weak: { label: "证据不足", color: "text-orange-400", bg: "bg-orange-500/10" },
};

export default function EvidenceAnalysis({ evidence }: Props) {
  if (!evidence || (!evidence.main_drivers?.length && !evidence.summary)) return null;

  const strength = STRENGTH_CONFIG[evidence.evidence_strength] || STRENGTH_CONFIG.weak;

  return (
    <div className="bg-gray-900/80 border border-gray-700/50 rounded-xl p-4 mt-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Zap size={14} className="text-orange-400" />
          <span className="text-sm font-semibold text-gray-200">证据分析</span>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${strength.color} ${strength.bg}`}>
          {strength.label}
        </span>
      </div>

      {evidence.summary && (
        <p className="text-xs text-gray-400 mb-3 leading-relaxed">{evidence.summary}</p>
      )}

      {evidence.main_drivers && evidence.main_drivers.length > 0 && (
        <div className="mb-2">
          <div className="text-[10px] text-gray-500 mb-1.5 uppercase tracking-wider">主要驱动因素</div>
          <div className="flex flex-wrap gap-1.5">
            {evidence.main_drivers.map((d, i) => (
              <span key={i} className="inline-flex items-center gap-1 text-xs bg-orange-500/10 text-orange-300 px-2 py-1 rounded-md">
                <ChevronRight size={10} />
                {d}
              </span>
            ))}
          </div>
        </div>
      )}

      {evidence.secondary_drivers && evidence.secondary_drivers.length > 0 && (
        <div>
          <div className="text-[10px] text-gray-500 mb-1.5 uppercase tracking-wider">次要因素</div>
          <div className="flex flex-wrap gap-1.5">
            {evidence.secondary_drivers.map((d, i) => (
              <span key={i} className="inline-flex items-center gap-1 text-xs bg-gray-700/50 text-gray-400 px-2 py-1 rounded-md">
                {d}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
