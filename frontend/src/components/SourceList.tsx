"use client";

import { SourceItem } from "@/types/chat";
import { ExternalLink, Database, Globe, BookOpen } from "lucide-react";

interface Props {
  sources: SourceItem[];
}

function sourceIcon(source: string) {
  const s = source.toLowerCase();
  if (s.includes("web") || s.includes("search")) return Globe;
  if (s.includes("知识库") || s.includes("rag")) return BookOpen;
  return Database;
}

function relevanceColor(score: number): string {
  if (score >= 0.7) return "text-green-400";
  if (score >= 0.4) return "text-yellow-400";
  return "text-gray-500";
}

function relevanceDot(score: number): string {
  if (score >= 0.7) return "bg-green-400";
  if (score >= 0.4) return "bg-yellow-400";
  return "bg-gray-500";
}

export default function SourceList({ sources }: Props) {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-2 space-y-1">
      <div className="text-[10px] text-gray-500 font-medium uppercase tracking-wider">
        来源
      </div>
      {sources.map((src, i) => {
        const Icon = sourceIcon(src.source);
        return (
          <div
            key={i}
            className="flex items-center gap-2 text-xs text-gray-400 bg-gray-800/40 rounded-lg px-2.5 py-1.5"
          >
            <Icon size={12} className="flex-shrink-0 text-gray-500" />
            <span className="text-gray-300 truncate">{src.title || src.source}</span>
            {src.page && src.page !== "-" && (
              <span className="flex-shrink-0 px-1.5 py-0.5 rounded bg-gray-700/60 text-[10px] text-gray-400 font-mono">
                p.{src.page}
              </span>
            )}
            {src.relevance_score != null && (
              <span className={`flex-shrink-0 flex items-center gap-1 text-[10px] ${relevanceColor(src.relevance_score)}`}>
                <span className={`inline-block w-1.5 h-1.5 rounded-full ${relevanceDot(src.relevance_score)}`} />
                {src.relevance_score.toFixed(2)}
              </span>
            )}
            {src.url && (
              <a
                href={src.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-shrink-0 inline-flex items-center text-brand-500 hover:text-brand-400 transition-colors"
              >
                <ExternalLink size={10} />
              </a>
            )}
            {src.published_at && (
              <span className="flex-shrink-0 text-gray-600">{src.published_at}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}
