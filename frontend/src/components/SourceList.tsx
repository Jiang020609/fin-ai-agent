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
            className="flex items-start gap-2 text-xs text-gray-400 bg-gray-800/40 rounded-lg px-2.5 py-1.5"
          >
            <Icon size={12} className="mt-0.5 flex-shrink-0 text-gray-500" />
            <div className="min-w-0 flex-1">
              <span className="text-gray-300">{src.title || src.source}</span>
              {src.url && (
                <a
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-0.5 ml-1.5 text-brand-500 hover:text-brand-400 transition-colors"
                >
                  <ExternalLink size={10} />
                </a>
              )}
              {src.published_at && (
                <span className="ml-1.5 text-gray-600">{src.published_at}</span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
