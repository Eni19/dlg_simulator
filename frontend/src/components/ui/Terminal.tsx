import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";
import { Terminal as TerminalIcon } from "lucide-react";

export interface LogLine {
  id: string;
  text: string;
  type: "info" | "metric" | "warn" | "success" | "error";
  timestamp?: number;
}

interface TerminalProps {
  logs: LogLine[];
  className?: string;
  maxLines?: number;
  title?: string;
  isRunning?: boolean;
}

/**
 * Terminal de Logs Interativo — simula um console ao vivo
 * para exibição do progresso das iterações do ataque DLG.
 */
export function Terminal({
  logs,
  className,
  maxLines = 25,
  title = "DLG Attack Logs",
  isRunning = false,
}: TerminalProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [visibleLogs, setVisibleLogs] = useState<LogLine[]>([]);

  // Auto-scroll para o final
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  // Limitar número de linhas visíveis
  useEffect(() => {
    setVisibleLogs(logs.slice(-maxLines));
  }, [logs, maxLines]);

  const lineColor = (type: LogLine["type"]) => {
    switch (type) {
      case "metric":
        return "text-emerald-400";
      case "warn":
        return "text-amber-400";
      case "success":
        return "text-emerald-300";
      case "error":
        return "text-red-400";
      default:
        return "text-graphite-300";
    }
  };

  return (
    <div
      className={cn(
        "overflow-hidden rounded-2xl border border-graphite-800 bg-[#0a0d11] shadow-2xl",
        className
      )}
    >
      {/* Header do Terminal */}
      <div className="flex items-center justify-between border-b border-graphite-800/50 px-5 py-3">
        <div className="flex items-center gap-2.5">
          <div className="flex gap-1.5">
            <div className="h-2.5 w-2.5 rounded-full bg-red-500/80" />
            <div className="h-2.5 w-2.5 rounded-full bg-amber-500/80" />
            <div className="h-2.5 w-2.5 rounded-full bg-emerald-500/80" />
          </div>
          <TerminalIcon className="ml-2 h-4 w-4 text-graphite-500" />
          <span className="text-xs font-medium text-graphite-400">{title}</span>
        </div>
        {isRunning && (
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
            <span className="text-[11px] font-medium text-emerald-500">RUNNING</span>
          </div>
        )}
      </div>

      {/* Corpo do Terminal */}
      <div
        ref={scrollRef}
        className="h-72 overflow-y-auto p-4 font-mono text-xs leading-relaxed"
      >
        {visibleLogs.length === 0 && (
          <div className="flex h-full items-center justify-center">
            <p className="animate-pulse text-graphite-600">
              Aguardando início do ataque...
            </p>
          </div>
        )}
        {visibleLogs.map((line) => (
          <div
            key={line.id}
            className={cn(
              "flex gap-2 transition-opacity duration-300",
              lineColor(line.type)
            )}
          >
            <span className="select-none text-graphite-700">›</span>
            <span className="whitespace-pre-wrap break-all">{line.text}</span>
          </div>
        ))}
      </div>

      {/* Footer — barra de status */}
      <div className="border-t border-graphite-800/50 px-5 py-2">
        <p className="text-[10px] text-graphite-600">
          {logs.length} lines · {isRunning ? "Streaming" : "Complete"}
        </p>
      </div>
    </div>
  );
}


