import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

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
 * Registro das etapas e contagens recebidas durante o experimento.
 */
export function Terminal({
  logs,
  className,
  maxLines = 25,
  title = "Etapas do experimento",
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

  return (
    <section className={cn("run-log", className)} aria-label={title}>
      <header className="run-log-header">
        <h3>{title}</h3>
        {isRunning && (
          <span className="run-log-live">
            <span className="status-dot running" />
            Em curso
          </span>
        )}
      </header>

      <div ref={scrollRef} className="run-log-body">
        {visibleLogs.length === 0 && (
          <div className="run-log-empty">
            <span className="hand-note">nada por aqui ainda</span>
            As etapas aparecerão aqui quando você iniciar o experimento.
          </div>
        )}
        {visibleLogs.map((line) => (
          <div key={line.id} className="run-log-line" data-type={line.type}>
            <span>›</span>
            <span className="whitespace-pre-wrap break-words min-w-0">{line.text}</span>
          </div>
        ))}
      </div>

      <footer className="run-log-footer">
        {logs.length} registros · {isRunning ? "Recebendo atualizações" : logs.length ? "Acompanhamento encerrado" : "Pronto para começar"}
      </footer>
    </section>
  );
}
