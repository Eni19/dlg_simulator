import { Card, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { Terminal, type LogLine } from "@/components/ui/Terminal";
import type { AttackConfig } from "@/lib/types";
import { Play, Square, Shield, Zap } from "lucide-react";

interface ExecutionPanelProps {
  config: AttackConfig;
  isRunning: boolean;
  logs: LogLine[];
  progress: number; // 0–100
  onStart: () => void;
  onAbort: () => void;
}

/**
 * Step 2 — Execução e Logs em Tempo Real.
 * Botão de ação principal + terminal de logs interativo.
 */
export function ExecutionPanel({
  config,
  isRunning,
  logs,
  progress,
  onStart,
  onAbort,
}: ExecutionPanelProps) {
  const envLabel =
    config.environment === "federated" ? "Federado (Flower)" : "Centralizado";
  const defenseLabel =
    config.defense === "none" ? "Sem defesa" : `Defesa: ${config.defense}`;

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Botão de Ação Principal */}
      <Card variant="highlight">
        <div className="flex flex-col items-center gap-5 py-4 sm:flex-row sm:justify-between">
          <div className="text-center sm:text-left">
            <h3 className="text-lg font-bold text-white">
              {isRunning ? "Ataque em Execução" : "Pronto para Iniciar"}
            </h3>
            <p className="mt-1 text-sm text-graphite-400">
              {envLabel} · Batch {config.batchSize} · {config.numIters}{" "}
              iterações · {defenseLabel}
            </p>
          </div>

          <div className="flex items-center gap-3">
            {isRunning ? (
              <Button
                variant="danger"
                size="lg"
                icon={<Square className="h-4 w-4" />}
                onClick={onAbort}
              >
                Cancelar espera
              </Button>
            ) : (
              <Button
                variant="primary"
                size="lg"
                icon={<Play className="h-4 w-4" />}
                onClick={onStart}
              >
                Iniciar Ataque DLG
              </Button>
            )}
          </div>
        </div>

        {/* Barra de Progresso */}
        {isRunning && (
          <div className="mt-2">
            <ProgressBar
              value={progress}
              size="lg"
              label="Progresso estimado no navegador"
            />
            <p className="mt-2 text-center text-xs text-graphite-500">
              Estimativa local, sem telemetria de iterações. Cancelar a espera
              não garante interromper o servidor.
            </p>
          </div>
        )}
      </Card>

      {/* Badges de Status */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[
          {
            label: "Ambiente",
            value: envLabel,
            icon: Zap,
          },
          {
            label: "Batch Size",
            value: `${config.batchSize} imagens`,
            icon: Zap,
          },
          {
            label: "Iterações",
            value: String(config.numIters),
            icon: Zap,
          },
          {
            label: "Defesa",
            value: defenseLabel,
            icon: Shield,
          },
        ].map((stat) => (
          <div
            key={stat.label}
            className="rounded-xl border border-graphite-800/60 bg-graphite-900/40 p-4 text-center"
          >
            <stat.icon className="mx-auto mb-1.5 h-4 w-4 text-graphite-500" />
            <p className="text-[10px] font-medium  text-graphite-500">
              {stat.label}
            </p>
            <p className="mt-0.5 text-sm font-semibold text-white">
              {stat.value}
            </p>
          </div>
        ))}
      </div>

      <p className="text-sm text-graphite-400">
        As etapas explicativas abaixo ilustram o algoritmo. Apenas a resposta
        final confirma o resultado da execução.
      </p>
      {/* Terminal de Logs */}
      <Terminal
        logs={logs}
        isRunning={isRunning}
        title={`DLG Attack · ${envLabel}`}
        maxLines={30}
      />
    </div>
  );
}
