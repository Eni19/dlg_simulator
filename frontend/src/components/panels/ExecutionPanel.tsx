import { Card, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { ProgressBar } from "@/components/ui/ProgressBar";
import { Terminal, type LogLine } from "@/components/ui/Terminal";
import type { AttackConfig } from "@/lib/types";
import { defenseLabel as describeDefense } from "@/lib/types";
import type { ExperimentProgress } from "@/lib/api";
import { Play, Square } from "lucide-react";

interface ExecutionPanelProps {
  config: AttackConfig;
  isRunning: boolean;
  logs: LogLine[];
  progress: number; // 0–100
  telemetry: ExperimentProgress | null;
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
  telemetry,
  onStart,
  onAbort,
}: ExecutionPanelProps) {
  const envLabel =
    config.environment === "federated" ? "Federado (Flower)" : "Centralizado";
  const defenseLabel =
    describeDefense(config.defense);

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Botão de Ação Principal */}
      <Card variant="highlight">
        <div className="flex flex-col items-center gap-5 py-4 sm:flex-row sm:justify-between">
          <div className="text-center sm:text-left">
            <h3 className="font-serif text-2xl font-medium text-white">
              {isRunning ? "A reconstrução está em andamento" : "Pronto para experimentar?"}
            </h3>
            <p className="mt-1 text-[15px] text-graphite-400">
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
                Iniciar experimento
              </Button>
            )}
          </div>
        </div>

        {/* Barra de Progresso */}
        {isRunning && (
          <div className="mt-2">
            {telemetry?.total != null && telemetry.current != null ? <ProgressBar
              value={progress}
              size="lg"
              label={telemetry.phase === "optimizing" ? "Ciclos DLG concluídos" : "Rodadas federadas concluídas"}
            /> : null}
            <p className="mt-2 text-sm" role="status" aria-live="polite">
              {telemetry?.text || "Conectando ao acompanhamento do experimento"}
              {telemetry?.total != null && telemetry.current != null
                ? ` · ${telemetry.current} de ${telemetry.total}` : ""}
              {telemetry?.loss != null ? ` · loss ${telemetry.loss.toExponential(3)}` : ""}
            </p>
            <p className="mt-2 text-center text-xs text-graphite-500">
              Contagem enviada pelo backend, sem estimativa de tempo restante.
              Um ciclo percorre as imagens do grupo e pode exigir vários ajustes internos.
              Cancelar a espera não garante interromper o servidor.
            </p>
          </div>
        )}
      </Card>

      <p className="text-[15px] leading-relaxed text-graphite-400">
        Acompanhe abaixo as etapas reais da execução. Depois dos ajustes,
        o laboratório compara a reconstrução com o original e calcula as medidas de semelhança.
      </p>
      {/* Terminal de Logs */}
      <Terminal
        logs={logs}
        isRunning={isRunning}
        title={`Etapas · ${envLabel}`}
        maxLines={30}
      />
    </div>
  );
}
