import { useState, useCallback, useRef, useEffect } from "react";
import type { AttackConfig, AttackResponse } from "@/lib/types";
import { DEFAULT_CONFIG } from "@/lib/types";
import type { LogLine } from "@/components/ui/Terminal";
import { runAttack, runFederated } from "@/lib/api";
import { uid } from "@/lib/utils";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { SetupPanel } from "@/components/panels/SetupPanel";
import { ExecutionPanel } from "@/components/panels/ExecutionPanel";
import { ResultsPanel } from "@/components/panels/ResultsPanel";
import { ChartsPanel } from "@/components/panels/ChartsPanel";
import { FederatedFlowPanel } from "@/components/panels/FederatedFlowPanel";
import {
  Settings,
  Play,
  BarChart3,
  Image,
  FlaskConical,
  Shield,
  Github,
} from "lucide-react";

type ActiveTab = "setup" | "execute" | "results" | "charts";
type RunState = "idle" | "running" | "success" | "error";

const TABS: { key: ActiveTab; label: string; icon: React.ComponentType<{ className?: string }> }[] = [
  { key: "setup", label: "Configuração", icon: Settings },
  { key: "execute", label: "Execução", icon: Play },
  { key: "results", label: "Resultados", icon: Image },
  { key: "charts", label: "Gráficos", icon: BarChart3 },
];

/** Sanitiza um número para evitar Infinity/NaN que quebram a UI */
function safeNum(n: number | undefined | null, fallback = 0): number {
  if (n == null || !isFinite(n)) return fallback;
  return n;
}

/** Sanitiza o AttackResponse para garantir que todos os campos são seguros */
function sanitizeResponse(res: AttackResponse): AttackResponse {
  return {
    ...res,
    lossHistory: (res.lossHistory ?? []).map((v) => safeNum(v, 1e-6)),
    metrics: res.metrics
      ? {
          ssimMean: safeNum(res.metrics.ssimMean, 0),
          psnrMean: safeNum(res.metrics.psnrMean, 0),
          mseMean: safeNum(res.metrics.mseMean, 0),
          ssimStd: safeNum(res.metrics.ssimStd, 0),
          psnrStd: safeNum(res.metrics.psnrStd, 0),
          defenseType: res.metrics.defenseType,
          informedAttacker: res.metrics.informedAttacker,
        }
      : null,
    reconstructedLabels: res.reconstructedLabels ?? [],
    groundTruthImages: res.groundTruthImages ?? [],
    reconstructedImages: res.reconstructedImages ?? [],
    errorImages: res.errorImages ?? [],
    snapshotIterations: res.snapshotIterations ?? [],
    snapshotImages: res.snapshotImages ?? [],
    device: res.device ?? "cpu",
    batchSize: res.batchSize ?? 0,
    message: res.message ?? "",
  };
}

/**
 * Dashboard principal — coordena os 4 steps do fluxo DLG.
 */
export function Dashboard() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("setup");
  const [config, setConfig] = useState<AttackConfig>({ ...DEFAULT_CONFIG });

  const [runState, setRunState] = useState<RunState>("idle");
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [progress, setProgress] = useState(0);
  const [response, setResponse] = useState<AttackResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const logSimRef = useRef<number | null>(null);
  const progressRef = useRef<number | null>(null);

  // Limpa timers ao desmontar
  useEffect(() => {
    return () => {
      if (logSimRef.current) clearInterval(logSimRef.current);
      if (progressRef.current) clearInterval(progressRef.current);
    };
  }, []);

  const addLog = useCallback((entry: { text: string; type?: LogLine["type"] }) => {
    setLogs((prev) => [
      ...prev,
      { id: uid(), text: entry.text, type: entry.type ?? "info", timestamp: Date.now() },
    ]);
  }, []);

  /** Processa a resposta do backend */
  const handleResponse = useCallback(
    (res: AttackResponse) => {
      setResponse(res);
      setRunState(res.success ? "success" : "error");
      setProgress(res.success ? 100 : 0);

      if (logSimRef.current) clearInterval(logSimRef.current);
      if (progressRef.current) clearInterval(progressRef.current);

      if (!res.success) {
        setError(res.message || "Erro desconhecido");
        addLog({ text: `✕ Falha: ${res.message}`, type: "error" });
      } else {
        const psnr = res.metrics?.psnrMean;
        const psnrStr = (psnr != null && isFinite(psnr)) ? psnr.toFixed(1) : "N/A";
        const lossCount = res.lossHistory?.length ?? 0;
        addLog({
          text: `✓ Ataque concluído — PSNR médio: ${psnrStr} dB (${lossCount} amostras de loss)`,
          type: "success",
        });
        setActiveTab("results");
      }
    },
    [addLog],
  );

  /** Inicia o ataque DLG */
  const handleStart = useCallback(async () => {
    if (runState === "running") return;

    setRunState("running");
    setProgress(0);
    setLogs([]);
    setResponse(null);
    setError(null);
    setActiveTab("execute");

    addLog({ text: "Iniciando ataque DLG...", type: "info" });
    addLog({
      text: `Ambiente: ${config.environment} · Batch: ${config.batchSize} · Iterações: ${config.numIters}`,
      type: "info",
    });

    if (config.defense !== "none") {
      addLog({
        text: `Defesa ativa: ${config.defense}`,
        type: "warn",
      });
    }

    abortRef.current = new AbortController();

    // Simulador de logs
    const simMessages = [
      "Carregando dataset CIFAR-100...",
      "Instanciando modelo LeNet (Sigmoid)...",
      "Computando gradientes ground-truth...",
      "Inicializando reconstrução dummy (smart init)...",
      "Otimização LBFGS em andamento...",
      "Round-robin + global phase...",
      "TV cosine annealing: 1e-5 → 1e-7",
      "Label anchoring via bias gradients...",
    ];

    let msgIdx = 0;
    logSimRef.current = window.setInterval(() => {
      if (msgIdx < simMessages.length) {
        addLog({ text: simMessages[msgIdx], type: "metric" });
        msgIdx++;
      }
    }, 2500);

    // Barra de progresso baseada nos ciclos do ataque
    // Para DLG centralizado, estima-se baseado nos ciclos e iterações.
    const totalCycles = Math.floor(config.numIters / Math.max(config.batchSize, 1));
    const msPerCycle = 220; // estimativa realista para CPU
    let estimatedDurationMs = totalCycles * msPerCycle;

    // Se for federado, adicionamos o tempo de treinamento por rodada
    if (config.environment === "federated") {
      const msPerClientRound = 800; // estimativa de treino local por cliente
      const flTime = config.flRounds * config.flClients * msPerClientRound;
      estimatedDurationMs += flTime;
    }

    addLog({
      text: `Duração estimada: ${(estimatedDurationMs / 1000).toFixed(0)}s`,
      type: "info",
    });

    const t0 = Date.now();
    progressRef.current = window.setInterval(() => {
      const elapsed = Date.now() - t0;
      // Crescimento assintótico suave até 95% para não travar nos 90%
      const pct = Math.min(95, Math.round((elapsed / estimatedDurationMs) * 100));
      setProgress(pct);
    }, 200);

    try {
      let rawRes: AttackResponse;

      if (config.environment === "federated") {
        addLog({
          text: `Simulação Federada: ${config.flClients} clientes, ${config.flRounds} rodadas`,
          type: "info",
        });
        rawRes = await runFederated(config, abortRef.current.signal);
      } else {
        rawRes = await runAttack(config, abortRef.current.signal);
      }

      handleResponse(rawRes);
    } catch (err: unknown) {
      if (logSimRef.current) clearInterval(logSimRef.current);
      if (progressRef.current) clearInterval(progressRef.current);
      setProgress(0);

      if (err instanceof Error && err.name === "AbortError") {
        setRunState("idle");
        addLog({ text: "Ataque abortado pelo usuário", type: "warn" });
        return;
      }
      const msg = err instanceof Error ? err.message : "Erro desconhecido";
      setRunState("error");
      setError(msg);
      addLog({ text: `Erro: ${msg}`, type: "error" });
    }
  }, [config, runState, addLog, handleResponse]);

  /** Aborta o ataque em andamento */
  const handleAbort = useCallback(() => {
    abortRef.current?.abort();
    if (logSimRef.current) clearInterval(logSimRef.current);
    if (progressRef.current) clearInterval(progressRef.current);
    setRunState("idle");
    setProgress(0);
    addLog({ text: "Ataque abortado pelo usuário", type: "warn" });
  }, [addLog]);

  return (
    <div className="min-h-screen bg-graphite-950 text-graphite-100">
      {/* Header */}
      <header className="sticky top-0 z-40 border-b border-graphite-800/60 bg-graphite-950/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-600/20 ring-1 ring-emerald-500/30">
              <Shield className="h-5 w-5 text-emerald-400" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white">
                DLG Attack
              </h1>
              <p className="text-[10px] font-medium uppercase tracking-wider text-graphite-500">
                Deep Leakage from Gradients
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <a
              href="https://flower.ai"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-lg px-3 py-1.5 text-xs text-graphite-500 transition-colors hover:text-graphite-300 hover:bg-graphite-800/50"
            >
              Flower.ai
            </a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs text-graphite-500 transition-colors hover:text-graphite-300 hover:bg-graphite-800/50"
            >
              <Github className="h-3.5 w-3.5" />
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Navegação de Abas */}
      <nav className="sticky top-[57px] z-30 border-b border-graphite-800/60 bg-graphite-950/80 backdrop-blur-xl">
        <div className="mx-auto max-w-5xl px-4 sm:px-6">
          <div className="flex gap-1 overflow-x-auto">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.key;
              const isComplete =
                (tab.key === "setup" && response !== null) ||
                (tab.key === "execute" && (runState === "running" || response !== null)) ||
                (tab.key === "results" && response !== null) ||
                (tab.key === "charts" && response !== null);

              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={`flex items-center gap-2 whitespace-nowrap border-b-2 px-4 py-3 text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? "border-emerald-500 text-emerald-400"
                      : isComplete
                        ? "border-transparent text-graphite-400 hover:text-graphite-200"
                        : "border-transparent text-graphite-600"
                  }`}
                >
                  <Icon
                    className={`h-4 w-4 ${isActive ? "text-emerald-400" : ""}`}
                  />
                  {tab.label}
                  {isActive && (
                    <span className="ml-0.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse-soft" />
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Conteúdo Principal */}
      <main className="mx-auto max-w-5xl px-4 py-6 sm:px-6">
        {/* Banner de Erro Global */}
        {runState === "error" && error && (
          <div className="mb-6 rounded-xl border border-red-500/20 bg-red-600/10 px-5 py-4">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-red-600/20">
                <FlaskConical className="h-4 w-4 text-red-400" />
              </div>
              <div>
                <p className="text-sm font-semibold text-red-400">
                  Erro na Execução
                </p>
                <p className="mt-0.5 text-xs text-graphite-400">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Step 1 — Setup */}
        {activeTab === "setup" && (
          <ErrorBoundary>
            <SetupPanel config={config} onChange={setConfig} isRunning={runState === "running"} />
          </ErrorBoundary>
        )}

        {/* Step 2 — Execução */}
        {activeTab === "execute" && (
          <ErrorBoundary>
            <ExecutionPanel
              config={config}
              isRunning={runState === "running"}
              logs={logs}
              progress={progress}
              onStart={handleStart}
              onAbort={handleAbort}
            />
            {/* Fluxo Federado: explicação visual quando ambiente federado */}
            {config.environment === "federated" && (
              <div className="mt-5">
                <FederatedFlowPanel config={config} isRunning={runState === "running"} />
              </div>
            )}
          </ErrorBoundary>
        )}

        {/* Step 3 — Resultados */}
        {activeTab === "results" && (
          <ErrorBoundary>
            <ResultsPanel
              response={response}
              isLoading={runState === "running"}
            />
          </ErrorBoundary>
        )}

        {/* Step 4 — Gráficos */}
        {activeTab === "charts" && (
          <ErrorBoundary>
            <ChartsPanel
              response={response}
              isLoading={runState === "running"}
            />
          </ErrorBoundary>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-graphite-800/60">
        <div className="mx-auto max-w-5xl px-4 py-4 sm:px-6">
          <p className="text-center text-[11px] text-graphite-600">
            DLG Attack Tool • CIFAR-100 + LeNet (Sigmoid) • Loss Composta: L2 +
            Cosine + TV • Flower.ai Design System
          </p>
        </div>
      </footer>
    </div>
  );
}
