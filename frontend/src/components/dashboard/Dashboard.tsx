import {
  useState,
  useCallback,
  useRef,
  useEffect,
  lazy,
  Suspense,
} from "react";
import type { AttackConfig, AttackResponse } from "@/lib/types";
import { DEFAULT_CONFIG } from "@/lib/types";
import type { LogLine } from "@/components/ui/Terminal";
import { checkHealth, runAttack, runFederated } from "@/lib/api";
import { uid } from "@/lib/utils";
import { ErrorBoundary } from "@/components/ui/ErrorBoundary";
import { ExecutionPanel } from "@/components/panels/ExecutionPanel";
import { ResultsPanel } from "@/components/panels/ResultsPanel";
import { FederatedFlowPanel } from "@/components/panels/FederatedFlowPanel";
import { LearningPanel } from "@/components/panels/LearningPanel";
import {
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  SlidersHorizontal,
  Play,
  ScanLine,
  ChartNoAxesCombined,
  Code2,
  Github,
  Menu,
  X,
  RotateCcw,
  Square,
} from "lucide-react";

const SetupPanel = lazy(() =>
  import("@/components/panels/SetupPanel").then((module) => ({
    default: module.SetupPanel,
  })),
);
const ChartsPanel = lazy(() =>
  import("@/components/panels/ChartsPanel").then((module) => ({
    default: module.ChartsPanel,
  })),
);
const CodePanel = lazy(() =>
  import("@/components/panels/CodePanel").then((module) => ({
    default: module.CodePanel,
  })),
);

type ActiveTab =
  "overview" | "setup" | "execute" | "results" | "charts" | "code";
type RunState = "idle" | "running" | "success" | "error";
const TABS = [
  { key: "overview" as const, label: "Visão geral", icon: BookOpen },
  { key: "setup" as const, label: "Experimento", icon: SlidersHorizontal },
  { key: "execute" as const, label: "Execução", icon: Play },
  { key: "results" as const, label: "Reconstruções", icon: ScanLine },
  { key: "charts" as const, label: "Análise", icon: ChartNoAxesCombined },
  { key: "code" as const, label: "Código", icon: Code2 },
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
  const [activeTab, setActiveTab] = useState<ActiveTab>(() => {
    const hash = window.location.hash.slice(1);
    return TABS.some((tab) => tab.key === hash)
      ? (hash as ActiveTab)
      : "overview";
  });
  const [config, setConfig] = useState<AttackConfig>({ ...DEFAULT_CONFIG });

  const [runState, setRunState] = useState<RunState>("idle");
  const [logs, setLogs] = useState<LogLine[]>([]);
  const [progress, setProgress] = useState(0);
  const [response, setResponse] = useState<AttackResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [menuOpen, setMenuOpen] = useState(false);
  const [health, setHealth] = useState<"checking" | "online" | "offline">(
    "checking",
  );
  const [device, setDevice] = useState("");
  useEffect(() => {
    let live = true;
    checkHealth()
      .then((result) => {
        if (live) {
          setHealth("online");
          setDevice(result.device);
        }
      })
      .catch(() => {
        if (live) setHealth("offline");
      });
    return () => {
      live = false;
    };
  }, []);
  const navigate = (tab: ActiveTab) => {
    setActiveTab(tab);
    setMenuOpen(false);
    window.scrollTo({ top: 0, behavior: "instant" });
  };
  const abortRef = useRef<AbortController | null>(null);
  const logSimRef = useRef<number | null>(null);
  const progressRef = useRef<number | null>(null);

  // Limpa timers ao desmontar
  useEffect(() => {
    return () => {
      if (logSimRef.current) clearInterval(logSimRef.current);
      if (progressRef.current) clearInterval(progressRef.current);
      abortRef.current?.abort();
    };
  }, []);

  const addLog = useCallback(
    (entry: { text: string; type?: LogLine["type"] }) => {
      setLogs((prev) => [
        ...prev,
        {
          id: uid(),
          text: entry.text,
          type: entry.type ?? "info",
          timestamp: Date.now(),
        },
      ]);
    },
    [],
  );

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
        addLog({ text: `Falha: ${res.message}`, type: "error" });
      } else {
        const psnr = res.metrics?.psnrMean;
        const psnrStr =
          psnr != null && isFinite(psnr) ? psnr.toFixed(1) : "N/A";
        const lossCount = res.lossHistory?.length ?? 0;
        addLog({
          text: `Ataque concluído — PSNR médio: ${psnrStr} dB (${lossCount} amostras de loss)`,
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

    const controller = new AbortController();
    abortRef.current = controller;

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
        addLog({
          text: `Sobre o algoritmo (etapa ilustrativa): ${simMessages[msgIdx]}`,
          type: "info",
        });
        msgIdx++;
      }
    }, 2500);

    // Barra de progresso baseada nos ciclos do ataque
    // Para DLG centralizado, estima-se baseado nos ciclos e iterações.
    const totalCycles = Math.floor(
      config.numIters / Math.max(config.batchSize, 1),
    );
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
      const pct = Math.min(
        95,
        Math.round((elapsed / estimatedDurationMs) * 100),
      );
      setProgress(pct);
    }, 200);

    try {
      let rawRes: AttackResponse;

      if (config.environment === "federated") {
        addLog({
          text: `Simulação Federada: ${config.flClients} clientes, ${config.flRounds} rodadas`,
          type: "info",
        });
        rawRes = await runFederated(config, controller.signal);
      } else {
        rawRes = await runAttack(config, controller.signal);
      }

      if (!controller.signal.aborted) handleResponse(sanitizeResponse(rawRes));
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
      if (logSimRef.current) clearInterval(logSimRef.current);
      if (progressRef.current) clearInterval(progressRef.current);
      setProgress(0);

      if (err instanceof Error && err.name === "AbortError") {
        setRunState("idle");
        addLog({
          text: "Espera cancelada no navegador. O servidor pode continuar processando.",
          type: "warn",
        });
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
    addLog({
      text: "Espera cancelada no navegador. O servidor pode continuar processando.",
      type: "warn",
    });
  }, [addLog]);

  const tabTitle = TABS.find((tab) => tab.key === activeTab)?.label;
  const defenseLabels = {
    none: "Sem defesa",
    noisy: "Ruído gaussiano",
    sparse: "Esparsificação",
    pruning: "Poda do modelo",
  };
  return (
    <div className="lab-shell">
      <a className="skip-link" href="#main-content">
        Pular para o conteúdo
      </a>
      <aside
        className={`lab-sidebar ${menuOpen ? "is-open" : ""}`}
        aria-label="Navegação principal"
      >
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            navigate("overview");
          }}
        >
          <svg viewBox="0 0 36 36" width="32" height="32" aria-hidden="true">
            <path
              d="M4 9h9v9H4zm19 0h9v9h-9zM13 23h10v9H13z"
              fill="currentColor"
            />
            <path
              d="m13 14 10 0M9 18l9 10 9-10"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            />
          </svg>
          <span>
            DLG<span className="brand-light"> / lab</span>
          </span>
        </a>
        <button
          className="mobile-close"
          aria-label="Fechar navegação"
          onClick={() => setMenuOpen(false)}
        >
          <X size={20} />
        </button>
        <div className="project-label">
          <span>Deep Leakage from Gradients</span>
          <small>Um experimento em privacidade</small>
        </div>
        <nav>
          {TABS.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.key}
                aria-current={activeTab === tab.key ? "page" : undefined}
                className={`sidebar-link ${activeTab === tab.key ? "active" : ""}`}
                onClick={() => navigate(tab.key)}
              >
                <Icon size={18} />
                <span>{tab.label}</span>
                {tab.key === "execute" && runState === "running" && (
                  <span className="status-dot running" />
                )}
                {tab.key === "results" && response?.success && (
                  <span className="result-dot" />
                )}
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="model-note">
            <span>O experimento</span>
            <strong>CIFAR-100 + LeNet</strong>
            <small>Imagens 32 × 32 · 100 classes</small>
          </div>
          <a
            href="https://github.com/Eni19/dlg_simulator"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={16} /> Repositório <ArrowUpRight size={14} />
          </a>
          <span className="author-note">Projeto de Enrique Susin</span>
        </div>
      </aside>
      {menuOpen && (
        <button
          className="menu-backdrop"
          aria-label="Fechar menu"
          onClick={() => setMenuOpen(false)}
        />
      )}
      <div className="lab-workspace">
        <header className="lab-topbar">
          <div className="breadcrumb">
            <button
              className="menu-toggle"
              aria-label="Abrir navegação"
              aria-expanded={menuOpen}
              onClick={() => setMenuOpen(!menuOpen)}
            >
              <Menu size={20} />
            </button>
            <span>Laboratório de privacidade</span>
            <span className="breadcrumb-slash">/</span>
            <strong>{tabTitle}</strong>
          </div>
          <div
            className="connection"
            title={
              health === "offline"
                ? "Inicie o backend FastAPI na porta 8000 para executar experimentos."
                : "Conexão com o backend"
            }
          >
            <span className={`status-dot ${health}`} />
            <span>
              {health === "online"
                ? `API conectada${device ? ` · ${device.toUpperCase()}` : ""}`
                : health === "checking"
                  ? "Conectando à API"
                  : "API desconectada"}
            </span>
          </div>
        </header>
        <main id="main-content" className="lab-sheet" tabIndex={-1}>
          <Suspense fallback={<p role="status">Carregando o laboratório…</p>}>
            {activeTab !== "overview" && activeTab !== "code" && (
              <div className="workspace-heading">
                <div>
                  <h1>
                    {activeTab === "setup"
                      ? "Configure o experimento"
                      : activeTab === "execute"
                        ? "Acompanhe a execução"
                        : activeTab === "results"
                          ? "O que foi reconstruído?"
                          : "Leia os resultados"}
                  </h1>
                  <p>
                    {activeTab === "setup"
                      ? "Uma hipótese por vez. Ajuste o cenário e observe o que muda."
                      : activeTab === "execute"
                        ? "Da requisição ao resultado da otimização."
                        : activeTab === "results"
                          ? "Compare os dados originais com o que os gradientes revelaram."
                          : "Convergência da otimização e impacto das defesas."}
                  </p>
                </div>
                {activeTab === "setup" && (
                  <button
                    className="quiet-action"
                    disabled={runState === "running"}
                    onClick={() =>
                      setConfig({
                        ...DEFAULT_CONFIG,
                        defenseParams: { ...DEFAULT_CONFIG.defenseParams },
                      })
                    }
                  >
                    <RotateCcw size={15} /> Restaurar padrões
                  </button>
                )}
              </div>
            )}
            {runState === "error" && error && (
              <div className="error-banner" role="alert">
                <strong>O experimento não foi concluído.</strong>
                <p>{error}</p>
                <button
                  className="text-action"
                  onClick={() => navigate("setup")}
                >
                  Revisar configuração <ArrowRight size={15} />
                </button>
              </div>
            )}
            {activeTab === "overview" && (
              <LearningPanel
                onExperiment={() => navigate("setup")}
                onCode={() => navigate("code")}
              />
            )}
            {activeTab === "code" && <CodePanel />}
            {activeTab === "setup" && (
              <div className="experiment-layout">
                <div className="panel-content">
                  <ErrorBoundary>
                    <fieldset
                      disabled={runState === "running"}
                      className="setup-fieldset"
                    >
                      <legend className="sr-only">
                        Parâmetros do experimento
                      </legend>
                      <SetupPanel
                        config={config}
                        onChange={setConfig}
                        isRunning={runState === "running"}
                      />
                    </fieldset>
                  </ErrorBoundary>
                </div>
                <aside className="experiment-summary">
                  <h2>Seu experimento</h2>
                  <dl>
                    <div>
                      <dt>Ambiente</dt>
                      <dd>
                        {config.environment === "federated"
                          ? "Federado"
                          : "Centralizado"}
                      </dd>
                    </div>
                    <div>
                      <dt>Batch</dt>
                      <dd>
                        {config.batchSize}{" "}
                        {config.batchSize === 1 ? "imagem" : "imagens"}
                      </dd>
                    </div>
                    <div>
                      <dt>Iterações</dt>
                      <dd>{config.numIters.toLocaleString("pt-BR")}</dd>
                    </div>
                    <div>
                      <dt>Seed</dt>
                      <dd>{config.seed}</dd>
                    </div>
                    <div>
                      <dt>Defesa</dt>
                      <dd>{defenseLabels[config.defense]}</dd>
                    </div>
                  </dl>
                  {config.environment === "federated" && (
                    <p>
                      {config.flClients} clientes · {config.flRounds} rodadas ·{" "}
                      {config.partitionerType.toUpperCase()}
                    </p>
                  )}
                  <button
                    className="primary-action"
                    disabled={runState === "running"}
                    onClick={handleStart}
                  >
                    <Play size={16} />
                    {runState === "running" ? "Em execução" : "Executar ataque"}
                  </button>
                  <p className="summary-note">
                    O processamento usa o backend Python. A duração depende das
                    iterações e do dispositivo.
                  </p>
                  {health === "offline" && (
                    <div className="backend-notice">
                      <strong>Backend indisponível</strong>
                      <p>
                        Você pode explorar a teoria e configurar o cenário. Para
                        executar, inicie a API local.
                      </p>
                      <code>python -m uvicorn app:app --port 8000</code>
                    </div>
                  )}
                  <div className="experiment-tip">
                    <h3>Comece pelo essencial</h3>
                    <p>
                      Use batch 1 e nenhuma defesa como referência. Depois
                      altere uma variável para comparar os efeitos.
                    </p>
                    <button
                      className="text-action"
                      onClick={() => navigate("overview")}
                    >
                      Revisar os conceitos <ArrowRight size={14} />
                    </button>
                  </div>
                </aside>
              </div>
            )}
            {activeTab === "execute" && (
              <div className="panel-content">
                <ErrorBoundary>
                  <ExecutionPanel
                    config={config}
                    isRunning={runState === "running"}
                    logs={logs}
                    progress={progress}
                    onStart={handleStart}
                    onAbort={handleAbort}
                  />
                  {config.environment === "federated" && (
                    <div className="mt-6">
                      <FederatedFlowPanel
                        config={config}
                        isRunning={runState === "running"}
                      />
                    </div>
                  )}
                </ErrorBoundary>
              </div>
            )}
            {activeTab === "results" && (
              <div className="panel-content">
                <ErrorBoundary>
                  <ResultsPanel
                    response={response}
                    isLoading={runState === "running"}
                  />
                </ErrorBoundary>
                {!response && runState !== "running" && (
                  <button
                    className="primary-action empty-action"
                    onClick={() => navigate("setup")}
                  >
                    Configurar experimento <ArrowRight size={16} />
                  </button>
                )}
              </div>
            )}
            {activeTab === "charts" && (
              <div className="panel-content">
                <ErrorBoundary>
                  <ChartsPanel
                    response={response}
                    isLoading={runState === "running"}
                  />
                </ErrorBoundary>
                {!response && runState !== "running" && (
                  <button
                    className="primary-action empty-action"
                    onClick={() => navigate("setup")}
                  >
                    Configurar experimento <ArrowRight size={16} />
                  </button>
                )}
              </div>
            )}
            <footer className="lab-footer">
              <span>
                DLG / lab <span>Aprender observando.</span>
              </span>
              <a
                href="https://github.com/Eni19/dlg_simulator"
                target="_blank"
                rel="noreferrer"
              >
                Explore a implementação <ArrowUpRight size={14} />
              </a>
            </footer>
          </Suspense>
        </main>
      </div>
      {runState === "running" && activeTab !== "execute" && (
        <div className="running-notice" role="status">
          <span className="status-dot running" />
          <button onClick={() => navigate("execute")}>
            Experimento em andamento · ver execução
          </button>
          <button aria-label="Cancelar espera" onClick={handleAbort}>
            <Square size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
