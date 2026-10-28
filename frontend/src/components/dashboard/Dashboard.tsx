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
import { SHOWCASE_BUILD, showcaseConfig, SHOWCASE_SAMPLES } from "@/lib/showcase";
import type { LogLine } from "@/components/ui/Terminal";
import { checkHealth, runExperimentStream, type ExperimentProgress } from "@/lib/api";
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
  Linkedin,
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
const LINKEDIN_URL = "https://www.linkedin.com/in/enrique-susin-3a4a36234/";
const TABS = [
  { key: "overview" as const, label: "Comece por aqui", icon: BookOpen },
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
  const [isShowcase, setIsShowcase] = useState(SHOWCASE_BUILD);
  const [activeTab, setActiveTab] = useState<ActiveTab>(() => {
    const hash = window.location.hash.slice(1);
    return TABS.some((tab) => tab.key === hash)
      ? (hash as ActiveTab)
      : "overview";
  });
  const [config, setConfig] = useState<AttackConfig>(() => SHOWCASE_BUILD ? showcaseConfig() : { ...DEFAULT_CONFIG });

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
          const showcase = SHOWCASE_BUILD || result.mode === "showcase";
          setIsShowcase(showcase);
          if (showcase) setConfig(previous => showcaseConfig(previous));
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
  const [telemetry, setTelemetry] = useState<ExperimentProgress | null>(null);

  // Encerra o acompanhamento ao desmontar.
  useEffect(() => {
    return () => {
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


      if (!res.success) {
        setError(res.message || "Erro desconhecido");
        addLog({ text: `Falha: ${res.message}`, type: "error" });
      } else {
        const psnr = res.metrics?.psnrMean;
        const psnrStr =
          psnr != null && isFinite(psnr) ? psnr.toFixed(1) : "N/A";
        const lossCount = res.lossHistory?.length ?? 0;
        addLog({
          text: `Experimento concluído — PSNR médio: ${psnrStr} dB (${lossCount} registros do erro da busca)`,
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

    addLog({ text: "Preparando uma tentativa de reconstrução...", type: "info" });
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

    setTelemetry(null);
    let lastLogKey = "";
    try {
      const rawRes = await runExperimentStream(config, (event) => {
        if (controller.signal.aborted) return;
        setTelemetry(event);
        setProgress(event.total && event.current != null
          ? Math.round(event.current / event.total * 100) : 0);
        const key = `${event.phase}:${event.text}`;
        const shouldLog = key !== lastLogKey || (event.current != null && event.total != null
          && (event.current === event.total || event.current % 10 === 0));
        if (shouldLog) {
          lastLogKey = key;
          const count = event.total && event.current != null ? ` (${event.current}/${event.total})` : "";
          const loss = event.loss != null ? ` · loss ${event.loss.toExponential(3)}` : "";
          addLog({ text: `${event.text}${count}${loss}`, type: "info" });
        }
      }, controller.signal);

      if (!controller.signal.aborted) handleResponse(sanitizeResponse(rawRes));
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
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
          {/* Uma imagem de 3 × 3 pixels que vai ganhando nitidez. */}
          <svg className="brand-mark" viewBox="0 0 30 30" width="30" height="30" aria-hidden="true">
            {[0.25, 0.5, 1, 0.5, 1, 0.75, 1, 0.75, 0.35].map((opacity, i) => (
              <rect key={i} x={(i % 3) * 10} y={Math.floor(i / 3) * 10}
                width="8.5" height="8.5" rx="2.2" fill="currentColor" opacity={opacity} />
            ))}
          </svg>
          <span>
            <span className="brand-name">DLG</span>
            <span className="brand-light">lab</span>
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
          <a
            href={LINKEDIN_URL}
            target="_blank"
            rel="noreferrer"
          >
            <Linkedin size={16} /> LinkedIn <ArrowUpRight size={14} />
          </a>
          <a
            className="author-note"
            href={LINKEDIN_URL}
            target="_blank"
            rel="noreferrer"
            aria-label="Projeto de Enrique Susin, perfil no LinkedIn"
          >
            Projeto de Enrique Susin
          </a>
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
                ? "O servidor que faz os cálculos precisa estar disponível para iniciar um experimento."
                : `Servidor de cálculo conectado. Dispositivo padrão: ${device || "a confirmar"}. Você pode escolher outro no experimento.`
            }
          >
            <span className={`status-dot ${health}`} />
            <span>
              {health === "online"
                ? isShowcase ? "Showcase online" : "Servidor conectado"
                : health === "checking"
                  ? "Conectando ao servidor"
                  : "Servidor desconectado"}
            </span>
          </div>
        </header>
        <main id="main-content" className="lab-sheet" tabIndex={-1}>
          <Suspense fallback={<p role="status">Carregando o laboratório…</p>}>
            <div key={activeTab} className="tab-view">
            {activeTab !== "overview" && activeTab !== "code" && (
              <div className="workspace-heading">
                <div>
                  <h1>
                    {activeTab === "setup"
                      ? isShowcase ? "Experimente uma reconstrução" : "Prepare seu experimento"
                      : activeTab === "execute"
                        ? "Acompanhe a execução"
                        : activeTab === "results"
                          ? "O que foi reconstruído?"
                          : "O que os números ajudam a enxergar?"}
                  </h1>
                  <p>
                    {activeTab === "setup"
                      ? isShowcase ? "Escolha um exemplo preparado e acompanhe o que o algoritmo consegue recuperar." : "Comece com a configuração pronta. Depois, mude uma escolha por vez e veja como ela afeta a reconstrução."
                      : activeTab === "execute"
                        ? "Veja o que o laboratório está fazendo e quantos ciclos de ajuste já foram concluídos."
                        : activeTab === "results"
                          ? "Observe cores, formas e detalhes: o que a tentativa conseguiu recuperar da imagem original?"
                          : "Use os gráficos para acompanhar a busca e comparar o efeito das defesas."}
                  </p>
                </div>
                {activeTab === "setup" && (
                  <button
                    className="quiet-action"
                    disabled={runState === "running"}
                    onClick={() =>
                      setConfig({
                        ...(isShowcase ? showcaseConfig() : DEFAULT_CONFIG),
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
                isShowcase={isShowcase}
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
                        isShowcase={isShowcase}
                        onStart={handleStart}
                        onChange={next => setConfig(isShowcase ? showcaseConfig(next) : next)}
                        isRunning={runState === "running"}
                      />
                    </fieldset>
                  </ErrorBoundary>
                </div>
                <aside className="experiment-summary">
                  <h2>Seu experimento</h2>
                  <dl>
                    {isShowcase && <div>
                      <dt>{config.batchSize === 1 ? "Exemplo" : "Exemplos"}</dt>
                      <dd>{config.imageIndices?.map(id => SHOWCASE_SAMPLES.find(sample => sample.id === id)?.name).join(" e ")}</dd>
                    </div>}
                    <div>
                      <dt>Ambiente</dt>
                      <dd>
                        {config.environment === "federated"
                          ? "Federado"
                          : "Centralizado"}
                      </dd>
                    </div>
                    <div>
                      <dt>Imagens por vez</dt>
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
                      <dt>Início aleatório</dt>
                      <dd>{config.seed}</dd>
                    </div>
                    <div>
                      <dt>Defesa</dt>
                      <dd>{defenseLabels[config.defense]}</dd>
                    </div>
                    <div>
                      <dt>Processamento</dt>
                      <dd>{config.device === "cuda" ? "GPU NVIDIA" : config.device === "cpu" ? "CPU" : "Automático"}</dd>
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
                    {runState === "running" ? "Em execução" : "Iniciar experimento"}
                  </button>
                  <p className="summary-note">
                    O laboratório fará os cálculos e mostrará a imagem original
                    ao lado da reconstrução. Mais iterações podem aumentar a espera.
                  </p>
                  {health === "offline" && (
                    <div className="backend-notice">
                      <strong>O servidor de cálculo está indisponível</strong>
                      <p>
                        Você pode continuar lendo e preparar o experimento.
                        Para executar nesta instalação local, inicie o servidor:
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
                    telemetry={telemetry}
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
              <nav className="footer-links" aria-label="Links do autor">
                <a href={LINKEDIN_URL} target="_blank" rel="noreferrer">
                  Fale com Enrique no LinkedIn <ArrowUpRight size={14} />
                </a>
                <a
                  href="https://github.com/Eni19/dlg_simulator"
                  target="_blank"
                  rel="noreferrer"
                >
                  Explore a implementação <ArrowUpRight size={14} />
                </a>
              </nav>
            </footer>
            </div>
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
