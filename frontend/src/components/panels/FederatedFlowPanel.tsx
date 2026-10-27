import { Card, CardHeader } from "@/components/ui/Card";
import type { AttackConfig } from "@/lib/types";
import { defenseLabel as describeDefense } from "@/lib/types";
import {
  Server,
  Users,
  Shield,
  Send,
  Download,
  Cpu,
  Layers,
  ArrowLeftRight,
  Eye,
  Zap,
  Wifi,
} from "lucide-react";

interface FederatedFlowPanelProps {
  config: AttackConfig;
  isRunning?: boolean;
}

/**
 * Visualização explicativa do fluxo de Ataque DLG em Ambiente Federado.
 *
 * Ilustra visualmente:
 * 1. Servidor → distribui modelo global para clientes
 * 2. Cada cliente treina localmente com seus dados privados
 * 3. Atacante intercepta gradientes do cliente alvo
 * 4. Servidor combina as atualizações a cada rodada
 * 5. Após as rodadas, DLG tenta reconstruir as imagens
 */
export function FederatedFlowPanel({
  config,
  isRunning,
}: FederatedFlowPanelProps) {
  const numClients = config.flClients;
  const numRounds = config.flRounds;
  const defense = config.defense;
  const defenseLabel = describeDefense(defense);

  const isServer = config.attackerRole === "malicious_server";
  const isClient = config.attackerRole === "malicious_client";
  const isMitm = config.attackerRole === "mitm";

  const interceptorIcon = isServer ? (
    <Server className="h-8 w-8" />
  ) : isClient ? (
    <Users className="h-8 w-8" />
  ) : (
    <Wifi className="h-8 w-8" />
  );
  const interceptorLabel = isServer
    ? "Servidor"
    : isClient
      ? "Outro participante"
      : "Alguém na rede";
  const interceptorDesc = isServer
    ? "Observa os gradientes"
    : isClient
      ? "Estima as contribuições"
      : "Observa atualizações em trânsito";

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Explicação visual interativa do fluxo */}
      <Card>
        <CardHeader
          title="O caminho das pistas no aprendizado em conjunto"
          subtitle="Este diagrama explica o cenário simulado. O andamento da execução aparece nos contadores acima."
        />
        <div className="space-y-6">
          {/* Animação de fluxo */}
          <div className="overflow-x-auto pb-4">
            <div className="flex items-center justify-between min-w-[700px] py-4 px-2">
              <FlowBox
                icon={<Server className="h-8 w-8" />}
                label="Servidor"
                sub="Modelo Global"
                color="  border-blue-500/40 text-blue-400 "
              />
              <FlowArrow animated />
              <FlowBox
                icon={<Users className="h-8 w-8" />}
                label={`${numClients} Clientes`}
                sub="Cada um aprende com seus exemplos"
                color="  border-emerald-500/40 text-emerald-400 "
              />
              <FlowArrow animated highlight />
              <FlowBox
                icon={interceptorIcon}
                label={interceptorLabel}
                sub={interceptorDesc}
                color="  border-amber-500/50 text-amber-400  ring-1 ring-amber-500/50"
                highlight
              />
              <FlowArrow animated highlight />
              <FlowBox
                icon={<ArrowLeftRight className="h-8 w-8" />}
                label="FedAvg"
                sub="Combina as atualizações"
                color="  border-emerald-500/40 text-emerald-400 "
              />
              <FlowArrow animated highlight />
              <FlowBox
                icon={<Zap className="h-8 w-8" />}
                label="Ataque DLG"
                sub="Tentativa de reconstrução"
                color="  border-red-500/50 text-red-400  ring-1 ring-red-500/50"
                highlight
              />
            </div>
          </div>

          {/* Legendas em grade */}
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <InfoCard
              title="O que pode ser revelado?"
              text="Os gradientes podem carregar informações sobre os exemplos de treino. O ataque tenta recuperar uma imagem a partir dessas pistas."
              icon={<Eye className="h-4 w-4 text-amber-400" />}
            />
            <InfoCard
              title="Aprender, compartilhar e combinar"
              text={`Durante ${numRounds} rodadas, os participantes aprendem localmente. O servidor combina suas atualizações por uma média chamada FedAvg.`}
              icon={<ArrowLeftRight className="h-4 w-4 text-blue-400" />}
            />
            <InfoCard
              title="O ponto de partida do ataque"
              text="O programa guarda uma cópia do modelo no momento do cálculo dos gradientes. Essa cópia ajuda a comparar pistas geradas nas mesmas condições."
              icon={<Cpu className="h-4 w-4 text-emerald-400" />}
            />
            <InfoCard
              title={
                defense !== "none" ? `Defesa: ${defenseLabel}` : "Sem Defesa"
              }
              text={
                defense === "none"
                  ? "As pistas ficam disponíveis sem uma defesa adicional. O que será recuperado depende das imagens e dos ajustes da busca."
                  : defense === "noisy"
                    ? "Variações aleatórias alteram as pistas compartilhadas. Compare quanto isso dificulta a reconstrução."
                    : defense === "sparse"
                      ? "Parte dos gradientes é zerada antes do compartilhamento, reduzindo as informações disponíveis."
                      : "Parte das conexões do modelo é removida. O efeito também depende de o ataque conhecer essa mudança."
              }
              icon={
                <Shield
                  className={`h-4 w-4 ${defense !== "none" ? "text-amber-400" : "text-graphite-500"}`}
                />
              }
            />
          </div>
        </div>
      </Card>

      {/* Detalhamento Passo a Passo */}
      <Card variant="highlight">
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Layers className="h-5 w-5 text-emerald-400" />
              Acompanhe a ideia, passo a passo
            </span>
          }
          subtitle="O que acontece nos bastidores em cada estágio"
        />
        <div className="space-y-6">
          {/* Estágios numerados */}
          <FlowStage
            num={1}
            title="Todos recebem uma versão do modelo"
            description="O servidor envia o modelo compartilhado aos participantes. Ele serve como ponto de partida para o aprendizado daquela rodada."
            icon={<Server className="h-5 w-5" />}
            color="blue"
          />

          <FlowStage
            num={2}
            title="Cada participante aprende com seus exemplos"
            description={`Os ${numClients} participantes comparam as respostas do modelo com as categorias das imagens. A diferença orienta os ajustes. Esse cálculo gera os gradientes, as pistas que investigamos aqui. Os exemplos são imagens públicas separadas entre participantes simulados.`}
            icon={<Users className="h-5 w-5" />}
            color="emerald"
          >
            {/* Cards de clientes */}
            <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {Array.from({ length: numClients }).map((_, idx) => {
                const isAttacker = idx === numClients - 1;
                return (
                  <div
                    key={idx}
                    className={`relative overflow-hidden rounded-xl border px-4 py-3 text-xs transition-colors hover:scale-[1.02] ${
                      isAttacker
                        ? "border-amber-500/50 bg-amber-950/30 "
                        : "border-graphite-700/50 bg-graphite-900/50 hover:bg-graphite-800/60"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {isAttacker ? (
                          <div className="p-1 rounded bg-amber-500/20">
                            <Eye className="h-3.5 w-3.5 text-amber-400" />
                          </div>
                        ) : (
                          <div className="p-1 rounded bg-graphite-700/50">
                            <Cpu className="h-3.5 w-3.5 text-graphite-400" />
                          </div>
                        )}
                        <span
                          className={`font-semibold ${
                            isAttacker ? "text-amber-400" : "text-graphite-200"
                          }`}
                        >
                          Cliente {idx}
                        </span>
                      </div>
                      {isAttacker && (
                        <span className="text-[9px] uppercase font-bold text-amber-500/80 tracking-wider">
                          Atacante
                        </span>
                      )}
                    </div>
                    <p className="text-graphite-400 leading-relaxed">
                      {isAttacker
                        ? isClient
                          ? "Atua normalmente, mas usa o modelo global para deduzir as atualizações dos outros."
                          : "Suas atualizações são observadas no cenário simulado."
                        : "Aprende com seus exemplos e prepara as atualizações."}
                    </p>
                    {defense !== "none" && (
                      <div className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-graphite-700 bg-graphite-950 px-2.5 py-1 text-[10px] font-medium text-graphite-300">
                        <Shield className="h-3 w-3 text-emerald-400" />
                        {defenseLabel}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </FlowStage>

          <FlowStage
            num={3}
            title={
              isServer
                ? "O servidor observa as atualizações"
                : isClient
                  ? "Um participante tenta separar as contribuições"
                  : "Alguém observa as atualizações na rede"
            }
            description={
              isServer
                ? "Neste cenário, o coordenador recebe as atualizações dos participantes. O laboratório simula esse acesso e usa as pistas do participante escolhido numa tentativa de reconstrução."
                : isClient
                  ? "Um participante conhece sua própria contribuição e observa a atualização combinada. Ele tenta usar essa informação para estimar o que veio dos demais. Esse cenário depende do número de participantes e do que está disponível para observação."
                  : "O laboratório supõe que alguém consegue observar as atualizações em trânsito entre um participante e o servidor. Essa é uma condição simulada para estudar o ataque."
            }
            icon={
              isServer ? (
                <Server className="h-5 w-5" />
              ) : isClient ? (
                <Users className="h-5 w-5" />
              ) : (
                <Wifi className="h-5 w-5" />
              )
            }
            color="amber"
            highlight
          />

          <FlowStage
            num={4}
            title="As contribuições formam um novo modelo"
            description="O servidor combina as atualizações numa média ponderada, o FedAvg. A versão resultante é compartilhada na rodada seguinte, e esse aprendizado se repete pelo número de rodadas escolhido."
            icon={<ArrowLeftRight className="h-5 w-5" />}
            color="violet"
          />

          <FlowStage
            num={5}
            title="Uma imagem é ajustada para explicar as pistas"
            description={`Depois da última rodada, este simulador tenta reconstruir as imagens a partir das pistas capturadas. O ataque começa com uma imagem ruidosa e compara os gradientes dela com os observados. Com ${config.numIters} iterações configuradas, a busca modifica a imagem para aproximar essas pistas. A qualidade da reconstrução varia conforme o cenário.`}
            icon={<Zap className="h-5 w-5" />}
            color="red"
            highlight
          />

        </div>
      </Card>
    </div>
  );
}

/* ── Subcomponentes ──────────────────────────────────────────────────── */

function FlowStage({
  num,
  title,
  description,
  icon,
  color,
  highlight,
  children,
}: {
  num: number;
  title: string;
  description: string;
  icon: React.ReactNode;
  color: "blue" | "emerald" | "amber" | "red" | "violet";
  highlight?: boolean;
  children?: React.ReactNode;
}) {
  const colorMap = {
    blue: "border-blue-500/30 bg-blue-600/10 text-blue-400",
    emerald: "border-emerald-500/30 bg-emerald-600/10 text-emerald-400",
    amber: "border-amber-500/30 bg-amber-600/10 text-amber-400",
    red: "border-red-500/30 bg-red-600/10 text-red-400",
    violet: "border-emerald-500/30 bg-emerald-600/10 text-emerald-400",
  };

  return (
    <div
      className={`relative rounded-xl border p-5 transition-colors hover:bg-graphite-800/20 ${
        highlight
          ? "border-amber-500/30 bg-amber-950/10 "
          : "border-graphite-800/50 bg-graphite-900/30"
      }`}
    >
      <div className="flex items-start gap-4">
        {/* Número do estágio */}
        <div
          className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl border text-lg font-bold  ${colorMap[color]}`}
        >
          {num}
        </div>
        <div className="flex-1 min-w-0 pt-0.5">
          <div className="flex items-center gap-2 mb-1.5">
            <span className={colorMap[color].split(" ")[2]}>{icon}</span>
            <h4 className="text-base font-semibold text-white tracking-tight">
              {title}
            </h4>
          </div>
          <p className="text-sm text-graphite-400 leading-relaxed mb-1">
            {description}
          </p>
          {children}
        </div>
      </div>
      {/* Conector vertical entre estágios com gradiente */}
      {num < 5 && (
        <div className="absolute left-[39px] top-[60px] bottom-[-20px] w-0.5 bg-graphite-900/40  to-transparent" />
      )}
    </div>
  );
}

function FlowBox({
  icon,
  label,
  sub,
  color,
  highlight,
}: {
  icon: React.ReactNode;
  label: string;
  sub: string;
  color: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`relative flex flex-1 flex-col items-center justify-center gap-1.5 rounded-2xl border bg-graphite-900/40 px-3 py-4 text-center transition-colors hover:-translate-y-1 ${
        highlight ? "z-10 scale-105" : ""
      } ${color}`}
    >
      <div className="mb-1">{icon}</div>
      <span className="text-sm font-bold tracking-tight">{label}</span>
      <span className="text-[10px] font-medium  opacity-75">{sub}</span>
    </div>
  );
}

function FlowArrow({
  animated,
  highlight,
}: {
  animated?: boolean;
  highlight?: boolean;
}) {
  return (
    <div className="relative flex flex-shrink-0 items-center justify-center w-8">
      <div
        className={`h-0.5 w-full ${highlight ? "bg-amber-500/50" : "bg-graphite-700"}`}
      >
        {animated && (
          <div
            className={`absolute top-1/2 left-0 h-1 w-3 -translate-y-1/2 rounded-full blur-[1px] animate-pulse-fast ${highlight ? "bg-amber-400 " : "bg-emerald-400 "}`}
            style={{ animation: "moveRight 2s infinite linear" }}
          />
        )}
      </div>
      <div
        className={`absolute -right-1 border-t-4 border-b-4 border-l-6 border-t-transparent border-b-transparent ${highlight ? "border-l-amber-500/50" : "border-l-graphite-700"}`}
        style={{ borderLeftWidth: "6px" }}
      />
      <style
        dangerouslySetInnerHTML={{
          __html: `
        @keyframes moveRight {
          0% { left: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { left: 100%; opacity: 0; }
        }
      `,
        }}
      />
    </div>
  );
}

function InfoCard({
  title,
  text,
  icon,
}: {
  title: string;
  text: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <p className="text-xs font-semibold text-graphite-200">{title}</p>
      </div>
      <p className="text-[11px] text-graphite-500 leading-relaxed">{text}</p>
    </div>
  );
}
