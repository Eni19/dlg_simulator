import { Card, CardHeader } from "@/components/ui/Card";
import type { AttackConfig } from "@/lib/types";
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
  Wifi
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
 * 4. DLG reconstrói imagens a partir dos gradientes
 * 5. Servidor agrega (FedAvg) os modelos atualizados
 */
export function FederatedFlowPanel({ config, isRunning }: FederatedFlowPanelProps) {
  const numClients = config.flClients;
  const numRounds = config.flRounds;
  const defense = config.defense;
  const defenseLabel =
    defense === "none"
      ? "Sem defesa"
      : defense === "noisy"
        ? "Ruído Gaussiano"
        : defense === "sparse"
          ? "Gradiente Esparso"
          : "Model Pruning";

  const isServer = config.attackerRole === "malicious_server";
  const isClient = config.attackerRole === "malicious_client";
  const isMitm = config.attackerRole === "mitm";

  const interceptorIcon = isServer ? <Server className="h-8 w-8" /> : isClient ? <Users className="h-8 w-8" /> : <Wifi className="h-8 w-8" />;
  const interceptorLabel = isServer ? "Servidor" : isClient ? "Cliente Atacante" : "Rede (MitM)";
  const interceptorDesc = isServer ? "Espiona Gradientes" : isClient ? "Infere Atualizações" : "Intercepta Pacotes";

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Explicação visual interativa do fluxo */}
      <Card>
        <CardHeader
          title="Arquitetura do Ataque Federado"
          subtitle="Acompanhe o ciclo de vida dos dados: desde o servidor até a reconstrução maliciosa."
        />
        <div className="space-y-6">
          {/* Animação de fluxo */}
          <div className="overflow-x-auto pb-4">
            <div className="flex items-center justify-between min-w-[700px] py-4 px-2">
              <FlowBox icon={<Server className="h-8 w-8" />} label="Servidor" sub="Modelo Global" color="from-blue-600/20 to-blue-900/20 border-blue-500/40 text-blue-400 shadow-[0_0_15px_rgba(59,130,246,0.15)]" />
              <FlowArrow animated />
              <FlowBox icon={<Users className="h-8 w-8" />} label={`${numClients} Clientes`} sub="Treino Local Privado" color="from-emerald-600/20 to-emerald-900/20 border-emerald-500/40 text-emerald-400 shadow-[0_0_15px_rgba(16,185,129,0.15)]" />
              <FlowArrow animated highlight />
              <FlowBox icon={interceptorIcon} label={interceptorLabel} sub={interceptorDesc} color="from-amber-600/20 to-amber-900/20 border-amber-500/50 text-amber-400 shadow-[0_0_20px_rgba(245,158,11,0.25)] ring-1 ring-amber-500/50" highlight />
              <FlowArrow animated highlight />
              <FlowBox icon={<Zap className="h-8 w-8" />} label="Ataque DLG" sub="Reconstrução LBFGS" color="from-red-600/20 to-red-900/20 border-red-500/50 text-red-400 shadow-[0_0_20px_rgba(239,68,68,0.25)] ring-1 ring-red-500/50" highlight />
              <FlowArrow animated />
              <FlowBox icon={<ArrowLeftRight className="h-8 w-8" />} label="FedAvg" sub="Agregação" color="from-violet-600/20 to-violet-900/20 border-violet-500/40 text-violet-400 shadow-[0_0_15px_rgba(139,92,246,0.15)]" />
            </div>
          </div>

          {/* Legendas em grade */}
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <InfoCard
              title="Ameaça de Privacidade"
              text="Gradientes carregam a 'assinatura' dos dados. O DLG explora essa assinatura invertendo a otimização."
              icon={<Eye className="h-4 w-4 text-amber-400" />}
            />
            <InfoCard
              title="Ciclo de Treinamento"
              text={`Por ${numRounds} rodadas, o servidor central distribui o modelo, clientes treinam e enviam atualizações, e o servidor faz a média (FedAvg).`}
              icon={<ArrowLeftRight className="h-4 w-4 text-blue-400" />}
            />
            <InfoCard
              title="Snapshot Essencial"
              text="O ataque copia o modelo exato (pesos congelados) ANTES do passo do otimizador local, garantindo compatibilidade no DLG."
              icon={<Cpu className="h-4 w-4 text-emerald-400" />}
            />
            <InfoCard
              title={defense !== "none" ? `Defesa: ${defenseLabel}` : "Sem Defesa"}
              text={
                defense === "none"
                  ? "Cenário ideal para o atacante: gradientes fiéis permitem reconstrução quase perfeita dos dados."
                  : defense === "noisy"
                    ? "Ruído mascara os gradientes, causando imagens embaçadas na reconstrução."
                    : defense === "sparse"
                      ? "Gradientes menores são podados, removendo detalhes de alta frequência."
                      : "Poda de pesos dificulta o pareamento no LBFGS se o atacante não for informado."
              }
              icon={<Shield className={`h-4 w-4 ${defense !== "none" ? "text-amber-400" : "text-graphite-500"}`} />}
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
              Detalhamento Passo a Passo
            </span>
          }
          subtitle="O que acontece nos bastidores em cada estágio"
        />
        <div className="space-y-6">
          {/* Estágios numerados */}
          <FlowStage
            num={1}
            title="Distribuição do Modelo (Servidor)"
            description="O servidor transmite os pesos globais atualizados da rede LeNet para todos os clientes participando da rodada."
            icon={<Server className="h-5 w-5" />}
            color="blue"
          />

          <FlowStage
            num={2}
            title="Treinamento Local (Clientes)"
            description={`Os ${numClients} clientes recebem o modelo global e começam a treinar localmente. Cada cliente calcula a perda (Loss) comparando as predições do modelo com seus dados privados reais. Em seguida, utilizam retropropagação (Backpropagation) para gerar os gradientes (as atualizações de peso necessárias para aprender com aqueles dados).`}
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
                    className={`relative overflow-hidden rounded-xl border px-4 py-3 text-xs transition-all hover:scale-[1.02] ${
                      isAttacker
                        ? "border-amber-500/50 bg-amber-950/30 shadow-[0_0_15px_rgba(245,158,11,0.1)]"
                        : "border-graphite-700/50 bg-graphite-900/50 hover:bg-graphite-800/60"
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {isAttacker ? (
                          <div className="p-1 rounded bg-amber-500/20"><Eye className="h-3.5 w-3.5 text-amber-400" /></div>
                        ) : (
                          <div className="p-1 rounded bg-graphite-700/50"><Cpu className="h-3.5 w-3.5 text-graphite-400" /></div>
                        )}
                        <span
                          className={`font-semibold ${
                            isAttacker ? "text-amber-400" : "text-graphite-200"
                          }`}
                        >
                          Cliente {idx}
                        </span>
                      </div>
                      {isAttacker && <span className="text-[9px] uppercase font-bold text-amber-500/80 tracking-wider">Atacante</span>}
                    </div>
                    <p className="text-graphite-400 leading-relaxed">
                      {isAttacker
                        ? isClient ? "Atua normalmente, mas usa o modelo global para deduzir as atualizações dos outros." : "Tem seus pacotes interceptados ao serem enviados."
                        : "Treina de forma honesta e prepara seus gradientes."}
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
            title={isServer ? "O Servidor Curioso (Interceptação)" : isClient ? "A Inferência (Cliente Malicioso)" : "O Homem no Meio (MitM)"}
            description={
              isServer 
                ? "Como o Servidor Central coordena a rodada, ele recebe os pacotes de gradientes brutos de cada cliente individualmente. Ao final de uma rodada de treino (ex: Rodada 1), antes de realizar a média aritmética (FedAvg), o servidor separa a atualização enviada pelo Cliente Alvo. Essa atualização contém a 'assinatura matemática' exata das imagens locais daquele cliente." 
                : isClient 
                  ? "Na transição para a próxima rodada, o servidor devolve o modelo global agregado (média de todos). O Cliente Malicioso sabe exatamente quais gradientes ele próprio enviou na rodada anterior. Através de subtração algébrica simples (Modelo Global Agregado - Contribuição Própria), o atacante isola perfeitamente os gradientes originais da vítima (funciona melhor quando há apenas 2 clientes)." 
                  : "Logo após o fim do treino local, o cliente transmite seus gradientes não-criptografados para a nuvem. Um atacante (escutando a rede Wi-Fi, ou comprometendo o provedor de internet) faz o 'sniffing' desse tráfego. Ele não tem acesso ao servidor nem ao dispositivo do cliente, mas rouba o pacote de gradientes no momento do envio."
            }
            icon={isServer ? <Server className="h-5 w-5" /> : isClient ? <Users className="h-5 w-5" /> : <Wifi className="h-5 w-5" />}
            color="amber"
            highlight
          />

          <FlowStage
            num={4}
            title="Otimização DLG (Deep Leakage from Gradients)"
            description={`É aqui que a mágica acontece, fora do fluxo oficial. O atacante congela o modelo global da rodada e gera uma "imagem de ruído" (Dummy Image). O algoritmo então passa essa imagem pelo modelo congelado para obter os "Gradientes Dummy". Uma função de perda (Loss) calcula a distância entre os Gradientes Dummy e os Gradientes Interceptados. Usando o potente otimizador L-BFGS por ${config.numIters} iterações, o atacante altera iterativamente os pixels da imagem de ruído até que a distância entre os gradientes chegue a zero. Quando os gradientes se igualam, a imagem gerada se torna idêntica à foto privada original.`}
            icon={<Zap className="h-5 w-5" />}
            color="red"
            highlight
          />

          <FlowStage
            num={5}
            title="Agregação FedAvg (Servidor)"
            description="O servidor recebe as atualizações, incluindo a do atacante. Calcula a média e o ciclo recomeça, cego ao vazamento de dados."
            icon={<ArrowLeftRight className="h-5 w-5" />}
            color="violet"
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
    violet: "border-violet-500/30 bg-violet-600/10 text-violet-400",
  };

  return (
    <div
      className={`relative rounded-xl border p-5 transition-colors hover:bg-graphite-800/20 ${
        highlight
          ? "border-amber-500/30 bg-amber-950/10 shadow-[inset_0_0_20px_rgba(245,158,11,0.05)]"
          : "border-graphite-800/50 bg-graphite-900/30"
      }`}
    >
      <div className="flex items-start gap-4">
        {/* Número do estágio */}
        <div
          className={`flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-xl border text-lg font-bold shadow-lg ${colorMap[color]}`}
        >
          {num}
        </div>
        <div className="flex-1 min-w-0 pt-0.5">
          <div className="flex items-center gap-2 mb-1.5">
            <span className={colorMap[color].split(" ")[2]}>{icon}</span>
            <h4 className="text-base font-semibold text-white tracking-tight">{title}</h4>
          </div>
          <p className="text-sm text-graphite-400 leading-relaxed mb-1">
            {description}
          </p>
          {children}
        </div>
      </div>
      {/* Conector vertical entre estágios com gradiente */}
      {num < 5 && (
        <div className="absolute left-[39px] top-[60px] bottom-[-20px] w-0.5 bg-gradient-to-b from-graphite-700 to-transparent" />
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
      className={`relative flex flex-1 flex-col items-center justify-center gap-1.5 rounded-2xl border bg-gradient-to-b px-3 py-4 text-center transition-all hover:-translate-y-1 ${
        highlight ? "z-10 scale-105" : ""
      } ${color}`}
    >
      <div className="mb-1">{icon}</div>
      <span className="text-sm font-bold tracking-tight">{label}</span>
      <span className="text-[10px] font-medium uppercase tracking-wider opacity-75">{sub}</span>
    </div>
  );
}

function FlowArrow({ animated, highlight }: { animated?: boolean; highlight?: boolean }) {
  return (
    <div className="relative flex flex-shrink-0 items-center justify-center w-8">
      <div className={`h-0.5 w-full ${highlight ? "bg-amber-500/50" : "bg-graphite-700"}`}>
        {animated && (
          <div className={`absolute top-1/2 left-0 h-1 w-3 -translate-y-1/2 rounded-full blur-[1px] animate-pulse-fast ${highlight ? "bg-amber-400 shadow-[0_0_8px_#f59e0b]" : "bg-emerald-400 shadow-[0_0_8px_#34d399]"}`} style={{ animation: 'moveRight 2s infinite linear' }} />
        )}
      </div>
      <div className={`absolute -right-1 border-t-4 border-b-4 border-l-6 border-t-transparent border-b-transparent ${highlight ? "border-l-amber-500/50" : "border-l-graphite-700"}`} style={{ borderLeftWidth: '6px' }} />
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes moveRight {
          0% { left: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { left: 100%; opacity: 0; }
        }
      `}} />
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
