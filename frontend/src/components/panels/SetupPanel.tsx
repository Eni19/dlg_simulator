import { Card, CardHeader } from "@/components/ui/Card";
import { Toggle } from "@/components/ui/Toggle";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";
import { DataDistributionPanel } from "./DataDistributionPanel";
import { ShowcaseSetup } from "./ShowcaseSetup";
import type { AttackConfig, DefenseType, Environment } from "@/lib/types";
import {
  Cpu,
  Globe,
  Network,
  Shield,
  Server,
  UserX,
  Wifi,
  AlertTriangle,
} from "lucide-react";

interface SetupPanelProps {
  config: AttackConfig;
  onChange: (config: AttackConfig) => void;
  isRunning: boolean;
  isShowcase?: boolean;
  onStart?: () => void;
}

/**
 * Step 1 — Configuração do Cenário.
 * Permite selecionar ambiente, batch size, defesas e hiperparâmetros.
 */
export function SetupPanel({ config, onChange, isRunning, isShowcase = false, onStart }: SetupPanelProps) {
  if (isShowcase) return <ShowcaseSetup config={config} onChange={onChange} onStart={onStart} isRunning={isRunning} />;
  const patch = (partial: Partial<AttackConfig>) =>
    onChange({ ...config, ...partial });

  const patchDefense = (partial: Partial<AttackConfig["defenseParams"]>) =>
    onChange({
      ...config,
      defenseParams: { ...config.defenseParams, ...partial },
    });

  const defenseOptions: { value: DefenseType; label: string }[] = [
    { value: "none", label: "Sem defesa" },
    { value: "noisy", label: "Adicionar ruído" },
    { value: "sparse", label: "Compartilhar menos pistas" },
    { value: "pruning", label: "Remover conexões (poda)" },
  ];

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Ambiente */}
      <Card>
        <CardHeader
          title="Qual cenário você quer explorar?"
          subtitle="Comece pelo centralizado, com uma tentativa direta de reconstrução. No federado, vários participantes aprendem juntos."
        />
        <Toggle<Environment>
          options={[
            {
              value: "centralized",
              label: "Centralizado",
              icon: <Cpu className="h-4 w-4" />,
            },
            {
              value: "federated",
              label: "Federado",
              icon: <Globe className="h-4 w-4" />,
            },
          ]}
          value={config.environment}
          onChange={(env) => patch({ environment: env })}
          className="w-full"
        />
      </Card>

      <Card>
        <CardHeader title="Onde fazer os cálculos?" subtitle="Use CPU ou a placa de vídeo disponível no computador que executa o laboratório." />
        <label htmlFor="experiment-device" className="block text-sm font-medium mb-2">
          Dispositivo
        </label>
        <select
          id="experiment-device"
          value={config.device}
          onChange={(event) => patch({ device: event.target.value as AttackConfig["device"] })}
          className="w-full rounded-lg border border-current/20 bg-transparent px-3 py-2 text-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2"
          aria-describedby="device-help"
        >
          <option value="cpu">CPU</option>
          <option value="cuda">GPU NVIDIA (CUDA)</option>
          <option value="auto">Automático (GPU quando disponível)</option>
        </select>
        <p id="device-help" className="mt-2 text-sm text-graphite-400">
          CPU é um bom ponto de partida. A GPU usa sua placa NVIDIA, mas pode ser mais lenta
          neste experimento pequeno. Automático prefere GPU quando disponível e usa CPU caso contrário.
        </p>
      </Card>

      {/* Configuração do Ataque */}
      <Card>
        <CardHeader
          title="Vamos começar com uma imagem?"
          subtitle="Uma imagem e 50 iterações são um teste inicial. Depois, aumente um valor por vez para observar a diferença."
        />
        <div className="grid gap-6 sm:grid-cols-2">
          <Slider
            label="Imagens por vez (batch)"
            suffix={config.batchSize === 1 ? " imagem" : " imagens"}
            value={config.batchSize}
            min={1}
            max={8}
            step={1}
            onChange={(v) => patch({ batchSize: v })}
            marks={[
              { value: 1, label: "1" },
              { value: 4, label: "4" },
              { value: 8, label: "8" },
            ]}
          />
          <Slider
            label="Tentativas de ajuste (iterações)"
            value={config.numIters}
            min={50}
            max={5000}
            step={50}
            onChange={(v) => patch({ numIters: v })}
            marks={[
              { value: 50, label: "50" },
              { value: 1000, label: "1k" },
              { value: 5000, label: "5k" },
            ]}
          />
        </div>

        <p className="mt-3 text-sm text-graphite-400">
          Mais iterações dão mais oportunidades de ajuste e aumentam o tempo de espera.
          Elas podem melhorar a reconstrução, mas o resultado também depende das imagens e das defesas.
        </p>

        {config.environment === "federated" && config.batchSize > 1 && (
          <div className="mt-4 flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-amber-200">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
            <div className="text-sm">
              <strong>Um desafio maior:</strong> as pistas de várias imagens se
              misturam no mesmo cálculo. Separá-las fica mais difícil. No cenário
              federado, use uma imagem por vez como referência antes de aumentar o grupo.
            </div>
          </div>
        )}

        <div className="mt-5 grid gap-6 sm:grid-cols-2">
          <Slider
            label="Início aleatório (seed)"
            value={config.seed}
            min={0}
            max={999}
            step={1}
            onChange={(v) => patch({ seed: v })}
          />
        </div>
        <p className="mt-2 text-sm text-graphite-400">
          A seed define o ponto de partida dos cálculos aleatórios. Pode manter esse valor no primeiro teste.
        </p>
      </Card>

      {/* Federated Learning */}
      {config.environment === "federated" && (
        <Card variant="highlight" animate>
          <CardHeader
            title={
              <span className="flex items-center gap-2">
                <Network className="h-4 w-4 text-emerald-400" />
                Aprendizado em conjunto
              </span>
            }
            subtitle="Cada participante, chamado de cliente, treina com seus exemplos. A cada rodada, o servidor combina o que eles aprenderam."
          />
          <div className="grid gap-6 sm:grid-cols-2">
            <Slider
              label="Participantes (clientes)"
              value={config.flClients}
              min={2}
              max={10}
              step={1}
              onChange={(v) => patch({ flClients: v })}
            />
            <Slider
              label="Rodadas de aprendizado"
              value={config.flRounds}
              min={1}
              max={50}
              step={1}
              onChange={(v) => patch({ flRounds: v })}
            />
          </div>

          {/* Seleção do Papel do Atacante */}
          <div className="mt-6 border-t border-emerald-900/30 pt-6">
            <h4 className="mb-3 text-sm font-medium text-emerald-100 flex items-center gap-2">
              <Shield className="h-4 w-4" /> Quem tem acesso às pistas?
            </h4>
            <div className="grid gap-3 sm:grid-cols-3">
              <button
                onClick={() => patch({ attackerRole: "malicious_server" })}
                className={`flex flex-col items-center gap-2 rounded-lg border p-3 text-center transition-colors ${
                  config.attackerRole === "malicious_server"
                    ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-300 "
                    : "bg-graphite-900/50 border-graphite-800 text-graphite-400 hover:border-graphite-700 hover:bg-graphite-800"
                }`}
              >
                <Server className="h-5 w-5" />
                <span className="text-sm font-medium">O servidor</span>
              </button>
              <button
                onClick={() => patch({ attackerRole: "malicious_client" })}
                className={`flex flex-col items-center gap-2 rounded-lg border p-3 text-center transition-colors ${
                  config.attackerRole === "malicious_client"
                    ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-300 "
                    : "bg-graphite-900/50 border-graphite-800 text-graphite-400 hover:border-graphite-700 hover:bg-graphite-800"
                }`}
              >
                <UserX className="h-5 w-5" />
                <span className="text-sm font-medium">Outro participante</span>
              </button>
              <button
                onClick={() => patch({ attackerRole: "mitm" })}
                className={`flex flex-col items-center gap-2 rounded-lg border p-3 text-center transition-colors ${
                  config.attackerRole === "mitm"
                    ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-300 "
                    : "bg-graphite-900/50 border-graphite-800 text-graphite-400 hover:border-graphite-700 hover:bg-graphite-800"
                }`}
              >
                <Wifi className="h-5 w-5" />
                <span className="text-sm font-medium">Alguém na rede</span>
              </button>
            </div>

            <p className="mt-3 text-sm text-graphite-400">
              {config.attackerRole === "malicious_server" &&
                "O coordenador do aprendizado observa as pistas enviadas por cada participante antes de combiná-las."}
              {config.attackerRole === "malicious_client" &&
                "Um participante tenta separar sua própria contribuição da atualização conjunta para estimar o que veio dos outros."}
              {config.attackerRole === "mitm" &&
                "Uma pessoa entre o participante e o servidor observa as atualizações em trânsito. Esse acesso é simulado no laboratório."}
            </p>
          </div>

          {/* Gráfico de Distribuição do Flower */}
          <div className="mt-6 border-t border-emerald-900/30 pt-6">
            <DataDistributionPanel
              config={config}
              onChange={onChange}
              isRunning={isRunning}
            />
          </div>
        </Card>
      )}

      {/* Defesas */}
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-emerald-500" />
              Defesas
            </span>
          }
          subtitle="Primeiro, experimente sem defesa. Depois, escolha uma forma de dificultar a reconstrução e compare o resultado."
        />

        {/* Escolha da defesa */}
        <Toggle
          className="mb-4"
          options={defenseOptions}
          value={config.defense}
          onChange={(defense) => patch({ defense })}
          label="Defesa"
        />

        <p className="text-sm text-graphite-400">
          {config.defense === "none" && "As pistas chegam ao ataque sem uma defesa adicional. Use esse resultado como referência."}
          {config.defense === "noisy" && "O ruído acrescenta variações aleatórias aos gradientes, como pequenas interferências nas pistas compartilhadas."}
          {config.defense === "sparse" && "Parte dos gradientes é zerada antes do compartilhamento. A técnica é chamada de esparsificação."}
          {config.defense === "pruning" && "A poda zera parte dos pesos, os números que representam as conexões do modelo. Isso muda as pistas disponíveis ao ataque."}
        </p>

        {/* Parâmetros da defesa selecionada */}
        {config.defense === "noisy" && (
          <div className="mt-4">
            <Slider
              label="Intensidade do Ruído (σ)"
              value={config.defenseParams.noiseStd}
              min={0.01}
              max={1.0}
              step={0.01}
              onChange={(v) => patchDefense({ noiseStd: v })}
              marks={[
                { value: 0.01, label: "0.01" },
                { value: 0.5, label: "0.5" },
                { value: 1.0, label: "1.0" },
              ]}
            />
          </div>
        )}

        {config.defense === "sparse" && (
          <div className="mt-4">
            <Slider
              label="Gradientes mantidos"
              suffix="%"
              value={Math.round(config.defenseParams.keepRatio * 100)}
              min={1}
              max={100}
              step={1}
              onChange={(v) => patchDefense({ keepRatio: v / 100 })}
              marks={[
                { value: 1, label: "1%" },
                { value: 50, label: "50%" },
                { value: 100, label: "100%" },
              ]}
            />
          </div>
        )}

        {config.defense === "pruning" && (
          <div className="mt-4 space-y-5">
            <Slider
              label="Conexões removidas pela poda"
              suffix="%"
              value={Math.round(config.defenseParams.pruningAmount * 100)}
              min={0}
              max={90}
              step={5}
              onChange={(v) => patchDefense({ pruningAmount: v / 100 })}
              marks={[
                { value: 0, label: "0%" },
                { value: 30, label: "30%" },
                { value: 60, label: "60%" },
                { value: 90, label: "90%" },
              ]}
            />

            {/* Atacante Informado vs Desinformado */}
            <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-graphite-200">
                    O ataque conhece a poda?
                  </p>
                  <p className="text-xs text-graphite-500 mt-0.5">
                    {config.informedAttacker
                      ? "Sim. A reconstrução usa a mesma versão do modelo, com as conexões removidas."
                      : "Não. A reconstrução usa o modelo anterior à poda, com informações diferentes das usadas pelo participante."}
                  </p>
                </div>
                <Switch
                  checked={config.informedAttacker}
                  onChange={(v) => patch({ informedAttacker: v })}
                />
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Hiperparâmetros da Perda */}
      <Card>
        <CardHeader
          title="Ajustes avançados da busca"
          subtitle="Você pode manter os valores prontos. Abra esta parte quando quiser investigar como o algoritmo escolhe os próximos ajustes."
        />
        <details>
          <summary className="cursor-pointer text-sm font-medium">Explorar os pesos e a suavização</summary>
          <p className="mt-3 text-sm text-graphite-400">
            Os pesos equilibram os sinais que orientam a reconstrução.
            A suavização tenta reduzir variações bruscas entre pontos vizinhos da imagem.
          </p>
        <div className="mt-4 grid gap-6 sm:grid-cols-2">
          <Slider
            label="Direção dos gradientes (cosseno)"
            value={config.cosineWeight}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => patch({ cosineWeight: v })}
          />
          <Slider
            label="Peso da categoria estimada"
            value={config.labelAnchorWeight}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => patch({ labelAnchorWeight: v })}
          />
          <Slider
            label="Suavização inicial da imagem (TV)"
            suffix=""
            value={config.tvCoeffStart}
            min={1e-8}
            max={1e-2}
            step={1e-6}
            onChange={(v) => patch({ tvCoeffStart: v })}
          />
          <Slider
            label="Suavização final da imagem (TV)"
            suffix=""
            value={config.tvCoeffEnd}
            min={1e-10}
            max={1e-3}
            step={1e-8}
            onChange={(v) => patch({ tvCoeffEnd: v })}
          />
        </div>
        </details>
      </Card>
    </div>
  );
}
