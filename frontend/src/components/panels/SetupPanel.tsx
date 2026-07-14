import { Card, CardHeader } from "@/components/ui/Card";
import { Toggle } from "@/components/ui/Toggle";
import { Slider } from "@/components/ui/Slider";
import { Switch } from "@/components/ui/Switch";
import { Button } from "@/components/ui/Button";
import { DataDistributionPanel } from "./DataDistributionPanel";
import type { AttackConfig, DefenseType, Environment } from "@/lib/types";
import {
  Cpu,
  Globe,
  Network,
  Shield,
  Layers,
  Hash,
  GitBranch,
  Zap,
  Server,
  UserX,
  Wifi,
  AlertTriangle
} from "lucide-react";

interface SetupPanelProps {
  config: AttackConfig;
  onChange: (config: AttackConfig) => void;
  isRunning: boolean;
}

/**
 * Step 1 — Configuração do Cenário.
 * Permite selecionar ambiente, batch size, defesas e hiperparâmetros.
 */
export function SetupPanel({ config, onChange, isRunning }: SetupPanelProps) {
  const patch = (partial: Partial<AttackConfig>) =>
    onChange({ ...config, ...partial });

  const patchDefense = (partial: Partial<AttackConfig["defenseParams"]>) =>
    onChange({
      ...config,
      defenseParams: { ...config.defenseParams, ...partial },
    });

  const defenseOptions: { value: DefenseType; label: string }[] = [
    { value: "none", label: "Sem Defesa" },
    { value: "noisy", label: "Noisy Gradients" },
    { value: "sparse", label: "Sparsification" },
    { value: "pruning", label: "Model Pruning" },
  ];

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Ambiente */}
      <Card>
        <CardHeader
          title="Ambiente de Execução"
          subtitle="Escolha entre ataque centralizado ou federado com Flower"
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
              label: "Federado (Flower)",
              icon: <Globe className="h-4 w-4" />,
            },
          ]}
          value={config.environment}
          onChange={(env) => patch({ environment: env })}
          className="w-full"
        />
      </Card>

      {/* Configuração do Ataque */}
      <Card>
        <CardHeader
          title="Parâmetros do Ataque"
          subtitle="Configure o tamanho do batch e iterações"
        />
        <div className="grid gap-6 sm:grid-cols-2">
          <Slider
            label="Tamanho do Batch"
            suffix=" imagens"
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
            label="Iterações do Ataque"
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

        {config.environment === "federated" && config.batchSize > 1 && (
          <div className="mt-4 flex items-start gap-3 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-amber-200">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
            <div className="text-sm">
              <strong>Aviso:</strong> O ataque DLG original tem enorme dificuldade em desembaraçar imagens (batch entanglement). 
              Usar um batch maior que 1 no ambiente federado provavelmente resultará em imagens com <strong>ruído puro</strong>.
            </div>
          </div>
        )}

        <div className="mt-5 grid gap-6 sm:grid-cols-2">
          <Slider
            label="Seed"
            value={config.seed}
            min={0}
            max={999}
            step={1}
            onChange={(v) => patch({ seed: v })}
          />
        </div>
      </Card>

      {/* Federated Learning */}
      {config.environment === "federated" && (
        <Card variant="highlight" animate>
          <CardHeader
            title={
              <span className="flex items-center gap-2">
                <Network className="h-4 w-4 text-emerald-400" />
                Federated Learning
              </span>
            }
            subtitle="Configuração da simulação Flower com FedAvg manual"
          />
          <div className="grid gap-6 sm:grid-cols-2">
            <Slider
              label="Número de Clientes"
              value={config.flClients}
              min={2}
              max={10}
              step={1}
              onChange={(v) => patch({ flClients: v })}
            />
            <Slider
              label="Rodadas de Agregação"
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
              <Shield className="h-4 w-4" /> Tipo de Ameaça (Ator Malicioso)
            </h4>
            <div className="grid gap-3 sm:grid-cols-3">
              <button
                onClick={() => patch({ attackerRole: "malicious_server" })}
                className={`flex flex-col items-center gap-2 rounded-lg border p-3 text-center transition-all ${
                  config.attackerRole === "malicious_server"
                    ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.15)]"
                    : "bg-graphite-900/50 border-graphite-800 text-graphite-400 hover:border-graphite-700 hover:bg-graphite-800"
                }`}
              >
                <Server className="h-5 w-5" />
                <span className="text-xs font-medium">Servidor Curioso</span>
              </button>
              <button
                onClick={() => patch({ attackerRole: "malicious_client" })}
                className={`flex flex-col items-center gap-2 rounded-lg border p-3 text-center transition-all ${
                  config.attackerRole === "malicious_client"
                    ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.15)]"
                    : "bg-graphite-900/50 border-graphite-800 text-graphite-400 hover:border-graphite-700 hover:bg-graphite-800"
                }`}
              >
                <UserX className="h-5 w-5" />
                <span className="text-xs font-medium">Cliente Malicioso</span>
              </button>
              <button
                onClick={() => patch({ attackerRole: "mitm" })}
                className={`flex flex-col items-center gap-2 rounded-lg border p-3 text-center transition-all ${
                  config.attackerRole === "mitm"
                    ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.15)]"
                    : "bg-graphite-900/50 border-graphite-800 text-graphite-400 hover:border-graphite-700 hover:bg-graphite-800"
                }`}
              >
                <Wifi className="h-5 w-5" />
                <span className="text-xs font-medium">Man-in-the-Middle</span>
              </button>
            </div>
            
            <p className="mt-3 text-xs text-graphite-400">
              {config.attackerRole === "malicious_server" && "O Servidor Central inspeciona os gradientes enviados individualmente pelos clientes antes da agregação global."}
              {config.attackerRole === "malicious_client" && "Um cliente da rede subtrai sua própria atualização do modelo global agregado para descobrir os gradientes da vítima."}
              {config.attackerRole === "mitm" && "Um hacker na rede intercepta o pacote de dados do cliente a caminho da nuvem."}
            </p>
          </div>

          {/* Gráfico de Distribuição do Flower */}
          <div className="mt-6 border-t border-emerald-900/30 pt-6">
            <DataDistributionPanel config={config} onChange={onChange} isRunning={isRunning} />
          </div>
        </Card>
      )}

      {/* Defesas */}
      <Card>
        <CardHeader
          title={
            <span className="flex items-center gap-2">
              <Shield className="h-4 w-4 text-amber-400" />
              Defesas
            </span>
          }
          subtitle="Ative técnicas de mitigação contra o ataque DLG"
        />

        {/* Toggle de defesa ativa */}
        <div className="mb-4 flex flex-wrap gap-2">
          {defenseOptions.map((opt) => (
            <button
              key={opt.value}
              onClick={() =>
                patch({
                  defense: opt.value,
                })
              }
              disabled={isRunning}
              className={`rounded-lg px-3.5 py-2 text-xs font-medium transition-all duration-200 ${
                config.defense === opt.value
                  ? "bg-amber-600/20 text-amber-400 border border-amber-500/30"
                  : "bg-graphite-800/60 text-graphite-400 border border-transparent hover:border-graphite-700"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

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
              label="Gradientes Mantidos (Keep Ratio)"
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
              label="Proporção de Poda (Model Pruning)"
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
                    Atacante Informado
                  </p>
                  <p className="text-xs text-graphite-500 mt-0.5">
                    {config.informedAttacker
                      ? "Atacante usa o mesmo modelo podado do cliente"
                      : "Atacante usa modelo original (sem poda) — reconstrução tende a ser pior"}
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
          title="Hiperparâmetros da Perda Composta"
          subtitle="Ajuste fino da loss: L2 + Cosine + TV + Label Anchoring"
        />
        <div className="grid gap-6 sm:grid-cols-2">
          <Slider
            label="Cosine Weight"
            value={config.cosineWeight}
            min={0}
            max={1}
            step={0.05}
            onChange={(v) => patch({ cosineWeight: v })}
          />
          <Slider
            label="Label Anchor Weight"
            value={config.labelAnchorWeight}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => patch({ labelAnchorWeight: v })}
          />
          <Slider
            label="TV Coeff Start"
            suffix=""
            value={config.tvCoeffStart}
            min={1e-8}
            max={1e-2}
            step={1e-6}
            onChange={(v) => patch({ tvCoeffStart: v })}
          />
          <Slider
            label="TV Coeff End"
            suffix=""
            value={config.tvCoeffEnd}
            min={1e-10}
            max={1e-3}
            step={1e-8}
            onChange={(v) => patch({ tvCoeffEnd: v })}
          />
        </div>
      </Card>
    </div>
  );
}
