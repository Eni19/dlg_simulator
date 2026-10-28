import { useState, useEffect } from "react";
import { Card, CardHeader } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Toggle } from "@/components/ui/Toggle";
import { Slider } from "@/components/ui/Slider";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { AttackConfig } from "@/lib/types";
import { PieChart, Loader2, Database } from "lucide-react";

interface DataDistributionPanelProps {
  config: AttackConfig;
  onChange: (config: AttackConfig) => void;
  isRunning: boolean;
}

export function DataDistributionPanel({
  config,
  onChange,
  isRunning,
}: DataDistributionPanelProps) {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchDistribution = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/data-distribution", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          num_clients: config.flClients,
          partitioner_type: config.partitionerType,
          alpha: config.dirichletAlpha,
        }),
      });
      const json = await res.json();
      if (!res.ok)
        throw new Error(json.detail || "Erro ao buscar distribuição");

      // Transform data for Recharts
      // Input: { "Cliente 0": { "0": 10, "1": 5 }, "Cliente 1": ... }
      // Output: [ { name: "Cliente 0", "0": 10, "1": 5 }, ... ]
      const formatted = Object.entries(json.distribution).map(
        ([clientName, counts]: [string, any]) => {
          return {
            name: clientName,
            ...counts,
          };
        },
      );
      setData(formatted);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Apaga os dados antigos se a configuração mudar
    setData([]);
  }, [config.flClients, config.partitionerType, config.dirichletAlpha]);

  // Lista única de classes (0-99)
  const classes = Array.from({ length: 100 }, (_, i) => i.toString());

  // Cores dinâmicas para as classes
  const getColor = (i: number) => `hsl(${(i * 137.5) % 360}, 70%, 50%)`;

  return (
    <Card className="animate-slide-up border-emerald-900/30">
      <CardHeader
        title={
          <span className="flex items-center gap-2">
            <Database className="h-5 w-5 text-emerald-400" />
            Uma prévia da divisão dos exemplos
          </span>
        }
        subtitle="Compare uma divisão mais equilibrada com uma em que cada participante recebe proporções diferentes de categorias."
      />
      <div className="space-y-6">
        <div className="grid gap-6 sm:grid-cols-2">
          <div>
            <p className="mb-2 text-xs font-semibold text-graphite-400 ">
              Como dividir os exemplos
            </p>
            <Toggle
              options={[
                { value: "iid", label: "Equilibrado (IID)" },
                { value: "dirichlet", label: "Desigual (Dirichlet)" },
              ]}
              value={config.partitionerType}
              onChange={(val) => onChange({ ...config, partitionerType: val })}
              className="w-full"
            />
          </div>

          {config.partitionerType === "dirichlet" && (
            <Slider
              label="Equilíbrio entre categorias (alfa)"
              value={config.dirichletAlpha}
              min={0.1}
              max={10.0}
              step={0.1}
              onChange={(v) => onChange({ ...config, dirichletAlpha: v })}
              marks={[
                { value: 0.1, label: "0.1 (Extremo)" },
                { value: 1.0, label: "1.0" },
                { value: 10.0, label: "10.0 (Quase IID)" },
              ]}
            />
          )}
        </div>

        <div className="flex justify-end">
          <Button
            onClick={fetchDistribution}
            disabled={loading || isRunning}
            variant="outline"
            className="gap-2"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <PieChart className="h-4 w-4" />
            )}
            Ver a divisão das imagens
          </Button>
        </div>

        {error && (
          <p className="text-sm text-red-400 border border-red-500/20 bg-red-500/10 p-3 rounded-lg">
            {error}
          </p>
        )}

        {data.length > 0 && (
          <div className="rounded-xl border border-graphite-800/60 bg-graphite-900/40 p-5 mt-4">
            <h4 className="text-sm font-semibold text-graphite-200 mb-4 text-center">
              Imagens de cada categoria por participante
            </h4>
            <div className="h-[400px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={data}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <XAxis
                    dataKey="name"
                    stroke="#57625f"
                    fontSize={12}
                    tickMargin={10}
                  />
                  <YAxis stroke="#57625f" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#fffdf9",
                      border: "1px solid #e6dccb",
                      borderRadius: "12px",
                      color: "#1f2a30",
                    }}
                    itemStyle={{ fontSize: "12px" }}
                    labelStyle={{
                      fontWeight: "bold",
                      marginBottom: "8px",
                      color: "#1f2a30",
                    }}
                    // Limita a exibição do tooltip para não estourar a tela com 100 classes
                    formatter={(value: number, name: string) => [
                      value,
                      `Classe ${name}`,
                    ]}
                    filterNull={false}
                  />
                  {classes.map((cls, i) => (
                    <Bar
                      key={cls}
                      dataKey={cls}
                      stackId="a"
                      fill={getColor(i)}
                    />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-4 text-xs text-graphite-500 text-center">
              Cada cor representa uma das 100 classes do CIFAR-100.
              {config.partitionerType === "iid"
                ? " No particionamento IID, as cores são distribuídas uniformemente."
                : " No particionamento Dirichlet, os clientes possuem focos em classes específicas (Non-IID)."}
            </p>
          </div>
        )}
      </div>
    </Card>
  );
}
