import { Card, CardHeader } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import type { AttackResponse, ChartDataPoint, DefenseChartDataPoint } from "@/lib/types";
import { fmt } from "@/lib/utils";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { BarChart3, TrendingDown } from "lucide-react";

interface ChartsPanelProps {
  response: AttackResponse | null;
  isLoading: boolean;
}

// Cores do tema
const EMERALD_MAIN = "#10b981";
const EMERALD_MUTED = "#34d399";
const GRAPHITE_GRID = "#3a424e";
const GRAPHITE_TEXT = "#68778a";

/**
 * Step 4 — Gráficos de Análise Avançada.
 * Curva de Loss + Comparação de Defesas.
 */
export function ChartsPanel({ response, isLoading }: ChartsPanelProps) {
  if (isLoading) {
    return (
      <div className="space-y-5 animate-slide-up">
        <Card>
          <CardHeader
            title="Gráficos de Análise"
            subtitle="Carregando dados..."
          />
          <Skeleton variant="rect" className="h-72 w-full" />
        </Card>
      </div>
    );
  }

  if (!response) {
    return (
      <div className="space-y-5 animate-slide-up">
        <Card>
          <div className="flex flex-col items-center gap-4 py-12 text-center">
            <div className="rounded-full bg-graphite-800 p-4">
              <BarChart3 className="h-8 w-8 text-graphite-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">
                Nenhum Gráfico Disponível
              </h3>
              <p className="mt-1 text-sm text-graphite-400 max-w-md">
                Execute o ataque DLG para visualizar a curva de convergência
                e a comparação de defesas.
              </p>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  // Sanitiza número para evitar Infinity/NaN que quebram Recharts
  const safe = (n: number, fallback = 1e-8): number =>
    !isFinite(n) || n <= 0 ? fallback : n;

  // Preparar dados para os gráficos
  const lossData: ChartDataPoint[] = (response.lossHistory ?? [])
    .filter((loss) => isFinite(loss) && loss > 0)
    .map((loss, idx) => ({
      iteration: idx * 10,
      loss: safe(loss, 1e-8),
    }));

  // Dados de comparação de defesas (reais ou fallback)
  const penalty = response.defensePenalty;
  const hasPenalty = penalty !== null && penalty !== undefined;
  const defenseData: DefenseChartDataPoint[] = hasPenalty
    ? [
        {
          name: "Sem Defesa",
          ssim: safe(penalty.ssimWithoutDefense, 0),
          psnr: safe(penalty.psnrWithoutDefense, 0),
          mse: safe(penalty.mseWithoutDefense, 0),
        },
        {
          name: `Com ${response.metrics?.defenseType ?? "Defesa"}`,
          ssim: safe(penalty.ssimWithDefense, 0),
          psnr: safe(penalty.psnrWithDefense, 0),
          mse: safe(penalty.mseWithDefense, 0),
        },
      ]
    : [
        {
          name: "Sem Defesa",
          ssim: safe(response.metrics?.ssimMean ?? 0, 0),
          psnr: safe(response.metrics?.psnrMean ?? 0, 0),
          mse: safe(response.metrics?.mseMean ?? 0, 0),
        },
      ];

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Curva de Perda (Loss Decay) */}
      <Card>
        <CardHeader
          title="Curva de Convergência (Loss Decay)"
          subtitle={`${lossData.length} pontos · Escala logarítmica`}
          action={
            <TrendingDown className="h-5 w-5 text-emerald-500" />
          }
        />
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={lossData}
              margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={GRAPHITE_GRID}
                strokeOpacity={0.4}
              />
              <XAxis
                dataKey="iteration"
                tick={{ fill: GRAPHITE_TEXT, fontSize: 11 }}
                tickLine={{ stroke: GRAPHITE_GRID }}
                axisLine={{ stroke: GRAPHITE_GRID }}
                label={{
                  value: "Iteração (ciclo × 10)",
                  position: "insideBottom",
                  offset: -5,
                  fill: GRAPHITE_TEXT,
                  fontSize: 11,
                }}
              />
              <YAxis
                scale="log"
                domain={["auto", "auto"]}
                tick={{ fill: GRAPHITE_TEXT, fontSize: 11 }}
                tickLine={{ stroke: GRAPHITE_GRID }}
                axisLine={{ stroke: GRAPHITE_GRID }}
                label={{
                  value: "Loss Composta",
                  angle: -90,
                  position: "insideLeft",
                  fill: GRAPHITE_TEXT,
                  fontSize: 11,
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0e1014",
                  border: "1px solid #3a424e",
                  borderRadius: "12px",
                  color: "#eaecef",
                  fontSize: "12px",
                }}
                formatter={(value: number) => [fmt(value, 6), "Loss"]}
                labelFormatter={(label: number) =>
                  `Iteração ${label}`
                }
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke={EMERALD_MAIN}
                strokeWidth={2}
                dot={false}
                activeDot={{
                  r: 5,
                  fill: EMERALD_MUTED,
                  stroke: "#0e1014",
                  strokeWidth: 2,
                }}
                animationDuration={1500}
                animationEasing="ease-out"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-3 text-center text-xs text-graphite-500">
          Loss composta: (1−α)·L2 + α·cosine + TV adaptativo + label anchoring
        </p>
      </Card>

      {/* Comparação de Defesas */}
      <Card>
        <CardHeader
          title="Impacto da Defesa na Reconstrução"
          subtitle={
            hasPenalty
              ? `Comparação com vs sem ${response.metrics?.defenseType ?? "defesa"}`
              : response.metrics?.defenseType
                ? `Defesa: ${response.metrics.defenseType}`
                : "Comparação de métricas por tipo de defesa"
          }
        />
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={defenseData}
              margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
            >
              <CartesianGrid
                strokeDasharray="3 3"
                stroke={GRAPHITE_GRID}
                strokeOpacity={0.4}
              />
              <XAxis
                dataKey="name"
                tick={{ fill: GRAPHITE_TEXT, fontSize: 11 }}
                tickLine={{ stroke: GRAPHITE_GRID }}
                axisLine={{ stroke: GRAPHITE_GRID }}
              />
              <YAxis
                tick={{ fill: GRAPHITE_TEXT, fontSize: 11 }}
                tickLine={{ stroke: GRAPHITE_GRID }}
                axisLine={{ stroke: GRAPHITE_GRID }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0e1014",
                  border: "1px solid #3a424e",
                  borderRadius: "12px",
                  color: "#eaecef",
                  fontSize: "12px",
                }}
              />
              <Legend
                wrapperStyle={{ fontSize: "11px", color: GRAPHITE_TEXT }}
              />
              <Bar
                dataKey="ssim"
                name="SSIM"
                fill={EMERALD_MAIN}
                radius={[6, 6, 0, 0]}
                animationDuration={1200}
              >
                {defenseData.map((_, idx) => (
                  <Cell
                    key={idx}
                    fill={idx === 0 ? EMERALD_MAIN : "#f59e0b"}
                    fillOpacity={idx === 0 ? 1 : 0.7}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-3 text-center text-xs text-graphite-500">
          {hasPenalty
            ? `Δ SSIM: ${penalty.deltaSsim >= 0 ? "+" : ""}${fmt(penalty.deltaSsim, 4)} · Uma defesa eficaz reduz SSIM`
            : "Execute múltiplos cenários para comparar impacto de diferentes defesas"}
        </p>
      </Card>
    </div>
  );
}
