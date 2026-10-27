import { Card, CardHeader } from "@/components/ui/Card";
import { Skeleton } from "@/components/ui/Skeleton";
import type {
  AttackResponse,
  ChartDataPoint,
  DefenseChartDataPoint,
} from "@/lib/types";
import { fmt } from "@/lib/utils";
import { defenseLabel } from "@/lib/types";
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
const EMERALD_MAIN = "#1b6c80";
const EMERALD_MUTED = "#2f70bb";
const GRAPHITE_GRID = "#e4e9ec";
const GRAPHITE_TEXT = "#526570";

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
                Os gráficos vão contar a história da busca
              </h3>
              <p className="mt-1 text-sm text-graphite-400 max-w-md">
                Primeiro, inicie um experimento. Depois, acompanhe como o erro
                da busca mudou e, quando houver uma comparação, o que a defesa alterou.
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
          name: `Com ${defenseLabel(response.metrics?.defenseType ?? "defesa")}`,
          ssim: safe(penalty.ssimWithDefense, 0),
          psnr: safe(penalty.psnrWithDefense, 0),
          mse: safe(penalty.mseWithDefense, 0),
        },
      ]
    : [
        {
          name: "Esta execução",
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
          title="Como o erro da busca mudou"
          subtitle={`${lossData.length} pontos · Escala logarítmica`}
          action={<TrendingDown className="h-5 w-5 text-emerald-500" />}
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
                  value: "Ciclos de ajuste",
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
                  value: "Erro da busca (loss)",
                  angle: -90,
                  position: "insideLeft",
                  fill: GRAPHITE_TEXT,
                  fontSize: 11,
                }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#ffffff",
                  border: "1px solid #3a424e",
                  borderRadius: "12px",
                  color: "#172b36",
                  fontSize: "12px",
                }}
                formatter={(value: number) => [fmt(value, 6), "Loss"]}
                labelFormatter={(label: number) => `Iteração ${label}`}
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
                  stroke: "#ffffff",
                  strokeWidth: 2,
                }}
                isAnimationActive={false}
                animationEasing="ease-out"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-3 text-center text-xs text-graphite-500">
          A loss é o erro usado para guiar os ajustes. Quando diminui, a busca se aproxima
          do seu objetivo numérico. Compare também as imagens para avaliar o que foi recuperado.
        </p>
      </Card>

      {/* Comparação de Defesas */}
      <Card>
        <CardHeader
          title="O que a defesa mudou?"
          subtitle={
            hasPenalty
              ? `Compare a reconstrução com e sem ${defenseLabel(response.metrics?.defenseType ?? "defesa")}`
              : response.metrics?.defenseType
                ? `Defesa: ${defenseLabel(response.metrics.defenseType)}`
                : "Sem defesa adicional: esta é sua referência inicial"
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
                  backgroundColor: "#ffffff",
                  border: "1px solid #3a424e",
                  borderRadius: "12px",
                  color: "#172b36",
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
                    fill={idx === 0 ? EMERALD_MAIN : "#a86f0c"}
                    fillOpacity={idx === 0 ? 1 : 0.7}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-3 text-center text-xs text-graphite-500">
          {hasPenalty
            ? `Mudança no SSIM: ${penalty.deltaSsim >= 0 ? "+" : ""}${fmt(penalty.deltaSsim, 4)}. Uma queda indica menor semelhança estrutural nesta comparação.`
            : "Há apenas um cenário neste resultado. Para comparar o efeito de uma defesa, ative uma na aba Experimento."}
        </p>
      </Card>
    </div>
  );
}
