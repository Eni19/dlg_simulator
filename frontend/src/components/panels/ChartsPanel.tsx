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
import { TrendingDown } from "lucide-react";

interface ChartsPanelProps {
  response: AttackResponse | null;
  isLoading: boolean;
}

// Cores do tema
const EMERALD_MAIN = "#1d6a72";
const EMERALD_MUTED = "#2b59a6";
const GRAPHITE_GRID = "#e3dccf";
const GRAPHITE_TEXT = "#57625f";

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
        <div className="empty-sheet">
          <svg className="empty-axes" viewBox="0 0 320 150" aria-hidden="true">
            <path d="M24 8 V126 H312" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            <path d="M18 16 L24 8 L30 16 M304 120 L312 126 L304 132" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            <text x="168" y="74" textAnchor="middle" className="hand-note" fill="var(--pen)">ainda sem dados</text>
          </svg>
          <h3>Os gráficos vão contar a história da busca</h3>
          <p>
            Primeiro, inicie um experimento. Depois, acompanhe como o erro
            da busca mudou e, quando houver uma comparação, o que a defesa alterou.
          </p>
        </div>
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
                  backgroundColor: "#fffdf9",
                  border: "1px solid #e6dccb",
                  borderRadius: "12px",
                  color: "#1f2a30",
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
                  backgroundColor: "#fffdf9",
                  border: "1px solid #e6dccb",
                  borderRadius: "12px",
                  color: "#1f2a30",
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
                    fill={idx === 0 ? EMERALD_MAIN : "#c08a2b"}
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
