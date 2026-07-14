import { useState, useMemo } from "react";
import { Card, CardHeader } from "@/components/ui/Card";
import { MetricsBadge } from "@/components/ui/MetricsBadge";
import { Skeleton } from "@/components/ui/Skeleton";
import { ImageLightbox } from "@/components/ui/ImageLightbox";
import type { AttackResponse } from "@/lib/types";
import { fmt } from "@/lib/utils";
import {
  Image as ImageIcon,
  RefreshCw,
  Shield,
  TrendingDown,
  Brain,
  Cpu,
  AlertTriangle,
  ZoomIn,
  SlidersHorizontal,
} from "lucide-react";

interface ResultsPanelProps {
  response: AttackResponse | null;
  isLoading: boolean;
}

/**
 * Step 3 — Visualização dos Resultados.
 * Galeria com zoom, slider de iteração, métricas e análise de defesa.
 */
export function ResultsPanel({ response, isLoading }: ResultsPanelProps) {
  // ── Estado do zoom ──
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxImages, setLightboxImages] = useState<string[]>([]);
  const [lightboxLabels, setLightboxLabels] = useState<string[]>([]);
  const [lightboxIdx, setLightboxIdx] = useState(0);

  // ── Estado do slider de iteração ──
  const [iterSliderIndex, setIterSliderIndex] = useState(-1); // -1 = resultado final

  const openZoom = (images: string[], labels: string[], idx: number) => {
    setLightboxImages(images);
    setLightboxLabels(labels);
    setLightboxIdx(idx);
    setLightboxOpen(true);
  };

  // ── Loading ──
  if (isLoading) {
    return (
      <div className="space-y-5 animate-slide-up">
        <Card>
          <CardHeader
            title="Resultados da Reconstrução"
            subtitle="Aguarde a conclusão do ataque DLG..."
          />
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="space-y-2">
                  <Skeleton variant="rect" className="aspect-square w-full" />
                  <Skeleton variant="line" lines={1} className="w-20" />
                </div>
              ))}
            </div>
          </div>
        </Card>
      </div>
    );
  }

  // ── Vazio ──
  if (!response) {
    return (
      <div className="space-y-5 animate-slide-up">
        <Card>
          <div className="flex flex-col items-center gap-4 py-12 text-center">
            <div className="rounded-full bg-graphite-800 p-4">
              <ImageIcon className="h-8 w-8 text-graphite-500" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">
                Nenhum Resultado
              </h3>
              <p className="mt-1 text-sm text-graphite-400 max-w-md">
                Execute o ataque DLG para visualizar as imagens reconstruídas
                e as métricas de qualidade (PSNR / SSIM).
              </p>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  const hasImages =
    response.groundTruthImages.length > 0 &&
    response.reconstructedImages.length > 0 &&
    response.errorImages.length > 0;

  const hasSnapshots =
    response.snapshotImages.length > 0 &&
    response.snapshotIterations.length > 0;

  const hasPenalty = response.defensePenalty !== null;
  const hasAccuracy = response.accuracyMetrics !== null;
  const hasSparsity =
    response.sparsityStats !== null && response.sparsityStats.sparsity > 0;

  const informed = response.metrics?.informedAttacker ?? true;
  const defenseType = response.metrics?.defenseType;

  // Imagens exibidas: snapshots no slider ou resultado final
  const displayImages: string[] =
    iterSliderIndex >= 0 && hasSnapshots
      ? response.snapshotImages[iterSliderIndex]
      : response.reconstructedImages;

  const displayLabel =
    iterSliderIndex >= 0 && hasSnapshots
      ? `Iteração ${response.snapshotIterations[iterSliderIndex]}`
      : "Resultado Final";

  return (
    <div className="space-y-5 animate-slide-up">
      {/* Sumário */}
      <Card variant={response.success ? "highlight" : "default"}>
        <CardHeader title="Sumário do Ataque" subtitle={response.message} />
        <div className="grid gap-4 sm:grid-cols-3">
          <SummaryItem
            label="Dispositivo"
            value={response.device.toUpperCase()}
            icon={<Cpu className="h-3.5 w-3.5 text-graphite-500" />}
          />
          <SummaryItem
            label="Batch Size"
            value={`${response.batchSize} imagens`}
          />
          <SummaryItem
            label="Loss Final"
            value={
              response.lossHistory.length > 0
                ? fmt(
                    response.lossHistory[response.lossHistory.length - 1],
                    6,
                  )
                : "—"
            }
          />
        </div>

        {response.metrics && (
          <div className="mt-5 flex flex-wrap items-center gap-4">
            <MetricsBadge
              psnr={response.metrics.psnrMean}
              ssim={response.metrics.ssimMean}
            />
            <span className="text-xs text-graphite-500">
              PSNR ±{fmt(response.metrics.psnrStd, 1)} · SSIM ±
              {fmt(response.metrics.ssimStd, 3)} · MSE{" "}
              {fmt(response.metrics.mseMean, 6)}
            </span>
            {defenseType && defenseType !== "none" && (
              <span className="inline-flex items-center gap-1 rounded-lg bg-amber-600/10 px-2.5 py-1 text-xs font-medium text-amber-400 border border-amber-500/20">
                <Shield className="h-3 w-3" />
                {defenseType}
                {!informed && " (desinformado)"}
              </span>
            )}
          </div>
        )}
      </Card>

      {/* Acurácia do Modelo */}
      {hasAccuracy && (
        <Card>
          <CardHeader
            title={
              <span className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-blue-400" />
                Acurácia do Modelo LeNet
              </span>
            }
            subtitle={`CIFAR-100 · ${response.accuracyMetrics!.numTestSamples} amostras de teste`}
          />
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
              <p className="text-[10px] font-medium uppercase tracking-wider text-graphite-500">
                Acurácia Top-1 / Top-5
              </p>
              <p className="mt-1 font-mono text-2xl font-bold text-blue-400">
                {response.accuracyMetrics!.testAccuracy}% / {response.accuracyMetrics!.testAccuracyTop5}%
              </p>
            </div>
            <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
              <p className="text-[10px] font-medium uppercase tracking-wider text-graphite-500">
                Loss Média (Teste)
              </p>
              <p className="mt-1 font-mono text-2xl font-bold text-graphite-300">
                {fmt(response.accuracyMetrics!.testLoss, 4)}
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Esparsidade do Modelo */}
      {hasSparsity && (
        <Card>
          <CardHeader
            title="Compressão do Modelo"
            subtitle="Estatísticas de esparsidade dos pesos podáveis"
          />
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
              <p className="text-[10px] font-medium uppercase tracking-wider text-graphite-500">
                Esparsidade
              </p>
              <p className="mt-1 font-mono text-xl font-bold text-amber-400">
                {(response.sparsityStats!.sparsity * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
              <p className="text-[10px] font-medium uppercase tracking-wider text-graphite-500">
                Pesos Zerados
              </p>
              <p className="mt-1 font-mono text-xl font-bold text-graphite-300">
                {response.sparsityStats!.zero.toLocaleString()}
              </p>
            </div>
            <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
              <p className="text-[10px] font-medium uppercase tracking-wider text-graphite-500">
                Razão de Compressão
              </p>
              <p className="mt-1 font-mono text-xl font-bold text-emerald-400">
                {fmt(response.sparsityStats!.compressionRatio, 2)}x
              </p>
            </div>
          </div>
        </Card>
      )}

      {/* Penalidade da Defesa */}
      {hasPenalty && (
        <Card variant="highlight">
          <CardHeader
            title={
              <span className="flex items-center gap-2">
                <TrendingDown className="h-4 w-4 text-amber-400" />
                Impacto da Defesa na Reconstrução
              </span>
            }
            subtitle="Comparação com vs sem defesa"
          />
          <div className="grid gap-4 sm:grid-cols-3">
            <PenaltyItem
              label="Δ SSIM"
              delta={response.defensePenalty!.deltaSsim}
              from={response.defensePenalty!.ssimWithoutDefense}
              to={response.defensePenalty!.ssimWithDefense}
              fmtFn={(v) => fmt(v, 4)}
              lowerIsWorse
            />
            <PenaltyItem
              label="Δ PSNR (dB)"
              delta={response.defensePenalty!.deltaPsnr}
              from={response.defensePenalty!.psnrWithoutDefense}
              to={response.defensePenalty!.psnrWithDefense}
              fmtFn={(v) => fmt(v, 1)}
              lowerIsWorse
            />
            <PenaltyItem
              label="Δ MSE"
              delta={response.defensePenalty!.deltaMse}
              from={response.defensePenalty!.mseWithoutDefense}
              to={response.defensePenalty!.mseWithDefense}
              fmtFn={(v) => fmt(v, 6)}
              higherIsWorse
            />
          </div>
          <PenaltyInterpretation penalty={response.defensePenalty!} />
        </Card>
      )}

      {/* Galeria de Reconstrução com Slider de Iteração */}
      <Card>
        <CardHeader
          title="Galeria de Reconstrução"
          subtitle={
            hasImages
              ? "Clique em qualquer imagem para ampliar · Navegue entre iterações com o slider"
              : "Execute o ataque para visualizar as imagens reconstruídas"
          }
          action={
            hasSnapshots ? (
              <SlidersHorizontal className="h-4 w-4 text-emerald-400" />
            ) : undefined
          }
        />

        {/* ── SLIDER DE ITERAÇÃO ── */}
        {hasSnapshots && (
          <div className="mb-5 rounded-xl border border-graphite-800/60 bg-graphite-900/40 px-5 py-4">
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs font-medium text-graphite-300 flex items-center gap-1.5">
                <SlidersHorizontal className="h-3.5 w-3.5 text-emerald-400" />
                Progresso da Reconstrução
              </p>
              <span className="text-xs font-mono font-semibold text-emerald-400">
                {displayLabel}
              </span>
            </div>
            <input
              type="range"
              min={-1}
              max={response.snapshotIterations.length - 1}
              value={iterSliderIndex}
              onChange={(e) => setIterSliderIndex(Number(e.target.value))}
              className="w-full h-2 rounded-lg appearance-none cursor-pointer
                bg-graphite-800
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-5 [&::-webkit-slider-thumb]:w-5
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-emerald-500
                [&::-webkit-slider-thumb]:shadow-lg
                [&::-webkit-slider-thumb]:cursor-pointer
                [&::-webkit-slider-thumb]:transition-all
                [&::-webkit-slider-thumb]:hover:scale-110"
            />
            <div className="flex justify-between mt-1.5">
              <span
                className={`text-[10px] cursor-pointer transition-colors ${
                  iterSliderIndex === -1
                    ? "text-emerald-400 font-semibold"
                    : "text-graphite-500 hover:text-graphite-400"
                }`}
                onClick={() => setIterSliderIndex(-1)}
              >
                Final
              </span>
              {response.snapshotIterations.map((iter, i) => (
                <span
                  key={i}
                  className={`text-[10px] cursor-pointer transition-colors ${
                    iterSliderIndex === i
                      ? "text-emerald-400 font-semibold"
                      : "text-graphite-500 hover:text-graphite-400"
                  }`}
                  onClick={() => setIterSliderIndex(i)}
                >
                  {iter}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* ── GRID DE IMAGENS (AMPLIADO) ── */}
        {hasImages ? (
          <div className="space-y-8">
            {Array.from({ length: response.batchSize }).map((_, i) => {
              const gtImg = response.groundTruthImages[i];
              const errImg = response.errorImages[i];
              const predLabel =
                response.reconstructedLabels?.[i] ?? null;

              // Monta listas para o lightbox (3 imagens por item)
              const lbImages = [gtImg, displayImages[i] ?? "", errImg];
              const lbLabels = [
                `Ground Truth #${i + 1}`,
                `Reconstrução #${i + 1} — ${displayLabel}`,
                `Erro Absoluto #${i + 1}`,
              ];

              return (
                <div
                  key={i}
                  className="overflow-hidden rounded-xl border border-graphite-800/60 bg-graphite-900/40"
                >
                  <div className="px-5 py-3 border-b border-graphite-800/40 flex items-center justify-between">
                    <p className="text-base font-semibold text-graphite-200">
                      Imagem #{i + 1}
                      {predLabel !== null && (
                        <span className="ml-2 text-xs font-normal text-graphite-500">
                          Label predito: {predLabel}
                        </span>
                      )}
                    </p>
                    <span className="text-[10px] text-graphite-600">
                      Clique para ampliar
                    </span>
                  </div>
                  {/* Layout ampliado: 3 colunas com imagens GRANDES */}
                  <div className="grid grid-cols-3 gap-1 bg-graphite-800/40 p-1">
                    {/* Ground Truth */}
                    <button
                      className="bg-graphite-950/60 p-4 text-left transition-colors hover:bg-graphite-900/60 group rounded-lg"
                      onClick={() => openZoom(lbImages, lbLabels, 0)}
                    >
                      <p className="mb-3 text-center text-xs font-semibold uppercase tracking-wider text-graphite-400">
                        Ground Truth
                      </p>
                      <div className="relative mx-auto flex items-center justify-center" style={{ width: 224, height: 224 }}>
                        <img
                          src={`data:image/png;base64,${gtImg}`}
                          alt={`Ground Truth ${i + 1}`}
                          className="rounded-lg"
                          style={{
                            width: 224,
                            height: 224,
                            imageRendering: "pixelated",
                            objectFit: "contain",
                          }}
                        />
                        <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-lg" style={{ background: "rgba(0,0,0,0.5)" }}>
                          <span className="flex items-center gap-1.5 rounded-lg bg-black/80 px-3 py-2 text-sm font-medium text-white">
                            <ZoomIn className="h-4 w-4" />
                            Ampliar
                          </span>
                        </div>
                      </div>
                    </button>

                    {/* Reconstrução */}
                    <button
                      className="bg-graphite-950/60 p-4 text-left transition-colors hover:bg-graphite-900/60 group rounded-lg"
                      onClick={() => openZoom(lbImages, lbLabels, 1)}
                    >
                      <p className="mb-3 text-center text-xs font-semibold uppercase tracking-wider text-emerald-400">
                        Reconstrução
                      </p>
                      <div className="relative mx-auto flex items-center justify-center" style={{ width: 224, height: 224 }}>
                        <img
                          src={`data:image/png;base64,${displayImages[i] ?? ""}`}
                          alt={`Reconstrução ${i + 1}`}
                          className="rounded-lg"
                          style={{
                            width: 224,
                            height: 224,
                            imageRendering: "pixelated",
                            objectFit: "contain",
                          }}
                        />
                        <div className="absolute inset-0 flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-lg" style={{ background: "rgba(0,0,0,0.4)" }}>
                          {response.perImageMetrics && response.perImageMetrics[i] && iterSliderIndex === -1 && (
                            <span className="rounded bg-emerald-500/90 px-3 py-1.5 text-sm font-bold text-white shadow-lg border border-emerald-400/30">
                              SSIM: {response.perImageMetrics[i].ssim.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>
                    </button>

                    {/* Erro Absoluto */}
                    <button
                      className="bg-graphite-950/60 p-4 text-left transition-colors hover:bg-graphite-900/60 group rounded-lg"
                      onClick={() => openZoom(lbImages, lbLabels, 2)}
                    >
                      <p className="mb-3 text-center text-xs font-semibold uppercase tracking-wider text-red-400">
                        Erro Absoluto
                      </p>
                      <div className="relative mx-auto flex items-center justify-center" style={{ width: 224, height: 224 }}>
                        <img
                          src={`data:image/png;base64,${errImg}`}
                          alt={`Erro absoluto ${i + 1}`}
                          className="rounded-lg"
                          style={{
                            width: 224,
                            height: 224,
                            imageRendering: "pixelated",
                            objectFit: "contain",
                          }}
                        />
                        <div className="absolute inset-0 flex flex-col items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-lg" style={{ background: "rgba(0,0,0,0.4)" }}>
                          {response.perImageMetrics && response.perImageMetrics[i] && iterSliderIndex === -1 && (
                            <span className="rounded bg-red-500/90 px-3 py-1.5 text-sm font-bold text-white shadow-lg border border-red-400/30">
                              MSE: {response.perImageMetrics[i].mse.toFixed(4)}
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-4">
            {Array.from({ length: response.batchSize }).map((_, i) => (
              <div
                key={i}
                className="relative overflow-hidden rounded-xl border border-graphite-800/60 bg-graphite-900/60 p-1"
              >
                <div className="aspect-square flex items-center justify-center bg-graphite-950/50 rounded-lg">
                  <div className="text-center">
                    <RefreshCw className="mx-auto h-6 w-6 text-graphite-600" />
                    <p className="mt-2 text-xs text-graphite-500">
                      Rec #{i + 1}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* ── LIGHTBOX (Zoom) ── */}
      {lightboxOpen && (
        <ImageLightbox
          images={lightboxImages}
          currentIndex={lightboxIdx}
          labels={lightboxLabels}
          onClose={() => setLightboxOpen(false)}
          onPrev={() =>
            setLightboxIdx((p) =>
              p - 1 < 0 ? lightboxImages.length - 1 : p - 1,
            )
          }
          onNext={() =>
            setLightboxIdx((p) =>
              p + 1 >= lightboxImages.length ? 0 : p + 1,
            )
          }
        />
      )}
    </div>
  );
}

/* ── Subcomponentes ──────────────────────────────────────────────────── */

function SummaryItem({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
      <p className="text-[10px] font-medium uppercase tracking-wider text-graphite-500 flex items-center gap-1.5">
        {icon}
        {label}
      </p>
      <p className="mt-0.5 font-mono text-sm font-semibold text-emerald-400">
        {value}
      </p>
    </div>
  );
}

function PenaltyItem({
  label,
  delta,
  from,
  to,
  fmtFn,
  lowerIsWorse,
  higherIsWorse,
}: {
  label: string;
  delta: number;
  from: number;
  to: number;
  fmtFn: (v: number) => string;
  lowerIsWorse?: boolean;
  higherIsWorse?: boolean;
}) {
  const isBad =
    (lowerIsWorse && delta < 0) || (higherIsWorse && delta > 0);
  const sign = delta >= 0 ? "+" : "";

  return (
    <div className="rounded-lg border border-graphite-800/40 bg-graphite-900/30 px-4 py-3">
      <p className="text-[10px] font-medium uppercase tracking-wider text-graphite-500">
        {label}
      </p>
      <p
        className={`mt-1 font-mono text-xl font-bold ${isBad ? "text-red-400" : "text-emerald-400"}`}
      >
        {sign}
        {fmtFn(delta)}
      </p>
      <p className="mt-0.5 text-[10px] text-graphite-500">
        {fmtFn(from)} → {fmtFn(to)}
      </p>
    </div>
  );
}

function PenaltyInterpretation({
  penalty,
}: {
  penalty: NonNullable<AttackResponse["defensePenalty"]>;
}) {
  let interpretation: string;
  let color: string;
  let Icon: typeof Shield;

  if (penalty.deltaSsim < -0.1) {
    interpretation =
      "A defesa causa degradação SIGNIFICATIVA na reconstrução — forte proteção de privacidade.";
    color = "text-emerald-400";
    Icon = Shield;
  } else if (penalty.deltaSsim < -0.03) {
    interpretation =
      "A defesa causa degradação MODERADA na reconstrução — proteção parcial.";
    color = "text-amber-400";
    Icon = AlertTriangle;
  } else {
    interpretation =
      "A defesa tem impacto MÍNIMO na reconstrução — proteção fraca ou inconclusiva.";
    color = "text-red-400";
    Icon = AlertTriangle;
  }

  return (
    <div className="mt-4 flex items-start gap-3 rounded-lg bg-graphite-900/40 border border-graphite-800/40 px-4 py-3">
      <Icon className="mt-0.5 h-4 w-4 flex-shrink-0" />
      <p className={`text-xs ${color}`}>{interpretation}</p>
    </div>
  );
}
