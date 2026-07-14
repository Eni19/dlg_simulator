import { cn, fmtPSNR, fmtSSIM, psnrColor, ssimColor } from "@/lib/utils";
import { Zap, Eye } from "lucide-react";

interface MetricsBadgeProps {
  psnr: number;
  ssim: number;
  className?: string;
  size?: "sm" | "md";
}

/**
 * Badge compacto exibindo métricas PSNR e SSIM.
 * Design inspirado no Flower.ai — minimalista com cores semânticas.
 */
export function MetricsBadge({ psnr, ssim, className, size = "md" }: MetricsBadgeProps) {
  const isSm = size === "sm";

  return (
    <div
      className={cn(
        "inline-flex items-center gap-3 rounded-xl border border-graphite-800/60 bg-graphite-900/70 px-4 py-2 backdrop-blur-sm",
        className
      )}
    >
      <div className="flex items-center gap-1.5">
        <Zap className={cn("text-emerald-500", isSm ? "h-3.5 w-3.5" : "h-4 w-4")} />
        <span className={cn("tabular-nums font-mono font-medium", isSm ? "text-xs" : "text-sm", psnrColor(psnr))}>
          {fmtPSNR(psnr)}
        </span>
      </div>
      <div className="h-4 w-px bg-graphite-700" />
      <div className="flex items-center gap-1.5">
        <Eye className={cn("text-emerald-500", isSm ? "h-3.5 w-3.5" : "h-4 w-4")} />
        <span className={cn("tabular-nums font-mono font-medium", isSm ? "text-xs" : "text-sm", ssimColor(ssim))}>
          {fmtSSIM(ssim)}
        </span>
      </div>
    </div>
  );
}

/**
 * Grade de badges de métricas para exibição lado a lado.
 */
export function MetricsGrid({
  metrics,
  className,
}: {
  metrics: { psnr: number; ssim: number; label?: string }[];
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap gap-3", className)}>
      {metrics.map((m, i) => (
        <div key={i} className="space-y-1">
          {m.label && (
            <p className="text-xs font-medium text-graphite-400">{m.label}</p>
          )}
          <MetricsBadge psnr={m.psnr} ssim={m.ssim} size="sm" />
        </div>
      ))}
    </div>
  );
}
