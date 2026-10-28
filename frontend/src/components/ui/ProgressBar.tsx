import { cn } from "@/lib/utils";

interface ProgressBarProps {
  value: number; // 0–100
  className?: string;
  size?: "sm" | "md" | "lg";
  color?: "emerald" | "amber" | "blue";
  animated?: boolean;
  label?: string;
}

/**
 * Indicador de progresso informado, com preenchimento sólido.
 */
export function ProgressBar({
  value,
  className,
  size = "md",
  color = "emerald",
  animated = true,
  label,
}: ProgressBarProps) {
  const clamped = Math.max(0, Math.min(100, value));
  const heights = { sm: "h-1.5", md: "h-2.5", lg: "h-3.5" };

  const fills: Record<string, string> = {
    emerald: "bg-emerald-500",
    amber: "bg-[#c08a2b]",
    blue: "bg-[#2b59a6]",
  };

  return (
    <div className={cn("space-y-1.5", className)}>
      {label !== undefined && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-graphite-400">{label}</span>
          <span className="text-xs tabular-nums font-semibold text-emerald-400">
            {clamped.toFixed(0)}%
          </span>
        </div>
      )}
      <div
        className={cn(
          "w-full overflow-hidden rounded-full bg-graphite-800",
          heights[size],
        )}
        role="progressbar"
        aria-label={label || "Progresso"}
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={cn(
            "h-full rounded-full transition-[width] duration-200 ease-out",
            fills[color],
            animated && "relative overflow-hidden",
          )}
          style={{ width: `${clamped}%` }}
        >
        </div>
      </div>
    </div>
  );
}
