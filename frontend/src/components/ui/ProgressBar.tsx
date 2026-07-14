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
 * Barra de progresso estilizada com gradiente esmeralda.
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

  const gradients: Record<string, string> = {
    emerald: "from-emerald-600 to-emerald-400",
    amber: "from-amber-600 to-amber-400",
    blue: "from-blue-600 to-blue-400",
  };

  return (
    <div className={cn("space-y-1.5", className)}>
      {label !== undefined && (
        <div className="flex items-center justify-between">
          <span className="text-xs text-graphite-400">{label}</span>
          <span className="text-xs font-mono font-semibold text-emerald-400">
            {clamped.toFixed(0)}%
          </span>
        </div>
      )}
      <div
        className={cn(
          "w-full overflow-hidden rounded-full bg-graphite-800",
          heights[size]
        )}
        role="progressbar"
        aria-valuenow={clamped}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={cn(
            "h-full rounded-full bg-gradient-to-r transition-all duration-300 ease-out",
            gradients[color],
            animated && "relative overflow-hidden"
          )}
          style={{ width: `${clamped}%` }}
        >
          {animated && (
            <div className="absolute inset-0 animate-pulse bg-white/10" />
          )}
        </div>
      </div>
    </div>
  );
}
