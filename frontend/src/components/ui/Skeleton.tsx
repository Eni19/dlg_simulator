import { cn } from "@/lib/utils";

interface SkeletonProps {
  className?: string;
  /** "line" | "circle" | "rect" | "card" */
  variant?: "line" | "circle" | "rect" | "card";
  /** Quantas linhas renderizar (apenas variant="line") */
  lines?: number;
}

/**
 * Skeleton screen — placeholder de carregamento animado.
 * Usado durante a execução do ataque DLG para indicar
 * que os resultados estão sendo processados.
 */
export function Skeleton({ className, variant = "line", lines = 3 }: SkeletonProps) {
  const base = "animate-pulse rounded-lg bg-graphite-800/60";

  if (variant === "circle") {
    return <div className={cn(base, "rounded-full", className)} />;
  }

  if (variant === "rect") {
    return <div className={cn(base, "rounded-xl", className)} />;
  }

  if (variant === "card") {
    return (
      <div className={cn("rounded-2xl border border-graphite-800 bg-graphite-900/50 p-6", className)}>
        <div className={cn(base, "mb-4 h-4 w-2/3")} />
        <div className="space-y-3">
          <div className={cn(base, "h-3 w-full")} />
          <div className={cn(base, "h-3 w-5/6")} />
          <div className={cn(base, "h-3 w-4/6")} />
        </div>
        <div className="mt-5 flex gap-3">
          <div className={cn(base, "h-8 w-20")} />
          <div className={cn(base, "h-8 w-20")} />
        </div>
      </div>
    );
  }

  // variant === "line"
  return (
    <div className={cn("space-y-2", className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <div
          key={i}
          className={cn(base, "h-3", i === lines - 1 ? "w-2/3" : "w-full")}
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  );
}
