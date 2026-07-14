import { cn } from "@/lib/utils";

interface CardProps {
  children: React.ReactNode;
  className?: string;
  /** Variante visual */
  variant?: "default" | "highlight" | "glass";
  /** Animação de entrada */
  animate?: boolean;
  onClick?: () => void;
}

/**
 * Card container reutilizável no padrão Flower.ai.
 */
export function Card({
  children,
  className,
  variant = "default",
  animate = false,
  onClick,
}: CardProps) {
  const variants = {
    default:
      "border-graphite-800/60 bg-graphite-900/40",
    highlight:
      "border-emerald-800/40 bg-emerald-950/20 ring-1 ring-emerald-500/10",
    glass:
      "border-graphite-800/40 bg-graphite-900/30 backdrop-blur-xl",
  };

  return (
    <div
      onClick={onClick}
      className={cn(
        "rounded-2xl border p-6 transition-all duration-300",
        variants[variant],
        animate && "animate-slide-up",
        onClick && "cursor-pointer hover:border-graphite-700 hover:bg-graphite-900/60",
        className
      )}
    >
      {children}
    </div>
  );
}

/**
 * Header de card com título e ação opcional.
 */
export function CardHeader({
  title,
  subtitle,
  action,
}: {
  title: React.ReactNode;
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-5 flex items-start justify-between">
      <div>
        <h3 className="text-base font-semibold text-white">{title}</h3>
        {subtitle && (
          <p className="mt-0.5 text-sm text-graphite-400">{subtitle}</p>
        )}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}
