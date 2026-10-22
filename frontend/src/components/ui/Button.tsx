import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";
import type { ButtonHTMLAttributes, ReactNode } from "react";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "danger";
  size?: "sm" | "md" | "lg";
  loading?: boolean;
  icon?: ReactNode;
  children: ReactNode;
}

/**
 * Botão estilizado no padrão Flower.ai.
 */
export function Button({
  variant = "primary",
  size = "md",
  loading = false,
  icon,
  children,
  className,
  disabled,
  ...rest
}: ButtonProps) {
  const sizes = {
    sm: "px-3.5 py-2 text-sm gap-1.5",
    md: "px-5 py-2.5 text-sm gap-2",
    lg: "px-8 py-3.5 text-base gap-2.5",
  };

  const variants = {
    primary: "bg-[#172b36] hover:bg-[#294553] text-white",
    secondary:
      "bg-graphite-800 hover:bg-graphite-700 text-graphite-200 border border-graphite-700/50",
    outline:
      "bg-transparent hover:bg-graphite-800/50 text-graphite-300 border border-graphite-700/50",
    ghost:
      "bg-transparent hover:bg-graphite-800/50 text-graphite-400 hover:text-graphite-200",
    danger:
      "bg-red-600/20 hover:bg-red-600/30 text-red-400 border border-red-500/20",
  };

  return (
    <button
      data-variant={variant}
      disabled={disabled || loading}
      className={cn(
        "inline-flex items-center justify-center rounded-md font-medium transition-colors duration-150 lab-button",
        "focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:ring-offset-2 focus:ring-offset-graphite-950",
        "disabled:cursor-not-allowed disabled:opacity-50",
        sizes[size],
        variants[variant],
        className,
      )}
      {...rest}
    >
      {loading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : icon ? (
        <span className="shrink-0">{icon}</span>
      ) : null}
      {children}
    </button>
  );
}
