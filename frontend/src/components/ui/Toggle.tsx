import { cn } from "@/lib/utils";

interface ToggleOption<T extends string> {
  value: T;
  label: string;
  icon?: React.ReactNode;
}

interface ToggleProps<T extends string> {
  options: ToggleOption<T>[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
  size?: "sm" | "md";
}

/**
 * Toggle / Segment Control estilizado no padrão Flower.ai.
 * Usado para alternar entre Ambiente Centralizado e Federado.
 */
export function Toggle<T extends string>({
  options,
  value,
  onChange,
  className,
  size = "md",
}: ToggleProps<T>) {
  const isSm = size === "sm";

  return (
    <div
      className={cn(
        "inline-flex rounded-xl border border-graphite-800 bg-graphite-900/50 p-1",
        className
      )}
      role="radiogroup"
    >
      {options.map((opt) => {
        const active = value === opt.value;
        return (
          <button
            key={opt.value}
            role="radio"
            aria-checked={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "relative flex items-center gap-2 rounded-lg px-4 py-2 font-medium transition-all duration-200",
              isSm ? "px-3 py-1.5 text-sm" : "px-4 py-2 text-sm",
              active
                ? "bg-emerald-600/20 text-emerald-400 shadow-sm"
                : "text-graphite-400 hover:text-graphite-200"
            )}
          >
            {opt.icon}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
