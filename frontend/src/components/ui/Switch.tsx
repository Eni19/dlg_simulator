import { cn } from "@/lib/utils";

interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: string;
  description?: string;
  disabled?: boolean;
  className?: string;
}

/**
 * Switch / Toggle checkbox no estilo Flower.ai.
 */
export function Switch({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  className,
}: SwitchProps) {
  return (
    <label
      className={cn(
        "flex items-start gap-3",
        disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer",
        className
      )}
    >
      <button
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200",
          "focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:ring-offset-2 focus:ring-offset-graphite-950",
          checked
            ? "bg-emerald-500"
            : "bg-graphite-700"
        )}
      >
        <span
          className={cn(
            "inline-block h-4 w-4 rounded-full bg-white shadow-md transition-transform duration-200",
            checked ? "translate-x-6" : "translate-x-1"
          )}
        />
      </button>
      {(label || description) && (
        <div className="space-y-0.5">
          {label && <span className="text-sm font-medium text-graphite-200">{label}</span>}
          {description && (
            <p className="text-xs text-graphite-500">{description}</p>
          )}
        </div>
      )}
    </label>
  );
}
