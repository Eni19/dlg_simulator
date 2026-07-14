import { cn } from "@/lib/utils";

interface SliderProps {
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  label?: string;
  suffix?: string;
  className?: string;
  disabled?: boolean;
  marks?: { value: number; label: string }[];
}

/**
 * Slider numérico customizado no estilo Flower.ai.
 * Usado para selecionar batch size, intensidade de poda, etc.
 */
export function Slider({
  value,
  min,
  max,
  step = 1,
  onChange,
  label,
  suffix = "",
  className,
  disabled = false,
  marks,
}: SliderProps) {
  const pct = ((value - min) / (max - min)) * 100;

  return (
    <div className={cn("space-y-2", className)}>
      {(label || suffix) && (
        <div className="flex items-center justify-between">
          {label && (
            <span className="text-sm font-medium text-graphite-300">{label}</span>
          )}
          <span className="text-sm font-semibold text-emerald-400 tabular-nums">
            {value}{suffix}
          </span>
        </div>
      )}

      <div className="relative">
        {/* Track */}
        <div className="h-2 rounded-full bg-graphite-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-emerald-700 to-emerald-500 transition-all duration-150"
            style={{ width: `${pct}%` }}
          />
        </div>

        {/* Thumb */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(Number(e.target.value))}
          className={cn(
            "absolute inset-0 h-2 w-full cursor-pointer appearance-none bg-transparent",
            "focus:outline-none",
            "[&::-webkit-slider-thumb]:appearance-none",
            "[&::-webkit-slider-thumb]:h-5",
            "[&::-webkit-slider-thumb]:w-5",
            "[&::-webkit-slider-thumb]:rounded-full",
            "[&::-webkit-slider-thumb]:bg-emerald-500",
            "[&::-webkit-slider-thumb]:shadow-lg",
            "[&::-webkit-slider-thumb]:shadow-emerald-500/30",
            "[&::-webkit-slider-thumb]:transition-transform",
            "[&::-webkit-slider-thumb]:duration-150",
            "[&::-webkit-slider-thumb]:hover:scale-110",
            "[&::-webkit-slider-thumb]:active:scale-95",
            "disabled:cursor-not-allowed disabled:opacity-40"
          )}
        />
      </div>

      {/* Marcas opcionais */}
      {marks && (
        <div className="flex justify-between px-0.5">
          {marks.map((m) => (
            <span
              key={m.value}
              className="text-[10px] text-graphite-500"
            >
              {m.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
