import { useId, type CSSProperties } from "react";
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
  const id = useId();
  const display =
    value !== 0 && Math.abs(value) < 0.001 ? value.toExponential(1) : value;
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex items-center justify-between gap-3">
        <label htmlFor={id} className="text-sm font-medium text-graphite-300">
          {label}
        </label>
        <output
          htmlFor={id}
          className="text-sm font-medium text-emerald-500 tabular-nums"
        >
          {display}
          {suffix}
        </output>
      </div>
      <input
        id={id}
        aria-label={label}
        type="range"
        className="lab-range"
        style={{ "--fill": `${max > min ? ((value - min) / (max - min)) * 100 : 0}%` } as CSSProperties}
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {marks && (
        <div className="flex justify-between text-xs text-graphite-500">
          {marks.map((m) => (
            <span key={m.value}>{m.label}</span>
          ))}
        </div>
      )}
    </div>
  );
}
