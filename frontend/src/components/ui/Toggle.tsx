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
export function Toggle<T extends string>({
  options,
  value,
  onChange,
  className,
}: ToggleProps<T>) {
  return (
    <div
      className={cn("lab-toggle", className)}
      role="radiogroup"
      aria-label="Selecionar opção"
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          aria-checked={value === opt.value}
          role="radio"
          onClick={() => onChange(opt.value)}
        >
          {opt.icon}
          {opt.label}
        </button>
      ))}
    </div>
  );
}
