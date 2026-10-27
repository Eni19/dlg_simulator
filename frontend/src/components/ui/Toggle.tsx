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
  label?: string;
}
export function Toggle<T extends string>({
  options,
  value,
  onChange,
  className,
  label = "Selecionar opção",
}: ToggleProps<T>) {
  return (
    <div
      className={cn("lab-toggle", className)}
      role="radiogroup"
      aria-label={label}
      onKeyDown={(event) => {
        if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const current = Math.max(0, options.findIndex(option => option.value === value));
        const next = event.key === "Home" ? 0 : event.key === "End" ? options.length - 1
          : (current + (["ArrowRight", "ArrowDown"].includes(event.key) ? 1 : -1) + options.length) % options.length;
        onChange(options[next].value);
        event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="radio"]')[next]?.focus();
      }}
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          aria-checked={value === opt.value}
          role="radio"
          tabIndex={value === opt.value ? 0 : -1}
          onClick={() => onChange(opt.value)}
        >
          {opt.icon}
          {opt.label}
        </button>
      ))}
    </div>
  );
}
