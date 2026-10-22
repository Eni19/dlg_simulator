import { cn } from "@/lib/utils";
interface CardProps {
  children: React.ReactNode;
  className?: string;
  variant?: "default" | "highlight" | "glass";
  animate?: boolean;
  onClick?: () => void;
}
export function Card({ children, className, onClick }: CardProps) {
  return (
    <div onClick={onClick} className={cn("lab-card", className)}>
      {children}
    </div>
  );
}
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
    <div className="lab-card-header">
      <div>
        <h3>{title}</h3>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {action && <div>{action}</div>}
    </div>
  );
}
