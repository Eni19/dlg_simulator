import { Component, type ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Error Boundary para evitar que um crash em um painel
 * faça a tela inteira ficar preta.
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="rounded-xl border border-red-500/20 bg-red-600/10 p-6 text-center">
          <p className="text-sm font-semibold text-red-400">
            Erro ao renderizar este painel
          </p>
          <p className="mt-1 text-xs text-graphite-400">
            {this.state.error?.message ?? "Erro desconhecido"}
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
