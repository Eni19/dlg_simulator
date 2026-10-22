import { X, ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useCallback, useRef } from "react";

interface ImageLightboxProps {
  images: string[];
  currentIndex: number;
  labels?: string[];
  onClose: () => void;
  onPrev: () => void;
  onNext: () => void;
}

/**
 * Modal de zoom para imagens — navegação entre múltiplas imagens
 * com teclado (← → ESC) e cliques.
 */
export function ImageLightbox({
  images,
  currentIndex,
  labels,
  onClose,
  onPrev,
  onNext,
}: ImageLightboxProps) {
  const hasMultiple = images.length > 1;
  const dialogRef = useRef<HTMLDivElement>(null);

  // Teclas
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && hasMultiple) onPrev();
      if (e.key === "ArrowRight" && hasMultiple) onNext();
    },
    [onClose, onPrev, onNext, hasMultiple],
  );

  useEffect(() => {
    const previousFocus = document.activeElement as HTMLElement | null;
    dialogRef.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== "Tab") return;
      const buttons =
        dialogRef.current?.querySelectorAll<HTMLButtonElement>("button");
      if (!buttons?.length) return;
      const first = buttons[0],
        last = buttons[buttons.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", trapFocus);
    return () => {
      document.removeEventListener("keydown", trapFocus);
      previousFocus?.focus();
    };
  }, []);

  useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = "";
    };
  }, [handleKeyDown]);

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-label="Visualização ampliada da reconstrução"
      className="image-lightbox fixed inset-0 z-50 flex items-center justify-center bg-black/90 animate-fade-in"
      onClick={onClose}
    >
      {/* Fechar */}
      <button
        onClick={onClose}
        className="absolute top-4 right-4 z-10 flex h-10 w-10 items-center justify-center rounded-full bg-graphite-800/80 text-graphite-300 transition-colors hover:bg-graphite-700 hover:text-white"
        aria-label="Fechar zoom"
      >
        <X className="h-5 w-5" />
      </button>

      {/* Navegação esquerda */}
      {hasMultiple && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onPrev();
          }}
          className="absolute left-4 z-10 flex h-12 w-12 items-center justify-center rounded-full bg-graphite-800/80 text-graphite-300 transition-colors hover:bg-graphite-700 hover:text-white"
          aria-label="Anterior"
        >
          <ChevronLeft className="h-6 w-6" />
        </button>
      )}

      {/* Imagem */}
      <div
        className="flex flex-col items-center max-w-[90vw] max-h-[90vh]"
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={`data:image/png;base64,${images[currentIndex]}`}
          alt={labels?.[currentIndex] ?? `Imagem ${currentIndex + 1}`}
          className="max-w-full max-h-[80vh] rounded-xl object-contain "
        />
        {labels && (
          <p className="mt-3 text-sm font-medium text-graphite-300">
            {labels[currentIndex]}
          </p>
        )}
        {/* Indicador de navegação */}
        {hasMultiple && (
          <div className="mt-3 flex gap-1.5">
            {images.map((_, i) => (
              <span
                key={i}
                className={`inline-block h-1.5 w-1.5 rounded-full transition-colors ${
                  i === currentIndex ? "bg-emerald-400 w-4" : "bg-graphite-600"
                }`}
              />
            ))}
          </div>
        )}
      </div>

      {/* Navegação direita */}
      {hasMultiple && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            onNext();
          }}
          className="absolute right-4 z-10 flex h-12 w-12 items-center justify-center rounded-full bg-graphite-800/80 text-graphite-300 transition-colors hover:bg-graphite-700 hover:text-white"
          aria-label="Próximo"
        >
          <ChevronRight className="h-6 w-6" />
        </button>
      )}
    </div>
  );
}
