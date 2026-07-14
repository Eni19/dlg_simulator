import clsx, { type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Utilitário para merge de classes Tailwind com suporte a condicionais.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/**
 * Formata um número para exibição compacta com precisão configurável.
 */
export function fmt(n: number, decimals = 4): string {
  if (!isFinite(n)) return "∞";
  return n.toFixed(decimals);
}

/**
 * Formata notação científica para exibição mais limpa.
 */
export function fmtSci(n: number): string {
  if (n === 0) return "0";
  if (Math.abs(n) >= 0.01 || n === 0) return fmt(n, 6);
  return n.toExponential(2);
}

/**
 * Formata PSNR com unidade dB.
 */
export function fmtPSNR(db: number): string {
  if (!isFinite(db)) return "∞ dB";
  return `${db.toFixed(1)} dB`;
}

/**
 * Formata SSIM com 4 casas.
 */
export function fmtSSIM(s: number): string {
  return s.toFixed(4);
}

/**
 * Classe de cor para métrica PSNR.
 */
export function psnrColor(db: number): string {
  if (!isFinite(db)) return "text-emerald-400";
  if (db >= 30) return "text-emerald-400";
  if (db >= 20) return "text-amber-400";
  return "text-red-400";
}

/**
 * Classe de cor para métrica SSIM.
 */
export function ssimColor(s: number): string {
  if (s >= 0.9) return "text-emerald-400";
  if (s >= 0.7) return "text-amber-400";
  return "text-red-400";
}

/**
 * Gera um ID único para uso em listas.
 */
let idCounter = 0;
export function uid(): string {
  return `dlg-${++idCounter}-${Date.now()}`;
}
