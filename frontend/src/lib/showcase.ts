import manifest from "../../../src/data/showcase_samples.json?raw";
import { DEFAULT_CONFIG, type AttackConfig } from "./types";

export const SHOWCASE_BUILD = import.meta.env.VITE_LAB_MODE === "showcase";
export const FULL_LAB_URL = "https://github.com/Eni19/dlg_simulator#executar-localmente";
export const SHOWCASE_ITERATIONS = [10, 25, 50];
export const SHOWCASE_PAIR_ITERATIONS = [...SHOWCASE_ITERATIONS, 100, 200];
export const showcaseIterations = (batchSize: number) => batchSize === 2 ? SHOWCASE_PAIR_ITERATIONS : SHOWCASE_ITERATIONS;
export const SHOWCASE_SAMPLES: { id: number; name: string; image: string }[] = JSON.parse(manifest).samples;

export function showcaseConfig(config: AttackConfig = DEFAULT_CONFIG): AttackConfig {
  const selected = [...new Set(config.imageIndices ?? [])]
    .filter(id => SHOWCASE_SAMPLES.some(sample => sample.id === id)).slice(0, 2);
  const imageIndices = selected.length ? selected : [SHOWCASE_SAMPLES[0].id];
  return {
    ...DEFAULT_CONFIG,
    defenseParams: { ...DEFAULT_CONFIG.defenseParams },
    device: "cpu", batchSize: imageIndices.length, environment: "centralized", defense: "none",
    numIters: showcaseIterations(imageIndices.length).includes(config.numIters) ? config.numIters : 50,
    imageIndices,
  };
}
