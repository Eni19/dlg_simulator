// =============================================================================
// Tipos do Projeto DLG Attack
// =============================================================================

// ── Domínio do Experimento ──────────────────────────────────────────────────

export type Environment = "centralized" | "federated";

export type DefenseType = "none" | "noisy" | "sparse" | "pruning";

export type AttackerRole = "malicious_server" | "malicious_client" | "mitm";

export interface AttackConfig {
  environment: Environment;
  batchSize: number; // 1–8
  numIters: number;
  seed: number;
  defense: DefenseType;
  defenseParams: DefenseParams;
  // Hiperparâmetros da perda
  tvCoeffStart: number; // default 1e-5
  tvCoeffEnd: number; // default 1e-7
  cosineWeight: number; // default 0.5
  labelAnchorWeight: number; // default 0.1
  // Federado
  flClients: number; // ≥2
  flRounds: number; // ≥1
  // Imagens específicas (null = aleatório)
  imageIndices: number[] | null;
  // Atacante informado (para pruning)
  informedAttacker: boolean;
  // Papel do atacante no ambiente federado
  attackerRole: AttackerRole;
  // Particionamento (flwr-datasets)
  partitionerType: "iid" | "dirichlet";
  dirichletAlpha: number;
}

export interface DefenseParams {
  noiseStd: number; // para noisy
  keepRatio: number; // para sparse
  pruningAmount: number; // para pruning (0–0.9)
}

export const DEFAULT_CONFIG: AttackConfig = {
  environment: "centralized",
  batchSize: 3,
  numIters: 800,
  seed: 50,
  defense: "none",
  defenseParams: {
    noiseStd: 0.1,
    keepRatio: 0.2,
    pruningAmount: 0.5,
  },
  tvCoeffStart: 1e-5,
  tvCoeffEnd: 1e-7,
  cosineWeight: 0.5,
  labelAnchorWeight: 0.1,
  flClients: 2,
  flRounds: 5,
  imageIndices: null,
  informedAttacker: true,
  attackerRole: "malicious_server",
  partitionerType: "iid",
  dirichletAlpha: 0.5,
};

// ── Respostas da API ────────────────────────────────────────────────────────

export interface LogEntry {
  iteration: number;
  totalIterations: number;
  loss: number;
  tvCoeff: number;
  timestamp: number;
}

export interface ImageMetrics {
  recIdx: number;
  gtIdx: number;
  ssim: number;
  psnr: number;
  mse: number;
  labelReal: number;
  labelPred: number;
}

export interface DefensePenalty {
  ssimWithoutDefense: number;
  psnrWithoutDefense: number;
  mseWithoutDefense: number;
  ssimWithDefense: number;
  psnrWithDefense: number;
  mseWithDefense: number;
  deltaSsim: number;
  deltaPsnr: number;
  deltaMse: number;
}

export interface AccuracyMetrics {
  testAccuracy: number;
  testAccuracyTop5: number;
  testLoss: number;
  numTestSamples: number;
}

export interface SparsityStats {
  total: number;
  nonzero: number;
  zero: number;
  sparsity: number;
  compressionRatio: number;
}

export interface AttackResponse {
  success: boolean;
  experimentName: string;
  batchSize: number;
  device: string;
  lossHistory: number[];
  metrics: {
    ssimMean: number;
    psnrMean: number;
    mseMean: number;
    ssimStd: number;
    psnrStd: number;
    defenseType?: string;
    informedAttacker?: boolean;
  } | null;
  reconstructedLabels: number[] | null;
  groundTruthImages: string[];
  reconstructedImages: string[];
  errorImages: string[];
  snapshotIterations: number[];
  snapshotImages: string[][];
  accuracyMetrics: AccuracyMetrics | null;
  defensePenalty: DefensePenalty | null;
  sparsityStats: SparsityStats | null;
  message: string;
  perImageMetrics: {
    ssim: number;
    psnr: number;
    mse: number;
  }[];
}

export interface FederatedResponse {
  success: boolean;
  numClients: number;
  numRounds: number;
  defenseType: string;
  dlgResultsCount: number;
  message: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  flowerAvailable: boolean;
  cudaAvailable: boolean;
}

export interface ModelInfo {
  architecture: string;
  totalParameters: number;
  trainableParameters: number;
  numClasses: number;
  inputShape: number[];
  layers: { name: string; type: string }[];
}

// ── Estado da UI ─────────────────────────────────────────────────────────────

export interface ExperimentState {
  config: AttackConfig;
  status: "idle" | "running" | "completed" | "error";
  response: AttackResponse | null;
  errorMessage: string | null;
}

// ── Dados para gráficos ─────────────────────────────────────────────────────

export interface ChartDataPoint {
  iteration: number;
  loss: number;
}

export interface DefenseChartDataPoint {
  name: string;
  ssim: number;
  psnr: number;
  mse: number;
}
