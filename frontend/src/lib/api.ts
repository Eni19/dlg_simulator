// =============================================================================
// API Client — Comunicação com o backend FastAPI
// =============================================================================

import type {
  AttackConfig,
  AttackResponse,
  FederatedResponse,
  HealthResponse,
  ModelInfo,
} from "./types";

const BASE_URL = "/api";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;

  const res = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    ...options,
  });

  if (!res.ok) {
    const body = await res.text();
    let detail = body;
    try {
      const parsed = JSON.parse(body);
      detail = parsed.detail || parsed.message || body;
    } catch {
      /* keep raw text */
    }
    throw new Error(detail || `HTTP ${res.status}: ${res.statusText}`);
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Mapeamento snake_case → camelCase
// ---------------------------------------------------------------------------

/**
 * O backend FastAPI retorna campos em snake_case (batch_size, loss_history...),
 * mas o frontend usa camelCase (batchSize, lossHistory...).
 * Esta função converte a resposta bruta para o formato do frontend.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function mapSnakeToCamel(raw: Record<string, any>): AttackResponse {
  const metricsRaw = raw.metrics as Record<string, unknown> | null | undefined;
  const labelsRaw = raw.reconstructed_labels as number[] | null | undefined;
  const lossRaw = raw.loss_history as number[] | null | undefined;
  const gtImagesRaw = raw.ground_truth_images as string[] | null | undefined;
  const recImagesRaw = raw.reconstructed_images as string[] | null | undefined;
  const errImagesRaw = raw.error_images as string[] | null | undefined;
  const accuracyRaw = raw.accuracy_metrics as Record<string, unknown> | null | undefined;
  const penaltyRaw = raw.defense_penalty as Record<string, unknown> | null | undefined;
  const sparsityRaw = raw.sparsity_stats as Record<string, unknown> | null | undefined;
  const perImageMetricsRaw = raw.per_image_metrics as Record<string, unknown>[] | null | undefined;

  return {
    success: Boolean(raw.success),
    experimentName: String(raw.experiment_name ?? "DLG Attack"),
    batchSize: Number(raw.batch_size ?? 0),
    device: String(raw.device ?? "cpu"),
    lossHistory: Array.isArray(lossRaw) ? lossRaw.map(Number) : [],
    metrics: metricsRaw
      ? {
          ssimMean: Number(metricsRaw.ssim_mean ?? 0),
          psnrMean: Number(metricsRaw.psnr_mean ?? 0),
          mseMean: Number(metricsRaw.mse_mean ?? 0),
          ssimStd: Number(metricsRaw.ssim_std ?? 0),
          psnrStd: Number(metricsRaw.psnr_std ?? 0),
          defenseType: metricsRaw.defense_type
            ? String(metricsRaw.defense_type)
            : undefined,
          informedAttacker: metricsRaw.informed_attacker !== undefined
            ? Boolean(metricsRaw.informed_attacker)
            : undefined,
        }
      : null,
    reconstructedLabels: Array.isArray(labelsRaw) ? labelsRaw.map(Number) : [],
    groundTruthImages: Array.isArray(gtImagesRaw) ? gtImagesRaw.map(String) : [],
    reconstructedImages: Array.isArray(recImagesRaw) ? recImagesRaw.map(String) : [],
    errorImages: Array.isArray(errImagesRaw) ? errImagesRaw.map(String) : [],
    snapshotIterations: Array.isArray(raw.snapshot_iterations)
      ? (raw.snapshot_iterations as number[]).map(Number)
      : [],
    snapshotImages: Array.isArray(raw.snapshot_images)
      ? (raw.snapshot_images as string[][]).map((arr: string[]) =>
          Array.isArray(arr) ? arr.map(String) : []
        )
      : [],
    accuracyMetrics: accuracyRaw
      ? {
          testAccuracy: Number(accuracyRaw.test_accuracy ?? 0),
          testAccuracyTop5: Number(accuracyRaw.test_accuracy_top5 ?? 0),
          testLoss: Number(accuracyRaw.test_loss ?? 0),
          numTestSamples: Number(accuracyRaw.num_test_samples ?? 0),
        }
      : null,
    defensePenalty: penaltyRaw
      ? {
          ssimWithoutDefense: Number(penaltyRaw.ssim_without_defense ?? 0),
          psnrWithoutDefense: Number(penaltyRaw.psnr_without_defense ?? 0),
          mseWithoutDefense: Number(penaltyRaw.mse_without_defense ?? 0),
          ssimWithDefense: Number(penaltyRaw.ssim_with_defense ?? 0),
          psnrWithDefense: Number(penaltyRaw.psnr_with_defense ?? 0),
          mseWithDefense: Number(penaltyRaw.mse_with_defense ?? 0),
          deltaSsim: Number(penaltyRaw.delta_ssim ?? 0),
          deltaPsnr: Number(penaltyRaw.delta_psnr ?? 0),
          deltaMse: Number(penaltyRaw.delta_mse ?? 0),
        }
      : null,
    sparsityStats: sparsityRaw
      ? {
          total: Number(sparsityRaw.total ?? 0),
          nonzero: Number(sparsityRaw.nonzero ?? 0),
          zero: Number(sparsityRaw.zero ?? 0),
          sparsity: Number(sparsityRaw.sparsity ?? 0),
          compressionRatio: Number(sparsityRaw.compression_ratio ?? 0),
        }
      : null,
    message: String(raw.message ?? ""),
    perImageMetrics: Array.isArray(perImageMetricsRaw)
      ? perImageMetricsRaw.map((m: any) => ({
          ssim: Number(m.ssim ?? 0),
          psnr: Number(m.psnr ?? 0),
          mse: Number(m.mse ?? 0),
        }))
      : [],
  };
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

/**
 * Verifica a saúde do serviço backend.
 */
export async function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

/**
 * Obtém informações sobre o modelo LeNet.
 */
export async function getModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/model-info");
}

/**
 * Converte AttackConfig para o payload esperado pelo backend.
 */
function configToPayload(config: AttackConfig): Record<string, unknown> {
  const base: Record<string, unknown> = {
    batch_size: config.batchSize,
    num_iters: config.numIters,
    seed: config.seed,
    tv_coeff_start: config.tvCoeffStart,
    tv_coeff_end: config.tvCoeffEnd,
    cosine_weight: config.cosineWeight,
    label_anchor_weight: config.labelAnchorWeight,
  };

  if (config.imageIndices && config.imageIndices.length === config.batchSize) {
    base["image_indices"] = config.imageIndices;
  }

  if (config.defense !== "none") {
    base["defense_type"] = config.defense;
    base["noise_std"] = config.defenseParams.noiseStd;
    base["keep_ratio"] = config.defenseParams.keepRatio;
    base["pruning_amount"] = config.defenseParams.pruningAmount;
    base["informed_attacker"] = config.informedAttacker;
  }

  return base;
}

/**
 * Executa o ataque DLG standalone (centralizado).
 * Aceita um AbortSignal opcional para cancelamento.
 */
export async function runAttack(
  config: AttackConfig,
  signal?: AbortSignal
): Promise<AttackResponse> {
  const endpoint =
    config.defense === "none" ? "/run-attack" : "/run-defense";
  const raw = await request<Record<string, unknown>>(endpoint, {
    method: "POST",
    body: JSON.stringify(configToPayload(config)),
    signal,
  });
  return mapSnakeToCamel(raw);
}

/**
 * Inicia uma simulação federada com Flower.
 * Aceita um AbortSignal opcional para cancelamento.
 * O backend agora retorna AttackResponse completo com imagens e métricas.
 */
export async function runFederated(
  config: AttackConfig,
  signal?: AbortSignal
): Promise<AttackResponse> {
  const payload: Record<string, unknown> = {
    num_clients: config.flClients,
    num_rounds: config.flRounds,
    defense_type: config.defense,
    noise_std: config.defenseParams.noiseStd,
    keep_ratio: config.defenseParams.keepRatio,
    pruning_amount: config.defenseParams.pruningAmount,
    dlg_iters: config.numIters,
    seed: config.seed,
    batch_size: config.batchSize,
    tv_coeff_start: config.tvCoeffStart,
    tv_coeff_end: config.tvCoeffEnd,
    cosine_weight: config.cosineWeight,
    label_anchor_weight: config.labelAnchorWeight,
  };

  const raw = await request<Record<string, unknown>>("/federated", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });

  // Backend agora retorna AttackResponse diretamente
  return mapSnakeToCamel(raw);
}

// ---------------------------------------------------------------------------
// Streaming de logs (EventSource / SSE)
// ---------------------------------------------------------------------------

export type LogCallback = (line: string) => void;

export function subscribeToLogs(
  _config: AttackConfig,
  onLog: LogCallback,
  onComplete: () => void,
  onError: (err: string) => void
): () => void {
  const url = `${BASE_URL}/stream-attack`;
  const eventSource = new EventSource(url);

  eventSource.addEventListener("log", (event) => {
    onLog(event.data);
  });

  eventSource.addEventListener("complete", () => {
    onComplete();
    eventSource.close();
  });

  eventSource.addEventListener("error", () => {
    onError("Stream connection error");
    eventSource.close();
  });

  return () => eventSource.close();
}
