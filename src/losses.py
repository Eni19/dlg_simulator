"""
Funções de perda para o ataque DLG.

Inclui:
- Total Variation (TV) loss: regularização espacial que penaliza variações
  bruscas entre pixels vizinhos, reduzindo artefatos de alta frequência.
- grad_matching_loss: Perda composta que combina MSE (L2) com Cosine Similarity
  no espaço de gradientes, conforme a fórmula:
      loss = (1 - COSINE_WEIGHT) * L2_loss + COSINE_WEIGHT * cosine_loss
- safe_grad_matching_loss_defense: Versão numericamente estável para uso
  com defesas (Noisy Gradients, Sparsification).

Constantes:
    TV_COEFF_START, TV_COEFF_END, COSINE_WEIGHT, LABEL_ANCHOR_WEIGHT,
    GLOBAL_PHASE_FREQ, LOGIT_CLAMP, GRAD_EPS
"""

import math
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constantes de Configuração da Perda
# ---------------------------------------------------------------------------

# Total Variation adaptativo
TV_COEFF_START: float = 1e-5  # TV inicial forte (evita caos no começo)
TV_COEFF_END: float = 1e-7    # TV final fraco (preserva detalhes finos)

# Peso da perda cosine similarity no espaço de gradientes
COSINE_WEIGHT: float = 0.5  # loss = (1-COSINE_WEIGHT)*L2 + COSINE_WEIGHT*cosine

# Peso do label anchoring analítico
LABEL_ANCHOR_WEIGHT: float = 0.1

# Fase global: a cada N ciclos round-robin, otimiza todas as imagens juntas
GLOBAL_PHASE_FREQ: int = 10

# Clamping de logits para evitar overflow numérico
LOGIT_CLAMP: float = 20.0

# Epsilon para evitar divisão por zero nos cálculos de gradiente
GRAD_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Total Variation Loss
# ---------------------------------------------------------------------------

def tv_loss(img: torch.Tensor) -> torch.Tensor:
    """
    Total Variation — penaliza variações bruscas entre pixels vizinhos.

    A TV loss atua como regularizador espacial, suavizando artefatos
    de alta frequência enquanto preserva bordas. No contexto do DLG,
    ajuda a evitar que o otimizador gere texturas caóticas.

    Fórmula:
        TV(x) = sum(|x_{i,j+1} - x_{i,j}|) + sum(|x_{i+1,j} - x_{i,j}|)

    Args:
        img: Tensor de imagem [B, C, H, W].

    Returns:
        Valor escalar da penalidade TV.
    """
    return (
        torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
        + torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    )


# ---------------------------------------------------------------------------
# Cosine Annealing do Coeficiente TV
# ---------------------------------------------------------------------------

def tv_cosine_coefficient(progress: float) -> float:
    """
    Calcula o coeficiente TV com cosine annealing.

    Começa em TV_COEFF_START (forte) e decai para TV_COEFF_END (fraco)
    seguindo uma curva de cosine annealing. Isso preserva estrutura
    no início da otimização e permite detalhes finos no final.

    Args:
        progress: Valor em [0, 1] indicando o progresso da otimização.

    Returns:
        Coeficiente TV para o passo atual.
    """
    progress = max(0.0, min(1.0, progress))
    return TV_COEFF_END + 0.5 * (TV_COEFF_START - TV_COEFF_END) * (
        1.0 + math.cos(math.pi * progress)
    )


# ---------------------------------------------------------------------------
# Perda de Casamento de Gradientes (Composta)
# ---------------------------------------------------------------------------

def grad_matching_loss(
    d_dy_dx: list[torch.Tensor],
    original_dy_dx: list[torch.Tensor],
    cosine_weight: float = COSINE_WEIGHT,
) -> torch.Tensor:
    """
    Perda composta: L2 (MSE) + Cosine Similarity no espaço de gradientes.

    Por que Cosine Similarity ajuda contra borramento:
    - MSE puro penaliza igualmente erros de baixa e alta frequência.
      Soluções borradas (baixa frequência) satisfazem MSE quase tão bem
      quanto soluções nítidas, criando planícies no espaço de busca.
    - Cosine similarity é invariante à escala e sensível à DIREÇÃO
      do vetor de gradiente → discrimina estruturas finas que L2 ignora.
    - A combinação quebra plateaus do L2 puro sem perder convexidade local.

    Fórmula:
        loss = (1 - cosine_weight) * L2_loss + cosine_weight * cosine_loss

    Args:
        d_dy_dx: Lista de gradientes reconstruídos (do modelo com dados dummy).
        original_dy_dx: Lista de gradientes originais (capturados).
        cosine_weight: Peso da componente cosine [0, 1].

    Returns:
        Valor escalar da perda composta.
    """
    # L2 loss: soma dos erros quadráticos por camada
    l2_loss = sum(
        ((gx - gy) ** 2).sum() for gx, gy in zip(d_dy_dx, original_dy_dx)
    )

    # Flatten todos os gradientes em vetores 1D para cosine similarity
    gx_flat = torch.cat([g.flatten() for g in d_dy_dx])
    gy_flat = torch.cat([g.flatten() for g in original_dy_dx])

    cosine_loss = 1.0 - F.cosine_similarity(
        gx_flat.unsqueeze(0), gy_flat.unsqueeze(0)
    )

    return (1.0 - cosine_weight) * l2_loss + cosine_weight * cosine_loss


# ---------------------------------------------------------------------------
# Perda de Casamento de Gradientes — Versão Robusta para Defesas
# ---------------------------------------------------------------------------

def safe_grad_matching_loss_defense(
    d_dy_dx: list[torch.Tensor],
    target_gradients: list[torch.Tensor],
    cosine_weight: float = COSINE_WEIGHT,
) -> torch.Tensor:
    """
    Perda L2 + cosine numericamente estável para experimentos com defesas.

    Mantém a mesma ideia da grad_matching_loss, mas:
    - Remove NaN/Inf dos gradientes antes da comparação.
    - Calcula cosine similarity manualmente com epsilon no denominador.
    - Evita NaN quando o gradiente comprimido fica quase todo zero.

    Args:
        d_dy_dx: Lista de gradientes reconstruídos.
        target_gradients: Lista de gradientes alvo (potencialmente ruidosos/comprimidos).
        cosine_weight: Peso da componente cosine [0, 1].

    Returns:
        Valor escalar da perda, sanitizado contra NaN/Inf.
    """
    l2_loss = 0.0
    gx_parts: list[torch.Tensor] = []
    gy_parts: list[torch.Tensor] = []

    for gx, gy in zip(d_dy_dx, target_gradients):
        gx_safe = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
        gy_safe = torch.nan_to_num(gy, nan=0.0, posinf=0.0, neginf=0.0)

        l2_loss = l2_loss + ((gx_safe - gy_safe) ** 2).sum()
        gx_parts.append(gx_safe.reshape(-1))
        gy_parts.append(gy_safe.reshape(-1))

    gx_flat = torch.cat(gx_parts)
    gy_flat = torch.cat(gy_parts)

    dot = torch.sum(gx_flat * gy_flat)
    denom = gx_flat.norm() * gy_flat.norm()

    cosine_sim = dot / denom.clamp_min(GRAD_EPS)
    cosine_sim = cosine_sim.clamp(-1.0, 1.0)
    cosine_loss = 1.0 - cosine_sim

    loss = (1.0 - cosine_weight) * l2_loss + cosine_weight * cosine_loss
    return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=1e6)
