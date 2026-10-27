"""
Funções auxiliares para o pipeline DLG.

Inclui:
- Carregamento de dados (CIFAR-100) e transformações.
- Conversão para one-hot encoding.
- Métricas de avaliação de qualidade de reconstrução (PSNR, SSIM, MSE).
- Alinhamento ótimo de reconstruções via Hungarian Algorithm.
- Sanitização de tensores (NaN/Inf).
- Utilitários de dispositivo e seed.
"""

import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision import datasets, transforms
from .runtime_paths import get_data_root


# ---------------------------------------------------------------------------
# Dispositivo e Reprodutibilidade
# ---------------------------------------------------------------------------

def get_device(verbose: bool = True, requested: str | None = None) -> torch.device:
    """
    Detecta o melhor dispositivo disponível para computação.

    DLG_DEVICE=cpu ou cuda força o dispositivo; auto é o padrão.
    CUDA explícito indisponível gera erro em vez de fallback silencioso.

    Ordem de prioridade:
        1. CUDA (NVIDIA GPU) — melhor performance
        2. CPU — default seguro; para modelos pequenos como LeNet,
           CPU é frequentemente mais rápida que aceleradores via
           fallback (DirectML/MPS sofrem com overhead de transferência
           em operações não suportadas nativamente).
        3. DirectML (AMD/Intel GPU) — ative com DLG_USE_DML=1
        4. MPS (Apple Silicon) — ative com DLG_USE_MPS=1

    Args:
        verbose: Se True, imprime o dispositivo selecionado.

    Returns:
        torch.device para o dispositivo selecionado.
    """
    import os

    requested = (requested if requested is not None else os.environ.get("DLG_DEVICE", "auto")).strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("DLG_DEVICE deve ser auto, cpu ou cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "DLG_DEVICE=cuda, mas CUDA está indisponível. "
            "Instale o PyTorch CUDA na venv e verifique o driver NVIDIA."
        )

    # A seleção explícita permite comparar CPU/GPU sem alterar o código.
    if requested == "cpu":
        device = torch.device("cpu")
        label = "CPU (DLG_DEVICE=cpu)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        label = f"CUDA ({torch.cuda.get_device_name(device)})"
    elif os.environ.get("DLG_USE_DML") == "1":
        try:
            import torch_directml  # type: ignore[import-untyped]
            if torch_directml.is_available():
                device = torch_directml.device()
                label = "DirectML (AMD/Intel GPU)"
            else:
                device = torch.device("cpu")
                label = "CPU (fallback — DirectML indisponível)"
        except ImportError:
            device = torch.device("cpu")
            label = "CPU (fallback — torch-directml não instalado)"
    elif os.environ.get("DLG_USE_MPS") == "1":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            label = "MPS (Apple Silicon)"
        else:
            device = torch.device("cpu")
            label = "CPU (fallback — MPS indisponível)"
    else:
        device = torch.device("cpu")
        label = "CPU"

    if verbose:
        print(f"[utils] Running on {label}")
    return device


def get_device_str(verbose: bool = True) -> str:
    """
    Retorna o nome do dispositivo como string para uso em APIs/relatórios.

    Returns:
        'cuda', 'dml', 'mps', ou 'cpu'.
    """
    dev = get_device(verbose=verbose)
    t = dev.type
    if t == "privateuseone":
        return "dml"
    return t


def set_seed(seed: int) -> None:
    """
    Define a semente aleatória para reprodutibilidade.

    Args:
        seed: Valor da semente.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Carregamento de Dados — CIFAR-100
# ---------------------------------------------------------------------------

# Médias e desvios padrão do CIFAR-100 (pré-computados)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar100_data(
    root: str | None = None,
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    download: bool = True,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Cria DataLoaders de treino e teste para CIFAR-100.

    Args:
        root: Diretório raiz para download/armazenamento do dataset.
        batch_size: Tamanho do batch para o DataLoader de treino.
        num_workers: Número de workers para carregamento paralelo.
        pin_memory: Se True, utiliza pin_memory para acelerar transferência GPU.
        download: Se True, faz download do dataset se não encontrado.

    Returns:
        Tupla (train_loader, test_loader).
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_dataset = datasets.CIFAR100(
        get_data_root(root), train=True, transform=train_transform, download=download
    )
    test_dataset = datasets.CIFAR100(
        get_data_root(root), train=False, transform=test_transform, download=download
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def get_raw_cifar100(
    root: str | None = None,
    download: bool = True,
) -> datasets.CIFAR100:
    """
    Retorna o dataset CIFAR-100 cru (sem transformações aplicadas),
    útil para selecionar imagens individuais para o ataque DLG.

    Args:
        root: Diretório raiz para o dataset.
        download: Se True, faz download se necessário.

    Returns:
        Dataset CIFAR-100 (sem transform).
    """
    return datasets.CIFAR100(get_data_root(root), download=download)


# Transformações para imagens individuais
def get_image_transform() -> transforms.Compose:
    """Retorna a transformação padrão para imagens CIFAR-100 (resize + tensor)."""
    return transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])


def get_to_pil_transform() -> transforms.ToPILImage:
    """Retorna transformação ToPILImage para visualização."""
    return transforms.ToPILImage()


# ---------------------------------------------------------------------------
# Codificação de Labels
# ---------------------------------------------------------------------------

def label_to_onehot(target: torch.Tensor, num_classes: int = 100) -> torch.Tensor:
    """
    Converte labels inteiros para one-hot encoding.

    Args:
        target: Tensor de labels [batch] com valores em [0, num_classes-1].
        num_classes: Número total de classes.

    Returns:
        Tensor one-hot [batch, num_classes].
    """
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcula cross-entropy loss para labels one-hot.

    Args:
        pred: Logits previstos [batch, num_classes].
        target: Labels one-hot [batch, num_classes].

    Returns:
        Valor escalar da loss.
    """
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


# ---------------------------------------------------------------------------
# Métricas de Qualidade de Imagem
# ---------------------------------------------------------------------------

def psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio entre dois tensores de imagem em [0, 1].

    Args:
        img1: Primeira imagem [1, C, H, W] ou [C, H, W].
        img2: Segunda imagem, mesma forma.

    Returns:
        PSNR em dB. Maior = melhor. Retorna inf se MSE = 0.
    """
    mse_val = F.mse_loss(img1, img2)
    if mse_val == 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse_val.item()))


def ssim_approx(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 7,
) -> float:
    """
    SSIM aproximado — implementação simplificada sem dependências externas.

    Calcula o Structural Similarity Index entre duas imagens usando
    pooling de média para estimar estatísticas locais.

    Args:
        img1: Primeira imagem [1, C, H, W].
        img2: Segunda imagem, mesma forma.
        window_size: Tamanho da janela para estatísticas locais.

    Returns:
        Valor SSIM médio no intervalo [0, 1]. Maior = mais similar.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sig1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    sig2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    sig12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu12

    ssim_map = ((2 * mu12 + C1) * (2 * sig12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)
    )
    return ssim_map.mean().item()


def mse_metric(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Mean Squared Error entre dois tensores de imagem em [0, 1].

    Args:
        img1: Primeira imagem [1, C, H, W].
        img2: Segunda imagem, mesma forma.

    Returns:
        Valor MSE. Menor = melhor.
    """
    return F.mse_loss(img1, img2).item()


# ---------------------------------------------------------------------------
# Sanitização de Tensores
# ---------------------------------------------------------------------------

def sanitize_image_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Remove NaN/Inf e restringe a imagem ao intervalo [0, 1].

    Args:
        x: Tensor de imagem, potencialmente com NaN/Inf.

    Returns:
        Tensor sanitizado e clampado em [0, 1].
    """
    x_detached = x.detach() if isinstance(x, torch.Tensor) else x
    return torch.nan_to_num(
        x_detached, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0, 1)


def finite_tensor(
    x: torch.Tensor,
    nan: float = 0.0,
    posinf: float = 0.0,
    neginf: float = 0.0,
) -> torch.Tensor:
    """
    Substitui NaN/Inf em um tensor detached por valores finitos.

    Args:
        x: Tensor de entrada.
        nan: Valor para substituir NaN.
        posinf: Valor para substituir +inf.
        neginf: Valor para substituir -inf.

    Returns:
        Tensor sem NaN/Inf.
    """
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


# ---------------------------------------------------------------------------
# Alinhamento Ótimo (Hungarian Algorithm)
# ---------------------------------------------------------------------------

def align_reconstructions(
    reconstructed: torch.Tensor,
    ground_truth: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Alinha reconstruções ao conjunto ground truth usando Hungarian Algorithm.

    O DLG reconstrói o CONJUNTO de imagens sem garantia de ordem.
    Esta função encontra a permutação ótima que minimiza o MSE total.

    Args:
        reconstructed: Tensor [N, C, H, W] das imagens reconstruídas.
        ground_truth: Tensor [N, C, H, W] das imagens originais.

    Returns:
        Tupla (rec_ind, gt_ind, cost_matrix):
        - rec_ind: índices das reconstruções na ordem ótima.
        - gt_ind: índices dos GTs correspondentes.
        - cost_matrix: matriz de custo NxN utilizada.
    """
    reconstructed = sanitize_image_tensor(reconstructed)
    ground_truth = sanitize_image_tensor(ground_truth)

    n = ground_truth.shape[0]
    cost_matrix = torch.zeros(n, n, device=reconstructed.device)

    for i in range(n):
        for j in range(n):
            val = F.mse_loss(
                reconstructed[i].unsqueeze(0), ground_truth[j].unsqueeze(0)
            )
            cost_matrix[i, j] = torch.nan_to_num(
                val, nan=1e6, posinf=1e6, neginf=1e6
            )

    cost_np = cost_matrix.detach().cpu().numpy()
    finite_mask = np.isfinite(cost_np)

    if not finite_mask.all():
        replacement = cost_np[finite_mask].max() * 10 if finite_mask.any() else 1e6
        cost_np = np.nan_to_num(
            cost_np, nan=replacement, posinf=replacement, neginf=replacement
        )

    rec_ind, gt_ind = linear_sum_assignment(cost_np)
    return rec_ind, gt_ind, torch.tensor(cost_np)


# ---------------------------------------------------------------------------
# Avaliação de Reconstruções
# ---------------------------------------------------------------------------

def evaluate_reconstruction_set(
    reconstructed: torch.Tensor,
    ground_truth: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    pred_logits: Optional[torch.Tensor] = None,
    name: str = "Experimento",
) -> pd.DataFrame:
    """
    Calcula MSE, PSNR e SSIM para cada reconstrução após alinhamento ótimo.

    Args:
        reconstructed: Tensor [N, C, H, W] das reconstruções.
        ground_truth: Tensor [N, C, H, W] dos ground truths.
        labels: (Opcional) Labels reais [N].
        pred_logits: (Opcional) Logits previstos [N, num_classes].
        name: Nome do experimento para identificação no DataFrame.

    Returns:
        DataFrame com colunas: experimento, rec_idx, gt_idx, mse, psnr, ssim,
        e opcionalmente label_real, label_pred.
    """
    rec_ind, gt_ind, _ = align_reconstructions(reconstructed, ground_truth)

    rows = []
    for rec_i, gt_i in zip(rec_ind, gt_ind):
        rec = sanitize_image_tensor(reconstructed[rec_i].unsqueeze(0))
        gt = sanitize_image_tensor(ground_truth[gt_i].unsqueeze(0))

        row = {
            "experimento": name,
            "rec_idx": int(rec_i),
            "gt_idx": int(gt_i),
            "mse": mse_metric(rec, gt),
            "psnr": psnr(rec, gt),
            "ssim": ssim_approx(rec, gt),
        }

        if labels is not None:
            row["label_real"] = int(labels[gt_i].item())

        if pred_logits is not None:
            row["label_pred"] = int(torch.argmax(pred_logits[rec_i]).item())

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estatísticas agregadas (média, std, min, max) das métricas
    agrupadas por experimento.

    Args:
        df: DataFrame retornado por evaluate_reconstruction_set.

    Returns:
        DataFrame com agregações.
    """
    return (
        df.groupby("experimento")[["mse", "psnr", "ssim"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )


# ---------------------------------------------------------------------------
# Utilitários de Gradientes
# ---------------------------------------------------------------------------

def clone_gradients(gradients: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Cria cópias detached dos gradientes para uso posterior.

    Args:
        gradients: Lista de tensores de gradiente.

    Returns:
        Lista de cópias detached.
    """
    return [g.detach().clone() for g in gradients]
