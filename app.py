"""
Aplicação FastAPI para o serviço de Ataque DLG.

Expõe as funcionalidades de simulação de ataque e defesa como rotas REST,
preparadas para deploy serverless no Vercel (via api/index.py).

Rotas:
    GET  /api/health         — Verificação de saúde do serviço.
    GET  /api/model-info     — Informações sobre o modelo LeNet.
    POST /api/run-attack     — Executa ataque DLG standalone.
    POST /api/run-defense    — Executa ataque DLG com defesas.
    POST /api/federated      — Inicia simulação federada com Flower.
    POST /api/intercept      — Intercepta gradientes e executa DLG.

Autor: Enrique Susin
"""

from __future__ import annotations

import base64
import io
import json
import logging
import traceback
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field, validator
from torchvision import transforms as tv_transforms

from src.models import create_lenet
from src.utils import (
    get_cifar100_data,
    get_device,
    get_device_str,
    get_image_transform,
    get_raw_cifar100,
    label_to_onehot,
    set_seed,
    evaluate_reconstruction_set,
    summarize_metrics,
)
from src.dlg_attack import DLGAttack
from src.federated_flower import (
    FlowerClient,
    AttackerClient,
    create_clients,
    intercept_and_attack,
    start_federated_simulation,
    FLOWER_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dlg-api")

app = FastAPI(
    title="DLG Attack API",
    description=(
        "API para simulação de ataques Deep Leakage from Gradients (DLG) "
        "em ambientes de Aprendizado Federado."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS — permite chamadas do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dispositivo
DEVICE = get_device(verbose=True)
DEVICE_STR = get_device_str(verbose=False)

# Dataset e modelo globais (carregados sob demanda)
_dataset: Optional[object] = None
_model: Optional[torch.nn.Module] = None


def get_dataset():
    """Retorna o dataset CIFAR-100 (lazy load)."""
    global _dataset
    if _dataset is None:
        _dataset = get_raw_cifar100(download=True)
    return _dataset


def get_model() -> torch.nn.Module:
    """Retorna o modelo LeNet base (lazy load)."""
    global _model
    if _model is None:
        _model = create_lenet(num_classes=100, device=DEVICE, seed=50)
    return _model


# ---------------------------------------------------------------------------
# Utilitários de Imagem e Avaliação
# ---------------------------------------------------------------------------

def tensor_to_base64(t: torch.Tensor) -> str:
    """
    Converte um tensor de imagem [C, H, W] em [0, 1] para string base64 PNG.

    Args:
        t: Tensor [3, 32, 32] com valores em [0, 1].

    Returns:
        String base64 da imagem PNG.
    """
    t_safe = t.detach().cpu().clamp(0, 1)
    pil_img = tv_transforms.ToPILImage()(t_safe)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_model_accuracy(
    model: torch.nn.Module,
    device: torch.device,
) -> dict:
    """
    Calcula a acurácia do modelo no conjunto de teste CIFAR-100.

    Args:
        model: Modelo LeNet a ser avaliado.
        device: Dispositivo de computação.

    Returns:
        Dicionário com:
        - test_accuracy: acurácia top-1 no teste.
        - test_loss: loss média no teste.
        - num_test_samples: número de amostras de teste.
    """
    from torch.utils.data import DataLoader
    from torchvision import datasets

    test_transform = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ])

    test_dataset = datasets.CIFAR100(
        "~/.torch", train=False, transform=test_transform, download=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=0
    )

    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            output = model(data)
            loss = F.cross_entropy(output, labels)
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct_top1 += pred.eq(labels).sum().item()
            
            _, top5_pred = output.topk(5, 1, True, True)
            correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            total += data.size(0)

    return {
        "test_accuracy": round(correct_top1 / max(total, 1) * 100, 2),
        "test_accuracy_top5": round(correct_top5 / max(total, 1) * 100, 2),
        "test_loss": round(total_loss / max(total, 1), 4),
        "num_test_samples": total,
    }


# ---------------------------------------------------------------------------
# Modelos de Requisição/Resposta
# ---------------------------------------------------------------------------

class AttackRequest(BaseModel):
    """Requisição para executar ataque DLG."""
    batch_size: int = Field(default=3, ge=1, le=8, description="Número de imagens no batch")
    num_iters: int = Field(default=800, ge=10, le=5000, description="Iterações do ataque")
    seed: int = Field(default=50, description="Semente para reprodutibilidade")
    tv_coeff_start: float = Field(default=1e-5, ge=1e-8, le=1e-2, description="TV inicial")
    tv_coeff_end: float = Field(default=1e-7, ge=1e-10, le=1e-3, description="TV final")
    cosine_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Peso cosine na loss")
    label_anchor_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Peso label anchoring")
    image_indices: Optional[list[int]] = Field(default=None, description="Índices de imagens específicas (None = aleatório)")

    @validator("image_indices")
    def validate_indices(cls, v, values):
        if v is not None and "batch_size" in values:
            if len(v) != values["batch_size"]:
                raise ValueError(f"image_indices deve ter tamanho igual a batch_size ({values['batch_size']})")
        return v


class DefenseRequest(AttackRequest):
    """Requisição para executar ataque DLG com defesas."""
    defense_type: str = Field(default="pruning", description="Tipo de defesa: 'none', 'noisy', 'sparse', 'pruning'")
    noise_std: float = Field(default=0.1, ge=0.0, le=1.0, description="Std do ruído (para noisy)")
    keep_ratio: float = Field(default=0.2, ge=0.01, le=1.0, description="Ratio de gradientes mantidos (para sparse)")
    pruning_amount: float = Field(default=0.5, ge=0.0, lt=1.0, description="Proporção de pesos podados (para pruning)")
    informed_attacker: bool = Field(default=True, description="Se True, atacante usa o mesmo modelo podado (informado). Se False, atacante usa modelo original (desinformado).")


class FederatedRequest(BaseModel):
    """Requisição para simulação federada."""
    num_clients: int = Field(default=2, ge=2, le=10, description="Número de clientes")
    num_rounds: int = Field(default=5, ge=1, le=50, description="Rodadas de agregação")
    defense_type: str = Field(default="none", description="Defesa: 'none', 'noisy', 'sparse', 'pruning'")
    noise_std: float = Field(default=0.1, ge=0.0, le=1.0)
    keep_ratio: float = Field(default=0.2, ge=0.01, le=1.0)
    pruning_amount: float = Field(default=0.5, ge=0.0, lt=1.0)
    dlg_iters: int = Field(default=800, ge=10, le=5000, description="Iterações do ataque DLG")
    batch_size: int = Field(default=1, ge=1, le=8, description="Tamanho do batch para o ataque DLG")
    seed: int = Field(default=50)
    attacker_role: str = Field(default="malicious_server", description="Papel do atacante: malicious_server, malicious_client, mitm")


class AttackResponse(BaseModel):
    """Resposta do ataque DLG."""
    success: bool
    experiment_name: str
    batch_size: int
    device: str
    loss_history: list[float]
    metrics: Optional[dict] = None
    reconstructed_labels: Optional[list[int]] = None
    ground_truth_images: list[str] = []
    reconstructed_images: list[str] = []
    error_images: list[str] = []
    snapshot_iterations: list[int] = []
    snapshot_images: list[list[str]] = []
    accuracy_metrics: Optional[dict] = None
    defense_penalty: Optional[dict] = None
    sparsity_stats: Optional[dict] = None
    message: str = ""
    per_image_metrics: list[dict] = []


class FederatedResponse(BaseModel):
    """Resposta da simulação federada."""
    success: bool
    num_clients: int
    num_rounds: int
    defense_type: str
    dlg_results_count: int
    message: str


class HealthResponse(BaseModel):
    """Resposta do health check."""
    status: str
    version: str
    device: str
    flower_available: bool
    cuda_available: bool


class DistributionRequest(BaseModel):
    num_clients: int = Field(default=10, ge=2, le=50)
    partitioner_type: str = Field(default="iid", description="iid ou dirichlet")
    alpha: float = Field(default=0.5, description="Alpha para dirichlet")


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "version": "1.0",
        "device": DEVICE_STR,
        "flower_available": FLOWER_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/api/data-distribution")
def data_distribution(request: DistributionRequest):
    """
    Usa flwr_datasets para gerar a distribuição de partições e
    retorna a contagem de classes por cliente para o frontend.
    """
    try:
        from flwr_datasets import FederatedDataset
        from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
        import numpy as np

        if request.partitioner_type == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=request.num_clients,
                partition_by="fine_label",
                alpha=request.alpha,
                min_partition_size=10,
                self_balancing=True,
            )
        else:
            partitioner = IidPartitioner(num_partitions=request.num_clients)

        fds = FederatedDataset(dataset="cifar100", partitioners={"train": partitioner})

        distribution = {}
        for client_id in range(request.num_clients):
            partition = fds.load_partition(client_id, "train")
            labels = partition["fine_label"]
            unique, counts = np.unique(labels, return_counts=True)
            distribution[f"Cliente {client_id}"] = {str(k): int(v) for k, v in zip(unique, counts)}

        return {
            "success": True,
            "distribution": distribution
        }
    except Exception as e:
        logger.error(f"Erro em data-distribution: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-info")
async def model_info():
    """Retorna informações sobre a arquitetura LeNet."""
    model = get_model()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    layers = []
    for name, module in model.named_modules():
        if name:  # pular o root
            layers.append({
                "name": name,
                "type": module.__class__.__name__,
            })

    return JSONResponse({
        "architecture": "LeNet (Sigmoid)",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "num_classes": 100,
        "input_shape": [3, 32, 32],
        "layers": layers,
    })


@app.post("/api/run-attack", response_model=AttackResponse)
async def run_attack(request: AttackRequest):
    """
    Executa o ataque DLG standalone.
    """
    try:
        set_seed(request.seed)
        dataset = get_dataset()
        model = get_model()
        model.eval()
        tp = get_image_transform()

        if request.image_indices is not None:
            indices = request.image_indices
        else:
            import time
            fresh_seed = int(time.time() * 1000) % (2**31)
            rng = np.random.RandomState(fresh_seed)
            indices = rng.choice(len(dataset), request.batch_size, replace=False).tolist()

        gt_images = []
        gt_labels = []
        for idx in indices:
            gt_images.append(tp(dataset[idx][0]))
            gt_labels.append(dataset[idx][1])

        gt_data = torch.stack(gt_images).to(DEVICE)
        gt_label = torch.tensor(gt_labels, dtype=torch.long, device=DEVICE)
        gt_onehot = label_to_onehot(gt_label, num_classes=100)

        from src.utils import cross_entropy_for_onehot
        model.zero_grad(set_to_none=True)
        out = model(gt_data)
        loss = cross_entropy_for_onehot(out, gt_onehot)
        dy_dx = torch.autograd.grad(loss, model.parameters())
        original_dy_dx = [g.detach().clone() for g in dy_dx]

        attack_model = create_lenet(num_classes=100, device=DEVICE, seed=request.seed)
        attack_model.eval()

        attack = DLGAttack(
            batch_size=request.batch_size,
            num_iters=request.num_iters,
            num_classes=100,
            device=DEVICE,
            cosine_weight=request.cosine_weight,
            tv_coeff_start=request.tv_coeff_start,
            tv_coeff_end=request.tv_coeff_end,
            label_anchor_weight=request.label_anchor_weight,
            seed=request.seed,
            verbose=True,
        )

        result = attack.run(
            target_gradients=original_dy_dx,
            attack_model=attack_model,
            experiment_name="API DLG Attack",
        )

        from src.utils import align_reconstructions
        rec_ind, gt_ind, _ = align_reconstructions(
            result["reconstructed"], gt_data
        )
        aligned_rec = result["reconstructed"][rec_ind]
        aligned_label = result["dummy_label"][rec_ind]

        metrics_df = evaluate_reconstruction_set(
            reconstructed=result["reconstructed"],
            ground_truth=gt_data,
            labels=gt_label,
            pred_logits=result["dummy_label"],
            name="DLG Attack",
        )

        per_image_metrics = []
        for i in range(request.batch_size):
            row = metrics_df[metrics_df['rec_idx'] == rec_ind[i]].iloc[0]
            per_image_metrics.append({
                "ssim": float(row["ssim"]),
                "mse": float(row["mse"]),
                "psnr": float(row["psnr"]),
            })

        metrics_dict = {
            "ssim_mean": float(metrics_df["ssim"].mean()),
            "psnr_mean": float(metrics_df["psnr"].mean()),
            "mse_mean": float(metrics_df["mse"].mean()),
            "ssim_std": float(metrics_df["ssim"].std()),
            "psnr_std": float(metrics_df["psnr"].std()),
        }

        predicted_labels = [
            int(torch.argmax(aligned_label[i]).item())
            for i in range(request.batch_size)
        ]

        gt_b64 = [tensor_to_base64(gt_data[j]) for j in gt_ind]
        rec_b64 = [tensor_to_base64(aligned_rec[i]) for i in range(request.batch_size)]
        err_b64 = [
            tensor_to_base64(torch.abs(gt_data[gt_ind[i]] - aligned_rec[i]))
            for i in range(request.batch_size)
        ]

        accuracy = compute_model_accuracy(model, DEVICE)
        from src.federated_flower import parameter_sparsity_stats
        sp_stats = parameter_sparsity_stats(model, only_prunable_weights=True)

        snapshot_images: list[list[str]] = []
        snapshot_iters: list[int] = []
        if result.get("snapshots"):
            for snap_idx, snap_tensor in enumerate(result["snapshots"]):
                snap_aligned = snap_tensor[rec_ind]
                snap_b64 = [
                    tensor_to_base64(snap_aligned[i])
                    for i in range(request.batch_size)
                ]
                snapshot_images.append(snap_b64)
                snapshot_iters.append(result["snapshot_iters"][snap_idx])

        return AttackResponse(
            success=True,
            experiment_name="DLG Attack",
            batch_size=request.batch_size,
            device=DEVICE_STR,
            loss_history=[float(l) for l in result["loss_history"]],
            metrics=metrics_dict,
            reconstructed_labels=predicted_labels,
            ground_truth_images=gt_b64,
            reconstructed_images=rec_b64,
            error_images=err_b64,
            snapshot_iterations=snapshot_iters,
            snapshot_images=snapshot_images,
            accuracy_metrics=accuracy,
            sparsity_stats=sp_stats,
            message="Ataque DLG centralizado concluído com sucesso.",
            per_image_metrics=per_image_metrics,
        )

    except Exception as e:
        logger.exception("Erro no ataque DLG")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-defense", response_model=AttackResponse)
async def run_defense(request: DefenseRequest):
    """
    Executa o ataque DLG com defesas aplicadas aos gradientes.
    """
    try:
        set_seed(request.seed)
        dataset = get_dataset()
        model = get_model()
        model.eval()
        tp = get_image_transform()

        if request.image_indices is not None:
            indices = request.image_indices
        else:
            import time
            fresh_seed = int(time.time() * 1000) % (2**31)
            rng = np.random.RandomState(fresh_seed)
            indices = rng.choice(len(dataset), request.batch_size, replace=False).tolist()

        gt_images, gt_labels = [], []
        for idx in indices:
            gt_images.append(tp(dataset[idx][0]))
            gt_labels.append(dataset[idx][1])

        gt_data = torch.stack(gt_images).to(DEVICE)
        gt_label = torch.tensor(gt_labels, dtype=torch.long, device=DEVICE)

        defense_params = {}
        if request.defense_type == "noisy":
            defense_params["noise_std"] = request.noise_std
        elif request.defense_type == "sparse":
            defense_params["keep_ratio"] = request.keep_ratio
        elif request.defense_type == "pruning":
            defense_params["amount"] = request.pruning_amount

        result = intercept_and_attack(
            model=model,
            data=gt_data,
            labels=gt_label,
            device=DEVICE,
            defense_type=request.defense_type,
            defense_params=defense_params,
            dlg_iters=request.num_iters,
            dlg_seed=request.seed,
            verbose=True,
            informed_attacker=request.informed_attacker,
        )

        dlg = result["dlg_result"]

        from src.utils import align_reconstructions
        rec_ind, gt_ind, _ = align_reconstructions(
            dlg["reconstructed"], gt_data
        )
        aligned_rec = dlg["reconstructed"][rec_ind]
        aligned_label = dlg["dummy_label"][rec_ind]

        metrics_df = evaluate_reconstruction_set(
            reconstructed=dlg["reconstructed"],
            ground_truth=gt_data,
            labels=gt_label,
            pred_logits=dlg["dummy_label"],
            name=f"DLG + {request.defense_type}",
        )

        per_image_metrics = []
        for i in range(result["batch_size"]):
            row = metrics_df[metrics_df['rec_idx'] == rec_ind[i]].iloc[0]
            per_image_metrics.append({
                "ssim": float(row["ssim"]),
                "mse": float(row["mse"]),
                "psnr": float(row["psnr"]),
            })

        metrics_dict = {
            "ssim_mean": float(metrics_df["ssim"].mean()),
            "psnr_mean": float(metrics_df["psnr"].mean()),
            "mse_mean": float(metrics_df["mse"].mean()),
            "ssim_std": float(metrics_df["ssim"].std()),
            "psnr_std": float(metrics_df["psnr"].std()),
            "defense_type": request.defense_type,
            "informed_attacker": request.informed_attacker,
        }

        predicted_labels = [
            int(torch.argmax(aligned_label[i]).item())
            for i in range(result["batch_size"])
        ]

        gt_b64 = [tensor_to_base64(gt_data[j]) for j in gt_ind]
        rec_b64 = [tensor_to_base64(aligned_rec[i]) for i in range(result["batch_size"])]
        err_b64 = [
            tensor_to_base64(torch.abs(gt_data[gt_ind[i]] - aligned_rec[i]))
            for i in range(result["batch_size"])
        ]

        accuracy = compute_model_accuracy(model, DEVICE)
        from src.federated_flower import parameter_sparsity_stats
        sp_stats = parameter_sparsity_stats(model, only_prunable_weights=True)

        snapshot_images: list[list[str]] = []
        snapshot_iters: list[int] = []
        if dlg.get("snapshots"):
            for snap_idx, snap_tensor in enumerate(dlg["snapshots"]):
                snap_aligned = snap_tensor[rec_ind]
                snap_b64 = [
                    tensor_to_base64(snap_aligned[i])
                    for i in range(result["batch_size"])
                ]
                snapshot_images.append(snap_b64)
                snapshot_iters.append(dlg["snapshot_iters"][snap_idx])

        defense_penalty = None
        if request.defense_type != "none":
            no_defense_result = intercept_and_attack(
                model=model,
                data=gt_data,
                labels=gt_label,
                device=DEVICE,
                defense_type="none",
                defense_params={},
                dlg_iters=request.num_iters,
                dlg_seed=request.seed,
                verbose=False,
                informed_attacker=True,
            )
            nd_dlg = no_defense_result["dlg_result"]
            nd_metrics_df = evaluate_reconstruction_set(
                reconstructed=nd_dlg["reconstructed"],
                ground_truth=gt_data,
                labels=gt_label,
                pred_logits=nd_dlg["dummy_label"],
                name="DLG sem defesa (referência)",
            )
            defense_penalty = {
                "ssim_without_defense": round(float(nd_metrics_df["ssim"].mean()), 4),
                "psnr_without_defense": round(float(nd_metrics_df["psnr"].mean()), 2),
                "mse_without_defense": round(float(nd_metrics_df["mse"].mean()), 6),
                "ssim_with_defense": metrics_dict["ssim_mean"],
                "psnr_with_defense": metrics_dict["psnr_mean"],
                "mse_with_defense": metrics_dict["mse_mean"],
                "delta_ssim": round(metrics_dict["ssim_mean"] - float(nd_metrics_df["ssim"].mean()), 4),
                "delta_psnr": round(metrics_dict["psnr_mean"] - float(nd_metrics_df["psnr"].mean()), 2),
                "delta_mse": round(metrics_dict["mse_mean"] - float(nd_metrics_df["mse"].mean()), 6),
            }

        attacker_info = "informado" if request.informed_attacker else "desinformado"
        return AttackResponse(
            success=True,
            experiment_name=f"DLG + {request.defense_type} ({attacker_info})",
            batch_size=result["batch_size"],
            device=DEVICE_STR,
            loss_history=[float(l) for l in dlg.get("loss_history", [])],
            metrics=metrics_dict,
            reconstructed_labels=predicted_labels,
            ground_truth_images=gt_b64,
            reconstructed_images=rec_b64,
            error_images=err_b64,
            snapshot_iterations=snapshot_iters,
            snapshot_images=snapshot_images,
            accuracy_metrics=accuracy,
            defense_penalty=defense_penalty,
            sparsity_stats=sp_stats,
            message=(
                f"Ataque com defesa '{request.defense_type}' concluído. "
                f"SSIM médio: {metrics_dict['ssim_mean']:.4f}"
            ),
            per_image_metrics=per_image_metrics,
        )

    except Exception as e:
        logger.exception("Erro no ataque com defesa")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/federated")
async def run_federated(request: FederatedRequest):
    """
    Inicia uma simulação de Aprendizado Federado com Flower.
    """
    if not FLOWER_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Flower (flwr) não está instalado. Instale com: pip install flwr"
        )

    try:
        defense_params = {}
        if request.defense_type == "noisy":
            defense_params["noise_std"] = request.noise_std
        elif request.defense_type == "sparse":
            defense_params["keep_ratio"] = request.keep_ratio
        elif request.defense_type == "pruning":
            defense_params["amount"] = request.pruning_amount

        batch_size = request.batch_size
        results = start_federated_simulation(
            num_clients=request.num_clients,
            num_rounds=request.num_rounds,
            defense_type=request.defense_type,
            defense_params=defense_params,
            pruning_amount=request.pruning_amount,
            device=DEVICE,
            seed=request.seed,
            attacker_enabled=True,
            attacker_role=request.attacker_role,
            verbose=True,
            dlg_batch_size=batch_size,
            dlg_num_iters=request.dlg_iters,
        )

        dlg_results = results.get("dlg_results", [])
        clients = results.get("clients", [])
        history = results.get("history", [])

        attacker_client = None
        for c in clients:
            if isinstance(c, AttackerClient) and c.dlg_result is not None:
                attacker_client = c
                break

        if attacker_client is None or len(dlg_results) == 0:
            return AttackResponse(
                success=True,
                experiment_name=f"Federated {request.num_clients}c/{request.num_rounds}r",
                batch_size=batch_size,
                device=DEVICE_STR,
                loss_history=[],
                metrics=None,
                reconstructed_labels=[],
                ground_truth_images=[],
                reconstructed_images=[],
                error_images=[],
                snapshot_iterations=[],
                snapshot_images=[],
                accuracy_metrics=None,
                defense_penalty=None,
                message=(
                    f"Simulação concluída: {request.num_clients} clientes, "
                    f"{request.num_rounds} rodadas. Nenhum ataque DLG executado."
                ),
            )

        dlg = dlg_results[0] if isinstance(dlg_results[0], dict) else dlg_results[0]
        gt_data = attacker_client.intercepted_data
        gt_labels = attacker_client.intercepted_labels

        if gt_data is None or gt_labels is None:
            return AttackResponse(
                success=True,
                experiment_name=f"Federated {request.num_clients}c/{request.num_rounds}r",
                batch_size=batch_size,
                device=DEVICE_STR,
                loss_history=[float(l) for l in dlg.get("loss_history", [])],
                metrics=None,
                reconstructed_labels=[],
                ground_truth_images=[],
                reconstructed_images=[],
                error_images=[],
                snapshot_iterations=[],
                snapshot_images=[],
                accuracy_metrics=None,
                defense_penalty=None,
                sparsity_stats=None,
                message="Simulação concluída, mas sem dados interceptados disponíveis.",
            )

        actual_bs = min(batch_size, gt_data.shape[0])
        gt_data = gt_data[:actual_bs]
        gt_labels = gt_labels[:actual_bs]
        dlg_reconstructed = dlg["reconstructed"][:actual_bs]
        dlg_label = dlg["dummy_label"][:actual_bs]

        from src.utils import align_reconstructions
        rec_ind, gt_ind, _ = align_reconstructions(dlg_reconstructed, gt_data)
        aligned_rec = dlg_reconstructed[rec_ind]
        aligned_label = dlg_label[rec_ind]

        metrics_df = evaluate_reconstruction_set(
            reconstructed=dlg_reconstructed,
            ground_truth=gt_data,
            labels=gt_labels,
            pred_logits=dlg_label,
            name=f"Federated {request.num_clients}c/{request.num_rounds}r",
        )

        metrics_dict = {
            "ssim_mean": float(metrics_df["ssim"].mean()),
            "psnr_mean": float(metrics_df["psnr"].mean()),
            "mse_mean": float(metrics_df["mse"].mean()),
            "ssim_std": float(metrics_df["ssim"].std()),
            "psnr_std": float(metrics_df["psnr"].std()),
            "defense_type": request.defense_type,
            "attacker_role": request.attacker_role,
        }

        per_image_metrics = []
        for i in range(actual_bs):
            row = metrics_df[metrics_df['rec_idx'] == rec_ind[i]].iloc[0]
            per_image_metrics.append({
                "ssim": float(row["ssim"]),
                "mse": float(row["mse"]),
                "psnr": float(row["psnr"]),
            })

        predicted_labels = [
            int(torch.argmax(aligned_label[i]).item())
            for i in range(actual_bs)
        ]

        role_msgs = {
            "malicious_server": "Servidor Curioso interceptou os gradientes de atualização do Cliente durante o FedAvg.",
            "malicious_client": "Cliente Malicioso inferiu os gradientes da vítima subtraindo sua própria contribuição do modelo global.",
            "mitm": "Ataque Man-in-the-Middle interceptou os pacotes de rede do Cliente a caminho do Servidor."
        }
        role_msg = role_msgs.get(request.attacker_role, "")
        message = f"{role_msg} SSIM médio: {metrics_dict['ssim_mean']:.4f}"

        gt_b64 = [tensor_to_base64(gt_data[j]) for j in gt_ind]
        rec_b64 = [tensor_to_base64(aligned_rec[i]) for i in range(actual_bs)]
        err_b64 = [
            tensor_to_base64(torch.abs(gt_data[gt_ind[i]] - aligned_rec[i]))
            for i in range(actual_bs)
        ]

        model = get_model()
        accuracy = compute_model_accuracy(model, DEVICE)

        snapshot_images: list[list[str]] = []
        snapshot_iters: list[int] = []
        if dlg.get("snapshots"):
            for snap_idx, snap_tensor in enumerate(dlg["snapshots"]):
                snap_actual = snap_tensor[:actual_bs]
                snap_aligned = snap_actual[rec_ind]
                snap_b64 = [
                    tensor_to_base64(snap_aligned[i])
                    for i in range(actual_bs)
                ]
                snapshot_images.append(snap_b64)
                snapshot_iters.append(dlg["snapshot_iters"][snap_idx])

        return AttackResponse(
            success=True,
            experiment_name=f"Federated {request.num_clients}c/{request.num_rounds}r",
            batch_size=actual_bs,
            device=DEVICE_STR,
            loss_history=[float(l) for l in dlg.get("loss_history", [])],
            metrics=metrics_dict,
            reconstructed_labels=predicted_labels,
            ground_truth_images=gt_b64,
            reconstructed_images=rec_b64,
            error_images=err_b64,
            snapshot_iterations=snapshot_iters,
            snapshot_images=snapshot_images,
            accuracy_metrics=accuracy,
            defense_penalty=None,
            sparsity_stats=None,
            message=message,
            per_image_metrics=per_image_metrics,
        )

    except Exception as e:
        logger.exception("Erro na simulação federada")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intercept")
async def intercept_gradients(request: DefenseRequest):
    """
    Rota simplificada que simula apenas a interceptação de gradientes:
    calcula gradientes → aplica defesa → executa DLG → retorna métricas.
    """
    return await run_defense(request)


# ---------------------------------------------------------------------------
# Ponto de entrada para desenvolvimento local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
