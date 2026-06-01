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

import io
import json
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.models import create_lenet
from src.utils import (
    get_cifar100_data,
    get_device,
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


class FederatedRequest(BaseModel):
    """Requisição para simulação federada."""
    num_clients: int = Field(default=2, ge=2, le=10, description="Número de clientes")
    num_rounds: int = Field(default=5, ge=1, le=50, description="Rodadas de agregação")
    defense_type: str = Field(default="none", description="Defesa: 'none', 'noisy', 'sparse', 'pruning'")
    noise_std: float = Field(default=0.1, ge=0.0, le=1.0)
    keep_ratio: float = Field(default=0.2, ge=0.01, le=1.0)
    pruning_amount: float = Field(default=0.5, ge=0.0, lt=1.0)
    dlg_iters: int = Field(default=800, ge=10, le=5000, description="Iterações do ataque DLG")
    seed: int = Field(default=50)


class AttackResponse(BaseModel):
    """Resposta do ataque DLG."""
    success: bool
    experiment_name: str
    batch_size: int
    device: str
    loss_history: list[float]
    metrics: Optional[dict] = None
    reconstructed_labels: Optional[list[int]] = None
    message: str


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


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Verifica a saúde do serviço e disponibilidade de dependências."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        device=DEVICE,
        flower_available=FLOWER_AVAILABLE,
        cuda_available=torch.cuda.is_available(),
    )


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

    Seleciona imagens aleatórias (ou específicas) do CIFAR-100,
    calcula os gradientes do modelo LeNet e executa o ataque DLG
    para reconstruí-las.
    """
    try:
        set_seed(request.seed)
        dataset = get_dataset()
        model = get_model()
        model.eval()

        tp = get_image_transform()

        # Selecionar imagens
        if request.image_indices is not None:
            indices = request.image_indices
        else:
            indices = np.random.choice(len(dataset), request.batch_size, replace=False).tolist()

        # Carregar imagens e labels
        gt_images = []
        gt_labels = []
        for idx in indices:
            gt_images.append(tp(dataset[idx][0]))
            gt_labels.append(dataset[idx][1])

        gt_data = torch.stack(gt_images).to(DEVICE)
        gt_label = torch.tensor(gt_labels, dtype=torch.long, device=DEVICE)
        gt_onehot = label_to_onehot(gt_label, num_classes=100)

        # Calcular gradientes originais
        from src.utils import cross_entropy_for_onehot

        model.zero_grad(set_to_none=True)
        out = model(gt_data)
        loss = cross_entropy_for_onehot(out, gt_onehot)
        dy_dx = torch.autograd.grad(loss, model.parameters())
        original_dy_dx = [g.detach().clone() for g in dy_dx]

        # Executar ataque
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

        # Métricas
        metrics_df = evaluate_reconstruction_set(
            reconstructed=result["reconstructed"],
            ground_truth=gt_data,
            labels=gt_label,
            pred_logits=result["dummy_label"],
            name="DLG Attack",
        )

        summary = summarize_metrics(metrics_df)
        metrics_dict = {
            "ssim_mean": float(metrics_df["ssim"].mean()),
            "psnr_mean": float(metrics_df["psnr"].mean()),
            "mse_mean": float(metrics_df["mse"].mean()),
            "ssim_std": float(metrics_df["ssim"].std()),
            "psnr_std": float(metrics_df["psnr"].std()),
        }

        predicted_labels = [
            int(torch.argmax(result["dummy_label"][i]).item())
            for i in range(request.batch_size)
        ]

        return AttackResponse(
            success=True,
            experiment_name="DLG Attack",
            batch_size=request.batch_size,
            device=DEVICE,
            loss_history=[float(l) for l in result["loss_history"]],
            metrics=metrics_dict,
            reconstructed_labels=predicted_labels,
            message=f"Ataque concluído. SSIM médio: {metrics_dict['ssim_mean']:.4f}",
        )

    except Exception as e:
        logger.exception("Erro no ataque DLG")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/run-defense", response_model=AttackResponse)
async def run_defense(request: DefenseRequest):
    """
    Executa o ataque DLG com defesas aplicadas aos gradientes.

    Suporta: Noisy Gradients, Gradient Sparsification e Model Pruning.
    Compara a qualidade da reconstrução com e sem defesa.
    """
    try:
        set_seed(request.seed)
        dataset = get_dataset()
        model = get_model()
        model.eval()

        tp = get_image_transform()

        # Selecionar imagens
        if request.image_indices is not None:
            indices = request.image_indices
        else:
            indices = np.random.choice(len(dataset), request.batch_size, replace=False).tolist()

        gt_images, gt_labels = [], []
        for idx in indices:
            gt_images.append(tp(dataset[idx][0]))
            gt_labels.append(dataset[idx][1])

        gt_data = torch.stack(gt_images).to(DEVICE)
        gt_label = torch.tensor(gt_labels, dtype=torch.long, device=DEVICE)

        # Construir parâmetros da defesa
        defense_params = {}
        if request.defense_type == "noisy":
            defense_params["noise_std"] = request.noise_std
        elif request.defense_type == "sparse":
            defense_params["keep_ratio"] = request.keep_ratio
        elif request.defense_type == "pruning":
            defense_params["amount"] = request.pruning_amount

        # Interceptar e atacar
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
        )

        dlg = result["dlg_result"]

        # Métricas
        metrics_df = evaluate_reconstruction_set(
            reconstructed=dlg["reconstructed"],
            ground_truth=gt_data,
            labels=gt_label,
            pred_logits=dlg["dummy_label"],
            name=f"DLG + {request.defense_type}",
        )

        metrics_dict = {
            "ssim_mean": float(metrics_df["ssim"].mean()),
            "psnr_mean": float(metrics_df["psnr"].mean()),
            "mse_mean": float(metrics_df["mse"].mean()),
            "defense_type": request.defense_type,
        }

        predicted_labels = [
            int(torch.argmax(dlg["dummy_label"][i]).item())
            for i in range(result["batch_size"])
        ]

        return AttackResponse(
            success=True,
            experiment_name=f"DLG + {request.defense_type}",
            batch_size=result["batch_size"],
            device=DEVICE,
            loss_history=[float(l) for l in dlg.get("loss_history", [])],
            metrics=metrics_dict,
            reconstructed_labels=predicted_labels,
            message=(
                f"Ataque com defesa '{request.defense_type}' concluído. "
                f"SSIM médio: {metrics_dict['ssim_mean']:.4f}"
            ),
        )

    except Exception as e:
        logger.exception("Erro no ataque com defesa")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/federated", response_model=FederatedResponse)
async def run_federated(request: FederatedRequest):
    """
    Inicia uma simulação de Aprendizado Federado com Flower.

    Cria múltiplos clientes locais, executa rodadas de FedAvg e,
    se configurado, captura gradientes para o ataque DLG.
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

        results = start_federated_simulation(
            num_clients=request.num_clients,
            num_rounds=request.num_rounds,
            defense_type=request.defense_type,
            defense_params=defense_params,
            pruning_amount=request.pruning_amount,
            device=DEVICE,
            seed=request.seed,
            attacker_enabled=True,
            verbose=True,
        )

        return FederatedResponse(
            success=True,
            num_clients=request.num_clients,
            num_rounds=request.num_rounds,
            defense_type=request.defense_type,
            dlg_results_count=len(results.get("dlg_results", [])),
            message=(
                f"Simulação concluída: {request.num_clients} clientes, "
                f"{request.num_rounds} rodadas, defesa '{request.defense_type}'."
            ),
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
