"""
Simulação de Aprendizado Federado (Federated Learning) com Flower (flwr).

Este módulo implementa:
- FlowerClient: Cliente customizado que treina localmente uma LeNet
  no CIFAR-100, expondo os gradientes para interceptação (ataque DLG).
- Defesas configuráveis: Model Pruning, Noisy Gradients, Gradient Sparsification.
- start_federated_simulation: Função principal que orquestra a simulação
  com múltiplos clientes e um servidor FedAvg.

Fluxo do ataque no Federated Learning:
    1. Cliente recebe parâmetros globais do servidor.
    2. Cliente treina localmente (fit), gerando gradientes.
    3. ANTES de enviar os gradientes ao servidor, o atacante os intercepta.
    4. O atacante executa DLG para reconstruir os dados de treino do cliente.
    5. Se defesa ativa, os gradientes são modificados antes do envio.

Uso típico:
    from src.federated_flower import start_federated_simulation
    results = start_federated_simulation(num_clients=2, num_rounds=5)
"""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Flower — NumPyClient é a única dependência real necessária
try:
    from flwr.client import NumPyClient
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    NumPyClient = object  # type: ignore
    import warnings
    warnings.warn(
        "Flower (flwr) não está instalado. "
        "Instale com: pip install flwr"
    )

from .models import LeNet, create_lenet, weights_init
from .utils import (
    cross_entropy_for_onehot,
    get_cifar100_data,
    get_device,
    label_to_onehot,
    set_seed,
)
from .dlg_attack import DLGAttack
from .losses import (
    COSINE_WEIGHT,
    LABEL_ANCHOR_WEIGHT,
    TV_COEFF_START,
    TV_COEFF_END,
)

# ---------------------------------------------------------------------------
# Configuração de Log
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Model Pruning
PRUNING_MODULE_TYPES = (nn.Conv2d, nn.Linear)

# Defesas
NOISE_STD_DEFAULT = 0.1
SPARSE_KEEP_RATIO_DEFAULT = 0.20


# ===================================================================
# Funções de Model Pruning
# ===================================================================

def _iter_prunable_weight_tensors(
    model: nn.Module,
    module_types: tuple = PRUNING_MODULE_TYPES,
):
    """Itera sobre os pesos das camadas elegíveis para poda."""
    for module_name, module in model.named_modules():
        if isinstance(module, module_types) and hasattr(module, "weight"):
            yield module_name, module, module.weight


def apply_global_magnitude_pruning_(
    model: nn.Module,
    amount: float,
    module_types: tuple = PRUNING_MODULE_TYPES,
) -> nn.Module:
    """
    Aplica poda global não estruturada por magnitude, in-place.

    Zera os pesos com menor valor absoluto até atingir a proporção `amount`.
    A arquitetura e a ordem dos parâmetros são preservadas — essencial para
    compatibilidade com o ataque DLG.

    Args:
        model: Modelo a ser podado.
        amount: Proporção de pesos a zerar [0.0, 1.0).
        module_types: Tipos de camada a considerar para poda.

    Returns:
        O próprio modelo (modificado in-place).
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("amount deve estar no intervalo [0.0, 1.0).")

    prunable = list(_iter_prunable_weight_tensors(model, module_types))
    if len(prunable) == 0 or amount == 0.0:
        return model

    with torch.no_grad():
        all_abs = torch.cat(
            [weight.detach().abs().flatten() for _, _, weight in prunable]
        )
        total = all_abs.numel()
        keep = max(1, int(round((1.0 - amount) * total)))

        if keep >= total:
            return model

        threshold = torch.topk(all_abs, keep, largest=True).values[-1]

        for _, _, weight in prunable:
            mask = (weight.detach().abs() >= threshold).to(weight.dtype)
            weight.mul_(mask)

    return model


def parameter_sparsity_stats(
    model: nn.Module,
    only_prunable_weights: bool = False,
) -> dict[str, float | int]:
    """
    Mede a esparsidade dos parâmetros do modelo.

    Args:
        model: Modelo a ser analisado.
        only_prunable_weights: Se True, considera apenas pesos podáveis.

    Returns:
        Dicionário com total, nonzero, zero, sparsity, compression_ratio.
    """
    tensors: list[torch.Tensor] = []
    if only_prunable_weights:
        tensors = [
            weight.detach()
            for _, _, weight in _iter_prunable_weight_tensors(model)
        ]
    else:
        tensors = [param.detach() for param in model.parameters()]

    if len(tensors) == 0:
        return {
            "total": 0, "nonzero": 0, "zero": 0,
            "sparsity": 0.0, "compression_ratio": float("nan"),
        }

    total = sum(t.numel() for t in tensors)
    nonzero = sum(torch.count_nonzero(t).item() for t in tensors)
    zero = total - nonzero
    sparsity = zero / total if total > 0 else 0.0
    compression_ratio = total / max(nonzero, 1)

    return {
        "total": int(total),
        "nonzero": int(nonzero),
        "zero": int(zero),
        "sparsity": float(sparsity),
        "compression_ratio": float(compression_ratio),
    }


# ===================================================================
# Funções de Defesa para Gradientes
# ===================================================================

def add_noisy_gradients(
    gradients: list[torch.Tensor],
    noise_std: float = NOISE_STD_DEFAULT,
) -> list[torch.Tensor]:
    """
    Adiciona ruído gaussiano RELATIVO à escala de cada tensor de gradiente.

    O ruído é proporcional ao desvio padrão de cada gradiente, evitando
    que gradientes pequenos sejam dominados por ruído absoluto.

    Args:
        gradients: Lista de tensores de gradiente.
        noise_std: Desvio padrão relativo do ruído.

    Returns:
        Lista de gradientes com ruído adicionado.
    """
    noisy_grads = []
    for g in gradients:
        g_clean = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        scale = g_clean.std(unbiased=False)

        if not torch.isfinite(scale) or scale.item() < 1e-8:
            scale = g_clean.abs().mean()
        if not torch.isfinite(scale) or scale.item() < 1e-8:
            scale = torch.tensor(1e-8, device=g_clean.device, dtype=g_clean.dtype)

        noise = torch.randn_like(g_clean) * noise_std * scale
        noisy_grads.append(
            torch.nan_to_num(g_clean + noise, nan=0.0, posinf=0.0, neginf=0.0)
        )

    return noisy_grads


def sparsify_gradients(
    grads: list[torch.Tensor],
    keep_ratio: float = SPARSE_KEEP_RATIO_DEFAULT,
) -> list[torch.Tensor]:
    """
    Mantém os maiores gradientes em valor absoluto e zera o restante.

    Args:
        grads: Lista de tensores de gradiente.
        keep_ratio: Proporção de gradientes a manter (top-K).

    Returns:
        Lista de gradientes esparsificados.
    """
    sparse_gradients = []
    for g in grads:
        g_clean = torch.nan_to_num(g.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        flat = g_clean.view(-1)
        k = max(1, int(flat.numel() * keep_ratio))
        k = min(k, flat.numel())

        if flat.numel() == 0:
            sparse_gradients.append(g_clean)
            continue

        threshold = torch.topk(flat.abs(), k).values[-1]
        mask = (g_clean.abs() >= threshold).float()
        sparse_gradients.append(g_clean * mask)

    return sparse_gradients


# ===================================================================
# Conversão de Parâmetros: PyTorch ↔ NumPy (Flower)
# ===================================================================

def get_parameters_from_model(model: nn.Module) -> list[np.ndarray]:
    """
    Extrai os parâmetros do modelo como lista de arrays NumPy.

    Args:
        model: Modelo PyTorch.

    Returns:
        Lista de arrays NumPy (um por parâmetro treinável).
    """
    return [
        param.detach().cpu().numpy()
        for param in model.parameters()
        if param.requires_grad
    ]


def set_model_parameters_from_list(
    model: nn.Module,
    parameters: list[np.ndarray],
) -> None:
    """
    Carrega parâmetros de uma lista NumPy para o modelo PyTorch.

    Itera apenas sobre parâmetros treináveis (requires_grad=True)
    e atualiza seus dados in-place usando torch.from_numpy.

    Args:
        model: Modelo PyTorch a ser atualizado.
        parameters: Lista de arrays NumPy com os novos valores.
    """
    device = next(model.parameters()).device
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) != len(parameters):
        raise ValueError(
            f"Mismatch: modelo tem {len(trainable_params)} parâmetros treináveis, "
            f"mas foram fornecidos {len(parameters)} arrays NumPy."
        )

    with torch.no_grad():
        for param, numpy_val in zip(trainable_params, parameters):
            new_val = torch.from_numpy(numpy_val.copy()).to(
                device=device, dtype=param.dtype
            )
            param.data.copy_(new_val)


def compute_model_gradients(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    criterion: Callable,
) -> list[torch.Tensor]:
    """
    Calcula os gradientes do modelo para um batch de dados.

    Args:
        model: Modelo PyTorch.
        data: Tensor de entrada [B, C, H, W].
        labels: Labels one-hot [B, num_classes].
        criterion: Função de perda.

    Returns:
        Lista de gradientes detached (um por parâmetro).
    """
    model.zero_grad(set_to_none=True)
    output = model(data)
    loss = criterion(output, labels)
    gradients = torch.autograd.grad(loss, model.parameters())
    return [g.detach().clone() for g in gradients]


# ===================================================================
# Cliente Flower Customizado
# ===================================================================

class FlowerClient(NumPyClient):
    """
    Cliente Federated Learning com suporte a ataque DLG e defesas.

    No método fit(), este cliente:
    1. Recebe os parâmetros globais do servidor.
    2. Treina localmente em seu subset de dados.
    3. Calcula e armazena os gradientes gerados (simulando o vazamento).
    4. Aplica a defesa configurada (se houver) antes de retornar.
    5. Expõe os gradientes para o atacante via atributo `intercepted_gradients`.

    Attributes:
        client_id: Identificador único do cliente.
        model: Modelo LeNet local.
        train_loader: DataLoader com os dados locais de treino.
        device: Dispositivo de computação.
        defense_type: Tipo de defesa ('none', 'noisy', 'sparse', 'pruning').
        defense_params: Parâmetros da defesa.
        intercepted_gradients: Últimos gradientes capturados (para o ataque DLG).
        intercepted_data: Último batch de dados usado (para avaliação do ataque).
        intercepted_labels: Últimos labels usados (para avaliação do ataque).
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        device: str = "cpu",
        defense_type: str = "none",
        defense_params: dict | None = None,
        local_epochs: int = 1,
        learning_rate: float = 0.01,
    ):
        """
        Inicializa o cliente Flower.

        Args:
            client_id: ID do cliente.
            model: Modelo LeNet local.
            train_loader: DataLoader com dados de treino do cliente.
            device: 'cuda' ou 'cpu'.
            defense_type: 'none', 'noisy', 'sparse', ou 'pruning'.
            defense_params: Dicionário com parâmetros da defesa.
                - Para 'noisy': {'noise_std': float}
                - Para 'sparse': {'keep_ratio': float}
                - Para 'pruning': {'amount': float}
            local_epochs: Épocas de treino local por rodada.
            learning_rate: Taxa de aprendizado para o otimizador SGD.
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.defense_type = defense_type
        self.defense_params = defense_params or {}
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate

        # Estado capturado para o ataque DLG
        self.intercepted_gradients: list[torch.Tensor] | None = None
        self.intercepted_data: torch.Tensor | None = None
        self.intercepted_labels: torch.Tensor | None = None
        self.intercepted_label_tensor: torch.Tensor | None = None
        self.model_at_capture: nn.Module | None = None  # snapshot do modelo no momento da captura

    # ------------------------------------------------------------------
    # Interface NumPyClient
    # ------------------------------------------------------------------

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        """
        Retorna os parâmetros atuais do modelo como arrays NumPy.

        Requerido pelo Flower para o servidor agregar os modelos.
        """
        return get_parameters_from_model(self.model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """
        Atualiza os parâmetros do modelo a partir do servidor.

        Args:
            parameters: Lista de arrays NumPy com os pesos.
        """
        set_model_parameters_from_list(self.model, parameters)

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """
        Executa treinamento local e expõe gradientes para interceptação.

        Fluxo corrigido para compatibilidade com DLG:
        1. Carrega parâmetros globais e aplica defesas (pruning).
        2. **ANTES** de treinar, captura gradientes do primeiro sub-batch
           usando a MESMA loss function que o DLGAttack usa
           (cross_entropy_for_onehot com labels one-hot).
        3. Salva snapshot do modelo LIMPO (antes de optimizer.step).
        4. Procede com o treinamento normal (F.cross_entropy + defesas).

        Args:
            parameters: Parâmetros globais atuais (do servidor).
            config: Configuração da rodada (não utilizado diretamente).

        Returns:
            Tupla (parameters, num_examples, metrics):
            - parameters: Parâmetros atualizados do modelo.
            - num_examples: Número de exemplos usados no treino.
            - metrics: Métricas adicionais.
        """
        self.set_parameters(parameters)
        self.model.train()

        # Aplicar Model Pruning se configurado
        pruning_amount = 0.0
        if self.defense_type == "pruning":
            pruning_amount = self.defense_params.get("amount", 0.0)
            apply_global_magnitude_pruning_(self.model, amount=pruning_amount)

        # ─── CAPTURA DE GRADIENTES PARA O ATAQUE DLG ─────────────────
        # Fazemos ANTES do treino, com o modelo limpo (mesmos pesos que
        # o atacante usará). Usamos cross_entropy_for_onehot para
        # garantir a mesma loss function que o DLGAttack.run() utiliza.
        # Capturamos apenas as primeiras dlg_capture_size imagens.
        captured_gradients: list[torch.Tensor] | None = None
        first_batch_data: torch.Tensor | None = None
        first_batch_labels: torch.Tensor | None = None

        # Determinar tamanho do sub-batch de captura
        dlg_capture_size = getattr(self, '_dlg_capture_size', 4)

        # Pegar o primeiro batch do dataloader para captura
        try:
            capture_iter = iter(self.train_loader)
            capture_data, capture_labels = next(capture_iter)
            capture_data = capture_data.to(self.device)
            capture_labels = capture_labels.to(self.device)

            # Sub-batch: capturar apenas as primeiras dlg_capture_size imagens
            sub_data = capture_data[:dlg_capture_size].detach().clone()
            sub_labels = capture_labels[:dlg_capture_size].detach().clone()

            first_batch_data = sub_data
            first_batch_labels = sub_labels

            # Snapshot do modelo ANTES do forward — o DLG atacante precisa
            # de exatamente este estado de pesos para recalcular gradientes
            self.model_at_capture = copy.deepcopy(self.model).to(self.device)
            self.model_at_capture.eval()

            # Calcular gradientes com a MESMA loss do DLGAttack:
            # cross_entropy_for_onehot(pred, onehot_labels)
            sub_onehot = label_to_onehot(sub_labels, num_classes=100)
            self.model.zero_grad(set_to_none=True)
            out_capture = self.model(sub_data)
            loss_capture = cross_entropy_for_onehot(out_capture, sub_onehot)
            grads_capture = torch.autograd.grad(
                loss_capture, self.model.parameters()
            )
            captured_gradients = [g.detach().clone() for g in grads_capture]
        except StopIteration:
            logger.warning(
                f"Cliente {self.client_id}: dataloader vazio, "
                f"captura de gradientes ignorada."
            )

        # Armazenar gradientes interceptados para o ataque DLG
        if captured_gradients is not None:
            self.intercepted_gradients = captured_gradients
            self.intercepted_data = first_batch_data
            self.intercepted_labels = first_batch_labels
            self.intercepted_label_tensor = label_to_onehot(
                first_batch_labels, num_classes=100
            )

        # ─── TREINAMENTO LOCAL NORMAL ─────────────────────────────────
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
        )

        total_examples = 0
        total_loss = 0.0

        for epoch in range(self.local_epochs):
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()

                # Forward — treino usa F.cross_entropy normal (labels inteiros)
                output = self.model(data)
                loss = F.cross_entropy(output, labels)
                loss.backward()

                # Aplicar defesa nos gradientes antes do step
                if self.defense_type == "noisy":
                    noise_std = self.defense_params.get("noise_std", NOISE_STD_DEFAULT)
                    for param in self.model.parameters():
                        if param.grad is not None:
                            g_clean = torch.nan_to_num(
                                param.grad, nan=0.0, posinf=0.0, neginf=0.0
                            )
                            scale = g_clean.std(unbiased=False)
                            if not torch.isfinite(scale) or scale.item() < 1e-8:
                                scale = g_clean.abs().mean()
                            if not torch.isfinite(scale) or scale.item() < 1e-8:
                                scale = torch.tensor(1e-8, device=g_clean.device, dtype=g_clean.dtype)
                            noise = torch.randn_like(g_clean) * noise_std * scale
                            param.grad.copy_(
                                torch.nan_to_num(g_clean + noise, nan=0.0, posinf=0.0, neginf=0.0)
                            )
                elif self.defense_type == "sparse":
                    keep_ratio = self.defense_params.get(
                        "keep_ratio", SPARSE_KEEP_RATIO_DEFAULT
                    )
                    for param in self.model.parameters():
                        if param.grad is not None:
                            g_clean = torch.nan_to_num(
                                param.grad, nan=0.0, posinf=0.0, neginf=0.0
                            )
                            flat = g_clean.view(-1)
                            k = max(1, int(flat.numel() * keep_ratio))
                            k = min(k, flat.numel())
                            if flat.numel() > 0:
                                threshold = torch.topk(flat.abs(), k).values[-1]
                                mask = (g_clean.abs() >= threshold).float()
                                param.grad.copy_(g_clean * mask)

                optimizer.step()

                total_examples += data.size(0)
                total_loss += loss.item() * data.size(0)

        avg_loss = total_loss / max(total_examples, 1)

        # Retornar parâmetros atualizados
        updated_params = get_parameters_from_model(self.model)

        metrics = {
            "client_id": self.client_id,
            "loss": float(avg_loss),
            "num_examples": total_examples,
            "defense": self.defense_type,
            "pruning_amount": pruning_amount,
        }

        return updated_params, total_examples, metrics

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        """
        Avalia o modelo nos dados locais.

        Args:
            parameters: Parâmetros do modelo a avaliar.
            config: Configuração da avaliação.

        Returns:
            Tupla (loss, num_examples, metrics).
        """
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, labels in self.train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                output = self.model(data)
                loss = F.cross_entropy(output, labels)
                total_loss += loss.item() * data.size(0)

                pred = output.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += data.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1)

        return float(avg_loss), total, {"accuracy": float(accuracy)}


# ===================================================================
# Cliente Malicioso (Atacante)
# ===================================================================

class AttackerClient(FlowerClient):
    """
    Cliente malicioso que, além de participar do treinamento federado,
    executa o ataque DLG sobre os gradientes interceptados.

    Esta classe estende FlowerClient e adiciona a capacidade de
    reconstruir dados a partir dos gradientes capturados.

    Uso:
        client = AttackerClient(...)
        client.fit(parameters, config)  # treina e captura gradientes
        result = client.attack_dlg()    # executa ataque DLG
    """

    def __init__(
        self,
        *args,
        dlg_num_iters: int = 800,
        dlg_batch_size: int | None = None,
        dlg_seed: int = 50,
        **kwargs,
    ):
        """
        Inicializa o cliente atacante.

        Args:
            *args: Argumentos do FlowerClient.
            dlg_num_iters: Iterações do ataque DLG.
            dlg_batch_size: Tamanho do batch a reconstruir (None = detecta automático).
            dlg_seed: Semente para reprodutibilidade do ataque.
            **kwargs: Argumentos adicionais do FlowerClient.
        """
        super().__init__(*args, **kwargs)
        self.dlg_num_iters = dlg_num_iters
        self.dlg_batch_size = dlg_batch_size
        self.dlg_seed = dlg_seed
        self.dlg_result: dict | None = None

    def attack_dlg(self, verbose: bool = True) -> dict | None:
        """
        Executa o ataque DLG sobre os gradientes interceptados.

        Returns:
            Dicionário com os resultados do ataque, ou None se não houver
            gradientes interceptados.
        """
        if self.intercepted_gradients is None:
            logger.warning("Nenhum gradiente interceptado. Execute fit() primeiro.")
            return None

        # Determinar batch_size automaticamente
        batch_size = self.dlg_batch_size
        if batch_size is None:
            batch_size = min(
                self.intercepted_data.shape[0]
                if self.intercepted_data is not None
                else 3,
                8,
            )

        # Limitar ao tamanho real do batch interceptado
        if self.intercepted_data is not None:
            batch_size = min(batch_size, self.intercepted_data.shape[0])

        # Criar uma cópia do modelo para o ataque
        # CRÍTICO: usar o snapshot do modelo NO MOMENTO da captura dos gradientes,
        # NÃO o modelo pós-treinamento (que foi atualizado por optimizer.step).
        # Se usarmos o modelo errado, os gradientes dummy nunca casarão com
        # os gradientes capturados → loss não converge.
        if self.model_at_capture is not None:
            attack_model = copy.deepcopy(self.model_at_capture).to(self.device)
        else:
            attack_model = copy.deepcopy(self.model).to(self.device)
        attack_model.eval()

        # Criar e executar o ataque
        attack = DLGAttack(
            batch_size=batch_size,
            num_iters=self.dlg_num_iters,
            num_classes=100,
            device=self.device,
            seed=self.dlg_seed,
            verbose=verbose,
        )

        result = attack.run(
            target_gradients=self.intercepted_gradients,
            attack_model=attack_model,
            experiment_name=f"DLG Client {self.client_id}",
        )

        self.dlg_result = result
        return result


# ===================================================================
# Simulação Federada
# ===================================================================

def create_clients(
    num_clients: int = 2,
    device: str = "cpu",
    defense_type: str = "none",
    defense_params: dict | None = None,
    data_root: str = "~/.torch",
    seed: int = 50,
    attacker_client: bool = False,
    dlg_num_iters: int = 800,
    dlg_batch_size: int = 1,
) -> list[FlowerClient]:
    """
    Cria uma lista de clientes Flower para a simulação.

    Cada cliente recebe um subset exclusivo dos dados de treino do CIFAR-100
    (particionamento IID por simplicidade).

    IMPORTANTE: Todos os clientes compartilham a mesma semente para
    inicialização do modelo, garantindo consistência com o modelo global.

    Args:
        num_clients: Número de clientes a criar.
        device: Dispositivo para os modelos.
        defense_type: Tipo de defesa aplicada ('none', 'noisy', 'sparse', 'pruning').
        defense_params: Parâmetros da defesa.
        data_root: Diretório raiz do CIFAR-100.
        seed: Semente para reprodutibilidade da partição.
        attacker_client: Se True, o último cliente é um AttackerClient.
        dlg_num_iters: Iterações do ataque DLG.
        dlg_batch_size: Número de imagens a reconstruir no DLG.

    Returns:
        Lista de FlowerClient (ou AttackerClient) instanciados.
    """
    set_seed(seed)

    # Carregar dataset completo
    from torchvision import datasets, transforms

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.CIFAR100(
        data_root, train=True, transform=train_transform, download=True
    )

    # Particionamento IID: cada cliente recebe ~1/num_clients dos dados
    total_size = len(train_dataset)
    indices = np.random.permutation(total_size)
    split_size = total_size // num_clients

    clients = []
    for i in range(num_clients):
        start = i * split_size
        end = start + split_size if i < num_clients - 1 else total_size
        client_indices = indices[start:end]

        client_dataset = Subset(train_dataset, client_indices)
        client_loader = DataLoader(
            client_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,  # 0 para evitar problemas em Windows/Flower
        )

        # Criar modelo — mesma seed para todos (consistência com modelo global)
        model = create_lenet(num_classes=100, device=device, seed=seed)

        # Escolher classe do cliente
        if attacker_client and i == num_clients - 1:
            client = AttackerClient(
                client_id=i,
                model=model,
                train_loader=client_loader,
                device=device,
                defense_type=defense_type,
                defense_params=defense_params,
                local_epochs=1,
                learning_rate=0.01,
                dlg_num_iters=dlg_num_iters,
                dlg_batch_size=dlg_batch_size,
                dlg_seed=seed,
            )
        else:
            client = FlowerClient(
                client_id=i,
                model=model,
                train_loader=client_loader,
                device=device,
                defense_type=defense_type,
                defense_params=defense_params,
                local_epochs=1,
                learning_rate=0.01,
            )

        # Configurar tamanho de captura para DLG em todos os clientes
        client._dlg_capture_size = dlg_batch_size

        clients.append(client)

    return clients


def fedavg_aggregate(
    results: list[tuple[list[np.ndarray], int, dict[str, Any]]],
    global_params: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Implementação manual do algoritmo FedAvg para agregação de parâmetros.

    Cada cliente retorna seus parâmetros locais atualizados. A agregação
    é feita por média ponderada pelo número de exemplos de cada cliente.

    Args:
        results: Lista de tuplas (parameters, num_examples, metrics)
                 retornadas por fit() de cada cliente.
        global_params: Parâmetros globais atuais (usados como base).

    Returns:
        Novos parâmetros globais agregados.
    """
    total_examples = sum(num_ex for _, num_ex, _ in results)
    if total_examples == 0:
        return global_params

    # Inicializar acumulador com zeros
    aggregated = [
        np.zeros_like(p, dtype=np.float64) for p in global_params
    ]

    # Média ponderada
    for params, num_ex, _ in results:
        weight = num_ex / total_examples
        for i, param in enumerate(params):
            aggregated[i] += param.astype(np.float64) * weight

    return aggregated


def start_federated_simulation(
    num_clients: int = 2,
    num_rounds: int = 5,
    defense_type: str = "none",
    defense_params: dict | None = None,
    pruning_amount: float = 0.0,
    device: str | None = None,
    seed: int = 50,
    data_root: str = "~/.torch",
    attacker_enabled: bool = True,
    attacker_role: str = "malicious_server",
    verbose: bool = True,
    dlg_batch_size: int = 3,
    dlg_num_iters: int = 800,
) -> dict[str, Any]:
    """
    Inicia uma simulação de Aprendizado Federado manual (FedAvg local).

    Diferente da API `start_simulation` do Flower (que exige Ray), esta
    implementação orquestra as rodadas FedAvg diretamente no processo atual,
    chamando fit() e evaluate() de cada cliente sequencialmente e
    agregando os parâmetros com média ponderada.

    Fluxo:
        1. Parâmetros globais iniciais vêm do primeiro cliente.
        2. A cada rodada:
           a. Distribui parâmetros globais para cada cliente.
           b. Cada cliente treina localmente (fit).
           c. Atacante captura gradientes do primeiro batch.
           d. Servidor agrega parâmetros via FedAvg.
        3. Após todas as rodadas, executa DLG sobre gradientes capturados.

    Args:
        num_clients: Número de clientes na simulação (≥2).
        num_rounds: Número de rodadas de agregação.
        defense_type: Defesa aplicada pelos clientes:
            'none', 'noisy', 'sparse', ou 'pruning'.
        defense_params: Parâmetros específicos da defesa.
        pruning_amount: Quantidade de poda (se defense_type='pruning').
        device: Dispositivo ('cuda' ou 'cpu'). None = auto-detecção.
        seed: Semente para reprodutibilidade.
        data_root: Diretório raiz do CIFAR-100.
        attacker_enabled: Se True, inclui um AttackerClient.
        attacker_role: Papel do atacante ('malicious_server', 'malicious_client', 'mitm').
        verbose: Se True, imprime logs detalhados.
        dlg_batch_size: Número de imagens a reconstruir no DLG.
        dlg_num_iters: Iterações do ataque DLG.

    Returns:
        Dicionário com:
        - 'clients': Lista de clientes (com gradientes interceptados).
        - 'dlg_results': Resultados do ataque DLG (se attacker_enabled).
        - 'history': Histórico de métricas por rodada.
        - 'global_params': Parâmetros globais finais.
    """
    if device is None:
        device = get_device(verbose=verbose)

    if verbose:
        mode = "CPU" if device == "cpu" else "CUDA"
        print(f"\n{'='*70}")
        print(f" Simulação Federada — FedAvg Manual + DLG Attack")
        print(f" Clientes: {num_clients} | Rodadas: {num_rounds}")
        print(f" Defesa: {defense_type} | Dispositivo: {mode}")
        print(f"{'='*70}")

    # Configurar defesa
    if defense_type == "pruning" and pruning_amount > 0:
        if defense_params is None:
            defense_params = {}
        defense_params.setdefault("amount", pruning_amount)

    # Criar clientes
    clients = create_clients(
        num_clients=num_clients,
        device=device,
        defense_type=defense_type,
        defense_params=defense_params,
        data_root=data_root,
        seed=seed,
        attacker_client=attacker_enabled,
        dlg_num_iters=dlg_num_iters,
        dlg_batch_size=dlg_batch_size,
    )

    if verbose:
        for c in clients:
            atk = " [ATACANTE]" if isinstance(c, AttackerClient) else ""
            print(f"  Cliente {c.client_id}: defesa={c.defense_type}{atk}")

    # Obter parâmetros globais iniciais do primeiro cliente
    global_params = clients[0].get_parameters({})

    # Histórico de métricas
    history: list[dict] = []

    # ═══════════════════════════════════════════════════════════════
    # LOOP DE RODADAS FEDAVG
    # ═══════════════════════════════════════════════════════════════
    for round_idx in range(1, num_rounds + 1):
        if verbose:
            print(f"\n--- Rodada {round_idx}/{num_rounds} ---")

        round_results: list[tuple[list[np.ndarray], int, dict[str, Any]]] = []

        for client in clients:
            # Treino local (fit) — fit() já chama set_parameters internamente
            updated_params, num_ex, metrics = client.fit(
                parameters=global_params,
                config={"round": round_idx},
            )
            round_results.append((updated_params, num_ex, metrics))

            if verbose:
                atk_tag = " [ATK]" if isinstance(client, AttackerClient) else ""
                loss_str = metrics.get("loss", "?")
                print(
                    f"  Cliente {client.client_id}{atk_tag}: "
                    f"loss={loss_str:.4f}, "
                    f"exemplos={num_ex}"
                )

        # Agregação FedAvg
        global_params = fedavg_aggregate(round_results, global_params)

        # Registrar métricas da rodada
        avg_loss = np.mean([
            m.get("loss", 0)
            for _, _, m in round_results
        ])
        total_ex = sum(num_ex for _, num_ex, _ in round_results)
        history.append({
            "round": round_idx,
            "avg_loss": float(avg_loss),
            "total_examples": total_ex,
        })

        if verbose:
            print(f"  [Servidor] Agregado — loss média: {avg_loss:.4f}")

    # ═══════════════════════════════════════════════════════════════
    # ATAQUE DLG (após última rodada)
    # ═══════════════════════════════════════════════════════════════
    dlg_results: list[dict] = []
    if attacker_enabled:
        for client in clients:
            if isinstance(client, AttackerClient):
                if client.intercepted_gradients is None:
                    if verbose:
                        print(
                            f"\n[Aviso] Cliente {client.client_id} (ATK) "
                            f"não tem gradientes interceptados."
                        )
                    continue

                if verbose:
                    print(
                        f"\n{'='*60}"
                        f"\n Executando ataque DLG no Cliente {client.client_id}..."
                        f"\n{'='*60}"
                    )

                result = client.attack_dlg(verbose=verbose)
                if result is not None:
                    dlg_results.append(result)

    # Sumário
    metrics_summary = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "defense_type": defense_type,
        "defense_params": defense_params,
        "device": device,
        "clients": clients,
        "dlg_results": dlg_results,
        "history": history,
        "global_params": global_params,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f" Simulação concluída.")
        print(f" Clientes: {num_clients} | Rodadas: {num_rounds}")
        print(f" Ataques DLG realizados: {len(dlg_results)}")
        print(f" Loss final (média): {history[-1]['avg_loss']:.4f}" if history else "")
        print(f"{'='*70}")

    return metrics_summary


# ===================================================================
# Função de Interceptação Isolada (para uso via API)
# ===================================================================

def intercept_and_attack(
    model: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
    defense_type: str = "none",
    defense_params: dict | None = None,
    dlg_iters: int = 800,
    dlg_seed: int = 50,
    verbose: bool = True,
    informed_attacker: bool = True,
) -> dict:
    """
    Função standalone que simula a interceptação de gradientes
    e executa o ataque DLG — útil para chamadas via API.

    Args:
        model: Modelo alvo (LeNet).
        data: Batch de dados [B, C, H, W].
        labels: Labels [B] (inteiros).
        device: Dispositivo.
        defense_type: Defesa aplicada ('none', 'noisy', 'sparse', 'pruning').
        defense_params: Parâmetros da defesa.
        dlg_iters: Iterações do ataque DLG.
        dlg_seed: Semente do ataque.
        verbose: Controla logs.
        informed_attacker: Se True, o atacante usa o mesmo modelo (informado).
            Se False, usa o modelo original sem poda (desinformado).
            Apenas relevante quando defense_type='pruning'.

    Returns:
        Dicionário com:
        - 'original_gradients': Gradientes sem defesa.
        - 'defended_gradients': Gradientes após defesa.
        - 'dlg_result': Resultado do ataque DLG.
    """
    device = device or get_device(verbose=False)
    model = model.to(device)
    model.eval()

    data = data.to(device)
    labels = labels.to(device)
    onehot_labels = label_to_onehot(labels, num_classes=100)

    # Calcular gradientes originais
    original_gradients = compute_model_gradients(
        model, data, onehot_labels, cross_entropy_for_onehot
    )

    # Aplicar defesa
    defended_gradients = original_gradients
    pruned_model = None  # referência ao modelo podado, se aplicável
    if defense_type == "noisy":
        noise_std = (defense_params or {}).get("noise_std", NOISE_STD_DEFAULT)
        defended_gradients = add_noisy_gradients(original_gradients, noise_std)
    elif defense_type == "sparse":
        keep_ratio = (defense_params or {}).get("keep_ratio", SPARSE_KEEP_RATIO_DEFAULT)
        defended_gradients = sparsify_gradients(original_gradients, keep_ratio)
    elif defense_type == "pruning":
        amount = (defense_params or {}).get("amount", 0.0)
        pruned_model = copy.deepcopy(model)
        apply_global_magnitude_pruning_(pruned_model, amount=amount)
        defended_gradients = compute_model_gradients(
            pruned_model, data, onehot_labels, cross_entropy_for_onehot
        )

    # Executar ataque DLG
    # Atacante informado (informed_attacker=True): usa o mesmo modelo que gerou
    #   os gradientes — se pruning, usa pruned_model.
    # Atacante desinformado (informed_attacker=False): usa sempre o modelo
    #   original — incompatível quando cliente usa pruning.
    if defense_type == "pruning" and informed_attacker and pruned_model is not None:
        attack_model = pruned_model
        attack_model.eval()
    else:
        attack_model = copy.deepcopy(model).to(device)
        attack_model.eval()

    batch_size = min(data.shape[0], 8)

    # Se pruning desinformado: usar loss mais robusta (safe) pois há mismatch
    use_safe = (defense_type == "pruning" and not informed_attacker)

    attack = DLGAttack(
        batch_size=batch_size,
        num_iters=dlg_iters,
        num_classes=100,
        device=device,
        seed=dlg_seed,
        use_safe_loss=use_safe,
        verbose=verbose,
    )

    dlg_result = attack.run(
        target_gradients=defended_gradients,
        attack_model=attack_model,
        experiment_name="API DLG Attack",
    )

    return {
        "original_gradients": original_gradients,
        "defended_gradients": defended_gradients,
        "dlg_result": dlg_result,
        "defense_type": defense_type,
        "batch_size": batch_size,
        "informed_attacker": informed_attacker,
    }
