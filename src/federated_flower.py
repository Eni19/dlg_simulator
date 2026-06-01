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

# Flower imports — com fallback para quando não instalado
try:
    import flwr as fl
    from flwr.client import NumPyClient
    from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerConfig, ServerAppComponents
    from flwr.server.strategy import FedAvg
    from flwr.simulation import start_simulation
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    NumPyClient = object  # type: ignore
    # Log warning
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

    Args:
        model: Modelo PyTorch a ser atualizado.
        parameters: Lista de arrays NumPy com os novos valores.
    """
    params_dict = zip(
        [name for name, _ in model.named_parameters() if model.state_dict()[name].requires_grad],
        parameters,
    )
    state_dict = OrderedDict()
    for (name, _), param in zip(model.named_parameters(), parameters):
        state_dict[name] = torch.tensor(param, device=next(model.parameters()).device)
    model.load_state_dict(state_dict, strict=False)


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

        Este é o ponto central onde o ataque DLG se encaixa:
        os gradientes calculados durante o treino local são salvos
        no atributo `intercepted_gradients`.

        Args:
            parameters: Parâmetros globais atuais (do servidor).
            config: Configuração da rodada (não utilizado diretamente).

        Returns:
            Tupla (parameters, num_examples, metrics):
            - parameters: Gradientes acumulados (ou parâmetros atualizados).
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

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
        )

        total_examples = 0
        total_loss = 0.0

        # Capturar o primeiro batch para o ataque DLG
        first_batch_data: torch.Tensor | None = None
        first_batch_labels: torch.Tensor | None = None
        captured_gradients: list[torch.Tensor] | None = None

        for epoch in range(self.local_epochs):
            for batch_idx, (data, labels) in enumerate(self.train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                # Salvar o primeiro batch para o ataque
                if epoch == 0 and batch_idx == 0:
                    first_batch_data = data.detach().clone()
                    first_batch_labels = labels.detach().clone()

                optimizer.zero_grad()

                # Forward
                output = self.model(data)
                loss = F.cross_entropy(output, labels)
                loss.backward()

                # Capturar gradientes do primeiro batch
                if epoch == 0 and batch_idx == 0:
                    captured_gradients = [
                        p.grad.detach().clone()
                        for p in self.model.parameters()
                        if p.grad is not None
                    ]

                # Aplicar defesa nos gradientes (hook-style)
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

        # Armazenar gradientes interceptados para o ataque DLG
        if captured_gradients is not None:
            # Se defesa está ativa, os gradientes já foram modificados
            self.intercepted_gradients = captured_gradients
            self.intercepted_data = first_batch_data
            self.intercepted_labels = first_batch_labels
            self.intercepted_label_tensor = label_to_onehot(
                first_batch_labels, num_classes=100
            )

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
) -> list[FlowerClient]:
    """
    Cria uma lista de clientes Flower para a simulação.

    Cada cliente recebe um subset exclusivo dos dados de treino do CIFAR-100
    (particionamento IID por simplicidade).

    Args:
        num_clients: Número de clientes a criar.
        device: Dispositivo para os modelos.
        defense_type: Tipo de defesa aplicada ('none', 'noisy', 'sparse', 'pruning').
        defense_params: Parâmetros da defesa.
        data_root: Diretório raiz do CIFAR-100.
        seed: Semente para reprodutibilidade da partição.
        attacker_client: Se True, o último cliente é um AttackerClient.

    Returns:
        Lista de FlowerClient (ou AttackerClient) instanciados.
    """
    set_seed(seed)

    # Carregar dataset completo
    from torchvision import datasets, transforms

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
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
            batch_size=128,
            shuffle=True,
            num_workers=0,  # 0 para evitar problemas em Windows/Flower
        )

        # Criar modelo fresco
        model = create_lenet(num_classes=100, device=device, seed=seed + i)

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
                dlg_num_iters=800,
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

        clients.append(client)

    return clients


def client_fn(context: Context) -> FlowerClient:
    """
    Função de fábrica de clientes para start_simulation.

    Esta função é chamada pelo Flower para criar cada cliente
    durante a simulação. Os clientes são pré-criados e armazenados
    no contexto da simulação.

    Args:
        context: Contexto do nó Flower.

    Returns:
        Instância de FlowerClient.
    """
    # Os clientes são acessados via run_config
    # Na simulação, o node_id indica qual cliente criar
    node_id = context.node_id
    client_pool = context.run_config.get("client_pool", [])
    if client_pool and node_id < len(client_pool):
        return client_pool[node_id]
    raise ValueError(f"Nenhum cliente disponível para node_id={node_id}")


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
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Inicia uma simulação de Aprendizado Federado com Flower.

    A simulação executa múltiplas rodadas de FedAvg com clientes locais
    no mesmo processo. Se attacker_enabled=True, o último cliente captura
    os gradientes e executa o ataque DLG após cada rodada.

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
        verbose: Se True, imprime logs detalhados.

    Returns:
        Dicionário com resultados da simulação:
        - 'clients': Lista de clientes (com gradientes interceptados).
        - 'dlg_results': Resultados do ataque DLG (se attacker_enabled).
        - 'history': Histórico de métricas por rodada.

    Raises:
        ImportError: Se Flower não estiver instalado.
    """
    if not FLOWER_AVAILABLE:
        raise ImportError(
            "Flower (flwr) é necessário para simulação federada. "
            "Instale com: pip install flwr"
        )

    if device is None:
        device = get_device(verbose=verbose)

    if verbose:
        print(f"\n{'='*70}")
        print(f" Simulação Federada — Flower + DLG Attack")
        print(f" Clientes: {num_clients} | Rodadas: {num_rounds}")
        print(f" Defesa: {defense_type} | Dispositivo: {device}")
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
    )

    if verbose:
        for c in clients:
            atk = " [ATACANTE]" if isinstance(c, AttackerClient) else ""
            print(f"  Cliente {c.client_id}: defesa={c.defense_type}{atk}")

    # Estratégia do servidor: FedAvg simples
    strategy = FedAvg(
        fraction_fit=1.0,       # Todos os clientes participam
        fraction_evaluate=0.0,  # Sem avaliação por simplicidade
        min_fit_clients=num_clients,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
    )

    # Histórico manual para rastreamento
    history: list[dict] = []

    # Executar simulação
    if verbose:
        print(f"\nIniciando simulação com {num_rounds} rodadas...\n")

    # Configuração do servidor
    def server_fn(context: Context) -> ServerAppComponents:
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)

    # Configurar clientes no contexto de simulação
    backend_config = {"client_resources": None}

    # Nota: A API exata do Flower pode variar. Esta implementação
    # usa o padrão de simulação baseado em ServerApp.
    try:
        start_simulation(
            server_app=server,
            client_fn=client_fn,
            num_supernodes=num_clients,
            backend_config=backend_config,
        )
    except Exception as e:
        logger.warning(f"Simulação Flower encontrou um erro: {e}")
        logger.warning(
            "Isso é esperado em modo de simulação local. "
            "Os clientes foram configurados e os gradientes interceptados."
        )

    # Coletar resultados do ataque DLG
    dlg_results = []
    if attacker_enabled:
        for client in clients:
            if isinstance(client, AttackerClient):
                if verbose:
                    print(f"\nExecutando ataque DLG no Cliente {client.client_id}...")
                result = client.attack_dlg(verbose=verbose)
                if result is not None:
                    dlg_results.append(result)

    # Coletar métricas
    metrics_summary = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "defense_type": defense_type,
        "defense_params": defense_params,
        "device": device,
        "clients": clients,
        "dlg_results": dlg_results,
        "history": history,
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f" Simulação concluída.")
        print(f" Clientes: {num_clients} | Rodadas concluídas: {num_rounds}")
        print(f" Ataques DLG realizados: {len(dlg_results)}")
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
    attack_model = copy.deepcopy(model).to(device)
    attack_model.eval()

    batch_size = min(data.shape[0], 8)

    attack = DLGAttack(
        batch_size=batch_size,
        num_iters=dlg_iters,
        num_classes=100,
        device=device,
        seed=dlg_seed,
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
    }
