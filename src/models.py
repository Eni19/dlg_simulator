"""
Arquitetura da rede LeNet e funções de inicialização de pesos.

Esta rede é utilizada como modelo alvo no ataque DLG.
A LeNet original com ativação Sigmoid é mantida conforme o código original,
pois o ataque DLG foi calibrado para esta arquitetura específica.

Classes:
    LeNet: Rede convolucional 4 camadas + FC com ativação Sigmoid.

Funções:
    weights_init: Inicialização uniforme dos pesos no intervalo [-0.5, 0.5].
    create_lenet: Factory function que cria e inicializa uma LeNet.
"""

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """
    Inicializa pesos e bias com distribuição uniforme no intervalo [-0.5, 0.5].

    Args:
        m: Módulo da rede neural a ser inicializado.
    """
    if hasattr(m, "weight") and m.weight is not None:
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias") and m.bias is not None:
        m.bias.data.uniform_(-0.5, 0.5)


class LeNet(nn.Module):
    """
    LeNet modificada para CIFAR-100 com ativação Sigmoid.

    Arquitetura:
        - Conv1: 3→12, kernel=5, pad=2, stride=2, Sigmoid
        - Conv2: 12→12, kernel=5, pad=2, stride=2, Sigmoid
        - Conv3: 12→12, kernel=5, pad=2, stride=1, Sigmoid
        - Conv4: 12→12, kernel=5, pad=2, stride=1, Sigmoid
        - FC: 768→100 (num_classes)

    O uso de Sigmoid (em vez de ReLU) é intencional: o ataque DLG
    original foi desenvolvido e testado com esta configuração exata.
    """

    def __init__(self, num_classes: int = 100):
        """
        Inicializa a LeNet.

        Args:
            num_classes: Número de classes de saída (default: 100 para CIFAR-100).
        """
        super().__init__()
        act = nn.Sigmoid

        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=2, stride=1),
            act(),
        )

        self.fc = nn.Sequential(
            nn.Linear(768, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass da rede.

        Args:
            x: Tensor de entrada [batch, 3, 32, 32].

        Returns:
            Logits [batch, num_classes].
        """
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def create_lenet(
    num_classes: int = 100,
    device: str | torch.device = "cpu",
    seed: int | None = None
) -> LeNet:
    """
    Cria e inicializa uma LeNet com a configuração padrão do ataque DLG.

    Args:
        num_classes: Número de classes (default: 100, CIFAR-100).
        device: torch.device ou string para instanciar o modelo.
        seed: Se fornecida, define a semente aleatória antes da inicialização
              dos pesos para reprodutibilidade.

    Returns:
        LeNet inicializada e movida para o dispositivo especificado.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Initialize on CPU so a seed identifies the same weights on CPU and CUDA.
    model = LeNet(num_classes=num_classes)
    model.apply(weights_init)
    model = model.to(device)
    model.eval()

    return model
