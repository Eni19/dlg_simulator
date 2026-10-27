"""
Deep Leakage from Gradients (DLG) — Motor do Ataque de Reconstrução.

Implementa o loop de otimização do ataque DLG com as seguintes melhorias:

[A] Inicialização Inteligente:
    Projeta os gradientes originais no espaço de imagem via hint espectral
    da primeira camada convolucional, criando pontos iniciais mais informativos
    que inicialização puramente aleatória.

[B] TV Adaptativo com Cosine Annealing:
    Coeficiente de Total Variation decai de TV_COEFF_START para TV_COEFF_END
    seguindo cosine annealing — preserva estrutura no início e detalhes no fim.

[C] Perda Composta L2 + Cosine:
    Combina MSE com cosine similarity entre vetores de gradiente flatten.
    Cosine é sensível à direção → discrimina estruturas finas que L2 ignora.

[D] Label Anchoring Analítico:
    Regularização suave que puxa os logits reconstruídos para a estimativa
    analítica obtida do gradiente do bias da última camada.

[E] Fase Global Periódica:
    A cada GLOBAL_PHASE_FREQ ciclos, um passo LBFGS otimiza TODAS as imagens
    simultaneamente, re-sincronizando o batch e capturando correlações.

Classes:
    DLGAttack: Encapsula toda a lógica do ataque DLG.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from .progress import report

from .losses import (
    COSINE_WEIGHT,
    GLOBAL_PHASE_FREQ,
    GRAD_EPS,
    LABEL_ANCHOR_WEIGHT,
    LOGIT_CLAMP,
    TV_COEFF_END,
    TV_COEFF_START,
    grad_matching_loss,
    safe_grad_matching_loss_defense,
    tv_cosine_coefficient,
    tv_loss,
)
from .utils import (
    clone_gradients,
    cross_entropy_for_onehot,
    sanitize_image_tensor,
    get_to_pil_transform,
)


class DLGAttack:
    """
    Executor do ataque Deep Leakage from Gradients (DLG).

    Dados os gradientes interceptados de um modelo, reconstrói as imagens
    de entrada que geraram esses gradientes usando otimização LBFGS com
    perda composta e regularização adaptativa.

    Attributes:
        batch_size: Número de imagens no batch a reconstruir.
        num_iters: Número total de iterações do ataque.
        num_classes: Número de classes da tarefa (100 para CIFAR-100).
        device: Dispositivo de computação ('cuda' ou 'cpu').
        cosine_weight: Peso da componente cosine na perda composta.
        tv_coeff_start: Coeficiente TV inicial (forte).
        tv_coeff_end: Coeficiente TV final (fraco).
        label_anchor_weight: Peso da regularização de label anchoring.
        global_phase_freq: Frequência da fase global.
        use_safe_loss: Se True, usa a versão numericamente estável da loss.
        verbose: Se True, imprime progresso durante o ataque.
    """

    def __init__(
        self,
        batch_size: int = 3,
        num_iters: int = 800,
        num_classes: int = 100,
        device: str | torch.device = "cpu",
        cosine_weight: float = COSINE_WEIGHT,
        tv_coeff_start: float = TV_COEFF_START,
        tv_coeff_end: float = TV_COEFF_END,
        label_anchor_weight: float = LABEL_ANCHOR_WEIGHT,
        global_phase_freq: int = GLOBAL_PHASE_FREQ,
        use_safe_loss: bool = False,
        seed: int | None = None,
        verbose: bool = True,
    ):
        """
        Inicializa o ataque DLG com todos os hiperparâmetros configuráveis.

        Args:
            batch_size: Número de imagens a reconstruir simultaneamente (máx. 8).
            num_iters: Número total de iterações do ataque.
            num_classes: Número de classes (100 = CIFAR-100).
            device: 'cuda' ou 'cpu'.
            cosine_weight: Peso da cosine similarity na perda composta.
            tv_coeff_start: Coeficiente TV inicial.
            tv_coeff_end: Coeficiente TV final.
            label_anchor_weight: Peso da regularização de label.
            global_phase_freq: Frequência de passos globais.
            use_safe_loss: Usa versão robusta da loss para defesas.
            seed: Semente para reprodutibilidade.
            verbose: Controla impressão de logs de progresso.
        """
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.num_classes = num_classes
        self.device = device
        self.cosine_weight = cosine_weight
        self.tv_coeff_start = tv_coeff_start
        self.tv_coeff_end = tv_coeff_end
        self.label_anchor_weight = label_anchor_weight
        self.global_phase_freq = global_phase_freq
        self.use_safe_loss = use_safe_loss
        self.verbose = verbose

        if seed is not None:
            torch.manual_seed(seed)

        # Cached transforms
        self._to_pil = get_to_pil_transform()

        # Estado interno populado durante run()
        self.history: list[list] = []
        self.loss_history: list[float] = []
        self.ssim_history: list[float] = []
        self.psnr_history: list[float] = []
        self.mse_history: list[float] = []

    # ------------------------------------------------------------------
    # Label Anchoring Analítico
    # ------------------------------------------------------------------

    def _make_anchor_from_gradients(
        self,
        target_gradients: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extrai estimativa analítica dos labels a partir do gradiente do bias.

        Para cross-entropy com one-hot, o gradiente do bias da última camada
        é: ∂L/∂b = softmax(logits) - onehot(y). Os labels VERDADEIROS
        correspondem às entradas com gradiente MAIS NEGATIVO.

        Args:
            target_gradients: Lista de gradientes capturados.

        Returns:
            Tupla (pred_labels, anchor):
            - pred_labels: Labels estimados [batch_size].
            - anchor: Tensor one-hot [batch_size, num_classes].
        """
        bias_grad = torch.nan_to_num(
            target_gradients[-1].detach(),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        pred_labels = torch.argsort(bias_grad)[: self.batch_size]

        anchor = torch.zeros(self.batch_size, self.num_classes, device=self.device)
        for i, lbl in enumerate(pred_labels.tolist()):
            anchor[i, lbl] = 1.0

        return pred_labels, anchor

    # ------------------------------------------------------------------
    # Inicialização Inteligente
    # ------------------------------------------------------------------

    def _smart_initialization(
        self,
        target_gradients: list[torch.Tensor],
        pred_labels: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """
        Inicializa imagens e labels dummy usando informação dos gradientes.

        Imagens: Combina ruído uniforme com hint espectral da primeira
        camada convolucional, decrescente por índice (imagens posteriores
        recebem menos informação do gradiente, promovendo diversidade).

        Labels: Inicializa próximos da estimativa analítica com pequena
        perturbação gaussiana.

        Args:
            target_gradients: Gradientes originais capturados.
            pred_labels: Labels estimados via bias gradient.

        Returns:
            Tupla (dummy_data_list, dummy_label_list) com tensores
            requerendo gradiente.
        """
        with torch.no_grad():
            dummy_data_list = []
            for k in range(self.batch_size):
                # Base: ruído uniforme
                # CPU RNG gives the same starting noise for a seed on each device.
                base = torch.rand(1, 3, 32, 32).to(self.device)

                # Hint espectral da primeira camada convolucional
                input_grad = target_gradients[0]  # [out_ch=12, in_ch=3, kH=5, kW=5]
                n_filters = input_grad.shape[0]  # 12

                f_idx = (k * 3) % n_filters
                filters = input_grad[f_idx: f_idx + 3]

                if filters.shape[0] < 3:
                    pad = input_grad[: 3 - filters.shape[0]]
                    filters = torch.cat([filters, pad], dim=0)

                hint = filters.mean(dim=1, keepdim=True)  # [3, 1, 5, 5]
                hint = F.interpolate(
                    hint, size=(32, 32), mode="bilinear", align_corners=False
                )
                hint = hint.squeeze(1).unsqueeze(0)  # [1, 3, 32, 32]
                hint = torch.nan_to_num(hint, nan=0.0, posinf=0.0, neginf=0.0)
                hint = (hint - hint.min()) / (hint.max() - hint.min() + 1e-8)

                # Combinação: alpha decresce com k para promover diversidade
                alpha = 0.25 / (k + 1)
                init = (1 - alpha) * base + alpha * hint
                dummy_data_list.append(init.clamp(0, 1).requires_grad_(True))

        dummy_label_list = []
        for k in range(self.batch_size):
            logit_init = torch.zeros(1, self.num_classes, device=self.device)
            lbl = pred_labels[k % len(pred_labels)].item()
            logit_init[0, lbl] = 5.0
            noise = torch.randn(logit_init.shape, dtype=logit_init.dtype).to(self.device)
            logit_init = logit_init + 0.1 * noise
            dummy_label_list.append(
                logit_init.clamp(-LOGIT_CLAMP, LOGIT_CLAMP).requires_grad_(True)
            )

        return dummy_data_list, dummy_label_list

    # ------------------------------------------------------------------
    # Reparo de Variáveis (estabilidade numérica)
    # ------------------------------------------------------------------

    def _repair_variables(
        self,
        dummy_data_list: list[torch.Tensor],
        dummy_label_list: list[torch.Tensor],
    ) -> None:
        """Corrige NaN/Inf in-place nas imagens e labels dummy."""
        for dd in dummy_data_list:
            dd.data.copy_(
                torch.nan_to_num(
                    dd.data, nan=0.0, posinf=1.0, neginf=0.0
                ).clamp(0.0, 1.0)
            )
        for dl in dummy_label_list:
            dl.data.copy_(
                torch.nan_to_num(
                    dl.data, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP
                ).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
            )

    # ------------------------------------------------------------------
    # Execução Principal do Ataque
    # ------------------------------------------------------------------

    def run(
        self,
        target_gradients: list[torch.Tensor],
        attack_model: nn.Module,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        experiment_name: str = "DLG Attack",
    ) -> dict:
        """
        Executa o ataque DLG completo.

        Args:
            target_gradients: Lista de gradientes interceptados (um por parâmetro
                              do modelo), detached e em CPU ou GPU.
            attack_model: Modelo usado pelo atacante para calcular gradientes
                          a partir das imagens dummy. Deve estar em eval mode.
            criterion: Função de perda usada no forward. Se None, usa
                       cross_entropy_for_onehot.
            experiment_name: Nome para identificação nos logs.

        Returns:
            Dicionário com:
            - 'reconstructed': Tensor [batch_size, 3, 32, 32] das imagens reconstruídas.
            - 'dummy_label': Tensor [batch_size, num_classes] dos logits recuperados.
            - 'history': Lista de listas de PIL Images por amostra.
            - 'loss_history': Lista de valores da loss.
            - 'target_gradients': Cópia sanitizada dos gradientes alvo.
            - 'snapshots': Lista de tensores [batch_size, 3, 32, 32] em checkpoints.
            - 'snapshot_iters': Lista de int com a iteração de cada snapshot.
        """
        if criterion is None:
            criterion = cross_entropy_for_onehot

        attack_model.eval()

        # Sanitizar gradientes alvo
        target_gradients = [
            torch.nan_to_num(
                g.detach().clone().to(self.device),
                nan=0.0, posinf=0.0, neginf=0.0,
            )
            for g in target_gradients
        ]

        if self.verbose:
            print(f"\n{'='*70}")
            print(f" {experiment_name}")
            print(f" Batch size: {self.batch_size} | Iterações: {self.num_iters}")
            print(f" Dispositivo: {self.device}")
            print(f"{'='*70}")

        # Label anchoring
        pred_labels, anchor_labels = self._make_anchor_from_gradients(target_gradients)

        if self.verbose:
            print(f"Labels analíticos estimados: {sorted(pred_labels.tolist())}")

        # Inicialização inteligente
        dummy_data_list, dummy_label_list = self._smart_initialization(
            target_gradients, pred_labels
        )

        # Selecionar a função de loss apropriada
        if self.use_safe_loss:
            matching_loss_fn = safe_grad_matching_loss_defense
        else:
            matching_loss_fn = grad_matching_loss

        # ── Configuração dos Otimizadores LBFGS ─────────────────────────
        # Um otimizador por amostra (round-robin) + um otimizador global

        lr_local = 0.05 if self.use_safe_loss else 1.0
        max_iter_local = 3 if self.use_safe_loss else 20
        history_size_local = 10 if self.use_safe_loss else 100
        lr_global = 0.02 if self.use_safe_loss else 0.5
        max_iter_global = 2 if self.use_safe_loss else 10
        history_size_global = 10 if self.use_safe_loss else 50

        optimizers = [
            torch.optim.LBFGS(
                [dummy_data_list[i], dummy_label_list[i]],
                lr=lr_local,
                max_iter=max_iter_local,
                history_size=history_size_local,
                tolerance_grad=1e-7 if not self.use_safe_loss else 1e-5,
                tolerance_change=1e-9 if not self.use_safe_loss else 1e-7,
                line_search_fn="strong_wolfe" if not self.use_safe_loss else None,
            )
            for i in range(self.batch_size)
        ]

        optimizer_global = torch.optim.LBFGS(
            dummy_data_list + dummy_label_list,
            lr=lr_global,
            max_iter=max_iter_global,
            history_size=history_size_global,
            tolerance_grad=1e-6 if not self.use_safe_loss else 1e-5,
            tolerance_change=1e-8 if not self.use_safe_loss else 1e-7,
            line_search_fn="strong_wolfe" if not self.use_safe_loss else None,
        )

        # Estado
        history: list[list] = [[] for _ in range(self.batch_size)]
        loss_history: list[float] = []
        last_valid_loss: float | None = None

        total_cycles = self.num_iters // self.batch_size
        report("optimizing", experiment_name, 0, total_cycles)
        _loss_log: list[float | None] = [None]

        # Checkpoints para snapshots da reconstrução
        snapshot_checkpoints = 10
        snapshot_interval = max(1, total_cycles // snapshot_checkpoints)
        snapshot_cycles = set(
            range(0, total_cycles, snapshot_interval)
        )
        # Garantir que o último ciclo está incluso
        snapshot_cycles.add(total_cycles - 1)
        snapshots: list[torch.Tensor] = []  # cada entrada: [B, C, H, W]
        snapshot_iters: list[int] = []       # iteração correspondente

        # ═══════════════════════════════════════════════════════════════
        # LOOP PRINCIPAL DO ATAQUE
        # ═══════════════════════════════════════════════════════════════
        for cycle in range(total_cycles):
            # Reparo de estabilidade numérica (para defesas)
            if self.use_safe_loss:
                self._repair_variables(dummy_data_list, dummy_label_list)

            # ── TV adaptativo com cosine annealing ─────────────────
            progress = cycle / max(total_cycles - 1, 1)
            tv_coeff = tv_cosine_coefficient(progress)

            # ── Fase global periódica ──────────────────────────────
            if cycle % self.global_phase_freq == 0 and cycle > 0:
                def closure_global():
                    optimizer_global.zero_grad()
                    if self.use_safe_loss:
                        self._repair_variables(dummy_data_list, dummy_label_list)

                    batch_data = torch.cat([d for d in dummy_data_list], dim=0)
                    batch_label = torch.cat(
                        [
                            F.softmax(
                                l.clamp(-LOGIT_CLAMP, LOGIT_CLAMP), dim=-1
                            )
                            for l in dummy_label_list
                        ],
                        dim=0,
                    )

                    pred = attack_model(batch_data)
                    d_loss = criterion(pred, batch_label)
                    d_dy_dx = torch.autograd.grad(
                        d_loss, attack_model.parameters(), create_graph=True
                    )

                    loss = matching_loss_fn(
                        d_dy_dx, target_gradients, self.cosine_weight
                    )
                    for dd in dummy_data_list:
                        loss = loss + tv_coeff * tv_loss(dd)

                    if self.use_safe_loss:
                        loss = torch.nan_to_num(
                            loss, nan=0.0, posinf=1e6, neginf=1e6
                        )
                    loss.backward()

                    if self.use_safe_loss:
                        torch.nn.utils.clip_grad_norm_(
                            dummy_data_list + dummy_label_list, max_norm=1.0
                        )

                    return loss

                try:
                    optimizer_global.step(closure_global)
                except RuntimeError as err:
                    if self.verbose:
                        print(
                            f"[{experiment_name}] Aviso: fase global ignorada "
                            f"no ciclo {cycle}: {err}"
                        )

                if self.use_safe_loss:
                    self._repair_variables(dummy_data_list, dummy_label_list)
                else:
                    with torch.no_grad():
                        for dd in dummy_data_list:
                            dd.data.clamp_(0.0, 1.0)

            # ── Loop round-robin por amostra ────────────────────────
            for target_idx in range(self.batch_size):
                opt = optimizers[target_idx]

                def closure(tidx: int = target_idx, _opt: torch.optim.LBFGS = opt):
                    _opt.zero_grad()
                    if self.use_safe_loss:
                        self._repair_variables(dummy_data_list, dummy_label_list)

                    parts_d, parts_l = [], []
                    for j in range(self.batch_size):
                        if j == tidx:
                            parts_d.append(dummy_data_list[j])
                            parts_l.append(
                                F.softmax(
                                    dummy_label_list[j].clamp(-LOGIT_CLAMP, LOGIT_CLAMP),
                                    dim=-1,
                                )
                            )
                        else:
                            parts_d.append(dummy_data_list[j].detach())
                            parts_l.append(
                                F.softmax(
                                    dummy_label_list[j]
                                    .detach()
                                    .clamp(-LOGIT_CLAMP, LOGIT_CLAMP),
                                    dim=-1,
                                )
                            )

                    batch_data = torch.cat(parts_d, dim=0)
                    batch_label = torch.cat(parts_l, dim=0)

                    pred = attack_model(batch_data)
                    d_loss = criterion(pred, batch_label)
                    d_dy_dx = torch.autograd.grad(
                        d_loss, attack_model.parameters(), create_graph=True
                    )

                    loss = matching_loss_fn(
                        d_dy_dx, target_gradients, self.cosine_weight
                    )
                    loss = loss + tv_coeff * tv_loss(dummy_data_list[tidx])

                    # Label anchoring
                    anchor_prob = anchor_labels[tidx].unsqueeze(0).clamp(1e-8, 1.0)
                    log_pred_prob = F.log_softmax(
                        dummy_label_list[tidx].clamp(-LOGIT_CLAMP, LOGIT_CLAMP),
                        dim=-1,
                    )
                    label_reg = F.kl_div(
                        log_pred_prob, anchor_prob, reduction="sum"
                    )
                    loss = loss + self.label_anchor_weight * label_reg

                    if self.use_safe_loss:
                        loss = torch.nan_to_num(
                            loss, nan=0.0, posinf=1e6, neginf=1e6
                        )
                    loss.backward()

                    if self.use_safe_loss:
                        torch.nn.utils.clip_grad_norm_(
                            [dummy_data_list[tidx], dummy_label_list[tidx]],
                            max_norm=1.0,
                        )

                    _loss_log[0] = float(loss.detach().cpu().item())
                    return loss

                try:
                    opt.step(closure)
                except RuntimeError as err:
                    if self.verbose:
                        print(
                            f"[{experiment_name}] Aviso: passo local {target_idx} "
                            f"ignorado no ciclo {cycle}: {err}"
                        )

                # Clamping pós-otimização
                if self.use_safe_loss:
                    self._repair_variables(dummy_data_list, dummy_label_list)
                else:
                    with torch.no_grad():
                        dummy_data_list[target_idx].data.clamp_(0.0, 1.0)

                # Histórico de visualização
                for j in range(self.batch_size):
                    history[j].append(
                        self._to_pil(
                            dummy_data_list[j].data.clamp(0, 1).squeeze(0).cpu()
                        )
                    )

            # ── Snapshot da reconstrução nos checkpoints ─────────────
            if cycle in snapshot_cycles:
                snap = torch.cat(
                    [
                        torch.nan_to_num(
                            d.data, nan=0.0, posinf=1.0, neginf=0.0
                        ).clamp(0, 1)
                        for d in dummy_data_list
                    ],
                    dim=0,
                )  # [B, C, H, W]
                snapshots.append(snap.cpu().clone())
                snapshot_iters.append(cycle * self.batch_size)

            report("optimizing", experiment_name, cycle + 1, total_cycles, _loss_log[0])

            # Logging
            current_loss = _loss_log[0]
            if cycle % 10 == 0 and current_loss is not None:
                if math.isfinite(current_loss):
                    last_valid_loss = current_loss
                elif last_valid_loss is not None:
                    current_loss = last_valid_loss

                loss_history.append(current_loss or 0.0)

                if self.verbose:
                    print(
                        f"cycle {cycle:4d}/{total_cycles}  |  "
                        f"loss={current_loss:.6f}  |  "
                        f"tv_coeff={tv_coeff:.2e}"
                    )

        # ═══════════════════════════════════════════════════════════════
        # Consolidação dos resultados
        # ═══════════════════════════════════════════════════════════════
        report("finalizing", "Consolidando reconstruções e snapshots")
        reconstructed = torch.cat(
            [
                torch.nan_to_num(d.data, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
                for d in dummy_data_list
            ],
            dim=0,
        )
        dummy_label = torch.cat(
            [
                torch.nan_to_num(
                    l.data, nan=0.0, posinf=LOGIT_CLAMP, neginf=-LOGIT_CLAMP
                ).clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
                for l in dummy_label_list
            ],
            dim=0,
        )

        self.history = history
        self.loss_history = loss_history

        return {
            "experiment_name": experiment_name,
            "reconstructed": reconstructed,
            "dummy_label": dummy_label,
            "history": history,
            "loss_history": loss_history,
            "target_gradients": target_gradients,
            "snapshots": snapshots,
            "snapshot_iters": snapshot_iters,
        }


# ---------------------------------------------------------------------------
# Função de conveniência
# ---------------------------------------------------------------------------

def run_dlg_attack(
    target_gradients: list[torch.Tensor],
    attack_model: nn.Module,
    batch_size: int = 3,
    num_iters: int = 800,
    num_classes: int = 100,
    device: str = "cpu",
    cosine_weight: float = COSINE_WEIGHT,
    tv_coeff_start: float = TV_COEFF_START,
    tv_coeff_end: float = TV_COEFF_END,
    label_anchor_weight: float = LABEL_ANCHOR_WEIGHT,
    global_phase_freq: int = GLOBAL_PHASE_FREQ,
    use_safe_loss: bool = False,
    seed: int | None = None,
    verbose: bool = True,
    experiment_name: str = "DLG Attack",
) -> dict:
    """
    Função de conveniência para executar o ataque DLG em uma linha.

    Args:
        target_gradients: Gradientes interceptados.
        attack_model: Modelo usado pelo atacante.
        batch_size: Tamanho do batch a reconstruir.
        num_iters: Iterações totais.
        num_classes: Número de classes.
        device: 'cuda' ou 'cpu'.
        cosine_weight: Peso cosine na perda composta.
        tv_coeff_start: Coeficiente TV inicial.
        tv_coeff_end: Coeficiente TV final.
        label_anchor_weight: Peso do label anchoring.
        global_phase_freq: Frequência da fase global.
        use_safe_loss: Se True, usa loss numericamente estável.
        seed: Semente aleatória.
        verbose: Controla logs.
        experiment_name: Nome do experimento.

    Returns:
        Dicionário com resultados do ataque (ver DLGAttack.run()).
    """
    attack = DLGAttack(
        batch_size=batch_size,
        num_iters=num_iters,
        num_classes=num_classes,
        device=device,
        cosine_weight=cosine_weight,
        tv_coeff_start=tv_coeff_start,
        tv_coeff_end=tv_coeff_end,
        label_anchor_weight=label_anchor_weight,
        global_phase_freq=global_phase_freq,
        use_safe_loss=use_safe_loss,
        seed=seed,
        verbose=verbose,
    )
    return attack.run(target_gradients, attack_model, experiment_name=experiment_name)
