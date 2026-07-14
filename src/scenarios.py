"""Cenários configuráveis de ataque DLG em ambiente de Aprendizado Federado.

Este módulo fornece uma infraestrutura completa para definir, executar e
comparar diferentes cenários de ataque *Deep Leakage from Gradients* (DLG)
sobre simulações federadas com Flower.  Inclui:

* :class:`AttackScenario` — dataclass que parametriza um cenário de ataque.
* :class:`ScenarioResult` — dataclass que encapsula o resultado de execução.
* :data:`PREDEFINED_SCENARIOS` — dicionário com cenários pré-configurados
  cobrindo variações de defesa (ruído, sparsification, pruning), tamanho
  de batch e número de rodadas.
* :func:`run_scenario` — executa um cenário individual com cronometragem e
  tratamento de erros.
* :func:`run_scenario_sweep` — executa uma lista de cenários em sequência e
  consolida as métricas em um único :class:`~pandas.DataFrame`.
* :func:`generate_report` — gera um relatório Markdown formatado a partir
  dos resultados do sweep.
* :func:`list_scenarios` — lista os nomes dos cenários pré-definidos.
"""

from __future__ import annotations

import copy
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from .models import create_lenet
from .utils import (
    get_device,
    set_seed,
    evaluate_reconstruction_set,
    summarize_metrics,
    get_image_transform,
    get_raw_cifar100,
    label_to_onehot,
)
from .federated_flower import (
    start_federated_simulation,
    AttackerClient,
)

# ---------------------------------------------------------------------------
# Logger do módulo
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# =========================================================================
# Dataclasses
# =========================================================================


@dataclass
class AttackScenario:
    """Define um cenário de ataque DLG no ambiente federado.

    Cada instância parametriza completamente uma simulação: tipo de defesa,
    hiperparâmetros do DLG, quantidade de clientes e rodadas, entre outros.

    Attributes:
        name: Identificador legível do cenário.
        defense_type: Tipo de defesa aplicada aos gradientes.
            Valores aceitos: ``"none"``, ``"noisy"``, ``"sparse"``,
            ``"pruning"``.
        defense_params: Parâmetros específicos da defesa (ex.:
            ``{"noise_std": 0.01}``).
        num_clients: Número de clientes na simulação federada.
        num_rounds: Número de rodadas de comunicação.
        dlg_batch_size: Tamanho do batch utilizado pelo ataque DLG.
        dlg_num_iters: Número de iterações de otimização do DLG.
        seed: Semente para reprodutibilidade.
        description: Descrição textual do cenário.
    """

    name: str
    defense_type: str = "none"  # "none", "noisy", "sparse", "pruning"
    defense_params: dict = field(default_factory=dict)
    num_clients: int = 2
    num_rounds: int = 5
    dlg_batch_size: int = 1
    dlg_num_iters: int = 800
    seed: int = 50
    description: str = ""


@dataclass
class ScenarioResult:
    """Resultado da execução de um cenário.

    Attributes:
        scenario: O cenário que foi executado.
        metrics_df: DataFrame com métricas de reconstrução (MSE, PSNR,
            SSIM, etc.), ou ``None`` se indisponível.
        dlg_results: Lista de dicionários com os resultados brutos do DLG.
        history: Histórico de treinamento federado por rodada.
        elapsed_seconds: Tempo total de execução em segundos.
        success: Indica se o cenário terminou sem erros.
        error_message: Mensagem de erro, caso ``success`` seja ``False``.
    """

    scenario: AttackScenario
    metrics_df: Optional[pd.DataFrame] = None
    dlg_results: list = field(default_factory=list)
    history: list = field(default_factory=list)
    elapsed_seconds: float = 0.0
    success: bool = False
    error_message: str = ""


# =========================================================================
# Cenários pré-definidos
# =========================================================================

PREDEFINED_SCENARIOS: dict[str, AttackScenario] = {
    # --- Linha de base (sem defesa) ---
    "baseline": AttackScenario(
        name="baseline",
        defense_type="none",
        dlg_batch_size=1,
        dlg_num_iters=800,
        description="DLG puro sem defesa — linha de base",
    ),
    "baseline_batch2": AttackScenario(
        name="baseline_batch2",
        defense_type="none",
        dlg_batch_size=2,
        description="DLG sem defesa com batch=2",
    ),
    "baseline_batch4": AttackScenario(
        name="baseline_batch4",
        defense_type="none",
        dlg_batch_size=4,
        description="DLG sem defesa com batch=4",
    ),
    # --- Ruído Gaussiano ---
    "noisy_low": AttackScenario(
        name="noisy_low",
        defense_type="noisy",
        defense_params={"noise_std": 0.01},
        description="Ruído gaussiano baixo (sigma=0.01)",
    ),
    "noisy_medium": AttackScenario(
        name="noisy_medium",
        defense_type="noisy",
        defense_params={"noise_std": 0.05},
        description="Ruído gaussiano médio (sigma=0.05)",
    ),
    "noisy_high": AttackScenario(
        name="noisy_high",
        defense_type="noisy",
        defense_params={"noise_std": 0.1},
        description="Ruído gaussiano alto (sigma=0.1)",
    ),
    "noisy_extreme": AttackScenario(
        name="noisy_extreme",
        defense_type="noisy",
        defense_params={"noise_std": 0.5},
        description="Ruído gaussiano extremo (sigma=0.5)",
    ),
    # --- Sparsification ---
    "sparse_50": AttackScenario(
        name="sparse_50",
        defense_type="sparse",
        defense_params={"keep_ratio": 0.5},
        description="Sparsification mantendo 50%",
    ),
    "sparse_20": AttackScenario(
        name="sparse_20",
        defense_type="sparse",
        defense_params={"keep_ratio": 0.2},
        description="Sparsification mantendo 20%",
    ),
    "sparse_10": AttackScenario(
        name="sparse_10",
        defense_type="sparse",
        defense_params={"keep_ratio": 0.1},
        description="Sparsification mantendo 10%",
    ),
    "sparse_01": AttackScenario(
        name="sparse_01",
        defense_type="sparse",
        defense_params={"keep_ratio": 0.01},
        description="Sparsification mantendo 1%",
    ),
    # --- Model Pruning ---
    "pruning_30": AttackScenario(
        name="pruning_30",
        defense_type="pruning",
        defense_params={"amount": 0.3},
        description="Model Pruning 30%",
    ),
    "pruning_50": AttackScenario(
        name="pruning_50",
        defense_type="pruning",
        defense_params={"amount": 0.5},
        description="Model Pruning 50%",
    ),
    "pruning_70": AttackScenario(
        name="pruning_70",
        defense_type="pruning",
        defense_params={"amount": 0.7},
        description="Model Pruning 70%",
    ),
    "pruning_90": AttackScenario(
        name="pruning_90",
        defense_type="pruning",
        defense_params={"amount": 0.9},
        description="Model Pruning 90%",
    ),
    # --- Variação de rodadas ---
    "multiround_r1": AttackScenario(
        name="multiround_r1",
        defense_type="none",
        num_rounds=1,
        description="Ataque na rodada 1",
    ),
    "multiround_r3": AttackScenario(
        name="multiround_r3",
        defense_type="none",
        num_rounds=3,
        description="Ataque na rodada 3",
    ),
    "multiround_r10": AttackScenario(
        name="multiround_r10",
        defense_type="none",
        num_rounds=10,
        description="Ataque na rodada 10",
    ),
}


# =========================================================================
# Execução de cenários
# =========================================================================


def run_scenario(
    scenario: AttackScenario,
    device: str | None = None,
    verbose: bool = True,
) -> ScenarioResult:
    """Executa um cenário de ataque DLG completo.

    Realiza a simulação federada via :func:`start_federated_simulation`,
    cronometra a execução, extrai métricas de reconstrução e encapsula
    tudo em um :class:`ScenarioResult`.

    Args:
        scenario: Instância de :class:`AttackScenario` com os parâmetros
            desejados.
        device: Dispositivo PyTorch (ex.: ``"cpu"``, ``"cuda"``).  Se
            ``None``, é detectado automaticamente via :func:`get_device`.
        verbose: Se ``True``, imprime informações de progresso no console.

    Returns:
        :class:`ScenarioResult` contendo métricas, resultados brutos do
        DLG, histórico de treinamento e metadados de execução.
    """
    if device is None:
        device = str(get_device(verbose=verbose))

    set_seed(scenario.seed)

    if verbose:
        print(f"\n{'='*70}")
        print(f" Cenário: {scenario.name}")
        print(f" {scenario.description}")
        print(f" Defesa: {scenario.defense_type} | Params: {scenario.defense_params}")
        print(f" Clientes: {scenario.num_clients} | Rodadas: {scenario.num_rounds}")
        print(f" DLG batch: {scenario.dlg_batch_size} | Iterações: {scenario.dlg_num_iters}")
        print(f"{'='*70}")

    start_time = time.time()

    try:
        sim_results = start_federated_simulation(
            num_clients=scenario.num_clients,
            num_rounds=scenario.num_rounds,
            defense_type=scenario.defense_type,
            defense_params=scenario.defense_params,
            pruning_amount=(
                scenario.defense_params.get("amount", 0.0)
                if scenario.defense_type == "pruning"
                else 0.0
            ),
            device=device,
            seed=scenario.seed,
            attacker_enabled=True,
            verbose=verbose,
            dlg_batch_size=scenario.dlg_batch_size,
            dlg_num_iters=scenario.dlg_num_iters,
        )

        elapsed = time.time() - start_time

        # --- Extração de métricas ---
        metrics_df = None
        dlg_results = sim_results.get("dlg_results", [])
        clients = sim_results.get("clients", [])

        if dlg_results:
            # Localiza o cliente atacante
            attacker = None
            for c in clients:
                if isinstance(c, AttackerClient):
                    attacker = c
                    break

            if attacker is not None and attacker.intercepted_data is not None:
                for dlg_res in dlg_results:
                    gt_data = attacker.intercepted_data[: scenario.dlg_batch_size]
                    gt_labels = (
                        attacker.intercepted_labels[: scenario.dlg_batch_size]
                        if attacker.intercepted_labels is not None
                        else None
                    )

                    try:
                        metrics_df = evaluate_reconstruction_set(
                            reconstructed=dlg_res["reconstructed"].to(gt_data.device),
                            ground_truth=gt_data,
                            labels=gt_labels,
                            pred_logits=dlg_res.get("dummy_label"),
                            name=scenario.name,
                        )
                    except Exception as e:
                        logger.warning(f"Erro ao calcular métricas: {e}")

        return ScenarioResult(
            scenario=scenario,
            metrics_df=metrics_df,
            dlg_results=dlg_results,
            history=sim_results.get("history", []),
            elapsed_seconds=elapsed,
            success=True,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"Cenário '{scenario.name}' falhou: {e}")
        return ScenarioResult(
            scenario=scenario,
            elapsed_seconds=elapsed,
            success=False,
            error_message=str(e),
        )


def run_scenario_sweep(
    scenarios: list[AttackScenario | str],
    device: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Executa uma sequência de cenários e consolida os resultados.

    Cada elemento de *scenarios* pode ser uma instância de
    :class:`AttackScenario` ou uma string correspondente a uma chave em
    :data:`PREDEFINED_SCENARIOS`.  As métricas de todos os cenários são
    concatenadas em um único :class:`~pandas.DataFrame`.

    Args:
        scenarios: Lista de cenários ou nomes de cenários pré-definidos.
        device: Dispositivo PyTorch.  Detectado automaticamente se
            ``None``.
        verbose: Se ``True``, imprime progresso e resumo final.

    Returns:
        :class:`~pandas.DataFrame` consolidado com colunas adicionais
        ``cenario``, ``defesa``, ``defesa_params``, ``tempo_s`` e
        ``sucesso``.

    Raises:
        ValueError: Se um nome de cenário não for encontrado em
            :data:`PREDEFINED_SCENARIOS`.
    """
    if device is None:
        device = str(get_device(verbose=verbose))

    # Resolução de nomes para instâncias
    resolved: list[AttackScenario] = []
    for s in scenarios:
        if isinstance(s, str):
            if s not in PREDEFINED_SCENARIOS:
                raise ValueError(
                    f"Cenário '{s}' não encontrado. "
                    f"Disponíveis: {list(PREDEFINED_SCENARIOS.keys())}"
                )
            resolved.append(PREDEFINED_SCENARIOS[s])
        else:
            resolved.append(s)

    all_rows: list[pd.DataFrame] = []
    results: list[ScenarioResult] = []

    for i, scenario in enumerate(resolved, 1):
        if verbose:
            print(f"\n{'#'*70}")
            print(f" SWEEP [{i}/{len(resolved)}]: {scenario.name}")
            print(f"{'#'*70}")

        result = run_scenario(scenario, device=device, verbose=verbose)
        results.append(result)

        if result.success and result.metrics_df is not None:
            df = result.metrics_df.copy()
            df["cenario"] = scenario.name
            df["defesa"] = scenario.defense_type
            df["defesa_params"] = str(scenario.defense_params)
            df["tempo_s"] = round(result.elapsed_seconds, 1)
            df["sucesso"] = True
            all_rows.append(df)
        else:
            all_rows.append(
                pd.DataFrame(
                    [
                        {
                            "cenario": scenario.name,
                            "defesa": scenario.defense_type,
                            "defesa_params": str(scenario.defense_params),
                            "tempo_s": round(result.elapsed_seconds, 1),
                            "sucesso": result.success,
                            "erro": result.error_message,
                            "mse": float("nan"),
                            "psnr": float("nan"),
                            "ssim": float("nan"),
                        }
                    ]
                )
            )

    consolidated = pd.concat(all_rows, ignore_index=True)

    if verbose:
        print(f"\n{'='*70}")
        print(f" RESULTADOS DO SWEEP")
        print(f"{'='*70}")
        summary_cols = [
            "cenario",
            "defesa",
            "mse",
            "psnr",
            "ssim",
            "tempo_s",
            "sucesso",
        ]
        available = [c for c in summary_cols if c in consolidated.columns]
        print(consolidated[available].to_string(index=False))

    return consolidated


# =========================================================================
# Relatório
# =========================================================================


def generate_report(
    results_df: pd.DataFrame,
    title: str = "Relatório de Cenários DLG",
) -> str:
    """Gera um relatório Markdown formatado a partir dos resultados do sweep.

    Agrupa os resultados por cenário, calcula médias das métricas e monta
    uma tabela Markdown legível.

    Args:
        results_df: DataFrame consolidado retornado por
            :func:`run_scenario_sweep`.
        title: Título do relatório.

    Returns:
        String contendo o relatório completo em formato Markdown.
    """
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Resumo por Cenário")
    lines.append("")
    lines.append(
        "| Cenário | Defesa | MSE (↓) | PSNR (↑) | SSIM (↑) | Tempo (s) |"
    )
    lines.append(
        "|---------|--------|---------|----------|----------|-----------|"
    )

    for name in results_df["cenario"].unique():
        subset = results_df[results_df["cenario"] == name]
        defense = (
            subset["defesa"].iloc[0] if "defesa" in subset.columns else "?"
        )
        mse_val = (
            subset["mse"].mean() if "mse" in subset.columns else float("nan")
        )
        psnr_val = (
            subset["psnr"].mean()
            if "psnr" in subset.columns
            else float("nan")
        )
        ssim_val = (
            subset["ssim"].mean()
            if "ssim" in subset.columns
            else float("nan")
        )
        tempo = (
            subset["tempo_s"].iloc[0] if "tempo_s" in subset.columns else 0
        )

        lines.append(
            f"| {name} | {defense} "
            f"| {mse_val:.6f} | {psnr_val:.1f} "
            f"| {ssim_val:.4f} | {tempo} |"
        )

    lines.append("")
    report = "\n".join(lines)
    print(report)
    return report


# =========================================================================
# Utilitário de listagem
# =========================================================================


def list_scenarios() -> list[str]:
    """Retorna a lista ordenada dos nomes de cenários pré-definidos.

    Returns:
        Lista de strings com as chaves de :data:`PREDEFINED_SCENARIOS`
        em ordem alfabética.
    """
    return sorted(PREDEFINED_SCENARIOS.keys())
