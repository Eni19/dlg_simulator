#!/usr/bin/env python3
"""
Script de demonstração do pipeline DLG modularizado.

Uso:
    python run_demo.py [--quick] [--defense TYPE] [--federated]

Exemplos:
    python run_demo.py                          # Ataque DLG completo
    python run_demo.py --quick                  # Versão rápida (100 iterações)
    python run_demo.py --defense pruning        # Ataque com Model Pruning
    python run_demo.py --defense noisy          # Ataque com Noisy Gradients
    python run_demo.py --defense sparse          # Ataque com Sparsification
    python run_demo.py --federated              # Simulação Federada com Flower

Autor: Enrique Susin
"""

import argparse
import sys
import math
import numpy as np
import torch
import torch.nn.functional as F
from random import randint

from src.models import create_lenet
from src.utils import (
    get_device,
    set_seed,
    get_image_transform,
    get_raw_cifar100,
    label_to_onehot,
    cross_entropy_for_onehot,
    evaluate_reconstruction_set,
    summarize_metrics,
    psnr,
    ssim_approx,
)
from src.dlg_attack import DLGAttack
from src.losses import (
    TV_COEFF_START,
    TV_COEFF_END,
    COSINE_WEIGHT,
    LABEL_ANCHOR_WEIGHT,
)


def run_demo_attack(args):
    """Executa uma demonstração do ataque DLG standalone."""
    device = get_device()
    set_seed(args.seed)

    print(f"\n{'='*70}")
    print(f" DEMO: Ataque DLG — Reconstrução de Imagens CIFAR-100")
    print(f" Batch: {args.batch_size} | Iterações: {args.num_iters}")
    print(f" Dispositivo: {device}")
    print(f"{'='*70}")

    # Dataset
    print("\n[1/5] Carregando CIFAR-100...")
    dataset = get_raw_cifar100(download=True)
    tp = get_image_transform()

    # Selecionar imagens aleatórias
    print(f"[2/5] Selecionando {args.batch_size} imagens aleatórias...")
    gt_images, gt_labels = [], []
    for _ in range(args.batch_size):
        idx = randint(0, len(dataset) - 1)
        gt_images.append(tp(dataset[idx][0]))
        gt_labels.append(dataset[idx][1])

    gt_data = torch.stack(gt_images).to(device)
    gt_label = torch.tensor(gt_labels, dtype=torch.long, device=device)
    gt_onehot = label_to_onehot(gt_label, num_classes=100)

    print(f"   Labels reais: {sorted(gt_label.tolist())}")

    # Modelo base
    print("[3/5] Criando modelo LeNet e calculando gradientes...")
    model = create_lenet(num_classes=100, device=device, seed=args.seed)
    model.eval()

    out = model(gt_data)
    loss_val = cross_entropy_for_onehot(out, gt_onehot)
    dy_dx = torch.autograd.grad(loss_val, model.parameters())
    original_dy_dx = [g.detach().clone() for g in dy_dx]

    print(f"   Parâmetros do modelo: {len(original_dy_dx)}")
    print(f"   Total elementos nos gradientes: {sum(g.numel() for g in original_dy_dx):,}")

    # Ataque DLG
    print(f"[4/5] Executando ataque DLG ({args.num_iters} iterações)...")
    attack_model = create_lenet(num_classes=100, device=device, seed=args.seed)
    attack_model.eval()

    attack = DLGAttack(
        batch_size=args.batch_size,
        num_iters=args.num_iters,
        num_classes=100,
        device=device,
        cosine_weight=args.cosine_weight,
        tv_coeff_start=args.tv_coeff_start,
        tv_coeff_end=args.tv_coeff_end,
        label_anchor_weight=args.label_anchor_weight,
        seed=args.seed,
        verbose=not args.quiet,
    )

    result = attack.run(
        target_gradients=original_dy_dx,
        attack_model=attack_model,
        experiment_name="DLG Demo",
    )

    # Avaliação
    print("[5/5] Avaliando reconstruções...")
    metrics_df = evaluate_reconstruction_set(
        reconstructed=result["reconstructed"],
        ground_truth=gt_data,
        labels=gt_label,
        pred_logits=result["dummy_label"],
        name="DLG Demo",
    )

    print(f"\n{'='*70}")
    print(f" RESULTADOS FINAIS")
    print(f"{'='*70}")

    for _, row in metrics_df.iterrows():
        print(
            f"  Rec {int(row['rec_idx'])} → GT {int(row['gt_idx'])}  |  "
            f"Label real: {int(row['label_real'])}  |  "
            f"Label pred: {int(row['label_pred'])}  |  "
            f"PSNR: {row['psnr']:.1f} dB  |  "
            f"SSIM: {row['ssim']:.4f}"
        )

    summary = summarize_metrics(metrics_df)
    print(f"\n  SSIM médio: {metrics_df['ssim'].mean():.4f} ± {metrics_df['ssim'].std():.4f}")
    print(f"  PSNR médio: {metrics_df['psnr'].mean():.1f} ± {metrics_df['psnr'].std():.1f} dB")
    print(f"  MSE médio:  {metrics_df['mse'].mean():.6f} ± {metrics_df['mse'].std():.6f}")
    print(f"  Loss final: {result['loss_history'][-1]:.6f}" if result["loss_history"] else "")

    return result, metrics_df


def run_demo_defense(args):
    """Executa ataque DLG com defesa."""
    from src.federated_flower import intercept_and_attack, apply_global_magnitude_pruning_

    device = get_device()
    set_seed(args.seed)

    print(f"\n{'='*70}")
    print(f" DEMO: Ataque DLG com Defesa — {args.defense.upper()}")
    print(f" Batch: {args.batch_size} | Iterações: {args.num_iters}")
    print(f" Dispositivo: {device}")
    print(f"{'='*70}")

    # Dados
    dataset = get_raw_cifar100(download=True)
    tp = get_image_transform()
    gt_images, gt_labels_list = [], []
    for _ in range(args.batch_size):
        idx = randint(0, len(dataset) - 1)
        gt_images.append(tp(dataset[idx][0]))
        gt_labels_list.append(dataset[idx][1])

    gt_data = torch.stack(gt_images).to(device)
    gt_label = torch.tensor(gt_labels_list, dtype=torch.long, device=device)

    # Modelo
    model = create_lenet(num_classes=100, device=device, seed=args.seed)

    # Configurar parâmetros da defesa
    defense_params = {}
    if args.defense == "noisy":
        defense_params["noise_std"] = args.noise_std
    elif args.defense == "sparse":
        defense_params["keep_ratio"] = args.keep_ratio
    elif args.defense == "pruning":
        defense_params["amount"] = args.pruning_amount

    # Executar interceptação + ataque
    result = intercept_and_attack(
        model=model,
        data=gt_data,
        labels=gt_label,
        device=device,
        defense_type=args.defense,
        defense_params=defense_params,
        dlg_iters=args.num_iters,
        dlg_seed=args.seed,
        verbose=not args.quiet,
    )

    dlg = result["dlg_result"]
    metrics_df = evaluate_reconstruction_set(
        reconstructed=dlg["reconstructed"],
        ground_truth=gt_data,
        labels=gt_label,
        pred_logits=dlg["dummy_label"],
        name=f"DLG + {args.defense}",
    )

    print(f"\n{'='*70}")
    print(f" RESULTADOS — Defesa: {args.defense}")
    print(f"{'='*70}")
    print(f"  SSIM médio: {metrics_df['ssim'].mean():.4f}")
    print(f"  PSNR médio: {metrics_df['psnr'].mean():.1f} dB")
    print(f"  MSE médio:  {metrics_df['mse'].mean():.6f}")

    return result, metrics_df


def run_demo_federated(args):
    """Executa simulação federada com Flower."""
    from src.federated_flower import start_federated_simulation, FLOWER_AVAILABLE

    if not FLOWER_AVAILABLE:
        print("\n[ERRO] Flower (flwr) não está instalado.")
        print("Instale com: pip install flwr")
        sys.exit(1)

    device = get_device()

    defense_params = {}
    if args.defense == "noisy":
        defense_params["noise_std"] = args.noise_std
    elif args.defense == "sparse":
        defense_params["keep_ratio"] = args.keep_ratio

    results = start_federated_simulation(
        num_clients=args.fl_clients,
        num_rounds=args.fl_rounds,
        defense_type=args.defense if args.defense != "pruning" else "pruning",
        defense_params=defense_params,
        pruning_amount=args.pruning_amount,
        device=device,
        seed=args.seed,
        attacker_enabled=True,
        verbose=not args.quiet,
        dlg_batch_size=args.batch_size,
        dlg_num_iters=args.num_iters,
    )

    print(f"\n{'='*70}")
    print(f" RESULTADOS — Simulação Federada")
    print(f"{'='*70}")
    print(f"  Clientes: {args.fl_clients}")
    print(f"  Rodadas: {args.fl_rounds}")
    print(f"  Defesa: {args.defense}")
    print(f"  Ataques DLG realizados: {len(results.get('dlg_results', []))}")

    return results


def run_demo_scenario(args):
    """Executa um cenário específico pré-definido."""
    from src.scenarios import run_scenario, PREDEFINED_SCENARIOS
    
    if args.scenario not in PREDEFINED_SCENARIOS:
        print(f"\n[ERRO] Cenário '{args.scenario}' não encontrado.")
        print(f"Cenários disponíveis: {', '.join(PREDEFINED_SCENARIOS.keys())}")
        sys.exit(1)
        
    scenario = PREDEFINED_SCENARIOS[args.scenario]
    
    # Sobrescrever iterações/batch se --quick for usado
    if args.quick:
        scenario.dlg_num_iters = 100
        if scenario.num_rounds > 2:
            scenario.num_rounds = 2
            
    device = get_device()
    result = run_scenario(scenario, device=device, verbose=not args.quiet)
    
    if result.success and result.metrics_df is not None:
        print(f"\n{'='*70}")
        print(f" RESULTADOS DO CENÁRIO: {args.scenario}")
        print(f"{'='*70}")
        df = result.metrics_df
        print(f"  SSIM médio: {df['ssim'].mean():.4f}")
        print(f"  PSNR médio: {df['psnr'].mean():.1f} dB")
        print(f"  MSE médio:  {df['mse'].mean():.6f}")
        
    return result


def run_demo_sweep(args):
    """Executa uma bateria (sweep) de vários cenários e gera relatório."""
    from src.scenarios import run_scenario_sweep, generate_report, PREDEFINED_SCENARIOS
    
    device = get_device()
    
    # Se args.sweep for "all", roda tudo, senão roda a lista especificada
    scenarios_to_run = args.sweep
    if "all" in scenarios_to_run:
        scenarios_to_run = list(PREDEFINED_SCENARIOS.keys())
        
    print(f"\nIniciando Sweep de {len(scenarios_to_run)} cenários...")
    df = run_scenario_sweep(scenarios_to_run, device=device, verbose=not args.quiet)
    
    generate_report(df, title=f"Relatório do Sweep ({len(scenarios_to_run)} cenários)")
    return df



# =========================================================================
# Ponto de Entrada
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DLG Attack — Pipeline Modularizado (Demo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python run_demo.py                           # Ataque DLG completo
  python run_demo.py --quick                   # Versão rápida
  python run_demo.py --defense pruning         # DLG com Model Pruning
  python run_demo.py --defense noisy           # DLG com Noisy Gradients
  python run_demo.py --federated               # Simulação Federada
        """,
    )

    # Modo de execução
    parser.add_argument("--federated", action="store_true", help="Executar simulação federada (custom parameters)")
    parser.add_argument("--scenario", type=str, help="Executar um cenário predefinido pelo nome (ex: baseline)")
    parser.add_argument("--sweep", nargs="+", help="Executar uma lista de cenários ou 'all' para todos")
    parser.add_argument("--quick", action="store_true", help="Modo rápido (menos iterações)")
    parser.add_argument("--quiet", action="store_true", help="Suprimir logs detalhados")

    # Parâmetros do ataque
    parser.add_argument("--batch-size", type=int, default=3, help="Tamanho do batch (1-8)")
    parser.add_argument("--num-iters", type=int, default=800, help="Iterações do ataque")
    parser.add_argument("--seed", type=int, default=50, help="Semente aleatória")

    # Hiperparâmetros da perda
    parser.add_argument("--cosine-weight", type=float, default=0.5, help="Peso cosine na loss")
    parser.add_argument("--tv-coeff-start", type=float, default=1e-5, help="TV inicial")
    parser.add_argument("--tv-coeff-end", type=float, default=1e-7, help="TV final")
    parser.add_argument("--label-anchor-weight", type=float, default=0.1, help="Peso label anchoring")

    # Defesa
    parser.add_argument("--defense", type=str, default="none",
                        choices=["none", "noisy", "sparse", "pruning"],
                        help="Tipo de defesa")
    parser.add_argument("--noise-std", type=float, default=0.1, help="Std do ruído")
    parser.add_argument("--keep-ratio", type=float, default=0.2, help="Ratio de gradientes mantidos")
    parser.add_argument("--pruning-amount", type=float, default=0.5, help="Proporção de poda")

    # Federated Learning
    parser.add_argument("--fl-clients", type=int, default=2, help="Número de clientes federados")
    parser.add_argument("--fl-rounds", type=int, default=5, help="Rodadas de agregação")

    args = parser.parse_args()

    # Modo rápido
    if args.quick:
        args.num_iters = 100
        args.fl_rounds = 2
        print("[MODO RÁPIDO] Iterações reduzidas para demonstração.")

    try:
        if args.sweep:
            run_demo_sweep(args)
        elif args.scenario:
            run_demo_scenario(args)
        elif args.federated:
            run_demo_federated(args)
        elif args.defense != "none":
            run_demo_defense(args)
        else:
            run_demo_attack(args)
    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
