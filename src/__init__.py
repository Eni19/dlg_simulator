"""
Deep Leakage from Gradients (DLG) — Ataque de Reconstrução de Gradientes.

Pacote modularizado para:
- Ataque DLG com perda composta (L2 + Cosine Similarity)
- Regularização TV adaptativa e Label Anchoring
- Defesas: Model Pruning, Noisy Gradients, Gradient Sparsification
- Simulação de Aprendizado Federado com Flower (flwr)
- Cenários pre-configurados de ataque DLG
- API web para exposição das funcionalidades

Autor: Enrique Susin
"""

def __getattr__(name):
    # Keep optional federated/scenario dependencies out of the online demo.
    if name in {"PREDEFINED_SCENARIOS", "run_scenario_sweep", "generate_report"}:
        from . import scenarios
        return getattr(scenarios, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "1.0.0"
__author__ = "Enrique Susin"
