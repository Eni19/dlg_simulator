"""
Deep Leakage from Gradients (DLG) — Ataque de Reconstrução de Gradientes.

Pacote modularizado para:
- Ataque DLG com perda composta (L2 + Cosine Similarity)
- Regularização TV adaptativa e Label Anchoring
- Defesas: Model Pruning, Noisy Gradients, Gradient Sparsification
- Simulação de Aprendizado Federado com Flower (flwr)
- API web para exposição das funcionalidades

Autor: Enrique Susin
"""

__version__ = "1.0.0"
__author__ = "Enrique Susin"
