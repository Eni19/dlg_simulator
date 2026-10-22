# DLG Attack

<!-- impeccable:product-schema 1 -->

## Platform

web

## Product Purpose

Laboratório didático de Deep Leakage from Gradients para explorar a reconstrução de imagens a partir de gradientes. O usuário solicitou uma reformulação visual completa para futura apresentação em seu portfólio, mantendo as funções e partes didáticas e explorando o código técnico existente.

## Users

Visitantes de portfólio e pessoas interessadas no experimento. A prioridade entre recrutadores, estudantes e especialistas permanece aberta; foi enviada uma pergunta ao usuário. Até a resposta, a implementação deve manter uma entrada compreensível e acesso aos controles técnicos.

## Capabilities and Constraints

React 18, TypeScript, Vite, Tailwind 3 e Recharts no frontend; FastAPI e PyTorch no backend. Preservar contratos da API, configuração centralizada e federada, três defesas, parâmetros de loss, distribuição IID/Dirichlet, execução, cancelamento da requisição, imagens, snapshots e métricas. Cancelar a requisição no navegador não garante interromper o processamento no servidor. Progresso atual é uma estimativa local, não telemetria do backend.

## Evidence on Hand

Código real em src/dlg_attack.py, src/losses.py, src/models.py, src/scenarios.py e src/federated_flower.py. Dataset CIFAR-100; LeNet com Sigmoid; LBFGS, inicialização espectral, perda L2/cosseno, TV adaptativa e label anchoring. Resultados devem vir da API; não inventar reconstruções ou métricas.

## Product Principles

- Preservar a precisão do experimento e distinguir explicações ilustrativas de medições.
- Tornar a teoria acessível sem esconder os parâmetros avançados.
- Relacionar decisões e resultados à implementação real.
- Preservar a stack e a lógica científica.
