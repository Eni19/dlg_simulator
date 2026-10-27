# DLG Attack

<!-- impeccable:product-schema 1 -->

## Platform

web

## Product Purpose

Laboratório didático de Deep Leakage from Gradients para explorar a reconstrução de imagens a partir de gradientes. O usuário solicitou uma reformulação visual completa para futura apresentação em seu portfólio, mantendo as funções e partes didáticas e explorando o código técnico existente.

## Users

Visitantes de portfólio e pessoas curiosas sobre IA e privacidade, incluindo leigos e quem não tem conhecimento aprofundado. A apresentação deve explicar o propósito, o funcionamento em termos abstratos e a origem do DLG antes dos detalhes técnicos. O código e as opções avançadas continuam acessíveis para quem quiser se aprofundar.

## Capabilities and Constraints

React 18, TypeScript, Vite, Tailwind 3 e Recharts no frontend; FastAPI e PyTorch no backend. Preservar contratos da API, configuração centralizada e federada, três defesas, parâmetros de loss, distribuição IID/Dirichlet, execução, cancelamento da requisição, imagens, snapshots e métricas. Cancelar a requisição no navegador não garante interromper o processamento no servidor. Progresso vem do backend por stream NDJSON: ciclos DLG e rodadas federadas realmente concluídos, sem estimativa de tempo restante.

## Evidence on Hand

Código real em src/dlg_attack.py, src/losses.py, src/models.py, src/scenarios.py e src/federated_flower.py. Dataset CIFAR-100; LeNet com Sigmoid; LBFGS, inicialização espectral, perda L2/cosseno, TV adaptativa e label anchoring. Resultados devem vir da API; não inventar reconstruções ou métricas.

## Product Principles

A linguagem e o vocabulário da experiência estão documentados em [CONTENT.md](CONTENT.md).

O deploy Vercel apresenta um showcase controlado: CPU, até duas imagens públicas
incluídas no projeto e 10/25/50 iterações (também 100/200 para duas imagens), sem defesas nem treino federado. As
possibilidades completas continuam visíveis como caminhos de aprofundamento via
GitHub, sem alarmes, cadeados decorativos ou linguagem de venda. Os limites devem
ser aplicados no backend. A instalação local preserva o laboratório completo,
com lotes de até 8 imagens e ajustes livres, cujo custo depende do hardware e
da configuração. Explicar essa diferença com linguagem convidativa.

- Preservar a precisão do experimento e distinguir explicações ilustrativas de medições.
- Tornar a teoria acessível sem esconder os parâmetros avançados.
- Relacionar decisões e resultados à implementação real.
- Preservar a stack e a lógica científica.
