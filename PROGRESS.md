# Acompanhamento real do experimento

A interface usa `POST /api/experiments/{attack|defense|federated}/stream` com o
mesmo payload dos endpoints anteriores. A resposta é NDJSON: eventos `progress`,
seguidos por um único `result` ou `error`. Linhas vazias são heartbeats, não
progresso. Os endpoints anteriores continuam disponíveis para clientes REST.

Cada ciclo DLG concluído emite `current`, `total` e a loss disponível. Um ciclo
contém passos LBFGS de todas as imagens; seu custo varia, portanto 50% dos ciclos
não significa metade do tempo. A contagem corresponde ao loop real
`num_iters // batch_size`, sem fingir executar iterações descartadas pelo algoritmo.

No treino federado, a contagem é de rodadas agregadas. Depois disso, cada ataque
tem seu próprio contador de ciclos. Preparação e avaliação exibem a etapa atual,
sem inventar uma porcentagem. Só o evento `result` confirma a conclusão completa.

O backend executa o cálculo fora do event loop HTTP, transmitindo atualizações
durante a computação. ContextVar isola a telemetria de cada requisição. Um lock
serializa experimentos iniciados pelo stream, pois o gerador aleatório do PyTorch
é global. A conexão única dispensa armazenamento de jobs e afinidade entre
instâncias. Não há recuperação automática do resultado após perder a conexão.

Cancelar a espera fecha o stream, mas não interrompe uma operação científica já
em andamento. O backend para de enfileirar eventos para a conexão encerrada.
Os heartbeats não aumentam o limite de duração de funções na Vercel.

Validação local: os ciclos 0–10 chegaram incrementalmente pela porta 5173 em CPU
e CUDA antes do resultado final. Uma execução real no navegador mostrou progresso
intermediário, terminou com as reconstruções e não teve overflow em 390 px.
Testes cobrem ordem dos eventos, erros, término prematuro, JSON/UTF-8 fragmentado
e sanitização de métricas não finitas.
