# NVIDIA RTX no Windows

O backend seleciona CUDA automaticamente quando disponível. Modelos, imagens e
gradientes do ataque e da simulação federada já usam o dispositivo selecionado.
O frontend continua no navegador; o processamento acontece no backend.

Em **Experimento → Processamento → Dispositivo**, escolha CPU, GPU NVIDIA
(CUDA) ou Automático. A seleção vale para cada requisição, inclusive defesas e
simulação federada, e tem prioridade sobre `DLG_DEVICE` do servidor. CPU é o
padrão inicial da interface. GPU indisponível retorna erro 422 sem trocar
silenciosamente para CPU; Automático prefere CUDA quando disponível.

Use a venv do projeto, com o servidor parado durante a instalação:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-full.txt
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Instalador oficial: https://pytorch.org/get-started/locally/
Esses comandos usam o runtime CUDA dos pacotes PyTorch; não é necessário compilar
o projeto com o CUDA Toolkit. Mantenha o driver NVIDIA compatível atualizado.

Inicie o backend normalmente ou todo o projeto com `vercel dev --local`.
`/api/health` deve informar `device: "cuda"` e `cuda_available: true`.

Para experimentos locais longos, prefira dois terminais:

```powershell
# Terminal 1
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
# Terminal 2
npm.cmd --prefix frontend run dev -- --host 127.0.0.1
```

Abra http://127.0.0.1:5173. O proxy Vite não impõe timeout de resposta.
O gateway permite esperar até 30 minutos pelo backend; o roteador de
`vercel dev` e a hospedagem ainda podem aplicar limites próprios. Esse ajuste
não aumenta `maxDuration` das funções em produção (atualmente 300 s).

Ataques simples e com defesas usam uma cópia local completa quando disponível;
caso contrário, buscam apenas as imagens escolhidas via Hugging Face Viewer.
O cache `hf-viewer-cifar100-v1` dentro de `DLG_DATA_DIR` (ou `~/.torch`) permite
repetir essas mesmas imagens sem internet. Índices novos precisam de conexão.
`DLG_DATA_SOURCE=local` exige o original local; `huggingface` força o modo online;
`auto` é o padrão. Cada consulta HTTP tem timeout de 15 segundos, sem fallback
automático para download do arquivo completo quando a fonte online falha.

Os assets do Viewer podem ser JPEG recomprimidos: não são garantidamente
idênticos pixel a pixel ao CIFAR original. Use `local` para comparações científicas
com resultados anteriores. Acurácia de classificação fica ausente quando não
há conjunto de teste original completo; métricas de reconstrução continuam
calculadas em relação às imagens efetivamente usadas no ataque.

Referência: https://huggingface.co/docs/dataset-viewer/rows

A simulação federada continua precisando do dataset para treinar os clientes.
Ao preparar o original completo, o torchvision baixa aproximadamente 169 MB.
Uma conexão lenta com o servidor do dataset pode consumir mais tempo que o
próprio ataque. Para preparar o cache antes de abrir o site:

```powershell
.\.venv\Scripts\python.exe -c "from src.utils import get_raw_cifar100; get_raw_cifar100(); print('CIFAR-100 pronto')"
```

Evite iniciar vários experimentos enquanto esse primeiro download está em curso.

Para exigir GPU, ou comparar com CPU, configure antes de iniciar o servidor:

```powershell
$env:DLG_DEVICE = "cuda"  # Falha com mensagem clara se CUDA estiver indisponível.
# $env:DLG_DEVICE = "cpu"
# $env:DLG_DEVICE = "auto"  # Padrão: CUDA quando disponível, senão CPU.
```

Benchmark do próprio otimizador DLG, com dados sintéticos e aquecimento:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_devices.py --iterations 10 --batch-size 3
```

O tempo não inclui download do CIFAR-100 ou requisições HTTP. Resultados variam
com batch, iterações, aquecimento e outras aplicações usando a GPU. LeNet é pequena;
GPU nem sempre vence CPU. Mantemos float32 e o algoritmo original, incluindo
derivadas de segunda ordem, sem introduzir precisão mista.

Validação local em 23/10/2026: RTX 3050 Laptop 6 GB, PyTorch 2.14.1+cu130,
batch 3 e 10 iterações após aquecimento: CPU 1,71 s; CUDA 3,02 s.
Esse teste confirma execução correta com saída finita na GPU, mas não indica
ganho de velocidade para essa carga. Compare suas configurações reais antes
de preferir CUDA; `DLG_DEVICE=cpu` permite manter CPU mesmo com CUDA instalado.

Nova comparação em 27/10/2026, 1 imagem e 50 iterações: CPU 10,19 s;
CUDA 12,43 s. O padrão inicial da interface agora é 1 imagem e 50 iterações
para testar o fluxo. Reconstruções mais refinadas podem exigir mais iterações.
Para este modelo pequeno, recomendamos iniciar o backend com `DLG_DEVICE=cpu`;
CUDA continua disponível por seleção explícita ou pelo modo automático.

Inicialização comparável entre dispositivos: pesos e ruídos iniciais são gerados
na CPU e transferidos para o dispositivo escolhido. O cálculo do ataque continua
na GPU quando selecionada. A API desativa TF32 para preservar float32 no matching
de gradientes de segunda ordem. Mesma seed não garante resultados bit a bit iguais
entre CPU e CUDA, pois os kernels e as trajetórias numéricas do LBFGS diferem.

Validação pela API com imagem 40055, seed 50, batch 1 e 50 iterações após o ajuste:
CUDA concluiu em 23,33 s (SSIM 0,99884); CPU em 9,75 s (SSIM 0,99854).
São resultados desta imagem/configuração, não uma garantia para todos os ataques.
Timeouts transitórios ao buscar imagens no Hugging Face agora têm uma repetição
limitada. Se persistirem, a API retorna 503 e identifica falha nos dados antes do
cálculo, sem atribuí-la ao CUDA.

A RTX local não fica disponível para funções da Vercel. O deploy Linux continua
com os pacotes CPU definidos em requirements.txt; CUDA é para este backend local
ou um host que ofereça GPU NVIDIA.
