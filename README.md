# DLG / lab

Um laboratório didático sobre reconstrução de imagens a partir de gradientes.
Conheça a ideia, acompanhe uma tentativa real e compare o original com o que foi
recuperado. Inspirado em *Deep Leakage from Gradients*, de Ligeng Zhu,
Zhijian Liu e Song Han (NeurIPS 2019).

## Duas formas de explorar

| Demonstração online | Versão completa no seu computador |
| --- | --- |
| Três imagens públicas incluídas no projeto | CIFAR-100 local ou imagens sob demanda |
| Uma ou duas imagens por lote, CPU | Lotes de até 8 imagens; CPU, GPU NVIDIA ou seleção automática |
| 10, 25 ou 50 iterações; também 100 e 200 para duas imagens | Grupos maiores e buscas mais longas |
| Reconstrução centralizada sem defesa | Defesas, ajustes avançados e simulação federada |
| Progresso real, imagens e métricas | Progresso real, imagens, métricas e código |

A demonstração faz os cálculos de verdade. Seus limites são aplicados também no
servidor. As imagens de exemplo estão em `src/data/showcase_samples.json`, com
categoria, origem e hash de cada PNG. Elas vêm do conjunto público CIFAR-100.

A versão completa permite configurar lotes de até 8 imagens, iterações, defesas,
pesos do ataque e cenários federados. Esses experimentos exigem mais processamento
e memória; o tempo depende do hardware e dos parâmetros escolhidos. Comece com
um lote pequeno e aumente aos poucos. A demonstração online é o perfil showcase.

## Executar localmente

Instale **Python 3.12**, **Node.js 24 (com npm)** e Git para clonar o projeto.
Essas são as versões usadas neste projeto. Uma GPU não é obrigatória.
No Windows/PowerShell:

```powershell
git clone https://github.com/Eni19/dlg_simulator.git dlg-lab
cd dlg-lab
.\start.cmd --setup
.\start.cmd
```

O primeiro comando instala as dependências em `.venv` e no frontend; execute-o
novamente quando os arquivos de dependências mudarem. O segundo inicia a API e a
interface **no mesmo terminal**, com o perfil completo. Ctrl+C encerra ambos.
O inicializador funciona independentemente da política de scripts do PowerShell.

Para verificar a instalação sem iniciar servidores:

```powershell
.\start.cmd --check
```

Se as portas padrão estiverem ocupadas:

```powershell
.\start.cmd --api-port 8010 --port 5180
```

Abra http://127.0.0.1:5173. A versão local oferece os cenários e os controles
completos. Para usar NVIDIA/CUDA, siga [GPU.md](GPU.md) e escolha o dispositivo
na tela Experimento. A GPU nem sempre é mais rápida para esta rede pequena.

Em Linux/macOS, use `python3.12 scripts/local.py --setup` e depois
`python3.12 scripts/local.py`. As mesmas opções estão disponíveis.

### O que será instalado

| Requisito | Papel no laboratório |
| --- | --- |
| PyTorch e torchvision | Rede neural, gradientes e reconstrução; CPU funciona sem NVIDIA |
| FastAPI, Uvicorn e Pydantic | API local e validação dos experimentos |
| NumPy, SciPy, pandas e Pillow | Cálculos, métricas e imagens |
| Flower (`flwr==1.42.0`) | Clientes federados com `NumPyClient` |
| Flower Datasets (`flwr-datasets[vision]==0.6.1`) | Dados e particionamento para os cenários federados |
| Matplotlib | Visualizações dos experimentos Python |
| Dependências npm do frontend | Interface React/Vite, instaladas pelo lockfile |

`requirements-full.txt` inclui `requirements.txt` e adiciona Flower, Flower
Datasets e Matplotlib. O inicializador instala esse conjunto completo.
Este laboratório coordena o FedAvg diretamente, na mesma máquina: **não precisa
de Ray, `flwr[simulation]`, Docker ou de um servidor Flower separado**. O extra
`simulation` pertence ao Simulation Runtime do Flower, que não é usado aqui.
Veja a [documentação oficial de instalação do Flower](https://flower.ai/docs/framework/how-to-install-flower.html).

CUDA é opcional e exige uma GPU NVIDIA compatível, driver atualizado e a edição
CUDA do PyTorch; os passos estão em [GPU.md](GPU.md). `--setup` não converte uma
instalação CPU em CUDA. `--check` informa se o PyTorch detecta CUDA.
É necessário acesso à internet na instalação e ao obter dados ainda não presentes
no cache. A primeira execução federada pode baixar datasets; ela não tem o mesmo
custo da demonstração com três imagens incluídas.

O cenário federado precisa de mais dados e recursos do que a demonstração.
As configurações iniciais pequenas ajudam a verificar a instalação antes de
aumentar a carga. Veja [PROGRESS.md](PROGRESS.md) para o acompanhamento real.

## Preparar a demonstração para Vercel

O deploy usa `requirements.txt`, com PyTorch CPU e sem Flower. Em Vercel,
`VERCEL=1` ativa o perfil showcase no backend; o frontend usa explicitamente
`npm run build:showcase` no deploy. Os limites não
podem ser ampliados por uma opção enviada pelo navegador. Não configure
`APP_URL` manualmente: a Vercel injeta esse endereço pelo binding do gateway.

Para testar o perfil online localmente, sem alterar a versão completa:

```powershell
# Terminal 1
$env:DLG_APP_MODE = "showcase"
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

```powershell
# Terminal 2
$env:VITE_LAB_MODE = "showcase"
$env:DLG_DEV_API_URL = "http://127.0.0.1:8001"
npm.cmd --prefix frontend run dev -- --host 127.0.0.1 --port 5174
```

Abra http://127.0.0.1:5174. O backend rejeita cenários federados, defesas, GPU,
imagens externas, parâmetros livres e buscas acima dos presets. A demonstração
não baixa datasets nem calcula acurácia sobre o conjunto completo. Cada instância
calcula um experimento por vez; uma instância ocupada pede uma nova tentativa.
As funções continuam sujeitas aos limites e ao escalonamento da hospedagem.

Configuração de serviços, rotas e requisitos do deploy: [DEPLOYMENT.md](DEPLOYMENT.md).
O funcionamento local não substitui a validação de um preview na Vercel.

## Pesquisa e autoria

[Artigo original, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html).
Este projeto é uma implementação didática de Enrique Susin inspirada nessa
pesquisa. Os resultados descrevem cada configuração testada, sem garantia de
reconstrução perfeita ou proteção universal contra outros ataques.

O tom e o vocabulário da interface estão documentados em [CONTENT.md](CONTENT.md).

O diagrama “Como a tentativa acontece” usa um exemplo real previamente registrado
em `frontend/src/data/learning-demo.json`: original, mapa normalizado dos gradientes
da primeira camada e reconstrução. Ele não executa um ataque ao abrir a página.
Para regenerá-lo com as dependências Python instaladas, execute
`.\.venv\Scripts\python.exe scripts/generate_learning_demo.py` (CPU, seed 50,
uma imagem incluída e 50 iterações). O mapa de gradientes representa valores do
modelo, não pixels da imagem original.
