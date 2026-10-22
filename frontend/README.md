# DLG / lab — Interface

Interface didática para o simulador Deep Leakage from Gradients. React 18, TypeScript, Vite, Tailwind CSS e Recharts; backend FastAPI/PyTorch existente na raiz do projeto.

## Executar localmente

Na pasta `frontend`:

```powershell
npm.cmd install
npm.cmd run dev
```

Abra http://127.0.0.1:5173. Para executar o processamento científico, inicie o backend em outro terminal, na raiz do repositório e no ambiente Python do projeto:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

O Vite encaminha `/api` para a porta 8000. A primeira execução científica pode precisar baixar o CIFAR-100. As dependências opcionais do cenário federado continuam sujeitas à configuração original do backend.

## Organização

- **Visão geral:** mecanismo interativo, conceitos, defesas e interpretação de SSIM/PSNR/MSE.
- **Experimento:** parâmetros originais, defesas, distribuição e resumo da configuração.
- **Execução:** início, estimativa de progresso, logs explicativos e cancelamento da espera.
- **Reconstruções:** imagens retornadas pela API, mapas de erro, snapshots e ampliação com teclado.
- **Análise:** convergência e comparação de defesa quando a resposta fornece dados pareados.
- **Código:** trechos e arquivos completos de quatro módulos Python reais, incorporados no build pelo Vite.

Os fragmentos `#setup`, `#code`, `#execute`, `#results` e `#charts` podem selecionar a tela ao abrir a aplicação. Configuração e resultado ficam em memória durante a sessão; não existe histórico persistente de experimentos nesta interface.

## Implementação visual

O sistema visual está em `src/lab.css`. `tailwind.config.js` adapta os nomes de cores dos componentes existentes à nova paleta. Gráficos, configuração e código são carregados sob demanda. Os contratos de ataque foram preservados; o envio de `attacker_role` ao endpoint federado foi corrigido para refletir a seleção da interface.

`PRODUCT.md` e `DESIGN.md`, na raiz, registram o contexto e a direção visual. O guia das skills também está na raiz.

## Verificação

```powershell
npm.cmd run build
```

A revisão de outubro de 2026 incluiu compilação TypeScript, inspeção em 1440/768/390px e testes de integração de interface em navegador isolado. Respostas da API foram interceptadas apenas dentro do navegador de teste para verificar os fluxos, parâmetros, erros, cancelamento e acessibilidade dos controles. Isso não valida a qualidade científica das reconstruções.

O Python disponível durante a revisão não tinha PyTorch, torchvision, FastAPI ou uvicorn instalados; portanto não foi executado um ataque real. O código científico e os notebooks foram preservados.

## Estados importantes

- A ausência do backend é indicada na interface; a leitura didática e a configuração continuam disponíveis.
- O progresso é estimado no navegador. Os logs de etapas são explicativos, não telemetria em tempo real do servidor.
- Cancelar a espera aborta a requisição no navegador, mas não garante encerrar a tarefa Python.
- A seleção IID/Dirichlet controla a visualização de distribuição. O endpoint federado atual tem seu próprio contrato; não se deve assumir que a visualização altera o particionamento do treinamento.
- A fonte IBM Plex é solicitada ao Google Fonts e possui fallback local. Para uso totalmente offline, os arquivos de fonte precisam ser hospedados junto ao site.
