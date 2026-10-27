# DLG / lab — Interface

Interface didática para o simulador Deep Leakage from Gradients. React 18, TypeScript, Vite, Tailwind CSS e Recharts; backend FastAPI/PyTorch existente na raiz do projeto.

## Executar localmente

Na raiz do repositório, instale e inicie a versão completa em um terminal:

```powershell
.\start.cmd --setup
.\start.cmd
```

Abra http://127.0.0.1:5173. Requisitos, Flower e CUDA estão no [README da raiz](../README.md#executar-localmente). Para trabalhar apenas na interface, com a API já iniciada, use na pasta `frontend`:

```powershell
npm.cmd ci
npm.cmd run dev
```

O Vite encaminha `/api` para a porta 8000. O modo completo usa o CIFAR-100 local ou imagens sob demanda; o cenário federado pode precisar baixar mais dados. Suas dependências estão em `requirements-full.txt`. O modo showcase usa três imagens incluídas no projeto, sem downloads, e mantém apenas CPU, lotes de uma ou duas imagens e 10/25/50 iterações, com opções de 100/200 para duas imagens. A versão completa permite lotes de até 8 imagens e mais configurações; o custo depende do hardware e dos parâmetros. As instruções completas e a prévia online estão no README da raiz.

## Vercel Services

A configuração de deploy está na raiz. Use `npx.cmd vercel@latest dev --local --listen 3005` na raiz para iniciar frontend, gateway e API com os bindings gerados pelo Vercel. O gateway consome `APP_URL` em runtime; essa variável não deve ser definida manualmente nem exposta no bundle Vite. Consulte `DEPLOYMENT.md` na raiz para o mapa de rotas e os detalhes pendentes de confirmação.

## Organização

- **Visão geral:** mecanismo interativo, conceitos, defesas e interpretação de SSIM/PSNR/MSE.
- **Experimento:** parâmetros originais, defesas, distribuição e resumo da configuração.
- **Execução:** progresso real enviado pelo backend, etapas, contagem de ciclos e cancelamento da espera.
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

A validação posterior do showcase incluiu reconstruções reais das três imagens incluídas, em CPU, com progresso recebido da API. O deploy ainda exige validação em um preview da Vercel.

## Estados importantes

- A ausência do backend é indicada na interface; a leitura didática e a configuração continuam disponíveis.
- O progresso vem do servidor por streaming, com contagem das iterações reais do ataque.
- Cancelar a espera aborta a requisição no navegador, mas não garante encerrar a tarefa Python.
- A seleção IID/Dirichlet controla a visualização de distribuição. O endpoint federado atual tem seu próprio contrato; não se deve assumir que a visualização altera o particionamento do treinamento.
- A fonte IBM Plex é solicitada ao Google Fonts e possui fallback local. Para uso totalmente offline, os arquivos de fonte precisam ser hospedados junto ao site.
