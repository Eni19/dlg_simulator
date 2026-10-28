# DLG / lab

## Direction

Caderno de investigação. Uma aplicação didática que parece o caderno de alguém
curioso: papel quente, tinta azul-petróleo, anotações à mão na margem e as
imagens 32 × 32 coladas como fotos. Escolhida pelo autor em 2026-10-28 para
substituir a direção anterior (`framed-sheet`, IBM Plex), que ficou fria,
"quadrada e crua" para um público de portfólio e de leigos curiosos.

Construída com as skills impeccable (modo Operate, com a visão geral mais
expressiva), design-taste-frontend e emil-design-eng. Onde o briefing pede
calor humano (fonte manuscrita, fotos com sombra), ele prevalece sobre as regras
gerais das skills; o restante delas continua valendo.

## Idea

No laboratório, o caminho imagem → pistas → reconstrução vira a anotação de uma
investigação. As imagens reais do experimento (o dinossauro, o ruído inicial, a
reconstrução) são as "fotos coladas"; as notas à mão dizem o que observar, nunca
inventam um resultado.

## System

- Fundo `--paper #f1eee8` com pontilhado de caderno (só no fundo, nunca sob o
  texto); folha `--sheet #fffdf9`, raio 24px, sem sombra; painéis internos em
  `--raised #f5f2ec`, raio 18–24px.
- Tinta `--text #1f2a30`; secundário `--ink-soft #3a464b` e `--muted #57625f`;
  azul-petróleo `--accent #1d6a72` em links, dados e foco.
- Marca-texto `--marker #ffe27a`: uma frase-chave por tela e a seleção de texto.
- Caneta `--pen #2b59a6`: só as notas manuscritas.
- Tipografia: Newsreader (títulos), Figtree (texto, 15–17.5px, tinta cheia),
  Caveat (notas à mão, rótulos de foto, assinatura) e IBM Plex Mono só em
  código, arquivos e registros reais.
- Botões principais em pílula na cor da tinta; links sublinhados com recuo.
  Separadores tracejados, como as linhas de um caderno.
- `.photo`: moldura branca, fita adesiva, leve inclinação e sombra de objeto
  físico; endireita no hover (só ponteiros finos).
- Navegação lateral de 244px; gaveta no celular.

## Motion

Curvas fortes (`--ease-out cubic-bezier(0.23, 1, 0.32, 1)`), 160–320ms na
interface. Botões respondem ao toque com `scale(0.97)`. A visão geral entra em
cascata (60ms entre elementos), o marca-texto se desenha e a nota "comece por
aqui!" aparece por último; trocas de aba usam um fade curto. Hover só em
`(hover: hover) and (pointer: fine)`. Com redução de movimento, nada se desloca
e tudo aparece no estado final.

## Product truth

Preservar endpoints e funções científicas. O frontend mostra contagens reais
por etapa, recebidas do backend via stream, sem estimar tempo restante. Cancelar
encerra a espera; o processamento Python pode continuar. Não fabricar imagens ou
métricas: as fotos da visão geral vêm de `learning-demo.json` (execução real
registrada) e os estados vazios mostram molduras vazias, não resultados falsos.

## Surfaces

Visão geral didática; experimento com resumo (completo e showcase); execução;
reconstruções e snapshots; análise; explorador de quatro arquivos Python.

## Validation

Capturas em `.taste-skill/redesign-*` (antes e depois, 1440/768/390px),
ignoradas pelo Git. Desvios conscientes do inspetor do taste: três famílias de
fonte (a manuscrita é o conceito), sombras nas fotos coladas, título da visão
geral maior que o de tela de app e fonte diferente da direção antiga registrada
em `.taste-skill/direction.json`.
