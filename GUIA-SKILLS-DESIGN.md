# Guia de skills de design para o Codex

Documento criado em 22/10/2026, com base nas skills instaladas nesta máquina.

## 1. O que foi instalado

Skills são instruções e recursos que o Codex consulta para executar um trabalho. Não são bibliotecas do site: não entram no bundle, não mudam o framework e não alteram a aplicação só por estarem instaladas.

| Skill | Nome para usar no Codex | Principal aplicação | Origem |
| --- | --- | --- | --- |
| Emil Design Engineering | `emil-design-eng` | Acabamento de componentes, interações e animações | [emilkowalski/skills](https://github.com/emilkowalski/skills) |
| Impeccable | `impeccable` | Planejamento, implementação, crítica e revisão de interfaces | [pbakaus/impeccable](https://github.com/pbakaus/impeccable) |
| Taste | `design-taste-frontend` | Direção de arte, composição, identidade e validação visual | [Leonxlnx/taste-skill](https://github.com/Leonxlnx/taste-skill) |

Foi instalada a skill principal de cada projeto. Os repositórios podem oferecer outras skills ou integrações que não fazem parte desta instalação.

## 2. Onde estão e como usar em outro projeto

As três foram instaladas globalmente para este usuário:

```text
C:\Users\enriq\.codex\skills\emil-design-eng\
C:\Users\enriq\.codex\skills\impeccable\
C:\Users\enriq\.codex\skills\design-taste-frontend\
```

Na mesma máquina e com o mesmo usuário/configuração do Codex, basta abrir outro projeto e pedir o uso delas. Não é necessário reinstalar, copiar arquivos para o repositório ou usar o ai-memory para ativá-las. Uma instalação que use outro `CODEX_HOME` pode ter uma pasta de skills diferente.

Exemplo de pedido no chat do Codex:

```text
Use as skills design-taste-frontend, impeccable e emil-design-eng
para redesenhar este projeto. Leia suas instruções antes de começar.
```

Também é possível mencionar uma skill com `$`, por exemplo:

```text
$emil-design-eng revise as interações e animações deste componente.
```

Esses exemplos são mensagens para o Codex, não comandos do PowerShell. Mencionar o nome e a tarefa em linguagem natural é suficiente quando a skill está disponível na sessão.

As skills não transportam o histórico deste projeto. Conteúdo, decisões e memória do projeto são assuntos separados; forneça o contexto do novo projeto no pedido.

## 3. Emil Design Engineering

### Para que serve

Orienta os detalhes que tornam uma interface consistente e responsiva: feedback ao clicar, transições, menus, modais, tooltips, curvas de animação e comportamento de componentes.

Um ponto central é decidir se uma interação precisa de animação antes de escolher o efeito. A frequência de uso, a finalidade e a percepção de velocidade importam mais que adicionar movimento por decoração.

### Quando usar

- Ao implementar ou revisar botões, menus, modais e outros componentes interativos.
- Quando uma interface funciona, mas parece lenta ou pouco refinada.
- Ao revisar animações e transições antes de apresentar o projeto.

### Exemplo

```text
Use emil-design-eng para revisar os botões, menus e modais.
Melhore o feedback e a fluidez sem atrasar as ações frequentes.
Respeite prefers-reduced-motion e preserve as funcionalidades.
```

A skill pede uma tabela de antes/depois e justificativa nas revisões de código de UI. A instalação atual contém instruções; bibliotecas de animação só devem ser adicionadas se a implementação precisar delas.

## 4. Impeccable

### Para que serve

Oferece um fluxo de design que considera produto, público, hierarquia, tipografia, cores, layout, responsividade e acessibilidade. Distingue uma reformulação visual completa de um refinamento do design existente.

Também distingue o objetivo de cada superfície: convencer e converter, executar uma tarefa, ler conteúdo ou explorar uma experiência, como um portfólio. Uma landing page e o painel de um mesmo produto podem exigir tratamentos diferentes.

### Pedidos úteis

Envie estas mensagens no chat do Codex:

| Mensagem | Objetivo |
| --- | --- |
| `$impeccable init` | Registrar contexto durável do produto em `PRODUCT.md` |
| `$impeccable shape página inicial` | Planejar a experiência e a interface |
| `$impeccable critique página inicial` | Avaliar clareza, hierarquia e experiência |
| `$impeccable audit interface` | Revisar acessibilidade, responsividade e qualidade técnica |
| `$impeccable polish interface` | Refinar os detalhes finais |
| `$impeccable typeset página inicial` | Melhorar tipografia e hierarquia textual |
| `$impeccable layout página inicial` | Ajustar composição, alinhamento e espaçamento |
| `$impeccable harden interface` | Trabalhar estados de erro, textos longos e casos extremos |

### Arquivos e requisitos

- `PRODUCT.md`: público, propósito, contexto, restrições e fatos do produto.
- `DESIGN.md`: registro do sistema visual, quando criado pelo fluxo de trabalho.
- A skill contém referências e um launcher Windows em `scripts/impeccable.cmd`.
- Na primeira execução, o launcher pode baixar o motor do Impeccable e precisar de rede.

A instalação feita aqui copiou a skill e seus recursos pelo instalador do Codex. Ela não executou o instalador completo do Impeccable e não configurou hooks automáticos. Usar a skill no chat e habilitar verificações automáticas após edições são etapas distintas.

## 5. Taste

### Para que serve

Ajuda a definir uma direção visual ligada ao conteúdo real do projeto, evitando composições genéricas. Trabalha conceito, tipografia, paleta, imagens e organização das seções, com inspeção do resultado no navegador.

O nome da pasta de origem no repositório é `skills/taste-skill`, mas o nome instalado para uso no Codex é `design-taste-frontend`.

### Quando usar

- No início de um redesign completo ou de uma página nova.
- Quando o site parece um template sem relação com o assunto.
- Para estruturar uma landing page, portfólio ou página de produto.
- Para dashboards e telas operacionais, seguindo o fluxo específico de app da skill.

### Exemplo

```text
Use design-taste-frontend para criar uma direção visual própria
para este projeto. Baseie o conceito no conteúdo e nos recursos reais.
Diferencie a página de apresentação da interface operacional.
Valide o resultado em desktop e celular.
```

### Requisitos e arquivos

- Node.js 22 ou superior para os scripts. Nesta máquina foi confirmado Node.js 24.20.0 na instalação.
- Chrome, Edge ou Chromium instalado para a inspeção automatizada no navegador.
- Acesso à rede quando o fluxo buscar imagens externas.
- Os scripts de Taste não exigem API keys, conforme as instruções instaladas.
- Os arquivos de trabalho são gerados em `.taste-skill/` no projeto a partir do qual os scripts são executados.

A skill inclui scripts como `draw.mjs` e `inspect.mjs`. O Codex deve consultar a documentação instalada para escolher os argumentos e o fluxo correto. A instalação confirmou a presença dos arquivos, mas não executou a inspeção de um site.

## 6. Como combinar as três

Uma divisão de responsabilidades recomendada para um redesign é:

1. **Impeccable:** entender o produto, os usuários, as restrições e as funções que precisam continuar funcionando.
2. **Taste:** estabelecer uma direção visual e uma composição ligadas ao assunto.
3. **Impeccable:** estruturar e implementar layout, componentes, estados e responsividade dentro dessa direção.
4. **Emil:** refinar o comportamento dos componentes e o movimento.
5. **Taste e Impeccable:** inspecionar a interface renderizada e corrigir problemas visuais e de usabilidade.

Essa divisão é uma recomendação de uso, não um mecanismo automático de coordenação entre as skills. Elas podem ter preferências diferentes; o briefing do projeto deve orientar a escolha. Evite aplicar regras contraditórias sem uma direção visual comum.

## 7. Prompt reutilizável para outro projeto

Copie o texto abaixo para uma conversa no projeto desejado e preencha os campos:

```text
Quero redesenhar a interface deste projeto usando as skills
impeccable, design-taste-frontend e emil-design-eng.
Leia as instruções das três antes de começar.

Projeto: [nome e finalidade]
Público: [quem usa ou avalia]
Objetivo principal: [o que a pessoa deve entender ou fazer]
Escopo: [páginas e telas incluídas]
Tecnologias e comando para rodar: [stack e comando, se conhecidos]
Conteúdo e recursos disponíveis: [textos, screenshots, imagens, logo]
Referências visuais: [links ou descrições, se houver]
Restrições: [marca, cores, fontes, dependências, prazo, dispositivos]

Trate este trabalho como uma reformulação visual completa.
Preserve os fatos, os dados e as funcionalidades existentes.
Use Impeccable para contexto e estrutura, Taste para direção visual
e Emil para interações e animações.
Documente as decisões e valide a interface no navegador,
incluindo celular, teclado e preferência por movimento reduzido.
Informe o que foi validado e qualquer limitação da verificação.
```

Para um ajuste pequeno, reduza o pedido: indique a tela ou componente e use apenas as skills que ajudam naquela tarefa.

## 8. Instalar em outra máquina

Peça ao Codex da outra máquina:

```text
Use skill-installer para instalar globalmente estas skills:

1. Repositório emilkowalski/skills
   Caminho: skills/emil-design-eng
   Nome: emil-design-eng

2. Repositório pbakaus/impeccable
   Caminho: .agents/skills/impeccable
   Nome: impeccable

3. Repositório Leonxlnx/taste-skill
   Caminho: skills/taste-skill
   Nome de destino: design-taste-frontend

Confira os caminhos atuais nos repositórios oficiais antes de instalar.
Preserve instalações existentes e informe os resultados.
```

Os caminhos acima foram verificados em 22/10/2026 e podem mudar no futuro. Para instalação manual, copie a pasta inteira de cada skill, incluindo scripts, referências e assets; copiar apenas `SKILL.md` é insuficiente para Taste e Impeccable.

O destino padrão é `~/.codex/skills/`, ou a pasta `skills` dentro de `CODEX_HOME` quando personalizado. Em outra máquina Windows, substitua o nome do usuário; não reutilize literalmente o caminho `C:\Users\enriq`.

## 9. Atualizações e problemas comuns

| Situação | O que fazer |
| --- | --- |
| Skill não aparece | Confirme o diretório de instalação e o `CODEX_HOME`; inicie uma nova conversa ou recarregue o Codex se necessário. |
| Funciona em um projeto, mas não em outro ambiente | Verifique se ambos usam o mesmo usuário, máquina e diretório de configuração. |
| Taste não consegue inspecionar a página | Confira Node.js 22+, navegador instalado e se o servidor local do projeto está acessível. |
| Impeccable não consegue executar seu launcher | Verifique permissões e rede; o contexto existente pode ser lido diretamente conforme o fallback da skill. |
| Skill existente ao tentar reinstalar | O instalador do Codex evita sobrescrever diretórios existentes. Peça atualização explícita e preserve customizações. |
| Esperava verificações automáticas do Impeccable | A cópia da skill não configurou hooks. Consulte a integração oficial antes de habilitá-los. |
| O resultado mudou após atualizar | Releia as instruções e registre a versão ou commit usado quando precisar reproduzir o fluxo. |

As skills estão disponíveis para novos trabalhos; cada projeto deve manter seu próprio briefing e decisões visuais. Reutilize as ferramentas, adaptando a direção ao conteúdo de cada projeto.
