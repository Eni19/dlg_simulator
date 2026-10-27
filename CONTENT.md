# Linguagem do laboratório

## Para quem escrevemos

Pessoas curiosas sobre IA e privacidade, incluindo quem está chegando ao assunto.
O visitante precisa entender a pergunta do experimento, experimentar com uma
configuração simples e interpretar o que viu antes de explorar o código.

## Voz

Conversar com um adulto interessado: claro, acolhedor e preciso. Convidar a
observar, comparar e investigar. Explicar uma ideia por vez, com frases concretas,
sem transformar o texto em aula formal ou infantilizar o leitor.

A primeira tela começa pela pergunta sobre imagens e aprendizado. Define
gradientes em linguagem comum, apresenta a origem da pesquisa e oferece um
primeiro experimento. Fórmulas e controles avançados ficam disponíveis por
expansão, depois da explicação em palavras.

## Termos consistentes

| Termo | Como apresentar |
| --- | --- |
| Modelo | A parte da IA que transforma uma imagem em uma previsão de categoria. |
| Gradientes | Números que orientam os ajustes do modelo. “Pistas” é uma analogia para a informação que eles podem carregar, não uma definição matemática. |
| Ataque | Tentativa controlada de descobrir informações a partir do acesso aos gradientes e ao modelo. |
| Batch | Grupo de imagens usadas juntas no cálculo; apresentar como “imagens por vez”. |
| Iterações | Oportunidades de ajuste configuradas. O progresso informa ciclos reais, que podem conter vários ajustes internos. |
| Loss | Erro numérico que guia a busca. Não equivale à semelhança visual da reconstrução. |
| Seed | Ponto de partida dos cálculos aleatórios. Não prometer que ela fixa a seleção das imagens ou produz resultados idênticos entre dispositivos. |
| Cliente federado | Participante simulado que aprende com seus próprios exemplos. |
| Rótulo | Categoria associada a uma imagem. |
| Poda | Zerar parte dos pesos, que representam conexões do modelo. |
| Esparsificação | Zerar parte dos gradientes antes do compartilhamento. |

## Compromissos com a precisão

- Distinguir os autores do método original do autor deste laboratório.
- O DLG foi apresentado por Ligeng Zhu, Zhijian Liu e Song Han na NeurIPS 2019.
  Fonte: https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html
- Apresentar este projeto como implementação didática inspirada na pesquisa,
  com seus próprios controles e escolhas de implementação.
- Dizer “tenta reconstruir”, sem prometer recuperação perfeita ou universal.
- Usar “em certas condições” ao falar do risco demonstrado pela pesquisa.
- Explicar que as imagens são públicas e os participantes federados são simulados.
- Uma reconstrução ruim não prova proteção contra todos os ataques.
- Uma loss menor não garante uma imagem visualmente melhor.
- A imagem original serve para avaliar os resultados; a busca usa gradientes.
- Percentuais descrevem trabalho concluído, nunca tempo restante inventado.

## Em cada etapa

Introdução: por que a pergunta importa e como a tentativa funciona.
Configuração: o que cada escolha muda e um ponto de partida recomendado.
Execução: o que está realmente acontecendo, com contagens do servidor.
Resultados: olhar imagens antes dos números; separar observação de conclusão.
Código: explicar a função de cada arquivo antes dos nomes dos algoritmos.

Manter as referências e os nomes técnicos para quem quiser se aprofundar.
Evitar jargão de infraestrutura no caminho principal; detalhes da execução
podem aparecer sob demanda.
