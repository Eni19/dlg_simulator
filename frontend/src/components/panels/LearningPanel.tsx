import { useState, type CSSProperties } from "react";
import demoSource from "@/data/learning-demo.json?raw";
import {
  ArrowRight,
  ArrowUpRight,
  ShieldCheck,
} from "lucide-react";

const demo = JSON.parse(demoSource) as {
  original: string; gradient: string; early: string; reconstruction: string;
  earlyIteration: number; iterations: number;
};

const stages = [
  {
    title: "A imagem de partida",
    text: "Uma pequena rede neural, o sistema de IA deste experimento, recebe uma imagem e sua categoria. Ao aprender com esse exemplo, calcula os ajustes que precisa fazer.",
    label: "Imagem",
    code: "x, y → LeNet → ∇θ L",
    image: demo.original,
  },
  {
    title: "As pistas do aprendizado",
    text: "Esses ajustes são orientados por números chamados gradientes. Eles indicam como o modelo deve mudar para errar menos e são as pistas que o ataque recebe.",
    label: "Pistas",
    code: "g = ∇θ L(fθ(x), y)",
    image: demo.gradient,
  },
  {
    title: "A tentativa de reconstrução",
    text: "O ataque começa com uma imagem ruidosa. Ele a modifica até que produza gradientes parecidos com os da imagem original. Assim, tenta descobrir qual exemplo gerou aquelas pistas.",
    label: "Reconstrução",
    code: "x′, y′ = argmin D(g′, g)",
    image: demo.reconstruction,
  },
];

export function LearningPanel({
  onExperiment,
  onCode,
  isShowcase = false,
}: {
  onExperiment: () => void;
  onCode: () => void;
  isShowcase?: boolean;
}) {
  const [stage, setStage] = useState(0);
  const [animateStage, setAnimateStage] = useState(false);
  const current = stages[stage];
  return (
    <div className="learning-panel">
      <section className="intro-layout" aria-labelledby="intro-title">
        <div className="intro-copy">
          <h1 id="intro-title">Uma IA pode revelar as imagens que viu?</h1>
          <p>
            Para aprender com uma imagem, uma inteligência artificial calcula
            como deve se ajustar. Os gradientes, números que orientam esses
            ajustes, podem carregar pistas sobre o exemplo usado no aprendizado.
            O que alguém conseguiria descobrir
            se tivesse acesso a elas?
          </p>
          <p>
            Aqui, você acompanha uma tentativa de refazer uma imagem a partir
            dessas pistas. Explore com calma, compare o antes e o depois e veja
            como pequenas mudanças afetam a reconstrução.
          </p>
          <div className="intro-actions">
            <button className="primary-action" onClick={onExperiment}>
              {isShowcase ? "Experimentar a demonstração" : "Fazer meu primeiro experimento"} <ArrowRight size={17} />
            </button>
            <button className="text-action" onClick={onCode}>
              Conhecer o código <ArrowUpRight size={16} />
            </button>
          </div>
          <p className="intro-note">
            {isShowcase
              ? "Você está no showcase online: escolha até duas das três imagens públicas do CIFAR-100 e acompanhe uma reconstrução real. Com duas imagens, explore buscas de até 200 iterações."
              : "Usamos pequenas imagens públicas do CIFAR-100, um conjunto de dados para pesquisa. Comece com 1 imagem e ajuste o experimento depois."}
          </p>
          <p className="intro-note">
            {isShowcase ? "Na versão completa, no seu computador, você pode" : "Neste laboratório, você pode"}
            {" "}explorar defesas e observar como elas afetam a reconstrução, além de investigar
            o ataque em aprendizado federado, quando vários participantes aprendem juntos.
            Também há lotes de até 8 imagens, CPU/GPU e ajustes avançados.
            {isShowcase && " Esses experimentos exigem mais processamento e memória; o tempo depende do hardware e da configuração."}
          </p>
        </div>
        <section className="mechanism mechanism-refined" aria-labelledby="mechanism-title">
          <header className="mechanism-intro">
            <h2 id="mechanism-title">Como a tentativa acontece</h2>
            <p>Da imagem às pistas. Das pistas, uma tentativa de voltar.</p>
          </header>
          <div className="mechanism-tabs" role="tablist" aria-label="Etapas da tentativa"
            data-motion={animateStage ? "on" : "off"}
            style={{ "--indicator-x": `calc(${stage * 100}% + ${stage * 4}px)` } as CSSProperties}
            onKeyDown={event => {
              if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
              event.preventDefault();
              const next = event.key === "Home" ? 0 : event.key === "End" ? 2
                : (stage + (event.key === "ArrowRight" ? 1 : -1) + stages.length) % stages.length;
              setAnimateStage(false);
              setStage(next);
              event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
            }}>
            {stages.map((s, i) => <button key={s.title} type="button" role="tab"
              id={`mechanism-tab-${i}`} aria-selected={stage === i}
              aria-controls="mechanism-panel" tabIndex={stage === i ? 0 : -1}
              onClick={event => { setAnimateStage(event.detail > 0); setStage(i); }}>{s.label}</button>)}
          </div>
          <div id="mechanism-panel" key={stage} className="mechanism-panel" data-motion={animateStage ? "on" : "off"}
            role="tabpanel" aria-labelledby={`mechanism-tab-${stage}`} tabIndex={0}>
          <figure className="mechanism-example">
            <div className="mechanism-example-images">
              {stage === 2 && <div>
                <img src={`data:image/png;base64,${demo.early}`} alt="Primeiro registro da tentativa de reconstrução do dinossauro" width={144} height={144} />
                <span>Primeiro registro</span>
              </div>}
              <div>
                <img src={`data:image/png;base64,${current.image}`}
                  alt={stage === 0 ? "Imagem original de um dinossauro, do conjunto público CIFAR-100" : stage === 1
                    ? "Mapa de valores dos gradientes da primeira camada: tons de azul representam sinais diferentes"
                    : "Reconstrução do dinossauro obtida em uma execução real do laboratório"}
                  width={144} height={144} />
                <span>{stage === 0 ? "Original · dinossauro" : stage === 1 ? "Gradientes em cores" : `Após ${demo.iterations} iterações`}</span>
              </div>
            </div>
            <figcaption className="sr-only">{current.title}: exemplo real registrado no laboratório.</figcaption>
          </figure>
          <div className="mechanism-explanation">
            <h3>{current.title}</h3>
            <p>{current.text}</p>
            {stage === 1 && <p className="mechanism-gradient-note">As cores representam valores dos gradientes da primeira camada, não pixels da imagem original.</p>}
            <details className="mechanism-math" key={stage}>
              <summary>Entender os cálculos</summary>
              <code>{current.code}</code>
            </details>
          </div>
          </div>
          <details className="mechanism-provenance">
            <summary>Sobre este exemplo real</summary>
            <p>
            Exemplo real previamente registrado · 1 imagem, CPU, seed 50 e 50 iterações.
            O resultado do seu experimento pode variar e aparece após a execução.
            </p>
          </details>
        </section>
      </section>
      <section className="research-context reading-section" aria-labelledby="research-heading">
        <h2 id="research-heading">De onde veio essa ideia?</h2>
        <p>
          Em 2019, <strong>Ligeng Zhu, Zhijian Liu e Song Han</strong> apresentaram
          o <em>Deep Leakage from Gradients</em>, ou DLG, na conferência NeurIPS.
          O trabalho mostrou que, em certas condições, gradientes compartilhados
          durante o aprendizado permitem recuperar exemplos usados no treino.
        </p>
        <p>
          Esse resultado ajuda a entender uma questão de privacidade: compartilhar
          os cálculos de um modelo também pode expor informações sobre seus dados.
          Este laboratório, desenvolvido por Enrique Susin, é uma implementação
          didática inspirada nessa pesquisa, com controles para explorar o ataque
          e comparar formas de dificultá-lo.
        </p>
        <a className="text-action"
          href="https://proceedings.neurips.cc/paper/2019/hash/60a6c4002cc7b29142def8871531281a-Abstract.html"
          target="_blank" rel="noreferrer">
          Conhecer o artigo original, em inglês <ArrowUpRight size={16} />
        </a>
      </section>
      <section className="explainer-section" aria-labelledby="learn-heading">
        <div className="section-heading">
          <h2 id="learn-heading">O que você pode descobrir aqui</h2>
          <p>
            Você pode começar com a configuração pronta e investigar uma pergunta por vez.
          </p>
        </div>
        <div className="learning-columns">
          <article>
            <h3>Quanto de uma imagem pode voltar?</h3>
            <p>
              Veja uma imagem original ao lado da tentativa de reconstrução.
              Depois, dê mais tentativas ao algoritmo e observe se as cores,
              os contornos e os detalhes ficam mais próximos.
            </p>
            <button className="text-action" onClick={onExperiment}>
              Preparar meu experimento <ArrowRight size={16} />
            </button>
          </article>
          <article>
            <h3>O que dificulta essa descoberta?</h3>
            <p>
              {isShowcase ? "Na versão completa, acrescente" : "Acrescente"} pequenas perturbações às pistas, compartilhe menos
              informações ou remova parte das conexões do modelo. Compare o
              resultado com e sem uma dessas defesas.
            </p>
            <div className="inline-note">
              <ShieldCheck size={16} /> O efeito de uma defesa depende do cenário e da sua intensidade.
            </div>
          </article>
          <article>
            <h3>Como saber se a tentativa deu certo?</h3>
            <p>
              Comece olhando as imagens. Você também pode percorrer registros
              da reconstrução ao longo da busca e consultar medidas de
              semelhança para apoiar sua interpretação.
            </p>
            <button className="text-action" onClick={onCode}>
              Entender a implementação <ArrowRight size={16} />
            </button>
          </article>
        </div>
      </section>
      <section className="loss-explainer" aria-labelledby="loss-heading">
        <div>
          <h2 id="loss-heading">Da imagem ruidosa aos primeiros contornos</h2>
          <p>
            A cada tentativa, o algoritmo compara as pistas da imagem que está
            criando com as pistas recebidas. Ele usa essa diferença para escolher
            o próximo ajuste. A imagem original fica reservada para a comparação
            dos resultados; ela não orienta essa busca.
          </p>
          <button className="text-action" onClick={onCode}>
            Ver como a busca foi programada <ArrowRight size={16} />
          </button>
        </div>
        <details className="formula-panel">
          <summary>Quero conhecer a matemática</summary>
          <code>L = (1 − α) L₂ + α Lcos + λ(t) TV + β Lanchor</code>
          <p>
            Neste laboratório, L é a função de perda: uma pontuação usada para
            orientar os ajustes. Ela combina diferenças entre os gradientes com
            termos que suavizam a imagem e orientam a categoria estimada.
            α, λ(t) e β controlam o peso de cada parte.
          </p>
        </details>
      </section>
      <section className="reading-section" aria-labelledby="reading-heading">
        <h2 id="reading-heading">Primeiro as imagens, depois os números</h2>
        <p className="reading-intro">
          Uma tentativa pode recuperar cores e formas sem reproduzir todos os
          detalhes. Os números abaixo ajudam a comparar imagens, mas vale olhar
          para elas antes de tirar uma conclusão.
        </p>
        <div className="metric-glossary">
          <div>
            <strong>SSIM</strong>
            <p>
              Compara a organização visual das imagens, como formas e contornos.
              Quanto mais próximo de 1, maior a semelhança estrutural.
            </p>
          </div>
          <div>
            <strong>
              PSNR <small>dB</small>
            </strong>
            <p>
              Resume o tamanho do erro entre as imagens numa escala em decibéis.
              Valores maiores costumam indicar menos erro. Isso, por si só,
              não diz se o objeto ficou reconhecível.
            </p>
          </div>
          <div>
            <strong>MSE</strong>
            <p>
              Compara as cores ponto a ponto. Valores menores indicam que os
              pixels ficaram numericamente mais próximos dos originais.
            </p>
          </div>
        </div>
        <details>
          <summary>Por que chamamos essa tentativa de ataque?</summary>
          <p>
            Na pesquisa em segurança, um ataque é uma forma de testar o que
            alguém conseguiria descobrir ou fazer com determinado acesso.
            Aqui, investigamos o que pode ser recuperado quando se conhecem os
            gradientes e o modelo que os gerou. As imagens usadas são públicas
            e a tentativa acontece dentro deste experimento controlado.
          </p>
        </details>
        <details>
          <summary>Por que começar com uma imagem por vez?</summary>
          <p>
            Quando várias imagens participam do mesmo cálculo, suas pistas se
            misturam. Separar o que veio de cada uma torna o desafio maior.
            Por isso, comece com uma imagem. No formulário, esse tamanho de grupo
            também é chamado de batch.
          </p>
        </details>
        <details>
          <summary>Como várias pessoas ou dispositivos podem treinar uma IA juntos?</summary>
          <p>
            No aprendizado federado, cada participante aprende com seus próprios
            exemplos e envia atualizações a um coordenador, o servidor, que as
            combina. Esses participantes são chamados de clientes. Aqui, todos
            são simulados pelo laboratório. Esse cenário permite investigar o
            que alguém com acesso às atualizações poderia reconstruir.
          </p>
        </details>
        <details>
          <summary>Uma imagem irreconhecível significa que os dados estão protegidos?</summary>
          <p>
            Esse resultado mostra o comportamento desta tentativa, com estas
            imagens e configurações. Ele não prova que os dados estariam
            protegidos de todos os ataques. Compare diferentes configurações e
            observe quais mudanças tornam a reconstrução mais difícil.
          </p>
        </details>
      </section>
    </div>
  );
}
