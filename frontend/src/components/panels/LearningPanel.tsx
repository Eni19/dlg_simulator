import { useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Database,
  Network,
  ScanLine,
  ShieldCheck,
} from "lucide-react";

const stages = [
  {
    title: "Dados privados",
    text: "As imagens ficam com o cliente. O modelo calcula como seus parâmetros devem mudar para aprender com elas.",
    icon: Database,
    code: "x, y → LeNet → ∇θ L",
  },
  {
    title: "Gradientes compartilhados",
    text: "A atualização viaja até o servidor. Compartilhar gradientes, em vez das imagens, não garante que os dados permaneçam privados.",
    icon: Network,
    code: "g = ∇θ L(fθ(x), y)",
  },
  {
    title: "Reconstrução por DLG",
    text: "O atacante ajusta imagens e rótulos candidatos até que seus gradientes se aproximem dos gradientes observados.",
    icon: ScanLine,
    code: "x′, y′ = argmin D(g′, g)",
  },
];

export function LearningPanel({
  onExperiment,
  onCode,
}: {
  onExperiment: () => void;
  onCode: () => void;
}) {
  const [stage, setStage] = useState(1);
  const current = stages[stage];
  const Icon = current.icon;
  return (
    <div className="learning-panel">
      <section className="intro-layout" aria-labelledby="intro-title">
        <div className="intro-copy">
          <h1 id="intro-title">O que seus gradientes revelam?</h1>
          <p>
            Enviar apenas gradientes protege as imagens de treinamento? Explore
            como o <strong>Deep Leakage from Gradients</strong> tenta
            reconstruir dados a partir das atualizações de uma rede neural.
          </p>
          <div className="intro-actions">
            <button className="primary-action" onClick={onExperiment}>
              Abrir experimento <ArrowRight size={17} />
            </button>
            <button className="text-action" onClick={onCode}>
              Explorar o código <ArrowUpRight size={16} />
            </button>
          </div>
          <p className="intro-note">
            CIFAR-100 · LeNet com Sigmoid · PyTorch + Flower
          </p>
        </div>
        <div className="mechanism" aria-label="Diagrama interativo do ataque">
          <div className="mechanism-heading">
            <span>Da atualização ao dado</span>
            <span>Explore o fluxo</span>
          </div>
          <div className="mechanism-route">
            {stages.map((s, i) => {
              const StageIcon = s.icon;
              return (
                <div className="route-item" key={s.title}>
                  <button
                    aria-label={s.title}
                    aria-pressed={stage === i}
                    onClick={() => setStage(i)}
                    className={
                      stage === i ? "route-node selected" : "route-node"
                    }
                  >
                    <StageIcon size={25} />
                  </button>
                  <span>{s.title}</span>
                  {i < 2 && <ArrowRight className="route-arrow" size={18} />}
                </div>
              );
            })}
          </div>
          <div className="gradient-map" aria-hidden="true">
            {Array.from({ length: 112 }, (_, i) => (
              <span
                key={i}
                style={{
                  opacity: 0.08 + ((Math.sin(i * 1.37 + stage) + 1) / 2) * 0.86,
                  backgroundColor:
                    i % 7 < 3 ? "var(--chart-1)" : "var(--chart-2)",
                }}
              />
            ))}
          </div>
          <div className="mechanism-detail" aria-live="polite">
            <Icon size={16} />
            <code>{current.code}</code>
            <p>{current.text}</p>
          </div>
          <span className="diagram-caption">
            Visualização conceitual · não representa uma execução
          </span>
        </div>
      </section>
      <section className="explainer-section" aria-labelledby="learn-heading">
        <div className="section-heading">
          <h2 id="learn-heading">Do conceito ao experimento</h2>
          <p>
            Altere uma hipótese. Observe a reconstrução. Investigue a causa.
          </p>
        </div>
        <div className="learning-columns">
          <article>
            <h3>Reconstruir</h3>
            <p>
              Escolha um batch de 1 a 8 imagens e ajuste as iterações. O
              otimizador LBFGS busca entradas que expliquem os gradientes
              interceptados.
            </p>
            <button className="text-action" onClick={onExperiment}>
              Configurar ataque <ArrowRight size={16} />
            </button>
          </article>
          <article>
            <h3>Proteger</h3>
            <p>
              Experimente ruído, esparsificação e poda do modelo. Compare a
              qualidade da reconstrução e o efeito da defesa nas métricas
              retornadas.
            </p>
            <div className="inline-note">
              <ShieldCheck size={16} /> Privacidade também é uma questão de
              configuração.
            </div>
          </article>
          <article>
            <h3>Entender</h3>
            <p>
              Compare originais, reconstruções e mapas de erro. Acompanhe
              snapshots e a curva de perda para interpretar o que a otimização
              conseguiu recuperar.
            </p>
            <button className="text-action" onClick={onCode}>
              Ver a implementação <ArrowRight size={16} />
            </button>
          </article>
        </div>
      </section>
      <section className="loss-explainer" aria-labelledby="loss-heading">
        <div>
          <h2 id="loss-heading">Uma busca guiada por gradientes</h2>
          <p>
            A implementação combina distância L2 e direção dos gradientes, com
            regularização espacial adaptativa e ancoragem dos rótulos.
          </p>
          <button className="text-action" onClick={onCode}>
            Ler a função de perda <ArrowRight size={16} />
          </button>
        </div>
        <div className="formula-panel">
          <code>L = (1 − α) L₂ + α Lcos + λ(t) TV + β Lanchor</code>
          <p>
            α equilibra magnitude e direção. λ(t) reduz a suavização ao longo da
            busca. β controla a ancoragem dos rótulos.
          </p>
        </div>
      </section>
      <section className="reading-section" aria-labelledby="reading-heading">
        <h2 id="reading-heading">Como interpretar o resultado</h2>
        <div className="metric-glossary">
          <div>
            <strong>SSIM</strong>
            <p>
              Similaridade estrutural. Valores mais próximos de 1 indicam maior
              semelhança entre a imagem original e a reconstruída.
            </p>
          </div>
          <div>
            <strong>
              PSNR <small>dB</small>
            </strong>
            <p>
              Relação sinal-ruído. Valores maiores costumam indicar menor erro,
              mas não garantem a recuperação dos detalhes semânticos.
            </p>
          </div>
          <div>
            <strong>MSE</strong>
            <p>
              Erro quadrático médio entre pixels. Quanto menor, maior a
              proximidade numérica; interprete junto das imagens.
            </p>
          </div>
        </div>
        <details>
          <summary>Por que usar batches menores?</summary>
          <p>
            Várias imagens compartilham o mesmo gradiente agregado. Separar suas
            contribuições torna a reconstrução mais difícil. No ambiente
            federado, comece com batch 1 para explorar o comportamento do
            ataque.
          </p>
        </details>
        <details>
          <summary>O que muda no aprendizado federado?</summary>
          <p>
            Os clientes treinam localmente e o servidor agrega atualizações. O
            simulador permite explorar servidor curioso, cliente malicioso e
            interceptação na rede, além de distribuições IID e Dirichlet.
          </p>
        </details>
        <details>
          <summary>Uma defesa impede qualquer reconstrução?</summary>
          <p>
            Não há uma garantia universal neste experimento. O resultado depende
            da intensidade da defesa, dos dados, dos parâmetros e do
            conhecimento do atacante. Em model pruning, compare o atacante
            informado e o desinformado.
          </p>
        </details>
      </section>
    </div>
  );
}
