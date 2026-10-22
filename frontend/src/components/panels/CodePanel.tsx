import { useState } from "react";
import { ArrowUpRight, Check, Copy } from "lucide-react";
import losses from "../../../../src/losses.py?raw";
import model from "../../../../src/models.py?raw";
import attack from "../../../../src/dlg_attack.py?raw";
import federated from "../../../../src/federated_flower.py?raw";

const files = [
  {
    name: "losses.py",
    path: "src/losses.py",
    title: "A função que guia a reconstrução",
    description:
      "L2 aproxima a magnitude dos gradientes; cosine similarity aproxima a direção. Total Variation regulariza a imagem e seu coeficiente decai ao longo da otimização.",
    source: losses,
    anchor: "def grad_matching_loss(",
    end: "# Perda",
  },
  {
    name: "models.py",
    path: "src/models.py",
    title: "O modelo alvo",
    description:
      "A LeNet do experimento usa quatro camadas convolucionais com ativação Sigmoid e uma camada totalmente conectada para as 100 classes do CIFAR-100.",
    source: model,
    anchor: "class LeNet(",
    end: "def create_lenet",
  },
  {
    name: "dlg_attack.py",
    path: "src/dlg_attack.py",
    title: "O motor do ataque",
    description:
      "O executor reúne inicialização inteligente, otimização LBFGS, fase global periódica, TV adaptativa e label anchoring. Explore a implementação completa no repositório.",
    source: attack,
    anchor: "class DLGAttack:",
    end: "",
  },
  {
    name: "federated_flower.py",
    path: "src/federated_flower.py",
    title: "O cenário distribuído",
    description:
      "Este módulo implementa clientes, agregação e os pontos de interceptação usados pelo cenário federado. As restrições do ambiente permanecem as mesmas do backend original.",
    source: federated,
    anchor: "",
    end: "",
  },
];

export function CodePanel() {
  const [selected, setSelected] = useState(0);
  const [copied, setCopied] = useState(false);
  const [showFull, setShowFull] = useState(false);
  const file = files[selected];
  const start = file.anchor ? Math.max(0, file.source.indexOf(file.anchor)) : 0;
  const remaining = file.source.slice(start);
  const end = file.end ? remaining.indexOf(file.end) : -1;
  const code = showFull
    ? file.source
    : (end > 0 ? remaining.slice(0, end) : remaining)
        .split("\n")
        .slice(0, selected < 2 ? 85 : 65)
        .join("\n")
        .trim();
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setCopied(false);
    }
  };
  return (
    <section className="code-panel">
      <div className="section-heading">
        <h1>Por dentro do código</h1>
        <p>Da interface à implementação científica, sem caixa-preta.</p>
      </div>
      <div className="source-tabs" aria-label="Arquivos da implementação">
        {files.map((f, i) => (
          <button
            key={f.name}
            aria-pressed={selected === i}
            onClick={() => {
              setSelected(i);
              setCopied(false);
              setShowFull(false);
            }}
          >
            {f.name}
          </button>
        ))}
      </div>
      <div className="code-intro">
        <div>
          <h2>{file.title}</h2>
          <p>{file.description}</p>
        </div>
        <a
          className="text-action"
          href={`https://github.com/Eni19/dlg_simulator/blob/HEAD/${file.path}`}
          target="_blank"
          rel="noreferrer"
        >
          Arquivo completo <ArrowUpRight size={16} />
        </a>
      </div>
      <div className="source-view">
        <div className="source-toolbar">
          <span>{file.path}</span>
          <button onClick={copy} aria-label="Copiar trecho de código">
            {copied ? <Check size={16} /> : <Copy size={16} />}{" "}
            {copied ? "Copiado" : "Copiar"}
          </button>
        </div>
        <pre tabIndex={0} aria-label={`Trecho de ${file.path}`}>
          <code>{code}</code>
        </pre>
      </div>
      <button className="text-action" onClick={() => setShowFull(!showFull)}>
        {showFull ? "Mostrar trecho selecionado" : "Explorar arquivo inteiro"}
      </button>
      <p className="source-note">
        Trecho carregado diretamente do arquivo Python deste projeto no build. A
        execução científica acontece no backend FastAPI.
      </p>
      <div className="implementation-path">
        <h2>O caminho de uma execução</h2>
        <p>
          <code>AttackConfig</code> na interface → <code>/api/run-attack</code>{" "}
          ou <code>/api/federated</code> → modelo e gradientes → otimização →
          imagens e métricas.
        </p>
        <a
          className="text-action"
          href="/api/docs"
          target="_blank"
          rel="noreferrer"
        >
          Documentação da API local <ArrowUpRight size={16} />
        </a>
      </div>
    </section>
  );
}
