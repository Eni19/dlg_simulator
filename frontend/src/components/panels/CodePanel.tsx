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
      "Como decidir se uma tentativa está se aproximando das pistas recebidas? Este arquivo reúne os cálculos usados nessa comparação. A função de perda, ou loss, combina a diferença entre gradientes com uma suavização da imagem.",
    source: losses,
    anchor: "def grad_matching_loss(",
    end: "# Perda",
  },
  {
    name: "models.py",
    path: "src/models.py",
    title: "A pequena rede que aprende com as imagens",
    description:
      "O modelo é a parte da IA que transforma uma imagem em uma previsão de categoria. Aqui usamos uma rede LeNet, com camadas que processam imagens pequenas e uma saída para as 100 categorias do CIFAR-100.",
    source: model,
    anchor: "class LeNet(",
    end: "def create_lenet",
  },
  {
    name: "dlg_attack.py",
    path: "src/dlg_attack.py",
    title: "O motor do ataque",
    description:
      "Este é o passo a passo da tentativa de reconstrução: criar uma imagem inicial, calcular suas pistas, comparar e ajustar. O otimizador LBFGS escolhe os ajustes, e o programa registra a evolução da busca.",
    source: attack,
    anchor: "class DLGAttack:",
    end: "",
  },
  {
    name: "federated_flower.py",
    path: "src/federated_flower.py",
    title: "Como os participantes aprendem juntos",
    description:
      "Este arquivo organiza os participantes simulados, combina suas atualizações e define em que momento o ataque recebe acesso às pistas. É a base do cenário de aprendizado federado.",
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
        <p>
          Quer entender como a ideia virou programa? Escolha uma parte abaixo.
          Você pode explorar o laboratório sem ler código; este espaço permite ir além.
        </p>
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
        Este trecho vem dos arquivos Python do projeto. Os cálculos acontecem
        no servidor do laboratório, usando PyTorch, uma ferramenta de aprendizado de máquina.
      </p>
      <div className="implementation-path">
        <h2>O caminho de uma execução</h2>
        <p>
          Suas escolhas são enviadas ao servidor. Ele prepara as imagens,
          calcula os gradientes, executa a busca e devolve as reconstruções e as
          medidas de comparação. A rota <code>/api/experiments/…/stream</code>{" "}
          também envia o andamento real enquanto os cálculos acontecem.
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
