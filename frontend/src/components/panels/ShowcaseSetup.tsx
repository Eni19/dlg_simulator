import { ArrowRight, ArrowUpRight, Check } from "lucide-react";
import { Card, CardHeader } from "@/components/ui/Card";
import { Toggle } from "@/components/ui/Toggle";
import type { AttackConfig } from "@/lib/types";
import { FULL_LAB_URL, SHOWCASE_SAMPLES, showcaseConfig, showcaseIterations } from "@/lib/showcase";

export function ShowcaseSetup({ config, onChange, onStart, isRunning }: {
  config: AttackConfig; onChange: (config: AttackConfig) => void;
  onStart?: () => void; isRunning: boolean;
}) {
  const selected = config.imageIndices ?? [SHOWCASE_SAMPLES[0].id];
  const toggleSample = (id: number) => {
    const next = selected.includes(id) ? selected.filter(value => value !== id) : [...selected, id];
    if (next.length >= 1 && next.length <= 2) onChange(showcaseConfig({ ...config, imageIndices: next }));
  };
  return <div className="space-y-5">
    <Card>
      <CardHeader title="Uma primeira reconstrução, aqui mesmo"
        subtitle="Você está no showcase online: uma demonstração com cálculos reais e ajustes preparados para explorar no navegador. Comece com uma imagem ou experimente duas juntas." />
      <fieldset>
        <legend className="mb-3 text-sm font-medium">Escolha uma ou duas imagens</legend>
        <div className="showcase-samples">
          {SHOWCASE_SAMPLES.map(sample => <button key={sample.id} type="button"
            aria-pressed={selected.includes(sample.id)}
            disabled={selected.length === 2 && !selected.includes(sample.id)}
            aria-describedby="showcase-selection-help"
            onClick={() => toggleSample(sample.id)}>
            <img src={`data:image/png;base64,${sample.image}`} alt={`Exemplo público: ${sample.name}`} />
            <span className="showcase-sample-name">{sample.name}{selected.includes(sample.id) && <Check size={15} aria-hidden="true" />}</span>
          </button>)}
        </div>
        <p className="mt-3 text-sm" role="status">{selected.length === 1 ? "1 imagem selecionada" : "2 imagens selecionadas · um único lote"}</p>
        <p id="showcase-selection-help" className="mt-2 text-sm text-graphite-400">
          {selected.length === 2 ? "Para trocar uma imagem, desmarque uma das selecionadas primeiro." : "Selecione mais uma imagem para formar um lote, também chamado de batch."}
        </p>
      </fieldset>
      <div className="mt-6">
        <p id="showcase-iterations" className="mb-2 text-sm font-medium">Iterações de ajuste</p>
        <Toggle options={showcaseIterations(selected.length).map(value => ({ value: String(value), label: String(value) }))} value={String(config.numIters)}
          onChange={value => onChange(showcaseConfig({ ...config, numIters: Number(value) }))}
          label="Iterações de ajuste" />
        <p className="mt-3 text-sm text-graphite-400">
          Mais iterações dão mais oportunidades de ajuste. Nesta experiência,
          usamos até duas imagens por lote, CPU e nenhuma defesa adicional.
          Com duas imagens, as pistas se misturam e a reconstrução pode ser mais difícil.
        </p>
        {selected.length === 2 && <p className="mt-3 text-sm text-graphite-400">
          Para dar mais espaço a esse desafio, você também pode escolher 100 ou 200
          iterações. A espera pode aumentar, sem garantia de uma reconstrução melhor.
        </p>}
      </div>
      {onStart && <button className="primary-action showcase-start" type="button" onClick={onStart} disabled={isRunning}>
        {isRunning ? "Em execução" : "Iniciar experimento"} <ArrowRight size={16} />
      </button>}
    </Card>
    <section className="showcase-extension" aria-labelledby="full-lab-heading">
      <h2 id="full-lab-heading">O próximo passo é no seu computador</h2>
      <p>Quer ir além do showcase? A versão completa permite lotes de até 8 imagens
        e muito mais liberdade para configurar os experimentos.</p>
      <p>Ela exige mais processamento e memória: o tempo de execução depende do
        seu hardware, do tamanho do lote e dos ajustes escolhidos. Você pode usar
        CPU ou uma GPU NVIDIA compatível. O código e o passo a passo estão no GitHub.</p>
      <ul className="showcase-features">
        {[
          ["Ataque em aprendizado federado", "Explore como o ataque tenta reconstruir imagens quando vários participantes aprendem juntos."],
          ["CPU, GPU e lotes de até 8 imagens", "Escolha o dispositivo e quantas imagens reconstruir juntas."],
          ["Defesas e ajustes avançados", "Configure iterações, pesos do ataque e defesas como ruído, menos gradientes e poda."],
        ].map(([title, description], index) => <li key={title}>
          <button disabled type="button" aria-describedby={`full-feature-${index}`}>
            <span>{title}</span><small>Versão completa</small>
          </button>
          <p id={`full-feature-${index}`}>{description}</p>
        </li>)}
      </ul>
      <a className="text-action" href={FULL_LAB_URL} target="_blank" rel="noreferrer">
        Abrir a versão completa no GitHub <ArrowUpRight size={16} />
      </a>
    </section>
  </div>;
}
