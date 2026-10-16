import { experiments } from '../lib/experiments.ts'

type Source = 'control' | 'implementation'

const sourceGroups: { id: Source; title: string; text: string; methods: string[] }[] = [
  {
    id: 'implementation', title: 'Published methods',
    text: 'The release contains all seven published comparison methods. ACE, MoRAM, OPR, TreeLoRA and OPHSD include upstream code or adaptations; SDFT and HCL-Stable use local implementations. The main results follow the paper.',
    methods: ['opr', 'treelora', 'hcl-stable', 'ace', 'moram', 'ophsd', 'sdft'],
  },
  {
    id: 'control', title: 'Project controls',
    text: 'These are not published methods. The study’s protocol defines them as reference points.',
    methods: ['base', 'static-harness', 'seq-lora', 'uniform-replay'],
  },
]

const METHOD_NAMES: Record<string, string> = { ace: 'ACE', moram: 'MoRAM', ophsd: 'OPHSD', sdft: 'SDFT' }

interface Change {
  id: string
  source: Source
  paper: { label: string; href: string }
  commit: string
  size: string
  kept: string
  changed: string[]
  why: string
  aligned: string
}

const CHANGES: Change[] = [
  {
    id: 'opr', source: 'implementation', paper: { label: 'arXiv:2605.29495', href: 'https://arxiv.org/abs/2605.29495' },
    commit: 'Yancey2024/OnPolicyReplay @ 1384d88',
    size: 'The branch bundles the upstream reference and an adapted official runner.',
    kept: 'OPR-RU generates responses for earlier tasks and ranks them with the shared scorer, with replay ratio 0.01 and equal task quotas.',
    changed: ['The runner maps the 25-task stream to the official data layout and uses the shared model, scorer and training budget.'],
    why: 'The adaptations place replay under the same TRACE+17 evaluation protocol.',
    aligned: 'The runnable adaptation is in opr/official; edits relative to opr/upstream carry a [HERO] marker.',
  },
  {
    id: 'treelora', source: 'implementation', paper: { label: 'ICML 2025', href: 'https://proceedings.mlr.press/v267/qian25b.html' },
    commit: 'ZinYY/TreeLoRA @ 1c7260c',
    size: 'The branch bundles the upstream KD tree and an adapted stage runner.',
    kept: 'The upstream tree, task fingerprints and regularized training step, with regularizer weight 0.5 and one adapter carried across stages.',
    changed: ['The runner uses Qwen3.5-4B and the shared update budget, and seals the adapter, optimizer, fingerprints and tree after each stage.'],
    why: 'The stage runner integrates the method with the common stream and evaluation schedule.',
    aligned: 'Focused tests compare the local training loop with a literal upstream transcription and check upstream file hashes.',
  },
  {
    id: 'hcl-stable', source: 'implementation', paper: { label: 'arXiv:2608.19013', href: 'https://arxiv.org/abs/2608.19013' },
    commit: 'Local implementation of the paper’s stable profile',
    size: 'A frozen model with a versioned Task Interface, Experience Memory, Capability Map and Adaptive Router.',
    kept: 'One candidate per component per stage; commit only with validation gain of at least 0.01 and no broken historical anchor.',
    changed: ['The components are rendered into one system prompt and served by the local backend.'],
    why: 'The stable profile evaluates harness changes under a bounded context and historical-retention check.',
    aligned: 'This is the branch’s local stable-profile implementation; it is not a claim of identical upstream inference.',
  },
]

const ALIGNMENT = [
  ['Model', 'Qwen3.5-4B at one pinned revision with a checked weights hash. By default the same model also fills every role inside ACE and HCL.'],
  ['Benchmark', 'TRACE+17: the 25 tasks in canonical order, with the same train, validation and test splits built with seed 42.'],
  ['Data isolation', 'Methods learn from the train split only. Validation gold reaches only evaluator callbacks, and test gold reaches only the scorer.'],
  ['Prompts and decoding', 'The official chat template with thinking off, greedy decoding, the union of the model and tokenizer end tokens, 5,120 new tokens and a 16,384-token input ceiling. Nothing is truncated: over-long inputs score zero, and over-long training records are excluded and recorded.'],
  ['Scorer', 'One pinned evaluator, copied byte-for-byte into every package together with the Apache-2.0 IFEval verifiers. Its tokenizer assets are hash-checked, conformance tests run against the canonical evaluator, and every reported number is recomputed from item scores.'],
  ['Evaluation schedule', 'The same sparse schedule for every method: full panels after tasks 4, 8, 16 and 25, the first eight tasks after 12 and 20, and otherwise the current task plus a 200-item general probe.'],
  ['Stage 0', 'Every learned method starts from the unchanged model’s measured predictions.'],
  ['Training budget', 'Every weight method gets 32 updates of 32 records per stage, on the same text-backbone linear layers, with LoRA rank 32 and alpha 64 unless the method defines its own adapter. HERD uses the same rank, learning rate and update budget.'],
  ['Matched state', 'Uniform replay holds exactly HERD’s residual-memory budget.'],
  ['Independence', 'Nine packages have standalone environments and scorer tests. OPR and OPHSD bundle the shared hero_baselines package into their jobs.'],
] as const

export function Baselines() {
  const method = (id: string) => experiments.methods.find((m) => m.id === id)
  const name = (id: string) => method(id)?.name ?? METHOD_NAMES[id] ?? id
  return (
    <section className="section" id="baselines">
      <div className="wrap experiments-content">
        <header>
          <h2 className="section-title reveal">Baselines</h2>
          <p className="experiment-lede">The paper compares eleven methods under one evaluation protocol. This section describes the implementations included in the arXiv branch, their adaptations and the shared evaluation settings.</p>
        </header>

        <div className="experiment-block">
          <h3><span className="baseline-step">1</span>Where the code comes from</h3>
          <div className="baseline-sources">{sourceGroups.map((group) => (
            <article key={group.id} className="baseline-source" data-source={group.id}>
              <h4>{group.title}</h4>
              <p>{group.text}</p>
              <ul>{group.methods.map((id) => {
                const label = method(id)?.short ?? METHOD_NAMES[id] ?? id
                return <li key={id}>{CHANGES.some((c) => c.id === id) ? <a href={`#baseline-${id}`}>{label}</a> : label}</li>
              })}</ul>
            </article>
          ))}</div>
        </div>

        <div className="experiment-block">
          <h3><span className="baseline-step">2</span>What was changed, and why</h3>
          <p className="experiment-reading">For each published method: how much of the original was changed, what stayed, why each change was needed, and whether the code still matches the original.</p>
          <div className="baseline-list">{CHANGES.map((c) => (
            <article className="baseline-card" key={c.id} id={`baseline-${c.id}`}>
              <header>
                <span className="eyebrow">{sourceGroups.find((g) => g.id === c.source)!.title}</span>
                <h4>{name(c.id)}</h4>
                <p className="experiment-note"><a href={c.paper.href} target="_blank" rel="noreferrer">{c.paper.label}</a> · <code>{c.commit}</code></p>
              </header>
              <dl>
                <div><dt>Size of change</dt><dd>{c.size}</dd></div>
                <div><dt>Kept from the original</dt><dd>{c.kept}</dd></div>
                <div><dt>Changed</dt><dd><ul>{c.changed.map((line) => <li key={line}>{line}</li>)}</ul></dd></div>
                <div><dt>Why it had to change</dt><dd>{c.why}</dd></div>
                <div data-reading="true"><dt>Still aligned with the original?</dt><dd>{c.aligned}</dd></div>
              </dl>
            </article>
          ))}</div>
        </div>

        <div className="experiment-block">
          <h3><span className="baseline-step">3</span>How the baselines are aligned</h3>
          <dl className="baseline-shared">{ALIGNMENT.map(([term, text]) =>
            <div key={term}><dt>{term}</dt><dd>{text}</dd></div>)}</dl>
        </div>

        <p className="experiment-note">Paper Appendix B describes OPR and TreeLoRA as paper-based implementations;
          this branch bundles upstream code. Dataset v1.1 also resamples NumGLUE-cm training and validation
          records and requires new runs; the displayed paper results do not establish results for that revision.</p>
        <p className="experiment-note">See <code>packages/baselines/README.md</code>, each runner and its bundled upstream metadata for implementation details.</p>
      </div>
    </section>
  )
}
