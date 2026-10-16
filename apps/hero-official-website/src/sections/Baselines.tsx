import { experiments } from '../lib/experiments.ts'

type Source = 'control' | 'rerun'

const sourceGroups: { id: Source; title: string; text: string; methods: string[] }[] = [
  {
    id: 'rerun', title: 'From official code',
    text: 'Every published method runs the authors’ code or an implementation aligned with it. The rows shown for OPR, TreeLoRA and HCL predate their reruns on the official code and will be replaced by those reruns. The rows for ACE, MoRAM, OPHSD and SDFT will be added when their runs complete.',
    methods: ['opr', 'treelora', 'hcl-stable', 'ace', 'moram', 'ophsd', 'sdft'],
  },
  {
    id: 'control', title: 'Project controls',
    text: 'These are not published methods. The study’s protocol defines them as reference points.',
    methods: ['base', 'static-harness', 'seq-lora', 'uniform-replay'],
  },
]

const RERUN_NAMES: Record<string, string> = { ace: 'ACE', moram: 'MoRAM', ophsd: 'OPHSD', sdft: 'SDFT' }

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
    id: 'opr', source: 'rerun', paper: { label: 'arXiv:2605.29495', href: 'https://arxiv.org/abs/2605.29495' },
    commit: 'Yancey2024/OnPolicyReplay @ 1384d88 (read only)',
    size: 'Algorithm as written. The changes are the reward function and the training budget.',
    kept: 'Algorithm 1. After each stage the model answers every past training prompt (temperature 0.1, 512 tokens). The OPR-RU ranking fills a buffer of ⌊ρ·n⌋ records, split evenly across earlier tasks, and the buffer is rebuilt each stage. Training is plain token-level cross-entropy with a cosine schedule. ρ = 0.01, the paper’s main setting.',
    changed: [
      'Rewards come from the shared scorer instead of per-task reward functions.',
      'LoRA rank 32 at 1e-4 instead of full fine-tuning at 1e-5, with a 16,384-token ceiling instead of 2,048.',
    ],
    why: 'The paper argues that the replay filter should use the evaluation metric, and the shared scorer is that metric for all 25 tasks. One 40 GB GPU per job rules out full fine-tuning, and every weight baseline uses the same adapter.',
    aligned: 'Algorithmically yes. The authors’ code uses a linear schedule and drops the quota remainder; the paper’s text is followed instead.',
  },
  {
    id: 'treelora', source: 'rerun', paper: { label: 'ICML 2025', href: 'https://proceedings.mlr.press/v267/qian25b.html' },
    commit: 'ZinYY/TreeLoRA @ 1c7260c (read only)',
    size: 'A full re-implementation with one documented gap: the internal-node bound of Eq. 2.',
    kept: 'Layer-wise LoRA-A fingerprints (as in the authors’ code), a lower-confidence-bound bandit that picks an earlier task per layer, a tree rebuilt after every task, a regularizer with λ = 0.5 ramped as upstream does, and one LoRA carried across tasks.',
    changed: [
      'Tree depth comes from the actual Qwen3.5 backbone instead of the paper’s 64-layer setting.',
      'Descent uses the leaf bound: Eq. 2’s internal-node form has no unambiguous reading, and the authors’ code does not implement it either.',
    ],
    why: 'The model differs from the paper’s, and the typeset equation cannot be transcribed without inventing a formula.',
    aligned: 'Partly. Two re-implementation choices weaken the mechanism.',
  },
  {
    id: 'hcl-stable', source: 'rerun', paper: { label: 'arXiv:2608.19013', href: 'https://arxiv.org/abs/2608.19013' },
    commit: 'Paper, Section 3 (no code at the time of the run)',
    size: 'A full clean-room reproduction, written before the authors released their code on 30 September 2026.',
    kept: 'A frozen model with a jointly versioned harness of Task Interface, Experience Memory, Capability Map and Adaptive Router. Candidate updates stay isolated until the Continual Evaluator verifies current-task improvement, historical retention and validity. The stable profile uses the paper’s strict setting, so a candidate may not break any historical anchor that the deployed harness still solves.',
    changed: [
      'Inference. The whole harness is rendered into one system prompt. The official code makes several calls per item: task-interface structuring, workflow planning, memory selection and the final answer.',
      'Experience memory. Nothing is written automatically. The official code writes raw memory with the gold answer as feedback and consolidates abstract memory and skills with the model.',
    ],
    why: 'The paper gave no code or prompts when the run started, so the reproduction had to design its own prompts and initial artifacts.',
    aligned: 'Partly. The evaluator’s acceptance rule follows the paper; the multi-call inference and the automatic memory do not. The next release reruns the official code.',
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
  ['Independence', 'Each package is self-contained, with its own lock file. A standalone test copies every package outside the repository, blocks imports between them and runs the scorer conformance cases.'],
] as const

export function Baselines() {
  const method = (id: string) => experiments.methods.find((m) => m.id === id)
  const name = (id: string) => method(id)?.name ?? RERUN_NAMES[id] ?? id
  return (
    <section className="section" id="baselines">
      <div className="wrap experiments-content">
        <header>
          <h2 className="section-title reveal">Baselines</h2>
          <p className="experiment-lede">Every comparison method runs on self-contained code under one contract. Seven have complete runs in this release; four are being rerun on their official code. This section answers three questions: where each method’s code comes from, what was changed in it and why, and how the methods are kept comparable.</p>
        </header>

        <div className="experiment-block">
          <h3><span className="baseline-step">1</span>Where the code comes from</h3>
          <div className="baseline-sources">{sourceGroups.map((group) => (
            <article key={group.id} className="baseline-source" data-source={group.id}>
              <h4>{group.title}</h4>
              <p>{group.text}</p>
              <ul>{group.methods.map((id) => {
                const label = method(id)?.short ?? RERUN_NAMES[id] ?? id
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

        <p className="experiment-note">Every method’s <code>SOURCE.md</code> under <code>packages/baselines/</code> records the inspected upstream commit and each adaptation.</p>
      </div>
    </section>
  )
}
