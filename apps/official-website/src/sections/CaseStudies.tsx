import { useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { ChartFigure, LineChart } from '../components/Charts.tsx'
import { Figure } from '../components/Figure.tsx'
import { vars } from '../lib/css.ts'
import { useCarousel, useInView } from '../lib/hooks.ts'

/* Appendix D of the paper: single test items followed through the stream. Every quoted output, score,
   token count and drift value is copied from the paper's case-study figure; seed 42 unless marked. */

const BUDGET = 0.05

/** One model output: who answered, whether it scored, a note, and the verbatim text. */
function Out({ who, ok, note, children }: { who: string; ok?: boolean; note?: string; children: ReactNode }) {
  return (
    <div className="cs-out" data-ok={ok === undefined ? undefined : String(ok)}>
      <div className="cs-out-head">
        <b>{who}</b>
        {ok !== undefined && <span className="cs-mark" aria-label={ok ? 'correct' : 'incorrect'}>{ok ? '✓' : '✗'}</span>}
        {note && <small>{note}</small>}
      </div>
      <pre>{children}</pre>
    </div>
  )
}

function Ask({ children, meta }: { children: ReactNode; meta?: string }) {
  return <p className="cs-ask">{children}{meta && <small> · {meta}</small>}</p>
}

function Chips<T extends string | number>({ options, value, onChange, label }: {
  options: { id: T; label: string }[]
  value: T
  onChange: (id: T) => void
  label: string
}) {
  return (
    <div className="viz-chips" role="group" aria-label={label}>
      {options.map((o) => (
        <button type="button" key={o.id} aria-pressed={value === o.id} onClick={() => onChange(o.id)}>{o.label}</button>
      ))}
    </div>
  )
}

/** A carousel figure that plays while on screen, holds under the pointer and advances on a click. */
function useScenes(count: number, ms: number) {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const [hover, setHover] = useState(false)
  const loop = useCarousel(count, ms, { active: inView, paused: hover })
  return { ref, step: loop.step, figure: { ref, onAdvance: loop.advance, onHover: setHover, hint: 'Paused · click for next' } }
}

/** Horizontal bars on a 0–1 scale; widths animate when the values change. */
function Bars({ rows }: { rows: { label: string; value: number; tone?: 'bad' | 'ghost' }[] }) {
  return (
    <div className="cs-bars">
      {rows.map((r) => (
        <div className="cs-bar" key={r.label} data-tone={r.tone}>
          <span>{r.label}</span>
          <span className="cs-bar-track"><i style={vars({ '--w': `${r.value * 100}%` })} /></span>
          <b>{r.value.toFixed(2)}</b>
        </div>
      ))}
    </div>
  )
}

/* ------------------------------------------------- (a) chain vs bare number */

const GSM_A = {
  2: [['Seq-LoRA', '10'], ['Uniform replay', '11'], ['OPR', '14'], ['TreeLoRA', '14']],
  25: [['Seq-LoRA', '26'], ['Uniform replay', '13'], ['OPR', '31'], ['TreeLoRA', '22']],
} as const
const TRACE = [
  { update: 1, d: 0.0, l: 0.0, note: 'The stage starts on the merged weights; the student has not moved, so base drift is zero.' },
  { update: 2, d: 0.192, l: 0.5, note: 'One update on one-letter C-STANCE answers lifts base drift to 0.192, almost four times the budget. The multiplier starts to rise.' },
  { update: 3, d: 0.161, l: 1.0, note: 'While the drift stays over budget, the multiplier keeps growing and pulls the student back towards the base model.' },
  { update: 4, d: 0.075, l: 1.25, note: 'Drift falls towards the budget; the multiplier is still positive.' },
  { update: 16, d: 0.048, l: 2.5, note: 'Base drift is back under 0.05; the multiplier has climbed to 2.5 and now begins to decay.' },
  { update: 32, d: 0.011, l: 0.0, note: 'The stage ends well inside the budget and the multiplier is back at zero, leaving learning alone.' },
]

function CaseA() {
  const [stage, setStage] = useState<2 | 25>(2)
  const s = useScenes(TRACE.length, 2600)
  const t = TRACE[s.step]
  return (
    <div className="cs-grid">
      <div className="cs-panel">
        <div className="viz-toolbar">
          <span className="viz-hint">GSM8K item 381</span>
          <Chips label="Stage" value={stage} onChange={setStage} options={[{ id: 2, label: 'After stage 2' }, { id: 25, label: 'After stage 25' }]} />
        </div>
        <Ask meta="gold 67">Cho hiked 14 km/h for 8 hours. Chloe hiked 9 km/h and stopped after 5 hours. How many kilometers farther did Cho hike?</Ask>
        <div className="cs-outs" key={stage}>
          <Out who="HERD" ok note={`${stage === 2 ? 240 : 226} tokens`}>… three-step derivation … 67</Out>
          <Out who="Base model" ok note="the same derivation">… three-step derivation … 67</Out>
          <p className="cs-label">The four parametric baselines: a bare number, median 3 tokens</p>
          <div className="cs-numbers">
            {GSM_A[stage].map(([who, n]) => (
              <Out key={who} who={who} ok={false}>{n}</Out>
            ))}
          </div>
        </div>
      </div>
      <Figure {...s.figure} label="Case a" title="Stage 1: base drift and its multiplier" schematic={false}
        progress={{ count: TRACE.length, index: s.step, label: `update ${t.update} of 32` }}
        note={<><b>Update {t.update}.</b> {t.note}</>}
        caption="HERD, seed 42, stage 1 (C-STANCE), at six of its 32 updates. Bars: base drift D⁰ against the 0.05 budget; dots: the multiplier λ₀.">
        <div className="cs-trace" style={vars({ '--b': `${(BUDGET / 0.2) * 100}%` })}>
          {TRACE.map((p, i) => (
            <div key={p.update} className="cs-trace-col" data-on={i === s.step || undefined} data-over={p.d > BUDGET || undefined}>
              <div className="cs-trace-area">
                <span className="cs-trace-bar" style={vars({ '--h': `${(p.d / 0.2) * 100}%` })}><em>{p.d.toFixed(3)}</em></span>
              </div>
              <div className="cs-trace-lambda">
                <span className="cs-trace-dot" style={vars({ '--l': `${(p.l / 2.5) * 100}%` })} title={`λ₀ = ${p.l.toFixed(2)}`} />
              </div>
              <small>{p.update}</small>
            </div>
          ))}
        </div>
        <p className="cs-label">Bars: D⁰ (0 to 0.2) against the dashed budget ε = 0.05. Rings in the strip below: λ₀ (0 to 2.5), now {t.l.toFixed(2)}.</p>
      </Figure>
    </div>
  )
}

/* -------------------------------------------- (b) one prompt, two conventions */

function CaseB() {
  const [item, setItem] = useState<'numglue' | 'gsm'>('numglue')
  return (
    <div className="cs-grid">
      <div className="cs-panel">
        <p className="cs-label">Stage-6 skill, routed by prefix to NumGLUE-cm prompts and seen only by the teacher</p>
        <pre className="cs-skill"><span className="cs-prefix">Solve the following math problem.</span>{'\n'}Output the final numerical answer … directly, without any explanatory text, step-by-step reasoning …</pre>
        <p className="cs-label">At test time HERD answers from its weights alone, with no skill in the prompt.</p>
      </div>
      <div className="cs-panel">
        <div className="viz-toolbar">
          <Chips label="Test item" value={item} onChange={setItem}
            options={[{ id: 'numglue', label: 'NumGLUE-cm item 23' }, { id: 'gsm', label: 'GSM8K item 381' }]} />
        </div>
        {item === 'numglue' ? (
          <div className="cs-outs" key="numglue">
            <Ask meta="gold 4320, strict match">Oliver runs at 24 m/s. How many meters in 3 minutes?</Ask>
            <Out who="HERD" ok note="5 tokens">4320</Out>
            <Out who="Base / static harness / HCL" ok={false} note="right number inside a sentence">3 minutes × 60 = 180 s, 24 × 180 = 4320 meters</Out>
          </div>
        ) : (
          <div className="cs-outs" key="gsm">
            <Ask meta="gold 67">The GSM8K item of case (a), at the same stage 6.</Ask>
            <Out who="HERD" ok note="211 tokens">… three-step derivation … 67</Out>
            <p className="cs-label">Parametric baselines: a bare number, 3 tokens</p>
            <div className="cs-numbers">
              {[['Seq-LoRA', '31'], ['OPR', '37'], ['MoRAM', '47']].map(([who, n]) => (
                <Out key={who} who={who} ok={false}>{n}</Out>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/* ------------------------------------------------- (c) transient label collapse */

const COLLAPSE = [
  { stage: 8, answer: 'C', ok: true, scores: [0.73, 0.82, 0.94] },
  { stage: 12, answer: 'neutral', ok: false, scores: [0.03, 0.05, 0.15] },
  { stage: 16, answer: 'C', ok: true, scores: [0.69, 0.76, 0.89] },
]
const HERD_PEAKS = [
  { seed: 42, peak: 0.136, lambda: 0.76, end: 0.020 },
  { seed: 43, peak: 0.119, lambda: 1.99, end: 0.023 },
  { seed: 44, peak: 0.114, lambda: 1.49, end: 0.051 },
]

function CaseC() {
  const s = useScenes(COLLAPSE.length, 3000)
  const c = COLLAPSE[s.step]
  return (
    <div className="cs-grid">
      <Figure {...s.figure} label="Case c" title="Seq-LoRA around stage 12" schematic={false}
        progress={{ count: COLLAPSE.length, index: s.step, label: `after stage ${c.stage}` }}
        note={c.stage === 12
          ? <><b>Stage 12.</b> After training on free-text Reddit summaries, Seq-LoRA answers the classification tasks in words. Mapping &ldquo;dovish&rdquo; and &ldquo;hawkish&rdquo; back to letters gives FOMC 0.82, its stage-8 score: the knowledge is intact, the format is not.</>
          : <><b>Stage {c.stage}.</b> {c.stage === 8 ? 'Before the summarization stage, Seq-LoRA answers with the option letter.' : 'Four stages later the letters are back, and none of this reaches Table 1, which is scored after stage 25.'}</>}
        caption="Seq-LoRA's strict scores on the three letter-answer tasks at stages 8, 12 and 16, and its answer on FOMC item 23 (gold C).">
        <Ask meta="gold C">Accordingly, considerable caution needed to be exercised in assessing the outlook for productivity …</Ask>
        <Out who={`Seq-LoRA · stage ${c.stage}`} ok={c.ok}>{c.answer}</Out>
        <Bars rows={[
          { label: 'C-STANCE', value: c.scores[0], tone: c.ok ? undefined : 'bad' },
          { label: 'FOMC', value: c.scores[1], tone: c.ok ? undefined : 'bad' },
          { label: 'ScienceQA', value: c.scores[2], tone: c.ok ? undefined : 'bad' },
          ...(c.stage === 12 ? [{ label: 'FOMC, semantic', value: 0.82, tone: 'ghost' as const }] : []),
        ]} />
      </Figure>
      <div className="cs-panel">
        <p className="cs-label">HERD at stage 12, all three seeds</p>
        <div className="cs-numbers">
          {HERD_PEAKS.map((p) => <Out key={p.seed} who={`seed ${p.seed}`} ok>C</Out>)}
        </div>
        <p className="cs-label">HERD&rsquo;s checkpoint probes open with the letter itself, so the same shift registers as checkpoint drift, and the multiplier pulls it back within the stage.</p>
        <div className="experiment-scroll" role="region" aria-label="HERD checkpoint drift at stage 12" tabIndex={0}>
          <table className="experiment-table cs-table">
            <thead><tr><th scope="col">Seed</th><th scope="col">D<sup>C</sup> peak</th><th scope="col">λ<sub>C</sub> peak</th><th scope="col">D<sup>C</sup> at stage end</th></tr></thead>
            <tbody>{HERD_PEAKS.map((p) => (
              <tr key={p.seed}><th scope="row">{p.seed}</th><td className="cs-over">{p.peak.toFixed(3)}</td><td>{p.lambda.toFixed(2)}</td>
                <td className={p.end > BUDGET ? 'cs-over' : undefined}>{p.end.toFixed(3)}</td></tr>
            ))}</tbody>
          </table>
        </div>
        <p className="cs-label">FOMC stays at 0.75 to 0.80. On MeetingBank, learned at stage 3, all three HERD seeds still write the ordinance title of item 65 at stage 25 (ROUGE-L 0.73), where Seq-LoRA writes &ldquo;The bill was passed.&rdquo; (0.00), without replaying any earlier answers.</p>
      </div>
    </div>
  )
}

/* -------------------------------------------------- (d) privileged context leaks */

function CaseD() {
  const [who, setWho] = useState<'sdft' | 'ophsd' | 'herd'>('sdft')
  return (
    <div className="cs-grid">
      <div className="cs-panel">
        <div className="viz-toolbar">
          <Chips label="Method" value={who} onChange={setWho}
            options={[{ id: 'sdft', label: 'SDFT' }, { id: 'ophsd', label: 'OPHSD' }, { id: 'herd', label: 'HERD' }]} />
        </div>
        <div className="cs-outs" key={who}>
          {who === 'sdft' && <>
            <Out who="SDFT · C-STANCE item 0 · stage 12" ok={false} note="cut at 5,120 tokens">B … <mark>This is an example for a response to the question:</mark>{'\n'}B{'\n'}<mark>**Thinking Process:**</mark> …</Out>
            <p className="cs-label">The teacher prompt shows a one- to three-token label and asks for &ldquo;a response of your own, including the thinking process&rdquo;, so the teacher invents one. Rollouts cut at 1,024 tokens are trained on regardless, so stopping is never supervised, and the loss settles at 0.005 to 0.008 because the teacher is an EMA of the student.</p>
          </>}
          {who === 'ophsd' && <>
            <Out who="OPHSD · GSM8K item 61 · stage 25" ok={false} note="gold 72"><mark>Following the solution plan</mark>, first note the discount applies after the subtotal, … 60 * 1.2 = 72 … gives 66.4.</Out>
            <p className="cs-label">The solve message gives the teacher a plan and a &ldquo;Reference answer (for verification only)&rdquo;; the student, whose prompt holds neither, begins to cite them. Outputs that mention a plan or reference answer rise from 0.01 at stage 1 to 0.17 at stage 25, and to 0.84 on the mathematics items, whose outputs shorten from about 1,180 to 405 tokens. Mathematics ends at 0.24 and instruction following at 0.38, against the base&rsquo;s 0.89 and 0.82.</p>
          </>}
          {who === 'herd' && <>
            <Out who="HERD · all 51,600 test outputs" ok note="0 leaks · at most 0.6% cut">no skill header, no &ldquo;Repeat your previous answer&rdquo; in any output</Out>
            <p className="cs-label">HERD&rsquo;s target is the verified answer replayed as the model&rsquo;s own turn, so there is no instruction text to copy, and its KL is scored on that answer&rsquo;s prefix rather than on the student&rsquo;s samples, where teacher and student can agree on a degenerate path.</p>
          </>}
        </div>
      </div>
      <ChartFigure title="SDFT: length and leakage compound"
        caption="Share of SDFT's test outputs cut at the 5,120-token ceiling, and share that reproduces the teacher's instruction verbatim."
        table={
          <table className="experiment-table">
            <thead><tr><th scope="col">Share</th>{[1, 4, 8, 12].map((s) => <th scope="col" key={s}>Stage {s}</th>)}</tr></thead>
            <tbody>
              <tr><th scope="row">Cut at ceiling</th>{[0.01, 0.21, 0.74, 0.94].map((v, k) => <td key={k}>{v.toFixed(2)}</td>)}</tr>
              <tr><th scope="row">Instruction copied</th>{[0, 0.23, 0.77, 0.95].map((v, k) => <td key={k}>{v.toFixed(2)}</td>)}</tr>
            </tbody>
          </table>
        }>
        <LineChart label="Share of SDFT outputs cut at the ceiling and copying the teacher instruction at stages 1, 4, 8 and 12."
          lines={[
            { id: 'cut', label: 'Cut off', color: 'var(--viz-weights)', points: [[1, 0.01], [4, 0.21], [8, 0.74], [12, 0.94]].map(([x, y]) => ({ x, y })) },
            { id: 'copy', label: 'Copied', color: 'var(--viz-context)', dashed: true, points: [[1, 0], [4, 0.23], [8, 0.77], [12, 0.95]].map(([x, y]) => ({ x, y })) },
          ]}
          x={{ min: 1, max: 12, ticks: [1, 4, 8, 12], label: 'Stage' }}
          y={{ min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1] }} />
      </ChartFigure>
    </div>
  )
}

/* ------------------------------------------------------------------ section */

const CASES: { id: string; title: string; meta: string; body: ReactNode; Visual: () => ReactNode }[] = [
  {
    id: 'a', title: 'Chain against bare number', meta: 'GSM8K · stages 1 → 25',
    body: <>The first two tasks, C-STANCE and FOMC, train on one-letter answers, and 32 cross-entropy updates on them make the short answer a habit: from stage 2 on, the four parametric baselines answer GSM8K with a bare number, median 3 tokens. The base model and HERD keep the same three-step derivation. The behaviours part at the first tokens, which is where the base anchor looks.</>,
    Visual: CaseA,
  },
  {
    id: 'b', title: 'One prompt, two conventions', meta: 'NumGLUE-cm and GSM8K · stage 6',
    body: <>NumGLUE-cm prompts differ from GSM8K&rsquo;s by one word and demand a bare number. HERD&rsquo;s stage-6 skill says so and is routed to the NumGLUE prefix alone: HERD answers 4320 on the NumGLUE item and keeps a 211-token chain on the GSM8K item, every parametric baseline answers both with a number, and the base, the static harness and HCL derive 4320 inside a sentence and score zero.</>,
    Visual: CaseB,
  },
  {
    id: 'c', title: 'Retention between the checkpoints', meta: 'FOMC and MeetingBank · stage 12',
    body: <>Some losses never reach Table 1. After stage 12, Reddit summaries with free-text answers, Seq-LoRA&rsquo;s three letter-answer tasks collapse and are back by stage 16. HERD&rsquo;s checkpoint probes catch the same shift as it happens.</>,
    Visual: CaseC,
  },
  {
    id: 'd', title: 'Privileged context leaks into the baselines', meta: 'SDFT and OPHSD',
    body: <>Both recursive self-distillation baselines give their teacher text the student never sees. SDFT collapses by leakage and length; OPHSD&rsquo;s outputs stay clean, at most 1.3% cut off, but its wording drifts toward the teacher&rsquo;s plan, and nothing pulls it back: its general probe never rises at any of the 25 stages.</>,
    Visual: CaseD,
  },
]

const GROUPS = [
  { title: 'HERD against the parametric baselines', ids: ['a', 'b', 'c'] },
  { title: 'The baselines’ privileged context', ids: ['d'] },
]

export function CaseStudies() {
  return (
    <section className="section" id="cases">
      <div className="wrap experiments-content">
        <header>
          <h2 className="section-title reveal">Case Studies</h2>
          <p className="experiment-lede">Single test items followed through the stream: where HERD keeps a behaviour the parametric baselines lose, and how the baselines&rsquo; privileged teacher context leaks into their outputs.</p>
          <p className="experiment-note">From the paper&rsquo;s Appendix D. Every quoted output is copied from the run records, seed 42 unless marked; an ellipsis marks text the paper shortens.</p>
        </header>
        {GROUPS.map((g) => (
          <div className="cs-group" key={g.title}>
            <h3>{g.title}</h3>
            {CASES.filter((c) => g.ids.includes(c.id)).map(({ id, title, meta, body, Visual }) => (
              <article className="experiment-block cs-case" key={id} id={`case-${id}`}>
                <h4><span className="cs-letter">{id}</span>{title}<small>{meta}</small></h4>
                <p className="experiment-reading">{body}</p>
                <Visual />
              </article>
            ))}
          </div>
        ))}
      </div>
    </section>
  )
}
