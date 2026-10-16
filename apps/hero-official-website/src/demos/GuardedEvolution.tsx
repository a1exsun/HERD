import { useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Figure } from '../components/Figure.tsx'
import { Tex } from '../components/Tex.tsx'
import { vars } from '../lib/css.ts'
import { useCarousel, useInView } from '../lib/hooks.ts'
import { TRACE8 } from '../lib/tasks.ts'
import { tx } from '../lib/tex.ts'

interface Proposal {
  /** Routing prefix: the longest common prefix of the task's training prompts. */
  prefix: string
  /** The skill's Format line, as written by the frozen base model. */
  format: string
  /** Size of the serialized registry once the skill is added. */
  registry: number
  /** Change in the frozen checkpoint's mean residual NLL of the verified answers in A_t (nats per token). */
  dNll: number
  /** Records of A_t shown to the proposer (at most 16) and mistakes of the frozen checkpoint (at most 4). */
  candidates: number
  mistakes: number
}

/** Illustrative proposals for the first eight task types; values demonstrate the paper's strict residual guard. */
const PROPOSALS: Proposal[] = [
  { prefix: '判断以下文本对指定对象的态度，选择一项：A.支持，B.反对，C.中立。…', format: 'Output a single character: A, B, or C.', registry: 431, dNll: 0.0023, candidates: 16, mistakes: 4 },
  { prefix: 'What is the monetary policy stance for the following text? …', format: 'Output exactly one letter (A, B, or C) representing the monetary policy stance.', registry: 874, dNll: 0.0279, candidates: 16, mistakes: 4 },
  { prefix: 'Write a summary of the following meeting transcripts. …', format: 'A single, concise sentence or short paragraph summarizing the recommendation, motion, or ordinance adopted.', registry: 1443, dNll: -0.0008, candidates: 10, mistakes: 4 },
  { prefix: '<s> ', format: 'The exact text following the last incomplete line of the code snippet, without explanations or formatting.', registry: 1830, dNll: -0.0768, candidates: 16, mistakes: 4 },
  { prefix: 'Choose an answer for the following question and give your reasons. …', format: 'The correct choice letter, then a paragraph explaining the reasoning.', registry: 2200, dNll: 0.0303, candidates: 16, mistakes: 4 },
  { prefix: 'Solve the following math problem. Question: ', format: 'The final numerical answer or entity name directly, without explanatory text or step-by-step reasoning.', registry: 2668, dNll: -0.0004, candidates: 16, mistakes: 4 },
  { prefix: 'Solve the following math problem. Question: ', format: 'The final numerical answer as a plain integer or decimal number, with no units, text, or explanation.', registry: 2768, dNll: -0.0014, candidates: 16, mistakes: 4 },
  { prefix: 'Provide a simplified version of the following paragraph in German. …', format: 'A simplified German summary of the input text in 2–4 sentences, focusing on the main event or conclusion.', registry: 3336, dNll: -0.0093, candidates: 13, mistakes: 4 },
]

const TAU = 0
const BUDGET = 24576
const RANGE = 0.1

interface Entry {
  stage: number
  proposal: Proposal
}

interface Decision {
  commit: boolean
  /** Stage whose skill the proposal replaces because the routing prefixes match. */
  replaces: number | null
  reasons: ReactNode[]
}

/** Replays the stream once. Every stage gets one proposal and one verdict. */
function decide(): Decision[] {
  const registry: Entry[] = []
  return PROPOSALS.map((proposal, stage) => {
    const replaced = registry.find((e) => e.proposal.prefix === proposal.prefix)
    const checks = { empty: proposal.format.length === 0, harm: proposal.dNll >= TAU, size: proposal.registry > BUDGET }
    const reasons: ReactNode[] = []
    if (checks.empty) reasons.push(<>the instructions are empty</>)
    if (checks.harm) reasons.push(<>the mean residual does not decrease</>)
    if (checks.size) reasons.push(<><Tex tex={tx`\mathrm{Tok}(K'_t)`} /> exceeds <Tex tex="B_K" /></>)
    const commit = reasons.length === 0
    if (commit) {
      if (replaced) registry.splice(registry.indexOf(replaced), 1)
      registry.push({ stage, proposal })
    }
    return { commit, replaces: replaced ? replaced.stage : null, reasons }
  })
}

const DECISIONS = decide()

/** The registry after the given number of settled stages. */
function registryAfter(settled: number): Entry[] {
  const registry: Entry[] = []
  for (let stage = 0; stage < settled; stage++) {
    if (!DECISIONS[stage].commit) continue
    const proposal = PROPOSALS[stage]
    const same = registry.findIndex((e) => e.proposal.prefix === proposal.prefix)
    if (same >= 0) registry.splice(same, 1)
    registry.push({ stage, proposal })
  }
  return registry
}

const signed = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(4)}`
const thousands = (v: number) => v.toLocaleString('en-US')

function HarmRow({ value }: { value: number }) {
  const pass = value < TAU
  const clamp = (v: number) => Math.max(-RANGE, Math.min(RANGE, v))
  const pct = (v: number) => 50 + (clamp(v) / RANGE) * 50
  const left = Math.min(pct(0), pct(value))
  const width = Math.abs(pct(value) - pct(0))
  return (
    <div className="guard-row" data-pass={pass}>
      <div className="guard-top">
        <span>
          <Tex tex={tx`\Delta\mathrm{NLL}`} />{' '}
          <span style={{ color: 'var(--muted)' }}>
            checkpoint with skill only on <Tex tex="A_t" />, must decrease
          </span>
        </span>
        <span className="val">{signed(value)}</span>
      </div>
      <div className="axis" aria-hidden="true">
        <span className="zero" />
        <span
          className="bar"
          style={vars({
            left: `${left}%`,
            width: `${Math.max(width, 0.6)}%`,
            '--c': pass ? 'var(--deploy)' : 'var(--optim)',
          })}
        />
        <span className="thr" style={{ left: `${pct(TAU)}%` }} />
      </div>
    </div>
  )
}

function SizeRow({ tokens }: { tokens: number }) {
  const pass = tokens <= BUDGET
  return (
    <div className="guard-row" data-pass={pass}>
      <div className="guard-top">
        <span>
          <Tex tex={tx`\mathrm{Tok}(K'_t)`} /> <span style={{ color: 'var(--muted)' }}>serialized registry, budget</span>{' '}
          <Tex tex="B_K" />
        </span>
        <span className="val">
          {thousands(tokens)} / {thousands(BUDGET)}
        </span>
      </div>
      <div className="meter-bar" aria-hidden="true">
        <i style={vars({ width: `${Math.min(100, (tokens / BUDGET) * 100)}%`, '--c': pass ? 'var(--deploy)' : 'var(--optim)' })} />
      </div>
    </div>
  )
}

export function GuardedEvolution() {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const [hover, setHover] = useState(false)
  // Two scenes per stage: the proposal under evaluation, then the stage after it has closed.
  const loop = useCarousel(PROPOSALS.length * 2, 3200, { active: inView, paused: hover })
  const stage = Math.floor(loop.step / 2)
  const closed = loop.step % 2 === 1

  const proposal = PROPOSALS[stage]
  const { commit, replaces, reasons } = DECISIONS[stage]
  const registry = registryAfter(closed ? stage + 1 : stage)
  // The newest committed skill is last, and its record holds the registry's serialized size.
  const shown = registry.at(-1)?.proposal.registry ?? 0

  return (
    <Figure
      ref={ref}
      label="Figure 4"
      title="Guarded evolution"
      schematic
      progress={{ count: PROPOSALS.length, index: stage, label: `Stage ${stage + 1} · ${TRACE8[stage].name}` }}
      onAdvance={loop.advance}
      onHover={setHover}
      hint="Paused · click for next step"
      legend={
        <span className="fig-params">
          <Tex tex={tx`\Delta\mathrm{NLL} < 0`} /> nats per token, <Tex tex={tx`B_K = 24{,}576`} /> tokens
        </span>
      }
      caption={
        <>
          Illustrative skill proposals and residual changes. The frozen base writes each skill; the checkpoint
          tests it without replaying the answer. Only strict improvement within the registry budget commits a skill.
          These values demonstrate the rule and are not experimental measurements.
        </>
      }
    >
      <div className="ge">
        <div className="spec">
          <div className="spec-title">
            <div>
              Skill registry <Tex tex={`K_{${closed ? stage + 1 : stage}}`} />
            </div>
            <span>
              {thousands(shown)} / {thousands(BUDGET)} tok
            </span>
          </div>
          <div className="meter">
            <div className="meter-bar">
              <i style={vars({ width: `${Math.min(100, (shown / BUDGET) * 100)}%` })} />
            </div>
          </div>
          <ul className="spec-list">
            {registry.length === 0 && (
              <li>
                <span>Empty</span>
                No skills yet. Unmatched prompts run without a skill.
              </li>
            )}
            {registry.map((e) => (
              <li
                key={e.stage}
                data-fresh={closed && e.stage === stage ? 'true' : undefined}
                style={!closed && replaces === e.stage ? { opacity: 0.45 } : undefined}
              >
                <span>{TRACE8[e.stage].name}</span>
                {e.proposal.format}
              </li>
            ))}
            {!closed && (
              <li key={`cand-${stage}`} data-new="true">
                <span>{TRACE8[stage].name}</span>
                {proposal.format}
              </li>
            )}
          </ul>
          <div className="trail" aria-label="Routing prefix of the proposed skill">
            <span>Prefix</span>
            <code style={{ fontSize: '0.74rem', overflowWrap: 'anywhere' }}>{JSON.stringify(proposal.prefix)}</code>
          </div>
        </div>

        <div className="guards">
          <div className="guard-row">
            <div className="guard-top">
              <span>
                Proposer <Tex tex={tx`\theta_0`} />{' '}
                <span style={{ color: 'var(--muted)' }}>
                  reads {proposal.candidates} candidates and {proposal.mistakes} mistakes, one greedy call
                </span>
              </span>
            </div>
          </div>
          <HarmRow value={proposal.dNll} />
          <SizeRow tokens={proposal.registry} />
          <div className="verdict" data-v={commit ? 'commit' : 'reject'}>
            <b>
              Stage {stage + 1} ({TRACE8[stage].name}): {commit ? 'committed' : 'rejected'}
            </b>
            <span>
              {commit ? (
                <>
                  The skill is non-empty, fits the budget, and lowers the mean residual, so{' '}
                  <Tex tex="K_t = K'_t" />.
                  {replaces !== null && <> It replaces the {TRACE8[replaces].name} skill, which has the same prefix.</>}
                </>
              ) : (
                <>
                  <Tex tex="K_t = K_{t-1}" />, because{' '}
                  {reasons.map((reason, i) => (
                    <span key={i}>
                      {i > 0 && ' and '}
                      {reason}
                    </span>
                  ))}
                  . No retry is made.
                </>
              )}
            </span>
          </div>
        </div>
      </div>
    </Figure>
  )
}
