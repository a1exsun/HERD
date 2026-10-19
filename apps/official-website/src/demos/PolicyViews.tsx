import { useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Figure } from '../components/Figure.tsx'
import { Tex } from '../components/Tex.tsx'
import { tx } from '../lib/tex.ts'
import { vars } from '../lib/css.ts'
import { useFrameLoop, useInView, useReducedMotion } from '../lib/hooks.ts'

type Dist = number[]

function kl(p: Dist, q: Dist): number {
  return p.reduce((sum, pi, i) => sum + (pi > 0 ? pi * Math.log(pi / q[i]) : 0), 0)
}

/** Geometric interpolation between two next-token distributions. */
function blend(a: Dist, b: Dist, w: number): Dist {
  const raw = a.map((ai, i) => Math.exp((1 - w) * Math.log(ai) + w * Math.log(b[i])))
  const z = raw.reduce((x, y) => x + y, 0)
  return raw.map((v) => v / z)
}

/** Largest blend weight whose divergence from the anchor stays at the given level. */
function weightAt(anchor: Dist, pull: Dist, level: number): number {
  let lo = 0
  let hi = 1
  if (kl(anchor, blend(anchor, pull, 1)) <= level) return 1
  for (let i = 0; i < 30; i++) {
    const mid = (lo + hi) / 2
    if (kl(anchor, blend(anchor, pull, mid)) <= level) lo = mid
    else hi = mid
  }
  return lo
}

/** The teacher sees the skill and the verified answer replayed, so its distribution is sharp on the answer token. */
const CURRENT = {
  prefix: '“…Which closing is correct for a letter?”',
  tokens: ['B', 'A', 'Thanks', 'The', 'other'],
  teacher: [0.9, 0.04, 0.03, 0.02, 0.01],
  start: [0.34, 0.3, 0.2, 0.1, 0.06],
}

/** An FOMC probe opens with the choice letter; drift pulls it towards the label word instead. */
const HISTORY = {
  prefix: '“…monetary policy stance for the following text?”',
  tokens: ['A', 'B', 'C', 'hawkish', 'other'],
  anchor: [0.06, 0.84, 0.06, 0.02, 0.02],
  pull: [0.1, 0.4, 0.1, 0.25, 0.15],
}

/** A GSM8K probe opens with reasoning; drift pulls it towards a bare number. */
const GENERAL = {
  prefix: '“…How many apples are left?”',
  tokens: ['To', 'First', 'Let', '42', 'other'],
  anchor: [0.5, 0.24, 0.18, 0.03, 0.05],
  pull: [0.25, 0.12, 0.1, 0.45, 0.08],
}

/** Drift inherited from earlier stages, present before this stage begins. */
const INHERITED = 0.12

interface BarsProps {
  tokens: string[]
  target: Dist
  student: Dist
  color: string
  label: string
}

function Bars({ tokens, target, student, color, label }: BarsProps) {
  const W = 330
  const H = 150
  const base = 112
  const step = W / tokens.length
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={{ width: '100%' }}>
      <line x1={0} x2={W} y1={base} y2={base} stroke="var(--line-strong)" strokeWidth={1} />
      {tokens.map((tok, i) => {
        const x = i * step + step / 2
        const ht = target[i] * 100
        const hs = student[i] * 100
        return (
          <g key={tok}>
            <rect x={x - 19} y={base - ht} width={17} height={ht} rx={3} fill={color} />
            <rect
              x={x + 2}
              y={base - hs}
              width={17}
              height={hs}
              rx={3}
              fill="var(--optim)"
              style={{ transition: 'y 0.12s linear, height 0.12s linear' }}
            />
            <text className="t-small" x={x} y={base + 17} textAnchor="middle" style={{ fontSize: 11 }}>
              {tok}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

interface RowProps {
  kicker: string
  color: string
  title: ReactNode
  prefix: string
  tokens: string[]
  target: Dist
  student: Dist
  targetName: ReactNode
  measure: ReactNode
  value: number
  budget?: number
  scale: number
}

function Row({ kicker, color, title, prefix, tokens, target, student, targetName, measure, value, budget, scale }: RowProps) {
  const over = budget !== undefined && value > budget
  return (
    <div className="pv-row">
      <div className="pv-info">
        <span className="kicker" style={vars({ '--kicker': color })}>
          <i />
          {kicker}
        </span>
        <b>{title}</b>
        <span className="pv-prefix">prefix {prefix}</span>
        <div className="legend">
          <span style={vars({ '--c': color })}>
            <i />
            <span>{targetName}</span>
          </span>
          <span style={vars({ '--c': 'var(--optim)' })}>
            <i />
            <span>student <Tex tex="p^S" /></span>
          </span>
        </div>
      </div>
      <Bars
        tokens={tokens}
        target={target}
        student={student}
        color={color}
        label={`Next-token distributions for the ${kicker} stream`}
      />
      <div className="readout">
        <div className="readout-top">
          <span>{measure}</span>
          <b style={over ? { color: 'var(--optim)' } : undefined}>{value.toFixed(3)}</b>
        </div>
        <div className="gauge">
          <i style={vars({ width: `${Math.min(100, (value / scale) * 100)}%`, '--c': over ? 'var(--optim)' : color })} />
          {budget !== undefined && <u style={{ left: `${Math.min(100, (budget / scale) * 100)}%` }} />}
        </div>
        <div className="readout-top">
          <span>
            {budget === undefined
              ? 'distillation objective'
              : over
                ? 'budget exceeded'
                : `slack ${(budget - value).toFixed(3)}`}
          </span>
          {budget !== undefined && <span>budget {budget.toFixed(2)}</span>}
        </div>
      </div>
    </div>
  )
}

const EPS_C = 0.05
const EPS_G = 0.05
const CAP_C = weightAt(HISTORY.anchor, HISTORY.pull, 0.9 * EPS_C)
const CAP_G = weightAt(GENERAL.anchor, GENERAL.pull, 0.9 * EPS_G)

/** One training stage takes this long on screen, then the result is held before the scenario changes. */
const RUN_MS = 9000
const HOLD_MS = 2600

const SCENARIOS = [
  { guarded: true, label: 'With drift budgets' },
  { guarded: false, label: 'Without drift budgets' },
]

export function PolicyViews() {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const reduced = useReducedMotion()
  const [view, setView] = useState({ scenario: 0, ms: 0 })

  useFrameLoop(inView && !reduced, (dt) => {
    setView((v) =>
      v.ms + dt >= RUN_MS + HOLD_MS
        ? { scenario: (v.scenario + 1) % SCENARIOS.length, ms: 0 }
        : { ...v, ms: v.ms + dt },
    )
  })

  const next = () => setView((v) => ({ scenario: (v.scenario + 1) % SCENARIOS.length, ms: 0 }))

  const { guarded, label } = SCENARIOS[view.scenario]
  // Without animation the figure shows the end of the stage, where the two scenarios differ most.
  const u = reduced ? 1 : Math.min(1, view.ms / RUN_MS)

  // Without constraints the student follows the task gradient on every stream.
  const wC = guarded ? Math.min(u * 0.9, CAP_C) : u * 0.9
  const rawG = INHERITED + u * 0.8
  const wG = guarded ? Math.min(rawG, CAP_G) : rawG
  // Holding the anchors costs a little plasticity on the current task.
  const held = guarded && (u * 0.9 > CAP_C || rawG > CAP_G)
  const wH = held ? Math.min(u, 0.55 + 0.35 * u) : u

  const sH = blend(CURRENT.start, CURRENT.teacher, wH)
  const sC = blend(HISTORY.anchor, HISTORY.pull, wC)
  const sG = blend(GENERAL.anchor, GENERAL.pull, wG)

  const epsC = EPS_C
  const epsG = EPS_G

  return (
    <Figure
      ref={ref}
      label="Figure 7"
      title="Teacher and anchors"
      progress={{ count: SCENARIOS.length, index: view.scenario, label: `${label} · ${Math.round(u * 100)}% of updates` }}
      onAdvance={next}
      hint="Click to switch scenario"
      caption={
        <>
          Teacher, anchor, and student next-token distributions on the current task, a checkpoint probe and a general probe
          during one stage, with and without the drift budgets. On the current task both views share a prefix of the
          verified answer; on probes, a prefix of the anchor&rsquo;s own greedy answer. Divergences are computed from
          the plotted distributions; the budgets match the experiments, and the distributions are illustrative.
        </>
      }
    >
      <div className="pv">
        <Row
          kicker="Current task"
          color="var(--harness)"
          title={<>Distillation teacher</>}
          prefix={CURRENT.prefix}
          tokens={CURRENT.tokens}
          target={CURRENT.teacher}
          student={sH}
          targetName={
            <>
              teacher <Tex tex="p^H" />, checkpoint with skill and replayed answer
            </>
          }
          measure={<Tex tex={tx`D_{\mathrm{KL}}(p^H \parallel p^S)`} />}
          value={kl(CURRENT.teacher, sH)}
          scale={0.8}
        />
        <Row
          kicker="Checkpoint probe"
          color="var(--guard)"
          title={<>Checkpoint anchor</>}
          prefix={HISTORY.prefix}
          tokens={HISTORY.tokens}
          target={HISTORY.anchor}
          student={sC}
          targetName={
            <>
              anchor <Tex tex="p^C" />, checkpoint without harness
            </>
          }
          measure={<Tex tex={tx`D_{\mathrm{KL}}(p^C \parallel p^S)`} />}
          value={kl(HISTORY.anchor, sC)}
          budget={epsC}
          scale={0.25}
        />
        <Row
          kicker="General probe"
          color="var(--predictor)"
          title={<>Base anchor</>}
          prefix={GENERAL.prefix}
          tokens={GENERAL.tokens}
          target={GENERAL.anchor}
          student={sG}
          targetName={
            <>
              anchor <Tex tex="p^0" />, immutable base model
            </>
          }
          measure={<Tex tex={tx`D_{\mathrm{KL}}(p^0 \parallel p^S)`} />}
          value={kl(GENERAL.anchor, sG)}
          budget={epsG}
          scale={0.25}
        />
      </div>
    </Figure>
  )
}
