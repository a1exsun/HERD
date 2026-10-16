import { useRef, useState } from 'react'
import { Figure } from '../components/Figure.tsx'
import { Edge, M, Node } from '../components/Svg.tsx'
import { useCarousel, useInView } from '../lib/hooks.ts'
import { TRACE8 } from '../lib/tasks.ts'

type Mode = 'learn' | 'serve'

const W = 360
const H = 250

function Frozen({ x, y }: { x: number; y: number }) {
  return (
    <g transform={`translate(${x} ${y})`} stroke="var(--predictor)" strokeWidth={1.5} strokeLinecap="round">
      <line x1={-6} x2={6} y1={0} y2={0} />
      <line x1={-3} x2={3} y1={-5.2} y2={5.2} />
      <line x1={-3} x2={3} y1={5.2} y2={-5.2} />
    </g>
  )
}

function Token({ x, y, label, kind }: { x: number; y: number; label: string; kind: 'x' | 'ctx' | 'y' }) {
  const w = kind === 'ctx' ? 92 : 40
  const fill = kind === 'ctx' ? 'var(--harness-soft)' : 'var(--surface)'
  const stroke = kind === 'ctx' ? 'var(--harness)' : kind === 'y' ? 'var(--deploy)' : 'var(--line-strong)'
  return (
    <g transform={`translate(${x - w / 2} ${y - 13})`}>
      <rect width={w} height={26} rx={6} fill="var(--surface)" />
      <rect width={w} height={26} rx={6} fill={fill} stroke={stroke} strokeWidth={1.3} />
      <text className="t-small" x={w / 2} y={17} textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
        {label}
      </text>
    </g>
  )
}

function NonParametric({ t, mode }: { t: number; mode: Mode }) {
  if (mode === 'serve') {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Non-parametric inference loads the harness and the frozen model">
        <Token x={46} y={70} label="x" kind="x" />
        <Edge d="M 68 70 L 106 70" flow kind="harness" />
        <Node x={170} y={70} w={120} h={48} kind="harness" active>
          <text textAnchor="middle" y={-2}>
            <M s={`H_{${t}}`} />
          </text>
          <text className="t-small" textAnchor="middle" y={15}>
            required at inference
          </text>
        </Node>
        <Edge d="M 170 96 L 170 136" flow kind="harness" />
        <Token x={150} y={152} label="context" kind="ctx" />
        <Token x={222} y={152} label="x" kind="x" />
        <Edge d="M 186 168 L 186 190" kind="plain" />
        <Node x={186} y={214} w={120} h={42} kind="predictor">
          <text textAnchor="middle" y={5}>
            <M s="θ_{0}" />
          </text>
        </Node>
        <Frozen x={232} y={214} />
        <Edge d="M 248 214 L 290 214" flow kind="deploy" />
        <Token x={318} y={214} label="ŷ" kind="y" />
      </svg>
    )
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Non-parametric continual learning evolves external state only">
      <text className="t-small" x={W / 2} y={34} textAnchor="middle">
        External-state adaptation
      </text>
      <Node x={84} y={84} w={112} h={48} kind="harness">
        <text textAnchor="middle" y={5}>
          <M s={`H_{${t - 1}}`} />
        </text>
      </Node>
      <Edge d="M 142 84 L 216 84" kind="harness" flow />
      <text className="t-small" x={180} y={72} textAnchor="middle">
        evolve
      </text>
      <Node key={t} x={276} y={84} w={112} h={48} kind="harness" active className="pop">
        <text textAnchor="middle" y={5}>
          <M s={`H_{${t}}`} />
        </text>
      </Node>
      <Node x={180} y={188} w={210} h={48} kind="predictor">
        <text textAnchor="middle" y={5}>
          <tspan className="t-label">Predictor </tspan>
          <M s="θ_{0}" />
        </text>
      </Node>
      <Frozen x={264} y={188} />
      <text className="t-small" x={W / 2} y={234} textAnchor="middle">
        frozen throughout the stream
      </text>
    </svg>
  )
}

function Hero({ t, mode }: { t: number; mode: Mode }) {
  if (mode === 'serve') {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="HERD inference runs the standalone predictor only">
        <Node x={170} y={70} w={120} h={48} kind="harness" dim>
          <text textAnchor="middle" y={-2}>
            <M s={`H_{${t}}`} />
          </text>
          <text className="t-small" textAnchor="middle" y={15}>
            not loaded
          </text>
        </Node>
        <text className="t-small" x={W / 2} y={124} textAnchor="middle">
          harness used during training only
        </text>
        <Token x={46} y={214} label="x" kind="x" />
        <Edge d="M 68 214 L 120 214" flow kind="predictor" />
        <Node x={186} y={214} w={124} h={46} kind="predictor" active>
          <text textAnchor="middle" y={5}>
            <M s={`~θ_{${t}}`} />
          </text>
        </Node>
        <Edge d="M 250 214 L 290 214" flow kind="deploy" />
        <Token x={318} y={214} label="ŷ" kind="y" />
        <text className="t-small" x={186} y={176} textAnchor="middle">
          harness knowledge internalized in parameters
        </text>
      </svg>
    )
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="HERD updates harness and predictor together">
      <text className="t-small" x={W / 2} y={34} textAnchor="middle">
        Updated harness and predictor recur together
      </text>
      <Node x={54} y={84} w={84} h={44} kind="harness">
        <text textAnchor="middle" y={5}>
          <M s={`H_{${t - 1}}`} />
        </text>
      </Node>
      <Node x={54} y={188} w={84} h={44} kind="predictor">
        <text textAnchor="middle" y={5}>
          <M s={t - 1 === 0 ? 'θ_{0}' : `~θ_{${t - 1}}`} />
        </text>
      </Node>
      <Edge d="M 98 84 C 116 84, 108 120, 126 124" kind="harness" flow />
      <Edge d="M 98 188 C 116 188, 108 152, 126 148" kind="predictor" flow />
      <Node x={180} y={136} w={104} h={84} kind="optim" band={28} active>
        <text className="t-head" textAnchor="middle" y={-24} style={{ fontSize: 12.5 }}>
          Stage {t} update
        </text>
        <text className="t-small" textAnchor="middle" y={6}>
          evolve harness,
        </text>
        <text className="t-small" textAnchor="middle" y={22}>
          constrained distillation
        </text>
      </Node>
      <Edge d="M 234 124 C 252 120, 244 84, 262 84" kind="harness" flow />
      <Edge d="M 234 148 C 252 152, 244 188, 262 188" kind="predictor" flow />
      <Node key={`h${t}`} x={306} y={84} w={84} h={44} kind="harness" active className="pop">
        <text textAnchor="middle" y={5}>
          <M s={`H_{${t}}`} />
        </text>
      </Node>
      <Node key={`p${t}`} x={306} y={188} w={84} h={44} kind="predictor" active className="pop">
        <text textAnchor="middle" y={5}>
          <M s={`~θ_{${t}}`} />
        </text>
      </Node>
    </svg>
  )
}

function Parametric({ t, mode }: { t: number; mode: Mode }) {
  if (mode === 'serve') {
    return (
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Parametric inference runs the trained predictor">
        <text className="t-small" x={W / 2} y={124} textAnchor="middle">
          no external state required
        </text>
        <Token x={46} y={214} label="x" kind="x" />
        <Edge d="M 68 214 L 120 214" flow kind="predictor" />
        <Node x={186} y={214} w={124} h={46} kind="predictor" active>
          <text textAnchor="middle" y={5}>
            <M s={`~θ_{${t}}`} />
          </text>
        </Node>
        <Edge d="M 250 214 L 290 214" flow kind="deploy" />
        <Token x={318} y={214} label="ŷ" kind="y" />
      </svg>
    )
  }
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Parametric continual learning trains the parameters">
      <text className="t-small" x={W / 2} y={34} textAnchor="middle">
        Parameter adaptation
      </text>
      <Node x={84} y={84} w={112} h={48} kind="predictor">
        <text textAnchor="middle" y={5}>
          <M s={t - 1 === 0 ? 'θ_{0}' : `~θ_{${t - 1}}`} />
        </text>
      </Node>
      <Edge d="M 142 84 L 216 84" kind="predictor" flow />
      <text className="t-small" x={180} y={72} textAnchor="middle">
        train
      </text>
      <Node key={t} x={276} y={84} w={112} h={48} kind="predictor" active className="pop">
        <text textAnchor="middle" y={5}>
          <M s={`~θ_{${t}}`} />
        </text>
      </Node>
      <g className="node" data-kind="plain" transform="translate(75 164)">
        <rect className="face" width={210} height={48} rx={10} strokeDasharray="5 4" />
        <text className="t-small" x={105} y={29} textAnchor="middle">
          optional retained training state
        </text>
      </g>
      <text className="t-small" x={W / 2} y={234} textAnchor="middle">
        each stage starts from the preceding checkpoint
      </text>
    </svg>
  )
}

const FOOT: Record<Mode, [string, string, string]> = {
  learn: [
    'External state is updated. The predictor stays frozen.',
    'The harness and the predictor are both updated and both carried forward.',
    'Parameters are updated from the preceding checkpoint.',
  ],
  serve: [
    'Inference requires the model and the harness.',
    'Inference uses the standalone predictor. The harness is not loaded.',
    'Inference uses the standalone predictor.',
  ],
}

const VIEWS: { mode: Mode; label: string }[] = [
  { mode: 'learn', label: 'Update at stage t' },
  { mode: 'serve', label: 'Inference' },
]

export function Paradigms() {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const [hover, setHover] = useState(false)
  // The loop walks through the stream in the learning view, then again in the inference view.
  const loop = useCarousel(TRACE8.length * VIEWS.length, 1900, { active: inView, paused: hover })
  const view = Math.floor(loop.step / TRACE8.length)
  const mode = VIEWS[view].mode
  const t = (loop.step % TRACE8.length) + 1

  return (
    <Figure
      ref={ref}
      label="Figure 2"
      title="Non-parametric CL, HERD, and parametric CL"
      schematic={false}
      progress={{ count: VIEWS.length, index: view, label: `${VIEWS[view].label} · stage ${t}, ${TRACE8[t - 1].name}` }}
      onAdvance={() => loop.goto(loop.step + TRACE8.length)}
      onHover={setHover}
      hint="Paused · click to switch view"
      caption={
        <>
          Storage and inference in continual learning. HERD recursively updates the inherited harness and predictor
          states, carries both forward, and deploys only the predictor.
        </>
      }
    >
      <div className="cards3">
        <div className="card">
          <div className="card-head">
            <span>(a)</span>
            <b>Non-parametric CL</b>
          </div>
          <NonParametric t={t} mode={mode} />
          <div className="card-foot">{FOOT[mode][0]}</div>
        </div>
        <div className="card" data-featured="true">
          <div className="card-head">
            <span>(b)</span>
            <b>HERD</b>
          </div>
          <Hero t={t} mode={mode} />
          <div className="card-foot">{FOOT[mode][1]}</div>
        </div>
        <div className="card">
          <div className="card-head">
            <span>(c)</span>
            <b>Parametric CL</b>
          </div>
          <Parametric t={t} mode={mode} />
          <div className="card-foot">{FOOT[mode][2]}</div>
        </div>
      </div>
    </Figure>
  )
}
