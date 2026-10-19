import { useMemo, useRef, useState } from 'react'
import { Figure } from '../components/Figure.tsx'
import { Edge, M } from '../components/Svg.tsx'
import { Tex } from '../components/Tex.tsx'
import { tx } from '../lib/tex.ts'
import { vars } from '../lib/css.ts'
import { useCarousel, useInView } from '../lib/hooks.ts'
import { mulberry32 } from '../lib/rng.ts'

type Matrix = number[][]

const N = 6
const R = 2
const CELL = 24
const STEPS = ['Initialization', 'Training', 'Merge', 'Next stage', 'Deployment']

function randomMatrix(rows: number, cols: number, rand: () => number, scale = 1): Matrix {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => (rand() * 2 - 1) * scale))
}

function product(b: Matrix, a: Matrix): Matrix {
  return b.map((row) => a[0].map((_, j) => row.reduce((sum, v, k) => sum + v * a[k][j], 0)))
}

interface GridProps {
  x: number
  y: number
  values: Matrix
  color: string
  /** Optional second layer, used to tint a merged matrix with its update. */
  overlay?: Matrix
  overlayColor?: string
  hidden?: boolean
  dim?: boolean
  pop?: boolean
  outline?: string
}

function Grid({ x, y, values, color, overlay, overlayColor, hidden, dim, pop, outline }: GridProps) {
  const rows = values.length
  const cols = values[0].length
  return (
    <g
      transform={`translate(${x} ${y})`}
      opacity={hidden ? 0 : dim ? 0.3 : 1}
      style={{ transition: 'opacity 0.5s' }}
    >
      <g className={pop ? 'pop' : undefined}>
        <rect width={cols * CELL} height={rows * CELL} fill="var(--surface)" />
        {values.map((row, i) =>
          row.map((v, j) => (
            <g key={`${i}-${j}`}>
              <rect
                x={j * CELL}
                y={i * CELL}
                width={CELL}
                height={CELL}
                fill={color}
                fillOpacity={Math.min(0.9, Math.abs(v))}
                style={{ transition: 'fill-opacity 1.1s ease' }}
              />
              {overlay && (
                <rect
                  x={j * CELL}
                  y={i * CELL}
                  width={CELL}
                  height={CELL}
                  fill={overlayColor}
                  fillOpacity={Math.min(0.85, Math.abs(overlay[i][j]))}
                  style={{ transition: 'fill-opacity 1.1s ease' }}
                />
              )}
              <rect
                x={j * CELL}
                y={i * CELL}
                width={CELL}
                height={CELL}
                fill="none"
                stroke="var(--line-strong)"
                strokeWidth={0.8}
              />
            </g>
          )),
        )}
        <rect
          width={cols * CELL}
          height={rows * CELL}
          fill="none"
          stroke={outline ?? 'var(--ink-2)'}
          strokeWidth={1.6}
          rx={2}
        />
      </g>
    </g>
  )
}

function Snow({ x, y }: { x: number; y: number }) {
  return (
    <g transform={`translate(${x} ${y})`} stroke="var(--predictor)" strokeWidth={1.8} strokeLinecap="round">
      <line x1={-7} x2={7} y1={0} y2={0} />
      <line x1={-3.5} x2={3.5} y1={-6} y2={6} />
      <line x1={-3.5} x2={3.5} y1={6} y2={-6} />
    </g>
  )
}

export function LoraMerge() {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const [hover, setHover] = useState(false)
  const loop = useCarousel(STEPS.length, 3600, { active: inView, paused: hover })
  const k = loop.step

  const data = useMemo(() => {
    const rand = mulberry32(42)
    const w = randomMatrix(N, N, rand, 0.75)
    const a = randomMatrix(R, N, rand, 0.9)
    const b = randomMatrix(N, R, rand, 0.9)
    const aNext = randomMatrix(R, N, rand, 0.9)
    const zeroB: Matrix = Array.from({ length: N }, () => Array.from({ length: R }, () => 0))
    const delta = product(b, a)
    return { w, a, b, aNext, zeroB, delta }
  }, [])

  const inherited = k >= 3
  const base = { x: 60, y: 118 }
  const bPos = { x: 286, y: 118 }
  const aPos = { x: 372, y: 166 }
  const dPos = { x: 606, y: 118 }
  const mPos = { x: 836, y: 118 }

  const notes = [
    <>
      <b>Initialization.</b> The inherited dense matrix <Tex tex={tx`\overline W_{t-1}`} /> is frozen. New factors{' '}
      <Tex tex={tx`\phi=(A,B)`} /> are initialized with <Tex tex="B = 0" />, so the effective update is zero.
    </>,
    <>
      <b>Training.</b> The distillation loss and the multiplier-weighted drift terms backpropagate into{' '}
      the adapter <Tex tex={tx`\phi`} /> only. Teacher and anchor targets stop gradients.
    </>,
    <>
      <b>Merge.</b> The final bf16 low-rank update is merged into the inherited weights:{' '}
      <Tex tex={tx`\overline W_t = \overline W_{t-1} + \tfrac{\alpha}{r} BA`} />.
    </>,
    <>
      <b>Next stage.</b> Stage <Tex tex="t+1" /> freezes the merged checkpoint and initializes new factors with a zero
      effective update.
    </>,
    <>
      <b>Deployment.</b> The merged weights <Tex tex={tx`\bar\theta_t`} /> are deployed without the harness. Their
      drift from both anchors is measured again and recorded with the stage.
    </>,
  ]

  const bValues = k === 0 || k >= 3 ? data.zeroB : data.b
  const aValues = inherited ? data.aNext : data.a
  const wLabel = inherited ? 't' : 't−1'

  return (
    <Figure
      ref={ref}
      label="Figure 8"
      title="LoRA update, merge, and inheritance"
      note={notes[k]}
      progress={{ count: STEPS.length, index: k, label: `${k + 1} · ${STEPS[k]}` }}
      onAdvance={loop.advance}
      onHover={setHover}
      hint="Paused · click for next step"
      legend={
        <div className="legend">
          <span style={vars({ '--c': 'var(--predictor)' })}>
            <i />
            Frozen weights
          </span>
          <span style={vars({ '--c': 'var(--optim)' })}>
            <i />
            Trainable factors
          </span>
        </div>
      }
      caption={
        <>
          Drift-constrained distillation at the parameter level. One adapted linear map computes{' '}
          <Tex tex={tx`h' = \overline W_{t-1}h + \tfrac{\alpha}{r} BAh`} />. The experiments use rank 32 and
          alpha 64 on every linear layer of the text backbone; the grids here are schematic (6 × 6, rank 2), and the
          update grid is the product of the two factor grids.
        </>
      }
    >
      <div className="fig-scroll" style={vars({ '--min': '860px' })}>
        <svg viewBox="0 0 1040 330" role="img" aria-label={`Low-rank update walkthrough, step ${k + 1} of 5: ${STEPS[k]}`}>
          {/* inherited dense weights */}
          <text x={base.x + (N * CELL) / 2} y={96} textAnchor="middle">
            <M s={`~W_{${wLabel}}`} size={17} />
          </text>
          <Grid
            key={`w-${inherited}`}
            x={base.x}
            y={base.y}
            values={data.w}
            color="var(--predictor)"
            overlay={inherited ? data.delta : undefined}
            overlayColor="var(--optim)"
            pop={k === 3}
            outline={k === 4 ? 'var(--deploy)' : undefined}
          />
          <Snow x={base.x + N * CELL + 14} y={base.y + 4} />
          <text className="t-small" x={base.x + (N * CELL) / 2} y={base.y + N * CELL + 24} textAnchor="middle">
            {k === 4 ? 'merged and deployed' : 'frozen within the stage'}
          </text>

          <text className="t-head" x={246} y={base.y + (N * CELL) / 2 + 7} textAnchor="middle" style={{ fontSize: 24 }} opacity={k === 4 ? 0.3 : 1}>
            +
          </text>

          {/* low-rank factors */}
          <text x={bPos.x + (R * CELL) / 2} y={96} textAnchor="middle" opacity={k === 4 ? 0.3 : 1}>
            <M s="B" size={17} />
          </text>
          <Grid x={bPos.x} y={bPos.y} values={bValues} color="var(--optim)" dim={k === 4} outline="var(--optim)" />
          <text x={aPos.x + (N * CELL) / 2} y={aPos.y - 18} textAnchor="middle" opacity={k === 4 ? 0.3 : 1}>
            <M s="A" size={17} />
          </text>
          <Grid
            key={`a-${inherited}`}
            x={aPos.x}
            y={aPos.y}
            values={aValues}
            color="var(--optim)"
            dim={k === 4}
            outline="var(--optim)"
            pop={k === 3}
          />
          <text className="t-small" x={bPos.x + 115} y={base.y + N * CELL + 24} textAnchor="middle" opacity={k === 4 ? 0.3 : 1}>
            {k === 0 || k === 3 ? 'new factors, zero effective update' : k === 1 ? 'trainable factors' : 'trained factors'}
          </text>

          {k === 1 && (
            <>
              <Edge kind="optim" flow d={`M 470 300 C 400 300, 350 290, ${bPos.x + 24} ${bPos.y + N * CELL + 34}`} />
              <text className="t-small" x={478} y={304} style={{ fill: 'var(--optim-ink)' }}>
                gradient of the constrained distillation loss
              </text>
            </>
          )}

          {/* merge */}
          <Edge kind="plain" d={`M ${aPos.x + N * CELL + 12} ${base.y + 72} L ${dPos.x - 14} ${base.y + 72}`} hidden={k !== 2} flow={k === 2} />
          <text x={dPos.x + (N * CELL) / 2} y={96} textAnchor="middle" opacity={k === 2 ? 1 : 0} style={{ transition: 'opacity 0.5s' }}>
            <tspan className="m">(α/r) </tspan>
            <M s="B" size={17} />
            <M s="A" size={17} />
          </text>
          <Grid key={`d-${k === 2}`} x={dPos.x} y={dPos.y} values={data.delta} color="var(--optim)" hidden={k !== 2} pop={k === 2} />
          <text className="t-small" x={dPos.x + (N * CELL) / 2} y={base.y + N * CELL + 24} textAnchor="middle" opacity={k === 2 ? 1 : 0}>
            low-rank update
          </text>

          <Edge kind="plain" d={`M ${dPos.x + N * CELL + 12} ${base.y + 72} L ${mPos.x - 14} ${base.y + 72}`} hidden={k !== 2} flow={k === 2} />
          <text x={mPos.x + (N * CELL) / 2} y={96} textAnchor="middle" opacity={k === 2 ? 1 : 0} style={{ transition: 'opacity 0.5s' }}>
            <M s="~W_{t}" size={17} />
          </text>
          <Grid
            key={`m-${k === 2}`}
            x={mPos.x}
            y={mPos.y}
            values={data.w}
            color="var(--predictor)"
            overlay={data.delta}
            overlayColor="var(--optim)"
            hidden={k !== 2}
            pop={k === 2}
          />
          <text className="t-small" x={mPos.x + (N * CELL) / 2} y={base.y + N * CELL + 24} textAnchor="middle" opacity={k === 2 ? 1 : 0}>
            merged dense weights
          </text>

          {k === 4 && (
            <g transform="translate(606 150)">
              <g className="node pop" data-kind="deploy">
              <rect className="under" width={374} height={92} rx={12} />
              <rect className="face" width={374} height={92} rx={12} />
              <text className="t-head" x={187} y={36} textAnchor="middle" style={{ fontSize: 16 }}>
                <tspan>Deploy merged </tspan>
                <M s="~θ_{t}" size={17} />
              </text>
              <text className="t-small" x={187} y={62} textAnchor="middle">
                weights alone, no harness · final drift audited against
                <tspan> </tspan>
                <M s="ε_{C}" size={13} />
                <tspan>, </tspan>
                <M s="ε_{0}" size={13} />
              </text>
              </g>
            </g>
          )}
        </svg>
      </div>
    </Figure>
  )
}
