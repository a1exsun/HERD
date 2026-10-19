import { useRef, useState } from 'react'
import { Figure } from '../components/Figure.tsx'
import { Edge, M, Node } from '../components/Svg.tsx'
import { Tex } from '../components/Tex.tsx'
import { tx } from '../lib/tex.ts'
import { vars } from '../lib/css.ts'
import { useCarousel, useInView } from '../lib/hooks.ts'
import { TRACE8 } from '../lib/tasks.ts'

const W = 1200
const H = 462
const X0 = 78
const COL0 = 208
const DX = 124
const Y_H = 84
const Y_T = 226
const Y_P = 368
const CAP_W = 88
const CAP_H = 42
const TILE_W = 100
const TILE_H = 58

const PHASES = ['Task data', 'Harness update', 'Distillation', 'Deployment']

const col = (i: number) => (i < 0 ? X0 : COL0 + i * DX)

export function StreamLoop() {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const [hover, setHover] = useState(false)
  const loop = useCarousel(TRACE8.length * PHASES.length, 1600, { active: inView, paused: hover })
  const s = Math.floor(loop.step / PHASES.length)
  const p = loop.step % PHASES.length
  const task = TRACE8[s]

  const notes = [
    <>
      <b>
        Stage {s + 1}, {task.name}.
      </b>{' '}
      The learner receives the training and validation data of the current task. Earlier training sets are not
      revisited.
    </>,
    <>
      <b>Harness update.</b> The base model writes a skill for {task.name}, and the guard commits it into{' '}
      <Tex tex={`K_{${s + 1}}`} /> because it lowers the checkpoint&rsquo;s mean residual on the window. After distillation and merge, high-residual examples are retained in{' '}
      <Tex tex={tx`\mathcal R_{${String(s + 1)}}`} />.
    </>,
    <>
      <b>Drift-constrained distillation.</b> The frozen <Tex tex={tx`\bar\theta_{${String(s)}}`} />, given the
      routed skill from <Tex tex={`K_{${s + 1}}`} /> and the verified answer, teaches the student on the bare prompt.
      The previous checkpoint and the base model <Tex tex={tx`\theta_0`} /> serve as drift anchors.
    </>,
    <>
      <b>Deployment.</b> <Tex tex={tx`\bar\theta_{${String(s + 1)}}`} /> is deployed without the harness.{' '}
      {s + 1 < TRACE8.length ? (
        <>
          <Tex tex={`H_{${s + 1}}`} /> and <Tex tex={tx`\bar\theta_{${String(s + 1)}}`} /> are passed to stage {s + 2}.
        </>
      ) : (
        'This is the last stage of the stream.'
      )}
    </>,
  ]

  return (
    <Figure
      ref={ref}
      label="Figure 1"
      title="HERD over the first eight stages of TRACE+17"
      schematic={false}
      note={notes[p]}
      progress={{ count: TRACE8.length, index: s, label: `Stage ${s + 1} · ${task.name}` }}
      onAdvance={loop.advance}
      onHover={setHover}
      hint="Paused · click for next step"
      legend={
        <div className="legend">
          <span style={vars({ '--c': 'var(--harness)' })}>
            <i />
            Harness
          </span>
          <span style={vars({ '--c': 'var(--predictor)' })}>
            <i />
            Weights
          </span>
          <span style={vars({ '--c': 'var(--optim)' })}>
            <i />
            Distillation
          </span>
          <span style={vars({ '--c': 'var(--guard)' })}>
            <i data-dash="true" />
            Drift anchors
          </span>
          <span style={vars({ '--c': 'var(--deploy)' })}>
            <i />
            Deployment
          </span>
        </div>
      }
      caption={
        <>
          At each stage the harness is updated, the weights are trained by drift-constrained distillation, and the
          weights alone are deployed. Both states are passed to the next stage.
        </>
      }
    >
      <div className="fig-scroll" style={vars({ '--min': '920px' })}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Stage ${s + 1} of 8, ${task.name}. Step: ${PHASES[p]}.`}
        >
          <text className="t-mono" x={24} y={34}>
            HARNESS STATE
          </text>
          <text className="t-mono" x={24} y={Y_T - 46}>
            TASK STREAM
          </text>
          <text className="t-mono" x={24} y={Y_P - 64}>
            WEIGHT STATE
          </text>

          <line x1={X0} x2={col(7)} y1={Y_H} y2={Y_H} stroke="var(--line)" strokeWidth={1.2} strokeDasharray="2 6" />
          <line x1={X0} x2={col(7)} y1={Y_P} y2={Y_P} stroke="var(--line)" strokeWidth={1.2} strokeDasharray="2 6" />

          {/* drift anchors, drawn under the capsules */}
          <Edge
            kind="guard"
            head={false}
            hidden={p !== 2}
            d={`M ${X0} ${Y_P + CAP_H / 2} Q ${(X0 + col(s)) / 2} ${Y_P + 62 + s * 5} ${col(s) + 12} ${Y_P + CAP_H / 2 + 2}`}
          />
          {s > 0 && (
            <Edge
              kind="guard"
              head={false}
              hidden={p !== 2}
              d={`M ${col(s - 1)} ${Y_P - CAP_H / 2} Q ${(col(s - 1) + col(s)) / 2} ${Y_P - 58} ${col(s) - 14} ${Y_P - CAP_H / 2 - 2}`}
            />
          )}
          {p === 2 && (
            <>
              <text className="t-small" x={(X0 + col(s)) / 2} y={Y_P + 62 + s * 3} textAnchor="middle" fill="var(--guard)">
                <tspan fill="var(--guard)">base anchor </tspan>
                <M s="ε_{0}" size={13} />
              </text>
              {s > 0 && (
                <text className="t-small" x={(col(s - 1) + col(s)) / 2} y={Y_P - 50} textAnchor="middle">
                  <tspan fill="var(--guard)">checkpoint anchor </tspan>
                  <M s="ε_{C}" size={13} />
                </text>
              )}
            </>
          )}

          <Node x={X0} y={Y_H} w={CAP_W} h={CAP_H} kind="harness" r={21} className={s > 0 || p > 0 ? 'past' : ''}>
            <text textAnchor="middle" y={5}>
              <M s="H_{0}" />
            </text>
          </Node>
          <Node x={X0} y={Y_P} w={CAP_W} h={CAP_H} kind="predictor" r={21} active={p === 2}>
            <text textAnchor="middle" y={5}>
              <M s="θ_{0}" />
            </text>
          </Node>
          <text className="t-small" x={X0} y={Y_H + 40} textAnchor="middle">
            empty
          </text>
          <text className="t-small" x={X0} y={Y_P - 30} textAnchor="middle">
            base model
          </text>

          {TRACE8.map((t, i) => {
            const cx = col(i)
            const current = i === s
            const showH = i < s || (current && p >= 1)
            const showP = i < s || (current && p >= 2)
            const px = col(i - 1) + CAP_W / 2
            return (
              <g key={t.name}>
                <Edge
                  kind="harness"
                  hidden={!showH}
                  dim={showH && !current}
                  flow={current && p === 1}
                  d={`M ${px} ${Y_H} L ${cx - CAP_W / 2 - 3} ${Y_H}`}
                />
                <Edge
                  kind="predictor"
                  hidden={!showP}
                  dim={showP && !current}
                  flow={current && p === 2}
                  d={`M ${px} ${Y_P} L ${cx - CAP_W / 2 - 3} ${Y_P}`}
                />
                <Edge
                  kind="harness"
                  hidden={!(current && p === 1)}
                  flow={current && p === 1}
                  d={`M ${cx} ${Y_T - TILE_H / 2} L ${cx} ${Y_H + CAP_H / 2 + 3}`}
                />
                <Edge
                  kind="optim"
                  hidden={!(current && p === 2)}
                  flow={current && p === 2}
                  d={`M ${cx} ${Y_T + TILE_H / 2} L ${cx} ${Y_P - CAP_H / 2 - 3}`}
                />
                <Edge
                  kind="optim"
                  hidden={!(current && p === 2)}
                  flow={current && p === 2}
                  d={`M ${cx + 30} ${Y_H + CAP_H / 2} C ${cx + 74} ${Y_H + 96}, ${cx + 74} ${Y_P - 96}, ${cx + 31} ${Y_P - CAP_H / 2 - 3}`}
                />

                <Node
                  x={cx}
                  y={Y_T}
                  w={TILE_W}
                  h={TILE_H}
                  kind="plain"
                  active={current}
                  dim={i > s}
                  className={i < s ? 'past' : ''}
                  onClick={() => loop.goto(i * PHASES.length)}
                >
                  <text className="t-mono" textAnchor="middle" y={-7}>
                    STAGE {i + 1}
                  </text>
                  <text className="t-label" textAnchor="middle" y={13} style={{ fontSize: 12.5 }}>
                    {t.name}
                  </text>
                </Node>

                <Node
                  key={`h-${showH}`}
                  x={cx}
                  y={Y_H}
                  w={CAP_W}
                  h={CAP_H}
                  kind="harness"
                  r={21}
                  hidden={!showH}
                  active={current && p === 1}
                  className={`${showH && current ? 'pop' : ''} ${i < s ? 'past' : ''}`}
                >
                  <text textAnchor="middle" y={5}>
                    <M s={`H_{${i + 1}}`} />
                  </text>
                </Node>

                <Node
                  key={`p-${showP}`}
                  x={cx}
                  y={Y_P}
                  w={CAP_W}
                  h={CAP_H}
                  kind={current && p === 3 ? 'deploy' : 'predictor'}
                  r={21}
                  hidden={!showP}
                  active={current && p >= 2}
                  className={`${showP && current && p === 2 ? 'pop' : ''} ${i < s ? 'past' : ''}`}
                >
                  <text textAnchor="middle" y={5}>
                    <M s={`~θ_{${i + 1}}`} />
                  </text>
                </Node>

                {current && p === 3 && (
                  <>
                    <rect
                      className="ring"
                      x={cx - CAP_W / 2 - 6}
                      y={Y_P - CAP_H / 2 - 6}
                      width={CAP_W + 12}
                      height={CAP_H + 12}
                      rx={27}
                    />
                    <text className="t-small" x={cx} y={Y_P + 46} textAnchor="middle">
                      <tspan fill="var(--deploy)" fontWeight={600}>
                        deployed without harness
                      </tspan>
                    </text>
                  </>
                )}
              </g>
            )
          })}

          {/* phase indicator */}
          <g transform={`translate(${W - 24} 30)`}>
            {PHASES.map((name, i) => (
              <g key={name} transform={`translate(${-(PHASES.length - 1 - i) * 22} 0)`}>
                <circle
                  r={i === p ? 6 : 4}
                  fill={i === p ? 'var(--ink)' : 'var(--line-strong)'}
                  style={{ transition: 'r 0.3s, fill 0.3s' }}
                />
              </g>
            ))}
            <text className="t-mono" x={-(PHASES.length - 1) * 22 - 16} y={4} textAnchor="end">
              {`${p + 1}/4 ${PHASES[p].toUpperCase()}`}
            </text>
          </g>
        </svg>
      </div>
    </Figure>
  )
}
