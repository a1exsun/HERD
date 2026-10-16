import { useRef, useState } from 'react'
import { Figure } from '../components/Figure.tsx'
import { Edge, M, Node } from '../components/Svg.tsx'
import { Tex } from '../components/Tex.tsx'
import { tx } from '../lib/tex.ts'
import { vars } from '../lib/css.ts'
import { useCarousel, useInView } from '../lib/hooks.ts'

const STEPS = ['Previous state', 'Harness update', 'Harness state', 'Predictor update', 'Updated state']

export function StageAnatomy() {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const [hover, setHover] = useState(false)
  const loop = useCarousel(STEPS.length, 3400, { active: inView, paused: hover })
  const k = loop.step

  /** A part is dimmed until the walkthrough reaches it, active on its own step. */
  const at = (step: number) => ({ active: k === step, dim: k < step })
  const flow = (step: number) => ({ flow: k === step, dim: k < step })

  const notes = [
    <>
      <b>Previous state.</b> Stage <Tex tex="t" /> starts from the harness checkpoint <Tex tex="H_{t-1}" /> and the
      predictor <Tex tex={tx`\bar\theta_{t-1}`} /> produced by stage <Tex tex="t-1" />.
    </>,
    <>
      <b>Harness update.</b> The frozen base model <Tex tex={tx`\theta_0`} /> writes the current task&rsquo;s skill
      from bounded evidence <Tex tex="A_t" /> and the mistakes of the frozen <Tex tex={tx`\bar\theta_{t-1}`} />. The
      skill is committed unless it harms the teacher, and the worst-explained examples are retained.
    </>,
    <>
      <b>Harness state.</b> The committed skill registry <Tex tex="K_t" /> and the residual memory{' '}
      <Tex tex={tx`\mathcal R_t`} /> form <Tex tex="H_t" />, with <Tex tex={tx`\mathrm{Tok}(H_t)\le B_M`} />.
    </>,
    <>
      <b>Predictor update.</b> Drift-constrained distillation trains on <Tex tex={tx`D_t^{\mathrm{train}}`} />: the
      teacher is the frozen <Tex tex={tx`\bar\theta_{t-1}`} /> with the routed skill and the replayed verified
      answer, and the student sees the bare prompt. Probes from <Tex tex={tx`\mathcal R_{t-1}`} /> and{' '}
      <Tex tex={tx`\mathcal G`} /> measure the two drifts, each held under its budget by a Lagrange multiplier.
    </>,
    <>
      <b>Updated state.</b> <Tex tex="H_t" /> and <Tex tex={tx`\bar\theta_t`} /> are passed to stage{' '}
      <Tex tex="t+1" />. Only <Tex tex={tx`\bar\theta_t`} /> is deployed.
    </>,
  ]

  return (
    <Figure
      ref={ref}
      label="Figure 3"
      title="One recursive stage"
      schematic={false}
      note={notes[k]}
      progress={{ count: STEPS.length, index: k, label: `${k + 1} · ${STEPS[k]}` }}
      onAdvance={loop.advance}
      onHover={setHover}
      hint="Paused · click for next step"
      caption={
        <>
          Bounded current evidence updates <Tex tex="H_{t-1}" /> into <Tex tex="H_t" />. The fixed framework routes the
          current task&rsquo;s skill from <Tex tex="K_t" /> into the teacher, and historical and general probes
          constrain training.
        </>
      }
    >
      <div className="fig-scroll" style={vars({ '--min': '940px' })}>
        <svg viewBox="0 0 1200 460" role="img" aria-label={`Stage walkthrough, step ${k + 1} of 5: ${STEPS[k]}`}>
          <text className="t-mono" x={36} y={38}>
            PREVIOUS STATE
          </text>
          <text className="t-mono" x={1164} y={38} textAnchor="end">
            UPDATED STATE
          </text>

          <Edge kind="harness" d="M 186 100 L 274 100" {...flow(1)} />
          <Edge kind="predictor" d="M 186 292 C 262 292, 318 228, 338 155" {...flow(1)} />
          <Edge kind="harness" d="M 502 100 L 566 100" {...flow(2)} />
          <Edge kind="harness" d="M 640 144 C 640 196, 724 178, 762 214" {...flow(3)} />
          <Edge kind="predictor" d="M 186 308 L 701 308" {...flow(3)} />
          <Edge kind="guard" d="M 626 386 C 676 386, 700 362, 736 330" hidden={k < 3} />
          <Edge kind="guard" d="M 626 432 C 706 432, 768 386, 796 330" hidden={k < 3} />
          <Edge kind="harness" d="M 714 100 L 1010 100" {...flow(4)} />
          <Edge kind="predictor" d="M 958 280 C 990 280, 984 300, 1010 300" {...flow(4)} />
          <Edge kind="deploy" d="M 1090 328 L 1090 370" {...flow(4)} />

          <Node x={110} y={100} w={150} h={52} kind="harness" active={k === 0} onClick={() => loop.goto(0)}>
            <text textAnchor="middle" y={6}>
              <M s="H_{t−1}" size={17} />
            </text>
          </Node>
          <Node
            x={110}
            y={300}
            w={150}
            h={52}
            kind="predictor"
            active={k === 0 || k === 1 || k === 3}
            onClick={() => loop.goto(0)}
          >
            <text textAnchor="middle" y={6}>
              <M s="~θ_{t−1}" size={17} />
            </text>
          </Node>
          <text className="t-small" x={110} y={350} textAnchor="middle">
            frozen during stage t
          </text>

          <Node x={390} y={100} w={224} h={108} kind="harness" band={32} {...at(1)} onClick={() => loop.goto(1)}>
            <text className="t-head" textAnchor="middle" y={-32}>
              1 · Harness update
            </text>
            <text className="t-label" textAnchor="middle" y={0}>
              skill by θ₀ + harm guard
            </text>
            <text className="t-label" textAnchor="middle" y={20}>
              residual retention
            </text>
            <text className="t-small" textAnchor="middle" y={42}>
              <tspan>evidence </tspan>
              <M s="A_{t}" size={13} />
              <tspan>, budget </tspan>
              <M s="B_{A}" size={13} />
            </text>
          </Node>

          <Node x={640} y={100} w={146} h={88} kind="harness" {...at(2)} onClick={() => loop.goto(2)}>
            <text textAnchor="middle" y={-6}>
              <M s="H_{t}" size={18} />
            </text>
            <text textAnchor="middle" y={20}>
              <tspan className="t-small">(</tspan>
              <M s="K_{t}" size={14} />
              <tspan className="t-small">, </tspan>
              <M s="cal:R_{t}" size={14} />
              <tspan className="t-small">)</tspan>
            </text>
          </Node>
          <text className="t-small" x={640} y={38} textAnchor="middle" opacity={k < 2 ? 0.3 : 1}>
            bounded harness state
          </text>

          <Node x={830} y={270} w={254} h={116} kind="optim" band={32} {...at(3)} onClick={() => loop.goto(3)}>
            <text className="t-head" textAnchor="middle" y={-36}>
              2 · Internalize predictor
            </text>
            <text className="t-label" textAnchor="middle" y={-4}>
              Drift-Constrained Distillation
            </text>
            <text className="t-small" textAnchor="middle" y={18}>
              <tspan>data </tspan>
              <M s="D_{t}^{train}" size={13} />
              <tspan>, teacher </tspan>
              <M s="~θ_{t−1}" size={13} />
              <tspan> + skill</tspan>
            </text>
            <text className="t-small" textAnchor="middle" y={40}>
              student starts at <M s="~θ_{t−1}" size={13} />
            </text>
          </Node>

          <Node x={500} y={386} w={252} h={40} kind="guard" {...at(3)}>
            <text className="t-small" textAnchor="middle" y={5} style={{ fill: 'var(--ink-2)' }}>
              <tspan>history probes from </tspan>
              <M s="cal:R_{t−1}" size={13} />
              <tspan>, anchor </tspan>
              <M s="~θ_{t−1}" size={13} />
            </text>
          </Node>
          <Node x={500} y={432} w={252} h={40} kind="guard" {...at(3)}>
            <text className="t-small" textAnchor="middle" y={5} style={{ fill: 'var(--ink-2)' }}>
              <tspan>general probes </tspan>
              <M s="cal:G" size={13} />
              <tspan>, anchor </tspan>
              <M s="θ_{0}" size={13} />
            </text>
          </Node>

          <Node x={1090} y={100} w={150} h={52} kind="harness" {...at(4)} onClick={() => loop.goto(4)}>
            <text textAnchor="middle" y={6}>
              <M s="H_{t}" size={17} />
            </text>
          </Node>
          <Node x={1090} y={300} w={150} h={52} kind="predictor" {...at(4)} onClick={() => loop.goto(4)}>
            <text textAnchor="middle" y={6}>
              <M s="~θ_{t}" size={17} />
            </text>
          </Node>
          <Node x={1090} y={398} w={190} h={52} kind="deploy" {...at(4)}>
            <text className="t-label" textAnchor="middle" y={-2}>
              <tspan>Deploy </tspan>
              <M s="~θ_{t}" size={14} />
              <tspan> only</tspan>
            </text>
            <text className="t-small" textAnchor="middle" y={16}>
              standalone predictor
            </text>
          </Node>
          <text className="t-small" x={1090} y={200} textAnchor="middle" opacity={k < 4 ? 0.3 : 1}>
            passed to stage t + 1
          </text>
        </svg>
      </div>
    </Figure>
  )
}
