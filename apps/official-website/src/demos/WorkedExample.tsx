import { useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { Figure } from '../components/Figure.tsx'
import { Tex } from '../components/Tex.tsx'
import { tx } from '../lib/tex.ts'
import { useCarousel, useInView } from '../lib/hooks.ts'

/* One ScienceQA item from the paper's Figure 1, learned at stage 5 and asked again at stage 12. */

const STEPS = ['Stage 5 · the task', 'Stage 5 · harness evolution', 'Stage 5 · distillation', 'Stage 12 · asked again']
const STAGES = Array.from({ length: 12 }, (_, i) => i + 1)

const ANSWERS = [
  { id: 'herd', name: 'HERD', text: 'B', ok: true, note: 'weights alone, three seeds' },
  { id: 'opr', name: 'OPR', text: 'A', ok: false, note: 'wrong choice' },
  { id: 'seq', name: 'Seq-LoRA', text: 'Thanks, Trudy', ok: false, note: 'right text, no choice letter' },
]

function Card({ step, k, kind, title, children }: { step: number; k: number; kind: string; title: ReactNode; children: ReactNode }) {
  return (
    <section className="we-card" data-kind={kind} data-state={k === step ? 'active' : k > step ? 'past' : 'future'}>
      <h5>{title}</h5>
      {children}
    </section>
  )
}

export function WorkedExample() {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const [hover, setHover] = useState(false)
  const loop = useCarousel(STEPS.length, 4200, { active: inView, paused: hover })
  const k = loop.step
  const stage = k === 3 ? 12 : 5

  const notes = [
    <><b>The task.</b> At stage 5 the stream reaches ScienceQA. The harness reads a bounded window{' '}
      <Tex tex="A_5" /> of whole records; the frozen checkpoint answers it first, and its mistakes go to the skill
      writer.</>,
    <><b>Harness evolution.</b> The base model writes a skill from the window and the mistakes, and the guard
      commits it because it lowers the checkpoint&rsquo;s mean residual on the window. After the merge, records that
      the skill and the new weights already predict yield their place: 0 to 1 ScienceQA samples stay in{' '}
      <Tex tex={tx`\mathcal R_5`} />.</>,
    <><b>Distillation.</b> The teacher, <Tex tex={tx`\bar\theta_4`} /> with the routed skill and the verified answer,
      opens with the choice letter; the student, the same weights with a fresh adapter and the bare prompt, learns
      the letter-first answer under the two drift budgets.</>,
    <><b>Asked again.</b> Seven stages later, after training on Reddit summaries, the same question goes to the
      deployed weights. HERD answers B from its weights alone across three seeds; OPR picks the wrong option and
      Seq-LoRA writes the right text without the letter. Across these stages Seq-LoRA&rsquo;s correct ScienceQA
      answers fall from 94 to 15.</>,
  ]

  return (
    <Figure
      ref={ref}
      label="Figure 4"
      title="One item through the stream"
      schematic={false}
      note={notes[k]}
      progress={{ count: STEPS.length, index: k, label: STEPS[k] }}
      onAdvance={loop.advance}
      onHover={setHover}
      hint="Paused · click for next step"
      caption={
        <>
          The example of the paper&rsquo;s Figure 1: one ScienceQA item learned at stage 5 and asked again at stage 12.
          The skill text and the stage-12 answers follow that figure; the drift anchors and the merge are shown in
          Figures 3 and 8.
        </>
      }
    >
      <div className="we">
        <ol className="we-track" aria-label={`Stage ${stage} of the stream`}>
          {STAGES.map((s) => (
            <li key={s} data-on={s === stage || undefined} data-mark={s === 5 || s === 12 || undefined}>
              <span>{s}</span>
              {s === 5 && <small>ScienceQA</small>}
              {s === 12 && <small>Reddit summary</small>}
            </li>
          ))}
        </ol>

        <div className="we-grid">
          <Card step={0} k={k} kind="plain" title={<>Task <Tex tex="A_5" /> · ScienceQA</>}>
            <p className="we-q">Which closing is correct for a letter?</p>
            <ul className="we-options">
              <li><b>A</b> thanks, Trudy</li>
              <li data-correct="true"><b>B</b> Thanks, Trudy</li>
            </ul>
          </Card>

          <Card step={1} k={k} kind="harness" title={<>Harness <Tex tex="H_5" /></>}>
            <p className="we-label">Skill in <Tex tex="K_5" /></p>
            <pre className="we-skill">Output the correct choice letter followed by a space, then …</pre>
            <p className="we-label"><Tex tex={tx`\mathcal R_5`} /> · 0–1 ScienceQA samples kept</p>
          </Card>

          <Card step={2} k={k} kind="optim" title="Drift-constrained distillation">
            <div className="we-row"><span><b>Teacher</b> with skill</span><code>B …</code></div>
            <div className="we-row"><span><b>Student</b> no skill</span><code>B …</code></div>
            <p className="we-label">The student learns the letter-first answer.</p>
          </Card>

          <Card step={3} k={k} kind="deploy" title={<>Deploy <Tex tex={tx`\bar\theta_{12}`} /> · no harness</>}>
            <p className="we-label">The same question at stage 12</p>
            {ANSWERS.map((a, i) => (
              <div key={a.id} className="we-answer" data-ok={a.ok} style={{ animationDelay: `${i * 180}ms` }}>
                <span><b>{a.name}</b> {a.ok ? '✓' : '✗'}</span>
                <code>{a.text}</code>
                <small>{a.note}</small>
              </div>
            ))}
          </Card>
        </div>
      </div>
    </Figure>
  )
}
