import { useState } from 'react'
import paper from '../../data/paper-results.json'
import { PaperResults } from './PaperResults.tsx'

const FULL = [4, 8, 16, 25]
const RETENTION = [12, 20]
const STAGES = paper.stream.map((t) => t.stage)

/** What is scored after stage t, by Appendix B: the current task always; every seen task and the full G-test at
    the anchor stages; tasks 1 to 8 at stages 12 and 20; the 200-item probe at every other stage. */
function schedule(t: number) {
  if (FULL.includes(t)) return { tasks: (j: number) => j <= t, general: 'full G-test, 1,000 items', kind: 'full' }
  if (RETENTION.includes(t)) return { tasks: (j: number) => j <= 8 || j === t, general: null, kind: 'retention' }
  return { tasks: (j: number) => j === t, general: '200-item G-test probe', kind: 'probe' }
}

function StreamSetup() {
  const [stage, setStage] = useState(12)
  const [task, setTask] = useState(12)
  const plan = schedule(stage)
  const scored = STAGES.filter(plan.tasks).length
  const info = paper.stream[task - 1]
  return (
    <figure className="experiment-chart viz-stream">
      <h4>The TRACE+17 stream and its evaluation schedule</h4>
      <div className="experiment-scroll" tabIndex={0} role="region" aria-label="Task stream and evaluation schedule">
        <div className="viz-stream-grid">
          {paper.stream.map((t) => (
            <button type="button" key={t.stage} className="viz-stream-task" data-source={t.source === 'TRACE' ? 'trace' : 'superni'}
              data-scored={plan.tasks(t.stage) || undefined} data-picked={task === t.stage || undefined} data-future={t.stage > stage || undefined}
              onClick={() => setTask(t.stage)} onMouseEnter={() => setTask(t.stage)} aria-label={`Task ${t.stage}: ${t.name}, ${t.metric}`}>
              <b>{t.stage}</b><span>{t.name.replace('SuperNI ', '')}</span>
            </button>
          ))}
          {STAGES.map((t) => {
            const kind = schedule(t).kind
            return (
              <button type="button" key={`s${t}`} className="viz-stream-eval" data-kind={kind} aria-pressed={stage === t}
                onClick={() => setStage(t)} aria-label={`Evaluation after stage ${t}`}>
                <i />
              </button>
            )
          })}
        </div>
      </div>
      <div className="viz-stream-legend" aria-hidden="true">
        <span data-kind="full"><i />all seen tasks and the full G-test</span>
        <span data-kind="retention"><i />tasks 1 to 8</span>
        <span data-kind="probe"><i />current task and the 200-item probe</span>
      </div>
      <div className="viz-stream-detail">
        <p><strong>After stage {stage}</strong>: {scored === 1 ? 'the current task' : `${scored} tasks`} scored{plan.general ? `, plus the ${plan.general}` : ''}. Click a dot to change the stage.</p>
        <p><strong>Task {info.stage} · {info.name}</strong> · {info.source} · {info.category} · scored by {info.metric}</p>
      </div>
      <figcaption>The eight TRACE tasks in their canonical order (stronger outline), then 17 Super-NaturalInstructions tasks; each has 500 training, 100 validation and 100 test examples. Stage 0 is the base model. This schedule is why the long-horizon charts below have points only at stages 8, 12, 16, 20 and 25, and the G-test at 4, 8, 16 and 25.</figcaption>
    </figure>
  )
}

export function Experiments() {
  return (
    <section className="section" id="experiments">
      <div className="wrap experiments-content">
        <header>
          <h2 className="section-title reveal">Experiments</h2>
          <p className="experiment-lede">The experiments ask three questions. Q1: does HERD outperform purely parametric and purely non-parametric methods? Q2: as the number of stages grows, does HERD maintain long-horizon stability? Q3: are harness evolution, high-residual memory and the two drift anchors all indispensable?</p>
          <p className="experiment-reading">TRACE+17 appends 17 Super-NaturalInstructions tasks to the eight TRACE tasks, because eight stages are too few to tell stability from slow collapse. Each of the 25 tasks has 500 training, 100 validation and 100 test examples. A G-test of five categories, 200 held-out items each, tracks general capability: knowledge, mathematics, commonsense, instruction following and multilingual understanding (MMLU, GSM8K, HellaSwag, IFEval, XCOPA). All methods start from Qwen3.5-4B with the same decoding, length limits and scorers; all weight updates share one budget, 32 updates of 32 records per stage. After every stage the current task is scored; after stages 4, 8, 16 and 25, all seen tasks and the G-test.</p>
          <p className="experiment-note">Acquisition is the mean of each task&rsquo;s score right after it is learned; AA is what remains after stage 25; F is what is lost in between. ΔC is the mean change over the five G-test categories and worst ΔC the largest drop. All numbers are means over three seeds, with standard deviations and 95% bootstrap intervals over items.</p>
        </header>

        <StreamSetup />
        <PaperResults />
      </div>
    </section>
  )
}
