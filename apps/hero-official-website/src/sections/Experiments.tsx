import { PaperResults } from './PaperResults.tsx'
import { useState } from 'react'
import { ChartFigure, Dumbbell, Forest, Legend, LineChart, MiniChart, ScatterChart } from '../components/Charts.tsx'
import { heat } from '../lib/css.ts'
import type { Line } from '../components/Charts.tsx'
import { experiments, isControl, method, methodColor, score, signed } from '../lib/experiments.ts'
import type { Compared, Depth, DepthSd, ExperimentMethod } from '../lib/experiments.ts'

const families: Record<string, string> = {
  control: 'Control', 'non-parametric': 'Context', parametric: 'Weights', hybrid: 'Hybrid', ours: 'This work',
}
const { tasks, categories, anchors } = experiments.protocol
const methods = experiments.methods
const hero = method('hero')!
const base = method('base')!
const learners = methods.filter((m) => !isControl(m))
const ordered = [hero, ...methods.filter((m) => m.id !== 'hero')]
const final = (m: ExperimentMethod) => m.depths['25']
const heroFinal = final(hero)
const heroSd = hero.depthsSd?.['25']
const seedCount = 3
const range = (values: number[]) => [Math.min(...values), Math.max(...values)] as const
const weights = learners.filter((m) => m.family === 'parametric')

/** Mean ± spread over seeds, when the method has several. */
function Value({ value, sd, digits = 3, sign }: { value: number | undefined; sd?: number; digits?: number; sign?: boolean }) {
  return (
    <>
      {sign ? signed(value, digits) : score(value, digits)}
      {sd !== undefined && <small>± {sd.toFixed(digits)}</small>}
    </>
  )
}

type Column = {
  id: string
  label: string
  better: 'high' | 'low'
  get: (d: Depth) => number | undefined
  sd: (d: DepthSd | undefined) => number | undefined
  sign?: boolean
}

const COLUMNS: Column[] = [
  { id: 'seen', label: 'Task mean', better: 'high', get: (d) => d.seenMean, sd: (d) => d?.seenMean },
  { id: 'acq', label: 'Acquisition', better: 'high', get: (d) => d.acquisition, sd: (d) => d?.acquisition },
  { id: 'forget', label: 'Forgetting', better: 'low', get: (d) => d.forgetting, sd: (d) => d?.forgetting },
  { id: 'bwt', label: 'BWT', better: 'high', get: (d) => d.signedBWT, sd: (d) => d?.signedBWT, sign: true },
  { id: 'general', label: 'G-test', better: 'high', get: (d) => d.general.macro, sd: (d) => d?.general?.macro },
  { id: 'change', label: 'ΔG vs base', better: 'high', get: (d) => d.general.change, sd: (d) => d?.general?.change, sign: true },
  { id: 'worst', label: 'Worst ΔG', better: 'high', get: (d) => d.general.worstCategoryChange,
    sd: (d) => d?.general?.worstCategoryChange, sign: true },
]

/** The G-test category that fell furthest below the unchanged model. */
const worstCategory = (d: Depth) => {
  const start = final(base).general.categories
  const drop = (id: string) => d.general.categories[id] - start[id]
  return categories.reduce((worst, c) => drop(c.id) < drop(worst.id) ? c : worst)
}

function Comparison({ depth }: { depth: number }) {
  const best = Object.fromEntries(COLUMNS.map((column) => {
    const values = learners.map((m) => column.get(m.depths[depth])).filter((v): v is number => v !== undefined)
    return [column.id, column.better === 'high' ? Math.max(...values) : Math.min(...values)]
  }))
  return (
    <div className="experiment-scroll" role="region" aria-label="Method comparison" tabIndex={0}>
      <table className="experiment-table">
        <caption>After {depth} tasks · bold marks the best learning method in each column · Learning methods: mean ± standard deviation over {seedCount} seeds</caption>
        <thead><tr>
          <th scope="col">Method</th>
          {COLUMNS.map((column) => <th scope="col" key={column.id}>{column.label} {column.better === 'high' ? '↑' : '↓'}</th>)}
        </tr></thead>
        <tbody>{ordered.map((m) => {
          const result = m.depths[depth]
          const sd = m.depthsSd?.[depth]
          return (
            <tr key={m.id} className={m.id === 'hero' ? 'experiment-hero' : undefined}>
              <th scope="row">{m.short}<small>{families[m.family]}</small></th>
              {COLUMNS.map((column) => {
                const value = column.get(result)
                const top = !isControl(m) && value !== undefined && Math.abs(value - best[column.id]) < 5e-7
                return (
                  <td key={column.id} className={top ? 'experiment-best' : undefined}>
                    <Value value={value} sd={column.sd(sd)} sign={column.sign} />
                    {column.id === 'general' && result.general.panel === 'probe' && <small>200-item probe</small>}
                    {column.id === 'worst' && value !== undefined && value < 0 && <small>{worstCategory(result).label}</small>}
                  </td>
                )
              })}
            </tr>
          )
        })}</tbody>
      </table>
    </div>
  )
}

function lineFor(m: ExperimentMethod, points: { stage: number; value: number; sd?: number }[], from: number, to: number): Line {
  // A control measured once is drawn across the whole stream.
  const flat = m.constant && points.length === 1
  return {
    id: m.id, label: m.short, color: methodColor[m.id], strong: m.id === 'hero', dashed: m.constant,
    markers: !m.constant,
    points: flat
      ? [{ x: from, y: points[0].value }, { x: to, y: points[0].value }]
      : points.map((p) => ({ x: p.stage, y: p.value, sd: p.sd })),
  }
}

function StageTable({ rows, columns }: { rows: number[]; columns: { id: string; label: string; at: (stage: number) => string }[] }) {
  return (
    <table className="experiment-table">
      <thead><tr><th scope="col">Stage</th>{columns.map((c) => <th scope="col" key={c.id}>{c.label}</th>)}</tr></thead>
      <tbody>{rows.map((stage) => (
        <tr key={stage}><th scope="row">{stage}</th>{columns.map((c) => <td key={c.id}>{c.at(stage)}</td>)}</tr>
      ))}</tbody>
    </table>
  )
}

function AnchorTrajectory() {
  const lines = ordered.map((m) => lineFor(m, m.series.seenMean, anchors[0], 25))
  return (
    <ChartFigure
      title="Average over the tasks learned so far"
      legend={<Legend items={ordered.map((m) => ({ id: m.id, label: m.short, color: methodColor[m.id], dashed: m.constant }))} />}
      caption={`Seen-task mean at the full evaluations after 4, 8, 16 and 25 tasks; the mean covers more tasks at each step, so a control's line moves with the tasks it is averaged over. Dashed lines are controls; the band is HERD's spread over ${seedCount} seeds.`}
      table={<StageTable rows={anchors} columns={ordered.map((m) => ({
        id: m.id, label: m.short, at: (stage: number) => score(m.depths[stage]?.seenMean),
      }))} />}
    >
      <LineChart wide lines={lines} label="Seen-task mean of every method after 4, 8, 16 and 25 tasks. Exact values in the data table."
        x={{ min: 4, max: 25, ticks: anchors, label: 'Tasks learned' }}
        y={{ min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], label: 'Seen-task mean' }} />
    </ChartFigure>
  )
}

function TradeOffs() {
  const dots = (y: (d: Depth) => number, x: (d: Depth) => number) => ordered.map((m) => ({
    id: m.id, label: m.short, color: methodColor[m.id], strong: m.id === 'hero', x: x(final(m)), y: y(final(m)),
  }))
  const maxForgetting = Math.max(...methods.map((m) => final(m).forgetting))
  const table = (columns: { label: string; get: (d: Depth) => number }[]) => (
    <table className="experiment-table">
      <thead><tr><th scope="col">Method</th>{columns.map((c) => <th scope="col" key={c.label}>{c.label}</th>)}</tr></thead>
      <tbody>{ordered.map((m) => <tr key={m.id}><th scope="row">{m.short}</th>
        {columns.map((c) => <td key={c.label}>{score(c.get(final(m)))}</td>)}</tr>)}</tbody>
    </table>
  )
  return (
    <div className="experiment-grid">
      <ChartFigure title="Learning against forgetting"
        caption="After 25 tasks. Acquisition is the mean score on each task right after it is learned; forgetting is the mean drop on earlier tasks by the end. Lower right is better."
        table={table([{ label: 'Acquisition', get: (d) => d.acquisition }, { label: 'Forgetting', get: (d) => d.forgetting }])}>
        <ScatterChart dots={dots((d) => d.forgetting, (d) => d.acquisition)}
          label="Acquisition against forgetting after 25 tasks for every method. Exact values in the data table."
          x={{ min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], label: 'Acquisition →' }}
          y={{ min: 0, max: Math.max(0.03, Math.ceil(maxForgetting * 100) / 100), ticks: [0, 0.01, 0.02, 0.03], label: 'Forgetting ↓' }} />
      </ChartFigure>
      <ChartFigure title="Stream tasks against general capability"
        caption="After 25 tasks. x: seen-task mean over the 25 stream tasks; y: G-test macro over five held-out categories. Upper right is better."
        table={table([{ label: 'Task mean', get: (d) => d.seenMean }, { label: 'G-test', get: (d) => d.general.macro }])}>
        <ScatterChart dots={dots((d) => d.general.macro, (d) => d.seenMean)}
          label="Seen-task mean against G-test macro after 25 tasks for every method. Exact values in the data table."
          x={{ min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], label: 'Task mean →' }}
          y={{ min: 0, max: 0.9, ticks: [0, 0.3, 0.6, 0.9], label: 'G-test macro →' }} />
      </ChartFigure>
    </div>
  )
}

function Trajectory({ measure, comparison }: { measure: 'firstEight' | 'generalProbe'; comparison: ExperimentMethod }) {
  const firstEight = measure === 'firstEight'
  const title = firstEight ? 'Retention of the first eight tasks' : 'General capability through the stream'
  const start = firstEight ? 8 : 0
  const shown = [hero, comparison, base]
  const lines = shown.map((m) => lineFor(m, m.series[measure], start, 25))
  const stages = [...new Set(lines.flatMap((line) => line.points.map((p) => p.x)))].sort((a, b) => a - b)
  return (
    <ChartFigure title={title}
      legend={<Legend items={shown.map((m) => ({ id: m.id, label: m.short, color: methodColor[m.id], dashed: m.constant }))} />}
      caption={firstEight
        ? 'Mean score on the same eight TRACE tasks; x-axis: tasks learned.'
        : 'Category macro on the nested 200-item G probe; x-axis: tasks learned.'}
      table={<StageTable rows={stages} columns={shown.map((m, i) => ({
        id: m.id, label: m.short, at: (stage: number) => score(lines[i].points.find((p) => p.x === stage)?.y ?? (m.constant ? lines[i].points[0].y : undefined)),
      }))} />}>
      <LineChart lines={lines} label={`${title}: HERD, ${comparison.short} and the unchanged model. Exact values in the data table.`}
        x={{ min: start, max: 25, ticks: firstEight ? [8, 12, 16, 20, 25] : [0, 4, 8, 12, 16, 20, 25], label: 'Tasks learned' }}
        y={{ min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1] }} />
    </ChartFigure>
  )
}

type TaskView = 'learned' | 'final' | 'change'

function TaskMatrix() {
  const [view, setView] = useState<TaskView>('final')
  const value = (m: ExperimentMethod, j: number) => {
    const t = m.tasks[j]
    return view === 'learned' ? t.learned : view === 'final' ? t.final : t.final - t.learned
  }
  const fill = (v: number) => view === 'change'
    ? heat(v >= 0 ? 'var(--deploy)' : 'var(--optim)', Math.abs(v) / 0.3)
    : heat('var(--predictor)', v)
  const mean = (m: ExperimentMethod) => m.tasks.reduce((sum, _, j) => sum + value(m, j), 0) / tasks.length
  const views: [TaskView, string][] = [['learned', 'Right after learning'], ['final', 'After 25 tasks'], ['change', 'Change']]
  return (
    <div className="experiment-block">
      <div className="experiment-toolbar">
        <h3>Task by task</h3>
        <div className="experiment-depths" role="group" aria-label="Score shown">{views.map(([id, label]) =>
          <button type="button" key={id} aria-pressed={view === id} onClick={() => setView(id)}>{label}</button>)}</div>
      </div>
      <div className="experiment-scroll" role="region" aria-label="Scores per task" tabIndex={0}>
        <table className="experiment-table experiment-matrix">
          <caption>
            {view === 'learned' && 'Each task’s test score after the stage that learned it. Controls are constant, so they show their single evaluation.'}
            {view === 'final' && 'Each task’s test score after all 25 tasks.'}
            {view === 'change' && 'Score after 25 tasks minus the score right after learning: green gained, red forgotten.'}
          </caption>
          <thead><tr>
            <th scope="col">Task</th>
            {ordered.map((m) => <th scope="col" key={m.id} className={m.id === 'hero' ? 'experiment-hero-col' : undefined}>{m.short}</th>)}
          </tr></thead>
          <tbody>
            {tasks.map((task, j) => (
              <tr key={task.stage}>
                <th scope="row"><span className="experiment-stage">{task.stage}</span>{task.label}<small>{task.category} · {task.metricLabel}</small></th>
                {ordered.map((m) => {
                  const v = value(m, j)
                  return <td key={m.id} style={{ background: fill(v) }}>{view === 'change' ? signed(v, 2) : score(v, 2)}</td>
                })}
              </tr>
            ))}
            <tr className="experiment-matrix-mean">
              <th scope="row">Mean of 25 tasks</th>
              {ordered.map((m) => <td key={m.id}>{view === 'change' ? signed(mean(m)) : score(mean(m))}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="experiment-note">Task scores use each task’s own metric (exact match, ROUGE-L, edit similarity or SARI) on 100 test items. HERD cells are the mean over {seedCount} seeds.</p>
    </div>
  )
}

function GeneralCategories() {
  const baseCategories = final(base).general.categories
  return (
    <div className="experiment-block">
      <h3>General capability by category</h3>
      <div className="experiment-scroll" role="region" aria-label="G-test by category" tabIndex={0}>
        <table className="experiment-table experiment-matrix">
          <caption>Full G-test after 25 tasks, 200 items per category. Shading shows the change from the unchanged model: green higher, red lower.</caption>
          <thead><tr>
            <th scope="col">Category</th>
            {ordered.map((m) => <th scope="col" key={m.id} className={m.id === 'hero' ? 'experiment-hero-col' : undefined}>{m.short}</th>)}
          </tr></thead>
          <tbody>
            {categories.map((c) => (
              <tr key={c.id}>
                <th scope="row">{c.label}<small>{c.source}</small></th>
                {ordered.map((m) => {
                  const v = final(m).general.categories[c.id]
                  const change = v - baseCategories[c.id]
                  return (
                    <td key={m.id} style={{ background: m.id === 'base' ? undefined : heat(change >= 0 ? 'var(--deploy)' : 'var(--optim)', Math.abs(change) / 0.6) }}>
                      {score(v)}{m.id !== 'base' && <small>{signed(change)}</small>}
                    </td>
                  )
                })}
              </tr>
            ))}
            <tr className="experiment-matrix-mean">
              <th scope="row">Macro</th>
              {ordered.map((m) => <td key={m.id}>{score(final(m).general.macro)}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <p className="experiment-note">The multiple-choice categories score the first character of the answer, so a model that learns to answer with the option letter gains there; mathematical reasoning and instruction following measure open-ended answers.</p>
    </div>
  )
}

const stageLine = (id: string, label: string, color: string, stages: number[], values: (number | null)[], sd?: (number | null)[]): Line => ({
  id, label, color,
  points: stages.flatMap((stage, i) => values[i] === null ? [] : [{ x: stage, y: values[i]!, sd: sd?.[i] ?? null }]),
})

/** Median and interquartile range of `values`. */
function quartiles(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b)
  const at = (q: number) => {
    const i = (sorted.length - 1) * q
    const lo = Math.floor(i)
    return sorted[lo] + (sorted[Math.ceil(i)] - sorted[lo]) * (i - lo)
  }
  return { median: at(0.5), q1: at(0.25), q3: at(0.75) }
}

const logTicks = (min: number, max: number) => {
  const ticks = []
  for (let e = Math.ceil(Math.log10(min)); e <= Math.floor(Math.log10(max)); e++) ticks.push(10 ** e)
  return ticks
}

function TrainingCurves() {
  const u = experiments.hero.updates
  const per = u.perStage
  const stages = [...new Set(u.stage)]
  // Every update of every stage, as the mean over seeds with its spread.
  const kl: Line = {
    id: 'kl', label: 'Distillation KL', color: 'var(--optim)', markers: false,
    points: u.x.map((x, i) => ({ x, y: u.distillKL[i], sd: u.distillKLSd[i] })),
  }
  // The same update of every stage, summarised across the 25 stages.
  const within = (values: number[]) => Array.from({ length: per }, (_, k) => {
    const q = quartiles(values.filter((_, i) => u.update[i] === k + 1))
    return { x: k + 1, y: q.median, lo: q.q1, hi: q.q3 }
  })
  const klWithin = within(u.distillKL)
  const gradWithin = within(u.gradNorm)
  const stageSummary = (values: number[]) => (stage: number) => {
    const own = values.filter((_, i) => u.stage[i] === stage)
    return { first: own[0], median: quartiles(own).median, last: own[own.length - 1] }
  }
  const klStage = stageSummary(u.distillKL)
  const gradStage = stageSummary(u.gradNorm)
  const ratio = quartiles(stages.map((st) => klStage(st).last / klStage(st).first)).median
  const klMin = 10 ** Math.floor(Math.log10(Math.min(...u.distillKL)))
  const klMax = 10 ** Math.ceil(Math.log10(Math.max(...u.distillKL)))
  const gMin = 10 ** Math.floor(Math.log10(Math.min(...u.gradNorm)))
  const gMax = 10 ** Math.ceil(Math.log10(Math.max(...u.gradNorm)))
  const withinTable = (rows: { x: number; y: number; lo: number; hi: number }[]) => (
    <table className="experiment-table">
      <thead><tr><th scope="col">Update</th><th scope="col">Median</th><th scope="col">Lower quartile</th><th scope="col">Upper quartile</th></tr></thead>
      <tbody>{rows.map((row) => <tr key={row.x}><th scope="row">{row.x}</th><td>{score(row.y)}</td><td>{score(row.lo)}</td><td>{score(row.hi)}</td></tr>)}</tbody>
    </table>
  )
  return (
    <>
      <ChartFigure wide title="Distillation loss at every update"
        legend={<Legend items={[{ id: 'kl', label: `Distillation KL, mean over ${seedCount} seeds`, color: 'var(--optim)' }]} />}
        caption={`The training objective, KL from the harness-conditioned teacher to the student on the verified answers, at each of the ${per} updates of every stage (log scale). Each stage starts a fresh adapter at learning rate ${u.learningRate}, decaying linearly to zero, so the curve restarts at every new task; the band is the spread over seeds. In the median stage the last update's loss is ${Math.round(ratio * 100)}% of the first.`}
        table={<StageTable rows={stages} columns={[
          { id: 'f', label: 'First update', at: (st) => score(klStage(st).first) },
          { id: 'm', label: 'Median update', at: (st) => score(klStage(st).median) },
          { id: 'l', label: 'Last update', at: (st) => score(klStage(st).last) },
          { id: 'gf', label: 'Gradient norm, first', at: (st) => score(gradStage(st).first, 2) },
          { id: 'gl', label: 'Gradient norm, last', at: (st) => score(gradStage(st).last, 2) },
        ]} />}>
        <LineChart wide label="Distillation KL at every update of the 25 stages on a log scale. Per-stage values in the data table."
          lines={[kl]} guides={stages.slice(0, -1)}
          x={{ min: 0, max: 25, ticks: [0, 5, 10, 15, 20, 25], label: 'Tasks learned' }}
          y={{ min: klMin, max: klMax, log: true, ticks: logTicks(klMin, klMax), label: 'KL (nats per token)' }} />
      </ChartFigure>

      <ChartFigure title="Loss within a stage"
        legend={<Legend items={[{ id: 'kl', label: 'Median over stages', color: 'var(--optim)' }]} />}
        caption={`Distillation KL at each update position, summarised over the 25 stages: the line is the median and the band the interquartile range (log scale).`}
        table={withinTable(klWithin)}>
        <LineChart label="Median and interquartile range of the distillation KL at each update position across stages. Exact values in the data table."
          lines={[{ id: 'kl', label: 'Distillation KL', color: 'var(--optim)', points: klWithin }]}
          x={{ min: 1, max: per, ticks: [1, 8, 16, 24, per], label: 'Update within the stage' }}
          y={{ min: klMin, max: klMax, log: true, ticks: logTicks(klMin, klMax), label: 'KL (nats per token)' }} />
      </ChartFigure>

      <ChartFigure title="Gradient norm within a stage"
        legend={<Legend items={[{ id: 'g', label: 'Median over stages', color: 'var(--predictor)' }]} />}
        caption="Gradient norm before clipping at 1.0, at each update position, summarised over the 25 stages: median and interquartile range (log scale). Values above 1 are clipped before the step."
        table={withinTable(gradWithin)}>
        <LineChart label="Median and interquartile range of the gradient norm at each update position across stages. Exact values in the data table."
          lines={[{ id: 'g', label: 'Gradient norm', color: 'var(--predictor)', points: gradWithin }]}
          refs={[{ y: 1, label: 'clip 1.0', color: 'var(--ink-2)' }]}
          x={{ min: 1, max: per, ticks: [1, 8, 16, 24, per], label: 'Update within the stage' }}
          y={{ min: gMin, max: gMax, log: true, ticks: logTicks(gMin, gMax), label: 'Gradient norm' }} />
      </ChartFigure>
    </>
  )
}

function PairedDifferences() {
  const c = experiments.hero.comparisons
  const shown = c.methods.filter((m) => ['parametric', 'hybrid'].includes(method(m.id)?.family ?? ''))
  const panels: { id: Compared; title: string; better: 'right' | 'left' }[] = [
    { id: 'seenMean', title: 'Task mean', better: 'right' },
    { id: 'acquisition', title: 'Acquisition', better: 'right' },
    { id: 'forgetting', title: 'Forgetting', better: 'left' },
    { id: 'generalMacro', title: 'G-test macro', better: 'right' },
  ]
  const all: { id: Compared; label: string }[] = [
    { id: 'seenMean', label: 'Task mean' }, { id: 'acquisition', label: 'Acquisition' }, { id: 'forgetting', label: 'Forgetting' },
    { id: 'signedBWT', label: 'BWT' }, { id: 'generalMacro', label: 'G-test' },
  ]
  return (
    <div className="experiment-block">
      <h3>Are the differences larger than test noise?</h3>
      <p className="experiment-reading">
        HERD minus each weight-updating baseline after {c.stage} tasks, with 95% intervals from a paired bootstrap:
        each of {c.draws.toLocaleString('en')} draws resamples the test items of every task and G-test category and applies
        the same resample to both methods. An interval that excludes zero is colored.
      </p>
      <div className="experiment-grid">
        {panels.map((panel, i) => (
          <ChartFigure key={panel.id} title={panel.title}
            caption={panel.better === 'left' ? 'Lower forgetting is better, so negative differences favour HERD.' : 'Positive differences favour HERD.'}
            table={i === panels.length - 1 ? (
              <table className="experiment-table">
                <thead><tr><th scope="col">HERD minus</th>{all.map((m) => <th scope="col" key={m.id}>{m.label}</th>)}</tr></thead>
                <tbody>{c.methods.map((m) => (
                  <tr key={m.id}><th scope="row">{method(m.id)?.short}</th>
                    {all.map((metric) => <td key={metric.id}>{signed(m[metric.id].difference)}
                      <small>{signed(m[metric.id].interval[0])} to {signed(m[metric.id].interval[1])}</small></td>)}</tr>
                ))}</tbody>
              </table>
            ) : undefined}>
            <Forest better={panel.better} format={(v) => signed(v, 3)}
              label={`${panel.title}: HERD minus each weight-updating baseline with 95% paired bootstrap intervals.`}
              rows={shown.map((m) => ({ id: m.id, label: method(m.id)!.short, value: m[panel.id].difference,
                lo: m[panel.id].interval[0], hi: m[panel.id].interval[1] }))} />
          </ChartFigure>
        ))}
      </div>
      <p className="experiment-note">
        HERD enters with its seed-mean item scores, so its forgetting here is computed after averaging the seeds and differs
        slightly from the mean of the per-seed values in the tables above. The intervals cover test-item sampling only, not
        variation across seeds or task orders, and each baseline is a single run. The data table under the last panel also lists the controls and HCL-Stable.
      </p>
    </div>
  )
}

function AccuracyMatrix() {
  const cells = new Map(experiments.hero.matrix.map((c) => [`${c.stage}|${c.task}`, c]))
  const stages = [...new Set(experiments.hero.matrix.map((c) => c.stage))].sort((a, b) => a - b)
  return (
    <div className="experiment-block">
      <h3>The full score matrix</h3>
      <div className="experiment-scroll" role="region" aria-label="HERD score matrix" tabIndex={0}>
        <table className="experiment-matrix-grid">
          <caption>
            HERD’s test score (×100) on task j after stage t, mean over {seedCount} seeds. The outlined diagonal is each task right after it is learned;
            full rows follow the evaluation schedule (all seen tasks after 4, 8, 16 and 25 tasks, the first eight after 12 and 20). Blank cells are not evaluated.
          </caption>
          <thead><tr><th scope="col">t \ j</th>{tasks.map((task) => <th scope="col" key={task.stage} title={task.label}>{task.stage}</th>)}</tr></thead>
          <tbody>{stages.map((stage) => (
            <tr key={stage}><th scope="row">{stage}</th>{tasks.map((_, j) => {
              const cell = cells.get(`${stage}|${j}`)
              return <td key={j} className={j + 1 === stage ? 'is-diagonal' : undefined}
                style={cell ? { background: heat('var(--predictor)', cell.value) } : undefined}
                title={cell ? `${tasks[j].label} after stage ${stage}: ${score(cell.value)} ± ${score(cell.sd)}` : undefined}>
                {cell ? Math.round(cell.value * 100) : ''}</td>
            })}</tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}

function Retention() {
  const cells = experiments.hero.matrix
  const learned = (j: number) => cells.find((c) => c.task === j && c.stage === j + 1)!
  return (
    <div className="experiment-block">
      <h3>Each of the first eight tasks through the stream</h3>
      <div className="experiment-minis">{tasks.slice(0, 8).map((task, j) => {
        const points = cells.filter((c) => c.task === j).sort((a, b) => a.stage - b.stage)
          .map((c) => ({ x: c.stage, y: c.value, sd: c.sd }))
        const change = points[points.length - 1].y - learned(j).value
        return (
          <figure className="experiment-mini" key={task.stage}>
            <figcaption><strong>{task.label}</strong><span className={change >= 0 ? 'is-gain' : 'is-loss'}>{signed(change, 2)}</span></figcaption>
            <MiniChart color={methodColor.hero} points={points} reference={base.tasks[j].final}
              label={`${task.label}: HERD's score from stage ${task.stage} to 25. The dashed line is the unchanged model.`}
              x={{ min: 1, max: 25, ticks: [1, 8, 16, 25] }} y={{ min: 0, max: 1, ticks: [0, 0.5, 1] }} />
          </figure>
        )
      })}</div>
      <p className="experiment-note">Score on each task from the stage that learned it to the end of the stream, mean over seeds with their spread; the number is the change from right after learning to after 25 tasks. Dashed: the unchanged model.</p>
    </div>
  )
}

function TaskScores() {
  return (
    <ChartFigure wide title="Every task, mean ± standard deviation"
      caption="Hollow: the mean right after learning. Filled: the mean after 25 tasks, with a bar spanning one standard deviation; green rows gained, red rows lost."
      table={
        <table className="experiment-table">
          <thead><tr><th scope="col">Task</th><th scope="col">After learning</th><th scope="col">After 25 tasks</th></tr></thead>
          <tbody>{tasks.map((task, j) => (
            <tr key={task.stage}><th scope="row">{task.stage}. {task.label}</th><td>{score(hero.tasks[j].learned)}</td>
              <td><Value value={hero.tasks[j].final} sd={hero.tasks[j].finalSd} /></td></tr>
          ))}</tbody>
        </table>
      }>
      <Dumbbell label="For each task, HERD's mean after learning and after 25 tasks, with the final standard deviation."
        fromLabel="right after learning" toLabel="after 25 tasks"
        x={{ min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], label: 'Test score' }}
        rows={tasks.map((task, j) => ({ id: String(task.stage), label: `${task.stage}. ${task.label}`,
          from: hero.tasks[j].learned, to: hero.tasks[j].final, sd: hero.tasks[j].finalSd }))} />
    </ChartFigure>
  )
}

function CategoryTrajectories() {
  const g = experiments.hero.generalByCategory
  return (
    <div className="experiment-block">
      <h3>General capability, category by category</h3>
      <div className="experiment-minis experiment-minis-five">{categories.map((c) => (
        <figure className="experiment-mini" key={c.id}>
          <figcaption><strong>{c.label}</strong><span>{c.source}</span></figcaption>
          <MiniChart color={methodColor.hero} reference={g.reference[c.id]}
            label={`${c.label}: HERD's 40-item probe score after every stage. The dashed line is the unchanged model.`}
            points={g.stages.flatMap((stage, i) => g.series[c.id].value[i] === null ? []
              : [{ x: stage, y: g.series[c.id].value[i]!, sd: g.series[c.id].sd[i] }])}
            x={{ min: 1, max: 25, ticks: [1, 8, 16, 25] }} y={{ min: 0, max: 1, ticks: [0, 0.5, 1] }} />
        </figure>
      ))}</div>
      <p className="experiment-note">The 40-item probe of each category after every stage, mean over seeds with their spread; dashed: the unchanged model on the same items. With 40 items one answer moves a score by 0.025, so single-stage changes should be read cautiously.</p>
    </div>
  )
}

function InsideHero() {
  const h = experiments.hero
  const s = h.harness.stages
  const stageTicks = [1, 5, 10, 15, 20, 25]
  const finalDrift = [...h.drift.series.checkpointDriftFinal.value, ...h.drift.series.generalDriftFinal.value]
    .filter((v): v is number => v !== null)
  const budget = h.drift.checkpointBudget
  const within = finalDrift.filter((v) => v <= budget).length
  const committed = h.guard.series.committed.value.filter((v) => v === 1).length
  const active = (key: 'checkpointMultiplierActive' | 'generalMultiplierActive') =>
    h.multipliers.series[key].value.reduce<number>((sum, v) => sum + (v ?? 0), 0) / s.length
  const updateLine = (id: string, label: string, color: string, values: number[]): Line => ({
    id, label, color, markers: false, points: h.updates.x.map((x, i) => ({ x, y: values[i] })),
  })
  const driftMax = Math.max(...h.updates.generalDrift, ...h.updates.checkpointDrift)
  const tokensMax = Math.max(...h.harness.series.harnessTokens.value.filter((v): v is number => v !== null))
  const fmtK = (v: number) => (v >= 1000 ? `${Math.round(v / 1000)}k` : String(v))
  return (
    <div className="experiment-block">
      <h3>Inside HERD</h3>
      <p className="experiment-reading">
        The dev snapshot records what the harness, guard, drift constraints and distillation did at each stage.
        The figures show the mean over {seedCount} seeds; bands show the spread.
      </p>
      <div className="experiment-grid">
        <TrainingCurves />

        <ChartFigure wide title={within === finalDrift.length ? 'Drift ends every stage within its budget' : 'Drift against its budget'}
          legend={<Legend items={[{ id: 'c', label: 'Checkpoint drift', color: 'var(--guard)' }, { id: 'g', label: 'Base drift', color: 'var(--harness)' }]} />}
          caption={`Worst-group KL at each of the ${h.updates.perStage} updates of every stage. A drift above the budget activates its Lagrange multiplier, on average in ${Math.round(active('checkpointMultiplierActive') * 100)}% of updates for checkpoint drift and ${Math.round(active('generalMultiplierActive') * 100)}% for base drift. At the end of the stages, ${within} of the ${finalDrift.length} measured drifts are within the budget; stage 1 has no earlier task, so no checkpoint drift.`}
          table={<StageTable rows={h.drift.stages} columns={[
            { id: 'cf', label: 'Checkpoint, end', at: (st) => score(h.drift.series.checkpointDriftFinal.value[st - 1]) },
            { id: 'cp', label: 'Checkpoint, peak', at: (st) => score(h.drift.series.checkpointDriftPeak.value[st - 1]) },
            { id: 'gf', label: 'Base, end', at: (st) => score(h.drift.series.generalDriftFinal.value[st - 1]) },
            { id: 'gp', label: 'Base, peak', at: (st) => score(h.drift.series.generalDriftPeak.value[st - 1]) },
          ]} />}>
          <LineChart wide label="Checkpoint and base drift at every update across the 25 stages, with the drift budget. Per-stage values in the data table."
            lines={[updateLine('g', 'Base drift', 'var(--harness)', h.updates.generalDrift),
              updateLine('c', 'Checkpoint drift', 'var(--guard)', h.updates.checkpointDrift)]}
            refs={[{ y: budget, label: `budget ε = ${budget}`, color: 'var(--optim)' }]}
            x={{ min: 0, max: 25, ticks: [0, 5, 10, 15, 20, 25], label: 'Tasks learned' }}
            y={{ min: 0, max: Math.ceil(driftMax * 10) / 10, ticks: [0, 0.1, 0.2, 0.3].filter((v) => v <= Math.ceil(driftMax * 10) / 10), label: 'KL (nats per token)' }} />
        </ChartFigure>

        <ChartFigure title="The harness grows with the stream"
          legend={<Legend items={[{ id: 't', label: 'Harness size', color: 'var(--harness)' }]} />}
          caption={`Serialized skill registry after each stage. The guard committed the proposed skill at ${committed} of ${s.length} stages in every seed; the registry budget is ${h.harness.budget.toLocaleString('en')} tokens.`}
          table={<StageTable rows={s} columns={[
            { id: 't', label: 'Tokens', at: (st) => score(h.harness.series.harnessTokens.value[st - 1], 0) },
            { id: 'k', label: 'Skills', at: (st) => score(h.harness.series.skills.value[st - 1], 0) },
            { id: 'p', label: 'Teacher NLL, previous', at: (st) => score(h.guard.series.teacherNLLPrevious.value[st - 1]) },
            { id: 'c', label: 'Teacher NLL, candidate', at: (st) => score(h.guard.series.teacherNLLCandidate.value[st - 1]) },
          ]} />}>
          <LineChart label="Harness size in tokens after each stage. Exact values in the data table."
            lines={[stageLine('t', 'Harness size', 'var(--harness)', s, h.harness.series.harnessTokens.value, h.harness.series.harnessTokens.sd)]}
            x={{ min: 1, max: 25, ticks: stageTicks, label: 'Stage' }}
            y={{ min: 0, max: Math.ceil(tokensMax / 5000) * 5000, ticks: [0, 5000, 10000, 15000].filter((v) => v <= Math.ceil(tokensMax / 5000) * 5000), label: 'Tokens', format: fmtK }} />
        </ChartFigure>

        <ChartFigure title="Distillation closes the teacher–student gap"
          legend={<Legend items={[{ id: 'f', label: 'First update', color: 'var(--optim)' }, { id: 'l', label: 'Last quarter of updates', color: 'var(--deploy)' }]} />}
          caption="KL from the harness-conditioned teacher to the student on the verified answers, at the first update of each stage and averaged over its last quarter."
          table={<StageTable rows={h.distillation.stages} columns={[
            { id: 'f', label: 'First update', at: (st) => score(h.distillation.series.distillKLFirst.value[st - 1]) },
            { id: 'l', label: 'Last quarter', at: (st) => score(h.distillation.series.distillKLLastQuarter.value[st - 1]) },
          ]} />}>
          <LineChart label="Distillation KL at the first update and over the last quarter of updates, per stage. Exact values in the data table."
            lines={[stageLine('f', 'First update', 'var(--optim)', h.distillation.stages, h.distillation.series.distillKLFirst.value, h.distillation.series.distillKLFirst.sd),
              stageLine('l', 'Last quarter', 'var(--deploy)', h.distillation.stages, h.distillation.series.distillKLLastQuarter.value, h.distillation.series.distillKLLastQuarter.sd)]}
            x={{ min: 1, max: 25, ticks: stageTicks, label: 'Stage' }}
            y={{ min: 0, max: 3.5, ticks: [0, 1, 2, 3], label: 'KL (nats per token)' }} />
        </ChartFigure>

        <ChartFigure title="When the constraints act"
          legend={<Legend items={[{ id: 'c', label: 'Checkpoint multiplier', color: 'var(--guard)' }, { id: 'g', label: 'Base multiplier', color: 'var(--harness)' }]} />}
          caption="Share of each stage's updates in which a drift exceeded its budget, so that its Lagrange multiplier was positive and the drift entered the loss."
          table={<StageTable rows={h.multipliers.stages} columns={[
            { id: 'c', label: 'Checkpoint', at: (st) => score(h.multipliers.series.checkpointMultiplierActive.value[st - 1], 2) },
            { id: 'g', label: 'Base', at: (st) => score(h.multipliers.series.generalMultiplierActive.value[st - 1], 2) },
          ]} />}>
          <LineChart label="Share of updates with an active checkpoint or base multiplier, per stage. Exact values in the data table."
            lines={[stageLine('c', 'Checkpoint multiplier', 'var(--guard)', h.multipliers.stages, h.multipliers.series.checkpointMultiplierActive.value, h.multipliers.series.checkpointMultiplierActive.sd),
              stageLine('g', 'Base multiplier', 'var(--harness)', h.multipliers.stages, h.multipliers.series.generalMultiplierActive.value, h.multipliers.series.generalMultiplierActive.sd)]}
            x={{ min: 1, max: 25, ticks: stageTicks, label: 'Stage' }}
            y={{ min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], label: 'Share of updates', format: (v) => `${Math.round(v * 100)}%` }} />
        </ChartFigure>

        <ChartFigure title="Residual memory and its budget"
          legend={<Legend items={[{ id: 'r', label: 'Tokens retained', color: 'var(--predictor)' }]} />}
          caption={`Records kept for the drift probes of earlier tasks, ranked by the frozen checkpoint's residual loss under the committed harness. After the last stage they cover ${score(h.residual.series.residualTasks.value.at(-1), 1)} of the 25 tasks on average. They are never replayed as training targets.`}
          table={<StageTable rows={h.residual.stages} columns={[
            { id: 'k', label: 'Tokens', at: (st) => score(h.residual.series.residualTokens.value[st - 1], 0) },
            { id: 'r', label: 'Records', at: (st) => score(h.residual.series.residualRecords.value[st - 1], 0) },
            { id: 't', label: 'Tasks', at: (st) => score(h.residual.series.residualTasks.value[st - 1], 1) },
          ]} />}>
          <LineChart label="Residual-memory tokens per stage with the task budget. Exact values in the data table."
            lines={[stageLine('r', 'Tokens retained', 'var(--predictor)', h.residual.stages, h.residual.series.residualTokens.value, h.residual.series.residualTokens.sd)]}
            refs={[{ y: h.residual.budget, label: `budget ${h.residual.budget.toLocaleString('en')}`, color: 'var(--ink-2)' }]}
            x={{ min: 1, max: 25, ticks: stageTicks, label: 'Stage' }}
            y={{ min: 0, max: 30000, ticks: [0, 10000, 20000, 30000], label: 'Tokens', format: fmtK }} />
        </ChartFigure>
      </div>
    </div>
  )
}

function AggregateSummary() {
  const h = experiments.hero
  const cols: { label: string; get: (d: Depth) => number; sd: (d: DepthSd | undefined) => number | undefined; sign?: boolean }[] = [
    { label: 'Task mean', get: (d) => d.seenMean, sd: (d) => d?.seenMean },
    { label: 'Acquisition', get: (d) => d.acquisition, sd: (d) => d?.acquisition },
    { label: 'Forgetting', get: (d) => d.forgetting, sd: (d) => d?.forgetting },
    { label: 'BWT', get: (d) => d.signedBWT, sd: (d) => d?.signedBWT, sign: true },
    { label: 'G-test', get: (d) => d.general.macro, sd: (d) => d?.general?.macro },
  ]
  const hours = (seconds: number) => (seconds / 3600).toFixed(1)
  const interval = (pair: [number, number] | null | undefined) => (pair ? `${score(pair[0])}–${score(pair[1])}` : '—')
  return (
    <div className="experiment-block">
      <h3>HERD mean and standard deviation</h3>
      <div className="experiment-scroll" role="region" aria-label="HERD aggregate statistics" tabIndex={0}>
        <table className="experiment-table">
          <caption>After 25 tasks. Training hours are the distillation time summed over the 25 stages on {h.hardware}; harness proposals and test decoding are not included.</caption>
          <thead><tr><th scope="col">Statistic</th>{cols.map((c) => <th scope="col" key={c.label}>{c.label}</th>)}<th scope="col">Training hours</th></tr></thead>
          <tbody>
            <tr className="experiment-hero"><th scope="row">Mean ± sd</th>
              {cols.map((c) => <td key={c.label}><Value value={c.get(heroFinal)} sign={c.sign} sd={c.sd(heroSd)} /></td>)}
              <td>{hours(h.trainingSeconds.value)}<small>± {Math.round(h.trainingSeconds.sd / 60)} min</small></td></tr>
            <tr><th scope="row">95% interval</th>
              <td>{interval(hero.intervals.seenMean['25'])}</td><td>{interval(hero.intervals.acquisition)}</td>
              <td>—</td><td>—</td><td>—</td><td>—</td></tr>
          </tbody>
        </table>
      </div>
      <p className="experiment-note">Intervals resample the 100 test items of each task (4,000 draws) from the seed-mean item scores. Learning-method aggregates use seeds 42, 43 and 44.</p>
    </div>
  )
}

export function Experiments() {
  const [comparisonId, setComparisonId] = useState('uniform-replay')
  const comparison = method(comparisonId)!
  const [weightLow, weightHigh] = range(weights.map((m) => final(m).seenMean))
  const [weightGLow, weightGHigh] = range(weights.map((m) => final(m).general.macro))
  const ci = hero.intervals.seenMean['25']
  return (
    <section className="section" id="experiments">
      <div className="wrap experiments-content">
        <header>
          <h2 className="section-title reveal">Experiments</h2>
          <p className="experiment-lede">Learning a new task is only half the job. A continual learner also needs to keep earlier skills and the general abilities it started with. We follow Qwen3.5-4B through 25 tasks to see how these goals develop together.</p>
          <p className="experiment-note">TRACE+17 · 8 TRACE + 17 SuperNI tasks · {methods.length - 1} comparison methods · paper results below · seeds 42, 43, 44</p>
        </header>

        <PaperResults />

        <details className="experiment-block">
          <summary>Supplementary interactive charts · dev snapshot ({experiments.generated})</summary>
        <div className="experiment-stats">
          {[
            ['HERD task mean', score(heroFinal.seenMean), `± ${score(heroSd?.seenMean)} over seeds · 95% CI ${score(ci?.[0])}–${score(ci?.[1])}`],
            ['HERD acquisition', score(heroFinal.acquisition), 'Mean score right after learning each task'],
            ['HERD forgetting', score(heroFinal.forgetting), `± ${score(heroSd?.forgetting)} · lower is better`],
            ['HERD G-test', score(heroFinal.general.macro), `${signed(heroFinal.general.change)} versus the unchanged model`],
          ].map(([label, value, note]) => <div key={label}><span>{label}</span><strong>{value}</strong><small>{note}</small></div>)}
        </div>

        <div className="experiment-block">
          <div className="experiment-toolbar">
            <h3>Learn new tasks. Keep earlier skills.</h3>
          </div>
          <Comparison depth={25} />
          <p className="experiment-note">Task mean covers all tasks learned so far. Acquisition is the mean score on each task right after it is learned. Forgetting is the mean loss on earlier tasks, and BWT its signed counterpart. G-test averages five held-out categories; math shows one of them.</p>
          <p className="experiment-reading">
            After 25 tasks the weight-updating baselines reach a task mean of {score(weightLow)}–{score(weightHigh)}, and HERD reaches {score(heroFinal.seenMean)} with forgetting of {score(heroFinal.forgetting)}.
            Where they differ is general capability: their G-test macro is {score(weightGLow)}–{score(weightGHigh)}, HERD’s is {score(heroFinal.general.macro)}, and its mathematical reasoning stays at {score(heroFinal.general.categories['mathematical-reasoning'])}.
          </p>
        </div>

        <PairedDifferences />

        <div className="experiment-block">
          <h3>Where each method lands</h3>
          <AnchorTrajectory />
          <TradeOffs />
        </div>

        <div className="experiment-block">
          <div className="experiment-toolbar">
            <h3>Follow what survives the stream</h3>
            <label className="experiment-select">Compare with{' '}
              <select value={comparisonId} onChange={(event) => setComparisonId(event.target.value)}>
                {methods.filter((m) => m.id !== 'hero' && m.id !== 'base').map((m) =>
                  <option key={m.id} value={m.id}>{m.name}</option>)}
              </select>
            </label>
          </div>
          <div className="experiment-grid">
            <Trajectory measure="firstEight" comparison={comparison} />
            <Trajectory measure="generalProbe" comparison={comparison} />
          </div>
          <p className="experiment-reading">The first plot holds the original eight tasks fixed while seventeen new ones arrive. The second follows general capability on the same 200-item probe. Together, they show whether progress on the current task lasts beyond that task.</p>
        </div>

        <TaskMatrix />
        <GeneralCategories />
        <AccuracyMatrix />
        <Retention />
        <div className="experiment-block">
          <h3>Learned, then kept</h3>
          <TaskScores />
        </div>
        <CategoryTrajectories />
        <InsideHero />
        <AggregateSummary />
        </details>
      </div>
    </section>
  )
}
