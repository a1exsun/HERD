import { useState } from 'react'
import type { ReactNode } from 'react'
import paper from '../../data/paper-results.json'
import { ChartFigure, LineChart, ScatterChart } from '../components/Charts.tsx'
import type { Dot, Line } from '../components/Charts.tsx'
import { vars } from '../lib/css.ts'
import { colorOf, familyLabel, fixed, method, methods, signed } from '../lib/methods.ts'
import type { Family, Method } from '../lib/methods.ts'

/* ------------------------------------------------------------ Table 1 */

const directions = [1, 1, -1, 1, 1, 1, 1]
const learning = methods.filter((m) => m.family !== 'Control')
const best = directions.map((d, i) => Math.max(...learning.map((m) => d * m.values[i])))
const FAMILIES: Family[] = ['Context', 'Weights', 'Hybrid', 'Control']
// HERD's spread over seeds, from the full row of Table 3, keyed by Table 1 column.
const HERD_SD = new Map(paper.herdSd.columns.map((col, k) => [col, paper.herdSd.values[k]]))

function ResultsTable() {
  const [sort, setSort] = useState<number | null>(null)
  const [hidden, setHidden] = useState<Family[]>([])
  const rows = methods.filter((m) => !hidden.includes(m.family))
  const sorted = sort === null ? rows : [...rows].sort((a, b) => directions[sort] * (b.values[sort] - a.values[sort]))
  const toggle = (f: Family) => setHidden((h) => (h.includes(f) ? h.filter((x) => x !== f) : [...h, f]))
  return (
    <div className="experiment-block">
      <div className="viz-toolbar">
        <div className="viz-chips" role="group" aria-label="Show method families">
          {FAMILIES.map((f) => (
            <button type="button" key={f} aria-pressed={!hidden.includes(f)} onClick={() => toggle(f)}>{familyLabel[f]}</button>
          ))}
        </div>
        <span className="viz-hint">Click a column to rank methods by it, best first</span>
      </div>
      <div className="experiment-scroll" role="region" aria-label="Paper Table 1" tabIndex={0}>
        <table className="experiment-table viz-table">
          <caption>Paper Table 1 · TRACE+17 results after stage 25 under a matched model, stream, splits, prompts and schedule; means over three seeds, HERD with ± one standard deviation (from Table 3). Bold: best learning method (controls excluded) per column.</caption>
          <thead><tr>
            <th scope="col"><button type="button" onClick={() => setSort(null)} aria-pressed={sort === null}>Method</button></th>
            {paper.columns.map((name, i) => (
              <th scope="col" key={name} aria-sort={sort === i ? 'descending' : undefined}>
                <button type="button" onClick={() => setSort(i)} aria-pressed={sort === i}>{name} {directions[i] > 0 ? '↑' : '↓'}</button>
              </th>
            ))}
          </tr></thead>
          <tbody>{sorted.map((m) => (
            <tr key={m.id} className={m.id === 'hero' ? 'experiment-hero' : undefined}>
              <th scope="row"><i className="viz-swatch" style={vars({ '--c': colorOf(m.id) })} />{m.name}<small>{familyLabel[m.family]}</small></th>
              {m.values.map((v, i) => (
                <td key={i} className={m.family !== 'Control' && v * directions[i] === best[i] ? 'experiment-best' : undefined}>
                  {fixed(v)}
                  {m.id === 'hero' && HERD_SD.has(i) && <small>± {HERD_SD.get(i)!.toFixed(3)}</small>}
                  {i < 2 || i === 4 ? <span className="viz-bar" style={vars({ '--w': `${v * 100}%`, '--c': colorOf(m.id) })} /> : null}
                </td>
              ))}
            </tr>
          ))}</tbody>
        </table>
      </div>
    </div>
  )
}

/* ----------------------------------------------------------- Frontier */

function dotsFor(x: (m: Method) => number, y: (m: Method) => number): Dot[] {
  return methods.filter((m) => m.id !== 'static-harness').map((m) => ({
    id: m.id, label: m.name, color: colorOf(m.id), x: x(m), y: y(m), strong: m.id === 'hero',
    hollow: m.family === 'Hybrid' || m.family === 'Control',
  }))
}

function Frontier() {
  const [zoom, setZoom] = useState(false)
  const general = dotsFor((m) => m.values[0], (m) => m.values[4])
  const forget = dotsFor((m) => m.values[1], (m) => m.values[2])
  const near = (dots: Dot[]) => dots.filter((d) => d.x >= 0.6)
  const table = (cols: [string, number][]) => (
    <table className="experiment-table">
      <thead><tr><th scope="col">Method</th>{cols.map(([c]) => <th scope="col" key={c}>{c}</th>)}</tr></thead>
      <tbody>{methods.map((m) => <tr key={m.id}><th scope="row">{m.name}</th>{cols.map(([c, i]) => <td key={c}>{fixed(m.values[i])}</td>)}</tr>)}</tbody>
    </table>
  )
  return (
    <div className="experiment-block">
      <div className="viz-toolbar">
        <h4>Where each method lands</h4>
        <div className="viz-chips" role="group" aria-label="Axis range">
          <button type="button" aria-pressed={!zoom} onClick={() => setZoom(false)}>All methods</button>
          <button type="button" aria-pressed={zoom} onClick={() => setZoom(true)}>Zoom on the weight-updating cluster</button>
        </div>
      </div>
      <div className="experiment-grid">
        <ChartFigure title="Retention against general capability"
          caption="AA after stage 25 against the G-test macro mean. Upper right is better. Filled dots: HERD (green) and the non-parametric (orange) or parametric (blue) baselines; rings: the hybrid OPHSD and the base control."
          table={table([['AA', 0], ['G-test', 4]])}>
          <ScatterChart dots={zoom ? near(general) : general}
            label="AA against G-test macro mean after stage 25 for every method. Exact values in the data table."
            x={zoom ? { min: 0.6, max: 0.72, ticks: [0.6, 0.64, 0.68, 0.72], label: 'AA →' } : { min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], label: 'AA →' }}
            y={zoom ? { min: 0.55, max: 0.8, ticks: [0.55, 0.6, 0.65, 0.7, 0.75, 0.8], label: 'G-test ↑' } : { min: 0, max: 0.9, ticks: [0, 0.3, 0.6, 0.9], label: 'G-test ↑' }} />
        </ChartFigure>
        <ChartFigure title="Acquisition against forgetting"
          caption="Acquisition, each task's score right after it is learned, against forgetting by stage 25. Lower right is better; SDFT's low forgetting comes with collapsed acquisition."
          table={table([['Acquisition', 1], ['Forgetting', 2]])}>
          <ScatterChart dots={zoom ? near(forget) : forget}
            label="Acquisition against forgetting after stage 25 for every method. Exact values in the data table."
            x={zoom ? { min: 0.6, max: 0.73, ticks: [0.6, 0.64, 0.68, 0.72], label: 'Acquisition →' } : { min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], label: 'Acquisition →' }}
            y={{ min: 0, max: 0.035, ticks: [0, 0.01, 0.02, 0.03], label: 'Forgetting ↓' }} />
        </ChartFigure>
      </div>
    </div>
  )
}

/* ------------------------------------------------- Paired comparisons */

// Which sign of each difference favours HERD: lower forgetting is better.
const FAVOUR = [1, 1, -1, 1, 1]
const WEIGHT_FREE = ['Base', 'Static harness', 'HCL-Stable']

function PairedComparisons() {
  const [metric, setMetric] = useState(0)
  const [all, setAll] = useState(false)
  const rows = paper.comparisons.rows.filter((r) => all || !WEIGHT_FREE.includes(r.name))
  const reach = Math.max(...rows.flatMap((r) => r.cells[metric].interval.map(Math.abs)), 0.01) * 1.1
  const at = (v: number) => `${((v + reach) / (2 * reach)) * 100}%`
  return (
    <div className="experiment-block">
      <div className="viz-toolbar">
        <h4>Is each difference larger than test noise?</h4>
        <div className="viz-chips" role="group" aria-label="Baselines shown">
          <button type="button" aria-pressed={!all} onClick={() => setAll(false)}>Weight-updating baselines</button>
          <button type="button" aria-pressed={all} onClick={() => setAll(true)}>All baselines</button>
        </div>
      </div>
      <div className="viz-chips" role="group" aria-label="Quantity compared">
        {paper.comparisons.columns.map((c, i) => <button type="button" key={c} aria-pressed={metric === i} onClick={() => setMetric(i)}>{c}</button>)}
      </div>
      <figure className="experiment-chart viz-forest">
        <div className="viz-forest-axis" aria-hidden="true">
          <span>{FAVOUR[metric] > 0 ? '← favours the baseline' : '← favours HERD'}</span>
          <span>{FAVOUR[metric] > 0 ? 'favours HERD →' : 'favours the baseline →'}</span>
        </div>
        {rows.map((r) => {
          const { difference, interval: [lo, hi], excludesZero } = r.cells[metric]
          const tone = !excludesZero ? 'none' : difference * FAVOUR[metric] > 0 ? 'herd' : 'base'
          return (
            <div className="viz-forest-row" key={r.name} data-tone={tone}
              title={`HERD − ${r.name}: ${signed(difference)} [${fixed(lo)}, ${fixed(hi)}]`}>
              <span>{r.name}</span>
              <span className="viz-forest-track">
                <i className="viz-forest-zero" style={{ left: at(0) }} />
                <i className="viz-forest-ci" style={{ left: at(lo), width: `calc(${at(hi)} - ${at(lo)})` }} />
                <i className="viz-forest-dot" style={{ left: at(difference) }} />
              </span>
              <b>{signed(difference)}<small>[{fixed(lo)}, {fixed(hi)}]</small></b>
            </div>
          )
        })}
        <figcaption>Paper Table 2 · HERD minus each baseline after stage 25, with 95% paired item-bootstrap intervals (4,000 draws, items resampled within task and G-test category). Coloured: the interval excludes zero; grey: a tie.</figcaption>
      </figure>
    </div>
  )
}

/* ------------------------------------------------------ Long horizon */

const lh = paper.longHorizon
type Series = Record<string, number[]>
const LINE_IDS = Object.keys(lh.firstEight.methods).filter((id) => id !== 'hero')

function linesFor(series: Series, stages: number[], focus: string): Line[] {
  return Object.entries(series).map(([id, values]) => ({
    id, label: method(id).name, color: colorOf(id), strong: id === 'hero', muted: id !== 'hero' && id !== focus,
    dashed: method(id).family === 'Hybrid',
    points: stages.map((s, i) => ({ x: s, y: values[i] })),
  }))
}

function StageTable({ series, stages }: { series: Series; stages: number[] }) {
  return (
    <table className="experiment-table">
      <thead><tr><th scope="col">Method</th>{stages.map((s) => <th scope="col" key={s}>Stage {s}</th>)}</tr></thead>
      <tbody>{Object.entries(series).map(([id, v]) => <tr key={id}><th scope="row">{method(id).name}</th>{v.map((x, i) => <td key={i}>{fixed(x)}</td>)}</tr>)}</tbody>
    </table>
  )
}

/** Diverging fill for a change in points: red for a loss, blue for a gain, grey at zero. */
const diverging = (points: number, clip: number) =>
  `color-mix(in srgb, ${points < 0 ? 'var(--viz-neg)' : 'var(--viz-pos)'} ${Math.round(Math.min(1, Math.abs(points) / clip) * 85)}%, var(--viz-mid))`

function CategoryHeatmap({ focus, onFocus }: { focus: string; onFocus: (id: string) => void }) {
  const rows = Object.entries(lh.categoryChange)
  return (
    <div className="experiment-scroll" role="region" aria-label="Category change at stage 25" tabIndex={0}>
      <table className="viz-heat">
        <caption>Change in each G-test category after stage 25 against the base model, in points, for the weight-updating methods. The outlined cell is each method&rsquo;s worst ΔC; click a row to follow that method in the charts above.</caption>
        <thead><tr><th scope="col">Method</th>{lh.categories.map((c) => <th scope="col" key={c.id}>{c.source}<small>{c.label}</small></th>)}</tr></thead>
        <tbody>{rows.map(([id, v]) => {
          const worst = Math.min(...v)
          return (
            <tr key={id} data-focus={id === focus || id === 'hero' ? 'true' : undefined}>
              <th scope="row">{id === 'hero' ? method(id).name : <button type="button" onClick={() => onFocus(id)}>{method(id).name}</button>}</th>
              {v.map((x, i) => (
                <td key={i} data-worst={x === worst && x < 0 ? 'true' : undefined} style={{ background: diverging(x, 70) }}
                  title={`${method(id).name}, ${lh.categories[i].source}: ${x > 0 ? '+' : ''}${x} points`}>
                  {x > 0 ? `+${x}` : String(x).replace('-', '−')}
                </td>
              ))}
            </tr>
          )
        })}</tbody>
        <tfoot><tr><th scope="row">Base score</th>{lh.categories.map((c) => <td key={c.id}>{c.base.toFixed(2)}</td>)}</tr></tfoot>
      </table>
    </div>
  )
}

function LongHorizon() {
  const [focus, setFocus] = useState('seq-lora')
  return (
    <div className="experiment-block">
      <div className="viz-toolbar">
        <span className="viz-hint">Compare HERD with</span>
        <div className="viz-chips" role="group" aria-label="Baseline shown beside HERD">
          {LINE_IDS.map((id) => (
            <button type="button" key={id} aria-pressed={focus === id} onClick={() => setFocus(id)}
              style={vars({ '--c': colorOf(id) })}>{method(id).name}</button>
          ))}
        </div>
      </div>
      <div className="experiment-grid">
        <ChartFigure title="Mean score on the first eight tasks"
          caption="The eight TRACE tasks at the retention stages 8, 12, 16, 20 and 25. Grey lines are the other methods; the dotted line is the base model."
          table={<StageTable series={lh.firstEight.methods} stages={lh.firstEight.stages} />}>
          <LineChart lines={linesFor(lh.firstEight.methods, lh.firstEight.stages, focus)}
            label={`Mean score on tasks 1 to 8 by stage for HERD and ${method(focus).name}, with the other methods in grey.`}
            refs={[{ y: lh.firstEight.base, label: 'Base' }]}
            x={{ min: 8, max: 25, ticks: lh.firstEight.stages, label: 'Stage' }}
            y={{ min: 0, max: 0.8, ticks: [0, 0.2, 0.4, 0.6, 0.8], format: (v) => v.toFixed(v === 0 ? 0 : 1) }} />
        </ChartFigure>
        <ChartFigure title="G-test macro mean"
          caption="General capability at the anchor stages 0, 4, 8, 16 and 25, over five held-out categories. The dotted line is the base model."
          table={<StageTable series={lh.gtest.methods} stages={lh.gtest.stages} />}>
          <LineChart lines={linesFor(lh.gtest.methods, lh.gtest.stages, focus)}
            label={`G-test macro mean by stage for HERD and ${method(focus).name}, with the other methods in grey.`}
            refs={[{ y: lh.gtest.base, label: 'Base' }]}
            x={{ min: 0, max: 25, ticks: lh.gtest.stages, label: 'Stage' }}
            y={{ min: 0, max: 0.9, ticks: [0, 0.3, 0.6, 0.9], format: (v) => v.toFixed(v === 0 ? 0 : 1) }} />
        </ChartFigure>
      </div>
      <CategoryHeatmap focus={focus} onFocus={setFocus} />
    </div>
  )
}

/* ---------------------------------------------------------- Ablations */

const ABLATION_NOTES: Record<string, ReactNode> = {
  '− skills': <>Without skills, the teacher sees only the verified answer and nothing is proposed or committed; memory, probes and anchors are unchanged. Acquisition falls by 4.5 points and AA by 5.0, the largest losses in both columns, while forgetting and general capability move by at most a point.</>,
  'uniform memory': <>Filling the memory by uniform sampling under the same budget leaves acquisition within half a point, but forgetting doubles, from 0.010 to 0.020, the value of uniform replay in Table 1. The checkpoint probes are drawn from the memory, so uniform selection anchors the checkpoint on records the weights already predict and leaves the hardest earlier records unguarded; the stage-end D<sup>C</sup> stays within budget at 0.034.</>,
  '− checkpoint anchor': <>Without the checkpoint anchor, forgetting rises to 0.030, close to the worst parametric baseline&rsquo;s 0.032, and the audited D<sup>C</sup> ends at 0.118, more than twice the budget.</>,
  '− base anchor': <>Without the base anchor, the checkpoint constraint remains active, but base drift accumulates: the stage-end D<sup>0</sup> grows from 0.104 at stage 4 to 0.212 at stage 25, worst ΔC falls from −0.031 to −0.490, and mathematics drops from 0.885 to 0.395 where the full model keeps 0.823.</>,
  '− both anchors': <>With neither anchor, the harness evolves and nothing constrains the weights: acquisition holds, but forgetting reaches 0.044, above every parametric baseline, and worst ΔC reaches −0.560, close to their −0.63 to −0.68.</>,
  'HERD (full)': <>Three seeds, shown with ± one standard deviation. Three times these spreads give the reference scales of 0.006 for acquisition, 0.024 for AA, 0.009 for forgetting, 0.018 for ΔC and 0.036 for worst ΔC; the variants are single runs, so changes inside these scales are greyed and not interpreted.</>,
  'SFT objective': <>Cross-entropy on the verified answers replaces distillation under the same adapter, budget and constraints. Both audited drifts stay within budget, yet the general loss doubles, to ΔC −0.123 and worst ΔC −0.320: the teacher distribution limits drift where the probes do not reach.</>,
  'εC = 0.02': <>The constraint binds: acquisition pays 1.8 points and worst ΔC improves by 4.5. The 0.5-point drop in forgetting is inside the seed spread.</>,
  'εC = 0.1': <>The constraint goes slack: forgetting rises by 1.1 points and the audited D<sup>C</sup> ends at 0.071, over 0.05 and under the budget it was given. The 0.4-point rise in acquisition is inside the seed spread. ε<sub>0</sub> was not swept: removing the base anchor is its ε<sub>0</sub> → ∞ limit and already costs 9.5 points of ΔC.</>,
  '+ harness (infer.)': <>The deployed weights of the full run, re-scored with the stage-25 harness attached to every prompt: forgetting falls by 0.3 points and the other metrics move by at most 0.2 points. The weights have already internalized the harness.</>,
}

// Forgetting is the one column where a rise is worse.
const harm = (v: number, i: number) => (i === 2 ? -v : v)
const BUDGET = 0.05
const DRIFT_MAX = 0.25
const ab = paper.ablations
const NOISE = ab.noise.map((v) => v * 100)
/** The checkpoint budget a variant ran with: the ε_C sweep changes it, every other run uses 0.05. */
const budgetOf = (name: string, i: number) => (i === 0 && name.startsWith('εC = ') ? Number(name.slice(5)) : BUDGET)
const sweep = (i: number) => [['εC = 0.02', 0.02], ['HERD (full)', 0.05], ['εC = 0.1', 0.1]].map(([name, x]) => ({
  x: x as number, y: ab.rows.find((r) => r.name === name)!.changes?.[i] ?? 0,
}))

function Ablations() {
  const [full, setFull] = useState(false)
  const rows = full ? ab.rows : ab.rows.filter((r) => ab.figure4.includes(r.name))
  const [picked, setPicked] = useState(ab.figure4[0])
  return (
    <div className="experiment-block">
      <div className="viz-toolbar">
        <div className="viz-chips" role="group" aria-label="Rows shown">
          <button type="button" aria-pressed={!full} onClick={() => setFull(false)}>Paper Figure 4</button>
          <button type="button" aria-pressed={full} onClick={() => setFull(true)}>Full Table 3: SFT objective and εC sweep</button>
        </div>
      </div>
      <div className="experiment-scroll" role="region" aria-label="Ablations" tabIndex={0}>
        <table className="viz-heat viz-ablation">
          <caption>Ablations after stage 25: changes from full HERD in points, red is worse (colours clipped at ±10); grey changes are inside three seed standard deviations. The bars are the audited drifts D<sup>C</sup> and D<sup>0</sup> against each run&rsquo;s budget (tick). Variants are single runs. Click a row to read what it shows.</caption>
          <thead><tr>
            <th scope="col">Variant</th>
            {ab.columns.map((c) => <th scope="col" key={c}>{c}</th>)}
            <th scope="col" className="viz-drift-head">Audited drift</th>
          </tr></thead>
          <tbody>{rows.map((row) => (
            <tr key={row.name} data-focus={picked === row.name ? 'true' : undefined}>
              <th scope="row"><button type="button" aria-pressed={picked === row.name} onClick={() => setPicked(row.name)}>{row.name}</button></th>
              {row.changes
                ? row.changes.map((v, i) => {
                    const noise = Math.abs(v) < NOISE[i]
                    return <td key={i} data-noise={noise || undefined} style={{ background: noise ? undefined : diverging(harm(v, i), 10) }}>{signed(v, 1)}</td>
                  })
                : row.values.map((v, i) => <td key={i}>{fixed(v)}<small>± {paper.herdSd.values[i].toFixed(3)}</small></td>)}
              <td className="viz-drift">
                {row.drifts.map((d, i) => {
                  const budget = budgetOf(row.name, i)
                  return (
                    <span key={i} className="viz-drift-row" data-over={d > budget ? 'true' : undefined}>
                      <b>{i === 0 ? 'Dᶜ' : 'D⁰'}</b>
                      <span className="viz-drift-track" style={vars({ '--w': `${(d / DRIFT_MAX) * 100}%`, '--budget': `${(budget / DRIFT_MAX) * 100}%` })} />
                      <em>{fixed(d)}</em>
                    </span>
                  )
                })}
              </td>
            </tr>
          ))}</tbody>
        </table>
      </div>
      <div className={full ? 'experiment-grid' : undefined}>
      <p className="viz-note" aria-live="polite"><strong>{picked}.</strong> {ABLATION_NOTES[picked]}</p>
      {full && (
        <ChartFigure title="The checkpoint budget trades acquisition against retention"
          caption="Changes from full HERD (εC = 0.05) in points as the checkpoint budget is tightened to 0.02 or loosened to 0.1. Single runs; for forgetting, lower is better."
          table={
            <table className="experiment-table">
              <thead><tr><th scope="col">ε<sub>C</sub></th>{['Acquisition', 'Forgetting', 'Worst ΔC'].map((c) => <th scope="col" key={c}>{c}</th>)}</tr></thead>
              <tbody>{[0, 1, 2].map((k) => <tr key={k}><th scope="row">{[0.02, 0.05, 0.1][k]}</th>{[0, 2, 4].map((i) => <td key={i}>{signed(sweep(i)[k].y, 1)}</td>)}</tr>)}</tbody>
            </table>
          }>
          <LineChart label="Change in acquisition, forgetting and worst ΔC against the checkpoint budget εC."
            lines={[
              { id: 'acq', label: 'Acquisition', color: 'var(--viz-hero)', strong: true, points: sweep(0) },
              { id: 'f', label: 'Forgetting', color: 'var(--viz-neg)', dashed: true, points: sweep(2) },
              { id: 'c', label: 'Worst ΔC', color: 'var(--viz-weights)', points: sweep(4) },
            ]}
            x={{ min: 0.02, max: 0.1, ticks: [0.02, 0.05, 0.1], label: 'εC' }}
            y={{ min: -4, max: 6, ticks: [-4, -2, 0, 2, 4, 6], label: 'Change in points' }} />
        </ChartFigure>
      )}
      </div>
    </div>
  )
}

/* --------------------------------------------------------------- Cost */

const COST_IDS: Record<string, string> = Object.fromEntries(methods.map((m) => [m.name, m.id]))
const LOG = { min: 1, max: 6 }
const logAt = (v: number) => `${((Math.log10(v) - LOG.min) / (LOG.max - LOG.min)) * 100}%`

function Cost() {
  return (
    <figure className="experiment-chart viz-cost">
      <h4>Training time and inference context</h4>
      <div className="viz-cost-head" aria-hidden="true"><span /><span>Seconds per stage, stage 1 → 25, and total (log scale)</span><span>Context</span></div>
      {paper.cost.rows.map((r) => {
        const [first, last, total, tokens] = r.values
        const id = COST_IDS[r.name]
        return (
          <div className="viz-cost-row" key={r.name} data-hero={id === 'hero' || undefined}
            title={`${r.name}: ${first.toLocaleString('en')} s at stage 1, ${last.toLocaleString('en')} s at stage 25, ${total.toLocaleString('en')} s in total; ${tokens.toLocaleString('en')} context tokens`}>
            <span><i className="viz-swatch" style={vars({ '--c': colorOf(id) })} />{r.name}</span>
            <span className="viz-cost-track">
              <i className="viz-cost-total" style={vars({ '--w': logAt(total), '--c': colorOf(id) })} />
              <i className="viz-cost-span" style={{ left: logAt(Math.min(first, last)), width: `calc(${logAt(Math.max(first, last))} - ${logAt(Math.min(first, last))})` }} />
              <i className="viz-cost-dot" data-end="first" style={{ left: logAt(first) }} />
              <i className="viz-cost-dot" data-end="last" style={{ left: logAt(last) }} />
              <em data-far={total > 2e5 || undefined} style={{ left: logAt(total) }}>{total.toLocaleString('en')} s</em>
            </span>
            <b>{tokens ? tokens.toLocaleString('en') : '0'}</b>
          </div>
        )
      })}
      <div className="viz-cost-axis" aria-hidden="true"><span /><span>{['10 s', '100 s', '10³ s', '10⁴ s', '10⁵ s', '10⁶ s'].map((t) => <i key={t}>{t}</i>)}</span><span /></div>
      <figcaption>Paper Table 4 · Bars: total training time over 25 stages. Hollow and filled dots: a stage&rsquo;s time at stage 1 and at stage 25. Context: tokens attached to every prompt at inference.</figcaption>
    </figure>
  )
}

/* ------------------------------------------------------------ Section */

export function PaperResults() {
  const hero = method('hero').values
  return (
    <>
      <div className="experiment-stats">
        {[[0, 'HERD AA'], [1, 'HERD acquisition'], [2, 'HERD forgetting'], [4, 'HERD G-test']].map(([i, label]) => (
          <div key={label}><span>{label}</span><strong>{hero[Number(i)].toFixed(3)}</strong><small>Paper · Table 1</small></div>
        ))}
      </div>

      <div className="experiment-block">
        <h3>Q1 · Main results after 25 stages</h3>
        <p className="experiment-reading">
          HERD lies on the frontier of the three families: it acquires as much as the strongest parametric method,
          retains more than any method that updates weights, loses the least general capability among them, and
          deploys without context. Its acquisition, 0.713, is statistically indistinguishable from Seq-LoRA&rsquo;s
          0.715, the plasticity floor (paired difference −0.002, interval [−0.010, 0.007]), and from uniform replay and
          TreeLoRA. Its AA of 0.706 is the highest in the table, significantly above Seq-LoRA, OPR, TreeLoRA and OPHSD
          and tied with uniform replay. Forgetting is 0.010, about half or less that of every weight-updating baseline
          but SDFT, whose 0.003 reflects collapse, not retention: it acquires only 0.024.
        </p>
      </div>
      <ResultsTable />
      <Frontier />
      <PairedComparisons />
      <div className="experiment-block">
        <p className="experiment-reading">
          ACE and HCL finish within 1.4 points of the unchanged base model: ACE&rsquo;s playbook stops growing by task
          2, and HCL commits none of its proposals in 25 stages, so its scores amount to the base model under a fixed
          system prompt. HERD, scored bare, is 27 points above both. Uniform replay holds exactly as many tokens as
          HERD&rsquo;s residual memory and still forgets like the others (F 0.020, ΔC −0.238): the gap is not in how
          much is stored, but in what is kept and what is constrained.
        </p>
      </div>

      <div className="experiment-block">
        <h3>Q2 · Long-horizon stability</h3>
        <p className="experiment-reading">
          On the first eight tasks, HERD&rsquo;s mean stays within half a point of its stage-8 level throughout,
          from 0.664 at stage 8 to 0.665 at stage 25. Every parametric baseline moves more: Seq-LoRA falls to 0.372 at stage 12 before recovering to
          0.637, TreeLoRA to 0.593, and OPR and MoRAM decline to 0.621 and 0.625. On the G-test, the parametric
          baselines take most of their loss by stage 4 or 8 and then plateau, while HERD loses 1.5 points by stage 4
          and 6.3 by stage 25.
        </p>
      </div>
      <LongHorizon />
      <div className="experiment-block">
        <p className="experiment-reading">
          The parametric baselines other than SDFT pay for new tasks with old abilities: ΔC is −0.20 to −0.24 and the
          worst category falls by 62 to 68 points. The loss concentrates in GSM8K and IFEval, which collapse within four
          stages (Seq-LoRA: from 0.89 and 0.82 to 0.18 and 0.33) and never recover, while knowledge, commonsense and
          multilingual scores stay within a few points of the base. HERD&rsquo;s largest loss, 13 points, falls in the
          same instruction-following category.
        </p>
        <p className="experiment-reading">
          The two recursive self-distillation baselines instead drift further at every checkpoint. OPHSD&rsquo;s G-test
          falls through 0.741, 0.661, 0.613 and 0.582, and its first eight tasks from 0.524 to 0.506. SDFT, whose
          teacher is asked to include its thinking process, produces answers that lengthen from 345 tokens at stage 1
          to the 5,120-token ceiling from stage 17 on, and ends at AA 0.021 and G-test 0.172. HERD also distills the
          model from itself under a privileged context, but its teacher context is a skill and the verified answer
          rather than text generated by the drifting model, and two anchors bound drift: the audited stage-25 drifts
          are 0.021 and 0.041 against budgets of 0.05.
        </p>
      </div>

      <div className="experiment-block">
        <h3>Q3 · Ablations: each component protects one quantity</h3>
        <p className="experiment-reading">
          Each row removes or replaces one component of HERD at a time and reports the change after stage 25. Without
          both anchors, the harness evolves and nothing constrains the weights: acquisition holds, and retention and
          general capability are those of a parametric baseline. OPHSD, whose harness never changes and whose drift is
          never budgeted, has lower acquisition and a larger general-capability loss than HERD (acquisition 0.639,
          worst ΔC −0.645). Only the full model, which evolves the harness under constraint, is on the frontier.
          Replacing distillation by cross-entropy on the verified answers, under the same adapter, budget and
          constraints, doubles the general loss, to ΔC −0.123 and worst ΔC −0.320, with both audited drifts within
          budget; this suggests that the teacher distribution limits drift where the probes do not reach. The budget
          ε<sub>C</sub> trades forgetting against acquisition in the expected direction.
        </p>
      </div>
      <Ablations />

      <div className="experiment-block">
        <h3>Cost</h3>
        <p className="experiment-reading">
          A HERD stage runs a fixed number of updates and probes, so its time tracks the task&rsquo;s answer length,
          not the stage index: 316 s at stage 1 and 284 s at stage 25, 10,723 s in total, within 3% of Seq-LoRA. The two
          recursive self-distillation baselines grow with the stage, SDFT from 1,999 to 3,626 s as its rollouts lengthen
          and OPHSD from 11,364 to 30,495 s as it regenerates its teacher context. ACE&rsquo;s 888,082 s go to the
          38,103 generator, reflector and curator calls of its offline trainer, and its playbook adds 15,410 tokens to
          every prompt; HERD deploys bare weights.
        </p>
      </div>
      <Cost />

      <div className="experiment-block">
        <h3>Limitations</h3>
        <p className="experiment-reading">
          The empirical validation is scoped to one base model and one 25-stage stream rather than a large-scale
          sweep. HERD also relies on a verified answer for every training record and assumes a stream with known task
          boundaries, one task per stage, and no revisits.
        </p>
      </div>
    </>
  )
}
