import { useEffect, useMemo, useRef, useState } from 'react'
import { Figure } from '../components/Figure.tsx'
import { Tex } from '../components/Tex.tsx'
import { tx } from '../lib/tex.ts'
import { vars } from '../lib/css.ts'
import { useInView, useReducedMotion } from '../lib/hooks.ts'
import { mulberry32 } from '../lib/rng.ts'

interface Rec {
  id: string
  source: 'retained' | 'new'
  tokens: number
  /** Residual under each of the three illustrative skill registries. */
  r: [number, number, number]
}

const MAX_TOKENS = 420
const MAX_R = 3.2

function makePool(): Rec[] {
  const rand = mulberry32(20261001)
  const pool: Rec[] = []
  for (let i = 0; i < 14; i++) {
    const source = i < 8 ? 'retained' : 'new'
    const tokens = Math.round((70 + rand() * 350) / 10) * 10
    const base = 0.5 + rand() * 2.6
    // A better skill explains some records, which lowers their residual.
    const explained1 = rand() < 0.4
    const explained2 = explained1 || rand() < 0.45
    pool.push({
      id: `z${String(i + 1).padStart(2, '0')}`,
      source,
      tokens,
      r: [base, explained1 ? base * 0.32 : base * 0.96, explained2 ? base * 0.28 : base * 0.92],
    })
  }
  return pool
}

const POOL = makePool()

/** The single TopBudget pass: keep a record only if its full serialization still fits. */
function scan(order: Rec[], budget: number) {
  const result: { rec: Rec; kept: boolean }[] = []
  let remaining = budget
  for (const rec of order) {
    const kept = rec.tokens <= remaining
    if (kept) remaining -= rec.tokens
    result.push({ rec, kept })
  }
  return result
}

type State = 'pending' | 'scanning' | 'kept' | 'evicted'

interface Scenario {
  label: string
  version: 0 | 1 | 2
  budget: number
}

/** The figure rotates through these settings, so no sliders are needed. */
const SCENARIOS: Scenario[] = [
  { label: 'Registry A · budget 1400', version: 0, budget: 1400 },
  { label: 'Registry A · budget 900', version: 0, budget: 900 },
  { label: 'Registry B · budget 1400', version: 1, budget: 1400 },
  { label: 'Registry C · budget 1400', version: 2, budget: 1400 },
]

const SCAN_MS = 420
const HOLD_MS = 3800

export function ResidualRetention() {
  const ref = useRef<HTMLElement>(null)
  const inView = useInView(ref)
  const reduced = useReducedMotion()
  const [view, setView] = useState({ scenario: 0, cursor: 0 })

  const { version, budget, label } = SCENARIOS[view.scenario]

  const order = useMemo(
    () => [...POOL].sort((a, b) => b.r[version] - a.r[version] || a.id.localeCompare(b.id)),
    [version],
  )
  const outcome = useMemo(() => scan(order, budget), [order, budget])

  const shown = reduced ? order.length : view.cursor
  const done = shown >= order.length

  useEffect(() => {
    if (!inView || reduced) return
    const id = window.setTimeout(
      () =>
        setView((v) =>
          v.cursor >= POOL.length
            ? { scenario: (v.scenario + 1) % SCENARIOS.length, cursor: 0 }
            : { ...v, cursor: v.cursor + 1 },
        ),
      done ? HOLD_MS : view.cursor === 0 ? 800 : SCAN_MS,
    )
    return () => window.clearTimeout(id)
  }, [inView, reduced, view, done])

  const next = () => setView((v) => ({ scenario: (v.scenario + 1) % SCENARIOS.length, cursor: 0 }))

  const kept = outcome.filter((o, i) => i < shown && o.kept)
  const used = kept.reduce((sum, o) => sum + o.rec.tokens, 0)

  const stateOf = (i: number): State => {
    if (i < shown) return outcome[i].kept ? 'kept' : 'evicted'
    if (i === shown) return 'scanning'
    return 'pending'
  }

  const fill = (r: number) => `color-mix(in srgb, var(--harness) ${Math.round(14 + (r / MAX_R) * 76)}%, var(--surface))`

  return (
    <Figure
      ref={ref}
      label="Figure 5"
      title="Residual retention"
      progress={{ count: SCENARIOS.length, index: view.scenario, label }}
      onAdvance={next}
      hint="Click for next setting"
      legend={
        <div className="legend">
          <span style={vars({ '--c': 'var(--harness)' })}>
            <i />
            carried over from <Tex tex={tx`\mathcal R^{\mathrm{task}}_{t-1}`} />
          </span>
          <span style={vars({ '--c': 'var(--ink)' })}>
            <i />
            new evidence from <Tex tex="A_t" />
          </span>
        </div>
      }
      caption={
        <>
          TopBudget selection under different committed skill registries and budgets. Registries A to C explain
          progressively more records. The residual is the target NLL under the frozen previous checkpoint executing
          the registry. Bar length shows token count and shading shows the residual. Values are illustrative.
        </>
      }
    >
      <div className="rr">
        <div className="rr-head">
          <b>
            Pool <Tex tex={tx`\mathcal U_t = \mathrm{Dedup}(\mathcal R^{\mathrm{task}}_{t-1} \cup A_t)`} />
          </b>
          <span>ordered by decreasing residual</span>
        </div>
        <div className="rr-pool">
          {order.map((rec, i) => {
            const state = stateOf(i)
            const r = rec.r[version]
            return (
              <div
                key={rec.id}
                className="rr-row"
                data-state={state}
                style={{ gridTemplateColumns: '128px minmax(0, 1fr) 118px' }}
              >
                <span>
                  {rec.id} · r {r.toFixed(2)}
                </span>
                <div className="rr-track">
                  <div
                    className="rr-rec"
                    style={vars({
                      width: `${(rec.tokens / MAX_TOKENS) * 100}%`,
                      '--fill': fill(r),
                      '--edge': rec.source === 'new' ? 'var(--ink)' : 'var(--harness)',
                    })}
                  />
                </div>
                <span className="rr-status">
                  {state === 'kept' && `kept · ${rec.tokens} tok`}
                  {state === 'evicted' && `skipped · ${rec.tokens} tok`}
                  {state === 'scanning' && `scanning · ${rec.tokens} tok`}
                  {state === 'pending' && `${rec.tokens} tok`}
                </span>
              </div>
            )
          })}
        </div>
        <div className="rr-bin">
          <div className="rr-head">
            <b>
              Retained <Tex tex={tx`\mathcal R^{\mathrm{task}}_t`} />
            </b>
            <span>
              {used} / {budget} tokens{done ? `, ${budget - used} unused` : ''}
            </span>
          </div>
          <div className="rr-bin-bar" aria-hidden="true">
            {kept.map((o) => (
              <i
                key={o.rec.id}
                style={vars({ width: `${(o.rec.tokens / budget) * 100}%`, '--fill': fill(o.rec.r[version]) })}
              />
            ))}
          </div>
        </div>
      </div>
    </Figure>
  )
}
