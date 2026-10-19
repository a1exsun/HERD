import { useState, useSyncExternalStore } from 'react'
import type { ReactNode } from 'react'
import { vars } from '../lib/css.ts'

/* Small SVG charts for the reported results. A chart is drawn in a fixed frame and scales with its
   container; colors come from the page tokens, so both themes work. Every chart has a hover layer:
   a crosshair and tooltip on line charts, a per-dot tooltip on scatters. */

const M = { left: 56, right: 18, top: 14, bottom: 42 }
const SIZE = { w: 440, h: 290 }
// On a phone the frame is about as wide as the card, so 11.5-unit text renders near 11.5px instead of shrinking.
const NARROW = { w: 330, h: 290 }
const NARROW_QUERY = '(max-width: 560px)'

function useNarrow() {
  return useSyncExternalStore(
    (cb) => {
      const mq = window.matchMedia(NARROW_QUERY)
      mq.addEventListener('change', cb)
      return () => mq.removeEventListener('change', cb)
    },
    () => window.matchMedia(NARROW_QUERY).matches,
    () => false,
  )
}

export interface Axis {
  min: number
  max: number
  ticks: number[]
  label?: string
  format?: (value: number) => string
}

interface Frame {
  w: number
  h: number
  right: number
  x: (value: number) => number
  y: (value: number) => number
}

function frame(x: Axis, y: Axis, right = M.right, narrow = false): Frame {
  const { w, h } = narrow ? NARROW : SIZE
  return {
    w, h, right,
    x: (v) => M.left + ((v - x.min) / (x.max - x.min)) * (w - M.left - right),
    y: (v) => h - M.bottom - ((v - y.min) / (y.max - y.min)) * (h - M.top - M.bottom),
  }
}

const fmt = (axis: Axis, v: number) => (axis.format ? axis.format(v) : String(v))
const r = (v: number) => Math.round(v * 10) / 10

function Axes({ x, y, f }: { x: Axis; y: Axis; f: Frame }) {
  return (
    <g className="chart-axes">
      {y.ticks.map((v) => (
        <g key={`y${v}`}>
          <line x1={M.left} x2={f.w - f.right} y1={r(f.y(v))} y2={r(f.y(v))} stroke="var(--line)" />
          <text x={M.left - 8} y={r(f.y(v)) + 4} textAnchor="end">{fmt(y, v)}</text>
        </g>
      ))}
      {x.ticks.map((v) => (
        <text key={`x${v}`} x={r(f.x(v))} y={f.h - M.bottom + 18} textAnchor="middle">{fmt(x, v)}</text>
      ))}
      {x.label && <text className="chart-label" x={(M.left + f.w - f.right) / 2} y={f.h - 6} textAnchor="middle">{x.label}</text>}
      {y.label && (
        <text className="chart-label" transform={`translate(13 ${(M.top + f.h - M.bottom) / 2}) rotate(-90)`} textAnchor="middle">
          {y.label}
        </text>
      )}
    </g>
  )
}

export interface Line {
  id: string
  label: string
  color: string
  points: { x: number; y: number }[]
  dashed?: boolean
  strong?: boolean
  /** Context lines: drawn thin and grey behind the others, left out of labels and the tooltip. */
  muted?: boolean
}

export interface Reference {
  y: number
  label: string
}

/** Spreads labels that would overlap vertically, keeping their order. */
function spread(ys: number[], gap: number) {
  const order = ys.map((y, i) => ({ y, i })).sort((a, b) => a.y - b.y)
  for (let k = 1; k < order.length; k++) order[k].y = Math.max(order[k].y, order[k - 1].y + gap)
  const out = [...ys]
  for (const o of order) out[o.i] = o.y
  return out
}

/** A line chart with direct end labels for the highlighted lines and a crosshair tooltip. */
export function LineChart({ lines, x, y, refs = [], label }: {
  lines: Line[]
  x: Axis
  y: Axis
  refs?: Reference[]
  label: string
}) {
  const narrow = useNarrow()
  const f = frame(x, y, narrow ? 92 : 104, narrow)
  const [at, setAt] = useState<number | null>(null)
  const shown = lines.filter((l) => !l.muted)
  const xs = [...new Set(lines.flatMap((l) => l.points.map((p) => p.x)))].sort((a, b) => a - b)
  const ends = shown.map((l) => l.points[l.points.length - 1])
  const labelY = spread(ends.map((p) => f.y(p.y) + 4), 13)
  const hovered = at === null ? [] : shown
    .map((l) => ({ l, p: l.points.find((p) => p.x === at) }))
    .filter((e): e is { l: Line; p: { x: number; y: number } } => e.p !== undefined)
    .sort((a, b) => b.p.y - a.p.y)
  const tipX = at === null ? 0 : f.x(at)
  const tipW = 150
  const tipLeft = tipX + 12 + tipW > f.w ? tipX - 12 - tipW : tipX + 12
  // Keep the box off the points it describes: below them when they sit in the upper half of the plot.
  const tipH = 22 + hovered.length * 16
  const upper = hovered.length > 0 && Math.min(...hovered.map(({ p }) => f.y(p.y))) < (M.top + f.h - M.bottom) / 2
  const tipTop = upper ? f.h - M.bottom - tipH - 6 : M.top
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} role="img" aria-label={label} onMouseLeave={() => setAt(null)}>
      <Axes x={x} y={y} f={f} />
      {refs.map((ref) => (
        <g key={ref.label}>
          <line x1={M.left} x2={f.w - f.right} y1={r(f.y(ref.y))} y2={r(f.y(ref.y))}
            stroke="var(--ink-2)" strokeDasharray="2 4" strokeWidth="1.4" />
          <text className="chart-ref" x={f.w - f.right + 6} y={r(f.y(ref.y)) + 4}>{ref.label}</text>
        </g>
      ))}
      {[...lines].sort((a, b) => Number(!a.muted) - Number(!b.muted) || Number(Boolean(a.strong)) - Number(Boolean(b.strong))).map((line) => (
        <g key={line.id}>
          <polyline points={line.points.map((p) => `${r(f.x(p.x))},${r(f.y(p.y))}`).join(' ')} fill="none"
            stroke={line.muted ? 'var(--line-strong)' : line.color} strokeWidth={line.strong ? 3 : line.muted ? 1.4 : 2}
            strokeLinejoin="round" strokeLinecap="round" strokeDasharray={line.dashed ? '6 4' : undefined} />
          {!line.muted && line.points.map((p) => (
            <circle key={p.x} cx={r(f.x(p.x))} cy={r(f.y(p.y))} r={line.strong ? 4.5 : 4} fill={line.color}
              stroke="var(--surface)" strokeWidth="2" />
          ))}
        </g>
      ))}
      {shown.map((line, i) => (
        <text key={`${line.id}-end`} className={line.strong ? 'chart-dot-label chart-strong' : 'chart-dot-label'}
          x={r(f.x(ends[i].x) + 10)} y={r(labelY[i])}>{line.label}</text>
      ))}
      {at !== null && (
        <g className="chart-tip" pointerEvents="none">
          <line x1={r(tipX)} x2={r(tipX)} y1={M.top} y2={f.h - M.bottom} stroke="var(--ink-2)" strokeWidth="1" />
          <rect x={r(tipLeft)} y={r(tipTop)} width={tipW} height={tipH} rx="6" />
          <text className="chart-tip-head" x={r(tipLeft + 10)} y={r(tipTop + 16)}>{x.label ?? ''} {fmt(x, at)}</text>
          {hovered.map(({ l, p }, k) => (
            <g key={l.id}>
              <circle cx={r(tipLeft + 14)} cy={r(tipTop + 30 + k * 16)} r="3.5" fill={l.color} />
              <text x={r(tipLeft + 24)} y={r(tipTop + 34 + k * 16)}>{l.label}</text>
              <text x={r(tipLeft + tipW - 10)} y={r(tipTop + 34 + k * 16)} textAnchor="end">{p.y.toFixed(3)}</text>
            </g>
          ))}
        </g>
      )}
      {xs.map((v, i) => {
        const lo = i === 0 ? M.left : (f.x(xs[i - 1]) + f.x(v)) / 2
        const hi = i === xs.length - 1 ? f.w - f.right : (f.x(v) + f.x(xs[i + 1])) / 2
        return <rect key={`hit${v}`} x={r(lo)} y={M.top} width={r(hi - lo)} height={f.h - M.top - M.bottom}
          fill="transparent" onMouseEnter={() => setAt(v)} />
      })}
    </svg>
  )
}

export interface Dot {
  id: string
  label: string
  color: string
  x: number
  y: number
  strong?: boolean
  /** A ring instead of a filled disc, the secondary encoding for neutral-coloured methods. */
  hollow?: boolean
}

interface Box { x0: number; y0: number; x1: number; y1: number }
const overlaps = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1

/** A scatter of methods. Each label takes the first nearby position that clears the other labels,
    the dots and the frame; a label pushed further away keeps a leader line to its dot. */
export function ScatterChart({ dots, x, y, label }: { dots: Dot[]; x: Axis; y: Axis; label: string }) {
  const f = frame(x, y, M.right, useNarrow())
  const [hover, setHover] = useState<string | null>(null)
  const plot: Box = { x0: M.left, y0: M.top, x1: f.w - M.right, y1: f.h - M.bottom }
  const discs = dots.map((dot) => {
    const cx = f.x(dot.x)
    const cy = f.y(dot.y)
    const d = dot.strong ? 9 : 7.5
    return { x0: cx - d, y0: cy - d, x1: cx + d, y1: cy + d }
  })
  const taken: Box[] = []
  const order = [...dots].sort((a, b) => Number(Boolean(b.strong)) - Number(Boolean(a.strong)))
  const labels = order.map((dot) => {
    const cx = f.x(dot.x)
    const cy = f.y(dot.y)
    const width = dot.label.length * (dot.strong ? 8 : 6.6) + 2
    const height = dot.strong ? 14 : 12
    const box = (lx: number, ly: number, anchor: 'start' | 'end' | 'middle'): Box => {
      const x0 = anchor === 'start' ? lx : anchor === 'end' ? lx - width : lx - width / 2
      return { x0, y0: ly - height + 3, x1: x0 + width, y1: ly + 3 }
    }
    const candidates: [number, number, 'start' | 'end' | 'middle'][] = []
    for (const step of [0, 1, 2, 3, 4]) {
      const lift = step * 13
      candidates.push([cx + 11, cy + 4 - lift, 'start'], [cx + 11, cy + 4 + lift, 'start'],
        [cx - 11, cy + 4 - lift, 'end'], [cx - 11, cy + 4 + lift, 'end'],
        [cx, cy - 12 - lift, 'middle'], [cx, cy + 20 + lift, 'middle'])
    }
    const fits = (b: Box) => b.x0 >= plot.x0 && b.x1 <= f.w && b.y0 >= 0 && b.y1 <= plot.y1
      && !taken.some((t) => overlaps(t, b)) && !discs.some((d) => overlaps(d, b))
    const [lx, ly, anchor] = candidates.find(([cxl, cyl, a]) => fits(box(cxl, cyl, a))) ?? candidates[0]
    taken.push(box(lx, ly, anchor))
    const far = Math.hypot(lx - cx, ly - 4 - cy) > 24
    return { dot, lx, ly, anchor, far, cx, cy }
  })
  const tip = dots.find((d) => d.id === hover)
  const tipW = 168
  const tipX = tip ? Math.min(f.x(tip.x) + 12, f.w - tipW - 4) : 0
  const tipY = tip ? Math.max(4, f.y(tip.y) - 58) : 0
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} role="img" aria-label={label}>
      <Axes x={x} y={y} f={f} />
      {labels.filter((l) => l.far).map((l) => (
        <line key={`${l.dot.id}-leader`} x1={r(l.cx)} y1={r(l.cy)} x2={r(l.anchor === 'end' ? l.lx + 2 : l.anchor === 'start' ? l.lx - 2 : l.lx)}
          y2={r(l.ly - 4)} stroke="var(--line-strong)" strokeWidth="1" />
      ))}
      {dots.map((dot) => (
        <g key={dot.id} onMouseEnter={() => setHover(dot.id)} onMouseLeave={() => setHover(null)}>
          <circle cx={r(f.x(dot.x))} cy={r(f.y(dot.y))} r="14" fill="transparent" />
          <circle cx={r(f.x(dot.x))} cy={r(f.y(dot.y))} r={(dot.strong ? 7 : 5.5) + (hover === dot.id ? 2 : 0)}
            fill={dot.hollow ? 'var(--surface)' : dot.color} stroke={dot.hollow ? dot.color : 'var(--surface)'}
            strokeWidth={dot.hollow ? 2.2 : 2} />
        </g>
      ))}
      {labels.map(({ dot, lx, ly, anchor }) => (
        <text key={dot.id} className={dot.strong ? 'chart-dot-label chart-strong' : 'chart-dot-label'} x={r(lx)} y={r(ly)}
          textAnchor={anchor}>{dot.label}</text>
      ))}
      {tip && (
        <g className="chart-tip" pointerEvents="none">
          <rect x={r(tipX)} y={r(tipY)} width={tipW} height="50" rx="6" />
          <text className="chart-tip-head" x={r(tipX + 10)} y={r(tipY + 17)}>{tip.label}</text>
          <text x={r(tipX + 10)} y={r(tipY + 32)}>{x.label?.replace(/ [→↓↑]$/, '')}: {tip.x.toFixed(3)}</text>
          <text x={r(tipX + 10)} y={r(tipY + 45)}>{y.label?.replace(/ [→↓↑]$/, '')}: {tip.y.toFixed(3)}</text>
        </g>
      )}
    </svg>
  )
}

export function Legend({ items }: { items: { id: string; label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="experiment-legend">
      {items.map((item) => (
        <span key={item.id} data-dashed={item.dashed ? 'true' : undefined} style={vars({ '--series-color': item.color })}>{item.label}</span>
      ))}
    </div>
  )
}

/** A chart card: title, the chart, a caption and the exact values behind it. */
export function ChartFigure({ title, children, caption, legend, table, wide }: {
  title: string
  children: ReactNode
  caption: ReactNode
  legend?: ReactNode
  table?: ReactNode
  wide?: boolean
}) {
  return (
    <figure className={wide ? 'experiment-chart experiment-wide' : 'experiment-chart'}>
      <h4>{title}</h4>
      {legend}
      {children}
      <figcaption>{caption}</figcaption>
      {table && (
        <details className="experiment-details">
          <summary>Data table</summary>
          <div className="experiment-scroll" tabIndex={0} role="region" aria-label={`${title} data`}>{table}</div>
        </details>
      )}
    </figure>
  )
}
