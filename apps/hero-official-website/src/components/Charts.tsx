import type { ReactNode } from 'react'
import { vars } from '../lib/css.ts'

/* Small SVG charts for the experiment record. A chart is drawn in a fixed frame (wide charts in a
   wider one, so their text keeps its size) and scales with its container; colors come from the
   page tokens, so both themes work. */

const M = { left: 56, right: 18, top: 14, bottom: 42 }
const SIZES = { normal: { w: 440, h: 290 }, wide: { w: 860, h: 300 } }

export interface Axis {
  min: number
  max: number
  ticks: number[]
  label?: string
  format?: (value: number) => string
  /** Logarithmic scale; `min` must be positive. */
  log?: boolean
}

interface Frame {
  w: number
  h: number
  x: (value: number) => number
  y: (value: number) => number
}

function frame(x: Axis, y: Axis, wide?: boolean): Frame {
  const { w, h } = SIZES[wide ? 'wide' : 'normal']
  return {
    w, h,
    x: (v) => M.left + ((v - x.min) / (x.max - x.min)) * (w - M.left - M.right),
    y: (v) => h - M.bottom - (y.log
      ? (Math.log10(Math.max(v, y.min)) - Math.log10(y.min)) / (Math.log10(y.max) - Math.log10(y.min))
      : (v - y.min) / (y.max - y.min)) * (h - M.top - M.bottom),
  }
}

const fmt = (axis: Axis, v: number) => (axis.format ? axis.format(v) : String(v))
const r = (v: number) => Math.round(v * 10) / 10

function Axes({ x, y, f }: { x: Axis; y: Axis; f: Frame }) {
  return (
    <g className="chart-axes">
      {y.ticks.map((v) => (
        <g key={`y${v}`}>
          <line x1={M.left} x2={f.w - M.right} y1={r(f.y(v))} y2={r(f.y(v))} stroke="var(--line)" />
          <text x={M.left - 8} y={r(f.y(v)) + 4} textAnchor="end">{fmt(y, v)}</text>
        </g>
      ))}
      {x.ticks.map((v) => (
        <text key={`x${v}`} x={r(f.x(v))} y={f.h - M.bottom + 18} textAnchor="middle">{fmt(x, v)}</text>
      ))}
      {x.label && <text className="chart-label" x={(M.left + f.w - M.right) / 2} y={f.h - 6} textAnchor="middle">{x.label}</text>}
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
  /** `lo` and `hi` bound the shaded band; without them the band is `y ± sd`. */
  points: { x: number; y: number; sd?: number | null; lo?: number; hi?: number }[]
  dashed?: boolean
  strong?: boolean
  markers?: boolean
}

export interface Reference {
  y: number
  label: string
  color?: string
}

interface LineChartProps {
  lines: Line[]
  x: Axis
  y: Axis
  refs?: Reference[]
  label: string
  wide?: boolean
  /** Faint vertical guides, such as stage boundaries. */
  guides?: number[]
}

/** A line chart. A wide chart is also drawn in the normal frame, which narrow screens show instead. */
export function LineChart(props: LineChartProps) {
  if (!props.wide) return <LineSvg {...props} />
  return (
    <>
      <LineSvg {...props} className="chart-wide" />
      <LineSvg {...props} wide={false} className="chart-narrow" />
    </>
  )
}

function LineSvg({ lines, x, y, refs = [], label, wide, guides = [], className }: LineChartProps & { className?: string }) {
  const f = frame(x, y, wide)
  const clamp = (v: number) => Math.max(y.min, Math.min(y.max, v))
  return (
    <svg className={className} viewBox={`0 0 ${f.w} ${f.h}`} role="img" aria-label={label}>
      <Axes x={x} y={y} f={f} />
      {guides.map((g) => (
        <line key={`g${g}`} x1={r(f.x(g))} x2={r(f.x(g))} y1={M.top} y2={f.h - M.bottom} stroke="var(--line)" strokeWidth="0.6" />
      ))}
      {refs.map((ref) => (
        <g key={ref.label}>
          <line x1={M.left} x2={f.w - M.right} y1={r(f.y(ref.y))} y2={r(f.y(ref.y))}
            stroke={ref.color ?? 'var(--ink-2)'} strokeDasharray="6 4" strokeWidth="1.4" />
          <text className="chart-ref" x={f.w - M.right} y={r(f.y(ref.y)) - 6} textAnchor="end" style={{ fill: ref.color ?? 'var(--ink-2)' }}>
            {ref.label}
          </text>
        </g>
      ))}
      {lines.map((line) => {
        const band = line.points.filter((p) => p.sd || p.hi !== undefined)
        const hi = (p: Line['points'][number]) => clamp(p.hi ?? p.y + p.sd!)
        const lo = (p: Line['points'][number]) => clamp(p.lo ?? p.y - p.sd!)
        return band.length > 1 ? (
          <polygon key={`${line.id}-band`} fill={line.color} fillOpacity="0.16" stroke="none"
            points={[...band.map((p) => `${r(f.x(p.x))},${r(f.y(hi(p)))}`),
              ...[...band].reverse().map((p) => `${r(f.x(p.x))},${r(f.y(lo(p)))}`)].join(' ')} />
        ) : null
      })}
      {lines.map((line) => (
        <g key={line.id}>
          <polyline points={line.points.map((p) => `${r(f.x(p.x))},${r(f.y(p.y))}`).join(' ')} fill="none"
            stroke={line.color} strokeWidth={line.strong ? 3 : 1.8} strokeLinejoin="round" strokeLinecap="round"
            strokeDasharray={line.dashed ? '5 5' : undefined} />
          {line.markers !== false && line.points.map((p) => (
            <circle key={p.x} cx={r(f.x(p.x))} cy={r(f.y(p.y))} r={line.strong ? 3.6 : 2.8} fill={line.color}>
              <title>{`${line.label}, ${fmt(x, p.x)}: ${fmt(y, p.y)}${p.sd ? ` ± ${fmt(y, p.sd)}` : ''}`}</title>
            </circle>
          ))}
        </g>
      ))}
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
}

interface Box { x0: number; y0: number; x1: number; y1: number }
const overlaps = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1

/** A scatter of methods. Each label takes the first nearby position that clears the other labels,
    the dots and the frame; a label pushed further away keeps a leader line to its dot. */
export function ScatterChart({ dots, x, y, label }: { dots: Dot[]; x: Axis; y: Axis; label: string }) {
  const f = frame(x, y)
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
  return (
    <svg viewBox={`0 0 ${f.w} ${f.h}`} role="img" aria-label={label}>
      <Axes x={x} y={y} f={f} />
      {labels.filter((l) => l.far).map((l) => (
        <line key={`${l.dot.id}-leader`} x1={r(l.cx)} y1={r(l.cy)} x2={r(l.anchor === 'end' ? l.lx + 2 : l.anchor === 'start' ? l.lx - 2 : l.lx)}
          y2={r(l.ly - 4)} stroke="var(--line-strong)" strokeWidth="1" />
      ))}
      {dots.map((dot) => (
        <circle key={dot.id} cx={r(f.x(dot.x))} cy={r(f.y(dot.y))} r={dot.strong ? 7 : 5.5} fill={dot.color}
          fillOpacity={dot.strong ? 1 : 0.85} stroke="var(--surface)" strokeWidth="1.5">
          <title>{`${dot.label}: ${fmt(x, dot.x)}, ${fmt(y, dot.y)}`}</title>
        </circle>
      ))}
      {labels.map(({ dot, lx, ly, anchor }) => (
        <text key={dot.id} className={dot.strong ? 'chart-dot-label chart-strong' : 'chart-dot-label'} x={r(lx)} y={r(ly)}
          textAnchor={anchor} style={{ fill: dot.strong ? dot.color : undefined }}>{dot.label}</text>
      ))}
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
      {children}
      <figcaption>
        {legend}
        {caption}
      </figcaption>
      {table && (
        <details className="experiment-details">
          <summary>Data table</summary>
          <div className="experiment-scroll" tabIndex={0} role="region" aria-label={`${title} data`}>{table}</div>
        </details>
      )}
    </figure>
  )
}

/** A small chart for small multiples: one series with its band, an optional dashed reference. */
export function MiniChart({ points, x, y, color, reference, label }: {
  points: { x: number; y: number; sd?: number | null }[]
  x: Axis
  y: Axis
  color: string
  reference?: number
  label: string
}) {
  const w = 220
  const h = 140
  const m = { left: 30, right: 8, top: 8, bottom: 22 }
  const px = (v: number) => m.left + ((v - x.min) / (x.max - x.min)) * (w - m.left - m.right)
  const py = (v: number) => h - m.bottom - ((Math.max(y.min, Math.min(y.max, v)) - y.min) / (y.max - y.min)) * (h - m.top - m.bottom)
  const band = points.filter((p) => p.sd)
  return (
    <svg className="chart-mini" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      {y.ticks.map((v) => (
        <g key={`y${v}`}>
          <line x1={m.left} x2={w - m.right} y1={r(py(v))} y2={r(py(v))} stroke="var(--line)" />
          <text x={m.left - 5} y={r(py(v)) + 3.5} textAnchor="end">{fmt(y, v)}</text>
        </g>
      ))}
      {x.ticks.map((v) => <text key={`x${v}`} x={r(px(v))} y={h - 6} textAnchor="middle">{fmt(x, v)}</text>)}
      {reference !== undefined && (
        <line x1={m.left} x2={w - m.right} y1={r(py(reference))} y2={r(py(reference))} stroke="var(--muted)" strokeDasharray="4 3" />
      )}
      {band.length > 1 && (
        <polygon fill={color} fillOpacity="0.18" points={[...band.map((p) => `${r(px(p.x))},${r(py(p.y + p.sd!))}`),
          ...[...band].reverse().map((p) => `${r(px(p.x))},${r(py(p.y - p.sd!))}`)].join(' ')} />
      )}
      <polyline points={points.map((p) => `${r(px(p.x))},${r(py(p.y))}`).join(' ')} fill="none" stroke={color} strokeWidth="2"
        strokeLinejoin="round" />
      {points.map((p) => (
        <circle key={p.x} cx={r(px(p.x))} cy={r(py(p.y))} r="2.6" fill={color}>
          <title>{`${fmt(x, p.x)}: ${fmt(y, p.y)}${p.sd ? ` ± ${fmt(y, p.sd)}` : ''}`}</title>
        </circle>
      ))}
    </svg>
  )
}

export interface DumbbellRow {
  id: string
  label: string
  from: number
  to: number
  sd?: number
}

/** One row per item: a hollow marker at `from`, a filled marker at `to`, the line between them colored by
    direction, and a band spanning one standard deviation around `to`. */
export function Dumbbell({ rows, x, label, fromLabel, toLabel }: {
  rows: DumbbellRow[]
  x: Axis
  label: string
  fromLabel: string
  toLabel: string
}) {
  const w = 640
  const rowH = 20
  const m = { left: 132, right: 16, top: 8, bottom: 30 }
  const h = m.top + rows.length * rowH + m.bottom
  const px = (v: number) => m.left + ((v - x.min) / (x.max - x.min)) * (w - m.left - m.right)
  return (
    <svg className="chart-dumbbell" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      {x.ticks.map((v) => (
        <g key={v}>
          <line x1={r(px(v))} x2={r(px(v))} y1={m.top} y2={h - m.bottom} stroke="var(--line)" />
          <text x={r(px(v))} y={h - m.bottom + 16} textAnchor="middle">{fmt(x, v)}</text>
        </g>
      ))}
      {rows.map((row, i) => {
        const cy = m.top + i * rowH + rowH / 2
        const color = row.to >= row.from ? 'var(--deploy)' : 'var(--optim)'
        return (
          <g key={row.id}>
            <text className="chart-row-label" x={m.left - 10} y={cy + 4} textAnchor="end">{row.label}</text>
            <line x1={r(px(row.from))} x2={r(px(row.to))} y1={cy} y2={cy} stroke={color} strokeWidth="2.4" />
            {row.sd !== undefined && <line x1={r(px(Math.max(x.min, row.to - row.sd)))} x2={r(px(Math.min(x.max, row.to + row.sd)))}
              y1={cy} y2={cy} stroke={color} strokeWidth="7" strokeOpacity="0.3" />}
            <circle cx={r(px(row.from))} cy={cy} r="4.2" fill="var(--surface)" stroke="var(--ink-2)" strokeWidth="1.5">
              <title>{`${row.label}, ${fromLabel}: ${fmt(x, row.from)}`}</title>
            </circle>
            <circle cx={r(px(row.to))} cy={cy} r="4.2" fill={color}>
              <title>{`${row.label}, ${toLabel}: ${fmt(x, row.to)}`}</title>
            </circle>
          </g>
        )
      })}
      {x.label && <text className="chart-label" x={(m.left + w - m.right) / 2} y={h - 4} textAnchor="middle">{x.label}</text>}
    </svg>
  )
}

export interface ForestRow {
  id: string
  label: string
  value: number
  lo: number
  hi: number
}

/** Estimates with intervals around a zero line; `better` says which side favors the first method. */
export function Forest({ rows, label, better, format }: {
  rows: ForestRow[]
  label: string
  better: 'right' | 'left'
  format: (value: number) => string
}) {
  const w = 440
  const rowH = 26
  const m = { left: 110, right: 30, top: 22, bottom: 30 }
  const h = m.top + rows.length * rowH + m.bottom
  // A symmetric axis on round ticks, about two on each side of zero.
  const reach = Math.max(...rows.flatMap((row) => [Math.abs(row.lo), Math.abs(row.hi)]), 1e-3)
  const magnitude = 10 ** Math.floor(Math.log10(reach / 2))
  const step = [1, 2, 2.5, 5, 10].map((k) => k * magnitude).find((k) => reach / k <= 2.5)!
  const extent = Math.ceil(reach / step) * step
  const px = (v: number) => m.left + ((v + extent) / (2 * extent)) * (w - m.left - m.right)
  const decimals = Math.max(0, -Math.floor(Math.log10(step) + 1e-9) + (String(step / magnitude).includes('.') ? 1 : 0))
  const ticks = Array.from({ length: Math.round((2 * extent) / step) + 1 }, (_, i) => Math.round((-extent + i * step) / magnitude) * magnitude)
  return (
    <svg className="chart-forest" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      <text className="chart-side" x={r(px(-extent / 2))} y={14} textAnchor="middle">{better === 'left' ? 'favours HERD' : 'favours baseline'}</text>
      <text className="chart-side" x={r(px(extent / 2))} y={14} textAnchor="middle">{better === 'right' ? 'favours HERD' : 'favours baseline'}</text>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={r(px(v))} x2={r(px(v))} y1={m.top} y2={h - m.bottom} stroke={v === 0 ? 'var(--ink-2)' : 'var(--line)'} />
          <text x={r(px(v))} y={h - m.bottom + 16} textAnchor="middle">{v === 0 ? '0' : `${v > 0 ? '+' : '−'}${Math.abs(v).toFixed(decimals)}`}</text>
        </g>
      ))}
      {rows.map((row, i) => {
        const cy = m.top + i * rowH + rowH / 2
        const favours = better === 'right' ? row.lo > 0 : row.hi < 0
        const against = better === 'right' ? row.hi < 0 : row.lo > 0
        const color = favours ? 'var(--deploy)' : against ? 'var(--optim)' : 'var(--muted)'
        return (
          <g key={row.id}>
            <text className="chart-row-label" x={m.left - 10} y={cy + 4} textAnchor="end">{row.label}</text>
            <line x1={r(px(row.lo))} x2={r(px(row.hi))} y1={cy} y2={cy} stroke={color} strokeWidth="2" />
            <rect x={r(px(row.value)) - 4} y={cy - 4} width="8" height="8" fill={color}>
              <title>{`HERD − ${row.label}: ${format(row.value)} [${format(row.lo)}, ${format(row.hi)}]`}</title>
            </rect>
          </g>
        )
      })}
    </svg>
  )
}
