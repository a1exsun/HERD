import type { MouseEvent, ReactNode } from 'react'

export type Kind = 'plain' | 'harness' | 'predictor' | 'optim' | 'guard' | 'deploy'

const KINDS: Kind[] = ['plain', 'harness', 'predictor', 'optim', 'guard', 'deploy']

/** Arrowheads shared by every diagram on the page. Rendered once at the app root. */
export function SvgDefs() {
  return (
    <svg aria-hidden="true" focusable="false" style={{ position: 'absolute', width: 0, height: 0 }}>
      <defs>
        {KINDS.map((k) => (
          <marker
            key={k}
            id={`arr-${k}`}
            viewBox="0 0 10 10"
            refX="8.6"
            refY="5"
            markerWidth="9"
            markerHeight="9"
            markerUnits="userSpaceOnUse"
            orient="auto-start-reverse"
          >
            <path d="M0.6 1 L9 5 L0.6 9 Z" className={`head head-${k}`} />
          </marker>
        ))}
      </defs>
    </svg>
  )
}

interface MProps {
  /** Mini notation: [~]base[_{sub}][^{sup}]. "~" draws a bar, "cal:" selects the calligraphic face. */
  s: string
  size?: number
}

const PATTERN = /^(~)?(cal:)?([^_^]+)(?:_\{([^}]*)\})?(?:\^\{([^}]*)\})?$/
const WORD = /^[a-z]{2,}$/i

/** A math symbol for use inside an SVG <text>, set in the same faces as the KaTeX formulas. */
export function M({ s, size = 15 }: MProps) {
  const match = PATTERN.exec(s)
  if (!match) return <tspan className="m">{s}</tspan>
  const [, bar, cal, base, sub, sup] = match
  const small = size * 0.72
  const drop = size * 0.26
  const rise = size * 0.4
  const baseClass = cal ? 'm-cal' : bar ? 'm m-bar' : 'm'
  return (
    <>
      <tspan className={baseClass} fontSize={size}>
        {base}
      </tspan>
      {sub !== undefined && (
        <>
          <tspan className={WORD.test(sub) ? 'm-rm' : 'm'} fontSize={small} dy={drop}>
            {sub}
          </tspan>
          <tspan fontSize={small} dy={-drop}>
            {'​'}
          </tspan>
        </>
      )}
      {sup !== undefined && (
        <>
          <tspan className={WORD.test(sup) ? 'm-rm' : 'm'} fontSize={small} dy={-rise}>
            {sup}
          </tspan>
          <tspan fontSize={small} dy={rise}>
            {'​'}
          </tspan>
        </>
      )}
    </>
  )
}

interface NodeProps {
  /** Center of the node. */
  x: number
  y: number
  w: number
  h: number
  kind?: Kind
  /** Optional darker title band across the top of the node. */
  band?: number
  r?: number
  active?: boolean
  dim?: boolean
  hidden?: boolean
  className?: string
  /** Makes the node a direct target. The click does not also advance the surrounding figure. */
  onClick?: () => void
  children?: ReactNode
}

export function Node({
  x,
  y,
  w,
  h,
  kind = 'plain',
  band,
  r = 10,
  active,
  dim,
  hidden,
  className,
  onClick,
  children,
}: NodeProps) {
  const handleClick = onClick
    ? (event: MouseEvent) => {
        event.stopPropagation()
        onClick()
      }
    : undefined
  const bandPath =
    band !== undefined
      ? `M0 ${r} a${r} ${r} 0 0 1 ${r} ${-r} h${w - 2 * r} a${r} ${r} 0 0 1 ${r} ${r} v${band - r} h${-w} z`
      : null
  // The CSS entrance animation sets a transform, so placement lives on a separate outer group.
  return (
    <g transform={`translate(${x - w / 2} ${y - h / 2})`}>
      <g
        className={`node${className ? ` ${className}` : ''}`}
        data-kind={kind}
        data-active={active || undefined}
        data-dim={dim || undefined}
        data-hidden={hidden || undefined}
        data-target={onClick ? 'true' : undefined}
        onClick={handleClick}
      >
        <rect className="under" width={w} height={h} rx={r} />
        <rect className="face" width={w} height={h} rx={r} />
        {bandPath && <path className="band" d={bandPath} />}
        <g transform={`translate(${w / 2} ${h / 2})`}>{children}</g>
      </g>
    </g>
  )
}

interface EdgeProps {
  d: string
  kind?: Kind
  flow?: boolean
  dim?: boolean
  hidden?: boolean
  head?: boolean
}

export function Edge({ d, kind = 'plain', flow, dim, hidden, head = true }: EdgeProps) {
  return (
    <path
      className="edge"
      d={d}
      data-kind={kind}
      data-flow={flow || undefined}
      data-dim={dim || undefined}
      data-hidden={hidden || undefined}
      markerEnd={head ? `url(#arr-${kind})` : undefined}
    />
  )
}
