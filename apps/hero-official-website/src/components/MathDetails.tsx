import type { ReactNode } from 'react'
import { Tex } from './Tex.tsx'

interface MathDetailsProps {
  title: string
  count: number
  children: ReactNode
}

/** Collapsed by default: the page explains with figures first and keeps the formulas one click away. */
export function MathDetails({ title, count, children }: MathDetailsProps) {
  return (
    <details className="math reveal">
      <summary>
        <span className="sigma" aria-hidden="true">
          f
        </span>
        <span>{title}</span>
        <span className="hint">
          {count} {count === 1 ? 'equation' : 'equations'}
        </span>
        <svg className="chev" viewBox="0 0 16 16" aria-hidden="true">
          <path
            d="M3 6l5 5 5-5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </summary>
      <div className="math-body">{children}</div>
    </details>
  )
}

interface EqProps {
  name: string
  tex: string
  children: ReactNode
}

export function Eq({ name, tex, children }: EqProps) {
  return (
    <div className="eq">
      <div className="eq-text">
        <b>{name}</b>
        <span>{children}</span>
      </div>
      <Tex tex={tex} block />
    </div>
  )
}
