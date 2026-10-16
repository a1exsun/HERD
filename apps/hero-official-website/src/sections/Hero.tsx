import { useEffect, useState } from 'react'
import { StreamLoop } from '../demos/StreamLoop.tsx'
import { vars } from '../lib/css.ts'
import { useReducedMotion } from '../lib/hooks.ts'

const AUTHORS = ['Xun Sun', 'Trang Vu', 'Thanh Thi Nguyen']

const LETTERS = [
  { letter: 'H', word: 'Harness', rest: 'arness', color: 'var(--harness)' },
  { letter: 'E', word: 'Evolving', rest: 'volving', color: 'var(--guard)' },
  { letter: 'R', word: 'Recursive', rest: 'ecursive', color: 'var(--predictor)' },
  { letter: 'D', word: 'Distillation', rest: 'istillation', color: 'var(--optim)' },
]

export function Hero() {
  const reduced = useReducedMotion()
  const [on, setOn] = useState(0)
  const [held, setHeld] = useState(false)

  useEffect(() => {
    if (reduced || held) return
    const id = window.setInterval(() => setOn((i) => (i + 1) % LETTERS.length), 1800)
    return () => window.clearInterval(id)
  }, [reduced, held])

  return (
    <section className="hero" id="top">
      <div className="wrap">
        <div className="hero-top">
          <div>
            <span className="eyebrow">Continual learning over NLP task streams</span>
            <h1 className="wordmark" aria-label="HERD">
              {LETTERS.map((l, i) => (
                <button
                  key={l.letter}
                  type="button"
                  data-on={i === on}
                  style={vars({ '--letter': l.color })}
                  onMouseEnter={() => {
                    setHeld(true)
                    setOn(i)
                  }}
                  onMouseLeave={() => setHeld(false)}
                  onFocus={() => {
                    setHeld(true)
                    setOn(i)
                  }}
                  onBlur={() => setHeld(false)}
                  aria-label={`${l.letter} stands for ${l.word}`}
                >
                  {l.letter}
                </button>
              ))}
            </h1>
            <div className="expansion" aria-hidden="true">
              {LETTERS.map((l, i) => (
                <span key={l.letter} data-on={i === on} style={vars({ '--letter': l.color })}>
                  <b>{l.letter}</b>
                  {l.rest}
                </span>
              ))}
            </div>
            <p className="hero-title">HERD: Harness-Evolving Recursive Distillation for Continual Learning</p>
          </div>

          <div className="hero-side">
            <p className="hero-lede">
              A continual-learning framework that recursively distills an evolving, bounded harness into a standalone
              predictor under two behavioral drift budgets.
            </p>
            <div className="byline">
              <ul className="authors" aria-label="Authors">
                {AUTHORS.map((name) => (
                  <li key={name}>{name}</li>
                ))}
              </ul>
              <span>Paper under review</span>
            </div>
            <div className="actions">
              <a className="btn btn-primary" href="#method">
                Method
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path
                    d="M8 3v10M3.5 8.5L8 13l4.5-4.5"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              </a>
              <a className="btn" href="/herd.pdf" target="_blank" rel="noreferrer">Paper (PDF)</a>
              <a className="btn" href="https://github.com/a1exsun/HERD/tree/arr" target="_blank" rel="noreferrer">Code</a>
            </div>
          </div>
        </div>

        <div className="hero-figure">
          <StreamLoop />
        </div>
      </div>
    </section>
  )
}
