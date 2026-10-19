import { useEffect, useState } from 'react'
import { StreamLoop } from '../demos/StreamLoop.tsx'
import { vars } from '../lib/css.ts'
import { useReducedMotion } from '../lib/hooks.ts'

const AUTHORS = [
  { name: 'Xun Sun', url: 'https://xsun.work' },
  { name: 'Thanh Thi Nguyen', url: 'https://sites.google.com/view/thanh-thi-nguyen' },
  { name: 'Thuy-Trang Vu', url: 'https://trangvu.github.io/' },
]

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
          </div>

          <div className="hero-side">
            <p className="hero-lede">
              A hybrid continual learning framework that learns efficiently in the harness and internalizes experience
              into the weights by drift-constrained distillation with dual anchors.
            </p>
            <div className="byline">
              <ul className="authors" aria-label="Authors">
                {AUTHORS.map((a) => (
                  <li key={a.name}><a href={a.url} target="_blank" rel="noreferrer">{a.name}</a></li>
                ))}
              </ul>
              <a href="https://www.monash.edu/" target="_blank" rel="noreferrer">Monash University</a>
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
              <span className="btn" aria-disabled="true">Paper · arXiv soon</span>
              <a className="btn" href="https://github.com/a1exsun/HERD" target="_blank" rel="noreferrer">Code</a>
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
