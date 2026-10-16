import { useEffect, useState, useSyncExternalStore } from 'react'

const LINKS = [
  { id: 'abstract', label: 'Abstract' },
  { id: 'method', label: 'Method' },
  { id: 'experiments', label: 'Experiments' },
  { id: 'baselines', label: 'Baselines' },
]

type Theme = 'light' | 'dark'

const DARK = '(prefers-color-scheme: dark)'

function subscribeScheme(cb: () => void) {
  const mq = window.matchMedia(DARK)
  mq.addEventListener('change', cb)
  return () => mq.removeEventListener('change', cb)
}

/** An explicit choice made on this page, if any. Without one the page follows the system. */
function storedChoice(): Theme | null {
  const set = document.documentElement.getAttribute('data-theme')
  return set === 'light' || set === 'dark' ? set : null
}

export function Nav() {
  const [active, setActive] = useState('')
  const [progress, setProgress] = useState(0)
  const [choice, setChoice] = useState<Theme | null>(storedChoice)
  const systemDark = useSyncExternalStore(
    subscribeScheme,
    () => window.matchMedia(DARK).matches,
    () => false,
  )
  const theme: Theme = choice ?? (systemDark ? 'dark' : 'light')

  useEffect(() => {
    const sections = LINKS.map((l) => document.getElementById(l.id)).filter((el): el is HTMLElement => el !== null)
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight
      setProgress(max > 0 ? Math.min(1, window.scrollY / max) : 0)
      const line = window.innerHeight * 0.35
      let found = ''
      for (const el of sections) {
        if (el.getBoundingClientRect().top <= line) found = el.id
      }
      setActive(found)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
    }
  }, [])

  const toggleTheme = () => {
    const next: Theme = theme === 'dark' ? 'light' : 'dark'
    document.documentElement.setAttribute('data-theme', next)
    try {
      localStorage.setItem('hero-theme', next)
    } catch {
      // Storage can be unavailable; the choice then lasts for this visit only.
    }
    setChoice(next)
  }

  return (
    <header className="nav">
      <div className="wrap nav-inner">
        <a className="brand" href="#top" aria-label="HERD, back to top">
          <span className="brand-mark" aria-hidden="true">
            <i />
            <i />
          </span>
          HERD
        </a>
        <nav className="nav-links" aria-label="Sections">
          {LINKS.map((l) => (
            <a key={l.id} href={`#${l.id}`} aria-current={active === l.id ? 'true' : undefined}>
              {l.label}
            </a>
          ))}
        </nav>
        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
        >
          {theme === 'dark' ? (
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <circle cx="10" cy="10" r="3.6" fill="currentColor" />
              <g stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
                <path d="M10 2.2v2M10 15.8v2M2.2 10h2M15.8 10h2M4.5 4.5l1.4 1.4M14.1 14.1l1.4 1.4M4.5 15.5l1.4-1.4M14.1 5.9l1.4-1.4" />
              </g>
            </svg>
          ) : (
            <svg viewBox="0 0 20 20" aria-hidden="true">
              <path d="M16.5 12.2A7 7 0 0 1 7.8 3.5a7 7 0 1 0 8.7 8.7z" fill="currentColor" />
            </svg>
          )}
        </button>
      </div>
      <div className="nav-progress" style={{ transform: `scaleX(${progress})` }} aria-hidden="true" />
    </header>
  )
}
