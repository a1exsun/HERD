import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react'
import type { RefObject } from 'react'

const REDUCED = '(prefers-reduced-motion: reduce)'

function subscribeReduced(cb: () => void) {
  const mq = window.matchMedia(REDUCED)
  mq.addEventListener('change', cb)
  return () => mq.removeEventListener('change', cb)
}

export function useReducedMotion(): boolean {
  return useSyncExternalStore(
    subscribeReduced,
    () => window.matchMedia(REDUCED).matches,
    () => false,
  )
}

/** True while the referenced element intersects the viewport. */
export function useInView<T extends Element>(ref: RefObject<T | null>, margin = '0px'): boolean {
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) setInView(e.isIntersecting)
      },
      { rootMargin: margin, threshold: 0.15 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [ref, margin])
  return inView
}

export interface Carousel {
  step: number
  /** Moves to the next scene and restarts the timer, so a click is never followed by an instant jump. */
  advance: () => void
  goto: (i: number) => void
}

interface CarouselOptions {
  /** The loop only runs while this is true, usually "figure is on screen". */
  active?: boolean
  /** Holds the current scene, for example while the pointer rests on the figure. */
  paused?: boolean
}

/** A scene loop without visible controls: it plays by itself and a click moves it forward. */
export function useCarousel(count: number, intervalMs: number, options: CarouselOptions = {}): Carousel {
  const { active = true, paused = false } = options
  const reduced = useReducedMotion()
  const [step, setStep] = useState(0)
  const [restart, setRestart] = useState(0)

  const running = active && !paused && !reduced

  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => setStep((s) => (s + 1) % count), intervalMs)
    return () => window.clearInterval(id)
  }, [running, count, intervalMs, restart])

  const advance = useCallback(() => {
    setStep((s) => (s + 1) % count)
    setRestart((n) => n + 1)
  }, [count])

  const goto = useCallback(
    (i: number) => {
      setStep(((i % count) + count) % count)
      setRestart((n) => n + 1)
    },
    [count],
  )

  return { step, advance, goto }
}

/** Calls the callback on every animation frame while enabled. */
export function useFrameLoop(enabled: boolean, onFrame: (dtMs: number) => void) {
  const cb = useRef(onFrame)
  useEffect(() => {
    cb.current = onFrame
  }, [onFrame])
  useEffect(() => {
    if (!enabled) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.min(64, now - last)
      last = now
      cb.current(dt)
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [enabled])
}
