import type { KeyboardEvent, ReactNode, Ref } from 'react'

export interface Progress {
  count: number
  index: number
  /** Names the scene on show, for example a stage or a scenario. */
  label: string
}

interface FigureProps {
  label: string
  title: string
  caption: ReactNode
  /** Marks figures whose values illustrate the mechanism and are not measurements. */
  schematic?: boolean
  note?: ReactNode
  legend?: ReactNode
  progress?: Progress
  /** Figures play by themselves. A click on the figure moves to the next scene. */
  onAdvance?: () => void
  onHover?: (hovering: boolean) => void
  hint?: string
  children: ReactNode
  ref?: Ref<HTMLElement>
}

export function Figure({
  label,
  title,
  caption,
  schematic = true,
  note,
  legend,
  progress,
  onAdvance,
  onHover,
  hint = 'Click for next',
  children,
  ref,
}: FigureProps) {
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!onAdvance || event.target !== event.currentTarget) return
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onAdvance()
    }
  }

  return (
    <figure className="fig reveal" ref={ref}>
      <div className="fig-head">
        <span className="fig-label">{label}</span>
        <span className="fig-title">{title}</span>
        <span className="fig-meta">
          {progress && (
            <span className="scene">
              <span className="scene-label">{progress.label}</span>
              <span className="dots" aria-hidden="true">
                {Array.from({ length: progress.count }, (_, i) => (
                  <i key={i} data-on={i === progress.index || undefined} />
                ))}
              </span>
            </span>
          )}
          {schematic && <span className="tag">Schematic</span>}
        </span>
      </div>
      <div
        className="fig-stage"
        data-clickable={onAdvance ? 'true' : undefined}
        data-hint={hint}
        role={onAdvance ? 'button' : undefined}
        tabIndex={onAdvance ? 0 : undefined}
        aria-label={onAdvance ? `${title}. Showing ${progress?.label ?? 'the current view'}. Activate for the next view.` : undefined}
        onClick={onAdvance}
        onKeyDown={onKeyDown}
        onMouseEnter={onHover ? () => onHover(true) : undefined}
        onMouseLeave={onHover ? () => onHover(false) : undefined}
      >
        {children}
      </div>
      {legend && <div className="fig-legend">{legend}</div>}
      {note !== undefined && <div className="fig-note">{note}</div>}
      <figcaption>{caption}</figcaption>
    </figure>
  )
}
