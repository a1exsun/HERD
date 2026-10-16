import type { CSSProperties } from 'react'

/** Inline custom properties, typed once so components stay free of casts. */
export function vars(values: Record<string, string | number>): CSSProperties {
  return values as CSSProperties
}

/** Background for a heat-map cell: the share `t` (0 to 1) of `color` mixed into the surface. */
export const heat = (color: string, t: number) =>
  `color-mix(in srgb, ${color} ${Math.round(Math.max(0, Math.min(1, t)) * 70)}%, var(--surface))`
