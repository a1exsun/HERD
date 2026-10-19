import type { CSSProperties } from 'react'

/** Inline custom properties, typed once so components stay free of casts. */
export function vars(values: Record<string, string | number>): CSSProperties {
  return values as CSSProperties
}
