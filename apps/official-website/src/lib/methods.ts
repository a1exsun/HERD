import paper from '../../data/paper-results.json'

export type Family = 'This work' | 'Control' | 'Context' | 'Weights' | 'Hybrid'

export interface Method {
  id: string
  name: string
  family: Family
  /** Table 1 row: AA, acquisition, forgetting, BWT, G-test, ΔC, worst ΔC. */
  values: number[]
}

export const methods = paper.rows as Method[]
export const method = (id: string) => methods.find((m) => m.id === id)!

/** Three categorical hues, validated for colour-vision deficiency in both themes; controls and the
    single hybrid method are neutral and always carry a direct label or dash pattern. */
export const familyColor: Record<Family, string> = {
  'This work': 'var(--viz-hero)',
  Context: 'var(--viz-context)',
  Weights: 'var(--viz-weights)',
  Hybrid: 'var(--ink-2)',
  Control: 'var(--muted)',
}

export const colorOf = (id: string) => familyColor[method(id).family]

export const familyLabel: Record<Family, string> = {
  'This work': 'This work',
  Context: 'Non-parametric',
  Weights: 'Parametric',
  Hybrid: 'Hybrid',
  Control: 'Control',
}

export const fixed = (v: number, digits = 3) => v.toFixed(digits).replace('-', '−')
export const signed = (v: number, digits = 3) => (v >= 0 ? '+' : '') + fixed(v, digits)
