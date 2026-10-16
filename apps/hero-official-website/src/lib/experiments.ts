export interface Depth {
  seenMean: number
  acquisition: number
  forgetting: number
  signedBWT: number
  items: number
  general: {
    panel: string
    macro: number
    change: number
    worstCategoryChange: number
    categories: Record<string, number>
  }
}

/** The spread over seeds of the numbers in a Depth. */
export type DepthSd = Partial<Omit<Depth, 'general'>> & {
  general?: Partial<Omit<Depth['general'], 'categories'>> & { categories?: Record<string, number> }
}

export interface Point {
  stage: number
  value: number
  sd?: number
}

export interface ExperimentMethod {
  id: string
  name: string
  short: string
  family: string
  status: string
  schedule: string
  constant: boolean
  depths: Record<string, Depth>
  depthsSd: Record<string, DepthSd> | null
  intervals: {
    seenMean: Record<string, [number, number]>
    vsBase: Record<string, [number, number]>
    acquisition: [number, number] | null
  }
  series: Record<'seenMean' | 'firstEight' | 'generalProbe' | 'current', Point[]>
  tasks: { learned: number; final: number; finalSd?: number }[]
  trainingSeconds: number | null
}

export interface StageColumns<K extends string> {
  stages: number[]
  series: Record<K, { value: (number | null)[]; sd: (number | null)[] }>
}

export type Compared = 'seenMean' | 'acquisition' | 'forgetting' | 'signedBWT' | 'generalMacro'

interface Spread {
  value: number
  sd: number
}

declare const __EXPERIMENTS__: {
  generated: string
  protocol: {
    tasks: { stage: number; label: string; category: string; metricLabel: string }[]
    categories: { id: string; label: string; source: string }[]
    anchors: number[]
  }
  methods: ExperimentMethod[]
  hero: {
    trainingSeconds: Spread
    hardware: string
    harness: StageColumns<'harnessTokens' | 'skills'> & { budget: number }
    guard: StageColumns<'teacherNLLPrevious' | 'teacherNLLCandidate' | 'committed'> & { tolerance: number }
    drift: StageColumns<'checkpointDriftFinal' | 'generalDriftFinal' | 'checkpointDriftPeak' | 'generalDriftPeak'> & {
      checkpointBudget: number
      generalBudget: number
    }
    multipliers: StageColumns<'checkpointMultiplierActive' | 'generalMultiplierActive'>
    distillation: StageColumns<'distillKLFirst' | 'distillKLLastQuarter'>
    residual: StageColumns<'residualRecords' | 'residualTasks' | 'residualTokens'> & { budget: number }
    matrix: { stage: number; task: number; value: number; sd: number }[]
    generalByCategory: StageColumns<string> & { reference: Record<string, number> }
    comparisons: {
      stage: number
      draws: number
      methods: ({ id: string } & Record<Compared, { difference: number; interval: [number, number] }>)[]
    }
    updates: {
      x: number[]
      stage: number[]
      update: number[]
      checkpointDrift: number[]
      generalDrift: number[]
      distillKL: number[]
      distillKLSd: number[]
      gradNorm: number[]
      gradNormSd: number[]
      perStage: number
      learningRate: number
    }
  }
}

export const experiments = __EXPERIMENTS__
export const method = (id: string) => experiments.methods.find((m) => m.id === id)
export const score = (value: number | null | undefined, digits = 3) =>
  value === undefined || value === null ? '—' : value.toFixed(digits)
export const signed = (value: number | null | undefined, digits = 3) =>
  value === undefined || value === null ? '—' : `${value > 0 ? '+' : value < 0 ? '−' : ''}${Math.abs(value).toFixed(digits)}`

/** One color per method, from the page's semantic tokens: HERD is the deployed predictor. */
export const methodColor: Record<string, string> = {
  hero: 'var(--deploy)',
  base: 'var(--muted)',
  'static-harness': 'var(--aux)',
  'hcl-stable': 'var(--harness)',
  ace: 'color-mix(in srgb, var(--harness) 65%, var(--optim))',
  'seq-lora': 'var(--predictor)',
  'uniform-replay': 'var(--guard)',
  opr: 'var(--optim)',
  treelora: 'var(--teal)',
  moram: 'color-mix(in srgb, var(--predictor) 60%, var(--guard))',
  sdft: 'color-mix(in srgb, var(--optim) 70%, var(--guard))',
  ophsd: 'color-mix(in srgb, var(--guard) 55%, var(--harness))',
}
export const isControl = (m: ExperimentMethod) => m.family === 'control'
