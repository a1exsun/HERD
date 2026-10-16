import { readFileSync } from 'node:fs'

// Presentation aggregates are versioned with the site; no local run archive is required.
export function readExperiments() {
  return JSON.parse(readFileSync(new URL('../data/experiments.json', import.meta.url), 'utf8'))
}
