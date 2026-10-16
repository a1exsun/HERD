import { existsSync, readFileSync, writeFileSync } from 'node:fs'

import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

if (!process.argv[2]) throw new Error('Usage: node scripts/export-experiments.mjs <outputs-directory>')
const record = pathToFileURL(resolve(process.argv[2]) + '/')
export const read = (path) => JSON.parse(readFileSync(new URL(path, record), 'utf8'))

const ANCHORS = ['4', '8', '16', '25']

// One point of a series, with the spread over seeds where the method has several.
const point = ({ stage, value, sd }) => (sd === undefined ? { stage, value } : { stage, value, sd })

// A per-stage table as parallel arrays: stages, then each series' values (and spread over seeds).
function columns(table, keys) {
  const stages = table.rows.map((row) => row.x)
  const series = {}
  for (const key of keys) {
    const values = table.values.filter((value) => value.series === key)
    const at = new Map(values.map((value) => [value.row, value]))
    series[key] = {
      value: table.rows.map((row) => at.get(row.id)?.value ?? null),
      sd: table.rows.map((row) => at.get(row.id)?.sd ?? null),
    }
  }
  return { stages, series }
}

// Each task's score when it was learned (the diagonal) and after the last stage. A constant
// method has one row, at stage 0.
function taskScores(results, tasks) {
  const cell = new Map(results.taskCells.map((c) => [`${c.stage}|${c.task}`, c]))
  const last = results.constant ? 0 : results.stages.planned
  return tasks.map((task) => {
    const learned = cell.get(`${results.constant ? 0 : task.stage}|${task.id}`)
    const final = cell.get(`${last}|${task.id}`)
    return {
      learned: learned?.score ?? null,
      final: final?.score ?? null,
      ...(final?.sd === undefined ? {} : { finalSd: final.sd }),
    }
  })
}

// Select only the aggregates used by the page. Item outputs, prompts, carried state and training
// configuration must not be included in the browser bundle.
export function readExperiments() {
  const protocol = read('protocol.json')
  const schema = read('schema.json')
  // The page compares the main method and baselines; ablations are separate paper diagnostics.
  const entries = schema.registry.filter((entry) => !entry.ablation
    && existsSync(new URL(`${entry.path}/results.json`, record)))
  const methods = entries.map((entry) => {
    const method = read(`${entry.path}/method.json`)
    const results = read(`${entry.path}/results.json`)
    const depths = Object.fromEntries(ANCHORS.filter((depth) => results.summary.depths[depth])
      .map((depth) => [depth, results.summary.depths[depth]]))
    return {
      id: entry.id,
      name: method.name,
      short: method.short,
      family: entry.family,
      status: entry.status,
      schedule: results.schedule,
      constant: results.constant,
      depths,
      depthsSd: results.summarySd?.depths ?? null,
      intervals: {
        seenMean: results.intervals.seenMean,
        vsBase: results.intervals.vsBase,
        acquisition: results.intervals.acquisition ?? null,
      },
      series: {
        seenMean: results.series.seenMean.map(point),
        firstEight: results.series.firstEight.map(point),
        generalProbe: results.series.generalProbe.map(point),
        current: results.series.current.map(point),
      },
      tasks: taskScores(results, protocol.stream.tasks),
      trainingSeconds: typeof results.cost?.trainingSeconds === 'number'
        ? results.cost.trainingSeconds : results.cost?.trainingSeconds?.value ?? null,
    }
  })

  const hero = read('hero/results.json')
  const tables = hero.tables
  const updates = tables.updates
  const byCell = new Map(updates.values.map((v) => [`${v.row}|${v.series}`, v]))
  const value = (row, series) => byCell.get(`${row}|${series}`)?.value ?? null
  const spreadOf = (row, series) => byCell.get(`${row}|${series}`)?.sd ?? null
  const compact = (v) => (v === null ? null : Number(v.toPrecision(4)))
  const taskIndex = new Map(protocol.stream.tasks.map((task, i) => [task.id, i]))
  const categoryIds = protocol.general.categories.map((c) => c.id)
  const COMPARED = ['seenMean', 'acquisition', 'forgetting', 'signedBWT', 'generalMacro']
  return {
    generated: schema.generated,
    protocol: {
      tasks: protocol.stream.tasks.map(({ stage, label, category, metricLabel }) => ({ stage, label, category, metricLabel })),
      categories: protocol.general.categories.map(({ id, label, source }) => ({ id, label, source })),
      anchors: protocol.schedule.anchors,
    },
    methods,
    hero: {
      trainingSeconds: hero.cost.trainingSeconds,
      hardware: hero.cost.hardware,
      harness: { ...columns(tables.harness, ['harnessTokens', 'skills']), budget: tables.harness.budget },
      guard: { ...columns(tables.guard, ['teacherNLLPrevious', 'teacherNLLCandidate', 'committed']), tolerance: tables.guard.harmTolerance },
      drift: {
        ...columns(tables.drift, ['checkpointDriftFinal', 'generalDriftFinal', 'checkpointDriftPeak', 'generalDriftPeak']),
        checkpointBudget: tables.drift.checkpointBudget, generalBudget: tables.drift.generalBudget,
      },
      multipliers: columns(tables.multipliers, ['checkpointMultiplierActive', 'generalMultiplierActive']),
      distillation: columns(tables.distillation, ['distillKLFirst', 'distillKLLastQuarter']),
      residual: { ...columns(tables.residual, ['residualRecords', 'residualTasks', 'residualTokens']), budget: tables.residual.budget },
      matrix: hero.taskCells.map((c) => ({ stage: c.stage, task: taskIndex.get(c.task), value: c.score, sd: c.sd })),
      generalByCategory: { ...columns(tables.generalByCategory, categoryIds), reference: tables.generalByCategory.reference },
      comparisons: {
        stage: hero.comparisons.stage,
        draws: hero.comparisons.draws,
        methods: hero.comparisons.methods.map((m) => ({
          id: m.id,
          ...Object.fromEntries(COMPARED.map((name) => [name, { difference: m[name].difference, interval: m[name].interval }])),
        })),
      },
      updates: {
        x: updates.rows.map((row) => row.x),
        stage: updates.rows.map((row) => row.stage),
        update: updates.rows.map((row) => row.update),
        checkpointDrift: updates.rows.map((row) => compact(value(row.id, 'checkpointDrift'))),
        generalDrift: updates.rows.map((row) => compact(value(row.id, 'generalDrift'))),
        distillKL: updates.rows.map((row) => compact(value(row.id, 'distillKL'))),
        distillKLSd: updates.rows.map((row) => compact(spreadOf(row.id, 'distillKL'))),
        gradNorm: updates.rows.map((row) => compact(value(row.id, 'gradNorm'))),
        gradNormSd: updates.rows.map((row) => compact(spreadOf(row.id, 'gradNorm'))),
        perStage: updates.updatesPerStage,
        learningRate: updates.learningRate,
      },
    },
  }
}

writeFileSync(new URL('../data/experiments.json', import.meta.url), JSON.stringify(readExperiments()) + '\n')
