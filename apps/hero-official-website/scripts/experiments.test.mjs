import assert from 'node:assert/strict'
import test from 'node:test'
import { readExperiments } from './experiments.mjs'

test('the standalone snapshot covers all paper methods, tasks and chart series', () => {
  const data = readExperiments()
  assert.equal(data.protocol.tasks.length, 25)
  assert.deepEqual(data.methods.map((m) => m.id).sort(), [
    'ace', 'base', 'hcl-stable', 'hero', 'moram', 'ophsd', 'opr', 'sdft',
    'seq-lora', 'static-harness', 'treelora', 'uniform-replay',
  ])
  for (const method of data.methods) {
    for (const depth of [4, 8, 16, 25]) {
      for (const value of [method.depths[depth].seenMean, method.depths[depth].forgetting,
        method.depths[depth].general.macro]) assert.ok(Number.isFinite(value))
    }
    assert.equal(method.tasks.length, 25)
    assert.ok(method.tasks.every((task) => Number.isFinite(task.learned) && Number.isFinite(task.final)))
    for (const points of Object.values(method.series)) {
      assert.ok(points.length > 0)
      assert.ok(points.every((point) => Number.isFinite(point.value)))
    }
  }
  for (const key of ['harness', 'drift', 'distillation', 'residual']) {
    assert.equal(data.hero[key].stages.length, 25)
  }
  for (const key of ['distillKL', 'gradNorm', 'checkpointDrift', 'generalDrift']) {
    assert.equal(data.hero.updates[key].length, 25 * data.hero.updates.perStage)
    assert.ok(data.hero.updates[key].every(Number.isFinite))
  }
})

test('the browser receives compact aggregates, not individual records or runtime state', () => {
  const data = readExperiments()
  assert.ok(Buffer.byteLength(JSON.stringify(data)) < 160_000)
  const visit = (value) => {
    if (!value || typeof value !== 'object') return
    for (const [key, child] of Object.entries(value)) {
      assert.ok(!['outputs', 'samples', 'state', 'configuration', 'accounting', 'taskCells',
        'generalCells', 'seeds', 'perSeed', 'aggregation'].includes(key), key)
      visit(child)
    }
  }
  visit(data)
})
