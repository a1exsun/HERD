import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import test from 'node:test'

const read = (path) => readFileSync(new URL(path, import.meta.url))
const paper = JSON.parse(read('../data/paper-results.json'))
const table = Object.fromEntries(paper.rows.map((row) => [row.id, row.values]))

test('the page carries paper and appendix tables, never per-item outputs', () => {
  assert.deepEqual(Object.keys(paper).sort(),
    ['ablations', 'columns', 'comparisons', 'cost', 'herdSd', 'longHorizon', 'rows', 'source', 'stream'])
  assert.equal(paper.rows.length, 12)
  assert.equal(paper.ablations.rows.length, 10)
  assert.equal(paper.stream.length, 25)
})

test('the tables agree with one another', () => {
  const full = paper.ablations.rows[0]
  assert.deepEqual(full.values, paper.herdSd.columns.map((i) => table.hero[i]))
  assert.deepEqual(paper.ablations.noise, paper.herdSd.values.map((v) => Math.round(v * 3000) / 1000))
  assert.equal(paper.cost.rows.find((r) => r.name === 'HERD').values[2], 10723)
  for (const row of paper.comparisons.rows) {
    // Bold intervals exclude zero; a printed bound of exactly 0 can round either way.
    for (const { excludesZero, interval: [lo, hi] } of row.cells) {
      assert.ok(excludesZero ? lo >= 0 || hi <= 0 : lo <= 0 && hi >= 0, row.name)
    }
  }
})

test('Figure 3 agrees with Table 1 and with the numbers quoted in the text', () => {
  const { firstEight, gtest } = paper.longHorizon
  for (const [id, values] of Object.entries(gtest.methods)) assert.equal(values.at(-1), table[id][4], id)
  // The text quotes 0.664 at stage 8 and 0.665 at stage 25.
  assert.deepEqual([firstEight.methods.hero[0], firstEight.methods.hero.at(-1)], [0.664, 0.665])
  assert.equal(firstEight.methods['seq-lora'][1], 0.372)
  assert.deepEqual(gtest.methods.ophsd.slice(1), [0.741, 0.661, 0.613, 0.582])
})
