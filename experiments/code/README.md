# Results pipeline

Turns your own runs into the paper's tables and figures. No results are shipped: every number comes from
predictions you produce with the runners.

1. Run each method with seeds 42, 43 and 44 (`--seed`), and HERD's ablations with seed 42
   ([stream README](../../packages/hero-drift-constrained-opd/stream/README.md#ablations)); the "+ harness" row
   re-scores the HERD runs with `stream --inference-harness`. The base model (`packages/baselines/base`, task and
   G-test profiles) is required: it is stage 0 of every general-capability change.
2. List the runs, relative to `experiments/runs`, in a copy of [`runs.example.json`](runs.example.json); a method
   left out is reported as scheduled and its rows stay empty.
3. Export, then draw:

```sh
uv run --project packages/evaluator/python python experiments/code/export-paper-experiments.py --runs runs.json
for script in experiments/code/paper-*.py; do uv run --with matplotlib --with numpy python "$script"; done
```

The export rescores every prediction against the test sets (checked against `datasets/`), averages each method
over its seeds with sample standard deviations, and computes AA, acquisition, F, BWT, ΔC, ΔC_min, the 95% item
bootstrap intervals (4,000 draws, within task and G-test category) and HERD's paired comparisons. It writes
`experiments/outputs/` (ignored by Git); the scripts write `output/paper/tables/*.tex` and
`output/paper/figures/*.pdf`.

| File | Output |
| --- | --- |
| `registry.json` | Method ids, families, labels and ablation metadata |
| `export-paper-experiments.py` | `experiments/outputs/` |
| `paper-main-table.py`, `paper-comparison-table.py`, `paper-cost-table.py`, `paper-ablation-table.py` | Main results, paired comparisons, cost and ablation tables |
| `paper-long-horizon-figure.py`, `paper-long-horizon-appendix-figure.py`, `paper-ablation-figure.py`, `paper-mechanism-figure.py`, `paper-task-matrix-figure.py` | Long-horizon, per-stage, ablation, mechanism and task-matrix figures |

`tests/test_export.py` runs the whole pipeline on a synthetic stream:

```sh
uv run --project packages/evaluator/python --with matplotlib --with numpy --with pytest pytest experiments/code/tests
```
