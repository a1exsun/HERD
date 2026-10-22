# Experiments

- `protocol.md`: task stream, evaluation schedule, decoding and metrics.
- `code/`: the results pipeline from your runs to the paper's tables and figures ([README](code/README.md)).
- `../datasets/general-test-v1.json`, `../datasets/general-probes-v1.json`: pinned data
  construction manifests, required by the data preparation scripts.
- `data/tokenizers/gpt2/`: the pinned scorer tokenizer and its license.

Everything else under this directory is local working state and ignored by Git:

- `data/`: materialized TRACE+17 (`trace-plus-17/`), the G-test and general
  probes, raw source snapshots and scorer assets; the tracked dataset contracts
  live under `../datasets/`.
- `runs/`: run outputs (bundles, predictions and stage records).
- `outputs/`: metrics exported from `runs/` by `code/export-paper-experiments.py`.
- `checkpoints/`: the local Qwen registry (`qwen3.5-4b/`) managed by
  `pnpm model:setup` and the DSH plugin.

No published result payloads are included in this release.
