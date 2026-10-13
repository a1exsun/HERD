# Experiments

- `protocol.md`: task stream, evaluation schedule, decoding and metrics.
- `code/summarize.py`: seed means, sample SD and paired item-bootstrap intervals.
- `../datasets/general-test-v1.json`, `../datasets/general-probes-v1.json`: pinned data
  construction manifests, required by the data preparation scripts.
- `data/tokenizers/gpt2/`: the pinned scorer tokenizer and its license.

Materialized datasets, run bundles, predictions, checkpoints, runtime state and
computed results are local and ignored by Git. Train and score through the
HERD stream or the relevant baseline's runner. No published result payloads
are included in this release.
