# Base: Qwen3.5-4B lower bound

The unchanged, post-trained `Qwen/Qwen3.5-4B` checkpoint (revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`) evaluated as a constant predictor. No
training, adapters, harness, system prompt, tools, retrieval or conversational state.

## Protocol

- One ordinary user prompt through the model's chat template, thinking disabled.
- BF16 greedy decoding with vLLM 0.31.0 (`--engine vllm`, the default, on CUDA):
  this package's `vllm_decode.py` schedules all pending items together; an item's `seconds` is its share of its
  call's wall time. Model/tokenizer EOS union as stop tokens. `--engine
  transformers` uses the Transformers backend for local MPS or CPU inference.
- Output ceiling 5,120 generated tokens (`--max-new-tokens`, recorded as
  `maxNewTokens`); input ceiling 16,384 template tokens. Longer inputs are recorded
  as `input_limit` failures with zero score and are never truncated.
- Task panel: all 2,500 TRACE+17 test records (`--task-dataset`, default
  `experiments/data/trace-plus-17`). G panel: the 1,000-item G-test with its
  200-item nested probe (`--profile general`).
- Scoring: the package-local copy of the pinned TRACE+17 evaluator, with
  unchanged prompts and the local IFEval verifiers. G-test choice answers use
  the shared v3 extraction rule in `experiments/protocol.md`; TRACE's official
  first-character choice rule is unchanged.

Because the predictor never changes, every stage reuses the same measured
predictions: forgetting, signed BWT, incremental gain and G change are zero by
construction, while AA and G scores are measured.

## Reproduction

On a CUDA node, from a repository checkout with `experiments/data` and the pinned
checkpoint, `bootstrap.sh` builds a job-local environment from
`requirements-eval.lock.txt` (vLLM and the scorer's libraries; Python 3.10, set
`BASE_PYTHON` to another interpreter) and runs `run.py --engine vllm --device cuda`:

```sh
packages/baselines/base/bootstrap.sh --profile full --output <task-run>
packages/baselines/base/bootstrap.sh --profile general --output <g-run>
```

Local Transformers runs (`--engine transformers`):

```sh
# Tasks (base .venv) and G (separate environment with the G scorer dependencies)
packages/baselines/base/.venv/bin/python packages/baselines/base/run.py \
  --profile full --output <task-run>
experiments/runtime/base-general-venv/bin/python packages/baselines/base/run.py \
  --profile general --output <g-run>

# Seal
experiments/runtime/base-general-venv/bin/python packages/baselines/base/combine.py \
  --tasks <task-run> --general <g-run> --output experiments/reports/<name>.json
```

`experiments/runtime/base-general-venv` is created with
`UV_PROJECT_ENVIRONMENT=$PWD/experiments/runtime/base-general-venv uv sync --locked --project packages/baselines/base --python 3.12`.
Run one inference process at a time so recorded operation times are solo-device times.

The package also runs outside the monorepo. Supply `--checkpoint /path/to/base`
and `--registry /path/to/registry.json` (the pinned model/revision and weights
hash are still verified), plus `--task-dataset` or `--general-dataset`. Set
`TRACE_GPT2_TOKENIZER` and `TRACE_PUNKT_TOKENIZER` to the pinned data files.
New manifests record the checkpoint path and actual scoring assets; no imports
reach another baseline or the repository evaluator.

## Run directory contract

- `manifest.json`: the immutable prediction contract (model, weights tree hash,
  template, decoding, libraries, panel hash, record IDs, scorer and dataset
  provenance). `items/NNNN.json`: atomically sealed, checksummed predictions with
  token IDs, stop reason and cost. `scores.json`, `metrics.json`,
  `cost-ledger.json`, `status.json`: derived outputs with hashes in the status.
- Resume re-validates every cached item. Host, scorer and wrapper changes are
  recorded per attempt in `executions/`; a change to the run's predictor class
  (`Predictor` or `VllmPredictor`) requires a compatibility audit, and the vLLM
  engine's module is pinned by the contract's `decoderSha256`. A process lock prevents concurrent writers.

## Verification and sealing

`verify.py --run <dir> [--output <file>]` independently re-derives panel identity,
item checksums, tokenization, decoding, stop reasons, scores, metrics, cost totals
and checkpoint immutability. `combine.py` requires identical predictors and
inference source for the task and G runs, verifies both, and lists length-stopped
outputs with their exact repetition period.

## Monitoring

`monitor.py --run <dir>` validates sealed items and the live process and writes a
task-aware ETA (`monitor-latest.json`, `monitor-history.jsonl`); it never edits
predictions or the contract. It expects the launcher to write `launch.json` (PID,
start time) and `preflight.json` (per-record input lengths) into the run directory.
