# TRACE+17

TRACE+17 is a deterministic 25-task continual-learning stream: the canonical TRACE-8
order followed by the 17 SuperNI tasks declared in the English paper, seed 42. Each
task has 500 train, 100 validation and 100 test records. NumGLUE-cm preserves its
256/48/100 selected prompts and fills train and validation by seeded sampling
with replacement within each split. Repeated occurrences have unique IDs and
`metadata.resampledFrom`; the 100 test records are unchanged. Normalized prompts
never repeat across splits or tasks, and every test and validation prompt fits the
16,384-token model input ceiling.

`manifest.json` is the tracked dataset contract: source revisions, ordered task
identity, metrics, realized split counts, construction rules, physical visibility
and SHA-256 hashes for every file.

## Build

Prepare the pinned source directories described in `.env.example` and export
`HERO_TRACE_SOURCE_ROOT`. The official archive can be downloaded with
`node scripts/archive-trace-source.mjs`; extract its `LLM-CL-Benchmark_500` and
`LLM-CL-Benchmark_5000` directories into `$HERO_TRACE_SOURCE_ROOT/trace/`.
Place the pinned SuperNI checkout at
`$HERO_TRACE_SOURCE_ROOT/superni/natural-instructions-v2.8/`.
`pnpm model:setup` supplies the pinned Qwen tokenizer used for input length checks.

```sh
pnpm dataset:build    # select, then complete
pnpm dataset:verify
```

1. **Select** (`pnpm dataset:select`, `config/datasets/trace-plus-17.json`):
   deduplicates TRACE LLM-CL-Benchmark_500 and the SuperNI tasks and samples the
   requested splits into `experiments/data/trace-plus-17-500/`
   (tracked manifest `selection-500/manifest.json`). Sources are read from
   `HERO_TRACE_SOURCE_ROOT` at the pinned upstream revisions.
2. **Complete** (`scripts/build-trace-plus-17.py`): excludes test prompts above the
   input ceiling and fills TRACE split shortfalls from LLM-CL-Benchmark_5000
   `train.json` in a deterministic hash order, test first, then validation, then
   train. Output: `experiments/data/trace-plus-17/`. The pool is read from
   `$HERO_TRACE_SOURCE_ROOT/trace/LLM-CL-Benchmark_5000/` (or `--pool <dir>`), and
   input lengths are counted with the Qwen3.5-4B tokenizer at revision
   `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, read from
   `experiments/checkpoints/qwen3.5-4b/base/`. It copies both manifests to this
   directory: `manifest.json` and `selection-500/manifest.json`.

The completed dataset version is 1.1. NumGLUE-cm's `resampling` manifest entry
records unique-prompt and added-record counts separately. Version 1.1 changes
the training/validation sample multiplicities; rerun experiments with this
manifest rather than combining it with prior run artifacts.

Training records and validation inputs are under `agent-visible/`; validation
targets and complete test records are under `hidden-scorer/` and are
evaluation-only.
