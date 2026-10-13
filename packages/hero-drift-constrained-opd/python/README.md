# HERO local Qwen runtime

This Python package holds the local runtime around the pinned
`Qwen/Qwen3.5-4B` checkpoint. Drift-Constrained OPD training is the stream
runner in [`../stream`](../stream); this project's environment also runs its
CPU tests.

- `hero_qwen.bridge` (`hero-qwen-bridge`): loopback-only OpenAI-compatible Qwen
  route used by DSH Web, plus `/hero/clip`, `/hero/token-counts` and
  `/hero/target-nll` for the console's harness proposal and guard.
- `hero_qwen.model_registry`: the local version registry (`pnpm model:setup`,
  `pnpm model:status`). `install-adapter --stream-run <run> --stage N` installs
  a stream stage verified against its `stage.json`, the plan and the run's
  parent chain; `install-adapter --adapter <dir>` installs a bare LoRA.
- `hero_qwen.config`: pinned model identity and artifact hashing.

`uv.lock` pins the local MPS Torch `2.14.0` and Transformers `5.2.0`.

```sh
pnpm opd:test       # python/tests and ../stream/tests
pnpm opd:validate
uv run --project packages/hero-drift-constrained-opd/python --extra bridge \
  hero-qwen-bridge --checkpoint /absolute/merged-checkpoint
```
