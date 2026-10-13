# Experiment baselines

Workspace package for the paper's comparison methods.
Every method uses the same scientific contract while keeping its runnable code
self-contained:

- base model: `Qwen/Qwen3.5-4B`;
- stream: the 25-stage `TRACE+17` manifest, seed 42;
- scorer: the same pinned TRACE+17 evaluator logic; the nine standalone packages
  below carry local snapshots, with conformance checked against the canonical evaluator;
- data isolation: training may read only the agent-visible train split, while
  validation/test targets remain scorer-only inputs.

The baseline directories do not import one another. Where two methods need the
same protocol, SFT, or evaluator behavior, that small implementation is copied
locally on purpose so each method can be staged and executed independently.
`base`, `static-harness`, `ace`, `hcl-stable`, `seq-lora`, `uniform-replay`,
`treelora`, `moram`, and `sdft` each have their own `pyproject.toml`, `uv.lock`,
evaluator, and tests; `uniform-replay`, `moram`, and `sdft` also carry the
hash-locked CUDA job-bundle runner (`prepare`/`train`/`evaluate`/`score`). OPR and
OPHSD are job-bundle runners (`opr/run.py`, `ophsd/run.py`) that copy the shared
`python/` package (`hero_baselines`) into their bundles.

| Method | Family | Paper |
| --- | --- | --- |
| `base` | constant | project control |
| `static-harness` | constant | project control |
| `ace` | non-parametric | [Agentic Context Engineering, arXiv:2510.04618](https://arxiv.org/abs/2510.04618) |
| `hcl-stable` | non-parametric | [Harness Continual Learning, arXiv:2608.19013](https://arxiv.org/abs/2608.19013) |
| `seq-lora` | parametric | standard control |
| `uniform-replay` | parametric | standard control |
| `opr` | parametric | [On-Policy Replay, arXiv:2605.29495](https://arxiv.org/abs/2605.29495) |
| `treelora` | parametric | [TreeLoRA, ICML 2025 / PMLR 267](https://proceedings.mlr.press/v267/qian25b.html) |
| `moram` | parametric | [MoRAM, ICML 2026 / arXiv:2506.21035](https://arxiv.org/abs/2506.21035) |
| `ophsd` | OPD | [Training with Harnesses, arXiv:2605.08741](https://arxiv.org/abs/2605.08741) |
| `sdft` | OPD | [Self-Distillation Enables Continual Learning, arXiv:2601.19897](https://arxiv.org/abs/2601.19897) |

HCL-stable runs on this Mac: every executor and proposer call is served by the
pinned Qwen3.5-4B in an in-process MLX engine (`local_backend.py`, continuous
batching with shared-prefix cache snapshots). ACE serves every role from the pinned checkpoint through in-process PyTorch MPS (`ace/local_backend.py`)
or a CUDA vLLM server (`ace/serve.sh`), selected by `prepare --device mps|cuda`. Both `run.py run` commands
train the stream in order while each sealed stage is evaluated alongside the next.

## Runtime entry points

| Method | Entry point |
| --- | --- |
| `base` | `uv run --locked --project packages/baselines/base python packages/baselines/base/run.py ...` |
| `static-harness` | `uv run --locked --project packages/baselines/static-harness python packages/baselines/static-harness/run.py ...` |
| `ace` | `uv run --locked --project packages/baselines/ace --extra local-model python packages/baselines/ace/run.py prepare --device mps ...`; then run the bundle's `run.py run\|train\|evaluate\|score ...` (CUDA uses `ACE_LOCAL_BASE_URL` and needs no extra) |
| `hcl-stable` | `uv run --locked --project packages/baselines/hcl-stable --extra local-model python packages/baselines/hcl-stable/run.py prepare\|run\|train\|evaluate\|score ...` |
| `seq-lora` | `uv run --locked --project packages/baselines/seq-lora python packages/baselines/seq-lora/run.py ...` |
| `uniform-replay` | `uv run --locked --project packages/baselines/uniform-replay python packages/baselines/uniform-replay/run.py prepare\|train\|evaluate\|score ...` |
| `opr` | `uv run --project packages/baselines/python python packages/baselines/opr/run.py prepare\|train\|evaluate\|rollout\|score ...` |
| `treelora` | `uv run --locked --project packages/baselines/treelora python packages/baselines/treelora/run.py ...` |
| `moram` | `uv run --locked --project packages/baselines/moram python packages/baselines/moram/run.py prepare\|train\|evaluate\|score ...` |
| `ophsd` | `uv run --project packages/baselines/python --extra ophsd python packages/baselines/ophsd/run.py prepare\|train\|score ...` |
| `sdft` | `uv run --locked --project packages/baselines/sdft python packages/baselines/sdft/run.py prepare\|train\|evaluate\|score ...` |

## Commands

```bash
pnpm --filter @hero/baselines python:test
pnpm --filter @hero/baselines runners:test
pnpm --filter @hero/baselines standalone:test
pnpm --filter @hero/baselines python:validate
```

The Python package also exposes `hero-baselines protocol` for a machine-readable
view of the shared model/dataset contract.

## Standalone execution and validation

Copy a single method directory and install with `uv sync --locked` from that
directory. Run `uv run --locked python run.py --help` for its CLI. External
inputs are model weights or an OpenAI-compatible Qwen endpoint, TRACE+17 data,
and the two pinned scoring assets:

```sh
export TRACE_GPT2_TOKENIZER=/absolute/data/tokenizers/gpt2/tokenizer.json
export TRACE_PUNKT_TOKENIZER=/absolute/data/nltk/punkt/english.pickle
uv run --locked pytest tests
```

Within the monorepo these assets resolve from `experiments/data`; both explicit
and default paths are checked against pinned SHA-256 values. `base` also accepts
`--checkpoint` and `--registry`, and records the actual checkpoint path for
verification. HCL-stable loads the pinned checkpoint itself under MLX (the `local-model`
extra); ACE uses either local PyTorch MPS or an OpenAI-compatible Qwen endpoint; static-harness uses an endpoint.
`uniform-replay`, `moram`, and `sdft` read the TRACE+17 and G-test files through
their manifests' SHA-256 values when preparing a bundle; `uniform-replay` also
needs the pinned checkpoint's tokenizer; `--memory-records` fixes its task reservoir capacity (default 128).
The static-harness orchestrator requires macOS for its sandboxed
Python tool; its model endpoint may run elsewhere.

`standalone:test` copies all nine packages and only the scoring data assets to a
temporary directory outside the repository, installs each lock in a fresh
environment, blocks project-package imports, and runs local tests plus scorer
conformance cases. It uses small CPU fixtures and a fake model client for the
tool loop; it does not run Qwen training or a full benchmark. The shared
`python:test` command covers the shared `hero_baselines` package used by OPR and OPHSD;
`runners:test` runs `opr/tests` and `ophsd/tests` (each runner's own evaluation
code) in separate processes, so neither imports the other's modules.
