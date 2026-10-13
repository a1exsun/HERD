# HERD stream runner

`hero.py` implements the stage algorithm; `run.py` prepares a self-contained
bundle, trains the stream, evaluates test prompts and scores predictions locally.

## Stage procedure

1. Fill the 8,192-token evidence window in seeded order, skipping records that
   do not fit whole. The frozen checkpoint answers the window; up to four
   mistakes and sixteen examples guide one greedy skill proposal from the base.
2. Keep the candidate unless it is empty, exceeds the 24,576-token registry
   budget, or raises answer-replay teacher NLL beyond `--harm-tolerance` (0.3).
3. Rank the retained/new record pool by residual under the pre-training
   checkpoint and committed harness. Keep whole records under 24,576 tokens;
   the remaining 8,192 tokens are reserved for general probes.
4. Train a fresh rank-32 LoRA, alpha 64, for 32 updates of 32 records with AdamW
   at 1e-4 decaying linearly to zero. The student sees the bare prompt; the
   frozen teacher receives the skill and verified answer replay. Forward KL is
   scored at verified-answer positions. Bare prompt plus target must fit 16,384
   tokens. Checkpoint probes come from the previous memory, up to eight per task;
   base probes are fixed across the stream. Both use at most 32 greedy tokens.
5. Audit both drifts, save the bf16 adapter, merge it, and evaluate the formal
   test schedule. Stage records retain the registry, guard, memory, losses,
   drift, adapter checksum and merged-weight fingerprint.

Each drift is the worst group's mean KL. Its multiplier uses bounded dual
ascent at rate 0.5; both full-method budgets are 0.05. Skills route by the
longest NFKC/whitespace/lowercase-normalized prefix. Task memory is never replayed
as training targets.

## Prepare and run

From the repository root, with materialized datasets and the pinned tokenizer:

```sh
python packages/hero-drift-constrained-opd/stream/run.py prepare \
  --repository . --output /path/bundle --variant full --seed 42
```

The bundle always contains the formal test prompts without their gold answers.
It includes no development/validation evaluation path or cluster launcher.
On a CUDA 12.4-compatible machine with Python 3.10, run:

```sh
bash /path/bundle/bootstrap.sh /path/bundle /path/output
```

`HERO_PYTHON` selects a different Python 3.10 interpreter. Alternatively, with
the bundle's pinned dependencies installed:

```sh
python /path/bundle/run.py stream --bundle /path/bundle \
  --checkpoint /path/qwen-base --output /path/output
python packages/hero-drift-constrained-opd/stream/run.py score /path/output \
  --repository . --output output/score.json
```

Decode greedily with thinking disabled, up to 5,120 new tokens and 16,384 input
tokens. Score the current task after each stage; all seen tasks and full G-test
at 4/8/16/25; tasks 1–8 at 12/20; the nested G probe otherwise. Stage 0 comes
from the separate base-model runner. Inputs over the ceiling get empty predictions.

## Seeds and resume

`--seed` controls evidence and batch order. `--init-seed` independently controls
LoRA initialization and defaults to `--seed`; stage t uses `init_seed + t`.
Each fresh adapter is seeded at creation, so a resumed stage does not depend on
prior RNG consumption. Cross-hardware bitwise equality is not assumed.

The output directory is bound to the bundle digest. Completed stages replay
checksum-verified adapters; use a distinct bundle/output for each seed or
variant. `--test-stages 16,25` restricts test decoding and stops after the last
requested stage. `--decode-tokens` controls batch memory (rows × longest row).

## Paper ablations

| `prepare --variant` | Change |
| --- | --- |
| `full` | Full method, default. |
| `no-harness` | No proposals or guard calls; empty skills. Answer replay, memory and anchors remain. |
| `uniform-residual` | Uniform SHA-256(seed:id) priorities over retained/new records, under the same token budget. |
| `no-checkpoint-anchor` | No checkpoint penalty or multiplier updates. |
| `no-base-anchor` | No base penalty or multiplier updates. |
| `no-anchors` | Neither penalty nor multiplier updates. |
| `sft` | Verified-answer cross-entropy replaces distillation; both constraints remain. |
| `drift-budget-tight` | Checkpoint budget 0.02. |
| `drift-budget-loose` | Checkpoint budget 0.1. |

Disabled anchors still measure drift. SFT skips teacher forwards and shares the
chunked, per-record token-mean accumulation with KL. Logs use `objectiveLoss`;
training summaries include `objective`, `objectiveLossFirst` and
`objectiveLossLastQuarter`. No-harness guard measurements are null. Uniform
residual selection is distinct from the fixed 128-record replay baseline.

For the inference-only `+ harness` variant, replay a completed full run's
adapter chain and attach its final routed harness. The source run is read-only:

```sh
python /path/bundle/run.py evaluate-harness --bundle /path/bundle \
  --run /path/output --checkpoint /path/qwen-base --output /path/with-harness
python packages/hero-drift-constrained-opd/stream/run.py score /path/with-harness \
  --repository . --diagonal-run /path/output --output output/with-harness.json
```

This evaluates all tasks and full G-test at the final stage without training.
Acquisition and the reference for forgetting use the original bare-prompt
diagonal; only the final row changes.
