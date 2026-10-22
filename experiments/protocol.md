# Evaluation protocol

HERD and every baseline are evaluated under one protocol. Its constants are
implemented in `packages/baselines/python/src/hero_baselines/protocol.py`,
`packages/baselines/base/run.py` and
`packages/hero-drift-constrained-opd/stream/run.py`.

## Shared controls

Use all 25 TRACE+17 tasks in canonical order with the seed-42 data
construction ([`../datasets/trace-plus-17`](../datasets/trace-plus-17)), and
the pinned `Qwen/Qwen3.5-4B` model and tokenizer at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with thinking disabled. A method
learns only from the agent-visible training split; each record's `answer` is its
training target and `references` hold every valid answer for scoring.
Validation gold and test records are scorer-only inputs (`hidden-scorer/`) and
never enter a model workspace. No discarded historical data re-enters the
learner.

Development and calibration use validation data. Test results are never used
for tuning, candidate selection or checkpoint selection.

## General capability

G-test has five equal-weight categories of 200 held-out items each: general
knowledge (MMLU), mathematical reasoning (GSM8K), commonsense reasoning
(HellaSwag), instruction following (IFEval) and multilingual understanding
(XCOPA). A fixed 200-item probe (40 per category) is nested in it, and a
disjoint 200-item development panel (40 per category) serves validation. Its
construction is recorded in `../datasets/general-test-v1.json`. No G-test, probe or
development record may enter proposals, residual ranking, drift constraints or
checkpoint selection. Methods that need general anchors draw them from the
reserved non-test records (`../datasets/general-probes-v1.json` and the G-test's
non-test split).

MMLU, HellaSwag and XCOPA use the same answer extraction in
`hero-task-scores-v3`. The last explicit answer declaration takes precedence;
otherwise the response must contain a single clearly labelled option at its
opening or closing. Markdown wrappers and letter case are normalized. Invalid
labels, ambiguous selections, option lists without a selection, and unresolved
answers score zero. Extraction never consults the reference answer. MMLU and
HellaSwag allow A–D; XCOPA allows A–B. TRACE's official first-character choice
scoring, GSM8K final-number scoring and IFEval strict verification are unchanged.

## Test schedule

Stage 0 is the base model (`packages/baselines/base`): all 25 task test sets and
the full G-test. After stage `t`:

- the current task's test set is scored at every stage (the diagonal);
- all seen tasks and the full G-test are scored at stages 4, 8, 16 and 25;
- tasks 1–8 are scored at stages 12 and 20;
- the 200-item G probe is scored at every other stage.

Each task cell uses all 100 test items. Decoding is greedy with thinking off,
at most 5,120 new tokens and 16,384 input tokens, with the model/tokenizer EOS
union as stop tokens. LoRA baselines decode their merged checkpoints with vLLM
0.31.0 (each package's own `vllm_decode.py`); HERD and MoRAM decode with
Transformers (MoRAM's input-dependent routing cannot be merged into the
weights); frozen-model methods use MLX, PyTorch's MPS backend or vLLM servers.
An input above the limit receives an empty prediction and stays in the
denominator. Failures score zero and are counted separately.

## Metrics

`R[t,j]` is the score on task `j` after stage `t`, each task scored by its
manifest metric. Task means weight every task equally.

- Acquisition: `mean_j R[j,j]`, the mean diagonal.
- Seen-task mean after stage `t`: the mean of `R[t,j]` over `j ≤ t` at the
  full-evaluation stages; at stage 25 this is AA.
- Forgetting: `F = mean_{j<T} max(0, R[j,j] - R[T,j])`.
- Signed BWT: `mean_{j<T} (R[T,j] - R[j,j])`.
- General capability: the macro mean over the five categories per evaluated
  panel. `ΔC` is the mean change of the five category scores after stage 25
  against stage 0, and `ΔC_min` the worst category's change.

Results are means over seeds 42, 43 and 44 with sample standard deviations.
Intervals are 95% item bootstraps (4,000 draws), resampled within task and
G-test category and paired between HERD and each baseline
(`code/export-paper-experiments.py`). NumGLUE-cm's 500/100/100 records include
within-split resampling; the test panel is unchanged.

## Execution

Prepare a bundle and execute it through the HERD stream or the relevant
baseline runner. HERD test predictions are scored locally with `run.py score`;
test gold never leaves the local archive.
