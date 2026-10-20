# HERD stream runner

Self-contained HERD implementation for the TRACE+17 stream: one GPU job runs every stage.
`hero.py` holds the algorithm, and `run.py` prepares bundles, runs the stream and scores test predictions.
The settings below are the paper's; `plan.json` records them.

## Stage t

Stage t receives the deployed predictor θ_{t-1} (merged weights) and the harness H_{t-1} = (K, R).

1. **Harness evolution.** The window A_t is drawn from the task's training records in a per-stage seeded order,
   whole records only (prompt + answer + 16 tokens each), until the next record would overflow B_A = 8,192 tokens.
   - The frozen θ_{t-1} (the model with its LoRA disabled) answers every record of A_t greedily; up to 4 answers
     that differ from the verified output after normalization are the mistakes shown to the writer.
   - The frozen base model θ_0 (kept on the CPU and moved to the GPU only for this call) writes the task's skill in
     one greedy call with thinking off, at most 768 new tokens. Its prompt (`PROPOSER` in `hero.py`) lists up to 16
     records of A_t, those whose answers open differently first, inputs clipped to 384 tokens (head and last third)
     and outputs to 128, then the mistakes, and asks for a skill under four headers: Format, Rules, Avoid, Examples.
   - The skill's routing prefix is the longest common prefix of the task's training prompts, cut back to a
     whitespace boundary; its instructions are the reply from its Format line on, clipped to 768 tokens. A prompt
     executes only the skill with the longest matching prefix (NFKC, collapsed whitespace, lowercase), shown as
     the system message; a new skill replaces one with the same prefix or task id.
   - The candidate registry is committed only if it lowers the mean residual of A_t under the frozen θ_{t-1}
     (the verified answer's mean NLL with only the routed skill and the bare prompt in context, no replayed answer)
     and its serialization fits B_K = 24,576 tokens. Otherwise K_{t-1} is kept.
2. **Drift-constrained distillation.**
   - The teacher is the frozen θ_{t-1}, given the routed skill as the system message, the prompt, the verified
     answer replayed as its own previous turn, and "Repeat your previous answer exactly, with no other text."
   - The student is θ_{t-1} plus a fresh rank-32 LoRA (alpha 64, no dropout, every text-decoder linear layer) and
     sees the bare prompt. The loss is the full-vocabulary KL(teacher ‖ student) at every position of the verified
     answer (plus the end-of-turn token). A record trains only if its bare prompt and answer fit 16,384 tokens;
     longer records are excluded.
   - Checkpoint drift: KL(θ_{t-1} ‖ student) on the first 32 greedy tokens of θ_{t-1} for up to 8 prompts (at most
     2,048 tokens) per task retained in R_{t-1}, grouped by task. Base drift: KL(θ_0 ‖ student) on the first 32
     greedy tokens of θ_0 for the reserved general set G (48 non-test GSM8K, IFEval and SuperNI records within
     B_G = 8,192 tokens), grouped by source. Anchor hidden states are cached once.
   - Each drift is the worst group's mean KL, held under ε = 0.05 by a multiplier with bounded dual ascent
     (rate 0.5). Each update measures both drifts, adds every active constraint's worst group with weight λ/ε,
     steps on the adapter and then moves the multipliers.
   - 32 updates of 32 records in a seeded shuffle, AdamW without weight decay, lr 1e-4 decaying linearly to zero,
     gradient clip 1.0; the final iterate is kept.
3. **Merge and refresh.** The bf16 adapter is saved and merged into θ_t. R_t = TopBudget(R_{t-1} ∪ A_t) under
   B_R − B_G = 24,576 tokens: records in decreasing residual under θ_t executing K_t, stable-id ties, each kept only
   if it fits whole.
4. **Test predictions** on the formal schedule: the current task after every stage; all seen tasks and the full
   G-test at stages 4, 8, 16 and 25; tasks 1–8 at stages 12 and 20; the 200-item G-test probe otherwise. Greedy,
   thinking off, at most 5,120 new tokens; an input over 16,384 tokens is an empty prediction. Stage 0 is the
   base-model run (`packages/baselines/base`).

`stage-NN/stage.json` records the registry, the guard, the memory, drift before and after training, the training
summary and the adapter's SHA-256; `training-log.jsonl` has every update.

## Use

```sh
python run.py prepare --repository ../../.. --output /path/bundle --seed 42
bash /path/bundle/bootstrap.sh /path/bundle /path/output      # on a CUDA 12.4 machine with CPython 3.10
python run.py score /path/output --repository ../../.. --output score.json
```

`score` needs the evaluator dependencies (`packages/evaluator/python`) next to `torch`. Completed stages are
replayed from their saved adapters, each verified against its `stage.json`, so rerunning the same command continues
the stream. `stream --test-stages 16,25` limits a job to those stages' predictions and stops after the last.
`stream --decode-tokens` sets the decoding batch (rows × longest row; 98,304 fits a 40 GB A100).

### Ablations

| Paper row | `prepare` option |
| --- | --- |
| − skills | `--no-skills` |
| uniform memory | `--memory uniform` |
| − checkpoint anchor | `--checkpoint-budget none` |
| − base anchor | `--general-budget none` |
| − both anchors | `--checkpoint-budget none --general-budget none` |
| SFT objective | `--objective sft` |
| ε_C = 0.02 / 0.1 | `--checkpoint-budget 0.02` / `0.1` |
| + harness (infer.) | `stream --inference-harness <full run>/stage-25/stage.json`, then `score --predictions test-predictions.harness.jsonl` |

A removed anchor is still measured and audited, never constrained.

## Tests

Tests run on CPU with a tiny random Qwen3.5 at `HERO_TINY_QWEN` (they skip without it):

```sh
HERO_TINY_QWEN=/path/tiny-qwen python -m pytest tests
```
