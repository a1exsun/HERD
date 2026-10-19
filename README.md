<div align="center">

# HERD

**Harness-Evolving Recursive Distillation for Continual Learning**

<p align="center">
  <a href="https://a1exsun.github.io/HERD/#top">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/readme/banner-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="public/readme/banner-light.svg">
      <img src="public/readme/banner-light.svg" width="100%" alt="HERD: Learn in the harness. Internalize in the weights.">
    </picture>
  </a>
  <br><sub>An evolving harness for learning. The weights alone for deployment.</sub>
</p>

<p>
  <a href="https://a1exsun.github.io/HERD/"><strong>Project website ↗</strong></a> &nbsp;·&nbsp;
  <strong>Paper: arXiv link coming soon</strong> &nbsp;·&nbsp;
  <a href="https://a1exsun.github.io/HERD/#method"><strong>Interactive demos ↗</strong></a> &nbsp;·&nbsp;
  <a href="#getting-started"><strong>Get started ↓</strong></a>
</p>
<p><sub>Qwen3.5-4B &nbsp; / &nbsp; TRACE+17 &nbsp; / &nbsp; 25 sequential tasks</sub></p>

</div>

HERD is a hybrid continual learning framework for language models: it learns efficiently in an external **harness** and internalizes that experience into the **weights** by distillation. On the harness side, it refines skills that capture reusable patterns and keeps, in a bounded memory, the high-residual examples the skills miss. On the parameter side, drift-constrained distillation with dual anchors, the previous checkpoint for per-stage drift and the base model for cumulative drift, reduces forgetting over long recursive runs. At deployment, **the weights run on their own**, without the skill registry or memory in the prompt.

<p align="center">
  <a href="#see-herd-in-action">Demos</a> &nbsp;·&nbsp;
  <a href="#how-it-works">How it works</a> &nbsp;·&nbsp;
  <a href="#getting-started">Quick start</a> &nbsp;·&nbsp;
  <a href="#results">Results</a> &nbsp;·&nbsp;
  <a href="#documentation">Documentation</a> &nbsp;·&nbsp;
  <a href="#citation">Citation</a>
</p>

<table>
  <tr>
    <td width="33%" valign="top"><h3>Explore the idea</h3><p>Watch skills and model weights evolve across a task stream.</p><a href="https://a1exsun.github.io/HERD/#method">Open the interactive walkthrough →</a></td>
    <td width="33%" valign="top"><h3>Run the code</h3><p>Set up the workspace, prepare data, and configure a continual-learning experiment.</p><a href="#getting-started">Start with the setup guide →</a></td>
    <td width="33%" valign="top"><h3>Build on HERD</h3><p>Inspect the harness, distillation runtime, shared scorers, and baseline implementations.</p><a href="#documentation">Find the relevant component →</a></td>
  </tr>
</table>

## See HERD in action

<p align="center">
  <a href="https://a1exsun.github.io/HERD/#top">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/readme/hero-stream-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="public/readme/hero-stream-light.svg">
      <img src="public/readme/hero-stream-light.svg" width="100%" alt="Animated task stream: receive task data, update the harness, distill, and deploy the weights; carry both learning states forward.">
    </picture>
  </a>
  <br><sub>Schematic · Reuses the website’s animated task-stream figure. Click to explore the live version.</sub>
</p>


## How it works

<table>
  <tr><th align="left">① Evolve the harness</th><th align="left">② Distill with two anchors</th></tr>
  <tr>
    <td width="50%" valign="top">The frozen base model writes a skill from a bounded window of the task and the checkpoint&rsquo;s mistakes. The guard commits it only if it lowers the checkpoint&rsquo;s mean residual on the window.</td>
    <td width="50%" valign="top">The previous checkpoint sees the skill and the verified answer; a fresh LoRA student sees the bare prompt. The previous checkpoint bounds per-stage drift on probes from the memory, and the base model bounds cumulative drift on general probes.</td>
  </tr>
  <tr><th align="left">③ Merge and refresh</th><th align="left">④ Deploy the weights</th></tr>
  <tr>
    <td valign="top">Merge the adapter into the inherited weights, then rescore whole records and retain those with high residuals within the memory budget.</td>
    <td valign="top">Carry the harness and merged weights into the next learning stage. Inference uses the weights alone, with no added harness context.</td>
  </tr>
</table>

<p align="center">
  <a href="https://a1exsun.github.io/HERD/#overview">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/readme/stage-anatomy-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="public/readme/stage-anatomy-light.svg">
      <img src="public/readme/stage-anatomy-light.svg" width="100%" alt="Animated HERD stage: bounded window, harness evolution, dual-anchor distillation, and deployment of the merged weights.">
    </picture>
  </a>
  <br><sub>Schematic · One stage of the recursive learning loop, rendered from the website component.</sub>
</p>


<details>
<summary><strong>Compare learning paradigms</strong> — what changes, what is carried, what is deployed</summary>

<p align="center">
  <a href="https://a1exsun.github.io/HERD/#overview">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/readme/paradigms-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="public/readme/paradigms-light.svg">
      <img src="public/readme/paradigms-light.svg" width="100%" alt="Animated comparison of non-parametric continual learning, HERD, and parametric continual learning.">
    </picture>
  </a>
  <br><sub>Schematic · HERD carries both external state and parameters between stages, and deploys the weights alone.</sub>
</p>


</details>

<details>
<summary><strong>Watch the LoRA merge</strong> — from a fresh adapter to the next checkpoint</summary>

<p align="center">
  <a href="https://a1exsun.github.io/HERD/#drift-constrained-distillation">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/readme/lora-merge-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="public/readme/lora-merge-light.svg">
      <img src="public/readme/lora-merge-light.svg" width="100%" alt="Animated LoRA update: train a rank-limited adapter, merge it into inherited weights, and pass the weights to the next stage.">
    </picture>
  </a>
  <br><sub>Schematic · The website’s LoRA animation, exported for inline playback.</sub>
</p>


</details>

<details>
<summary><strong>Objective, memory, and drift constraints</strong></summary>

Distillation minimizes token-averaged forward KL from teacher to student on verified-answer prefixes. Earlier retained records supply checkpoint-anchor probes, rather than replay targets.

Each anchor measures the **worst group's mean KL** over the first 32 greedy response tokens. Checkpoint probes are grouped by task; general probes are grouped by source. Both budgets are 0.05, and dual ascent adjusts their multipliers. These probe constraints do not guarantee zero forgetting.

See the paper (arXiv link coming soon) and the [stream runner guide](packages/hero-drift-constrained-opd/stream/README.md) for the objective and execution details.

</details>

## Getting started

You can browse the [live demos](https://a1exsun.github.io/HERD/#method) without installing anything. For local use, choose the website preview or experiment setup below. Existing package names, environment variables, and CLI paths use `hero`.

### Preview the website locally

Requires **Node.js ≥22.19.0** and **pnpm 11.12.0**. No model download or GPU is needed.

```sh
git clone https://github.com/a1exsun/HERD.git
cd HERD
pnpm install --frozen-lockfile
pnpm site:dev
```

Open the local URL printed by Vite. The page includes the interactive figures and the main results reported in the paper.

### Run an experiment

Use the checkout above and run all following commands from its root.

#### 1. Set up the workspace

Requires **Node.js ≥22.19.0**, **pnpm 11.12.0**, **Python 3.12**, and **uv**. GPU bundles pin their CUDA dependencies separately.

```sh
cp .env.example .env
# Replace this path with the source directory described in .env.example.
export HERO_TRACE_SOURCE_ROOT=/path/to/raw
pnpm install --frozen-lockfile
pnpm build
```

<details>
<summary><strong>Focused verification commands</strong></summary>

Run the checks for the component you changed:

```sh
pnpm test          # TypeScript workspace tests
pnpm eval:test     # Shared Python scorers
pnpm opd:test      # Distillation runtime and stream tests
pnpm opd:validate  # Python source compilation checks
pnpm analysis:test # Statistical analysis and data preparation tests
```

These checks do not run GPU training. See the DSH section for integration checks.

</details>

#### 2. Prepare the model and data

Follow [datasets/trace-plus-17/README.md](datasets/trace-plus-17/README.md) for source revisions, directory layout, and dataset terms. Prepare the model first: the builder needs its tokenizer.

```sh
pnpm model:setup
pnpm dataset:build
pnpm dataset:verify
```

**Dataset v1.1.** All tasks have 500/100/100 records. NumGLUE-cm expands 256/48 training/validation prompts by deterministic within-split sampling with replacement; its 100 test records stay unchanged. Repeats have unique IDs and `metadata.resampledFrom`. JSONL files remain local. Use the same manifest across methods: v1.1 changes sample multiplicities and requires new runs, separate from prior artifacts and the reported paper results.

Also prepare G-test, reserved general probes, and scorer assets as described in [experiments/protocol.md](experiments/protocol.md) and the stream README; `dataset:build` covers TRACE+17 only.

#### 3. Configure and run an experiment

```sh
uv run --locked --project packages/hero-drift-constrained-opd/python --python 3.12 \
  python packages/hero-drift-constrained-opd/stream/run.py prepare --help
uv run --locked --project packages/hero-drift-constrained-opd/python --python 3.12 \
  python packages/hero-drift-constrained-opd/stream/run.py stream --help
```

The CLI provides `prepare` (bundle), `stream` (GPU run), `report` (validation), and `score` (test scoring). See [packages/hero-drift-constrained-opd/stream/README.md](packages/hero-drift-constrained-opd/stream/README.md) for execution and cluster submission. Set `--split test` when preparing a run for the formal test schedule.

<details>
<summary><strong>Paper configuration</strong></summary>

CLI defaults differ from the paper. Set these parameters explicitly and save `plan.json` with each run.

| Parameter | Paper setting |
| --- | --- |
| Stages | 25 |
| LoRA rank / alpha / dropout | 32 / 64 / 0 |
| Updates / records per update | 32 / 32 |
| Learning rate | 1e-4, linearly decayed to zero |
| Training sequence limit | 16,384 tokens |
| Current-task window | 8,192 tokens |
| Skill registry / individual skill | 24,576 / 768 tokens |
| Memory budget | 32,768 tokens, including 8,192 reserved for general probes |
| Checkpoint probes | Up to 8 per retained task; first 32 greedy tokens |
| Checkpoint / base drift budgets | 0.05 / 0.05 |
| Dual-ascent rate | 0.5 |

The runner's skill guard also differs from the paper's strict residual test: it accepts a candidate whose replay-conditioned teacher NLL rises by at most `--harm-tolerance`. Check the saved plan against the manuscript. Baseline entry points and dependencies are in [packages/baselines/README.md](packages/baselines/README.md).

</details>

<details>
<summary><strong>Score and analyze your runs</strong></summary>

#### 4. Score and analyze results

Decode and score with the method runner, then analyze its scored item panels:

```sh
python3 experiments/code/summarize.py /path/scored-panels.json \
  --reference /path/baseline-panels.json --output output/comparison.json
```

Input: `taskOrder` and `runs`, with `learned`, `final`, `generalBase`, and `generalFinal` panels per seed. Each panel maps tasks or categories to `{item_id: score}`. Keep failed predictions as zero; match item IDs and manifests across methods. Omit `--reference` for a single method. See [experiments/code/README.md](experiments/code/README.md) for the schema.

Outputs include stream metrics, seed means, sample SD, and 95% item-bootstrap intervals (4,000 draws). Seeds remain fixed: intervals measure item-sampling uncertainty, not training-seed uncertainty.

The repository does not include raw prediction archives, trained checkpoints, or full run archives; the website shows only the results reported in the paper. Runners and analysis write outputs locally. The reported results below come from the paper.

</details>

<details>
<summary><strong>Optional: local DeepSeek Harness (DSH) console</strong></summary>

#### 5. Open the local DSH interface

To open the optional DeepSeek Harness (DSH) console:

```sh
pnpm model:setup  # Download the pinned model if it is not already available.
pnpm dsh:web
```

`dsh:web` installs the plugin and starts the host. Open <http://127.0.0.1:3080/hero/>. Model bridge: <http://127.0.0.1:8192/v1>.

Optional checks: `pnpm model:smoke` and `pnpm dsh:smoke`. They require the model/host environment; the model smoke runs inference.

</details>

## Results

<table>
  <thead>
    <tr>
      <th align="center">Final task performance ↑</th>
      <th align="center">New-task acquisition ↑</th>
      <th align="center">Forgetting ↓</th>
      <th align="center">Added inference context</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>0.706</strong><br><sub>AA after stage 25</sub></td>
      <td align="center"><strong>0.713</strong><br><sub>Mean score when learned</sub></td>
      <td align="center"><strong>0.010</strong><br><sub>Acquisition-to-final loss</sub></td>
      <td align="center"><strong>0 tokens</strong><br><sub>Merged weights only</sub></td>
    </tr>
  </tbody>
</table>

<p align="center"><sub>TRACE+17 · Means over seeds 42, 43, and 44.</sub></p>


HERD has the highest mean AA in this comparison, though the paired 95% interval for its AA difference from uniform replay includes zero. Acquisition is statistically indistinguishable from Seq-LoRA. General-capability loss is the smallest among weight-updating methods: **ΔC = −0.063**, with a worst-category change of **−0.130**.

<p align="center">
  <a href="https://a1exsun.github.io/HERD/#experiments">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="public/readme/results-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="public/readme/results-light.svg">
      <img src="public/readme/results-light.svg" width="100%" alt="Long-horizon stability on TRACE+17: HERD keeps its mean on the first eight tasks and loses the least G-test score, while Seq-LoRA dips at stage 12 and loses most of its general capability by stage 4.">
    </picture>
  </a>
  <br><sub>Drawn from the website's chart component and the paper's reported values. Click to explore the interactive version.</sub>
</p>

<p><a href="https://a1exsun.github.io/HERD/#experiments"><strong>Explore the results on the website →</strong></a></p>

<details>
<summary><strong>Full comparison · TRACE+17 after stage 25</strong></summary>

<table>
  <thead>
    <tr><th>Method</th><th>Family</th><th>AA ↑</th><th>Acq. ↑</th><th>F ↓</th><th>BWT ↑</th><th>G-test ↑</th><th>ΔC ↑</th><th>ΔC<sub>min</sub> ↑</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>HERD</strong></td><td>Hybrid · this work</td><td><strong>0.706</strong></td><td>0.713</td><td>0.010</td><td>−0.007</td><td>0.774</td><td>−0.063</td><td>−0.130</td></tr>
    <tr><td>Base</td><td>Control</td><td>0.419</td><td>0.419</td><td>0.000</td><td>0.000</td><td>0.837</td><td>0.000</td><td>0.000</td></tr>
    <tr><td>Static harness</td><td>Control</td><td>0.432</td><td>0.432</td><td>0.000</td><td>0.000</td><td>0.791</td><td>−0.046</td><td>−0.135</td></tr>
    <tr><td>HCL-Stable</td><td>Non-parametric</td><td>0.433</td><td>0.435</td><td>0.003</td><td>−0.002</td><td><strong>0.823</strong></td><td><strong>−0.014</strong></td><td><strong>−0.085</strong></td></tr>
    <tr><td>ACE</td><td>Non-parametric</td><td>0.427</td><td>0.428</td><td><strong>0.001</strong></td><td><strong>−0.001</strong></td><td>0.748</td><td>−0.089</td><td>−0.320</td></tr>
    <tr><td>Seq-LoRA</td><td>Parametric</td><td>0.694</td><td><strong>0.715</strong></td><td>0.025</td><td>−0.022</td><td>0.597</td><td>−0.240</td><td>−0.665</td></tr>
    <tr><td>Uniform replay</td><td>Parametric</td><td>0.699</td><td>0.711</td><td>0.020</td><td>−0.013</td><td>0.599</td><td>−0.238</td><td>−0.675</td></tr>
    <tr><td>OPR</td><td>Parametric</td><td>0.690</td><td>0.705</td><td>0.024</td><td>−0.016</td><td>0.614</td><td>−0.223</td><td>−0.625</td></tr>
    <tr><td>TreeLoRA</td><td>Parametric</td><td>0.686</td><td>0.711</td><td>0.032</td><td>−0.026</td><td>0.605</td><td>−0.232</td><td>−0.635</td></tr>
    <tr><td>MoRAM</td><td>Parametric</td><td>0.693</td><td>0.701</td><td>0.019</td><td>−0.009</td><td>0.637</td><td>−0.200</td><td>−0.650</td></tr>
    <tr><td>SDFT</td><td>Parametric</td><td>0.021</td><td>0.024</td><td>0.003</td><td>−0.003</td><td>0.172</td><td>−0.665</td><td>−0.770</td></tr>
    <tr><td>OPHSD</td><td>Hybrid</td><td>0.620</td><td>0.639</td><td>0.021</td><td>−0.020</td><td>0.582</td><td>−0.255</td><td>−0.645</td></tr>
  </tbody>
</table>

**Metrics.** AA: final mean over 25 tasks. Acq.: mean score at acquisition. F: mean nonnegative acquisition-to-final drop over tasks 1–24. BWT: the signed mean change over the same tasks. G-test: macro mean over five categories. ΔC and ΔC<sub>min</sub>: mean and worst-category changes from the base. Scores are on a 0–1 scale; values are manuscript means. Bold: best learning method per column, controls excluded.

**Baseline context.** In these runs, SDFT collapses, so its low F does not indicate retention. ACE hits the input ceiling and rolls back later updates; HCL-Stable accepts no proposals. Results apply to these implementations and budgets.

</details>

<details>
<summary><strong>Evaluation protocol and scope</strong></summary>

### Evaluation protocol

| Item | Configuration |
| --- | --- |
| Base model | Qwen3.5-4B; pinned checkpoint revision in the runners |
| Task stream | TRACE+17: 8 TRACE tasks followed by 17 Super-NaturalInstructions tasks |
| Splits | Target: 500 train / 100 validation / 100 test per task; see the dataset note above for NumGLUE-cm resampling |
| General-capability test | 1,000 held-out items: 200 each from MMLU, GSM8K, HellaSwag, IFEval, and XCOPA |
| Update budget | 32 updates × 32 records per stage for weight-updating methods |
| Main-run seeds | 42, 43, 44; mean and sample standard deviation |
| Full evaluation | All seen tasks and full G-test after stages 4, 8, 16, and 25; current-task evaluation after every stage |
| Retention panels | Tasks 1–8 additionally at stages 12 and 20; a 200-item G-test subset at non-anchor stages |
| Test decoding | Greedy, thinking off; at most 16,384 input and 5,120 generated tokens |
| Uncertainty | 95% item-bootstrap intervals with 4,000 draws, resampled within task or category |

General-anchor training probes are disjoint from G-test. Main comparisons use three seeds; ablations use one. Evaluation covers one base model and one task order; some forgetting and general-capability loss remain.

</details>

## Documentation

| Start here | What you will find |
| --- | --- |
| Paper (arXiv link coming soon) | Method, evaluation, ablations, and limitations |
| [Experiment protocol](experiments/protocol.md) | Task stream, budgets, metrics, and evaluation schedule |
| [Dataset guide](datasets/trace-plus-17/README.md) | Sources, revisions, construction, and dataset terms |
| [Stream runner](packages/hero-drift-constrained-opd/stream/README.md) | Prepare, train, report, and score |
| [Baselines](packages/baselines/README.md) | Comparison methods, dependencies, and provenance |
| [Analysis](experiments/code/README.md) | Scored-panel schema and statistical summaries |
| [Website](apps/official-website/README.md) | Interactive figures and local development |

<details>
<summary><strong>Repository map</strong></summary>

| Path | Purpose |
| --- | --- |
| [`packages/hero-drift-constrained-opd/`](packages/hero-drift-constrained-opd/) | HERD stream runner and Python distillation runtime |
| [`packages/hero-harness-evolving/`](packages/hero-harness-evolving/) | Harness evolution components |
| [`packages/hero-dsh-plugin/`](packages/hero-dsh-plugin/), [`apps/console/`](apps/console/) | Local DSH integration |
| [`packages/baselines/`](packages/baselines/) | Comparison methods and provenance |
| [`packages/evaluator/`](packages/evaluator/) | Shared evaluation contracts and scorers |
| [`packages/hero-data-preprocessing/`](packages/hero-data-preprocessing/) | Dataset construction tools |
| [`config/datasets/`](config/datasets/), [`datasets/trace-plus-17/`](datasets/trace-plus-17/) | Dataset configurations and manifests |
| [`experiments/code/`](experiments/code/) | Statistical analysis |
| [`apps/official-website/`](apps/official-website/) | Project website and source figures |

</details>

<details>
<summary><strong>Refresh the README animations</strong></summary>

The inline SVG animations are generated from the website’s React figures. They include light and dark variants and honor reduced-motion preferences. To regenerate them after editing a source figure:

```sh
pnpm --filter @hero/official-website readme:figures
```

Commit the generated files in [`public/readme/`](public/readme/) with the relevant source change. GitHub displays the exported animation; click a figure to use the interactive website.

</details>

## Compute

A HERD stage runs a fixed number of updates and probes, so its training time tracks answer length, not the stage index: 316 s at stage 1, 284 s at stage 25, and 10,723 s in total, against Seq-LoRA's 10,463 s. OPHSD, which regenerates its teacher context by rollouts at every stage, needs 659,000 s. ACE adds its 15,410-token playbook to every inference prompt; HERD deploys bare weights.

## Questions and contributions

Use [GitHub Issues](https://github.com/a1exsun/HERD/issues) for questions, reproducible bugs, and proposed improvements. For an experiment issue, include the method, seed, saved plan, code revision, environment, and a minimal error log. Remove credentials and private infrastructure details.

For a contribution, keep the change focused and run the checks for the affected component. Website changes use `pnpm site:check`; runtime checks are listed under [Getting started](#getting-started). Do not alter recorded experimental results to make a check pass.

## Citation

If HERD informs your work, please cite the paper:

```bibtex
@misc{sun2026herd,
  title  = {HERD: Harness-Evolving Recursive Distillation for Continual Learning},
  author = {Sun, Xun and Nguyen, Thanh Thi and Vu, Thuy-Trang},
  year   = {2026},
  note   = {Preprint},
  url    = {https://a1exsun.github.io/HERD/}
}
```

## License and attribution

This repository does not currently declare a project-wide license. Third-party code and datasets retain their original terms; consult the [baseline provenance](packages/baselines/README.md), [dataset guide](datasets/trace-plus-17/README.md), component license files, and `UPSTREAM.json` records before reuse.

<p align="center"><a href="https://a1exsun.github.io/HERD/"><strong>Explore HERD on the project website ↗</strong></a></p>
