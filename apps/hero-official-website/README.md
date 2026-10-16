# HERD project page

Single-page site that introduces HERD with interactive figures. Built with Vite, React and TypeScript.
Fonts and KaTeX are bundled, so the built page has no external dependencies.

Run these commands from the repository root (dependencies use the root pnpm workspace lockfile).

```sh
pnpm install
pnpm --filter @hero/official-website dev      # local preview
pnpm --filter @hero/official-website build    # type-check and build into dist/
pnpm --filter @hero/official-website deploy   # build, then publish to the Cloudflare Worker named in wrangler.jsonc
```

## Layout

- `src/sections/` holds the page sections in reading order.
- `src/demos/` holds one component per interactive figure.
- `src/styles.css` defines the color tokens. They follow the paper figures: orange harness, blue predictor,
  red optimization, purple constraints, green deployment.

Figures have no playback controls. Each one loops by itself, holds while the pointer rests on it where that helps
reading, and moves to the next scene on a click.

Figures marked "Schematic" use invented values to show a mechanism. Every number in the Experiments section
comes from the experiment outputs: the method comparison at 4, 8, 16 and 25 tasks, the seen-task trajectory,
the acquisition–forgetting and task–general trade-offs, retention and general-capability trajectories,
per-task and per-category matrices, HERD's harness, drift, multiplier, distillation and residual-memory
records per stage, and its per-seed results. HERD is shown as the mean over its seeds.

## Experiment data

`data/experiments.json` is the presentation-only snapshot exported from `dev`
commit `c05385231d9856d2f8bafc95a4613c80ffddf434`. It includes chart aggregates,
not predictions, prompts, checkpoints or run archives. The release builds without
`experiments/outputs/`, which is intentionally absent from `arr`.

To explicitly replace the snapshot from a local output directory:

```sh
node apps/hero-official-website/scripts/export-experiments.mjs /absolute/path/to/outputs
pnpm site:check
```

The page retains its review-time noindex setting. Deployment requires the operator's
own Cloudflare account configuration; no account or production route is bundled.

## Paper alignment and current differences

The method description and `data/paper-results.json` follow `docs/paper/herd.pdf`.
All 84 Table 1 values were checked against page 6. The JSON records the PDF SHA-256;
`public/herd.pdf` links to that exact repository PDF. Main results retain the paper's
printed precision. The supplementary interactive charts preserve the unmodified
presentation aggregates exported from `dev`; they are not a reconstruction of the PDF.

| Area | Paper | Current branch / snapshot |
| --- | --- | --- |
| Skill guard | Strictly lower mean residual, no answer replay (Sections 2.3 and A) | `packages/hero-drift-constrained-opd/stream/run.py`, `evolve`: replay-conditioned teacher NLL may rise by 0.05 |
| Residual refresh | Score under merged weights after distillation (Section 2.3) | The stream runner selects residuals before training under the previous checkpoint |
| Main results | HERD forgetting 0.010; other Table 1 values as printed | The dev snapshot has HERD forgetting 0.009 and several other different rounded aggregates; retained only as labelled supplementary charts |
| Baseline provenance | Appendix B describes OPR and TreeLoRA as paper-based implementations | The branch includes `opr/official`, `opr/upstream` and `treelora/upstream`; the baseline section describes the bundled code |
| Dataset revision | Paper results refer to the reported experiments | Root README documents v1.1 NumGLUE-cm resampling and requires new runs; do not attribute the paper scores to v1.1 |

This website change does not modify research code, the PDF, datasets or experimental
results. The Code button points to the published `arr` research code from which the
local `arXiv` branch was created; update it when `arXiv` is published. No arXiv identifier
has been assigned here, and the supplied PDF remains the anonymous review manuscript.
