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
