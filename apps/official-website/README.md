# HERD project page

Single-page site that introduces HERD with interactive figures, published at <https://a1exsun.github.io/HERD/>.
Built with Vite, React and TypeScript. Fonts and KaTeX are bundled, so the built page has no external dependencies.

Run these commands from the repository root (dependencies use the root pnpm workspace lockfile).

```sh
pnpm install
pnpm site:dev                                  # local preview
pnpm site:check                                # lint, test, type-check and build into dist/
```

Pushing changes under `apps/official-website/` to `main` deploys the site to GitHub Pages
(`.github/workflows/pages.yml`).

`pnpm --filter @hero/official-website readme:figures` renders the site's figures to animated, theme-aware SVG in
`herd-code/public/readme/` for the repository README; rerun it after changing a figure and commit the output.

## Layout

- `src/sections/` holds the page sections in reading order.
- `src/demos/` holds one component per interactive figure.
- `src/styles.css` defines the color tokens. They follow the paper figures: orange harness, blue weights,
  red optimization, purple constraints, green deployment.

Figures have no playback controls. Each one loops by itself, holds while the pointer rests on it where that helps
reading, and moves to the next scene on a click. Figures marked "Schematic" use invented values to show a mechanism.

## Paper alignment

The text, notation and reported results follow the final version of the paper in `herd-paper/`. The page does
not bundle the PDF; the Paper button points to arXiv once the preprint is posted.

- `data/paper-results.json` holds only what the paper's main text reports: Table 1, the Figure 3 long-horizon
  series and the Figure 4 ablations. Table 1 and Figure 4 are parsed from
  `herd-paper/tables/*.tex`; the Figure 3 series match every value the text quotes and Table 1's G-test column,
  which `pnpm test` checks. The page carries no experiment snapshot, per-stage outputs or appendix-level results.
- The results are drawn in HTML and SVG by the page itself (sortable table, frontier scatter, linked long-horizon
  charts and heatmap, ablation heatmap); the paper's raster figures are not used.

When the paper changes, update the section text and `data/paper-results.json` together.

### Known differences between the paper and the research code

The page describes the paper. The stream runner in `packages/hero-drift-constrained-opd/stream/run.py` differs in
two places, and is unchanged here:

| Area | Paper | Runner |
| --- | --- | --- |
| Skill guard | Strictly lower mean residual, no answer replay | Replay-conditioned teacher NLL may rise by `--harm-tolerance` |
| Residual refresh | Scored under the merged weights after distillation | Residuals selected before training under the previous checkpoint |
