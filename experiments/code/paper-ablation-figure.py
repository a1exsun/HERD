#!/usr/bin/env python3
"""Draw the paper's ablation figure (Q4) from experiments/outputs.

A single-column heatmap of each core variant's difference from full HERD at stage 25, in
percentage points, over the five quantities of Q1 (Acq., AA, F, ΔC, ΔC_min), with a strip of the
audited stage-25 drifts D^C and D^0 against their budgets. Red always means worse than full HERD
(the F column is flipped); the colour scale is clipped at ±CLIP points and the number in every cell
is unclipped. Per column, the variant that hurts most is outlined. Rows come from the registry in
experiments/outputs/schema.json (core-tier ablations, in registry order) plus the optional
"deployed with harness" row of Q3, read from the registry id in EXTRA_ROWS when it exists.

Rows whose results.json does not exist yet are drawn as hatched grey placeholders. Row reading is
shared with paper-ablation-table.py.

    python3 experiments/code/paper-ablation-figure.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors, patches  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/outputs"
FIGURE = ROOT / "output/paper/figures/hero-ablation.pdf"
CLIP = 10.0  # percentage points
GRAY = "#8a8a8a"

spec = importlib.util.spec_from_file_location("ablation_table", Path(__file__).with_name("paper-ablation-table.py"))
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)

# (header, lower-is-better, getter) in the figure's order; same quantities as the table.
METRICS = [
    ("Acq.", False, lambda d: d["acquisition"]),
    ("AA", False, lambda d: d["seenMean"]),
    ("F", True, lambda d: d["forgetting"]),
    (r"$\Delta C$", False, lambda d: d["general"]["change"]),
    (r"$\Delta C_{\min}$", False, lambda d: d["general"]["worstCategoryChange"]),
]
DRIFT = [(r"$D^C$", "checkpointDriftFinal", "checkpoint_budget"), (r"$D^0$", "generalDriftFinal", "general_budget")]
# Rows outside the ablation family, appended after the core tier: (registry id, label).
EXTRA_ROWS = [("hero-with-harness", "+ harness (infer.)")]
# Shorter row labels than the table's, to fit a column; keyed by registry id.
ROW_LABELS = {
    "no-harness": r"$-$ skills",
    "uniform-residual": "uniform memory",
    "no-checkpoint-anchor": r"$-$ ckpt anchor",
    "no-base-anchor": r"$-$ base anchor",
    "no-anchors": r"$-$ both anchors",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.5,
    "pdf.fonttype": 42,
})


def extra_rows(full: dict) -> list[dict]:
    registry = table.read(OUT / "schema.json")["registry"]
    rows = []
    for rid, label in EXTRA_ROWS:
        row = {"id": rid, "label": label, "tier": "extra", "placeholder": False, "budgets": full["budgets"]}
        entry = next((e for e in registry if e["id"] == rid), None)
        path = OUT / entry["path"] / "results.json" if entry else None
        if path and path.is_file():
            results = table.read(path)
            depth = (results.get("summary") or {}).get("depths", {}).get(table.DEPTH)
            if depth:
                row.update(depth=depth, drift=table.final_drift(results))
        rows.append(row)
    return rows


def main() -> None:
    full, variants = table.rows()
    rows = [r for r in variants if r["tier"] == "core"] + extra_rows(full)
    n, m = len(rows), len(METRICS)

    delta = np.full((n, m), np.nan)  # signed difference, percentage points
    harm = np.full((n, m), np.nan)  # positive = worse than full HERD
    for i, r in enumerate(rows):
        if "depth" not in r:
            continue
        for j, (_, lower, get) in enumerate(METRICS):
            d = 100 * (get(r["depth"]) - get(full["depth"]))
            delta[i, j] = d
            harm[i, j] = d if lower else -d
    real = np.array([("depth" in r) and not r["placeholder"] for r in rows])

    for r in rows:
        r["label"] = ROW_LABELS.get(r["id"], r["label"])
    fig = plt.figure(figsize=(3.03, 0.36 * n + 0.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[m, 2.0], wspace=0.06, left=0.30, right=0.99, top=0.84, bottom=0.08)
    ax = fig.add_subplot(grid[0])
    bx = fig.add_subplot(grid[1], sharey=ax)

    norm = colors.TwoSlopeNorm(vmin=-CLIP, vcenter=0, vmax=CLIP)
    cmap = plt.get_cmap("RdBu_r")
    ax.imshow(np.clip(harm, -CLIP, CLIP), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for i, r in enumerate(rows):
        for j in range(m):
            if np.isnan(delta[i, j]):
                ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f2f2f2", edgecolor="white", hatch="////", linewidth=0))
                continue
            text = f"{delta[i, j]:+.1f}"
            dark = abs(harm[i, j]) > 0.6 * CLIP
            if r["placeholder"]:
                color = "#e6e6e6" if dark else GRAY
            else:
                color = "white" if dark else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5, color=color,
                    fontstyle="italic" if r["placeholder"] else "normal")
    # Outline the variant that hurts most in each column, among real rows only.
    for j in range(m):
        column = np.where(real, harm[:, j], np.nan)
        if np.all(np.isnan(column)):
            continue
        i = int(np.nanargmax(column))
        ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.0))
    for i in range(n + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.2)
    for j in range(m + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.2)

    ax.set_xticks(range(m))
    ax.set_xticklabels([h for h, _, _ in METRICS])
    ax.xaxis.tick_top()
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] + (r"$^\dagger$" if ("depth" not in r or r["placeholder"]) else "") for r in rows])
    for label, r in zip(ax.get_yticklabels(), rows):
        if "depth" not in r or r["placeholder"]:
            label.set_color(GRAY)
    ax.tick_params(length=0, pad=2)
    for side in ax.spines.values():
        side.set_visible(False)
    # Full HERD's absolute values above the columns.
    for j, (_, _, get) in enumerate(METRICS):
        ax.text(j, -0.5 - 0.62, f"{get(full['depth']):.3f}", ha="center", va="bottom", fontsize=6.5, color=GRAY)
    ax.text(-0.5, -0.5 - 0.62, "HERD", ha="right", va="bottom", fontsize=6.5, color=GRAY)

    # Drift strip: horizontal bars per row, budgets dashed.
    width = 0.34
    limit = 0.0
    shades = [cmap(0.18), cmap(0.34)]  # D^C darker, D^0 lighter
    for k, (_, key, budget) in enumerate(DRIFT):
        offset = (k - 0.5) * (width + 0.04)
        for i, r in enumerate(rows):
            value = (r.get("drift") or {}).get(key)
            if value is None:
                continue
            over = value > r["budgets"][budget]
            if r["placeholder"]:
                color = "#d9d9d9" if k else "#bfbfbf"
            else:
                color = cmap(0.9) if over else shades[k]
            bx.barh(i + offset, value, height=width, color=color, edgecolor="none")
            limit = max(limit, value)
    eps = full["budgets"][DRIFT[0][2]]
    bx.axvline(eps, color="black", linestyle=(0, (2, 1.5)), linewidth=0.6)
    bx.set_xlim(0, max(limit * 1.12, 2 * eps))
    bx.set_xticks([])
    bx.tick_params(axis="y", left=False, labelleft=False)
    for side in bx.spines.values():
        side.set_visible(False)
    bx.text(0.0, -0.5, r"$D^C$ / $D^0$", ha="left", va="bottom", fontsize=8)
    bx.text(eps, -0.5 - 0.62, r"$\varepsilon$", ha="center", va="bottom", fontsize=7, color="black")

    pending = [r for r in rows if "depth" not in r or r["placeholder"]]
    if pending:
        fig.text(0.30, 0.01, r"$^\dagger$ run pending", fontsize=6, color=GRAY, va="bottom")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, format="pdf")
    done = sum(real)
    print(f"wrote {FIGURE.relative_to(ROOT)} ({done}/{n} rows with results)")


if __name__ == "__main__":
    main()
