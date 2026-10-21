#!/usr/bin/env python3
"""Draw the paper's long-horizon figure (Figure 3) from experiments/outputs.

Everything shown is read from the registry in experiments/outputs/schema.json, the stream and
G-test definitions in protocol.json, and each method's results.json and method.json, in the
same way as experiments/code/paper-main-table.py draws Table 1. No number is written here.

    (a) Mean score on the first eight tasks at the retention stages (series.firstEight), every
        learning method in registry order with HERD first; the base model as a dotted reference.
    (b) Full G-test macro mean through the stream (series.generalFull), starting from the base
        model at stage 0; the base model as a dotted reference.
    (c) Change of every G-test category after the last stage against the base model, in points,
        for the weight-updating methods (summary.depths[<last>].general.categories); the outlined
        cell is each method's worst category.

Registry entries that carry a `tier` (the ablations) are left to the ablation figure. Series
styles are keyed by method id; an unknown id falls back to a neutral style, so a new baseline
appears without editing this file.

    python3 experiments/code/paper-long-horizon-figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/outputs"
FIGURE = ROOT / "output/paper/figures/long-horizon.pdf"

# Shared figure palette.
INK, MUTED, GUIDE = "#333333", "#666666", "#c3c3c3"
RED, BLUE, ORANGE, GREEN, PURPLE = "#a43f39", "#356493", "#a76120", "#3c7650", "#785394"

# Dual encoding per method: colour, marker and line style, so the curves separate in grayscale.
STYLE = {
    "hero": dict(color=RED, marker="o", linestyle="-", linewidth=1.8, markersize=4.2, zorder=10),
    "seq-lora": dict(color=BLUE, marker="s", linestyle="--"),
    "uniform-replay": dict(color=PURPLE, marker="^", linestyle="-."),
    "opr": dict(color=ORANGE, marker="D", linestyle=":"),
    "treelora": dict(color=GREEN, marker="v", linestyle=(0, (5, 2, 1, 2))),
    "moram": dict(color="#6f93b8", marker="p", linestyle=(0, (3, 1, 1, 1, 1, 1))),
    "sdft": dict(color="#8c8c8c", marker="x", linestyle=":"),
    "ophsd": dict(color="#2a8c8c", marker="*", linestyle="-", markersize=5),
    "ace": dict(color="#9a9a9a", marker="<", linestyle="--"),
    "hcl-stable": dict(color="#bdbdbd", marker=">", linestyle=":"),
}
FALLBACK = dict(color=MUTED, marker=".", linestyle="-")
WEIGHT_UPDATING = {"ours", "parametric", "hybrid"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman"],
    "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "pdf.compression": 9,
})


def read(path: Path):
    return json.loads(path.read_text())


def methods() -> list[dict]:
    """Every registry method with results, HERD first, then registry order; ablations left out."""
    rows = []
    for entry in read(OUT / "schema.json")["registry"]:
        directory = OUT / entry["path"]
        if entry.get("tier") or not (directory / "results.json").is_file():
            continue
        rows.append({
            "id": entry["id"],
            "family": entry["family"],
            "short": read(directory / "method.json")["short"],
            "results": read(directory / "results.json"),
        })
    return [r for r in rows if r["family"] == "ours"] + [r for r in rows if r["family"] != "ours"]


def style(method: dict) -> dict:
    s = dict(linewidth=1.0, markersize=3.4, zorder=3)
    s.update(STYLE.get(method["id"], FALLBACK))
    return s


def series(method: dict, key: str) -> list[dict]:
    return sorted(method["results"]["series"].get(key, []), key=lambda p: p["stage"])


def draw_lines(ax, rows: list[dict], key: str, base_value: float, prepend_base: bool) -> None:
    for m in rows:
        points = series(m, key)
        if not points:
            continue
        stages = [p["stage"] for p in points]
        values = [p["value"] for p in points]
        sds = [p.get("sd", 0.0) for p in points]
        if prepend_base:
            stages, values, sds = [0, *stages], [base_value, *values], [0.0, *sds]
        s = style(m)
        ax.errorbar(stages, values, yerr=sds, label=m["short"], elinewidth=0.5, capsize=0,
                    ecolor=s["color"], **s)
    ax.axhline(base_value, color=INK, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("Stage")


def draw_heatmap(ax, rows: list[dict], categories: list[dict], base_categories: dict, last: str,
                 clip: float) -> None:
    ids = [c["id"] for c in categories]
    cmap = LinearSegmentedColormap.from_list("paper-diverging", [RED, "#ffffff", BLUE])
    norm = Normalize(-clip, clip)
    for i, m in enumerate(rows):
        scores = m["results"]["summary"]["depths"][last]["general"]["categories"]
        deltas = [100 * (scores[c] - base_categories[c]) for c in ids]
        worst = min(range(len(ids)), key=lambda j: deltas[j])
        for j, d in enumerate(deltas):
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=cmap(norm(d)), edgecolor="white",
                                   linewidth=0.6))
            dark = abs(d) > clip * 0.55
            text = f"{d:+.0f}" if round(d) != 0 else "0"
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=6.8,
                    color="white" if dark else INK, fontweight="bold" if m["family"] == "ours" else "normal")
        ax.add_patch(Rectangle((worst, i), 1, 1, facecolor="none", edgecolor=INK, linewidth=0.9,
                               zorder=5))
    ax.set_xlim(0, len(ids))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(ids))])
    ax.set_xticklabels([f"{c['source']}\n{base_categories[c['id']]:.2f}" for c in categories],
                       fontsize=6.4, linespacing=1.1)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([m["short"] for m in rows], fontsize=6.8)
    for label, m in zip(ax.get_yticklabels(), rows):
        if m["family"] == "ours":
            label.set_fontweight("bold")
    ax.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def main() -> None:
    rows = methods()
    protocol = read(OUT / "protocol.json")
    categories = protocol["general"]["categories"]
    retention = protocol["schedule"]["retention"]
    base = next(r for r in rows if r["id"] == "base")
    base_general = base["results"]["summary"]["final"]["general"]
    last = str(max(int(k) for k in next(r for r in rows if r["family"] == "ours")["results"]["summary"]["depths"]))
    learners = [r for r in rows if r["family"] != "control"]
    weight_updating = [r for r in learners if r["family"] in WEIGHT_UPDATING]
    stages_a = sorted({p["stage"] for m in learners for p in series(m, "firstEight")})
    stages_b = [0, *sorted({p["stage"] for m in learners for p in series(m, "generalFull")})]
    base_first_eight = series(base, "firstEight")[0]["value"]

    fig = plt.figure(figsize=(6.3, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.32], wspace=0.46,
                          left=0.075, right=0.995, top=0.91, bottom=0.30)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, k]) for k in range(3))

    draw_lines(ax_a, learners, "firstEight", base_first_eight, prepend_base=False)
    ax_a.set_title(f"(a) Mean score on the first {retention['tasks']} tasks", loc="left")
    ax_a.set_ylabel("Mean score")
    ax_a.set_xticks(stages_a)
    ax_a.set_ylim(0, 0.75)
    ax_a.set_yticks([0, 0.2, 0.4, 0.6])

    draw_lines(ax_b, learners, "generalFull", base_general["macro"], prepend_base=True)
    ax_b.set_title(f"(b) {protocol['general']['name']} macro mean", loc="left")
    ax_b.set_ylabel("Macro mean")
    ax_b.set_xticks(stages_b)
    ax_b.set_ylim(0.1, 0.9)
    ax_b.set_yticks([0.2, 0.4, 0.6, 0.8])

    clip = 40
    draw_heatmap(ax_c, weight_updating, categories, base_general["categories"], last, clip)
    ax_c.set_title(f"(c) Category change at stage {last}, points", loc="left")

    handles, labels = ax_a.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, tuple) else h for h in handles]
    handles.append(Line2D([], [], color=INK, linewidth=0.6, linestyle=(0, (1, 1.5))))
    labels.append("Base model")
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, columnspacing=1.2,
               handlelength=2.6, handletextpad=0.5, bbox_to_anchor=(0.5, -0.005))

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print(f"Wrote {FIGURE.relative_to(ROOT)}: {len(learners)} learners, "
          f"{len(weight_updating)} weight-updating rows, stage {last}, clip ±{clip} points.")


if __name__ == "__main__":
    main()
