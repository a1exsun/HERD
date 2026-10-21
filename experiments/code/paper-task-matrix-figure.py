#!/usr/bin/env python3
"""Draw the appendix's per-task figure from experiments/outputs.

Everything shown is read from the registry in experiments/outputs/schema.json, the stream in
protocol.json, and each method's results.json and method.json. No number is written here.

    (a) The score on each task at the stage it is learned (series.current), for HERD, the
        plasticity floor Seq-LoRA, the strongest non-parametric method by AA, and the base model.
    (b) The change on each task between the stage it was learned and the last stage, in points,
        for every weight-updating method (taskCells: R[T, j] - R[j, j]); the outlined cell is each
        method's largest loss.

Series styles are shared with the Figure 3 script by method id.

    python3 experiments/code/paper-task-matrix-figure.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/outputs"
FIGURE = ROOT / "output/paper/figures/task-matrix.pdf"
CLIP = 20  # points

spec = importlib.util.spec_from_file_location("long_horizon", Path(__file__).with_name("paper-long-horizon-figure.py"))
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
INK, RED, BLUE = shared.INK, shared.RED, shared.BLUE


def retention(method: dict, tasks: list[dict], last: int) -> list[float | None]:
    cells = {(c["task"], c["stage"]): c["score"] for c in method["results"]["taskCells"]}
    out = []
    for task in tasks:
        learned, final = cells.get((task["id"], task["stage"])), cells.get((task["id"], last))
        out.append(None if learned is None or final is None else 100 * (final - learned))
    return out


def main() -> None:
    rows = shared.methods()
    protocol = shared.read(OUT / "protocol.json")
    tasks = protocol["stream"]["tasks"]
    last = len(tasks)
    by_id = {r["id"]: r for r in rows}
    hero = next(r for r in rows if r["family"] == "ours")
    context = max((r for r in rows if r["family"] == "non-parametric"),
                  key=lambda r: r["results"]["summary"]["depths"][str(last)]["seenMean"])
    lines = [hero, by_id["seq-lora"], context, by_id["base"]]
    weight_updating = [r for r in rows if r["family"] in shared.WEIGHT_UPDATING]

    fig = plt.figure(figsize=(7.05, 3.55))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.45], hspace=0.62, left=0.095, right=0.995, top=0.93, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    stages = [t["stage"] for t in tasks]
    for m in lines:
        points = {p["stage"]: p["value"] for p in m["results"]["series"]["current"]}
        s = shared.style(m)
        if m["id"] == "base":
            s = dict(color=INK, marker="", linestyle=(0, (1, 1.5)), linewidth=0.8, zorder=1)
        ax.plot(stages, [points[t] for t in stages], label=m["short"] if m["id"] != "base" else "Base model", **s)
    ax.set_xlim(0.5, last + 0.5)
    ax.set_xticks(stages)
    ax.set_xticklabels([t["label"].replace("SuperNI ", "") for t in tasks], rotation=60, ha="right", fontsize=6.2)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Score when learned")
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_title("(a) Score on each task at the stage it is learned", loc="left")
    ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=6.8, handlelength=2.2, columnspacing=1.0,
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0.0)

    bx = fig.add_subplot(gs[1])
    cmap = LinearSegmentedColormap.from_list("paper-diverging", [RED, "#ffffff", BLUE])
    norm = Normalize(-CLIP, CLIP)
    for i, m in enumerate(weight_updating):
        deltas = retention(m, tasks, last)
        worst = min((j for j, d in enumerate(deltas) if d is not None), key=lambda j: deltas[j])
        for j, d in enumerate(deltas):
            if d is None:
                bx.add_patch(Rectangle((j, i), 1, 1, facecolor="#f2f2f2", edgecolor="white", linewidth=0.6))
                continue
            bx.add_patch(Rectangle((j, i), 1, 1, facecolor=cmap(norm(d)), edgecolor="white", linewidth=0.6))
            dark = abs(d) > CLIP * 0.55
            text = f"{d:+.0f}" if round(d) != 0 else "0"
            bx.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=5.9,
                    color="white" if dark else INK, fontweight="bold" if m["family"] == "ours" else "normal")
        bx.add_patch(Rectangle((worst, i), 1, 1, facecolor="none", edgecolor=INK, linewidth=0.9, zorder=5))
    bx.set_xlim(0, last)
    bx.set_ylim(len(weight_updating), 0)
    bx.set_xticks([j + 0.5 for j in range(last)])
    bx.set_xticklabels([t["label"].replace("SuperNI ", "") for t in tasks], rotation=60, ha="right", fontsize=6.2)
    bx.set_yticks([i + 0.5 for i in range(len(weight_updating))])
    bx.set_yticklabels([m["short"] for m in weight_updating], fontsize=6.8)
    for label, m in zip(bx.get_yticklabels(), weight_updating):
        if m["family"] == "ours":
            label.set_fontweight("bold")
    bx.tick_params(length=0)
    for side in ("top", "right", "bottom", "left"):
        bx.spines[side].set_visible(False)
    bx.set_title(f"(b) Change from the stage a task is learned to stage {last}, points", loc="left")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print(f"Wrote {FIGURE.relative_to(ROOT)}: {len(lines)} acquisition lines, "
          f"{len(weight_updating)} retention rows over {last} tasks, clip ±{CLIP} points.")


if __name__ == "__main__":
    main()
