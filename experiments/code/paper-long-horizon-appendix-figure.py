#!/usr/bin/env python3
"""Draw the appendix's per-stage view of Figure 3 (Figure 6) from experiments/outputs.

Everything shown is read from the registry in experiments/outputs/schema.json, the G-test
definition in protocol.json, and each method's results.json and method.json, in the same way as
experiments/code/paper-long-horizon-figure.py draws Figure 3. No number is written here.

    (a) One panel per G-test category: the full-panel score at stages 0, 4, 8, 16 and 25 for every
        weight-updating method (summary.depths[<stage>].general.categories), starting from the
        base model, with the base score as a dotted reference.
    (b) HERD's drift audit per stage (tables.drift): the audited stage-end checkpoint drift D^C and
        base drift D^0 (filled) and their highest within-stage measurement (hollow), on a log axis,
        against the budgets (dashed).

Series styles are shared with the Figure 3 script by method id.

    python3 experiments/code/paper-long-horizon-appendix-figure.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/outputs"
FIGURE = ROOT / "output/paper/figures/long-horizon-appendix.pdf"

spec = importlib.util.spec_from_file_location("long_horizon", Path(__file__).with_name("paper-long-horizon-figure.py"))
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
INK, GUIDE, ORANGE, PURPLE = shared.INK, shared.GUIDE, shared.ORANGE, shared.PURPLE

DRIFT = [  # (series id of the stage-end value, series id of the peak, label, colour)
    ("checkpointDriftFinal", "checkpointDriftPeak", r"$D^C$", PURPLE),
    ("generalDriftFinal", "generalDriftPeak", r"$D^0$", ORANGE),
]


def table_series(table: dict, series_id: str) -> list[tuple[int, float]]:
    rows = {r["id"]: int(r["x"]) for r in table["rows"]}
    return sorted((rows[v["row"]], v["value"]) for v in table["values"] if v["series"] == series_id)


def main() -> None:
    rows = shared.methods()
    protocol = shared.read(OUT / "protocol.json")
    categories = protocol["general"]["categories"]
    base = next(r for r in rows if r["id"] == "base")
    base_categories = base["results"]["summary"]["final"]["general"]["categories"]
    hero = next(r for r in rows if r["family"] == "ours")
    weight_updating = [r for r in rows if r["family"] in shared.WEIGHT_UPDATING]
    anchors = sorted(int(k) for k in hero["results"]["summary"]["depths"])

    fig = plt.figure(figsize=(7.05, 2.5))
    outer = fig.add_gridspec(1, 2, width_ratios=[2.55, 1.0], wspace=0.22,
                             left=0.065, right=0.995, top=0.84, bottom=0.30)
    inner = outer[0].subgridspec(1, len(categories), wspace=0.12)
    axes = [fig.add_subplot(inner[0, k]) for k in range(len(categories))]
    for ax, category in zip(axes, categories):
        cid = category["id"]
        for m in weight_updating:
            depths = m["results"]["summary"]["depths"]
            stages = [0, *(s for s in anchors if str(s) in depths)]
            values = [base_categories[cid], *(depths[str(s)]["general"]["categories"][cid] for s in stages[1:])]
            s = shared.style(m)
            ax.plot(stages, values, label=m["short"], **s)
        ax.axhline(base_categories[cid], color=INK, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
        ax.set_title(category["source"], fontsize=7.5, pad=2)
        ax.set_xticks([0, 8, 16, 25])
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.grid(axis="y", color="#e4e4e4", linewidth=0.4)
        ax.set_axisbelow(True)
        if ax is not axes[0]:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("G-test score")
    axes[len(axes) // 2].set_xlabel("Stage")
    fig.text(0.065, 0.93, "(a) General capability by category, weight-updating methods",
             fontsize=8, fontweight="bold", ha="left")

    bx = fig.add_subplot(outer[1])
    drift = hero["results"]["tables"]["drift"]
    budgets = {"checkpointDriftFinal": drift["checkpointBudget"], "generalDriftFinal": drift["generalBudget"]}
    handles = []
    for final_id, peak_id, label, colour in DRIFT:
        final = table_series(drift, final_id)
        peak = table_series(drift, peak_id)
        bx.plot([s for s, _ in final], [v for _, v in final], color=colour, marker="s", markersize=3.2,
                linewidth=1.0, zorder=4)
        bx.plot([s for s, _ in peak], [v for _, v in peak], color=colour, marker="o", markersize=3.2,
                markerfacecolor="white", linestyle="none", zorder=3)
        handles.append((Line2D([], [], color=colour, marker="s", markersize=3.2, linewidth=1.0), f"{label} end of stage"))
        handles.append((Line2D([], [], color=colour, marker="o", markersize=3.2, markerfacecolor="white",
                               linestyle="none"), f"{label} peak"))
    budget = budgets["checkpointDriftFinal"]
    bx.axhline(budget, color=INK, linewidth=0.6, linestyle="--", zorder=2)
    handles.append((Line2D([], [], color=INK, linewidth=0.6, linestyle="--"),
                    rf"budget $\varepsilon_C=\varepsilon_0={budget:g}$"))
    bx.set_yscale("log")
    bx.set_ylim(0.005, 0.6)
    bx.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    bx.set_yticklabels(["0.005", "0.010", "0.020", "0.050", "0.100", "0.200", "0.500"])
    bx.set_xticks([1, 5, 10, 15, 20, 25])
    bx.set_xlabel("Stage")
    bx.set_ylabel("Drift, nats/token (log)")
    bx.grid(axis="y", color="#e4e4e4", linewidth=0.4)
    bx.set_axisbelow(True)
    bx.legend([h for h, _ in handles], [l for _, l in handles], loc="upper right", frameon=False,
              fontsize=6.3, handlelength=1.6, borderaxespad=0.2, labelspacing=0.25)
    bx.set_title("(b) HERD drift audit per stage", loc="left", fontsize=8, fontweight="bold")

    handles_a, labels_a = axes[0].get_legend_handles_labels()
    fig.legend(handles_a, labels_a, loc="lower center", ncol=len(labels_a), frameon=False, columnspacing=1.2,
               handlelength=2.6, handletextpad=0.5, bbox_to_anchor=(0.5, -0.01))

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print(f"Wrote {FIGURE.relative_to(ROOT)}: {len(weight_updating)} weight-updating methods, "
          f"{len(categories)} categories, HERD drift over {len(drift['rows'])} stages.")


if __name__ == "__main__":
    main()
