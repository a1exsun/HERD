#!/usr/bin/env python3
"""Draw the appendix's stage-wise mechanism diagnostics of HERD from experiments/outputs.

Every panel is read from hero/results.json (tables.harness, tables.residual, tables.multipliers,
tables.distillation, tables.drift) and, for the two drift panels, from the ablation variants'
tables.drift under the registry in schema.json. No number is written here.

    (a) Harness size in tokens and skills in the registry per stage, against the registry budget.
    (b) Residual memory: records and tasks represented per stage; tokens against the task budget.
    (c) Share of updates with an active multiplier per stage, checkpoint and base.
    (d) Distillation KL at the first update and over the last quarter of updates per stage.
    (e) Audited stage-end checkpoint drift D^C for full HERD and the variants that change it.
    (f) Audited stage-end base drift D^0 for full HERD and the variants that change it.

    python3 experiments/code/paper-mechanism-figure.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/outputs"
FIGURE = ROOT / "output/paper/figures/mechanism-diagnostics.pdf"

spec = importlib.util.spec_from_file_location("long_horizon", Path(__file__).with_name("paper-long-horizon-figure.py"))
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)
INK, MUTED, RED, BLUE, ORANGE, GREEN, PURPLE = (shared.INK, shared.MUTED, shared.RED, shared.BLUE, shared.ORANGE,
                                                shared.GREEN, shared.PURPLE)

# Variants drawn in the drift panels, by registry id: (label, style).
CHECKPOINT_VARIANTS = [
    ("no-checkpoint-anchor", r"$-$ ckpt anchor", dict(color=BLUE, marker="s", linestyle="--")),
    ("no-anchors", r"$-$ both anchors", dict(color=MUTED, marker="^", linestyle="-.")),
    ("drift-budget-tight", r"$\varepsilon_C=0.02$", dict(color=GREEN, marker="v", linestyle=":")),
    ("drift-budget-loose", r"$\varepsilon_C=0.1$", dict(color=ORANGE, marker="D", linestyle=":")),
]
BASE_VARIANTS = [
    ("no-base-anchor", r"$-$ base anchor", dict(color=BLUE, marker="s", linestyle="--")),
    ("no-anchors", r"$-$ both anchors", dict(color=MUTED, marker="^", linestyle="-.")),
]
FULL = dict(color=RED, marker="o", linestyle="-", linewidth=1.5, zorder=10)


def series(table: dict, series_id: str) -> tuple[list[int], list[float]]:
    rows = {r["id"]: int(r["x"]) for r in table["rows"]}
    points = sorted((rows[v["row"]], v["value"]) for v in table["values"] if v["series"] == series_id)
    return [s for s, _ in points], [v for _, v in points]


def registry_results(entry_id: str) -> dict | None:
    for entry in shared.read(OUT / "schema.json")["registry"]:
        if entry["id"] == entry_id and (OUT / entry["path"] / "results.json").is_file():
            return shared.read(OUT / entry["path"] / "results.json")
    return None


def finish(ax, title: str, ylabel: str, stages: list[int]) -> None:
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    ax.set_xticks([1, 5, 10, 15, 20, 25])
    ax.set_xlim(min(stages) - 0.5, max(stages) + 0.5)
    ax.grid(axis="y", color="#e4e4e4", linewidth=0.4)
    ax.set_axisbelow(True)


def main() -> None:
    hero = registry_results("hero")
    tables = hero["tables"]
    marker = dict(markersize=2.6, linewidth=0.9)

    fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.7))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.93, bottom=0.11, wspace=0.5, hspace=0.62)
    (a, b, c), (d, e, f) = axes

    # (a) harness
    harness = tables["harness"]
    stages, tokens = series(harness, "harnessTokens")
    _, skills = series(harness, "skills")
    a.plot(stages, tokens, color=ORANGE, marker="o", label="Harness tokens", **marker)
    a.axhline(harness["budget"], color=ORANGE, linewidth=0.6, linestyle="--")
    a.text(stages[-1], harness["budget"], rf"$B_K$ = {harness['budget']:,}", ha="right", va="bottom", fontsize=6.3,
           color=ORANGE)
    a.set_ylim(0, harness["budget"] * 1.18)
    a.set_yticks([0, 8000, 16000, 24000])
    a.set_yticklabels(["0", "8k", "16k", "24k"])
    a2 = a.twinx()
    a2.plot(stages, skills, color=INK, marker="s", linestyle=":", label="Skills (right)", **marker)
    a2.set_ylim(0, len(stages) * 1.18)
    a2.set_yticks([0, 10, 20])
    a2.spines["top"].set_visible(False)
    finish(a, "(a) Harness size", "Tokens", stages)
    handles = a.get_legend_handles_labels()[0] + a2.get_legend_handles_labels()[0]
    a.legend(handles, [h.get_label() for h in handles], loc="upper left", frameon=False, fontsize=6.3,
             handlelength=1.8)

    # (b) residual memory
    residual = tables["residual"]
    stages, records = series(residual, "residualRecords")
    _, tasks = series(residual, "residualTasks")
    _, mem_tokens = series(residual, "residualTokens")
    b.plot(stages, records, color=PURPLE, marker="o", label="Records", **marker)
    b.set_ylim(0, max(records) * 1.35)
    b2 = b.twinx()
    b2.plot(stages, tasks, color=INK, marker="s", linestyle=":", label="Tasks represented (right)", **marker)
    b2.set_ylim(0, len(stages) * 1.35 * 0.75)
    b2.set_yticks([0, 8, 16])
    b2.spines["top"].set_visible(False)
    full_from = next((s for s, t in zip(stages, mem_tokens) if t >= 0.99 * residual["budget"]), None)
    if full_from is not None:
        b.axvspan(full_from - 0.5, stages[-1] + 0.5, color="#f2f2f2", zorder=0)
        b.text(full_from, max(records) * 1.28, f"memory at budget from stage {full_from}", fontsize=6.0,
               color=MUTED, ha="left", va="top")
    finish(b, "(b) Residual memory", "Records", stages)
    handles = b.get_legend_handles_labels()[0] + b2.get_legend_handles_labels()[0]
    b.legend(handles, [h.get_label() for h in handles], loc="lower right", frameon=False, fontsize=6.3,
             handlelength=1.8)

    # (c) multipliers
    multipliers = tables["multipliers"]
    stages, ck = series(multipliers, "checkpointMultiplierActive")
    _, ba = series(multipliers, "generalMultiplierActive")
    width = 0.4
    c.bar([s - width / 2 for s in stages], ck, width, color=PURPLE, label=r"$\lambda_C>0$")
    c.bar([s + width / 2 for s in stages], ba, width, color=ORANGE, label=r"$\lambda_0>0$")
    c.set_ylim(0, 1.0)
    c.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    finish(c, "(c) Updates with an active multiplier", "Share of updates", stages)
    c.legend(loc="upper right", frameon=False, fontsize=6.3, handlelength=1.2)

    # (d) distillation KL
    distillation = tables["distillation"]
    stages, first = series(distillation, "distillKLFirst")
    _, last = series(distillation, "distillKLLastQuarter")
    d.plot(stages, first, color=MUTED, marker="o", markerfacecolor="white", label="First update", **marker)
    d.plot(stages, last, color=RED, marker="o", label="Last quarter of updates", **marker)
    d.set_yscale("log")
    d.set_ylim(0.001, 5)
    d.set_yticks([0.001, 0.01, 0.1, 1])
    d.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    finish(d, "(d) Distillation KL", "Nats/token (log)", stages)
    d.legend(loc="lower right", frameon=False, fontsize=6.3, handlelength=1.8)

    # (e), (f) drift across variants
    drift = tables["drift"]
    for ax, series_id, budget_key, variants, title, ylabel in [
        (e, "checkpointDriftFinal", "checkpointBudget", CHECKPOINT_VARIANTS, r"(e) Stage-end $D^C$ by variant",
         "Nats/token (log)"),
        (f, "generalDriftFinal", "generalBudget", BASE_VARIANTS, r"(f) Stage-end $D^0$ by variant",
         "Nats/token (log)"),
    ]:
        stages, values = series(drift, series_id)
        ax.plot(stages, values, label="HERD (full)", markersize=2.8, **FULL)
        for vid, label, style in variants:
            results = registry_results(vid)
            if results is None:
                continue
            vs, vv = series(results["tables"]["drift"], series_id)
            ax.plot(vs, vv, label=label, markersize=2.6, linewidth=0.9, **style)
        ax.axhline(drift[budget_key], color=INK, linewidth=0.6, linestyle="--")
        ax.set_yscale("log")
        ax.set_ylim(0.0015, 0.4)
        ax.set_yticks([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])
        ax.set_yticklabels(["0.002", "0.005", "0.01", "0.02", "0.05", "0.1", "0.2"])
        finish(ax, title, ylabel, stages)
        ax.legend(loc="lower center", frameon=False, fontsize=5.9, handlelength=1.6, ncol=2, labelspacing=0.2,
                  columnspacing=0.8, borderaxespad=0.1)
    for ax in (d, e, f):
        ax.set_xlabel("Stage")

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    print(f"Wrote {FIGURE.relative_to(ROOT)}: six panels over {len(stages)} stages.")


if __name__ == "__main__":
    main()
