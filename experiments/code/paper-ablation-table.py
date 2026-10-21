#!/usr/bin/env python3
"""Write the paper's ablation table (Table 2) from experiments/outputs.

Rows come from the registry in experiments/outputs/schema.json: HERD first, then every entry of
family `ablation`, core tier before optional tier, in registry order. Columns are the five
quantities of Q1 (Acq., AA, F, ΔC, ΔC_min) at stage 25, each ablation cell carrying its difference
from full HERD, plus the stage-25 drift audit D^C and D^0 against the budgets ε_C and ε_0.
Bold marks, per column, the ablation that hurts most; a drift value over its budget is also bold.
A last row, read from the registry id in EXTRA_ROWS, re-scores the deployed weights with the
stage-25 harness attached (inference only); it is outside the bold marks and carries its own note.

A variant whose results.json does not exist yet still gets its row of gray "--" cells and a visible
note. A row whose results lack `summarySd` at stage 25 is marked single-seed.

    python3 experiments/code/paper-ablation-table.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/outputs"
TABLE = ROOT / "output/paper/tables/ablations.tex"
DEPTH = "25"
TIERS = ["core", "optional"]
GRAY = "gray!70"

# (header, lower-is-better, value getter, sd getter, signed)
COLUMNS = [
    (r"Acq.$\uparrow$", False, lambda d: d["acquisition"], lambda s: s.get("acquisition"), False),
    (r"AA$\uparrow$", False, lambda d: d["seenMean"], lambda s: s.get("seenMean"), False),
    (r"F$\downarrow$", True, lambda d: d["forgetting"], lambda s: s.get("forgetting"), False),
    (r"$\Delta C\uparrow$", False, lambda d: d["general"]["change"], lambda s: s.get("general", {}).get("change"), True),
    (r"$\Delta C_{\min}\uparrow$", False, lambda d: d["general"]["worstCategoryChange"],
     lambda s: s.get("general", {}).get("worstCategoryChange"), True),
]
DRIFT = [(r"$D^C$", "checkpointDriftFinal", "checkpoint_budget"), (r"$D^0$", "generalDriftFinal", "general_budget")]

# Row labels when a variant has no method.json yet; method.json `short` overrides.
LABELS = {
    "no-harness": r"$-$ skills",
    "uniform-residual": "uniform memory",
    "no-checkpoint-anchor": r"$-$ checkpoint anchor",
    "no-base-anchor": r"$-$ base anchor",
    "no-anchors": r"$-$ both anchors",
    "sft": "SFT objective",
}
PARAMETERS = {"training.checkpoint_budget": r"$\varepsilon_C$", "training.general_budget": r"$\varepsilon_0$"}
# Rows outside the ablation family, appended after the tiers: (registry id, label, note).
EXTRA_ROWS = [("hero-with-harness", "+ harness (infer.)",
               r"\textsuperscript{\ddag}\,inference only: the deployed weights of the full run re-scored with the "
               r"stage-25 harness attached to every prompt.")]

def read(path: Path):
    return json.loads(path.read_text())


def number(value: float, signed: bool, digits: int = 3) -> str:
    text = f"{abs(value):.{digits}f}"
    if signed and value > 0:
        return f"$+${text}"
    if value < 0:
        return f"$-${text}"
    return text


def gray(text: str) -> str:
    return rf"\textcolor{{{GRAY}}}{{{text}}}"


def final_drift(results: dict) -> dict[str, float | None]:
    table = (results.get("tables") or {}).get("drift")
    if not table:
        return {}
    last = max((row["id"] for row in table["rows"]), key=lambda r: float(r))
    return {v["series"]: v["value"] for v in table["values"] if v["row"] == last}


def rows() -> tuple[dict, list[dict]]:
    registry = read(OUT / "schema.json")["registry"]
    hero_entry = next(e for e in registry if e["id"] == "hero")
    hero = read(OUT / hero_entry["path"] / "results.json")
    budgets = {k: hero["configuration"]["training"].get(k, 0.05) for _, _, k in DRIFT}
    full = {
        "id": "hero", "label": "HERD (full)", "tier": None, "placeholder": False,
        "depth": hero["summary"]["depths"][DEPTH],
        "sd": ((hero.get("summarySd") or {}).get("depths") or {}).get(DEPTH),
        "drift": final_drift(hero), "budgets": budgets,
    }
    variants = []
    for tier in TIERS:
        for entry in registry:
            if entry.get("family") != "ablation" or entry.get("tier", "core") != tier:
                continue
            directory = OUT / entry["path"]
            label = LABELS.get(entry["id"])
            ablation = entry.get("ablation", {})
            if ablation.get("kind") == "sweep":
                name = PARAMETERS.get(ablation.get("parameter"), ablation.get("parameter"))
                label = f"{name} $=$ {ablation.get('value')}"
            if (directory / "method.json").is_file():
                label = read(directory / "method.json").get("short", label)
            row = {"id": entry["id"], "label": label or entry["id"], "tier": tier, "status": entry.get("status"),
                   "placeholder": False, "budgets": budgets}
            if (directory / "results.json").is_file():
                results = read(directory / "results.json")
                depth = (results.get("summary") or {}).get("depths", {}).get(DEPTH)
                if depth:
                    row["depth"] = depth
                    row["sd"] = ((results.get("summarySd") or {}).get("depths") or {}).get(DEPTH)
                    row["drift"] = final_drift(results)
            variants.append(row)
    return full, variants


def extra_rows(full: dict) -> list[dict]:
    registry = read(OUT / "schema.json")["registry"]
    rows = []
    for rid, label, note in EXTRA_ROWS:
        entry = next((e for e in registry if e["id"] == rid), None)
        path = OUT / entry["path"] / "results.json" if entry else None
        if not (path and path.is_file()):
            continue
        results = read(path)
        depth = (results.get("summary") or {}).get("depths", {}).get(DEPTH)
        if not depth:
            continue
        rows.append({"id": rid, "label": label, "note": note, "tier": "extra", "placeholder": False,
                     "budgets": full["budgets"], "depth": depth, "sd": None, "drift": final_drift(results)})
    return rows


def main() -> None:
    full, variants = rows()
    scored = [r for r in variants if "depth" in r and not r["placeholder"]]
    worst = []
    for _, lower, get, _, _ in COLUMNS:
        deltas = [get(r["depth"]) - get(full["depth"]) for r in scored]
        worst.append((max if lower else min)(deltas) if deltas else None)

    header = ["Variant", *(c[0] for c in COLUMNS), *(d[0] for d in DRIFT)]
    lines = [
        "% Generated by experiments/code/paper-ablation-table.py from experiments/outputs; do not edit by hand.",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "c" * len(COLUMNS) + "cc}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    def cells(row: dict, reference: bool) -> list[str]:
        out = []
        for (_, _, get, spread, signed), top in zip(COLUMNS, worst):
            value = get(row["depth"])
            text = number(value, signed)
            sd = spread(row["sd"]) if row.get("sd") else None
            if sd is not None:
                text += rf"{{\scriptsize$\pm${sd:.3f}}}"
            if not reference:
                delta = value - get(full["depth"])
                bold = not row["placeholder"] and top is not None and abs(delta - top) < 5e-7
                text += rf"\,{{\scriptsize({number(delta, True)})}}"
                if bold:
                    text = rf"\textbf{{{text}}}"
            out.append(gray(text) if row["placeholder"] else text)
        for _, key, budget in DRIFT:
            value = (row.get("drift") or {}).get(key)
            if value is None:
                out.append(gray("--"))
                continue
            text = number(value, False)
            if value > row["budgets"][budget] + 5e-7:
                text = rf"\textbf{{{text}}}"
            out.append(gray(text) if row["placeholder"] else text)
        return out

    lines.append(" & ".join([rf"\textbf{{{full['label']}}}", *cells(full, True)]) + r" \\")
    previous = None
    pending = single = False
    for row in variants:
        if previous is not None and row["tier"] != previous:
            lines.append(r"\midrule")
        if previous is None:
            lines.append(r"\midrule")
        previous = row["tier"]
        label = row["label"]
        if "depth" not in row:
            pending = True
            lines.append(" & ".join([gray(label + r"\textsuperscript{\dag}"), *([gray("--")] * (len(COLUMNS) + len(DRIFT)))]) + r" \\")
            continue
        if row["placeholder"]:
            pending = True
            label = gray(label + r"\textsuperscript{\dag}")
        elif not row.get("sd"):
            single = True
            label += r"\textsuperscript{*}"
        lines.append(" & ".join([label, *cells(row, False)]) + r" \\")
    extras = extra_rows(full)
    if extras:
        lines.append(r"\midrule")
        for row in extras:
            lines.append(" & ".join([row["label"] + r"\textsuperscript{\ddag}", *cells(row, False)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    notes = [row["note"] for row in extras]
    if pending:
        notes.append(r"\textsuperscript{\dag}\,run pending: gray cells are placeholders, not results.")
    if single:
        notes.append(r"\textsuperscript{*}\,single seed.")
    if notes:
        lines.append(r"\par\smallskip{\scriptsize " + " ".join(notes) + "}")
    lines.append("")

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text("\n".join(lines))
    done = sum(1 for r in variants if "depth" in r and not r["placeholder"])
    print(f"wrote {TABLE.relative_to(ROOT)} ({done}/{len(variants)} variants with results, stage {DEPTH})")


if __name__ == "__main__":
    main()
