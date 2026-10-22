"""Export run outputs to experiments/outputs, from which the paper's tables and figures are drawn.

    uv run --project packages/evaluator/python python experiments/code/export-paper-experiments.py --runs runs.json

`runs.json` maps method ids of `registry.json` to their runs under experiments/runs (see `runs.example.json`):

    {"base": {"seeds": [["base-tasks", "base-general"]], "hardware": "..."},
     "seq-lora": {"seeds": ["seq-lora-s42", "seq-lora-s43", "seq-lora-s44"], "hardware": "..."},
     "hero": {"seeds": ["hero-s42", "hero-s43", "hero-s44"], "hardware": "..."},
     "hero-with-harness": {"seeds": ["hero-s42", "hero-s43", "hero-s44"],
                           "predictions": "test-predictions.harness.jsonl"}}

A seed is one directory, or a list of directories whose predictions are pooled (the base model's task and G-test
runs). Layouts, by method kind:

    constant (base, static harness)   <run>/items/*.json or <run>/predictions/<group>.jsonl, scored as stage 0
    stream (the other baselines)      <run>/output/stage-NN/predictions/<group>.jsonl and training.json
    HERD and its variants             <run>/bundle/plan.json and <run>/output/stage-NN/{stage.json,
                                      training-log.jsonl, test-predictions.jsonl}

Every cell is rescored per item with the shared scorer against the test sets under experiments/data, each checked
against its tracked manifest. A method's numbers are the mean over its seeds, with the sample standard deviation
(`summarySd`, per-cell `sd`); intervals are 95% item bootstraps (4,000 draws, items resampled within task and G-test
category), and HERD's comparisons pair the same resample between HERD and each baseline. The base model is stage 0
of every method's general-capability changes and is required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import math
import random
import statistics
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).with_name("registry.json")
sys.path.insert(0, str(ROOT / "packages/evaluator/python/src"))
from hero_evaluation.scoring import VERSION, score_predictions  # noqa: E402

ANCHORS, DRAWS, SEED, LEVEL = (4, 8, 16, 25), 4000, 42, 0.95
RETENTION_STAGES, RETENTION_TASKS = (12, 20), 8
CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning", "instruction-following",
              "multilingual-understanding")
CATEGORY_LABELS = {"general-knowledge": "General knowledge", "mathematical-reasoning": "Mathematical reasoning",
                   "commonsense-reasoning": "Commonsense reasoning", "instruction-following": "Instruction following",
                   "multilingual-understanding": "Multilingual understanding"}
CATEGORY_SOURCES = {"mmlu": "MMLU", "gsm8k": "GSM8K", "hellaswag": "HellaSwag", "ifeval": "IFEval", "xcopa": "XCOPA"}
METRIC_LABELS = {"strict_exact_match": "Exact match", "exact_match": "Exact match", "rouge_l": "ROUGE-L",
                 "edit_similarity": "Edit similarity", "sari": "SARI", "leading_choice_accuracy": "Accuracy",
                 "final_number_match": "Final-number match", "ifeval_strict": "Strict verification"}
FAMILIES = [
    {"id": "control", "label": "Controls", "description": "Constant predictors: no learning across the stream."},
    {"id": "non-parametric", "label": "Non-parametric", "description": "The model is frozen; an external context or harness evolves."},
    {"id": "parametric", "label": "Parametric", "description": "The model's parameters are updated at every stage."},
    {"id": "hybrid", "label": "Hybrid", "description": "A harness is distilled into the parameters."},
    {"id": "ours", "label": "This work", "description": "HERD."}]


def read(path: Path):
    return json.loads(path.read_text())


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def number(value: float) -> float:
    return round(value, 6)


def spread(values: list[float]) -> dict:
    """The mean and sample standard deviation; individual seed values are not exported."""
    return {"value": number(mean(values)), "sd": number(statistics.stdev(values)) if len(values) > 1 else 0.0}


def settings(value):
    """A plan section as published: the settings, without the hashes that pin internal files."""
    if isinstance(value, dict):
        return {key: settings(item) for key, item in value.items() if not key.lower().endswith("sha256")}
    return [settings(item) for item in value] if isinstance(value, list) else value


def interval(samples: list[list[float]], generator: random.Random) -> list[float]:
    """Percentile interval of the mean over groups, resampling items within each group."""
    draws = sorted(mean(mean(generator.choices(group, k=len(group))) for group in samples) for _ in range(DRAWS))
    tail = (1 - LEVEL) / 2
    return [number(draws[int(tail * DRAWS)]), number(draws[int((1 - tail) * DRAWS) - 1])]


def table(row_list: list[dict], series: list[dict], values: list[dict], **extra) -> dict:
    return {"rows": row_list, "series": series, "values": values, **extra}


def stage_rows(stages) -> list[dict]:
    return [{"id": str(s), "x": s, "label": f"Stage {s}"} for s in stages]


# ---------------------------------------------------------------------------
# protocol


class Protocol:
    """The test sets, G-test and their metrics, verified against the tracked manifests."""

    def __init__(self, repository: Path) -> None:
        data = repository / "experiments/data"
        manifest = read(data / "trace-plus-17/manifest.json")
        tracked = {t["id"]: t["files"]["testGold"]["sha256"]
                   for t in read(repository / "datasets/trace-plus-17/manifest.json")["tasks"]}
        self.manifest, self.tasks = manifest, [t["id"] for t in manifest["tasks"]]
        self.gold, self.panel, self.group_metric = {}, {}, {}
        for task in manifest["tasks"]:
            items = verified_rows(data / "trace-plus-17" / task["files"]["testGold"]["path"], tracked[task["id"]])
            self.gold[task["id"]] = {r["id"]: r for r in items}
            self.panel[task["id"]] = [r["id"] for r in items]
            self.group_metric[task["id"]] = task["metric"]
        report = read(repository / "datasets/general-test-v1.json")
        general_file = next(f for f in report["files"] if f["path"] == "hidden-scorer/general-test.jsonl")
        general = verified_rows(data / "general-test/v1" / general_file["path"], general_file["sha256"])
        self.general = {r["id"]: r for r in general}
        self.probe = set(read(data / "general-test/v1/hidden-scorer/probe-ids.json"))
        self.categories = {c["category"]: c for c in report["categories"]}
        for category in CATEGORIES:
            group = "general:" + category
            self.panel[group] = [r["id"] for r in general if r["category"] == category]
            self.group_metric[group] = self.categories[category]["metric"]
        self.base_items: dict[str, dict[str, float]] = {}

    def reference(self, group: str, item: str) -> dict:
        return self.general[item] if group.startswith("general:") else self.gold[group][item]


def verified_rows(path: Path, expected: str) -> list[dict]:
    if not path.is_file():
        sys.exit(f"Missing test set {path}; rebuild it with `pnpm dataset:build`.")
    if file_hash(path) != expected:
        sys.exit(f"{path} differs from its tracked manifest.")
    return rows(path)


# ---------------------------------------------------------------------------
# one seed


def kind_of(entry: dict) -> str:
    if entry["family"] == "control":
        return "constant"
    return "hero" if entry["path"] == "hero" or entry["path"].startswith("ablations/") else "stream"


def group_of(path: Path) -> str:
    return path.stem.replace("__", ":", 1)


def predictions_by_stage(directories: list[Path], kind: str, name: str) -> dict[int, dict[str, list[dict]]]:
    out: dict[int, dict[str, list[dict]]] = {}
    for directory in directories:
        if kind == "constant":
            groups = out.setdefault(0, {})
            for path in sorted((directory / "items").glob("*.json")):
                row = read(path)
                groups.setdefault(row["group"], []).append(row)
            for path in sorted((directory / "predictions").glob("*.jsonl")):
                groups.setdefault(group_of(path), []).extend(rows(path))
            continue
        for stage_dir in sorted((directory / "output").glob("stage-*")):
            stage = int(stage_dir.name.split("-")[1])
            if kind == "hero":
                if (stage_dir / name).is_file():
                    for row in rows(stage_dir / name):
                        out.setdefault(stage, {}).setdefault(row["group"], []).append(row)
            else:
                for path in sorted((stage_dir / "predictions").glob("*.jsonl")):
                    out.setdefault(stage, {})[group_of(path)] = rows(path)
    return out


def score_cells(predictions: dict[int, dict[str, list[dict]]], protocol: Protocol, label: str):
    """Per-item scores of every evaluated (stage, group), and the count of outputs stopped by the length ceiling."""
    items, stops = {}, 0
    for stage, groups in sorted(predictions.items()):
        for group, found in groups.items():
            ids = [r["id"] for r in found]
            panel = protocol.panel[group]
            if group.startswith("general:") and ids != panel:
                panel = [i for i in panel if i in protocol.probe]  # the 200-item probe
            if ids != panel:
                raise ValueError(f"{label}: stage {stage} {group} does not cover its test panel in order")
            scored = score_predictions(protocol.group_metric[group], found, [protocol.reference(group, i) for i in ids])
            items[(stage, group)] = {i["id"]: i["score"] for i in scored["itemScores"]}
            stops += sum(r.get("stopReason", r.get("finishReason")) == "length" for r in found)
    return items, stops


def general_cells(items: dict[tuple[int, str], dict[str, float]], protocol: Protocol) -> list[dict]:
    """Category scores of every evaluated panel; a full panel also yields its nested probe."""
    cells = []
    for (stage, group), scores in sorted(items.items(), key=lambda entry: entry[0][0]):
        if not group.startswith("general:"):
            continue
        category = group.split(":", 1)[1]
        probe = [score for i, score in scores.items() if i in protocol.probe]
        if len(scores) > len(probe):
            cells.append({"stage": stage, "category": category, "panel": "full", "items": len(scores),
                          "score": number(mean(scores.values()))})
        cells.append({"stage": stage, "category": category, "panel": "probe", "items": len(probe),
                      "score": number(mean(probe))})
    return cells


def summarize(items: dict[tuple[int, str], dict[str, float]], general: list[dict], protocol: Protocol,
              constant: bool) -> tuple[dict, dict]:
    """Stream metrics at every complete row and the per-stage series, from per-item scores. Forgetting and signed
    BWT compare each diagonal cell with the row's cell on the items the two share; general-capability changes are
    against the base model on the same panel."""
    at = (lambda stage: 0) if constant else (lambda stage: stage)
    cell = lambda stage, task: items.get((at(stage), task))  # noqa: E731
    base = {panel: {category: mean(score for i, score in protocol.base_items[f"general:{category}"].items()
                                   if panel == "full" or i in protocol.probe) for category in CATEGORIES}
            for panel in ("full", "probe")}
    panels: dict[tuple[int, str], dict[str, float]] = {}
    for c in general:
        panels.setdefault((c["stage"], c["panel"]), {})[c["category"]] = c["score"]
    panels = {key: scores for key, scores in panels.items() if len(scores) == len(CATEGORIES)}

    def general_summary(stage: int) -> dict | None:
        for panel in ("full", "probe"):
            scores = panels.get((at(stage), panel))
            if scores:
                changes = {category: scores[category] - base[panel][category] for category in CATEGORIES}
                return {"panel": panel, "macro": number(mean(scores.values())),
                        "change": number(mean(changes.values())), "worstCategoryChange": number(min(changes.values())),
                        "categories": {category: number(score) for category, score in scores.items()}}
        return None

    evaluated = sorted({stage for stage, group in items if not group.startswith("general:")})
    rows_at = list(range(1, len(protocol.tasks) + 1)) if constant else evaluated
    diagonal = {j + 1: cell(j + 1, task) for j, task in enumerate(protocol.tasks) if cell(j + 1, task)}
    summary: dict = {"depths": {}}
    series: dict = {"current": [{"stage": stage, "task": protocol.tasks[stage - 1], "value": number(mean(scores.values()))}
                                for stage, scores in diagonal.items()],
                    "seenMean": [], "firstEight": [], "generalProbe": [], "generalFull": []}
    for stage in rows_at:
        seen = protocol.tasks[:stage]
        if all(cell(stage, task) for task in seen) and all(j in diagonal for j in range(1, stage + 1)):
            final = {task: cell(stage, task) for task in seen}
            shared = [[i for i in diagonal[j + 1] if i in final[task]] for j, task in enumerate(seen[:-1])]
            paired = [(mean(diagonal[j + 1][i] for i in ids), mean(final[task][i] for i in ids))
                      for (j, task), ids in zip(enumerate(seen[:-1]), shared)]
            depth = {"seenMean": number(mean(mean(final[task].values()) for task in seen)),
                     "acquisition": number(mean(mean(diagonal[j].values()) for j in range(1, stage + 1))),
                     "items": min(len(final[task]) for task in seen)}
            if paired:
                depth["forgetting"] = number(mean(max(0.0, before - after) for before, after in paired))
                depth["signedBWT"] = number(mean(after - before for before, after in paired))
            if general_summary(stage):
                depth["general"] = general_summary(stage)
            if stage in ANCHORS:
                summary["depths"][str(stage)] = depth
                series["seenMean"].append({"stage": stage, "value": depth["seenMean"]})
        if stage >= RETENTION_TASKS and all(cell(stage, task) for task in protocol.tasks[:RETENTION_TASKS]):
            series["firstEight"].append({"stage": stage, "value": number(mean(
                mean(cell(stage, task).values()) for task in protocol.tasks[:RETENTION_TASKS]))})
    for (stage, panel), scores in sorted(panels.items()):
        series["generalProbe" if panel == "probe" else "generalFull"].append(
            {"stage": stage, "value": number(mean(scores.values())),
             "change": number(mean(scores[c] - base[panel][c] for c in CATEGORIES))})
    if diagonal:
        summary["acquisition"] = {"value": number(mean(mean(scores.values()) for scores in diagonal.values())),
                                  "stages": len(diagonal)}
    if evaluated or constant:
        last = len(protocol.tasks) if constant else evaluated[-1]
        summary["latest"] = {"stage": last}
        if general_summary(last):
            summary["latest"]["general"] = general_summary(last)
        if last in diagonal:
            summary["latest"]["current"] = number(mean(diagonal[last].values()))
    summary["final"] = summary["depths"].get(str(len(protocol.tasks)))
    return summary, series


def intervals(items: dict[tuple[int, str], dict[str, float]], protocol: Protocol, constant: bool) -> dict:
    """Item-bootstrap intervals for the seen-task mean at each complete anchor row, its paired difference from the
    base model, and the mean diagonal."""
    generator = random.Random(SEED)
    out: dict = {"level": LEVEL, "draws": DRAWS, "unit": "test items resampled within task", "seenMean": {}, "vsBase": {}}
    for anchor in ANCHORS:
        seen = protocol.tasks[:anchor]
        stage = 0 if constant else anchor
        if not all((stage, task) in items for task in seen):
            continue
        out["seenMean"][str(anchor)] = interval([list(items[(stage, t)].values()) for t in seen], generator)
        out["vsBase"][str(anchor)] = interval(
            [[score - protocol.base_items[t][i] for i, score in items[(stage, t)].items()] for t in seen], generator)
    diagonal = [list(items[(0 if constant else j + 1, t)].values()) for j, t in enumerate(protocol.tasks)
                if (0 if constant else j + 1, t) in items]
    if diagonal:
        out["acquisition"] = interval(diagonal, generator)
    return out


def seed_record(entry: dict, seed, config: dict, protocol: Protocol, runs: Path) -> dict:
    kind = kind_of(entry)
    directories = [runs / name for name in ([seed] if isinstance(seed, str) else seed)]
    label = f"{entry['id']} {seed}"
    items, stops = score_cells(predictions_by_stage(directories, kind, config.get("predictions", "test-predictions.jsonl")),
                               protocol, label)
    if not items:
        raise ValueError(f"{label}: no test predictions found")
    record = {"items": items, "lengthStops": stops, "predictions": sum(len(v) for v in items.values()),
              "plan": None, "training": {}, "stages": [], "logs": {}}
    run = directories[0]
    if (run / "bundle/plan.json").is_file():
        record["plan"] = read(run / "bundle/plan.json")
    for stage_dir in sorted((run / "output").glob("stage-*")) if kind != "constant" else []:
        stage = int(stage_dir.name.split("-")[1])
        if kind == "hero" and (stage_dir / "stage.json").is_file():
            record["stages"].append(read(stage_dir / "stage.json"))
            record["logs"][stage] = rows(stage_dir / "training-log.jsonl")
            record["training"][stage] = record["stages"][-1]["training"]
        elif (stage_dir / "training.json").is_file():
            record["training"][stage] = read(stage_dir / "training.json")
    return record


# ---------------------------------------------------------------------------
# a method over its seeds


def seed_mean(trees: list):
    """The mean over seeds of matching JSON trees; values that agree, such as panels and item counts, are kept."""
    first = trees[0]
    if isinstance(first, dict):
        return {key: seed_mean([t[key] for t in trees]) for key in first if all(isinstance(t, dict) and key in t for t in trees)}
    if isinstance(first, list):
        if all(isinstance(e, dict) and "stage" in e for t in trees for e in t):
            common = [e["stage"] for e in first if all(any(x["stage"] == e["stage"] for x in t) for t in trees)]
            matched = [[next(x for x in t if x["stage"] == stage) for t in trees] for stage in common]
            return [{**seed_mean(group), **({"sd": spread([g["value"] for g in group])["sd"]} if "value" in group[0] else {})}
                    for group in matched]
        return first if all(t == first for t in trees) else None
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return first if all(t == first for t in trees) else number(mean(trees))
    return first if all(t == first for t in trees) else None


def seed_sd(trees: list):
    """Sample standard deviations of metrics, including zero; stage and item counts are metadata."""
    first = trees[0]
    if isinstance(first, dict):
        out = {key: seed_sd([t[key] for t in trees]) for key in first
               if key not in ("stage", "stages", "items") and all(isinstance(t, dict) and key in t for t in trees)}
        return {key: value for key, value in out.items() if value not in (None, {})} or None
    if isinstance(first, (int, float)) and not isinstance(first, bool):
        return number(statistics.stdev(trees))
    return None


def mean_items(records: list[dict]) -> dict[tuple[int, str], dict[str, float]]:
    """Seed-mean item scores of the cells every seed evaluated."""
    keys = set.intersection(*(set(r["items"]) for r in records))
    return {key: {i: mean(r["items"][key][i] for r in records) for i in records[0]["items"][key]} for key in keys}


def cell_mean(cells: list[list[dict]], key) -> list[dict]:
    grouped: dict = {}
    for seed_cells in cells:
        for c in seed_cells:
            grouped.setdefault(key(c), []).append(c)
    return [{**{k: v for k, v in group[0].items() if k != "score"}, "score": number(mean(c["score"] for c in group)),
             "sd": spread([c["score"] for c in group])["sd"]} for group in grouped.values() if len(group) == len(cells)]


def adapter_state(directory: Path) -> dict | None:
    """Rank, trainable parameters and size of a saved adapter, read from its safetensors header."""
    weights = directory / "adapter_model.safetensors"
    if not weights.is_file():
        return None
    with weights.open("rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    configuration = read(directory / "adapter_config.json")
    return {"rank": configuration["r"], "alpha": configuration["lora_alpha"], "tensors": len(tensors),
            "parameters": sum(math.prod(t["shape"]) for t in tensors), "bytes": weights.stat().st_size}


def method_result(entry: dict, config: dict, records: list[dict], protocol: Protocol) -> dict:
    kind = kind_of(entry)
    constant = kind == "constant"
    per_seed = []
    for record in records:
        items = record["items"]
        tasks = [{"stage": stage, "task": group, "metric": protocol.group_metric[group], "items": len(scores),
                  "score": number(mean(scores.values()))}
                 for (stage, group), scores in sorted(items.items()) if not group.startswith("general:")]
        general = [{**c, "metric": protocol.group_metric["general:" + c["category"]]} for c in general_cells(items, protocol)]
        summary, series = summarize(items, general, protocol, constant)
        per_seed.append({"tasks": tasks, "general": general, "summary": summary, "series": series})
    items = mean_items(records)
    evaluated = len({stage for stage, _ in items})
    trained = min(len(r["training"]) for r in records)
    result = {"schema": "hero-experiments/results-v3", "method": entry["id"], "constant": constant,
              "stages": {"planned": len(protocol.tasks), "trained": trained,
                         "evaluated": len(protocol.tasks) if constant else evaluated},
              "schedule": "constant" if constant else "standard",
              "summary": seed_mean([s["summary"] for s in per_seed]),
              "series": seed_mean([s["series"] for s in per_seed]),
              "intervals": intervals(items, protocol, constant),
              "taskCells": cell_mean([s["tasks"] for s in per_seed], lambda c: (c["stage"], c["task"])),
              "generalCells": cell_mean([s["general"] for s in per_seed], lambda c: (c["stage"], c["category"], c["panel"])),
              "outputs": {"predictions": records[0]["predictions"], "lengthStops": spread([r["lengthStops"] for r in records])},
              "seeds": len(records), "tables": {}}
    if len(records) > 1:
        result["summarySd"] = seed_sd([s["summary"] for s in per_seed])
    plan = records[0]["plan"]
    if plan:
        result["configuration"] = settings({key: value for key, value in plan.items()
                                            if key not in ("files", "tasks", "stages", "seed", "schemaVersion")})
        if "training" in result["configuration"] and isinstance(result["configuration"]["training"], dict):
            result["configuration"]["training"].pop("seed", None)
    else:
        result["configuration"] = {}
    result["configuration"].setdefault("evaluation", {})["scorerVersion"] = VERSION
    stages = sorted(set.intersection(*(set(r["training"]) for r in records))) if trained else []
    result["accounting"] = [{"stage": s, "task": protocol.tasks[s - 1],
                             "training": {"seconds": round(mean(r["training"][s]["seconds"] for r in records))}}
                            for s in stages]
    result["cost"] = {"hardware": config.get("hardware", "")}
    if stages:
        result["cost"]["trainingSeconds"] = spread([sum(r["training"][s]["seconds"] for s in stages) for r in records])
    last = [r["training"][stages[-1]] for r in records] if stages else []
    if last and "bestPlaybookTokens" in last[0]:
        result["state"] = {"kind": "playbook", "stage": stages[-1], "tokens": round(mean(t["bestPlaybookTokens"] for t in last))}
    elif last and "harnessQwenTokens" in last[0]:
        result["state"] = {"kind": "harness", "stage": stages[-1], "tokens": round(mean(t["harnessQwenTokens"] for t in last))}
    elif kind == "hero":
        result["state"] = {"kind": "adapter and harness", "stage": stages[-1],
                           "adapter": adapter_state(Path(records[0]["output"]) / f"stage-{stages[-1]:02d}/adapter"),
                           "harnessTokens": spread([r["stages"][-1]["harnessTokens"] for r in records]),
                           "skills": spread([len(json.loads(r["stages"][-1]["harness"])["skills"]) for r in records])}
        result["tables"] = hero_tables(records)
    result["tables"]["generalByCategory"] = general_by_category(cell_mean(
        [s["general"] for s in per_seed], lambda c: (c["stage"], c["category"], c["panel"])), protocol)
    result["items"] = items  # dropped before writing; HERD's comparisons use it
    return result


def general_by_category(general: list[dict], protocol: Protocol) -> dict:
    probe = [c for c in general if c["panel"] == "probe"]
    reference = general_cells({(0, group): scores for group, scores in protocol.base_items.items()
                               if group.startswith("general:")}, protocol)
    return table(stage_rows(sorted({c["stage"] for c in probe})),
                 [{"id": category, "label": CATEGORY_LABELS[category]} for category in CATEGORIES],
                 [{"row": str(c["stage"]), "series": c["category"], "value": c["score"], "sd": c["sd"], "items": c["items"]}
                  for c in probe], unit="score",
                 reference={c["category"]: c["score"] for c in reference if c["panel"] == "probe"})


# ---------------------------------------------------------------------------
# HERD tables


def hero_measures(record: dict) -> dict[int, dict[str, float | None]]:
    """What every stage of one seed did: the harness, the guard, the residual memory, drift and distillation."""
    out = {}
    for s in record["stages"]:
        log, guard, drift = record["logs"][s["stage"]], s["guard"], s["drift"]
        residual = guard["windowResidual"] if guard else None
        out[s["stage"]] = {
            "harnessTokens": s["harnessTokens"], "skills": len(json.loads(s["harness"])["skills"]),
            "committed": float(guard["committed"]) if guard else None,
            "windowResidualPrevious": residual["previous"] if residual else None,
            "windowResidualCandidate": residual["candidate"] if residual else None,
            "guardGain": residual["previous"] - residual["candidate"] if residual else None,
            "checkpointDriftInitial": drift["initial"]["checkpoint"], "checkpointDriftFinal": drift["final"]["checkpoint"],
            # Stage 1 has no earlier task to probe, so it has no checkpoint drift.
            "checkpointDriftPeak": max(r["checkpointDrift"] for r in log) if drift["initial"]["checkpoint"] is not None else None,
            "generalDriftInitial": drift["initial"]["general"], "generalDriftFinal": drift["final"]["general"],
            "generalDriftPeak": max(r["generalDrift"] for r in log),
            "checkpointMultiplierActive": mean(r["checkpointMultiplier"] > 0 for r in log),
            "generalMultiplierActive": mean(r["generalMultiplier"] > 0 for r in log),
            "distillKLFirst": s["training"]["lossFirst"], "distillKLLastQuarter": s["training"]["lossLastQuarter"],
            "gradNorm": mean(r["gradNorm"] for r in log),
            "residualRecords": len(s["residual"]), "residualTokens": sum(r["tokens"] for r in s["residual"]),
            "residualTasks": len(s["residualTasks"]),
            "trainingRecords": s["trainingRecords"], "excludedOverLength": s["excludedOverLength"],
            "trainingSeconds": s["training"]["seconds"], "stageSeconds": s["seconds"],
            "testSeconds": (s.get("test") or {}).get("seconds", 0.0), "peakMemoryGB": s["peakMemoryGB"]}
    return out


def hero_tables(records: list[dict]) -> dict:
    plan = records[0]["plan"]
    measures = [hero_measures(record) for record in records]
    stages = sorted(set.intersection(*(set(m) for m in measures)))

    def per_stage(series: list[tuple[str, str]], unit: str, **extra) -> dict:
        return table(stage_rows(stages), [{"id": key, "label": label} for key, label in series],
                     [{"row": str(s), "series": key, **spread([m[s][key] for m in measures])}
                      for s in stages for key, _ in series if all(m[s][key] is not None for m in measures)],
                     unit=unit, **extra)

    harness, training = plan["harness"], plan["training"]
    tables = {
        "harness": per_stage([("harnessTokens", "Harness size (tokens)"), ("skills", "Skills in the registry")], "tokens",
                             budget=harness["harnessBudgetTokens"], skillBudget=harness["skillBudgetTokens"]),
        "guard": per_stage([("windowResidualPrevious", "Window residual, previous registry"),
                            ("windowResidualCandidate", "Window residual, candidate registry"),
                            ("guardGain", "Residual reduction by the candidate"),
                            ("committed", "Seeds that commit the candidate")], "nats per token"),
        "drift": per_stage([("checkpointDriftInitial", "Checkpoint drift at stage start"),
                            ("checkpointDriftPeak", "Checkpoint drift, highest update"),
                            ("checkpointDriftFinal", "Checkpoint drift at stage end"),
                            ("generalDriftInitial", "Base drift at stage start"),
                            ("generalDriftPeak", "Base drift, highest update"),
                            ("generalDriftFinal", "Base drift at stage end")], "nats per token",
                           checkpointBudget=training["checkpoint_budget"], generalBudget=training["general_budget"]),
        "multipliers": per_stage([("checkpointMultiplierActive", "Updates with an active checkpoint multiplier"),
                                  ("generalMultiplierActive", "Updates with an active base multiplier")], "share"),
        "distillation": per_stage([("distillKLFirst", "Training loss, first update"),
                                   ("distillKLLastQuarter", "Training loss, last quarter of updates"),
                                   ("gradNorm", "Mean gradient norm before clipping")], "nats per token",
                                  objective=training.get("objective", "distill")),
        "residual": per_stage([("residualRecords", "Records retained"), ("residualTokens", "Tokens retained"),
                               ("residualTasks", "Tasks represented")], "count",
                              budget=harness["residualBudgetTokens"] - harness["generalReserveTokens"]),
        "training": per_stage([("trainingRecords", "Training records"), ("excludedOverLength", "Records over the length limit"),
                               ("trainingSeconds", "Training seconds"), ("stageSeconds", "Stage seconds"),
                               ("testSeconds", "Test decoding seconds"), ("peakMemoryGB", "Peak GPU memory (GB)")], "mixed"),
    }
    updates = sorted({(s, r["update"]) for s in stages for r in records[0]["logs"][s]})
    logged = {"loss": "Training loss", "gradNorm": "Gradient norm before clipping",
              "checkpointDrift": "Checkpoint drift", "generalDrift": "Base drift",
              "checkpointMultiplier": "Checkpoint multiplier", "generalMultiplier": "Base multiplier",
              "targetTokens": "Target tokens in the batch"}
    per_update = {name: {key: [] for key in updates} for name in logged}
    for record in records:
        for s in stages:
            for r in record["logs"][s]:
                for name in per_update:
                    if (s, r["update"]) in per_update[name]:
                        per_update[name][(s, r["update"])].append(r[name])
    count = max(u for _, u in updates)
    tables["updates"] = table(
        [{"id": f"{s}.{u}", "x": number(s - 1 + u / count), "stage": s, "update": u} for s, u in updates],
        [{"id": name, "label": label} for name, label in logged.items()],
        [{"row": f"{s}.{u}", "series": name, **spread(per_update[name][(s, u)])}
         for s, u in updates for name in per_update if len(per_update[name][(s, u)]) == len(records)],
        unit="nats per token", updatesPerStage=count, learningRate=training["learning_rate"],
        schedule="linear decay to zero within each stage",
        checkpointBudget=training["checkpoint_budget"], generalBudget=training["general_budget"])
    return tables


def comparisons(hero: dict[tuple[int, str], dict[str, float]], baselines: dict[str, tuple[dict, bool]],
                protocol: Protocol) -> dict:
    """HERD minus each baseline after the last stage, with paired item-bootstrap intervals: every draw resamples the
    items of each task and G-test category once and applies the same resample to both methods (seed-mean items)."""
    import numpy as np

    last = len(protocol.tasks)
    categories = [f"general:{c}" for c in CATEGORIES]

    def arrays(items: dict, constant: bool) -> dict:
        at = lambda stage, group: np.array([items[(0 if constant else stage, group)][i]  # noqa: E731
                                            for i in protocol.panel[group]])
        return {"final": [at(last, t) for t in protocol.tasks],
                "learned": [at(j + 1, t) for j, t in enumerate(protocol.tasks)],
                "general": [at(last, g) for g in categories]}

    def metrics(a: dict, task_draws: list, general_draws: list) -> dict:
        final = np.stack([scores[draw].mean(-1) for scores, draw in zip(a["final"], task_draws)])
        learned = np.stack([scores[draw].mean(-1) for scores, draw in zip(a["learned"], task_draws)])
        general = np.stack([scores[draw].mean(-1) for scores, draw in zip(a["general"], general_draws)])
        return {"seenMean": final.mean(0), "acquisition": learned.mean(0),
                "forgetting": np.maximum(0.0, learned[:-1] - final[:-1]).mean(0),
                "signedBWT": (final[:-1] - learned[:-1]).mean(0), "generalMacro": general.mean(0)}

    generator = np.random.default_rng(SEED)
    sizes = [len(protocol.panel[t]) for t in protocol.tasks] + [len(protocol.panel[g]) for g in categories]
    draws = [generator.integers(0, n, size=(DRAWS, n)) for n in sizes]
    identity = [np.arange(n) for n in sizes]
    ours = arrays(hero, False)
    point_ours = metrics(ours, identity[:last], identity[last:])
    drawn_ours = metrics(ours, draws[:last], draws[last:])
    tail = (1 - LEVEL) / 2
    out = []
    for method, (items, constant) in baselines.items():
        theirs = arrays(items, constant)
        point = metrics(theirs, identity[:last], identity[last:])
        drawn = metrics(theirs, draws[:last], draws[last:])
        entry = {"id": method}
        for name in point_ours:
            difference = drawn_ours[name] - drawn[name]
            entry[name] = {"hero": number(float(point_ours[name])), "method": number(float(point[name])),
                           "difference": number(float(point_ours[name] - point[name])),
                           "interval": [number(float(np.quantile(difference, tail))),
                                        number(float(np.quantile(difference, 1 - tail)))]}
        out.append(entry)
    return {"stage": last, "level": LEVEL, "draws": DRAWS, "unit": "test items resampled within task and category, "
            "paired between HERD and the method", "heroItems": "seed-mean item scores",
            "generalPanel": "full G-test", "methods": out}


# ---------------------------------------------------------------------------
# outputs


def protocol_file(protocol: Protocol) -> dict:
    tasks = [{"stage": t["position"], "id": t["id"], "source": t["source"],
              "label": t["sourceTaskId"] if t["source"] == "trace" else f"SuperNI {t['sourceTaskId']}",
              "category": t["category"], "metric": t["metric"], "metricLabel": METRIC_LABELS[t["metric"]],
              "items": t["realized"]["test"]} for t in protocol.manifest["tasks"]]
    general = protocol.categories
    return {"schema": "hero-experiments/protocol-v1",
            "model": {"id": "Qwen/Qwen3.5-4B", "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "thinking": False},
            "stream": {"name": protocol.manifest["name"], "seed": protocol.manifest["seed"], "tasks": tasks},
            "general": {"name": "G-test", "categories": [
                {"id": c, "label": CATEGORY_LABELS[c], "source": CATEGORY_SOURCES.get(general[c]["source"], general[c]["source"]),
                 "metric": general[c]["metric"], "metricLabel": METRIC_LABELS.get(general[c]["metric"], general[c]["metric"]),
                 "items": general[c]["testCount"], "probe": general[c]["probeCount"]} for c in CATEGORIES]},
            "schedule": {"anchors": list(ANCHORS), "retention": {"stages": list(RETENTION_STAGES), "tasks": RETENTION_TASKS},
                         "itemsPerCell": 100, "maxNewTokens": 5120, "maxInputTokens": 16384,
                         "intervals": {"level": LEVEL, "draws": DRAWS}},
            "families": FAMILIES}


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"  {path}")


def export(runs_config: dict, repository: Path, runs: Path, outputs: Path) -> None:
    registry = read(REGISTRY)
    unknown = set(runs_config) - {e["id"] for e in registry}
    if unknown:
        raise SystemExit(f"Not in registry.json: {sorted(unknown)}")
    if "base" not in runs_config:
        raise SystemExit("runs.json needs the base model: it is stage 0 of every general-capability change")
    protocol = Protocol(repository)
    write(outputs / "protocol.json", protocol_file(protocol))
    order = sorted((e for e in registry if e["id"] in runs_config),
                   key=lambda e: (e["id"] != "base", kind_of(e) == "hero"))
    results: dict[str, dict] = {}
    for entry in order:
        config = runs_config[entry["id"]]
        print(entry["id"])
        records = [seed_record(entry, seed, config, protocol, runs) for seed in config["seeds"]]
        for record, seed in zip(records, config["seeds"]):
            record["output"] = str(runs / (seed if isinstance(seed, str) else seed[0]) / "output")
        if entry["id"] == "base":
            protocol.base_items = {group: scores for (_, group), scores in mean_items(records).items()}
        results[entry["id"]] = method_result(entry, config, records, protocol)
    if "hero" in results:
        baselines = {e["id"]: (results[e["id"]]["items"], results[e["id"]]["constant"]) for e in order
                     if e["id"] in results and kind_of(e) != "hero"}
        results["hero"]["comparisons"] = comparisons(results["hero"]["items"], baselines, protocol)
    schema = {"registry": []}
    for entry in registry:
        listed = {key: entry[key] for key in ("id", "family", "path", "tier", "ablation", "reference") if key in entry}
        if entry["id"] in results:
            result = results[entry["id"]]
            stages = result["stages"]
            listed["status"] = "complete" if stages["evaluated"] == stages["planned"] else "stopped"
            listed["stages"] = stages
            write(outputs / entry["path"] / "results.json", {k: v for k, v in result.items() if k != "items"})
        else:
            listed["status"] = "scheduled"
        write(outputs / entry["path"] / "method.json", {"id": entry["id"], "short": entry["short"], "family": entry["family"],
                                                        **({"ablation": entry["ablation"]} if "ablation" in entry else {})})
        schema["registry"].append(listed)
    write(outputs / "schema.json", schema)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True, help="JSON map of registry ids to their seeds' runs")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "experiments/runs")
    parser.add_argument("--outputs", type=Path, default=ROOT / "experiments/outputs")
    parser.add_argument("--repository", type=Path, default=ROOT, help="where experiments/data and datasets/ live")
    args = parser.parse_args()
    export(read(args.runs), args.repository.resolve(), args.runs_root.resolve(), args.outputs.resolve())


def run_with_deep_stack(target) -> None:
    """The TRACE rouge package computes LCS recursively; outputs at the 5,120-token ceiling need a deep stack."""
    import threading

    sys.setrecursionlimit(200_000)
    threading.stack_size(1 << 30)
    errors: list[BaseException] = []

    def guarded() -> None:
        try:
            target()
        except BaseException as error:  # noqa: BLE001 - re-raised in the main thread
            errors.append(error)

    worker = threading.Thread(target=guarded)
    worker.start()
    worker.join()
    if errors:
        raise errors[0]


if __name__ == "__main__":
    run_with_deep_stack(main)
