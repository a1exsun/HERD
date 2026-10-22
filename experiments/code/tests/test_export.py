"""End-to-end check of the results pipeline on a synthetic stream: runs -> export -> every paper table and figure.

    uv run --project packages/evaluator/python --with matplotlib --with pytest pytest experiments/code/tests
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export", CODE / "export-paper-experiments.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

TASKS = 25
ITEMS = 4
CATEGORIES = export.CATEGORIES


def jsonl(path: Path, rows) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def repository(root: Path) -> tuple[list[str], dict[str, list[dict]], list[dict]]:
    tasks, gold, entries = [], {}, []
    for position in range(1, TASKS + 1):
        task_id = f"trace:T{position:02d}"
        rows = [{"id": f"{task_id}-{k}", "prompt": f"q{k}", "answer": str(k), "references": [str(k)]} for k in range(ITEMS)]
        path = f"{position:02d}/test.jsonl"
        digest = jsonl(root / "experiments/data/trace-plus-17" / path, rows)
        tasks.append(task_id)
        gold[task_id] = rows
        entries.append({"position": position, "id": task_id, "source": "trace", "sourceTaskId": f"T{position:02d}",
                        "category": "Synthetic", "metric": "strict_exact_match", "realized": {"test": ITEMS},
                        "files": {"testGold": {"path": path, "sha256": digest}}})
    dump(root / "experiments/data/trace-plus-17/manifest.json", {"name": "TRACE+17", "seed": 42, "tasks": entries})
    dump(root / "datasets/trace-plus-17/manifest.json", {"tasks": entries})
    general = [{"id": f"{c}-{k}", "prompt": f"{k}+1?", "category": c, "metric": "final_number_match",
                "answer": str(k + 1), "references": [str(k + 1)]} for c in CATEGORIES for k in range(ITEMS)]
    digest = jsonl(root / "experiments/data/general-test/v1/hidden-scorer/general-test.jsonl", general)
    dump(root / "experiments/data/general-test/v1/hidden-scorer/probe-ids.json",
         [r["id"] for r in general if int(r["id"].rsplit("-", 1)[1]) < 2])
    dump(root / "datasets/general-test-v1.json", {
        "files": [{"path": "hidden-scorer/general-test.jsonl", "sha256": digest}],
        "categories": [{"category": c, "source": "gsm8k", "metric": "final_number_match", "testCount": ITEMS,
                        "probeCount": 2} for c in CATEGORIES]})
    return tasks, gold, general


def answers(rows: list[dict], accuracy: float, rng: random.Random) -> list[dict]:
    return [{"id": r["id"], "prediction": r["answer"] if rng.random() < accuracy else "wrong", "stopReason": "eos"}
            for r in rows]


def hero_run(directory: Path, tasks, gold, general, rng: random.Random, harness_predictions: bool) -> None:
    dump(directory / "bundle/plan.json", {
        "seed": 42, "harness": {"harnessBudgetTokens": 24576, "skillBudgetTokens": 768, "residualBudgetTokens": 32768,
                                "generalReserveTokens": 8192},
        "training": {"checkpoint_budget": 0.05, "general_budget": 0.05, "learning_rate": 1e-4, "objective": "distill"},
        "lora": {"rank": 32, "alpha": 64, "dropout": 0.0}})
    probe = [r for r in general if int(r["id"].rsplit("-", 1)[1]) < 2]
    for stage in range(1, TASKS + 1):
        out = directory / "output" / f"stage-{stage:02d}"
        full = stage in (4, 8, 16, 25)
        seen = range(1, stage + 1) if full else range(1, 9) if stage in (12, 20) else []
        groups = sorted(set(seen) | {stage})
        rows = [{**p, "group": tasks[j - 1]} for j in groups for p in answers(gold[tasks[j - 1]], 0.8, rng)]
        rows += [{**p, "group": "general:" + r["category"], "panel": "full" if full else "probe"}
                 for r, p in zip(general if full else probe, answers(general if full else probe, 0.9, rng))]
        for name in ("test-predictions.jsonl",) + (("test-predictions.harness.jsonl",) if harness_predictions else ()):
            jsonl(out / name, rows)
        jsonl(out / "training-log.jsonl", [{"update": u, "loss": 0.5 / u, "checkpointDrift": 0.03, "generalDrift": 0.04,
                                            "checkpointMultiplier": 0.0, "generalMultiplier": 0.1 * (u == 1),
                                            "gradNorm": 1.0, "targetTokens": 64} for u in range(1, 5)])
        dump(out / "stage.json", {
            "stage": stage, "harness": json.dumps({"skills": [{"id": t} for t in tasks[:stage]]}), "harnessTokens": 100 * stage,
            "guard": {"windowResidual": {"previous": 1.0, "candidate": 0.8}, "committed": True},
            "residual": [{"id": "r", "tokens": 10, "taskId": tasks[0]}], "residualTasks": [tasks[0]],
            "trainingRecords": 500, "excludedOverLength": 0,
            "drift": {"initial": {"checkpoint": None if stage == 1 else 0.0, "general": 0.0},
                      "final": {"checkpoint": None if stage == 1 else 0.02, "general": 0.04}},
            "training": {"lossFirst": 0.5, "lossLastQuarter": 0.1, "seconds": 300.0}, "test": {"seconds": 10.0},
            "seconds": 400.0, "peakMemoryGB": 27.0})


def stream_run(directory: Path, training: dict, tasks, gold, general, rng: random.Random) -> None:
    for stage in range(1, TASKS + 1):
        out = directory / "output" / f"stage-{stage:02d}"
        dump(out / "training.json", training)
        full = stage in (4, 8, 16, 25)
        for j in sorted(set(range(1, stage + 1) if full else range(1, 9) if stage in (12, 20) else []) | {stage}):
            jsonl(out / "predictions" / f"{tasks[j - 1].replace(':', '__')}.jsonl", answers(gold[tasks[j - 1]], 0.7, rng))
        for c in CATEGORIES:
            panel = [r for r in general if r["category"] == c and (full or int(r["id"].rsplit("-", 1)[1]) < 2)]
            jsonl(out / "predictions" / f"general__{c}.jsonl", answers(panel, 0.6, rng))


def test_runs_to_paper_tables(tmp_path):
    rng = random.Random(0)
    tasks, gold, general = repository(tmp_path)
    runs = tmp_path / "experiments/runs"
    every_task = [p for t in tasks for p in answers(gold[t], 0.4, rng)]
    for k, row in enumerate(every_task + answers(general, 0.9, rng)):
        group = row["id"].rsplit("-", 1)[0]
        dump(runs / "base" / "items" / f"{k:04d}.json", {**row, "group": group if group.startswith("trace:") else "general:" + group})
    for seed in ("s42", "s43"):
        for method, training in (("seq-lora", {"seconds": 100.0}), ("hcl-stable", {"seconds": 50.0, "harnessQwenTokens": 39})):
            stream_run(runs / f"{method}-{seed}", training, tasks, gold, general, rng)
        hero_run(runs / f"hero-{seed}", tasks, gold, general, rng, True)
    hero_run(runs / "no-harness", tasks, gold, general, rng, False)
    config = {"base": {"seeds": [["base"]], "hardware": "Apple M4 Max"},
              "hcl-stable": {"seeds": ["hcl-stable-s42", "hcl-stable-s43"], "hardware": "NVIDIA H200"},
              "seq-lora": {"seeds": ["seq-lora-s42", "seq-lora-s43"], "hardware": "NVIDIA H200"},
              "hero": {"seeds": ["hero-s42", "hero-s43"], "hardware": "NVIDIA A100"},
              "hero-with-harness": {"seeds": ["hero-s42", "hero-s43"], "predictions": "test-predictions.harness.jsonl"},
              "no-harness": {"seeds": ["no-harness"]}}
    outputs = tmp_path / "experiments/outputs"
    export.export(config, tmp_path, runs, outputs)

    hero = json.loads((outputs / "hero/results.json").read_text())
    final = hero["summary"]["depths"]["25"]
    assert set(final) >= {"seenMean", "acquisition", "forgetting", "signedBWT", "general"}
    assert set(final["general"]) >= {"change", "worstCategoryChange"} and "25" in hero["summarySd"]["depths"]
    assert [m["id"] for m in hero["comparisons"]["methods"]] == ["base", "hcl-stable", "seq-lora"]
    assert hero["tables"]["drift"]["values"] and hero["cost"]["trainingSeconds"]["value"] == 300.0 * TASKS
    base = json.loads((outputs / "baselines/base/results.json").read_text())
    assert base["summary"]["depths"]["25"]["general"]["change"] == 0.0
    registry = json.loads((outputs / "schema.json").read_text())["registry"]
    assert {e["id"]: e["status"] for e in registry}["ace"] == "scheduled"

    (tmp_path / "experiments/code").mkdir()
    for script in sorted(CODE.glob("paper-*.py")):  # the scripts read and write relative to their repository
        shutil.copy(script, tmp_path / "experiments/code")
    for script in sorted((tmp_path / "experiments/code").glob("paper-*.py")):
        run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True)
        assert run.returncode == 0, f"{script.name}: {run.stderr[-2000:]}"
    produced = {p.name for p in (tmp_path / "output/paper").rglob("*") if p.is_file()}
    assert {"main-results.tex", "comparisons.tex", "cost.tex", "ablations.tex"} <= produced
    assert {"long-horizon.pdf", "hero-ablation.pdf", "mechanism-diagnostics.pdf", "task-matrix.pdf"} <= produced
    assert "Seq-LoRA" in (tmp_path / "output/paper/tables/main-results.tex").read_text()
    assert "HCL-Stable & 50 & 50 & 1,250 & 39" in (tmp_path / "output/paper/tables/cost.tex").read_text()
