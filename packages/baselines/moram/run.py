"""MoRAM (ICML 2026) over the TRACE+17 stream: prepare a job bundle, run atomic GPU jobs, score locally.

prepare   Build a self-contained bundle: plan, stage training records, prompt-only evaluation inputs
          (no test gold), this directory's code (including the `upstream/` layer), CUDA locks and `job.sh`.
train     Stage N: load the stage N-1 atom bank and the stream's optimizer state, train the new atoms on
          the current task, consolidate them (upstream `increment_task`) and seal the bank.
evaluate  Stage N: greedy evaluation with the sealed stage N bank on the base control's sparse schedule;
          per-item resumable, optionally one group subset per job.
score     Score sealed predictions with this directory's evaluator and derive stream metrics.

Each GPU job runs exactly one (stage, phase[, subset]) unit and leaves sealed files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from moram_cl import (  # noqa: E402
    MoramConfig, TrainingProgress, atom_counts, increment_all, inject_moram, load_moram_state,
    load_trainable_atom_state, make_optimizer, moram_state, moram_targets, summary_dict, supervised_ids,
    train_stage, trainable_atom_state,
)
from protocol import (  # noqa: E402
    BASE_MODEL_ID, EVAL_MAX_INPUT_TOKENS, EVAL_MAX_NEW_TOKENS, STAGE_COUNT, evaluated_task_indices,
    general_panel_kind, load_manifest, read_jsonl,
)

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_TREE_SHA256 = "e2361c06ba172642a18b1472f4758449bcddf20072e371921642bfba28004dc5"
CHECKPOINT_FILES = ("config.json", "merges.txt", "model.safetensors-00001-of-00002.safetensors",
                    "model.safetensors-00002-of-00002.safetensors", "model.safetensors.index.json",
                    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                    "video_preprocessor_config.json", "vocab.json")
GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")
PLAN_VERSION = "moram-plan-v2"
CHECKPOINT_EVERY = 8  # updates between mid-stage checkpoints (current atoms, optimizer, RNG, progress)
CHECKPOINT_FORMAT = "moram-trainable-atoms-v1"
# Shared update budget of the parametric baselines: 32 updates of 32 records per stage.
UPDATES, BATCH_SIZE = 32, 32
BUNDLE_CODE = ("run.py", "moram_cl.py", "microbatch.py", "protocol.py", "upstream/__init__.py", "upstream/moram.py",
               "upstream/lora_layer.py", "upstream/LICENSE.txt", "bootstrap.sh",
               "requirements-cuda.lock.txt", "requirements-kernels.lock.txt")
TRAIN_FIELDS = ("id", "prompt", "answer", "references", "metadata")


# ---------------------------------------------------------------------------
# Hashing and IO


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        name = child.relative_to(path).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(file_hash(child)))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def group_file(group: str) -> str:
    return group.replace(":", "__") + ".jsonl"


def stage_dir(output: Path, position: int) -> Path:
    return output / f"stage-{position:02d}"


def scientific_plan_hash(plan: dict[str, Any]) -> str:
    """Plan hash without its file manifest: bundles with equal hashes run the same experiment."""
    science = {key: value for key, value in plan.items() if key != "files"}
    return hashlib.sha256(json.dumps(science, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verified_rows(root: Path, spec: dict[str, Any]) -> list[dict[str, Any]]:
    if file_hash(root / spec["path"]) != spec["sha256"]:
        raise ValueError(f"Dataset hash mismatch: {spec['path']}")
    return read_jsonl(root / spec["path"])


def manifest_file(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    return next(entry for entry in manifest["files"] if entry["path"] == path)


# ---------------------------------------------------------------------------
# prepare (local)


def build_plan(dataset: Path, general_test: Path, *, profile: str, stages: int, updates: int, batch_size: int,
               config: MoramConfig, seed: int = 42, train_limit: int | None, eval_limit: int | None,
               general_limit: int | None) -> tuple[dict[str, Any], dict[str, list], dict[str, list]]:
    manifest = load_manifest(dataset)
    tasks = manifest["tasks"]
    eval_files: dict[str, list] = {}
    for task in tasks:
        rows = verified_rows(dataset, task["files"]["testGold"])
        eval_files[task["id"]] = [{"id": r["id"], "prompt": r["prompt"]} for r in rows]
    test_manifest = json.loads((general_test / "manifest.json").read_text())
    general_rows = verified_rows(general_test, manifest_file(test_manifest, "hidden-scorer/general-test.jsonl"))
    probe_ids = set(json.loads((general_test / "hidden-scorer/probe-ids.json").read_text()))
    for category in GENERAL_CATEGORIES:
        eval_files["general:" + category] = [{"id": r["id"], "prompt": r["prompt"]} for r in general_rows
                                             if r["category"] == category]
    train_files: dict[str, list] = {}
    plan_stages: list[dict[str, Any]] = []
    for position in range(1, stages + 1):
        task = tasks[position - 1]
        records = verified_rows(dataset, task["files"]["train"])
        records = records[:train_limit] if train_limit else records
        train_files[f"{position:02d}"] = [{key: r[key] for key in TRAIN_FIELDS if key in r} for r in records]
        evaluations = []
        kind = general_panel_kind(position)
        for index in evaluated_task_indices(position):
            ids = [r["id"] for r in eval_files[tasks[index]["id"]]]
            evaluations.append({"group": tasks[index]["id"], "ids": ids[:eval_limit] if eval_limit else ids})
        for category in GENERAL_CATEGORIES:
            ids = [r["id"] for r in eval_files["general:" + category] if kind == "full" or r["id"] in probe_ids]
            evaluations.append({"group": "general:" + category, "panel": kind,
                                "ids": ids[:general_limit] if general_limit else ids})
        plan_stages.append({"stage": position, "taskId": task["id"], "metric": task["metric"],
                            "trainRecords": len(records), "evaluations": evaluations})
    needed = {e["group"] for s in plan_stages for e in s["evaluations"]}
    eval_files = {group: rows for group, rows in eval_files.items() if group in needed}
    plan = {
        "schemaVersion": PLAN_VERSION, "method": "moram", "paper": "arXiv:2506.21035 (ICML 2026)",
        "upstream": {"repository": "https://github.com/Artificer-AI-Lab/MoRAM",
                     "commit": "aaa51d8ec67171e1bc5e1a1168351588561256c4"},
        "profile": profile, "model": BASE_MODEL_ID, "revision": REVISION, "baseTreeSha256": BASE_TREE_SHA256,
        "dataset": {"name": manifest["name"], "seed": manifest["seed"], "manifestSha256": file_hash(dataset / "manifest.json")},
        "generalTestManifestSha256": file_hash(general_test / "manifest.json"),
        "stages": plan_stages,
        "moram": asdict(config),
        "training": {"updates": updates, "batchSize": batch_size, "schedule": "constant, no warmup",
                     "optimizer": "Adam, one state carried across the whole stream",
                     "loss": "per-record token-mean cross-entropy, averaged over the batch",
                     "microBatchTokens": 24576,
                     "records": "D_t, seeded shuffle, reshuffled each epoch", "seed": seed,
                     "randomDraws": "initial atoms from seed*1000, re-initialized atoms after stage N from "
                                    "seed*1000+N, dropout from seed+N; restored on resume",
                     "maxSequenceTokens": EVAL_MAX_INPUT_TOKENS,
                     "targets": "text-backbone Linear modules (no vision, MTP or LM head)"},
        "evaluation": {"decoding": "greedy", "maxNewTokens": EVAL_MAX_NEW_TOKENS, "maxInputTokens": EVAL_MAX_INPUT_TOKENS,
                       "thinking": False, "eos": "union(generation_config, tokenizer)",
                       "adapter": "unmerged MoRAM layers in eval mode (self-activation gate active)",
                       "stageZero": "the base control's formal predictions"},
    }
    return plan, train_files, eval_files


def render_job(remote_root: str, account: str) -> str:
    """Generic MLeRP unit: one 40 GB GPU. Job name, walltime and the unit come from sbatch:
    sbatch --job-name=moram-s01-train --time=06:00:00 job.sh train --stage 1"""
    remote = remote_root.rstrip("/")
    return "\n".join([
        "#!/bin/bash",
        f"#SBATCH --account={account}",
        "#SBATCH --partition=BigCats",
        "#SBATCH --qos=lion",
        "#SBATCH --gres=gpu:40gb:1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=12",
        "#SBATCH --mem=96G",
        f"#SBATCH --output={remote}/logs/%x-%j.out",
        f"#SBATCH --error={remote}/logs/%x-%j.err",
        "set -euo pipefail",
        f'bash {remote}/bundle/bootstrap.sh {remote}/bundle {remote}/output "$@"',
        "",
    ])


def prepare(args) -> None:
    root = Path(args.repository).resolve()
    bundle = args.output.resolve()
    if bundle.exists():
        raise SystemExit(f"Bundle exists: {bundle}; use a new path")
    config = MoramConfig()
    plan, train_files, eval_files = build_plan(
        root / "experiments/data/trace-plus-17", root / "experiments/data/general-test/v1", profile=args.profile,
        stages=args.stages, updates=args.updates, batch_size=args.batch_size, config=config, seed=args.seed,
        train_limit=args.train_limit, eval_limit=args.eval_limit, general_limit=args.general_limit)
    for key, rows in train_files.items():
        write_jsonl(bundle / "inputs/train" / f"{key}.jsonl", rows)
    for group, rows in eval_files.items():
        write_jsonl(bundle / "inputs/eval" / group_file(group), rows)
    for name in BUNDLE_CODE:
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / name, bundle / name)
    if args.remote_root:
        (bundle / "job.sh").write_text(render_job(args.remote_root, args.account), encoding="utf-8")
    plan["files"] = {path.relative_to(bundle).as_posix(): file_hash(path)
                     for path in sorted(bundle.rglob("*")) if path.is_file()}
    atomic_json(bundle / "plan.json", plan)
    print(json.dumps({"bundle": str(bundle), "stages": len(plan["stages"]),
                      "planSha256": file_hash(bundle / "plan.json")}))


# ---------------------------------------------------------------------------
# GPU units


class Unit:
    """Verified bundle, pinned checkpoint, recorded runtime contract and the tokenizer for one GPU job."""

    def __init__(self, args):
        import torch
        from transformers import AutoTokenizer

        self.bundle, self.output = args.bundle.resolve(), args.output.resolve()
        self.plan = json.loads((self.bundle / "plan.json").read_text())
        if self.plan["schemaVersion"] != PLAN_VERSION:
            raise ValueError("Unsupported plan")
        for relative, expected in self.plan["files"].items():
            if file_hash(self.bundle / relative) != expected:
                raise ValueError(f"Bundle file changed: {relative}")
        if not 1 <= args.stage <= len(self.plan["stages"]):
            raise ValueError(f"Stage {args.stage} is outside the plan")
        self.stage = self.plan["stages"][args.stage - 1]
        self.dir = stage_dir(self.output, args.stage)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no implicit device fallback")
        self.device = args.device
        self.checkpoint = args.checkpoint.resolve()
        if args.fetch:
            from huggingface_hub import hf_hub_download

            self.checkpoint.mkdir(parents=True, exist_ok=True)
            for name in CHECKPOINT_FILES:
                if not (self.checkpoint / name).is_file():
                    hf_hub_download(BASE_MODEL_ID, name, revision=REVISION, local_dir=self.checkpoint)
            shutil.rmtree(self.checkpoint / ".cache", ignore_errors=True)
        if tree_hash(self.checkpoint) != self.plan["baseTreeSha256"]:
            raise ValueError("Base checkpoint differs from the pinned tree hash")
        contract = {"scientificPlanSha256": scientific_plan_hash(self.plan), "baseTreeSha256": self.plan["baseTreeSha256"],
                    "libraries": {name: __import__(name).__version__ for name in ("torch", "transformers")}}
        path = self.output / "contract.json"
        if path.is_file():
            if json.loads(path.read_text()) != contract:
                raise ValueError("Unit runs with a different scientific plan, checkpoint or library set")
        else:
            atomic_json(path, contract)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.config = MoramConfig(**self.plan["moram"])
        self.job = {"slurmJobId": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    def sealed_state(self, position: int) -> tuple[Path, Path]:
        directory = stage_dir(self.output, position)
        record = directory / "training.json"
        if not record.is_file():
            raise ValueError(f"Stage {position} is not trained")
        sealed = json.loads(record.read_text())
        if file_hash(directory / "moram.safetensors") != sealed["moramSha256"] or \
                file_hash(directory / "optimizer.pt") != sealed["optimizerSha256"]:
            raise ValueError(f"Stage {position} state hash mismatch")
        return directory / "moram.safetensors", directory / "optimizer.pt"

    def model(self, state: Path | None):
        """Base checkpoint with every target Linear replaced by the upstream MoRAM layer; `state` restores a bank."""
        import torch
        from safetensors.torch import load_file
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(self.checkpoint, dtype=torch.bfloat16,
                                                            local_files_only=True).to(self.device)
        configured = model.generation_config.eos_token_id
        self.eos = sorted(set(([configured] if isinstance(configured, int) else list(configured or []))
                              + [self.tokenizer.eos_token_id]))
        inject_moram(model, self.config, moram_targets(model))
        if state is not None:
            load_moram_state(model, load_file(str(state)))
        return model


def require_valid_resume(work: Path, stage: int, resumed: bool) -> None:
    """Preserve failed checkpoints; after stage three, never restart interrupted training."""
    if (work / "checkpoint/progress.json").is_file() and not resumed:
        raise RuntimeError("Training checkpoint failed verification; preserving it without retraining")
    if stage > 3 and not resumed and work.exists() and any(work.iterdir()):
        raise RuntimeError("After stage 3, interrupted training requires a verified checkpoint; no retraining")


def train_unit(args) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    unit = Unit(args)
    position, directory, training, config = args.stage, unit.dir, unit.plan["training"], unit.config
    if (directory / "training.json").is_file():
        unit.sealed_state(position)
        print(json.dumps({"stage": position, "phase": "train", "status": "already sealed"}), flush=True)
        return
    previous = unit.sealed_state(position - 1) if position > 1 else None
    base_state_sha = file_hash(previous[0]) if previous else None
    records, excluded = [], []
    for record in read_jsonl(unit.bundle / "inputs/train" / f"{position:02d}.jsonl"):
        try:
            supervised_ids(unit.tokenizer, record, max_length=training["maxSequenceTokens"])
        except ValueError:
            excluded.append(record["id"])
            continue
        records.append(record)
    records_sha = hashlib.sha256(json.dumps([r["id"] for r in records]).encode()).hexdigest()

    # Frozen banks come from the hash-verified preceding seal; only current atoms need checkpoint copies.
    work = directory / "training.tmp"
    checkpoint_dir = work / "checkpoint"
    resume = None
    if (checkpoint_dir / "progress.json").is_file():
        saved = json.loads((checkpoint_dir / "progress.json").read_text())
        if (saved["recordsSha256"] == records_sha
                and saved["format"] == CHECKPOINT_FORMAT
                and saved["baseStage"] == position - 1 and saved["baseMoramSha256"] == base_state_sha
                and file_hash(checkpoint_dir / "atoms.safetensors") == saved["atomsSha256"]
                and file_hash(checkpoint_dir / "optimizer.pt") == saved["optimizerSha256"]
                and file_hash(checkpoint_dir / "rng.pt") == saved["rngSha256"]):
            resume = TrainingProgress(saved["update"], tuple(saved["losses"]), saved["targetTokens"], saved["seconds"])
    require_valid_resume(work, position, resume is not None)
    if resume is None:
        shutil.rmtree(work, ignore_errors=True)  # fresh stage, or an interrupted stage within the first three
        work.mkdir(parents=True)
        torch.manual_seed(training["seed"] * 1000)  # stage 1's initial atoms; later stages load theirs
        model = unit.model(previous[0] if previous else None)
        optimizer = make_optimizer(model, config)
        if previous:  # upstream keeps one optimizer for the whole stream
            optimizer.load_state_dict(torch.load(previous[1], map_location=unit.device))
        torch.manual_seed(training["seed"] + position)  # dropout draws
    else:
        lines = (work / "training-log.jsonl").read_text().splitlines()[:resume.update]
        (work / "training-log.jsonl").write_text("".join(line + "\n" for line in lines))
        model = unit.model(previous[0] if previous else None)
        load_trainable_atom_state(model, load_file(str(checkpoint_dir / "atoms.safetensors")))
        optimizer = make_optimizer(model, config)
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=unit.device))
        rng = torch.load(checkpoint_dir / "rng.pt")
        torch.set_rng_state(rng["cpu"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
        print(json.dumps({"stage": position, "phase": "train", "resumedAtUpdate": resume.update}), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def save_checkpoint(progress: TrainingProgress) -> None:
        staging = work / "checkpoint.new"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        save_file(trainable_atom_state(model), str(staging / "atoms.safetensors"))
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save({"cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, staging / "rng.pt")
        atomic_json(staging / "progress.json", {
            "update": progress.update, "losses": list(progress.losses), "targetTokens": progress.target_tokens,
            "seconds": progress.seconds, "recordsSha256": records_sha,
            "format": CHECKPOINT_FORMAT, "baseStage": position - 1, "baseMoramSha256": base_state_sha,
            "atomsSha256": file_hash(staging / "atoms.safetensors"), "optimizerSha256": file_hash(staging / "optimizer.pt"),
            "rngSha256": file_hash(staging / "rng.pt")})
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.replace(staging, checkpoint_dir)
        print(json.dumps({"stage": position, "phase": "train", "checkpoint": progress.update}), flush=True)

    atomic_json(work / "started.json", {"stage": position, "job": unit.job})
    with (work / "training-log.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        summary = train_stage(model, unit.tokenizer, optimizer, records, updates=training["updates"],
                              batch_size=training["batchSize"], config=config, seed=training["seed"] + position,
                              max_length=training["maxSequenceTokens"],
                              log=lambda row: log.write(json.dumps(row) + "\n"), resume=resume,
                              checkpoint_every=CHECKPOINT_EVERY, checkpoint=save_checkpoint)
    increment_all(model, seed=training["seed"] * 1000 + position)
    counts = atom_counts(model)
    save_file(moram_state(model), str(work / "moram.safetensors"))
    torch.save(optimizer.state_dict(), work / "optimizer.pt")
    for name in ("moram.safetensors", "optimizer.pt", "training-log.jsonl"):
        os.replace(work / name, directory / name)
    atomic_json(directory / "training.json", {
        "stage": position, **summary_dict(summary), "excludedOverLength": excluded, "recordsSha256": records_sha,
        "atoms": counts, "resumedAtUpdate": resume.update if resume else 0, "job": unit.job,
        "moramSha256": file_hash(directory / "moram.safetensors"),
        "optimizerSha256": file_hash(directory / "optimizer.pt"),
        "trainingLogSha256": file_hash(directory / "training-log.jsonl")})
    shutil.rmtree(work)
    print(json.dumps({"stage": position, "phase": "train", "meanLoss": summary.mean_loss, "atoms": counts,
                      "excluded": len(excluded), "seconds": summary.seconds}), flush=True)


def greedy_predict(model, tokenizer, prompt: str, eos) -> dict[str, Any]:
    """Identical decoding to the base control's Predictor: unbatched greedy, EOS union, no truncation."""
    import torch

    started = time.perf_counter()
    device = next(model.parameters()).device
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False,
                                           return_dict=True, return_tensors="pt").to(device)
    count = inputs["input_ids"].shape[-1]
    if count > EVAL_MAX_INPUT_TOKENS:
        return {"prediction": "", "outputTokenIds": [], "promptTokens": count, "outputTokens": 0,
                "seconds": time.perf_counter() - started, "stopReason": "input_limit", "itemStatus": "failed"}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=EVAL_MAX_NEW_TOKENS, do_sample=False,
                                eos_token_id=list(eos), pad_token_id=tokenizer.pad_token_id)
    tokens = output[0, count:].tolist()
    return {"prediction": tokenizer.decode(tokens, skip_special_tokens=True), "outputTokenIds": tokens,
            "promptTokens": count, "outputTokens": len(tokens), "seconds": time.perf_counter() - started,
            "stopReason": "eos" if tokens and tokens[-1] in eos else "length", "itemStatus": "complete"}


def evaluate_unit(args) -> None:
    unit = Unit(args)
    directory, stage = unit.dir, unit.stage
    groups = [e["group"] for e in stage["evaluations"]]
    selected = args.groups.split(",") if args.groups else groups
    if set(selected) - set(groups):
        raise ValueError(f"Groups outside stage {stage['stage']}: {sorted(set(selected) - set(groups))}")
    if (directory / "evaluation.json").is_file():
        print(json.dumps({"stage": stage["stage"], "phase": "evaluate", "status": "already sealed"}), flush=True)
        return
    model = unit.model(unit.sealed_state(stage["stage"])[0]).eval()
    model.requires_grad_(False)
    for evaluation in stage["evaluations"]:
        group = evaluation["group"]
        output = directory / "predictions" / group_file(group)
        if group not in selected or output.is_file():
            continue
        prompts = {r["id"]: r["prompt"] for r in read_jsonl(unit.bundle / "inputs/eval" / group_file(group))}
        items = directory / "items" / group.replace(":", "__")
        items.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, identifier in enumerate(evaluation["ids"]):
            path = items / f"{index:04d}.json"
            if path.is_file():
                row = json.loads(path.read_text())
                if row["id"] != identifier:
                    raise ValueError(f"Cached prediction does not match the plan: {path}")
            else:
                row = {"id": identifier, "group": group, "stage": stage["stage"],
                       **greedy_predict(model, unit.tokenizer, prompts[identifier], unit.eos)}
                atomic_json(path, row)
            rows.append(row)
        write_jsonl(output, rows)
        print(json.dumps({"stage": stage["stage"], "group": group, "items": len(rows),
                          "seconds": sum(r["seconds"] for r in rows)}), flush=True)
    paths = [directory / "predictions" / group_file(g) for g in groups]
    if all(path.is_file() for path in paths):
        atomic_json(directory / "evaluation.json", {
            "stage": stage["stage"], "trainingSha256": file_hash(directory / "training.json"),
            "predictions": {path.relative_to(directory).as_posix(): file_hash(path) for path in paths}})
        print(json.dumps({"stage": stage["stage"], "phase": "evaluate", "status": "sealed"}), flush=True)


# ---------------------------------------------------------------------------
# score (local)


def stream_metrics(cells: list[dict[str, Any]], order: list[str]) -> dict[str, Any]:
    """The paper's metrics from the sparse matrix: AA and forgetting use the diagonal and the final row."""
    matrix = {(c["stage"], c["task"]): c["score"] for c in cells}
    final_stage = max((c["stage"] for c in cells), default=0)
    metrics: dict[str, Any] = {"completedStages": final_stage}
    diagonal = [matrix[(j + 1, order[j])] for j in range(final_stage) if (j + 1, order[j]) in matrix]
    if diagonal:
        metrics["acquisitionMeanDiagonal"] = sum(diagonal) / len(diagonal)
    last = len(order)
    if final_stage == last and all((last, task) in matrix for task in order):
        final = [matrix[(last, task)] for task in order]
        metrics["AA"] = sum(final) / len(final)
        earlier = list(enumerate(order[:-1], start=1))
        metrics["forgetting"] = sum(max(0.0, matrix[(j, t)] - matrix[(last, t)]) for j, t in earlier) / len(earlier)
        metrics["signedBWT"] = sum(matrix[(last, t)] - matrix[(j, t)] for j, t in earlier) / len(earlier)
    return metrics


def score(args) -> None:
    from protocol import canonical_score

    # TRACE ROUGE-L (the `rouge` package) reconstructs its LCS recursively; long predictions need depth.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
    root = Path(args.repository).resolve()
    run = args.run.resolve()
    plan = json.loads((args.bundle.resolve() / "plan.json").read_text())
    dataset = root / "experiments/data/trace-plus-17"
    manifest = load_manifest(dataset)
    if file_hash(dataset / "manifest.json") != plan["dataset"]["manifestSha256"]:
        raise ValueError("Dataset manifest differs from the plan")
    gold = {t["id"]: {r["id"]: r for r in verified_rows(dataset, t["files"]["testGold"])} for t in manifest["tasks"]}
    metrics_by_task = {t["id"]: t["metric"] for t in manifest["tasks"]}
    general = {r["id"]: r for r in read_jsonl(root / "experiments/data/general-test/v1/hidden-scorer/general-test.jsonl")}
    cells, general_cells, failures, length_stops, atoms = [], [], 0, 0, []
    for stage in plan["stages"]:
        directory = stage_dir(run, stage["stage"])
        if (directory / "training.json").is_file():
            atoms.append({"stage": stage["stage"], **json.loads((directory / "training.json").read_text())["atoms"]})
        if not (directory / "evaluation.json").is_file():
            continue
        sealed = json.loads((directory / "evaluation.json").read_text())["predictions"]
        for evaluation in stage["evaluations"]:
            path = directory / "predictions" / group_file(evaluation["group"])
            if file_hash(path) != sealed[path.relative_to(directory).as_posix()]:
                raise ValueError(f"Prediction file changed: {path}")
            rows = read_jsonl(path)
            if [r["id"] for r in rows] != evaluation["ids"]:
                raise ValueError(f"Prediction coverage differs from the plan: {path}")
            failures += sum(r["itemStatus"] != "complete" for r in rows)
            length_stops += sum(r["stopReason"] == "length" for r in rows)
            group = evaluation["group"]
            if group.startswith("general:"):
                references = [general[r["id"]] for r in rows]
                result = canonical_score(references[0]["metric"], rows, references)
                general_cells.append({"stage": stage["stage"], "category": group.split(":", 1)[1],
                                      "panel": evaluation["panel"], "items": len(rows), "score": result["score"]})
            else:
                result = canonical_score(metrics_by_task[group], rows, [gold[group][r["id"]] for r in rows])
                cells.append({"stage": stage["stage"], "task": group, "items": len(rows), "score": result["score"]})
    order = [t["id"] for t in manifest["tasks"]][:len(plan["stages"])]
    metrics = {**stream_metrics(cells, order), "itemFailures": failures, "lengthStops": length_stops}
    by_stage: dict[int, list] = {}
    for cell in general_cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    metrics["generalByStage"] = [{"stage": s, "panel": rows[0]["panel"],
                                  "macro": sum(r["score"] for r in rows) / len(rows)}
                                 for s, rows in sorted(by_stage.items()) if len(rows) == len(GENERAL_CATEGORIES)]
    report = {"method": "moram", "profile": plan["profile"], "run": str(run),
              "planSha256": file_hash(args.bundle.resolve() / "plan.json"), "atomsByStage": atoms,
              "taskCells": cells, "generalCells": general_cells, "metrics": metrics}
    atomic_json(args.output, report)
    print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--repository", default=".")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=("smoke", "trace", "full"), required=True)
    p.add_argument("--stages", type=int, default=STAGE_COUNT)
    p.add_argument("--seed", type=int, default=42, help="training seed (the paper reports seeds 42, 43 and 44)")
    p.add_argument("--updates", type=int, default=UPDATES)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--train-limit", type=int, help="Smoke only: first N training records per task")
    p.add_argument("--eval-limit", type=int, help="Smoke only: first N test items per task")
    p.add_argument("--general-limit", type=int, help="Smoke only: first N G items per category")
    p.add_argument("--remote-root", help="Render job.sh for this MLeRP directory")
    p.add_argument("--account", help="Slurm account; required with --remote-root")
    for name in ("train", "evaluate"):
        u = commands.add_parser(name)
        u.add_argument("--stage", type=int, required=True)
        u.add_argument("--bundle", type=Path, default=HERE)
        u.add_argument("--output", type=Path, required=True)
        u.add_argument("--checkpoint", type=Path, required=True)
        u.add_argument("--fetch", action="store_true", help="Download the pinned checkpoint first")
        u.add_argument("--device", default="cuda")
        if name == "evaluate":
            u.add_argument("--groups", help="Comma-separated evaluation groups for this job (default: all)")
    s = commands.add_parser("score")
    s.add_argument("--repository", default=".")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare" and args.remote_root and not args.account:
        parser.error("--account is required with --remote-root")
    if args.command == "prepare":
        if args.profile == "trace" and (args.stages != 8 or args.train_limit or args.eval_limit or args.general_limit
                                        or args.updates != 32 or args.batch_size != 32):
            parser.error("The trace profile is the eight TRACE tasks with the complete data and shared update budget")
        full = args.profile == "full"
        if full and (args.stages != STAGE_COUNT or args.train_limit or args.eval_limit or args.general_limit
                     or args.updates != UPDATES or args.batch_size != BATCH_SIZE):
            parser.error("The full profile uses the complete stream, data and shared update budget")
        prepare(args)
    elif args.command == "train":
        train_unit(args)
    elif args.command == "evaluate":
        evaluate_unit(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
