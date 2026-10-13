"""TreeLoRA (ICML 2025) over the TRACE+17 stream: prepare a job bundle, run atomic GPU jobs, score locally.

prepare   Build a self-contained bundle: plan, stage training records, prompt-only evaluation inputs
          (no test gold), this directory's code (including `upstream/`), CUDA locks and `job.sh`.
train     Stage N: load the stage N-1 adapter and optimizer and the sealed fingerprints of tasks 1..N-1,
          rebuild the upstream KD tree, train the task with the upstream TreeLoRA step, then seal the
          adapter, the optimizer, the task's fingerprint and the rebuilt tree.
evaluate  Stage N: greedy vLLM evaluation of the stage N adapter on the plan's evaluation schedule;
          per-item resumable, optionally one group subset per job.
score     Score sealed predictions with this directory's evaluator and derive stream metrics.

`--stage` takes N or A-B; a GPU job runs its (stage, phase[, subset]) units in order inside one
process and every unit leaves sealed files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from functools import lru_cache
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

from controls import LoraControlConfig  # noqa: E402
from vllm_decode import ENGINE  # noqa: E402
from protocol import (  # noqa: E402
    BASE_MODEL_ID, EVAL_MAX_INPUT_TOKENS, EVAL_MAX_NEW_TOKENS, STAGE_COUNT, evaluated_task_indices,
    general_panel_kind, load_manifest, read_jsonl,
)
from sft import TrainingProgress, summary_dict, supervised_token_ids, text_backbone_lora_targets  # noqa: E402
from treelora_cl import (  # noqa: E402
    TreeLoraConfig, make_optimizer, new_tree, restore_tree_epoch_state, train_stage, tree_dict, tree_epoch_state,
)

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_TREE_SHA256 = "e2361c06ba172642a18b1472f4758449bcddf20072e371921642bfba28004dc5"
CHECKPOINT_FILES = ("config.json", "merges.txt", "model.safetensors-00001-of-00002.safetensors",
                    "model.safetensors-00002-of-00002.safetensors", "model.safetensors.index.json",
                    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                    "video_preprocessor_config.json", "vocab.json")
GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")
PLAN_VERSION = "treelora-plan-v2"
UPSTREAM_COMMIT = "1c7260c42b34e1961283797c742f08b9c3842501"
CHECKPOINT_EVERY = 8  # default updates between mid-stage checkpoints (adapter, optimizer, tree, RNG, progress)
MIN_FREE_BYTES = 1 << 30  # a unit starts only with this much free space on the output volume
UPSTREAM_FILES = tuple(sorted(json.loads((HERE / "upstream/UPSTREAM.json").read_text())["files"]))
BUNDLE_CODE = ("run.py", "controls.py", "protocol.py", "sft.py", "microbatch.py", "treelora_cl.py",
               "upstream/UPSTREAM.json", *(f"upstream/{name}" for name in UPSTREAM_FILES), "bootstrap.sh",
               "requirements-cuda.lock.txt", "requirements-kernels.lock.txt", "requirements-eval.lock.txt",
               "vllm_decode.py")
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


@lru_cache(maxsize=None)
def checkpoint_tree_hash(path: Path) -> str:
    """The base checkpoint is read-only within a job, so each unit of the job reuses its hash."""
    return tree_hash(path)


def stage_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    stages = list(range(int(first), int(last or first) + 1))
    if not stages:
        raise argparse.ArgumentTypeError(f"Empty stage range: {text}")
    return stages


# ---------------------------------------------------------------------------
# prepare (local)


def build_plan(dataset: Path, general_test: Path, *, profile: str, stages: int, updates: int, batch_size: int,
               seed: int = 42, train_limit: int | None, eval_limit: int | None,
               general_limit: int | None) -> tuple[dict[str, Any], dict[str, list], dict[str, list]]:
    manifest = load_manifest(dataset)
    budget, config = LoraControlConfig(seed=seed), TreeLoraConfig()
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
        "schemaVersion": PLAN_VERSION, "method": "treelora", "paper": "ICML 2025, PMLR 267:50066-50085",
        "upstream": {"repository": "https://github.com/ZinYY/TreeLoRA", "commit": UPSTREAM_COMMIT},
        "profile": profile,
        "model": BASE_MODEL_ID, "revision": REVISION, "baseTreeSha256": BASE_TREE_SHA256,
        "dataset": {"name": manifest["name"], "seed": manifest["seed"], "manifestSha256": file_hash(dataset / "manifest.json")},
        "generalTestManifestSha256": file_hash(general_test / "manifest.json"),
        "stages": plan_stages,
        "treelora": {**asdict(config),
                     "step": "upstream Tree_LoRA.train_one_task per record (micro-step), one optimizer step per batch",
                     "epoch": "one pass over the stage's records; a final partial pass joins the last epoch",
                     "loraRows": "one row per LoRA module, zero-padded to the widest module"},
        "training": {"updates": updates, "batchSize": batch_size, "schedule": "constant, no warmup",
                     "optimizer": "Adam, one state carried across the whole stream",
                     "loss": "per-record token-mean cross-entropy, averaged over the batch",
                     "microBatchTokens": 24576,
                     "records": "D_t only, seeded shuffle, reshuffled each epoch",
                     "maxSequenceTokens": EVAL_MAX_INPUT_TOKENS, "seed": budget.seed,
                     "lora": {"targets": "text-backbone Linear modules (no vision, MTP or LM head)",
                              "carriedAcrossStages": True}},
        "evaluation": {"decoding": "greedy", "engine": ENGINE, "maxNewTokens": EVAL_MAX_NEW_TOKENS, "maxInputTokens": EVAL_MAX_INPUT_TOKENS,
                       "thinking": False, "eos": "union(generation_config, tokenizer)",
                       "stageZero": "the base control's formal predictions", "schedule": "protocol"},
    }
    return plan, train_files, eval_files


def render_job(remote_root: str, account: str) -> str:
    """Generic MLeRP unit: one 40 GB GPU. Job name, walltime and the unit come from sbatch:
    sbatch --job-name=treelora-s01-train --time=06:00:00 job.sh train --stage 1"""
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
    plan, train_files, eval_files = build_plan(
        root / "experiments/data/trace-plus-17", root / "experiments/data/general-test/v1", profile=args.profile,
        stages=args.stages, updates=args.updates, batch_size=args.batch_size, seed=args.seed, train_limit=args.train_limit,
        eval_limit=args.eval_limit, general_limit=args.general_limit)
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
        if checkpoint_tree_hash(self.checkpoint) != self.plan["baseTreeSha256"]:
            raise ValueError("Base checkpoint differs from the pinned tree hash")
        science = {"scientificPlanSha256": scientific_plan_hash(self.plan), "baseTreeSha256": self.plan["baseTreeSha256"]}
        libraries = ("torch", "transformers", "peft")
        if args.command == "evaluate":
            # Evaluation runs in its own vLLM environment: it must evaluate the run trained under this plan, and
            # keeps its own library contract next to its predictions.
            trained = self.output / "contract.json"
            if not trained.is_file() or {key: json.loads(trained.read_text())[key] for key in science} != science:
                raise ValueError("Evaluation needs the training run's contract under the same scientific plan")
            self.predictions = stage_dir((args.predictions_root or self.output).resolve(), args.stage)
            contract = {**science, "engine": ENGINE,
                        "libraries": {name: __import__(name).__version__ for name in (*libraries, "vllm")}}
            path = self.predictions.parent / "contract-evaluate.json"
        else:
            contract = {**science, "libraries": {name: __import__(name).__version__ for name in libraries}}
            path = self.output / "contract.json"
        if path.is_file():
            if json.loads(path.read_text()) != contract:
                raise ValueError("Unit runs with a different scientific plan, checkpoint or library set")
        else:
            atomic_json(path, contract)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        self.tokenizer.padding_side = "left"
        self.job = {"slurmJobId": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    def sealed_adapter(self, position: int) -> Path:
        directory = stage_dir(self.output, position)
        record = directory / "training.json"
        if not record.is_file():
            raise ValueError(f"Stage {position} is not trained")
        if tree_hash(directory / "adapter") != json.loads(record.read_text())["adapterSha256"]:
            raise ValueError(f"Stage {position} adapter hash mismatch")
        return directory / "adapter"

    def sealed_fingerprint(self, position: int):
        """Task `position`'s KD-tree fingerprint (upstream `current_grad` at its `end_task`), on the unit's device."""
        from safetensors.torch import load_file

        directory = stage_dir(self.output, position)
        record = json.loads((directory / "training.json").read_text())
        if file_hash(directory / "fingerprint.safetensors") != record["fingerprintSha256"]:
            raise ValueError(f"Stage {position} fingerprint hash mismatch")
        return load_file(str(directory / "fingerprint.safetensors"), device=self.device)["fingerprint"]

    def sealed_optimizer(self, position: int) -> Path:
        directory = stage_dir(self.output, position)
        if file_hash(directory / "optimizer.pt") != json.loads((directory / "training.json").read_text())["optimizerSha256"]:
            raise ValueError(f"Stage {position} optimizer hash mismatch")
        return directory / "optimizer.pt"

    def model(self, adapter: Path | None, *, trainable: bool):
        """Base checkpoint plus the carried LoRA (fresh at stage 1); merged into the weights for inference."""
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForImageTextToText

        base = AutoModelForImageTextToText.from_pretrained(self.checkpoint, dtype=torch.bfloat16,
                                                           local_files_only=True).to(self.device)
        configured = base.generation_config.eos_token_id
        self.eos = sorted(set(([configured] if isinstance(configured, int) else list(configured or []))
                              + [self.tokenizer.eos_token_id]))
        if adapter is not None:
            model = PeftModel.from_pretrained(base, adapter, is_trainable=trainable)
            return model if trainable else model.merge_and_unload(safe_merge=True)
        if not trainable:
            raise ValueError("Inference needs a trained adapter")
        config = self.plan["treelora"]
        return get_peft_model(base, LoraConfig(r=config["rank"], lora_alpha=config["lora_alpha"],
                                               lora_dropout=config["lora_dropout"], bias="none",
                                               target_modules=text_backbone_lora_targets(base), task_type="CAUSAL_LM"))


def rng_state() -> dict[str, Any]:
    import torch

    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    torch.set_rng_state(state["cpu"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def train_unit(args) -> None:
    import torch

    unit = Unit(args)
    position, directory, training = args.stage, unit.dir, unit.plan["training"]
    config = TreeLoraConfig(**{name: unit.plan["treelora"][name] for name in TreeLoraConfig.__dataclass_fields__})
    if (directory / "training.json").is_file():
        unit.sealed_adapter(position)
        print(json.dumps({"stage": position, "phase": "train", "status": "already sealed"}), flush=True)
        return
    if position > 1:
        unit.sealed_adapter(position - 1)
    records, excluded = [], []
    for record in read_jsonl(unit.bundle / "inputs/train" / f"{position:02d}.jsonl"):
        try:
            supervised_token_ids(unit.tokenizer, record, max_length=training["maxSequenceTokens"])
        except ValueError:
            excluded.append(record["id"])
            continue
        records.append(record)
    records_sha = hashlib.sha256(json.dumps([r["id"] for r in records]).encode()).hexdigest()
    task_id = position - 1  # upstream's 0-based task index
    tree = new_tree(len(unit.plan["stages"]), config.reg,
                    [unit.sealed_fingerprint(previous) for previous in range(1, position)])

    # Mid-stage checkpoint: training.tmp/checkpoint/{adapter, optimizer.pt, tree.pt, rng.pt, progress.json}.
    work = directory / "training.tmp"
    checkpoint_dir = work / "checkpoint"
    resume = None
    if (checkpoint_dir / "progress.json").is_file():
        saved = json.loads((checkpoint_dir / "progress.json").read_text())
        if (saved["recordsSha256"] == records_sha and tree_hash(checkpoint_dir / "adapter") == saved["adapterSha256"]
                and all(file_hash(checkpoint_dir / name) == saved[key] for name, key in
                        (("optimizer.pt", "optimizerSha256"), ("tree.pt", "treeSha256"), ("rng.pt", "rngSha256")))):
            resume = TrainingProgress(saved["update"], tuple(saved["losses"]), saved["targetTokens"], saved["seconds"])
    if resume is None:
        shutil.rmtree(work, ignore_errors=True)  # nothing verifiable to continue from
        work.mkdir(parents=True)
        torch.manual_seed(training["seed"] + position)  # LoRA initialisation (stage 1), dropout, tree_search draws
        model = unit.model(unit.sealed_adapter(position - 1) if position > 1 else None, trainable=True)
        optimizer = make_optimizer(model, config)
        if position > 1:  # upstream keeps one optimizer for the whole stream
            optimizer.load_state_dict(torch.load(unit.sealed_optimizer(position - 1), map_location=unit.device))
        rows = sum("lora_A" in name for name, _ in model.named_parameters())
        selection = torch.zeros((task_id, rows), dtype=torch.long, device=unit.device) if task_id else None
    else:
        lines = (work / "training-log.jsonl").read_text().splitlines()[:resume.update]
        (work / "training-log.jsonl").write_text("".join(line + "\n" for line in lines))
        model = unit.model(checkpoint_dir / "adapter", trainable=True)
        optimizer = make_optimizer(model, config)
        optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=unit.device))
        saved_tree = torch.load(checkpoint_dir / "tree.pt", map_location=unit.device, weights_only=False)
        restore_tree_epoch_state(tree, saved_tree["epoch"], task_id, unit.device)
        selection = saved_tree["selection"]
        restore_rng_state(torch.load(checkpoint_dir / "rng.pt", weights_only=False))
        print(json.dumps({"stage": position, "phase": "train", "resumedAtUpdate": resume.update}), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def save_checkpoint(progress: TrainingProgress) -> None:
        staging = work / "checkpoint.new"
        shutil.rmtree(staging, ignore_errors=True)
        model.save_pretrained(staging / "adapter", safe_serialization=True)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save({"epoch": tree_epoch_state(tree), "selection": selection}, staging / "tree.pt")
        torch.save(rng_state(), staging / "rng.pt")
        atomic_json(staging / "progress.json", {
            "update": progress.update, "losses": list(progress.losses), "targetTokens": progress.target_tokens,
            "seconds": progress.seconds, "recordsSha256": records_sha,
            "adapterSha256": tree_hash(staging / "adapter"), "optimizerSha256": file_hash(staging / "optimizer.pt"),
            "treeSha256": file_hash(staging / "tree.pt"), "rngSha256": file_hash(staging / "rng.pt")})
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
        os.replace(staging, checkpoint_dir)
        print(json.dumps({"stage": position, "phase": "train", "checkpoint": progress.update}), flush=True)

    with (work / "training-log.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        summary = train_stage(model, unit.tokenizer, optimizer, tree, records, task_id=task_id,
                              updates=training["updates"], batch_size=training["batchSize"], config=config,
                              seed=training["seed"] + position, max_length=training["maxSequenceTokens"],
                              selection=selection, log=lambda row: log.write(json.dumps(row) + "\n"), resume=resume,
                              checkpoint_every=args.checkpoint_every, checkpoint=save_checkpoint)
    from safetensors.torch import save_file

    tree.end_task(task_id=task_id)  # upstream, after each task: seal the fingerprint and rebuild the tree
    fingerprint = tree.all_accumulate_grads[task_id].detach().cpu().contiguous()
    save_file({"fingerprint": fingerprint}, str(work / "fingerprint.safetensors"))
    atomic_json(work / "gradient-tree.json", tree_dict(tree.kd_tree_root))
    torch.save(optimizer.state_dict(), work / "optimizer.pt")
    model.save_pretrained(work / "adapter", safe_serialization=True)
    for name in ("adapter", "optimizer.pt", "fingerprint.safetensors", "gradient-tree.json", "training-log.jsonl"):
        os.replace(work / name, directory / name)
    shutil.rmtree(work)
    atomic_json(directory / "training.json", {
        "stage": position, **summary_dict(summary), "excludedOverLength": excluded, "recordsSha256": records_sha,
        "reg": config.reg, "historicalTasks": task_id,
        "selectedCounts": selection.tolist() if selection is not None else None,
        "fingerprintShape": list(fingerprint.shape), "fingerprintBytes": fingerprint.numel() * fingerprint.element_size(),
        "fingerprintSha256": file_hash(directory / "fingerprint.safetensors"),
        "gradientTreeSha256": file_hash(directory / "gradient-tree.json"),
        "optimizerSha256": file_hash(directory / "optimizer.pt"),
        "resumedAtUpdate": resume.update if resume else 0, "checkpointEvery": args.checkpoint_every, "job": unit.job,
        "adapterSha256": tree_hash(directory / "adapter"),
        "trainingLogSha256": file_hash(directory / "training-log.jsonl")})
    print(json.dumps({"stage": position, "phase": "train", "meanLoss": summary.mean_loss,
                      "excluded": len(excluded), "seconds": summary.seconds}), flush=True)


def evaluate_unit(args) -> None:
    """Greedy evaluation of the stage's merged checkpoint with vLLM (see vllm_decode); per-item resumable.

    Predictions go to `--predictions-root` (default: the run's output), so a sealed run can be re-evaluated
    without touching its earlier predictions."""
    from vllm_decode import evaluate_items

    unit = Unit(args)
    directory, stage = unit.predictions, unit.stage
    groups = [e["group"] for e in stage["evaluations"]]
    selected = args.groups.split(",") if args.groups else groups
    if set(selected) - set(groups):
        raise ValueError(f"Groups outside stage {stage['stage']}: {sorted(set(selected) - set(groups))}")
    if (directory / "evaluation.json").is_file():
        print(json.dumps({"stage": stage["stage"], "phase": "evaluate", "status": "already sealed"}), flush=True)
        return
    pending = []
    for evaluation in stage["evaluations"]:
        group = evaluation["group"]
        if group not in selected or (directory / "predictions" / group_file(group)).is_file():
            continue
        prompts = {r["id"]: r["prompt"] for r in read_jsonl(unit.bundle / "inputs/eval" / group_file(group))}
        items = directory / "items" / group.replace(":", "__")
        items.mkdir(parents=True, exist_ok=True)
        for index, identifier in enumerate(evaluation["ids"]):
            path = items / f"{index:04d}.json"
            if path.is_file():
                if json.loads(path.read_text())["id"] != identifier:
                    raise ValueError(f"Cached prediction does not match the plan: {path}")
            else:
                pending.append((path, {"id": identifier, "group": group, "stage": stage["stage"]}, prompts[identifier]))
    if pending:
        evaluate_items(unit.model(unit.sealed_adapter(stage["stage"]), trainable=False).eval(), unit.checkpoint, unit.tokenizer, unit.eos, pending, atomic_json,
                       max_input_tokens=EVAL_MAX_INPUT_TOKENS, max_new_tokens=EVAL_MAX_NEW_TOKENS)
    for evaluation in stage["evaluations"]:
        group = evaluation["group"]
        output = directory / "predictions" / group_file(group)
        if group not in selected or output.is_file():
            continue
        items = directory / "items" / group.replace(":", "__")
        rows = [json.loads((items / f"{index:04d}.json").read_text()) for index in range(len(evaluation["ids"]))]
        write_jsonl(output, rows)
        print(json.dumps({"stage": stage["stage"], "group": group, "items": len(rows),
                          "seconds": sum(r["seconds"] for r in rows)}), flush=True)
    paths = [directory / "predictions" / group_file(g) for g in groups]
    if all(path.is_file() for path in paths):
        atomic_json(directory / "evaluation.json", {
            "stage": stage["stage"], "engine": ENGINE, "trainingSha256": file_hash(unit.dir / "training.json"),
            "predictions": {path.relative_to(directory).as_posix(): file_hash(path) for path in paths}})
        print(json.dumps({"stage": stage["stage"], "phase": "evaluate", "status": "sealed"}), flush=True)


# ---------------------------------------------------------------------------
# score (local)


def stream_metrics(cells: list[dict[str, Any]], order: list[str],
                   item_scores: dict[tuple[int, str], dict[str, float]] | None = None) -> dict[str, Any]:
    """The paper's metrics from the sparse matrix: AA and forgetting use the diagonal and the final row.

    With `item_scores`, forgetting and signed BWT read the final row on each diagonal cell's own items,
    so a diagonal measured on a subset of the final panel is compared item for item."""
    matrix = {(c["stage"], c["task"]): c["score"] for c in cells}
    last = len(order)
    if item_scores:
        def final_on_diagonal_items(j: int, task: str) -> float:
            diagonal, final = item_scores[(j, task)], item_scores[(last, task)]
            return sum(final[identifier] for identifier in diagonal) / len(diagonal)
    else:
        def final_on_diagonal_items(j: int, task: str) -> float:
            return matrix[(last, task)]
    final_stage = max((c["stage"] for c in cells), default=0)
    metrics: dict[str, Any] = {"completedStages": final_stage}
    diagonal = [matrix[(j + 1, order[j])] for j in range(final_stage) if (j + 1, order[j]) in matrix]
    if diagonal:
        metrics["acquisitionMeanDiagonal"] = sum(diagonal) / len(diagonal)
    if final_stage == last and all((last, task) in matrix for task in order):
        final = [matrix[(last, task)] for task in order]
        metrics["AA"] = sum(final) / len(final)
        earlier = [(j, t, final_on_diagonal_items(j, t)) for j, t in enumerate(order[:-1], start=1)]
        metrics["forgetting"] = sum(max(0.0, matrix[(j, t)] - end) for j, t, end in earlier) / len(earlier)
        metrics["signedBWT"] = sum(end - matrix[(j, t)] for j, t, end in earlier) / len(earlier)
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
    cells, general_cells, failures, length_stops = [], [], 0, 0
    item_scores: dict[tuple[int, str], dict[str, float]] = {}
    for stage in plan["stages"]:
        directory = stage_dir(run, stage["stage"])
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
                item_scores[(stage["stage"], group)] = {i["id"]: i["score"] for i in result["itemScores"]}
    order = [t["id"] for t in manifest["tasks"]][:len(plan["stages"])]
    metrics = {**stream_metrics(cells, order, item_scores), "itemFailures": failures, "lengthStops": length_stops}
    by_stage: dict[int, list] = {}
    for cell in general_cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    metrics["generalByStage"] = [{"stage": s, "panel": rows[0]["panel"],
                                  "macro": sum(r["score"] for r in rows) / len(rows)}
                                 for s, rows in sorted(by_stage.items()) if len(rows) == len(GENERAL_CATEGORIES)]
    report = {"method": "treelora", "profile": plan["profile"],
              "evaluationSchedule": plan["evaluation"].get("schedule", "protocol"),
              "reg": plan["treelora"]["reg"],
              "run": str(run), "planSha256": file_hash(args.bundle.resolve() / "plan.json"),
              "taskCells": cells, "generalCells": general_cells, "metrics": metrics}
    atomic_json(args.output, report)
    print(json.dumps(metrics, indent=2))


def run_units(args, phases) -> None:
    """Run each stage's phases in order inside this process: one job start-up serves many sealed units."""
    import gc

    import torch

    if getattr(args, "groups", None) and len(args.stage) != 1:
        raise ValueError("--groups selects panels of a single stage")
    output = args.output.resolve()
    volume = next(path for path in (output, *output.parents) if path.exists())
    for position in args.stage:
        for phase in phases:
            if shutil.disk_usage(volume).free < MIN_FREE_BYTES:
                raise SystemExit(f"Less than {MIN_FREE_BYTES >> 30} GiB free on {volume}; "
                                 f"archive sealed stages before stage {position}")
            phase(argparse.Namespace(**{**vars(args), "stage": position}))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--repository", default=".")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=("smoke", "full"), required=True)
    p.add_argument("--stages", type=int, default=STAGE_COUNT)
    p.add_argument("--seed", type=int, default=42, help="training seed (the paper reports seeds 42, 43 and 44)")
    p.add_argument("--updates", type=int, default=LoraControlConfig().fixed_updates)
    p.add_argument("--batch-size", type=int, default=LoraControlConfig().effective_batch_size)
    p.add_argument("--train-limit", type=int, help="Smoke only: first N training records per task")
    p.add_argument("--eval-limit", type=int, help="Smoke only: first N test items per task")
    p.add_argument("--general-limit", type=int, help="Smoke only: first N G items per category")
    p.add_argument("--remote-root", help="Render job.sh for this MLeRP directory")
    p.add_argument("--account", help="Slurm account; required with --remote-root")
    for name in ("train", "evaluate"):
        u = commands.add_parser(name)
        u.add_argument("--stage", type=stage_range, required=True, help="N or A-B, run in order")
        u.add_argument("--bundle", type=Path, default=HERE)
        u.add_argument("--output", type=Path, required=True)
        u.add_argument("--checkpoint", type=Path, required=True)
        u.add_argument("--fetch", action="store_true", help="Download the pinned checkpoint first")
        u.add_argument("--device", default="cuda")
        if name == "train":
            u.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                           help="Updates between mid-stage checkpoints; 0 restarts an interrupted stage")
        if name == "evaluate":
            u.add_argument("--groups", help="Comma-separated evaluation groups of a single stage (default: all)")
            u.add_argument("--predictions-root", type=Path,
                           help="Write predictions here instead of the run's output (re-evaluating a sealed run)")
    s = commands.add_parser("score")
    s.add_argument("--repository", default=".")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare" and args.remote_root and not args.account:
        parser.error("--account is required with --remote-root")
    if args.command == "prepare":
        full = args.profile == "full"
        if full and (args.stages != STAGE_COUNT or args.train_limit or args.eval_limit or args.general_limit
                     or args.updates != LoraControlConfig().fixed_updates
                     or args.batch_size != LoraControlConfig().effective_batch_size):
            parser.error("The full profile uses the complete stream, data and shared update budget")
        prepare(args)
    elif args.command == "train":
        run_units(args, (train_unit,))
    elif args.command == "evaluate":
        run_units(args, (evaluate_unit,))
    else:
        score(args)


if __name__ == "__main__":
    main()
