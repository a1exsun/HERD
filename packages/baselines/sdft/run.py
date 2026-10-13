"""SDFT (arXiv:2601.19897) over the TRACE+17 stream: prepare a job bundle, run atomic GPU jobs, score locally.

prepare   Build a self-contained bundle: plan, stage training records (demonstrations included),
          prompt-only evaluation inputs (no test gold), this directory's code, CUDA locks and `job.sh`.
train     Stage N: rebuild the merged stage N-1 checkpoint from the sealed adapter chain, train a fresh
          LoRA by on-policy self-distillation from the EMA demonstration-conditioned teacher, seal it.
evaluate  Stage N: greedy vLLM evaluation of the merged stage N checkpoint on the base control's sparse
          schedule; per-item resumable, optionally one group subset per job.
score     Score sealed predictions with this directory's evaluator and derive stream metrics.

Each GPU job runs exactly one (stage, phase[, subset]) unit and leaves sealed files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
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
from sdft import (  # noqa: E402
    CHECKPOINT_ENCODING, EXECUTION_ONLY_FIELDS, TEACHER_TEMPLATE, EmaTeacher, SdftConfig, StageProgress, lora_targets, summary_dict,
    train_stage,
)

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_TREE_SHA256 = "e2361c06ba172642a18b1472f4758449bcddf20072e371921642bfba28004dc5"
CHECKPOINT_FILES = ("config.json", "merges.txt", "model.safetensors-00001-of-00002.safetensors",
                    "model.safetensors-00002-of-00002.safetensors", "model.safetensors.index.json",
                    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                    "video_preprocessor_config.json", "vocab.json")
GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")
PLAN_VERSION = "sdft-plan-v3"
CHECKPOINT_EVERY = 4  # updates between mid-stage checkpoints (adapter, EMA teacher, optimizer, RNG, progress)
ROLLOUT_ENDPOINT_VARIABLE = "HERO_ROLLOUT_ENDPOINT"
BUNDLE_CODE = ("run.py", "controls.py", "protocol.py", "sdft.py", "endpoint.py", "bootstrap.sh",
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
    """Plan hash without its file manifest and execution-only settings: bundles with equal hashes run the
    same experiment and may continue one output directory."""
    science = {key: value for key, value in plan.items() if key != "files"}
    science["sdft"] = {key: value for key, value in science["sdft"].items() if key not in EXECUTION_ONLY_FIELDS}
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
               config: SdftConfig, seed: int = 42, train_limit: int | None, eval_limit: int | None,
               general_limit: int | None, rollout_server: bool = False) -> tuple[dict[str, Any], dict[str, list], dict[str, list]]:
    manifest = load_manifest(dataset)
    lora = LoraControlConfig(seed=seed)
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
        # The demonstration c is the record's own reference; nothing else from D_t is visible to the teacher.
        train_files[f"{position:02d}"] = [{key: r[key] for key in TRAIN_FIELDS if key in r} for r in records]
        evaluations = []
        for index in evaluated_task_indices(position):
            ids = [r["id"] for r in eval_files[tasks[index]["id"]]]
            evaluations.append({"group": tasks[index]["id"], "ids": ids[:eval_limit] if eval_limit else ids})
        kind = general_panel_kind(position)
        for category in GENERAL_CATEGORIES:
            ids = [r["id"] for r in eval_files["general:" + category] if kind == "full" or r["id"] in probe_ids]
            evaluations.append({"group": "general:" + category, "panel": kind,
                                "ids": ids[:general_limit] if general_limit else ids})
        plan_stages.append({"stage": position, "taskId": task["id"], "metric": task["metric"],
                            "trainRecords": len(records), "evaluations": evaluations})
    needed = {e["group"] for s in plan_stages for e in s["evaluations"]}
    eval_files = {group: rows for group, rows in eval_files.items() if group in needed}
    plan = {
        "schemaVersion": PLAN_VERSION, "method": "sdft", "paper": "arXiv:2601.19897", "profile": profile,
        "model": BASE_MODEL_ID, "revision": REVISION, "baseTreeSha256": BASE_TREE_SHA256,
        "dataset": {"name": manifest["name"], "seed": manifest["seed"], "manifestSha256": file_hash(dataset / "manifest.json")},
        "generalTestManifestSha256": file_hash(general_test / "manifest.json"),
        "stages": plan_stages,
        "sdft": {**asdict(config), "teacherTemplate": TEACHER_TEMPLATE.template,
                 "reference": "github.com/idanshen/Self-Distillation@d77573212fa0a3ae2eeb64b9b44db1c251f75e3e",
                 "demonstration": "the record's first reference",
                 "prompts": "chat template, thinking disabled, left-truncated to max_prompt_length",
                 "teacher": "a separate copy of the stage-start model; after every optimizer step each parameter "
                            "follows ref <- (1 - alpha) ref + alpha theta in the model dtype, theta being the "
                            "student's effective weight (W0 + scaling * B A for adapted Linear layers)",
                 "loss": "full-vocabulary forward KL(teacher || student) per completion "
                         "token in the model dtype, first num_loss_tokens_to_skip tokens masked, token mean per "
                         "sequence, mean over the batch",
                 "rollouts": "one on-policy completion per prompt from the current student, top-p 1, no top-k",
                 **({"rolloutServer": "vLLM; the student's adapter chain served as one LoRA with stacked factors",
                     "importanceSampling": "each sequence weighted by the mean over its loss tokens of "
                                           "min(exp(student - vLLM log-probability), importance_sampling_cap)"}
                    if rollout_server else {})},
        "training": {"updates": updates, "batchSize": batch_size, "learningRate": lora.learning_rate,
                     "schedule": "linear warmup over ceil(0.1 * updates), then cosine to 0",
                     "optimizer": "AdamW, weight decay 0, one per stage", "gradientClip": lora.gradient_clip,
                     "records": "D_t prompts in a fixed order: Dataset.shuffle(seed), then one torch.randperm "
                                "per epoch, incomplete last batch dropped", "seed": lora.seed,
                     "seeding": "transformers.set_seed(seed) at every stage start, RNG state kept across resumes",
                     "lora": {"rank": lora.rank, "alpha": lora.alpha, "dropout": lora.dropout,
                              "targets": "text-backbone Linear modules (no vision, MTP or LM head)",
                              "perStage": "fresh adapter, merged into the weights after the stage"}},
        "evaluation": {"decoding": "greedy", "engine": ENGINE, "maxNewTokens": EVAL_MAX_NEW_TOKENS, "maxInputTokens": EVAL_MAX_INPUT_TOKENS,
                       "thinking": False, "eos": "union(generation_config, tokenizer)",
                       "stageZero": "the base control's formal predictions"},
    }
    return plan, train_files, eval_files


def render_job(remote_root: str, account: str) -> str:
    """Generic MLeRP unit: one 40 GB GPU. Job name, walltime and the unit come from sbatch:
    sbatch --job-name=sdft-s01-train --time=20:00:00 job.sh train --stage 1"""
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
    config = SdftConfig(ema_alpha=args.ema_alpha)
    plan, train_files, eval_files = build_plan(
        root / "experiments/data/trace-plus-17", root / "experiments/data/general-test/v1", profile=args.profile,
        stages=args.stages, updates=args.updates, batch_size=args.batch_size, config=config, seed=args.seed,
        train_limit=args.train_limit, eval_limit=args.eval_limit, general_limit=args.general_limit,
        rollout_server=args.rollout_server)
    for key, rows in train_files.items():
        write_jsonl(bundle / "inputs/train" / f"{key}.jsonl", rows)
    for group, rows in eval_files.items():
        write_jsonl(bundle / "inputs/eval" / group_file(group), rows)
    for name in BUNDLE_CODE:
        shutil.copy2(HERE / name, bundle / name)
    if args.remote_root:
        (bundle / "job.sh").write_text(render_job(args.remote_root, args.account), encoding="utf-8")
    plan["files"] = {path.relative_to(bundle).as_posix(): file_hash(path)
                     for path in sorted(bundle.rglob("*")) if path.is_file()}
    atomic_json(bundle / "plan.json", plan)
    print(json.dumps({"bundle": str(bundle), "stages": len(plan["stages"]), "emaAlpha": config.ema_alpha,
                      "planSha256": file_hash(bundle / "plan.json")}))


# ---------------------------------------------------------------------------
# GPU units


class Unit:
    """Verified bundle, pinned checkpoint, recorded runtime contract and the tokenizer for one GPU job."""

    def __init__(self, args):
        import torch
        from transformers import AutoTokenizer

        self.bundle, self.output = args.bundle.resolve(), args.output.resolve()
        self.adapter_root = (getattr(args, "adapter_root", None) or self.output).resolve()
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
        self.config = SdftConfig(**{f.name: self.plan["sdft"][f.name] for f in fields(SdftConfig)})
        self.job = {"slurmJobId": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    def sealed_adapter(self, position: int) -> Path:
        directory = stage_dir(self.output, position)
        record = directory / "training.json"
        if not record.is_file():
            raise ValueError(f"Stage {position} is not trained")
        adapter = stage_dir(self.adapter_root, position) / "adapter"
        if tree_hash(adapter) != json.loads(record.read_text())["adapterSha256"]:
            raise ValueError(f"Stage {position} adapter hash mismatch")
        return adapter

    def merged_model(self, through: int):
        """Base checkpoint with the sealed stage adapters 1..through merged in stream order."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(self.checkpoint, dtype=torch.bfloat16,
                                                            local_files_only=True).to(self.device)
        configured = model.generation_config.eos_token_id
        self.eos = sorted(set(([configured] if isinstance(configured, int) else list(configured or []))
                              + [self.tokenizer.eos_token_id]))
        for position in range(1, through + 1):
            model = PeftModel.from_pretrained(model, self.sealed_adapter(position)).merge_and_unload(safe_merge=True)
        model.requires_grad_(False)
        return model


def rng_snapshot() -> dict[str, Any]:
    """Python, numpy, torch and CUDA RNG states, as the Trainer keeps them in a checkpoint."""
    import random

    import numpy as np
    import torch

    return {"python": random.getstate(), "numpy": np.random.get_state(), "cpu": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict[str, Any]) -> None:
    import random

    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def require_valid_resume(work: Path, stage: int, resumed: bool) -> None:
    """Preserve failed checkpoints; after stage three, never restart interrupted training."""
    if (work / "checkpoint/progress.json").is_file() and not resumed:
        raise RuntimeError("Training checkpoint failed verification; preserving it without retraining")
    if stage > 3 and not resumed and work.exists() and any(work.iterdir()):
        raise RuntimeError("After stage 3, interrupted training requires a verified checkpoint; no retraining")


def checkpoint_record(directory: Path):
    path = directory / "progress.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if (tree_hash(directory / "adapter") == saved["adapterSha256"]
            and saved["teacherEncoding"] == CHECKPOINT_ENCODING
            and file_hash(directory / "teacher.xor.gz") == saved["teacherSha256"]
            and file_hash(directory / "optimizer.pt") == saved["optimizerSha256"]
            and file_hash(directory / "rng.pt") == saved["rngSha256"]):
        return saved
    return None


def recover_pending_checkpoint(work: Path) -> None:
    """Finish an interrupted verified checkpoint directory transaction in the same run."""
    pending, checkpoint = work / "checkpoint.new", work / "checkpoint"
    if not pending.exists():
        return
    saved = checkpoint_record(pending)
    if saved is None:
        raise RuntimeError("Pending checkpoint failed verification; existing state preserved")
    previous = checkpoint_record(checkpoint)
    if previous is not None and saved["nextUpdate"] < previous["nextUpdate"]:
        raise RuntimeError("Pending checkpoint would roll back progress")
    shutil.rmtree(checkpoint, ignore_errors=True)
    os.replace(pending, checkpoint)


def persist_checkpoint(staging: Path, checkpoint: Path) -> None:
    """Copy measured scratch state to persistent storage before retiring the previous checkpoint."""
    size = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
    if shutil.disk_usage(checkpoint.parent).free < size + 512 * 1024 ** 2:
        raise RuntimeError("Checkpoint exceeds persistent capacity; previous checkpoint preserved")
    destination = checkpoint.with_name("checkpoint.new")
    if destination.exists():
        raise RuntimeError("Unresolved checkpoint.new must be verified before continuation")
    shutil.copytree(staging, destination)
    if tree_hash(destination) != tree_hash(staging):
        raise RuntimeError("Persistent checkpoint copy failed verification; previous checkpoint preserved")
    shutil.rmtree(checkpoint, ignore_errors=True)
    os.replace(destination, checkpoint)


def train_unit(args) -> None:
    import copy

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    unit = Unit(args)
    position, directory, training, config = args.stage, unit.dir, unit.plan["training"], unit.config
    if (directory / "training.json").is_file():
        unit.sealed_adapter(position)
        print(json.dumps({"stage": position, "phase": "train", "status": "already sealed"}), flush=True)
        return
    records = read_jsonl(unit.bundle / "inputs/train" / f"{position:02d}.jsonl")
    base = unit.merged_model(position - 1)

    # Mid-stage checkpoint: training.tmp/checkpoint/{adapter, teacher.xor.gz, optimizer.pt, progress.json}.
    work = directory / "training.tmp"
    checkpoint_dir = work / "checkpoint"
    progress, optimizer_state, teacher_checkpoint, rng_state = None, None, None, None
    recover_pending_checkpoint(work)
    saved = checkpoint_record(checkpoint_dir)
    if saved is not None:
        progress = StageProgress(saved["nextUpdate"], tuple(saved["losses"]), saved["seconds"])
        optimizer_state = torch.load(checkpoint_dir / "optimizer.pt", map_location=unit.device)
        teacher_checkpoint = checkpoint_dir / "teacher.xor.gz"
        rng_state = torch.load(checkpoint_dir / "rng.pt", weights_only=False)
    # The teacher is a separate copy of the stage-start weights, taken before PEFT wraps `base` in place.
    teacher_model = copy.deepcopy(base)
    require_valid_resume(work, position, progress is not None)
    if progress is None:
        shutil.rmtree(work, ignore_errors=True)  # fresh stage, or an interrupted stage within the first three
        work.mkdir(parents=True)
        lora = training["lora"]
        set_seed(training["seed"])  # seeds the adapter initialisation as well as the stage
        model = get_peft_model(base, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                                bias="none", target_modules=lora_targets(base), task_type="CAUSAL_LM"))
    else:
        lines = (work / "training-log.jsonl").read_text().splitlines()
        kept = [line for line in lines if json.loads(line)["update"] < progress.next_update]
        (work / "training-log.jsonl").write_text("".join(line + "\n" for line in kept))
        model = PeftModel.from_pretrained(base, checkpoint_dir / "adapter", is_trainable=True)
        print(json.dumps({"stage": position, "phase": "train", "resumedAtUpdate": progress.next_update}), flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    teacher = EmaTeacher(teacher_model, model, config.ema_alpha)
    if teacher_checkpoint is not None:
        teacher.load_checkpoint(teacher_checkpoint)

    def save_checkpoint(state: StageProgress, optimizer_state_dict) -> None:
        staging = Path(tempfile.mkdtemp(prefix="sdft-checkpoint-", dir=unit.checkpoint.parent))
        model.save_pretrained(staging / "adapter", safe_serialization=True)
        teacher.save_checkpoint(staging / "teacher.xor.gz")
        torch.save(optimizer_state_dict, staging / "optimizer.pt")
        torch.save(rng_snapshot(), staging / "rng.pt")
        atomic_json(staging / "progress.json", {
            "nextUpdate": state.next_update, "losses": list(state.losses), "seconds": state.seconds,
            "adapterSha256": tree_hash(staging / "adapter"), "teacherEncoding": CHECKPOINT_ENCODING,
            "teacherSha256": file_hash(staging / "teacher.xor.gz"),
            "optimizerSha256": file_hash(staging / "optimizer.pt"), "rngSha256": file_hash(staging / "rng.pt")})
        persist_checkpoint(staging, checkpoint_dir)
        shutil.rmtree(staging)
        print(json.dumps({"stage": position, "phase": "train", "checkpoint": state.next_update}), flush=True)

    sampler = None
    if unit.plan["sdft"].get("rolloutServer"):
        from endpoint import EndpointSampler

        sampler = EndpointSampler(os.environ[ROLLOUT_ENDPOINT_VARIABLE], [unit.sealed_adapter(k) for k in range(1, position)],
                                  Path(os.environ.get("HERO_ROLLOUT_SCRATCH", "/dev/shm/sdft-rollouts")),
                                  f"sdft-s{position:02d}")
    if rng_state is not None:
        restore_rng(rng_state)
    atomic_json(work / "started.json", {"stage": position, "job": unit.job})
    with (work / "training-log.jsonl").open("a", encoding="utf-8", buffering=1) as log:
        summary = train_stage(model, unit.tokenizer, stage=position, task_id=unit.stage["taskId"], records=records,
                              updates=training["updates"], batch_size=training["batchSize"],
                              learning_rate=training["learningRate"], gradient_clip=training["gradientClip"],
                              seed=training["seed"], eos=unit.eos, config=config, teacher=teacher,
                              log=lambda row: log.write(json.dumps(row, ensure_ascii=False) + "\n"),
                              progress=progress, optimizer_state=optimizer_state, checkpoint_every=CHECKPOINT_EVERY,
                              save_checkpoint=save_checkpoint, sampler=sampler)
    model.save_pretrained(work / "adapter", safe_serialization=True)
    for name in ("adapter", "training-log.jsonl"):
        os.replace(work / name, directory / name)
    atomic_json(directory / "training.json", {
        **summary_dict(summary), "trainRecords": len(records), "emaAlpha": config.ema_alpha,
        "resumedAtUpdate": progress.next_update if progress else 0, "job": unit.job,
        "rolloutServer": sampler.server if sampler is not None else None,
        "adapterSha256": tree_hash(directory / "adapter"),
        "trainingLogSha256": file_hash(directory / "training-log.jsonl")})
    shutil.rmtree(work)
    print(json.dumps({"stage": position, "phase": "train", "meanLoss": summary.mean_loss,
                      "truncated": len(summary.truncated_prompts), "seconds": summary.seconds}), flush=True)


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
        evaluate_items(unit.merged_model(stage["stage"]).eval(), unit.checkpoint, unit.tokenizer, unit.eos, pending, atomic_json,
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
    cells, general_cells, failures, length_stops = [], [], 0, 0
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
    order = [t["id"] for t in manifest["tasks"]][:len(plan["stages"])]
    metrics = {**stream_metrics(cells, order), "itemFailures": failures, "lengthStops": length_stops}
    by_stage: dict[int, list] = {}
    for cell in general_cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    metrics["generalByStage"] = [{"stage": s, "panel": rows[0]["panel"],
                                  "macro": sum(r["score"] for r in rows) / len(rows)}
                                 for s, rows in sorted(by_stage.items()) if len(rows) == len(GENERAL_CATEGORIES)]
    report = {"method": "sdft", "profile": plan["profile"], "emaAlpha": plan["sdft"]["ema_alpha"], "run": str(run),
              "planSha256": file_hash(args.bundle.resolve() / "plan.json"),
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
    p.add_argument("--rollout-server", action="store_true",
                   help=f"Sample rollouts from the vLLM server at ${ROLLOUT_ENDPOINT_VARIABLE}")
    p.add_argument("--ema-alpha", type=float, default=SdftConfig().ema_alpha,
                   help="EMA teacher rate (paper sweep 0.01, 0.02, 0.05; released default 0.01)")
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
        u.add_argument("--stage", type=int, required=True)
        u.add_argument("--bundle", type=Path, default=HERE)
        u.add_argument("--output", type=Path, required=True)
        u.add_argument("--checkpoint", type=Path, required=True)
        u.add_argument("--adapter-root", type=Path, help="Verified job-local sealed adapter chain")
        u.add_argument("--fetch", action="store_true", help="Download the pinned checkpoint first")
        u.add_argument("--device", default="cuda")
        if name == "evaluate":
            u.add_argument("--groups", help="Comma-separated evaluation groups for this job (default: all)")
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
        if args.profile == "trace" and (args.stages != 8 or args.train_limit or args.eval_limit or args.general_limit
                                        or args.updates != 32 or args.batch_size != 32):
            parser.error("The trace profile is the eight TRACE tasks with the complete data and shared update budget")
        full = args.profile == "full"
        if full and (args.stages != STAGE_COUNT or args.train_limit or args.eval_limit or args.general_limit
                     or args.updates != LoraControlConfig().fixed_updates
                     or args.batch_size != LoraControlConfig().effective_batch_size):
            parser.error("The full profile uses the complete stream, data and shared update budget")
        if not 0 < args.ema_alpha <= 1:
            parser.error("--ema-alpha must be in (0, 1]")
        prepare(args)
    elif args.command == "train":
        train_unit(args)
    elif args.command == "evaluate":
        evaluate_unit(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
