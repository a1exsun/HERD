"""On-Policy Replay (OPR-RU, arXiv:2605.29495) over the TRACE+17 stream: prepare a job
bundle, run atomic GPU jobs, score locally.

The method code is `official/`, driven one stage at a time by this runner, which only
supplies what its `main.py` loop normally gets from its environment: the data
directory, and the shared protocol evaluation in place of `tools/eval.py`.
Edits relative to the reference copy in `upstream/` carry a `[HERO]` marker.

prepare   Build a self-contained bundle: plan, `data/<task>/train.jsonl` in the method's
          layout (plus the shared evaluator's gold for the SuperNI rewards), prompt-only
          evaluation inputs (no test gold), the method code, scorer and decoding
          snapshots, CUDA locks and the generic Slurm script `job.sh`.
train     Stage N: `OnPolicyReplayTrainer.train()` (scripts/train.sh, ms-swift LoRA SFT on train.jsonl plus the
          buffer of stage N-1); seals the adapter.
rollout   Stage N: `OnPolicyReplayTrainer.generate()` (tools/generate_opr_ru.py, vLLM) on the stage N weights;
          seals the buffer that stage N+1 trains on.
evaluate  Stage N: greedy vLLM evaluation of the stage N weights on the base control's sparse schedule;
          per-item resumable, optionally one group subset per job.
schedule  Print the Slurm submission script for a stage window (two dependency lanes).
score     Score sealed predictions with the shared evaluator and derive stream metrics.

The method's loop is `train(); test(); generate()` per task; train N+1 needs only rollout N, so evaluation runs
off the critical path.
"""
from __future__ import annotations

import argparse
import ast
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
SCORER_SOURCE = Path("packages/evaluator/python/src")
# This package's own modules (vllm_decode) and, inside a job bundle, its hero_baselines and scorer copies.
sys.path.insert(0, str(HERE))
if (HERE / "hero_baselines").is_dir():  # inside a job bundle
    sys.path.insert(0, str(HERE / "scorer" / SCORER_SOURCE))

from hero_baselines.protocol import (  # noqa: E402
    BASE_MODEL_ID, EVAL_MAX_INPUT_TOKENS, EVAL_MAX_NEW_TOKENS, STAGE_COUNT, evaluated_task_indices,
    general_panel_kind, load_manifest, read_jsonl,
)
from vllm_decode import ENGINE  # noqa: E402

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_TREE_SHA256 = "e2361c06ba172642a18b1472f4758449bcddf20072e371921642bfba28004dc5"
CHECKPOINT_FILES = ("config.json", "merges.txt", "model.safetensors-00001-of-00002.safetensors",
                    "model.safetensors-00002-of-00002.safetensors", "model.safetensors.index.json",
                    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                    "video_preprocessor_config.json", "vocab.json")
GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")
PLAN_VERSION = "opr-official-plan-v1"
UPSTREAM_COMMIT = "1384d8823b250acf2725dc983aea6ac6e64a4283"
RHO = 0.01  # Paper setting: five slots, with the integer per-task quotas
OFFICIAL_FILES = ("core/OPRTrainer.py", "scripts/train.sh", "tools/eval.py", "tools/generate_opr_ru.py")
# Atomic job sizing for `schedule` on one A100 40GB.
TRAIN_WALLTIME = "04:00:00"
ROLLOUT_WALLTIME = "04:00:00"
EVAL_SHARD_ITEMS = 300
EVAL_SECONDS_PER_ITEM = 45
JOB_SETUP_SECONDS = 1800
BUNDLE_MODULES = ("__init__.py", "protocol.py")
SCORER_TOKENIZER = Path("experiments/data/tokenizers/gpt2/tokenizer.json")


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


def official_name(task_id: str) -> str:
    """The method data directory name: TRACE tasks keep the upstream names, SuperNI tasks become superni-NNN
    (a ':' would be read as a subset separator by ms-swift's --dataset)."""
    source, name = task_id.split(":", 1)
    return name if source == "trace" else f"{source}-{name}"


def official_dataset_lists(official: Path) -> list[list[str]]:
    """The literal `dataset_list` of the trainer and of the rollout script."""
    lists = []
    for relative in ("core/OPRTrainer.py", "tools/generate_opr_ru.py"):
        tree = ast.parse((official / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets = node.targets if isinstance(node, ast.Assign) else []
            if any(getattr(t, "attr", getattr(t, "id", None)) == "dataset_list" for t in targets):
                lists.append(ast.literal_eval(node.value))
    return lists


def scientific_plan_hash(plan: dict[str, Any]) -> str:
    """Hash of the plan without its file manifest: bundles with equal hashes run the same experiment and may
    continue one output directory."""
    return hashlib.sha256(json.dumps({k: v for k, v in plan.items() if k != "files"}, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# prepare (local)


def build_plan(dataset: Path, general: Path, official: Path, *, profile: str, stages: int, train_limit: int | None,
               eval_limit: int | None, general_limit: int | None) -> tuple[dict[str, Any], dict, dict, dict]:
    manifest = load_manifest(dataset)
    tasks = manifest["tasks"]
    names = [official_name(task["id"]) for task in tasks]
    if any(found != names for found in official_dataset_lists(official)):
        raise ValueError("The method dataset_list differs from the TRACE+17 stream")
    train_files, gold_files, eval_files, plan_stages = {}, {}, {}, []
    for task in tasks:
        spec = task["files"]["testGold"]
        if file_hash(dataset / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {spec['path']}")
        eval_files[task["id"]] = [{"id": r["id"], "prompt": r["prompt"]} for r in read_jsonl(dataset / spec["path"])]
    general_rows = read_jsonl(general / "hidden-scorer/general-test.jsonl")
    probe_ids = set(json.loads((general / "hidden-scorer/probe-ids.json").read_text()))
    for category in GENERAL_CATEGORIES:
        eval_files["general:" + category] = [{"id": r["id"], "prompt": r["prompt"]} for r in general_rows
                                             if r["category"] == category]
    for position, (task, name) in enumerate(zip(tasks, names), start=1):
        spec = task["files"]["train"]
        if file_hash(dataset / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {spec['path']}")
        rows = read_jsonl(dataset / spec["path"])
        rows = rows[:train_limit] if train_limit else rows
        # The method data/<task>/train.jsonl holds prompt and answer only (tools/data_preprocess.py).
        train_files[name] = [{"prompt": r["prompt"], "answer": r["answer"]} for r in rows]
        if task["id"].startswith("superni:"):
            gold_files[name] = [{key: r[key] for key in ("id", "prompt", "answer", "references", "metadata") if key in r}
                                | {"metric": task["metric"]} for r in rows]
        if position > stages:
            continue
        evaluations = []
        for index in evaluated_task_indices(position):
            ids = [r["id"] for r in eval_files[tasks[index]["id"]]]
            evaluations.append({"group": tasks[index]["id"], "ids": ids[:eval_limit] if eval_limit else ids})
        kind = general_panel_kind(position)
        for category in GENERAL_CATEGORIES:
            ids = [r["id"] for r in eval_files["general:" + category] if kind == "full" or r["id"] in probe_ids]
            evaluations.append({"group": "general:" + category, "panel": kind,
                                "ids": ids[:general_limit] if general_limit else ids})
        plan_stages.append({"stage": position, "taskId": task["id"], "officialName": name, "metric": task["metric"],
                            "trainRecords": len(rows), "rollout": position < STAGE_COUNT,
                            "evaluations": evaluations})
    needed = {e["group"] for s in plan_stages for e in s["evaluations"]}
    eval_files = {group: rows for group, rows in eval_files.items() if group in needed}
    plan = {
        "schemaVersion": PLAN_VERSION, "method": "opr", "paper": "arXiv:2605.29495", "profile": profile,
        "upstream": {"repository": "https://github.com/Yancey2024/OnPolicyReplay", "commit": UPSTREAM_COMMIT},
        "model": BASE_MODEL_ID, "revision": REVISION, "baseTreeSha256": BASE_TREE_SHA256,
        "dataset": {"name": manifest["name"], "seed": manifest["seed"], "manifestSha256": file_hash(dataset / "manifest.json"),
                    "trainLimit": train_limit},
        "generalManifestSha256": file_hash(general / "manifest.json"),
        "stages": plan_stages,
        "opr": {"variant": "OPR-RU", "rewardType": "ru", "rho": RHO,
                "officialSha256": {relative: file_hash(official / relative) for relative in OFFICIAL_FILES}},
        "evaluation": {"decoding": "greedy", "engine": ENGINE, "maxNewTokens": EVAL_MAX_NEW_TOKENS,
                       "maxInputTokens": EVAL_MAX_INPUT_TOKENS, "thinking": False,
                       "eos": "union(generation_config, tokenizer)", "stageZero": "the base control's formal predictions"},
    }
    return plan, train_files, gold_files, eval_files


def render_job(remote_root: str, account: str) -> str:
    """Generic MLeRP unit: one 40 GB GPU on BigCats/lion. Job name, walltime and the unit come from sbatch:
    sbatch --job-name=opr-s01-train --time=01:00:00 job.sh train --stage 1"""
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
    plan, train_files, gold_files, eval_files = build_plan(
        root / "experiments/data/trace-plus-17", root / "experiments/data/general-test/v1", HERE / "official",
        profile=args.profile, stages=args.stages, train_limit=args.train_limit, eval_limit=args.eval_limit,
        general_limit=args.general_limit)
    for name, rows in train_files.items():
        write_jsonl(bundle / "data" / name / "train.jsonl", rows)
    for name, rows in gold_files.items():
        write_jsonl(bundle / "data" / name / "gold.jsonl", rows)
    for group, rows in eval_files.items():
        write_jsonl(bundle / "inputs/eval" / group_file(group), rows)
    for relative in OFFICIAL_FILES:
        (bundle / "official" / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "official" / relative, bundle / "official" / relative)
    package = root / "packages/baselines/python/src/hero_baselines"
    (bundle / "hero_baselines").mkdir(parents=True)
    for name in BUNDLE_MODULES:
        shutil.copy2(package / name, bundle / "hero_baselines" / name)
    # The shared evaluator, laid out as in the repository so its GPT-2 tokenizer path resolves.
    shutil.copytree(root / SCORER_SOURCE / "hero_evaluation", bundle / "scorer" / SCORER_SOURCE / "hero_evaluation",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (bundle / "scorer" / SCORER_TOKENIZER).parent.mkdir(parents=True)
    shutil.copy2(root / SCORER_TOKENIZER, bundle / "scorer" / SCORER_TOKENIZER)
    shutil.copy2(Path(__file__), bundle / "run.py")
    for name in ("requirements-cuda.lock.txt", "requirements-eval.lock.txt",
                 "vllm_decode.py", "bootstrap.sh"):
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
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no implicit device fallback")
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
        if args.command == "evaluate":
            # Evaluation runs in its own vLLM environment: it must evaluate the run trained under this plan, and
            # keeps its own library contract next to its predictions.
            trained = self.output / "contract.json"
            if not trained.is_file() or {key: json.loads(trained.read_text())[key] for key in science} != science:
                raise ValueError("Evaluation needs the training run's contract under the same scientific plan")
            self.predictions = stage_dir((args.predictions_root or self.output).resolve(), args.stage)
            contract = {**science, "engine": ENGINE,
                        "libraries": {name: __import__(name).__version__ for name in ("torch", "transformers", "peft", "vllm")}}
            path = self.predictions.parent / "contract-evaluate.json"
        else:
            # train and rollout run the method scripts in one environment (ms-swift training, vLLM rollouts).
            contract = {**science, "libraries": {name: __import__(name).__version__
                                                 for name in ("torch", "transformers", "peft", "swift", "trl", "vllm")}}
            path = self.output / "contract.json"
        if path.is_file():
            if json.loads(path.read_text()) != contract:
                raise ValueError("Unit runs with a different scientific plan, checkpoint or library set")
        else:
            atomic_json(path, contract)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        self.job = {"slurmJobId": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(0)}

    def sealed_adapter(self, position: int) -> Path:
        directory = stage_dir(self.output, position)
        record = directory / "training.json"
        if not record.is_file():
            raise ValueError(f"Stage {position} is not trained")
        if tree_hash(directory / "adapter") != json.loads(record.read_text())["adapterSha256"]:
            raise ValueError(f"Stage {position} adapter hash mismatch")
        return directory / "adapter"

    def model(self, adapter: Path, *, trainable: bool):
        """The base checkpoint with the stage's LoRA merged into its bf16 weights, for the vLLM rollouts and
        evaluation (training is ms-swift's, in train.sh)."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText

        if trainable:
            raise ValueError("train.sh trains; the runner only loads stage weights for inference")
        base = AutoModelForImageTextToText.from_pretrained(self.checkpoint, dtype=torch.bfloat16,
                                                           local_files_only=True).to("cuda")
        configured = base.generation_config.eos_token_id
        self.eos = sorted(set(([configured] if isinstance(configured, int) else list(configured or []))
                              + [self.tokenizer.eos_token_id]))
        return PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)

    def workspace(self) -> Path:
        """The method's repository layout in node-local scratch: core/, scripts/, tools/ and data/<task>/, with
        its scripts run from the root."""
        work = self.checkpoint.parent / "workspace"
        shutil.rmtree(work, ignore_errors=True)
        shutil.copytree(self.bundle / "official", work)
        shutil.copytree(self.bundle / "data", work / "data")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(self.bundle / "scorer" / SCORER_SOURCE), *filter(None, [os.environ.get("PYTHONPATH")])])
        os.chdir(work)
        sys.path.insert(0, str(work / "core"))
        return work

    def trainer(self, model_path: Path, work: Path):
        from OPRTrainer import OnPolicyReplayTrainer

        trainer = OnPolicyReplayTrainer(str(model_path), str(work / "output"), self.plan["opr"]["rewardType"],
                                        self.plan["opr"]["rho"])
        trainer.task_id = self.stage["stage"] - 1
        return trainer


def train_unit(args) -> None:
    """OnPolicyReplayTrainer.train() for task N: the LoRA SFT on train.jsonl plus buffer.jsonl."""
    unit = Unit(args)
    position, directory = args.stage, unit.dir
    if (directory / "training.json").is_file():
        unit.sealed_adapter(position)
        print(json.dumps({"stage": position, "phase": "train", "status": "already sealed"}), flush=True)
        return
    work = unit.workspace()
    trainer = unit.trainer(unit.checkpoint, work)
    buffer_sha = None
    if position > 1:
        previous = stage_dir(unit.output, position - 1)
        if not (previous / "replay.json").is_file():
            raise ValueError(f"Stage {position - 1} replay buffer is not sealed")
        buffer_sha = json.loads((previous / "replay.json").read_text())["bufferSha256"]
        if file_hash(previous / "buffer.jsonl") != buffer_sha:
            raise ValueError(f"Stage {position - 1} replay buffer changed")
        shutil.copy2(previous / "buffer.jsonl", work / "data" / unit.stage["officialName"] / "buffer.jsonl")
        trainer.adapter_path = str(unit.sealed_adapter(position - 1))
    started = time.perf_counter()
    trainer.train()
    seconds = time.perf_counter() - started
    checkpoint = Path(trainer.adapter_path)
    shutil.rmtree(directory / "adapter.tmp", ignore_errors=True)
    shutil.copytree(checkpoint, directory / "adapter.tmp")
    os.replace(directory / "adapter.tmp", directory / "adapter")
    # ms-swift's run record: arguments and the per-step training log of the versioned output directory.
    for name in ("args.json", "logging.jsonl"):
        if (checkpoint.parent / name).is_file():
            shutil.copy2(checkpoint.parent / name, directory / f"swift-{name}")
    atomic_json(directory / "training.json", {
        "stage": position, "checkpoint": checkpoint.name, "bufferSha256": buffer_sha, "seconds": seconds,
        "job": unit.job, "adapterSha256": tree_hash(directory / "adapter")})
    print(json.dumps({"stage": position, "phase": "train", "checkpoint": checkpoint.name, "seconds": seconds}), flush=True)


def rollout_unit(args) -> None:
    """OnPolicyReplayTrainer.generate() after task N: the rollouts, rewards and buffer for task N+1."""
    from vllm_decode import export_checkpoint, release_cuda

    unit = Unit(args)
    position, directory = args.stage, unit.dir
    if not unit.stage["rollout"]:
        raise ValueError(f"Stage {position} builds no replay buffer")
    if (directory / "replay.json").is_file():
        print(json.dumps({"stage": position, "phase": "rollout", "status": "already sealed"}), flush=True)
        return
    work = unit.workspace()
    model = unit.model(unit.sealed_adapter(position), trainable=False)
    merged = export_checkpoint(model, unit.checkpoint, work / "merged")
    del model
    release_cuda()  # the rollout script starts its own vLLM engine
    trainer = unit.trainer(merged, work)
    started = time.perf_counter()
    trainer.generate()
    seconds = time.perf_counter() - started
    shutil.copy2(work / "data" / trainer.dataset_list[position] / "buffer.jsonl", directory / "buffer.jsonl")
    rows = read_jsonl(directory / "buffer.jsonl")
    source = {}
    for name in trainer.dataset_list[:position]:
        for row in read_jsonl(work / "data" / name / "train.jsonl"):
            source.setdefault(row["prompt"], name)
    atomic_json(directory / "replay.json", {
        "stage": position, "trainingSha256": file_hash(directory / "training.json"), "bufferSize": trainer.buffer_size,
        "perTaskQuota": int(trainer.buffer_size / (trainer.task_id + 1)), "replaySize": len(rows),
        "replayTasks": [source.get(row["prompt"]) for row in rows], "seconds": seconds, "job": unit.job,
        "bufferSha256": file_hash(directory / "buffer.jsonl")})
    print(json.dumps({"stage": position, "phase": "rollout", "status": "sealed", "replay": len(rows),
                      "seconds": seconds}), flush=True)


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
        evaluate_items(unit.model(unit.sealed_adapter(stage["stage"]), trainable=False).eval(), unit.checkpoint,
                       unit.tokenizer, unit.eos, pending, atomic_json, max_input_tokens=EVAL_MAX_INPUT_TOKENS,
                       max_new_tokens=EVAL_MAX_NEW_TOKENS)
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
# schedule (local): the Slurm chain for a stage window


def seconds_to_walltime(seconds: int) -> str:
    seconds = min(max(seconds, 3600), 24 * 3600 - 60)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:00"


def stage_units(plan: dict[str, Any], position: int) -> list[dict[str, Any]]:
    """Atomic units of one stage: train, evaluation shards (groups packed to at most EVAL_SHARD_ITEMS items)
    and the rollout."""
    stage = plan["stages"][position - 1]
    units = [{"lane": "critical", "phase": "train", "args": [], "time": TRAIN_WALLTIME}]
    shards, current, items = [], [], 0
    for evaluation in stage["evaluations"]:
        if current and items + len(evaluation["ids"]) > EVAL_SHARD_ITEMS:
            shards.append((current, items))
            current, items = [], 0
        current.append(evaluation["group"])
        items += len(evaluation["ids"])
    shards.append((current, items))
    for groups, count in shards:
        units.append({"lane": "evaluate", "phase": "evaluate", "args": ["--groups", ",".join(groups)],
                      "time": seconds_to_walltime(JOB_SETUP_SECONDS + count * EVAL_SECONDS_PER_ITEM)})
    if stage["rollout"] and position < len(plan["stages"]):
        units.append({"lane": "critical", "phase": "rollout", "args": [], "time": ROLLOUT_WALLTIME})
    return units


def schedule(args) -> None:
    """Print a login-node bash script submitting stages first..last as two dependency lanes: the critical lane
    (train N, rollout N, train N+1, ...) and the evaluation lane (shards of every stage in order, each after its
    stage's train). Failed critical units cancel their dependants."""
    plan = json.loads((args.bundle.resolve() / "plan.json").read_text())
    skip = set(args.skip.split(",")) if args.skip else set()
    lines = ["#!/bin/bash", "set -euo pipefail", f"cd {args.remote_root.rstrip('/')}/bundle",
             "submit() { local name=$1 time=$2 dependency=$3; shift 3",
             '  sbatch --parsable --job-name="$name" --time="$time" ${dependency:+--dependency=$dependency} '
             '--kill-on-invalid-dep=yes job.sh "$@"; }',
             f"critical={args.after_critical or ''}", f"evaluate={args.after_evaluate or ''}"]
    for position in range(args.first, args.last + 1):
        for index, unit in enumerate(stage_units(plan, position)):
            name = f"opr-{plan['profile']}-s{position:02d}-{unit['phase']}" + (f"-{index:02d}" if unit["phase"] == "evaluate" else "")
            if name in skip:  # sealed, or still running from an earlier submission
                continue
            command = " ".join([unit["phase"], "--stage", str(position), *unit["args"]])
            if unit["lane"] == "critical":
                dependency = '${critical:+afterok:$critical}'
                lines.append(f'critical=$(submit {name} {unit["time"]} "{dependency}" {command})')
                if unit["phase"] == "train":
                    lines.append("train=$critical")
            else:
                dependency = 'afterok:$train${evaluate:+,afterany:$evaluate}'
                lines.append(f'evaluate=$(submit {name} {unit["time"]} "{dependency}" {command})')
            lines.append(f'echo "{name} ${{{"critical" if unit["lane"] == "critical" else "evaluate"}}} '
                         f'{unit["time"]} {command}"')
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# score (local)


def score(args) -> None:
    from hero_baselines.protocol import canonical_score, stream_metrics

    # TRACE ROUGE-L (the `rouge` package) reconstructs its LCS recursively, one frame per token of prediction
    # plus reference; long MeetingBank predictions exceed Python's default depth. The score is unchanged.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))

    root = Path(args.repository).resolve()
    run = args.run.resolve()
    plan = json.loads((args.bundle.resolve() / "plan.json").read_text())
    dataset = root / "experiments/data/trace-plus-17"
    manifest = load_manifest(dataset)
    if file_hash(dataset / "manifest.json") != plan["dataset"]["manifestSha256"]:
        raise ValueError("Dataset manifest differs from the plan")
    gold = {t["id"]: {r["id"]: r for r in read_jsonl(dataset / t["files"]["testGold"]["path"])} for t in manifest["tasks"]}
    metrics_by_task = {t["id"]: t["metric"] for t in manifest["tasks"]}
    general = {r["id"]: r for r in read_jsonl(root / "experiments/data/general-test/v1/hidden-scorer/general-test.jsonl")}
    cells, general_cells, replay, failures, length_stops = [], [], [], 0, 0
    for stage in plan["stages"]:
        directory = stage_dir(run, stage["stage"])
        if (directory / "replay.json").is_file():
            sealed = json.loads((directory / "replay.json").read_text())
            replay.append({"stage": stage["stage"], "nextReplaySize": sealed["replaySize"],
                           "replayTasks": sealed["replayTasks"]})
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
    order = [t["id"] for t in manifest["tasks"]]
    metrics: dict[str, Any] = {**stream_metrics(cells, order), "itemFailures": failures, "lengthStops": length_stops}
    by_stage = {}
    for cell in general_cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    metrics["generalByStage"] = [{"stage": s, "panel": rows[0]["panel"],
                                  "macro": sum(r["score"] for r in rows) / len(rows)}
                                 for s, rows in sorted(by_stage.items()) if len(rows) == len(GENERAL_CATEGORIES)]
    report = {"method": "opr", "profile": plan["profile"], "rho": plan["opr"]["rho"], "run": str(run),
              "planSha256": file_hash(args.bundle.resolve() / "plan.json"),
              "taskCells": cells, "generalCells": general_cells, "replay": replay, "metrics": metrics}
    atomic_json(args.output, report)
    print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--repository", default=".")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", choices=("smoke", "full"), required=True)
    p.add_argument("--stages", type=int, default=STAGE_COUNT)
    p.add_argument("--train-limit", type=int, help="Smoke only: first N training records per task")
    p.add_argument("--eval-limit", type=int, help="Smoke only: first N test items per task")
    p.add_argument("--general-limit", type=int, help="Smoke only: first N G items per category")
    p.add_argument("--remote-root", help="Render job.sh for this MLeRP directory")
    p.add_argument("--account", help="Slurm account; required with --remote-root")
    for name in ("train", "evaluate", "rollout"):
        u = commands.add_parser(name)
        u.add_argument("--stage", type=int, required=True)
        u.add_argument("--bundle", type=Path, default=HERE)
        u.add_argument("--output", type=Path, required=True)
        u.add_argument("--checkpoint", type=Path, required=True)
        u.add_argument("--fetch", action="store_true", help="Download the pinned checkpoint first")
        if name == "evaluate":
            u.add_argument("--groups", help="Comma-separated evaluation groups for this job (default: all)")
            u.add_argument("--predictions-root", type=Path,
                           help="Write predictions here instead of the run's output (re-evaluating a sealed run)")
    c = commands.add_parser("schedule")
    c.add_argument("--bundle", type=Path, required=True)
    c.add_argument("--remote-root", required=True)
    c.add_argument("--first", type=int, required=True)
    c.add_argument("--last", type=int, required=True)
    c.add_argument("--after-critical", help="Job the first train waits for (the previous window's last rollout)")
    c.add_argument("--after-evaluate", help="Job the first evaluation shard queues behind")
    c.add_argument("--skip", help="Comma-separated unit job names not to submit (sealed or running)")
    s = commands.add_parser("score")
    s.add_argument("--repository", default=".")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare" and args.remote_root and not args.account:
        parser.error("--account is required with --remote-root")
    if args.command == "prepare":
        if args.profile == "full" and (args.stages != STAGE_COUNT or args.train_limit or args.eval_limit
                                       or args.general_limit):
            parser.error("The full profile uses the registered stream and complete panels")
        prepare(args)
    elif args.command == "score":
        score(args)
    elif args.command == "schedule":
        schedule(args)
    else:
        {"train": train_unit, "evaluate": evaluate_unit, "rollout": rollout_unit}[args.command](args)


if __name__ == "__main__":
    main()
