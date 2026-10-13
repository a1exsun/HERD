"""OPHSD (arXiv:2605.08741) over the TRACE+17 stream: prepare a job bundle,
train on GPU, score locally.

prepare  Build a self-contained bundle: plan, stage training records, prompt-only
         evaluation inputs (no gold), code snapshot and CUDA job files.
train    Run the bundle on a GPU: per stage, OPHSD with a fresh LoRA whose frozen
         teacher is the stage-initial checkpoint, then merge; resumes from sealed
         adapters and mid-stage checkpoints. Training jobs do not evaluate.
evaluate Greedy vLLM evaluation of one shard of one trained stage on the base control's
         sparse schedule, rebuilding the stage checkpoint from its adapter chain.
score    Score predictions with the shared evaluator and derive stream metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
# This package's own modules (vllm_decode) and, inside a job bundle, its hero_baselines copy.
sys.path.insert(0, str(HERE))

from hero_baselines.controls import LoraControlConfig  # noqa: E402
from hero_baselines.ophsd import (  # noqa: E402
    EMBEDDING_MODEL, EMBEDDING_REVISION, EMBEDDING_TREE_SHA256, EXECUTION_ONLY_FIELDS, UPSTREAM_COMMIT, OphsdConfig,
    StageProgress, select_harness, summary_dict, train_stage,
)
from vllm_decode import ENGINE  # noqa: E402
from hero_baselines.protocol import (  # noqa: E402
    BASE_MODEL_ID, EVAL_MAX_INPUT_TOKENS, EVAL_MAX_NEW_TOKENS, STAGE_COUNT, evaluated_task_indices,
    general_panel_kind, load_manifest, read_jsonl,
)

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_TREE_SHA256 = "e2361c06ba172642a18b1472f4758449bcddf20072e371921642bfba28004dc5"
CHECKPOINT_FILES = ("config.json", "merges.txt", "model.safetensors-00001-of-00002.safetensors",
                    "model.safetensors-00002-of-00002.safetensors", "model.safetensors.index.json",
                    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                    "video_preprocessor_config.json", "vocab.json")
EMBEDDING_FILES = ("config.json", "model.safetensors", "special_tokens_map.json", "tokenizer.json",
                   "tokenizer_config.json", "vocab.txt")
GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")
PLAN_VERSION = "ophsd-plan-v2"
# A stage's training is not started with less job time than this; training checkpoints
# every few updates and before the deadline, evaluation resumes per item.
STAGE_START_RESERVE_SECONDS = 3600
BUNDLE_MODULES = ("__init__.py", "controls.py", "ophsd.py", "protocol.py", "sft.py")


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def semantic_plan_digest(plan: dict[str, Any]) -> str:
    """Digest of what a run means: everything except bundle files and execution-only settings."""
    meaning = {key: value for key, value in plan.items() if key != "files"}
    meaning["ophsd"] = {key: value for key, value in plan["ophsd"].items() if key not in EXECUTION_ONLY_FIELDS}
    return hashlib.sha256(json.dumps(meaning, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def truncate_log(path: Path, next_update: int) -> None:
    """Drop log rows from updates at or after the checkpoint (they are re-run)."""
    if path.is_file():
        rows = [line for line in path.read_text(encoding="utf-8").splitlines()
                if line and json.loads(line)["update"] < next_update]
        path.write_text("".join(row + "\n" for row in rows), encoding="utf-8")


def group_file(group: str) -> str:
    return group.replace(":", "__") + ".jsonl"


# ---------------------------------------------------------------------------
# prepare (local)


def build_plan(dataset: Path, general: Path, *, profile: str, stages: int, updates: int, batch_size: int,
               seed: int = 42, eval_limit: int | None, general_limit: int | None,
               learning_rate: float | None = None) -> tuple[dict[str, Any], dict[str, list], dict[str, list]]:
    manifest = load_manifest(dataset)
    lora = LoraControlConfig(seed=seed)
    train_files, eval_files, plan_stages = {}, {}, []
    tasks = manifest["tasks"]
    for task in tasks:
        spec = task["files"]["testGold"]
        if file_hash(dataset / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {spec['path']}")
        rows = read_jsonl(dataset / spec["path"])
        eval_files[task["id"]] = [{"id": r["id"], "prompt": r["prompt"]} for r in rows]
    general_rows = read_jsonl(general / "hidden-scorer/general-test.jsonl")
    probe_ids = set(json.loads((general / "hidden-scorer/probe-ids.json").read_text()))
    for category in GENERAL_CATEGORIES:
        eval_files["general:" + category] = [{"id": r["id"], "prompt": r["prompt"]} for r in general_rows
                                             if r["category"] == category]
    for position in range(1, stages + 1):
        task = tasks[position - 1]
        spec = task["files"]["train"]
        if file_hash(dataset / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {spec['path']}")
        records = [{"id": r["id"], "prompt": r["prompt"], "answer": r["answer"],
                    **({"references": r["references"]} if "references" in r else {})}
                   for r in read_jsonl(dataset / spec["path"])]
        train_files[f"{position:02d}"] = records
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
                            "harness": select_harness(records), "trainRecords": len(records),
                            "evaluations": evaluations})
    needed = {e["group"] for s in plan_stages for e in s["evaluations"]}
    eval_files = {group: rows for group, rows in eval_files.items() if group in needed}
    plan = {
        "schemaVersion": PLAN_VERSION, "method": "ophsd", "paper": "arXiv:2605.08741", "profile": profile,
        "upstream": {"repository": "https://github.com/zzy1127/OPHSD-On-Policy-Harness-Self-Distillation",
                     "commit": UPSTREAM_COMMIT},
        "model": BASE_MODEL_ID, "revision": REVISION, "baseTreeSha256": BASE_TREE_SHA256,
        "embedding": {"model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION, "treeSha256": EMBEDDING_TREE_SHA256},
        "dataset": {"name": manifest["name"], "seed": manifest["seed"], "manifestSha256": file_hash(dataset / "manifest.json")},
        "generalManifestSha256": file_hash(general / "manifest.json"),
        "stages": plan_stages,
        "training": {"updates": updates, "batchSize": batch_size,
                     "learningRate": lora.learning_rate if learning_rate is None else learning_rate,
                     "gradientClip": lora.gradient_clip, "weightDecay": OphsdConfig().weight_decay, "seed": lora.seed,
                     "records": "verl's train sampler: one torch.randperm per epoch from a generator seeded with "
                                "seed + stage, incomplete last batch dropped",
                     "loraInit": "transformers.set_seed(seed + stage) before the fresh LoRA is created",
                     "lora": {"rank": lora.rank, "alpha": lora.alpha, "dropout": lora.dropout,
                              "targets": "text-backbone Linear modules (no vision, MTP or LM head)",
                              "mergeAfterStage": True}},
        "ophsd": OphsdConfig().__dict__,
        "evaluation": {"decoding": "greedy", "engine": ENGINE, "maxNewTokens": EVAL_MAX_NEW_TOKENS, "maxInputTokens": EVAL_MAX_INPUT_TOKENS,
                       "thinking": False, "eos": "union(generation_config, tokenizer)",
                       "stageZero": "the base control's formal predictions"},
    }
    return plan, train_files, eval_files


def render_job(args, command: str) -> str:
    """MLeRP batch script on one 40 GB GPU (BigCats/lion); bundle at <remote>/bundle, output at <remote>/output.

    `train` trains stages in order (walltime `--walltime`); `evaluate` evaluates one shard of one
    trained stage with vLLM, given by OPHSD_STAGE and OPHSD_SHARD (i/n).
    """
    remote = args.remote_root.rstrip("/")
    header = ["#!/bin/bash", f"#SBATCH --job-name=ophsd-{args.profile}-{command}", f"#SBATCH --account={args.account}",
              "#SBATCH --partition=BigCats", "#SBATCH --qos=lion", "#SBATCH --gres=gpu:40gb:1", "#SBATCH --ntasks=1",
              "#SBATCH --cpus-per-task=12", "#SBATCH --mem=96G",
              f"#SBATCH --time={args.walltime if command == 'train' else args.eval_walltime}",
              f"#SBATCH --output={remote}/logs/%j.out", f"#SBATCH --error={remote}/logs/%j.err", "set -euo pipefail"]
    if command == "train":
        split = f"OPHSD_SPLIT_AT={args.split_at} " if args.split_at else ""
        body = ['deadline=$(date -d "$(squeue -h -j "$SLURM_JOB_ID" -o %e)" +%s)',
                f"{split}bash {remote}/bundle/bootstrap.sh {remote}/bundle {remote}/output train "
                f"--deadline \"$deadline\""]
    else:
        body = [f"bash {remote}/bundle/bootstrap.sh {remote}/bundle {remote}/output evaluate "
                '--stage "$OPHSD_STAGE" --shard "$OPHSD_SHARD"']
    return "\n".join(header + body + [""])


def prepare(args) -> None:
    root = Path(args.repository).resolve()
    bundle = args.output.resolve()
    if bundle.exists():
        raise SystemExit(f"Bundle exists: {bundle}; use a new path")
    plan, train_files, eval_files = build_plan(
        root / "experiments/data/trace-plus-17", root / "experiments/data/general-test/v1",
        profile=args.profile, stages=args.stages, updates=args.updates, batch_size=args.batch_size, seed=args.seed,
        eval_limit=args.eval_limit, general_limit=args.general_limit, learning_rate=args.learning_rate)
    for key, rows in train_files.items():
        write_jsonl(bundle / "inputs/train" / f"{key}.jsonl", rows)
    for group, rows in eval_files.items():
        write_jsonl(bundle / "inputs/eval" / group_file(group), rows)
    package = root / "packages/baselines/python/src/hero_baselines"
    (bundle / "hero_baselines").mkdir(parents=True)
    for name in BUNDLE_MODULES:
        shutil.copy2(package / name, bundle / "hero_baselines" / name)
    shutil.copytree(package / "ophsd_official", bundle / "hero_baselines" / "ophsd_official",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(Path(__file__), bundle / "run.py")
    for name in ("requirements-cuda.lock.txt", "requirements-kernels.lock.txt", "requirements-eval.lock.txt",
                 "vllm_decode.py", "bootstrap.sh"):
        shutil.copy2(HERE / name, bundle / name)
    if args.remote_root:
        (bundle / "job.sh").write_text(render_job(args, "train"), encoding="utf-8")
        (bundle / "job-evaluate.sh").write_text(render_job(args, "evaluate"), encoding="utf-8")
    plan["files"] = {path.relative_to(bundle).as_posix(): file_hash(path)
                     for path in sorted(bundle.rglob("*")) if path.is_file()}
    atomic_json(bundle / "plan.json", plan)
    print(json.dumps({"bundle": str(bundle), "stages": len(plan["stages"]),
                      "harnesses": [s["harness"] for s in plan["stages"]],
                      "planSha256": file_hash(bundle / "plan.json")}))


# ---------------------------------------------------------------------------
# train (GPU)


def fetch(repository: str, revision: str, files, destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    destination.mkdir(parents=True, exist_ok=True)
    for name in files:
        if not (destination / name).is_file():
            hf_hub_download(repository, name, revision=revision, local_dir=destination)
    shutil.rmtree(destination / ".cache", ignore_errors=True)


def verify_bundle(bundle: Path, plan: dict[str, Any]) -> None:
    for relative, expected in plan["files"].items():
        if file_hash(bundle / relative) != expected:
            raise ValueError(f"Bundle file changed: {relative}")


def embedder(path: Path, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()

    def embed(texts: list[str]) -> list[list[float]]:
        """The `Embedder` (sentence-transformers, normalize_embeddings=True): the model's
        sentence_bert_config lowercases (do_lower_case) and truncates at 512; CLS pooling; L2 norm."""
        texts = [text.lower() for text in texts]
        vectors = []
        for start in range(0, len(texts), 32):
            encoded = tokenizer(texts[start:start + 32], padding=True, truncation=True, max_length=512,
                                return_tensors="pt").to(device)
            with torch.inference_mode():
                cls = model(**encoded).last_hidden_state[:, 0]
            vectors.extend(torch.nn.functional.normalize(cls.float(), dim=-1).cpu().tolist())
        return vectors

    return embed


def evaluate_groups(model, checkpoint: Path, tokenizer, bundle: Path, stage_dir: Path, stage: dict[str, Any],
                    evaluations, eos, label: str) -> None:
    """Greedy vLLM predictions for the given panels of a stage (see vllm_decode); cached items are reused."""
    from vllm_decode import evaluate_items

    started = time.perf_counter()
    pending = []
    for evaluation in evaluations:
        group = evaluation["group"]
        prompts = {r["id"]: r["prompt"] for r in read_jsonl(bundle / "inputs/eval" / group_file(group))}
        items = stage_dir / "items" / group.replace(":", "__")
        items.mkdir(parents=True, exist_ok=True)
        for index, identifier in enumerate(evaluation["ids"]):
            path = items / f"{index:04d}.json"
            if path.is_file():
                if json.loads(path.read_text())["id"] != identifier:
                    raise ValueError(f"Cached prediction does not match the plan: {path}")
            else:
                pending.append((path, {"id": identifier, "group": group, "stage": stage["stage"]}, prompts[identifier]))
    if pending:
        evaluate_items(model.eval(), checkpoint, tokenizer, eos, pending, atomic_json,
                       max_input_tokens=EVAL_MAX_INPUT_TOKENS, max_new_tokens=EVAL_MAX_NEW_TOKENS)
    print(json.dumps({"stage": stage["stage"], "groups": [e["group"] for e in evaluations], "decoded": len(pending)}),
          flush=True)
    seconds = stage_dir / "evaluation-seconds" / f"{label}.json"
    previous = json.loads(seconds.read_text())["seconds"] if seconds.is_file() else 0.0
    atomic_json(seconds, {"seconds": previous + time.perf_counter() - started})


def finalize_stage(stage_dir: Path, stage: dict[str, Any]) -> bool:
    """Seal the stage's predictions once every planned item exists; False while any is missing."""
    hashes = {}
    for evaluation in stage["evaluations"]:
        items = stage_dir / "items" / evaluation["group"].replace(":", "__")
        paths = [items / f"{index:04d}.json" for index in range(len(evaluation["ids"]))]
        if not all(path.is_file() for path in paths):
            return False
        rows = [json.loads(path.read_text()) for path in paths]
        if [row["id"] for row in rows] != evaluation["ids"]:
            raise ValueError(f"Stage {stage['stage']} {evaluation['group']} predictions do not match the plan")
        output = stage_dir / "predictions" / group_file(evaluation["group"])
        write_jsonl(output, rows)
        hashes[output.relative_to(stage_dir).as_posix()] = file_hash(output)
    seconds = sum(json.loads(path.read_text())["seconds"] for path in (stage_dir / "evaluation-seconds").glob("*.json"))
    atomic_json(stage_dir / "complete.json", {"stage": stage["stage"], "engine": ENGINE, "predictions": hashes,
                                              "evaluationSeconds": seconds,
                                              "trainingSha256": file_hash(stage_dir / "training.json")})
    return True


def load_run(args, torch):
    """Verify bundle and pinned models, then load the base model and tokenizer for this run."""
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    from hero_baselines.ophsd import OphsdConfig as Config

    bundle, output = args.bundle.resolve(), args.output.resolve()
    plan = json.loads((bundle / "plan.json").read_text())
    if plan["schemaVersion"] != PLAN_VERSION:
        raise ValueError("Unsupported plan")
    verify_bundle(bundle, plan)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no implicit device fallback")
    checkpoint, embedding = args.checkpoint.resolve(), args.embedding.resolve()
    if args.fetch:
        fetch(BASE_MODEL_ID, REVISION, CHECKPOINT_FILES, checkpoint)
        fetch(EMBEDDING_MODEL, EMBEDDING_REVISION, EMBEDDING_FILES, embedding)
    if tree_hash(checkpoint) != plan["baseTreeSha256"]:
        raise ValueError("Base checkpoint differs from the pinned tree hash")
    if tree_hash(embedding) != plan["embedding"]["treeSha256"]:
        raise ValueError("Embedding model differs from the pinned tree hash")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                                        local_files_only=True).to(args.device)
    model.requires_grad_(False)
    configured = model.generation_config.eos_token_id
    eos = sorted(set(([configured] if isinstance(configured, int) else list(configured or [])) + [tokenizer.eos_token_id]))
    return bundle, output, plan, Config(**plan["ophsd"]), tokenizer, model, eos, embedding


def evaluate(args) -> None:
    """Evaluate one shard of one trained stage, rebuilding its checkpoint from the sealed adapter chain."""
    import torch
    from peft import PeftModel

    bundle, output, plan, config, tokenizer, model, eos, _ = load_run(args, torch)
    contract = json.loads((output / "contract.json").read_text())
    if contract["semanticPlanSha256"] != semantic_plan_digest(plan):
        raise ValueError("Evaluation bundle differs from the run's scientific plan")
    # Evaluation runs in its own vLLM environment and records it beside the training contract.
    evaluation_contract = {"semanticPlanSha256": contract["semanticPlanSha256"], "engine": ENGINE,
                           "libraries": {name: __import__(name).__version__ for name in ("torch", "transformers", "peft", "vllm")}}
    path = output / "contract-evaluate.json"
    if path.is_file() and json.loads(path.read_text()) != evaluation_contract:
        raise ValueError("Evaluation runs with a different plan or library set")
    atomic_json(path, evaluation_contract)
    index, count = (int(part) for part in args.shard.split("/"))
    stage = plan["stages"][args.stage - 1]
    stage_dir = output / f"stage-{args.stage:02d}"
    if (stage_dir / "complete.json").is_file():
        print(json.dumps({"stage": args.stage, "alreadyComplete": True}), flush=True)
        return
    for previous in plan["stages"][:args.stage]:
        directory = output / f"stage-{previous['stage']:02d}"
        recorded = json.loads((directory / "training.json").read_text())
        if tree_hash(directory / "adapter") != recorded["adapterSha256"]:
            raise ValueError(f"Stage {previous['stage']} adapter hash mismatch")
        model = PeftModel.from_pretrained(model, directory / "adapter").merge_and_unload(safe_merge=True)
    model.requires_grad_(False)
    mine = [evaluation for position, evaluation in enumerate(stage["evaluations"]) if position % count == index]
    evaluate_groups(model, args.checkpoint.resolve(), tokenizer, bundle, stage_dir, stage, mine, eos, f"shard-{index}-of-{count}")
    print(json.dumps({"stage": args.stage, "shard": args.shard, "sealed": finalize_stage(stage_dir, stage)}), flush=True)


def train(args) -> None:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import set_seed

    from hero_baselines.sft import text_backbone_lora_targets

    bundle, output, plan, config, tokenizer, model, eos, embedding = load_run(args, torch)
    output.mkdir(parents=True, exist_ok=True)
    # What a resumed run must share with the run it continues; execution-only settings may differ.
    contract = {"semanticPlanSha256": semantic_plan_digest(plan), "baseTreeSha256": plan["baseTreeSha256"],
                "embeddingTreeSha256": plan["embedding"]["treeSha256"],
                "libraries": {name: __import__(name).__version__ for name in ("torch", "transformers", "peft")}}
    contract_path = output / "contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Resume with a different scientific plan, checkpoint, embedding model or library set")
    else:
        atomic_json(contract_path, contract)

    training = plan["training"]
    embed = embedder(embedding, args.device)
    through = args.through_stage or len(plan["stages"])

    for stage in plan["stages"][:through]:
        position = stage["stage"]
        stage_dir = output / f"stage-{position:02d}"
        adapter, saved = stage_dir / "adapter", stage_dir / "checkpoint"
        trained = (stage_dir / "complete.json").is_file() or (stage_dir / "training.json").is_file()
        if not trained and args.deadline and time.time() + STAGE_START_RESERVE_SECONDS > args.deadline:
            print(json.dumps({"stage": position, "deferred": "insufficient job time"}), flush=True)
            break
        if trained:
            recorded = json.loads((stage_dir / "training.json").read_text())
            if tree_hash(adapter) != recorded["adapterSha256"]:
                raise ValueError(f"Stage {position} adapter hash mismatch")
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload(safe_merge=True)
            model.requires_grad_(False)
            print(json.dumps({"stage": position, "rebuilt": True}), flush=True)
        else:
            records = read_jsonl(bundle / "inputs/train" / f"{position:02d}.jsonl")
            if select_harness(records) != stage["harness"]:
                raise ValueError("Harness selection differs from the plan")
            progress, optimizer_state = None, None
            if (saved / "state.json").is_file():
                state = json.loads((saved / "state.json").read_text())
                if tree_hash(saved / "adapter") != state["adapterSha256"] or \
                        file_hash(saved / "optimizer.pt") != state["optimizerSha256"]:
                    raise ValueError(f"Stage {position} checkpoint hash mismatch")
                model = PeftModel.from_pretrained(model, saved / "adapter", is_trainable=True)
                optimizer_state = torch.load(saved / "optimizer.pt", map_location=args.device)
                progress = StageProgress(state["nextUpdate"], state["losses"], state["seconds"],
                                         state["skippedSamples"], state["skippedSteps"])
                truncate_log(stage_dir / "training-log.jsonl", progress.next_update)
                print(json.dumps({"stage": position, "resumedAtUpdate": progress.next_update}), flush=True)
            else:
                if stage_dir.exists():
                    shutil.rmtree(stage_dir)  # interrupted before any checkpoint
                stage_dir.mkdir(parents=True)
                set_seed(training["seed"] + position)  # the fresh LoRA's initialisation draws
                model = get_peft_model(model, LoraConfig(
                    r=training["lora"]["rank"], lora_alpha=training["lora"]["alpha"],
                    lora_dropout=training["lora"]["dropout"], bias="none",
                    target_modules=text_backbone_lora_targets(model), task_type="CAUSAL_LM"))
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            def save_checkpoint(state: StageProgress, optimizer_state_dict) -> None:
                staging = stage_dir / "checkpoint.tmp"
                shutil.rmtree(staging, ignore_errors=True)
                model.save_pretrained(staging / "adapter", safe_serialization=True)
                torch.save(optimizer_state_dict, staging / "optimizer.pt")
                atomic_json(staging / "state.json", {"nextUpdate": state.next_update, "losses": state.losses,
                                                     "seconds": state.seconds, "skippedSamples": state.skipped_samples,
                                                     "skippedSteps": state.skipped_steps,
                                                     "adapterSha256": tree_hash(staging / "adapter"),
                                                     "optimizerSha256": file_hash(staging / "optimizer.pt")})
                shutil.rmtree(saved, ignore_errors=True)
                os.replace(staging, saved)
                print(json.dumps({"stage": position, "checkpointAtUpdate": state.next_update}), flush=True)

            log_handle = (stage_dir / "training-log.jsonl").open("a", encoding="utf-8", buffering=1)
            summary = train_stage(model, tokenizer, stage=position, task_id=stage["taskId"], records=records,
                                  kind=stage["harness"], embed=embed,
                                  updates=training["updates"], batch_size=training["batchSize"],
                                  learning_rate=training["learningRate"], gradient_clip=training["gradientClip"],
                                  seed=training["seed"], eos=eos, config=config,
                                  log=lambda row: log_handle.write(json.dumps(row, ensure_ascii=False) + "\n"),
                                  progress=progress, optimizer_state=optimizer_state,
                                  save_checkpoint=save_checkpoint, deadline=args.deadline)
            log_handle.close()
            if summary is None:
                print(json.dumps({"stage": position, "deferred": "job deadline inside the stage"}), flush=True)
                break
            model.gradient_checkpointing_disable()
            staging = stage_dir / "adapter.tmp"
            model.save_pretrained(staging, safe_serialization=True)
            os.replace(staging, adapter)
            atomic_json(stage_dir / "training.json", {**summary_dict(summary), "adapterSha256": tree_hash(adapter),
                                                      "trainingLogSha256": file_hash(stage_dir / "training-log.jsonl")})
            shutil.rmtree(saved, ignore_errors=True)
            model = model.merge_and_unload(safe_merge=True)
            model.requires_grad_(False)
            print(json.dumps({"stage": position, "trained": summary.mean_loss, "seconds": summary.seconds}), flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    done = sum((output / f"stage-{s['stage']:02d}" / "complete.json").is_file() for s in plan["stages"])
    atomic_json(output / "status.json", {"status": "complete" if done == len(plan["stages"]) else "partial",
                                         "completedStages": done, "plannedStages": len(plan["stages"]),
                                         "updatedAt": time.time()})


# ---------------------------------------------------------------------------
# score (local)


def score(args) -> None:
    from hero_baselines.protocol import canonical_score, stream_metrics

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
    cells, general_cells, failures, length_stops = [], [], 0, 0
    for stage in plan["stages"]:
        stage_dir = run / f"stage-{stage['stage']:02d}"
        complete = stage_dir / "complete.json"
        if not complete.is_file():
            continue
        sealed = json.loads(complete.read_text())["predictions"]
        for evaluation in stage["evaluations"]:
            path = stage_dir / "predictions" / group_file(evaluation["group"])
            if file_hash(path) != sealed[path.relative_to(stage_dir).as_posix()]:
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
    report = {"method": "ophsd", "profile": plan["profile"], "run": str(run), "planSha256": file_hash(args.bundle.resolve() / "plan.json"),
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
    p.add_argument("--profile", choices=("smoke", "full", "sensitivity"), required=True)
    p.add_argument("--learning-rate", type=float, help="Sensitivity profile only: override the shared learning rate")
    p.add_argument("--stages", type=int, default=STAGE_COUNT)
    p.add_argument("--seed", type=int, default=42, help="training seed (the paper reports seeds 42, 43 and 44)")
    p.add_argument("--updates", type=int, default=LoraControlConfig().fixed_updates)
    p.add_argument("--batch-size", type=int, default=LoraControlConfig().effective_batch_size)
    p.add_argument("--eval-limit", type=int, help="Smoke only: first N test items per task")
    p.add_argument("--general-limit", type=int, help="Smoke only: first N G items per category")
    p.add_argument("--remote-root", help="Render job.sh for this MLeRP directory")
    p.add_argument("--account", help="Slurm account; required with --remote-root")
    p.add_argument("--walltime", default="02:00:00")
    p.add_argument("--eval-walltime", default="08:00:00")
    p.add_argument("--split-at", type=int, help="Stop after this stage, then resume in a fresh process")
    t = commands.add_parser("train")
    t.add_argument("--bundle", type=Path, default=HERE)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--checkpoint", type=Path, required=True)
    t.add_argument("--embedding", type=Path, required=True)
    t.add_argument("--fetch", action="store_true", help="Download the pinned checkpoint and embedding model first")
    t.add_argument("--device", default="cuda")
    t.add_argument("--through-stage", type=int)
    t.add_argument("--deadline", type=float, help="Job end (epoch seconds); no stage training starts without its reserve")
    e = commands.add_parser("evaluate")
    e.add_argument("--bundle", type=Path, default=HERE)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--embedding", type=Path, required=True)
    e.add_argument("--fetch", action="store_true")
    e.add_argument("--device", default="cuda")
    e.add_argument("--stage", type=int, required=True)
    e.add_argument("--shard", default="0/1", help="i/n: evaluate panels whose position modulo n is i")
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
        if full and (args.stages != STAGE_COUNT or args.eval_limit or args.general_limit
                     or args.updates != LoraControlConfig().fixed_updates
                     or args.batch_size != LoraControlConfig().effective_batch_size):
            parser.error("The full profile uses the registered stream, budget and complete panels")
        if args.learning_rate is not None and args.profile != "sensitivity":
            parser.error("--learning-rate is only allowed with the sensitivity profile")
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        # The TRACE rouge package computes LCS recursively; outputs that run to the
        # 5,120-token ceiling need a deep stack. Scoring semantics are unchanged.
        import threading

        sys.setrecursionlimit(200_000)
        threading.stack_size(1 << 30)
        worker = threading.Thread(target=score, args=(args,))
        worker.start()
        worker.join()


if __name__ == "__main__":
    main()
