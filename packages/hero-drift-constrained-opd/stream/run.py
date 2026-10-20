"""HERD over the TRACE+17 stream: prepare a self-contained bundle locally, then run the stream in one GPU job.

prepare  Bundle: plan, training records and test prompts of the first S tasks, the G-test prompts, the reserved
         general probes G, this directory's code, CUDA locks, scorer assets.
stream   GPU: per stage evolve the harness, run drift-constrained distillation, merge, refresh the residual memory
         and write the formal-schedule test predictions. Completed stages are replayed from their saved adapters,
         so a restarted job continues after the last completed stage.
score    Score the test predictions locally with the hidden gold and derive the stream metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hero  # noqa: E402

BASE_MODEL_ID = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CHECKPOINT_FILES = ("config.json", "merges.txt", "model.safetensors-00001-of-00002.safetensors",
                    "model.safetensors-00002-of-00002.safetensors", "model.safetensors.index.json",
                    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
                    "video_preprocessor_config.json", "vocab.json")
CODE = ("run.py", "hero.py", "scoring.py", "scorer_assets.py", "bootstrap.sh", "requirements-cuda.lock.txt",
        "requirements-kernels.lock.txt")
ASSETS = ("tokenizers/gpt2/tokenizer.json", "nltk/punkt/english.pickle")
PLAN_VERSION = "hero-stream-plan-v2"
FULL_EVALUATION_STAGES = (4, 8, 16, 25)  # all seen tasks and the full G-test
TRACE_PANEL_STAGES = (12, 20)  # the first eight tasks
GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")


def protocol_tasks(stage: int) -> list[int]:
    """1-based stages whose task is scored after `stage` under the formal sparse schedule."""
    seen = range(1, stage + 1) if stage in FULL_EVALUATION_STAGES else range(1, 9) if stage in TRACE_PANEL_STAGES else []
    return sorted(set(seen) | {stage})


def protocol_general_panel(stage: int) -> str:
    return "full" if stage in FULL_EVALUATION_STAGES else "probe"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# prepare (local)


def prepare(args) -> None:
    root = Path(args.repository).resolve()
    data = root / "experiments/data"
    bundle = args.output.resolve()
    if bundle.exists():
        raise SystemExit(f"Bundle exists: {bundle}; use a new path")
    manifest = json.loads((data / "trace-plus-17/manifest.json").read_text())
    tasks = []
    for task in manifest["tasks"][:args.stages]:
        files = task["files"]
        name = f"{task['position']:02d}"
        write_jsonl(bundle / "inputs/train" / f"{name}.jsonl", read_jsonl(data / "trace-plus-17" / files["train"]["path"]))
        test = read_jsonl(data / "trace-plus-17" / files["testGold"]["path"])  # prompts only: gold stays local
        write_jsonl(bundle / "inputs/test" / f"{name}.jsonl", [{"id": r["id"], "prompt": r["prompt"]} for r in test])
        tasks.append({"stage": task["position"], "taskId": task["id"], "metric": task["metric"], "file": name})
    probes = set(json.loads((data / "general-test/v1/hidden-scorer/probe-ids.json").read_text()))
    write_jsonl(bundle / "inputs/general-test.jsonl",
                [{"id": r["id"], "prompt": r["prompt"], "category": r["category"], "probe": r["id"] in probes}
                 for r in read_jsonl(data / "general-test/v1/hidden-scorer/general-test.jsonl")])
    write_jsonl(bundle / "inputs/general-probes.jsonl", general_probes(root, data))
    for name in CODE:
        shutil.copy2(HERE / name, bundle / name)
    shutil.copytree(HERE / "ifeval", bundle / "ifeval", ignore=shutil.ignore_patterns("__pycache__"))
    for asset in ASSETS:
        (bundle / "assets" / asset).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data / asset, bundle / "assets" / asset)
    plan = {"schemaVersion": PLAN_VERSION, "model": BASE_MODEL_ID, "revision": REVISION, "seed": args.seed,
            "tasks": tasks,
            "harness": {"skills": not args.no_skills, "memory": args.memory,
                        "skillBudgetTokens": 768, "harnessBudgetTokens": 24576, "windowBudgetTokens": 8192,
                        "residualBudgetTokens": 32768, "generalReserveTokens": 8192,
                        "proposer": {"model": BASE_MODEL_ID, "revision": REVISION, "weights": "theta_0, frozen",
                                     "decoding": "greedy, thinking off", "maxNewTokens": 768, "candidates": 16,
                                     "candidateInputTokens": 384, "candidateOutputTokens": 128, "mistakes": 4}},
            "training": asdict(hero.StageConfig(checkpoint_budget=args.checkpoint_budget,
                                                general_budget=args.general_budget, objective=args.objective,
                                                seed=args.seed)),
            "lora": {"rank": 32, "alpha": 64, "dropout": 0.0},
            "limits": {"trainTokens": 16384, "targetTokens": 16384, "probeTokens": 32, "probePromptTokens": 2048,
                       "historicalProbesPerTask": 8, "mistakeNewTokens": 1536},
            "evaluation": {"schedule": "diagonal; all seen tasks and full G-test at 4/8/16/25; tasks 1-8 at 12/20; "
                                       "200-item G probe otherwise; stage 0 = the base model run",
                           "decoding": "greedy, thinking off", "maxNewTokens": 5120, "maxInputTokens": 16384}}
    plan["files"] = {p.relative_to(bundle).as_posix(): file_hash(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    write_json(bundle / "plan.json", plan)
    print(json.dumps({"bundle": str(bundle), "stages": len(tasks), "plan": file_hash(bundle / "plan.json")}))


def general_probes(root: Path, data: Path, budget: int = 8192) -> list[dict[str, Any]]:
    """Reserved G for the base anchor, filled in order under B_G serialized tokens: the non-test GSM8K and IFEval
    records, then the SuperNI general probes, all disjoint from the G-test."""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root / "experiments/checkpoints/qwen3.5-4b/base/tokenizer.json"))
    candidates = [r for r in read_jsonl(data / "general-test/v1/non-test/general.jsonl")
                  if r["category"] in ("mathematical-reasoning", "instruction-following")]
    candidates += read_jsonl(data / "general-probes/v1/non-test/general.jsonl")
    selected, used = [], 0
    for record in candidates:
        size = len(tokenizer.encode(json.dumps({k: record.get(k) for k in ("id", "prompt", "answer")},
                                               ensure_ascii=False)).ids)
        if used + size <= budget:
            selected.append(record)
            used += size
    return selected


# ---------------------------------------------------------------------------
# stream (GPU)


class Stream:
    def __init__(self, args) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.bundle, self.output = args.bundle.resolve(), args.output.resolve()
        self.plan = json.loads((self.bundle / "plan.json").read_text())
        for relative, expected in self.plan["files"].items():
            if file_hash(self.bundle / relative) != expected:
                raise ValueError(f"Bundle file changed: {relative}")
        checkpoint = args.checkpoint.resolve()
        if args.fetch:
            from huggingface_hub import hf_hub_download

            for name in CHECKPOINT_FILES:
                if not (checkpoint / name).is_file():
                    hf_hub_download(BASE_MODEL_ID, name, revision=REVISION, local_dir=checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                                                 local_files_only=True).to(args.device)
        self.model.requires_grad_(False)
        self.proposer = None
        if self.plan["harness"]["skills"]:
            # The skill writer is the frozen base model theta_0, kept on the CPU and moved to the GPU only to write.
            self.proposer = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.bfloat16,
                                                                        local_files_only=True).eval()
            self.proposer.requires_grad_(False)
        self.eos = hero.eos_ids(self.model, self.tokenizer)
        self.pad = self.tokenizer.pad_token_id
        self.end_of_turn = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.head32 = hero.backbone_and_head(self.model)[1].weight.detach().float()
        self.training_config = hero.StageConfig(**self.plan["training"])
        self.divergence = hero.Divergence(self.head32)
        self.limits = self.plan["limits"]
        self.test_stages = args.test_stages
        self.decode_tokens = args.decode_tokens
        # Inference-only re-scoring: the registry of this stage record is attached to every test prompt.
        self.inference_harness = (hero.Harness(tuple(json.loads(json.loads(args.inference_harness.read_text())
                                                                ["harness"])["skills"]))
                                  if args.inference_harness else None)
        self.predictions_name = "test-predictions.harness.jsonl" if self.inference_harness else "test-predictions.jsonl"
        self.output.mkdir(parents=True, exist_ok=True)
        self.job = {"slurmJobId": os.environ.get("SLURM_JOB_ID"), "host": platform.node(),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else args.device,
                    "libraries": {n: __import__(n).__version__ for n in ("torch", "transformers", "peft")}}

    # -- helpers

    def say(self, **row) -> None:
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def adapter_dir(self, stage: int) -> Path:
        return self.output / f"stage-{stage:02d}" / "adapter"

    def tokens(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def propose(self, prompt: str) -> str:
        """theta_0 writes a skill: one greedy call with thinking off."""
        import torch

        settings = self.plan["harness"]["proposer"]
        self.proposer.to(hero.backbone_and_head(self.model)[1].weight.device)
        out = hero.generate(self.proposer, [hero.chat_ids(self.tokenizer, prompt)],
                            max_new_tokens=settings["maxNewTokens"], eos=self.eos, pad_id=self.pad)[0]
        self.proposer.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self.tokenizer.decode(out["tokens"], skip_special_tokens=True).strip()

    def residuals(self, model, harness: "hero.Harness", records) -> list[float]:
        """Residual of each record: mean NLL of its verified answer with only the routed skill and the bare prompt
        in the prefill."""
        rows = [(hero.chat_ids(self.tokenizer, r["prompt"], harness.system(r["prompt"])), self.tokens(r["answer"]))
                for r in records]
        return hero.target_nll(model, rows, self.head32)

    def window(self, task: dict[str, Any], records) -> list[dict[str, Any]]:
        """A_t: whole training records in a seeded order fixed per stage, each counted as prompt + answer + 16
        tokens, until the next record would overflow B_A."""
        shuffled = records[:]
        random.Random(self.plan["seed"] + task["stage"]).shuffle(shuffled)
        window, used = [], 0
        for r in shuffled:
            size = len(self.tokens(r["prompt"])) + len(self.tokens(r["answer"])) + 16
            if used + size > self.plan["harness"]["windowBudgetTokens"]:
                break
            window.append({**r, "tokens": size, "taskId": task["taskId"]})
            used += size
        return window

    def evolve(self, harness: "hero.Harness", task: dict[str, Any], records, window, model):
        """Harness evolution on A_t: theta_0 writes the task's skill from A_t and the frozen checkpoint's mistakes,
        and the candidate registry is committed only if it lowers the mean residual of A_t under the frozen
        checkpoint (the model with its LoRA disabled) and fits B_K."""
        tok, settings, proposer = self.tokenizer, self.plan["harness"], self.plan["harness"]["proposer"]
        lengths = sorted(len(self.tokens(r["answer"])) for r in records)
        cap = max(64, min(self.limits["mistakeNewTokens"], 2 * lengths[-1] + 16))
        with hero.frozen(model):
            outs = hero.generate(model, [hero.chat_ids(tok, r["prompt"]) for r in window],
                                 max_new_tokens=cap, eos=self.eos, pad_id=self.pad)
        predictions = [tok.decode(o["tokens"], skip_special_tokens=True).strip() for o in outs]
        mistakes = [(r, y) for r, y in zip(window, predictions)
                    if hero.normalize(y) != hero.normalize(r["answer"])][:proposer["mistakes"]]
        candidates = hero.distinct_first(window)[:proposer["candidates"]]
        proposal = self.propose(hero.proposal_prompt(candidates, mistakes, tok,
                                                     input_tokens=proposer["candidateInputTokens"],
                                                     output_tokens=proposer["candidateOutputTokens"]))
        skill = hero.skill_from_proposal(task["taskId"], hero.common_prefix([r["prompt"] for r in records]), proposal,
                                         tok, settings["skillBudgetTokens"])
        candidate = harness.with_skill(skill)
        mean = lambda spec: sum(values := self.residuals(model, spec, window)) / len(values)
        residual = {"previous": mean(harness), "candidate": mean(candidate)}
        committed = (residual["candidate"] < residual["previous"]
                     and candidate.tokens(tok) <= settings["harnessBudgetTokens"])
        self.say(stage=task["stage"], task=task["taskId"], windowResidual=residual, committed=committed,
                 skill=skill["instructions"])
        guard = {"windowResidual": residual, "committed": committed, "skill": skill, "proposal": proposal,
                 "mistakes": len(mistakes), "window": len(window)}
        return (candidate if committed else harness), guard

    def refresh(self, model, harness: "hero.Harness", pool, stage: int) -> list[dict[str, Any]]:
        """R_t: records of R_{t-1} u A_t in decreasing residual under the merged weights executing K_t (ties by id),
        each entering only if it fits whole into B_R - B_G. The uniform-memory ablation orders them at random."""
        settings = self.plan["harness"]
        records = sorted(pool, key=lambda r: r["id"])
        if settings["memory"] == "uniform":
            rng = random.Random(self.plan["seed"] + stage)
            scores = {r["id"]: rng.random() for r in records}
        else:
            scores = dict(zip((r["id"] for r in records), self.residuals(model, harness, records)))
        return hero.top_budget(records, scores, settings["residualBudgetTokens"] - settings["generalReserveTokens"])

    def predict(self, model, items, *, system=None) -> list[dict[str, Any]]:
        prompts = [hero.chat_ids(self.tokenizer, r["prompt"], system(r["prompt"]) if system else None) for r in items]
        evaluation = self.plan["evaluation"]
        fits = [k for k, p in enumerate(prompts) if len(p) <= evaluation["maxInputTokens"]]
        outputs = hero.generate(model, [prompts[k] for k in fits], max_new_tokens=evaluation["maxNewTokens"],
                                eos=self.eos, pad_id=self.pad, token_budget=self.decode_tokens)
        rows = [{"id": r["id"], "prediction": "", "outputTokens": 0, "stopReason": "input_limit"} for r in items]
        for k, out in zip(fits, outputs):
            rows[k] = {"id": items[k]["id"], "prediction": self.tokenizer.decode(out["tokens"], skip_special_tokens=True),
                       "outputTokens": len(out["tokens"]), "stopReason": out["stopReason"]}
        return rows

    def wants_test(self, stage: int) -> bool:
        return stage in self.test_stages and not (self.output / f"stage-{stage:02d}" / self.predictions_name).is_file()

    def evaluate_test(self, model, stage: int) -> dict[str, Any]:
        """Formal schedule; predictions only, scored locally by `run.py score`."""
        started, predictions = time.perf_counter(), []
        system = self.inference_harness.system if self.inference_harness else None
        for position in protocol_tasks(stage):
            task = self.plan["tasks"][position - 1]
            rows = self.predict(model, read_jsonl(self.bundle / "inputs/test" / f"{task['file']}.jsonl"), system=system)
            predictions += [{**row, "group": task["taskId"]} for row in rows]
        panel = protocol_general_panel(stage)
        items = [r for r in read_jsonl(self.bundle / "inputs/general-test.jsonl") if panel == "full" or r["probe"]]
        predictions += [{**row, "group": "general:" + item["category"], "panel": panel}
                        for row, item in zip(self.predict(model, items, system=system), items)]
        write_jsonl(self.output / f"stage-{stage:02d}" / self.predictions_name, predictions)
        return {"stage": stage, "testItems": len(predictions), "generalPanel": panel,
                "lengthStops": sum(p["stopReason"] == "length" for p in predictions),
                "inputLimit": sum(p["stopReason"] == "input_limit" for p in predictions),
                "seconds": time.perf_counter() - started}

    # -- stream

    def run(self) -> None:
        import torch
        from peft import LoraConfig, PeftModel

        plan, tok = self.plan, self.tokenizer
        os.environ.setdefault("TRACE_GPT2_TOKENIZER", str(self.bundle / "assets/tokenizers/gpt2/tokenizer.json"))
        os.environ.setdefault("TRACE_PUNKT_TOKENIZER", str(self.bundle / "assets/nltk/punkt/english.pickle"))
        tasks = plan["tasks"]
        train = {t["stage"]: read_jsonl(self.bundle / "inputs/train" / f"{t['file']}.jsonl") for t in tasks}

        # Reserved general probes: theta_0's own first greedy tokens, scored once by theta_0 (the base anchor).
        general_records = read_jsonl(self.bundle / "inputs/general-probes.jsonl")
        general_prompts = [hero.chat_ids(tok, r["prompt"]) for r in general_records]
        completions = hero.generate(self.model, general_prompts, max_new_tokens=self.limits["probeTokens"],
                                    eos=self.eos, pad_id=self.pad)
        source = lambda r: r["taskId"] if r["taskId"] in ("general:gsm8k", "general:ifeval") else "superni"
        general = [hero.Pair(p, c["tokens"], group=source(r)) for p, c, r
                   in zip(general_prompts, completions, general_records)]
        hero.anchor_targets(self.model, general)

        harness, residual = hero.Harness(), []
        lora = plan["lora"]
        lora_config = LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
                                 target_modules=text_backbone_lora_targets(self.model), task_type="CAUSAL_LM")
        for task in tasks[:max(self.test_stages)]:
            stage, directory = task["stage"], self.output / f"stage-{task['stage']:02d}"
            if (directory / "stage.json").is_file():  # replay a completed stage
                record = json.loads((directory / "stage.json").read_text())
                harness = hero.Harness(tuple(json.loads(record["harness"])["skills"]))
                residual = record["residual"]
                if adapter_sha256(self.adapter_dir(stage)) != record["adapterSha256"]:
                    raise RuntimeError(f"Stage {stage} adapter bytes differ from its stage record")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_dir(stage),
                                                       autocast_adapter_dtype=False).merge_and_unload()
                # The merged-weight fingerprint is exact only on the same hardware class (bf16 rounding differs by
                # 1 ulp across CPU/GPU kernels), so a mismatch is reported, not fatal.
                self.say(stage=stage, replayed=True,
                         sameWeights=weights_fingerprint(self.model) == record["weightsFingerprint"])
                if self.wants_test(stage):  # test predictions missing from a replayed stage
                    self.say(test=self.evaluate_test(self.model, stage))
                continue
            if self.inference_harness:
                raise SystemExit(f"--inference-harness re-scores a trained run; stage {stage} is not trained")
            stage_started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model = hero.initialize_adapter(self.model, lora_config, seed=plan["seed"], stage=stage)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            records = train[stage]
            window = self.window(task, records)

            # 1. Harness evolution on A_t (the no-skills ablation proposes and commits nothing).
            guard = None
            if plan["harness"]["skills"]:
                harness, guard = self.evolve(harness, task, records, window, model)

            # 2. Checkpoint probes from R_{t-1}: theta_{t-1}'s first greedy tokens on bare prompts, grouped by task.
            by_task: dict[str, list] = {}
            for r in residual:
                ids = hero.chat_ids(tok, r["prompt"])
                if len(ids) <= self.limits["probePromptTokens"]:
                    by_task.setdefault(r["taskId"], []).append(ids)
            historical_prompts = [(name, ids) for name in sorted(by_task)
                                  for ids in by_task[name][:self.limits["historicalProbesPerTask"]]]
            historical = []
            if historical_prompts:
                with hero.frozen(model):
                    outs = hero.generate(model, [p for _, p in historical_prompts],
                                         max_new_tokens=self.limits["probeTokens"], eos=self.eos, pad_id=self.pad)
                historical = [hero.Pair(p, o["tokens"], group=name) for (name, p), o in zip(historical_prompts, outs)]
                hero.anchor_targets(model, historical)

            # 3. Drift-constrained distillation on the current task's verified answers.
            training = []
            for r in records:
                teacher = hero.teacher_ids(tok, r["prompt"], harness.route(r["prompt"]), r["answer"])
                gold = self.tokens(r["answer"])[:self.limits["targetTokens"]] + [self.end_of_turn]
                student = hero.chat_ids(tok, r["prompt"])
                if len(student) + len(gold) <= self.limits["trainTokens"]:  # longer records are excluded
                    training.append({"studentIds": student, "teacherIds": teacher, "goldIds": gold})
            initial = {"checkpoint": hero.measure_drift(model, self.divergence, historical),
                       "general": hero.measure_drift(model, self.divergence, general)}
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "training-log.jsonl").open("w", encoding="utf-8", buffering=1) as handle:
                def log(row):
                    handle.write(json.dumps(row) + "\n")
                    if row["update"] % 4 == 0 or row["update"] == 1:
                        self.say(stage=stage, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
                summary = hero.train_stage(model, training, divergence=self.divergence, historical=historical,
                                           general=general, config=replace(self.training_config, seed=self.training_config.seed + stage),
                                           log=log)
            final = {"checkpoint": hero.measure_drift(model, self.divergence, historical),
                     "general": hero.measure_drift(model, self.divergence, general)}

            # 4. Merge: the bf16 adapter is saved and merged into the dense weights theta_t.
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(torch.bfloat16)
            model.save_pretrained(self.adapter_dir(stage).with_suffix(".tmp"), safe_serialization=True)
            self.model = model.merge_and_unload()
            self.model.requires_grad_(False)

            # 5. Residual refresh under the merged weights executing K_t.
            residual = self.refresh(self.model, harness, {r["id"]: r for r in residual + window}.values(), stage)

            evaluation = self.evaluate_test(self.model, stage) if self.wants_test(stage) else None
            shutil.rmtree(self.adapter_dir(stage), ignore_errors=True)  # left by a job stopped before stage.json
            os.replace(self.adapter_dir(stage).with_suffix(".tmp"), self.adapter_dir(stage))
            write_json(directory / "stage.json", {
                "stage": stage, "taskId": task["taskId"], "harness": harness.serialize(),
                "harnessTokens": harness.tokens(tok), "guard": guard,
                "residual": residual, "residualTasks": sorted({r["taskId"] for r in residual}),
                "trainingRecords": len(training), "excludedOverLength": len(records) - len(training),
                "drift": {"initial": initial, "final": final}, "training": summary,
                "test": evaluation, "adapterSha256": adapter_sha256(self.adapter_dir(stage)),
                "weightsFingerprint": weights_fingerprint(self.model),
                "seconds": time.perf_counter() - stage_started, "job": self.job,
                "peakMemoryGB": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None})
            self.say(stage=stage, drift=final, test=evaluation)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def adapter_sha256(directory: Path) -> str:
    """Content digest of a saved stage adapter (config and weights)."""
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        digest.update(name.encode())
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def weights_fingerprint(model) -> str:
    """Order-sensitive digest of every parameter's exact bits."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().reshape(-1).view(torch.uint8 if parameter.element_size() == 1 else torch.int16
                                                          if parameter.element_size() == 2 else torch.int32)
                      .sum(dtype=torch.int64).cpu().numpy().tobytes())
    return digest.hexdigest()


def text_backbone_lora_targets(model) -> list[str]:
    """Every linear layer of the text decoder: the vision tower, the multi-token-prediction head and the output
    head are left out."""
    import torch

    excluded = ("visual", "vision", "mtp", "lm_head")
    return sorted(name for name, module in model.named_modules()
                  if isinstance(module, torch.nn.Linear) and not any(part in name.lower() for part in excluded))


# ---------------------------------------------------------------------------
# score (local)


def score(args) -> None:
    """Score formal-schedule test predictions locally with the hidden gold; derive the paper's stream metrics."""
    from scoring import score_predictions

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))  # TRACE ROUGE-L recurses on long outputs
    data = Path(args.repository).resolve() / "experiments/data"
    manifest = json.loads((data / "trace-plus-17/manifest.json").read_text())
    gold = {t["id"]: {r["id"]: r for r in read_jsonl(data / "trace-plus-17" / t["files"]["testGold"]["path"])}
            for t in manifest["tasks"]}
    metric = {t["id"]: t["metric"] for t in manifest["tasks"]}
    general = {r["id"]: r for r in read_jsonl(data / "general-test/v1/hidden-scorer/general-test.jsonl")}
    cells, general_cells = {}, []
    for path in sorted(args.run.resolve().glob(f"stage-*/{args.predictions}")):
        stage = int(path.parent.name.split("-")[1])
        by_group: dict[str, list] = {}
        for row in read_jsonl(path):
            by_group.setdefault(row["group"], []).append(row)
        for group, rows in by_group.items():
            if group.startswith("general:"):
                value = sum(score_predictions(general[r["id"]]["metric"], [r], [general[r["id"]]])["score"]
                            for r in rows) / len(rows)
                general_cells.append({"stage": stage, "category": group.split(":", 1)[1], "panel": rows[0]["panel"],
                                      "items": len(rows), "score": value})
            else:
                cells[(stage, group)] = score_predictions(metric[group], rows, [gold[group][r["id"]] for r in rows])["score"]
    order = [t["id"] for t in manifest["tasks"]]
    last = max(stage for stage, _ in cells)
    diagonal = [cells[(j, order[j - 1])] for j in range(1, last + 1) if (j, order[j - 1]) in cells]
    metrics: dict[str, Any] = {"completedStages": last, "acquisitionMeanDiagonal": sum(diagonal) / len(diagonal)}
    row = [cells.get((last, order[j - 1])) for j in range(1, last + 1)]
    if all(value is not None for value in row):
        metrics["seenMean"] = sum(row) / len(row)
        earlier = range(1, last)
        if earlier:
            metrics["forgetting"] = sum(max(0.0, cells[(j, order[j - 1])] - cells[(last, order[j - 1])])
                                        for j in earlier) / len(earlier)
            metrics["signedBWT"] = sum(cells[(last, order[j - 1])] - cells[(j, order[j - 1])]
                                       for j in earlier) / len(earlier)
    by_stage: dict[int, list] = {}
    for cell in general_cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    metrics["generalByStage"] = [{"stage": st, "panel": rows[0]["panel"],
                                  "categories": {c["category"]: c["score"] for c in rows},
                                  "macro": sum(c["score"] for c in rows) / len(rows)}
                                 for st, rows in sorted(by_stage.items()) if len(rows) == len(GENERAL_CATEGORIES)]
    report = {"taskCells": [{"stage": st, "task": t, "score": v} for (st, t), v in sorted(cells.items())],
              "generalCells": general_cells, "metrics": metrics}
    write_json(args.output, report)
    print(json.dumps(metrics, indent=2))


def budget(value: str) -> float | None:
    return None if value == "none" else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--repository", default=".")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-budget", type=budget, default=0.05,
                   help="epsilon_C; 'none' removes the checkpoint anchor (drift is still audited)")
    p.add_argument("--general-budget", type=budget, default=0.05,
                   help="epsilon_0; 'none' removes the base anchor (drift is still audited)")
    p.add_argument("--no-skills", action="store_true",
                   help="ablation: nothing is proposed or committed; the teacher sees only the verified answer")
    p.add_argument("--memory", choices=("residual", "uniform"), default="residual",
                   help="ablation 'uniform': the memory is filled by uniform sampling under the same budget")
    p.add_argument("--objective", choices=("distill", "sft"), default="distill",
                   help="ablation 'sft': cross-entropy on the verified answers under the same adapter and constraints")
    s = commands.add_parser("stream")
    s.add_argument("--bundle", type=Path, default=HERE)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--fetch", action="store_true")
    s.add_argument("--device", default="cuda")
    s.add_argument("--decode-tokens", type=int, default=98304,
                   help="evaluation decoding batch: rows x longest row within this many tokens (98,304 fits 40 GB)")
    s.add_argument("--test-stages", type=lambda v: {int(x) for x in v.split(",")}, default=set(range(1, 26)),
                   help="comma-separated stages whose test predictions this job writes; the job stops after the "
                        "last one, so several jobs can share one output after the stages are trained")
    s.add_argument("--inference-harness", type=Path,
                   help="a stage.json of a trained run: re-score that run's stages with its registry attached to "
                        "every test prompt (written to test-predictions.harness.jsonl)")
    c = commands.add_parser("score")
    c.add_argument("run", type=Path)
    c.add_argument("--repository", default=".")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--predictions", default="test-predictions.jsonl",
                   help="test-predictions.harness.jsonl scores an --inference-harness re-scoring")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "stream":
        Stream(args).run()
    else:
        score(args)


if __name__ == "__main__":
    main()
