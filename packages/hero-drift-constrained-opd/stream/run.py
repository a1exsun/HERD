"""HERO over the TRACE+17 stream: prepare a self-contained bundle locally, then run the stream in one GPU job.

prepare  Bundle: training records, test prompts without gold, reserved general probes, code and CUDA locks.
stream   GPU: evolve the harness, retain residuals, train and merge; evaluate on the formal test schedule.
score    Score formal-schedule predictions locally with hidden gold.
evaluate-harness  Re-score the final full-run weights with their harness; no training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
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
VARIANTS = ("full", "no-harness", "uniform-residual", "no-checkpoint-anchor", "no-base-anchor", "no-anchors",
            "sft", "drift-budget-tight", "drift-budget-loose")
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


def training_config(args) -> hero.StageConfig:
    budget = {"drift-budget-tight": 0.02, "drift-budget-loose": 0.1}.get(args.variant, args.checkpoint_budget)
    return hero.StageConfig(updates=args.updates, batch_size=args.batch_size, learning_rate=args.learning_rate,
                            checkpoint_budget=None if args.variant in ("no-checkpoint-anchor", "no-anchors") else budget,
                            general_budget=None if args.variant in ("no-base-anchor", "no-anchors") else args.general_budget,
                            dual_rate=args.dual_rate, seed=args.seed,
                            objective="sft" if args.variant == "sft" else "distill")


def bind_output(output: Path, plan: Path, mode: str) -> None:
    """Never resume a different seed/variant in the same output directory."""
    record = {"planSha256": file_hash(plan), "mode": mode}
    marker = output / "run-spec.json"
    if marker.exists():
        if json.loads(marker.read_text()) != record:
            raise ValueError("Output belongs to a different bundle or evaluation mode; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Nonempty output has no run specification; use a new output directory")
    else:
        write_json(marker, record)


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
        train = read_jsonl(data / "trace-plus-17" / files["train"]["path"])
        name = f"{task['position']:02d}"
        write_jsonl(bundle / "inputs/train" / f"{name}.jsonl", train[:args.train_limit] if args.train_limit else train)
        # Prompts only: test gold stays in the local scorer.
        test = read_jsonl(data / "trace-plus-17" / files["testGold"]["path"])
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
            "initializationSeed": args.seed if args.init_seed is None else args.init_seed,
            "variant": args.variant,
            "tasks": tasks,
            "harness": {"skillBudgetTokens": args.skill_tokens, "harnessBudgetTokens": args.harness_tokens,
                        "evidenceBudgetTokens": 8192, "residualBudgetTokens": 32768, "generalReserveTokens": 8192,
                        "harmTolerance": args.harm_tolerance,  # nats/token of teacher NLL on A_t
                        "proposer": {"model": BASE_MODEL_ID, "revision": REVISION, "weights": "theta_0, frozen",
                                     "decoding": "greedy, thinking off", "maxNewTokens": 768, "candidates": 16,
                                     "candidateInputTokens": 384, "candidateOutputTokens": 128, "mistakes": 4}},
            "training": asdict(training_config(args)),
            "lora": {"rank": args.rank, "alpha": 2 * args.rank, "dropout": 0.0},
            "limits": {"trainTokens": args.train_tokens, "targetTokens": args.target_tokens, "probeTokens": 32, "historicalProbesPerTask": args.probes_per_task,
                       "mistakeNewTokens": 1536},
            "evaluation": {"split": "test", "decoding": "greedy, thinking off",
                           "schedule": "diagonal; all seen tasks and full G-test at 4/8/16/25; tasks 1-8 at "
                                       "12/20; 200-item G probe otherwise; stage 0 = the base model run",
                           "maxNewTokens": 5120, "maxInputTokens": 16384}}
    plan["files"] = {p.relative_to(bundle).as_posix(): file_hash(p) for p in sorted(bundle.rglob("*")) if p.is_file()}
    write_json(bundle / "plan.json", plan)
    print(json.dumps({"bundle": str(bundle), "stages": len(tasks), "plan": file_hash(bundle / "plan.json")}))


def general_probes(root: Path, data: Path, budget: int = 8192) -> list[dict[str, Any]]:
    """Reserved G for the base anchor, filled in order under B_G serialized tokens: the non-test GSM8K and IFEval
    records (open-ended reasoning and instruction following), then the SuperNI general probes. All are disjoint
    from G-test and the development panel. G focuses on open-ended reasoning and instruction following;
    multiple-choice records are excluded."""
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
        if self.plan["schemaVersion"] != PLAN_VERSION:
            raise ValueError(f"Expected {PLAN_VERSION}; prepare a new bundle with explicit initialization settings")
        for relative, expected in self.plan["files"].items():
            if file_hash(self.bundle / relative) != expected:
                raise ValueError(f"Bundle file changed: {relative}")
        bind_output(self.output, self.bundle / "plan.json", args.command)
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
        # The proposer is the frozen base model theta_0, kept on the CPU and moved to the GPU only to write a skill.
        self.proposer = None
        if args.command == "stream" and self.plan["variant"] != "no-harness":
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
        self.log_handle = None
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

    def evolve(self, harness: "hero.Harness", task: dict[str, Any], records, model):
        """Guarded harness evolution on bounded evidence A_t: theta_0 writes the task's skill from A_t's examples, and
        the skill is committed unless it breaks the frozen teacher (the model with its LoRA disabled)."""
        tok, settings, proposer = self.tokenizer, self.plan["harness"], self.plan["harness"]["proposer"]
        shuffled = records[:]
        random.Random(self.plan["seed"] + task["stage"]).shuffle(shuffled)  # per stage, so replay does not shift it
        evidence, used = [], 0
        for r in shuffled:
            size = len(self.tokens(r["prompt"])) + len(self.tokens(r["answer"])) + 16
            if used + size <= settings["evidenceBudgetTokens"]:
                evidence.append({**r, "tokens": size, "taskId": task["taskId"]})
                used += size
        if self.plan["variant"] == "no-harness":
            return hero.Harness(), evidence, {"enabled": False, "teacherNLL": {"previous": None, "candidate": None},
                "gain": None, "committed": False, "skill": None, "proposal": None, "mistakes": None,
                "evidence": len(evidence)}
        # The frozen checkpoint's own answers on A_t: its mistakes ground the skill's Avoid list.
        with hero.frozen(model):
            outs = hero.generate(model, [hero.chat_ids(tok, r["prompt"]) for r in evidence],
                                 max_new_tokens=self.mistake_caps[task["stage"]], eos=self.eos, pad_id=self.pad)
        predictions = [tok.decode(o["tokens"], skip_special_tokens=True).strip() for o in outs]
        mistakes = [(r, y) for r, y in zip(evidence, predictions)
                    if hero.normalize(y) != hero.normalize(r["answer"])][:proposer["mistakes"]]
        candidates = hero.distinct_first(evidence)[:proposer["candidates"]]
        proposal = self.propose(hero.proposal_prompt(candidates, mistakes, tok,
                                                     input_tokens=proposer["candidateInputTokens"],
                                                     output_tokens=proposer["candidateOutputTokens"]))
        skill = hero.skill_from_proposal(task["taskId"], hero.common_prefix([r["prompt"] for r in records]), proposal,
                                         tok, settings["skillBudgetTokens"])
        candidate = harness.with_skill(skill)

        def teacher_nll(spec: hero.Harness) -> float:  # the frozen teacher's NLL of the verified targets on A_t
            rows = [(hero.teacher_ids(tok, r["prompt"], spec.route(r["prompt"]), r["answer"]),
                     self.tokens(r["answer"])[:self.limits["targetTokens"]] + [self.end_of_turn]) for r in evidence]
            return sum(hero.target_nll(model, rows, self.head32)) / len(rows)

        # Commit unless the skill breaks the teacher: with the verified answer replayed, the teacher's NLL is already
        # near zero, so a gain is rarely measurable; a large rise means the skill fights the replay.
        nll = {"previous": teacher_nll(harness), "candidate": teacher_nll(candidate)}
        gain = nll["previous"] - nll["candidate"]
        committed = (bool(skill["instructions"]) and -gain <= settings["harmTolerance"]
                     and candidate.tokens(tok) <= settings["harnessBudgetTokens"])
        self.say(stage=task["stage"], task=task["taskId"], teacherNLL=nll, committed=committed,
                 skill=skill["instructions"])
        guard = {"enabled": True, "teacherNLL": nll, "gain": gain, "committed": committed, "skill": skill, "proposal": proposal,
                 "mistakes": len(mistakes), "evidence": len(evidence)}
        return (candidate if committed else harness), evidence, guard

    def answer_cap(self, records, quantile: float, scale: float, low: int, high: int) -> int:
        lengths = sorted(len(self.tokens(r["answer"])) for r in records)
        return max(low, min(high, int(scale * lengths[min(len(lengths) - 1, int(quantile * len(lengths)))]) + 16))

    def predict(self, model, items, *, system=None) -> list[dict[str, Any]]:
        prompts = [hero.chat_ids(self.tokenizer, r["prompt"], system(r["prompt"]) if system else None) for r in items]
        fits = [k for k, p in enumerate(prompts) if len(p) <= self.plan["evaluation"]["maxInputTokens"]]
        outputs = hero.generate(model, [prompts[k] for k in fits], max_new_tokens=self.plan["evaluation"]["maxNewTokens"], eos=self.eos,
                                pad_id=self.pad, token_budget=self.decode_tokens)
        rows = [{"id": r["id"], "prediction": "", "outputTokens": 0, "stopReason": "input_limit"} for r in items]
        for k, out in zip(fits, outputs):
            rows[k] = {"id": items[k]["id"], "prediction": self.tokenizer.decode(out["tokens"], skip_special_tokens=True),
                       "outputTokens": len(out["tokens"]), "stopReason": out["stopReason"]}
        return rows

    def wants_test(self, stage: int) -> bool:
        return (stage in self.test_stages
                and not (self.output / f"stage-{stage:02d}/test-predictions.jsonl").is_file())

    def evaluate_test(self, model, stage: int, *, system=None, full_panel=False) -> dict[str, Any]:
        """Formal schedule; predictions only, scored locally by `run.py score`."""
        started, predictions = time.perf_counter(), []
        for position in (range(1, stage + 1) if full_panel else protocol_tasks(stage)):
            task = self.plan["tasks"][position - 1]
            rows = self.predict(model, read_jsonl(self.bundle / "inputs/test" / f"{task['file']}.jsonl"),
                                system=system)
            predictions += [{**row, "group": task["taskId"]} for row in rows]
        panel = "full" if full_panel else protocol_general_panel(stage)
        items = [r for r in read_jsonl(self.bundle / "inputs/general-test.jsonl") if panel == "full" or r["probe"]]
        predictions += [{**row, "group": "general:" + item["category"], "panel": panel}
                        for row, item in zip(self.predict(model, items, system=system), items)]
        write_jsonl(self.output / f"stage-{stage:02d}/test-predictions.jsonl", predictions)
        return {"stage": stage, "testItems": len(predictions), "generalPanel": panel,
                "lengthStops": sum(p["stopReason"] == "length" for p in predictions),
                "inputLimit": sum(p["stopReason"] == "input_limit" for p in predictions),
                "seconds": time.perf_counter() - started}

    # -- stream

    def evaluate_harness(self, source: Path) -> None:
        """Replay a completed full run, then evaluate only its final weights with K_T."""
        from peft import PeftModel

        source = source.resolve()
        if source == self.output or source in self.output.parents:
            raise ValueError("Harness evaluation output must be separate from the training run")
        if self.plan['variant'] != 'full' or self.plan['evaluation']['split'] != 'test':
            raise ValueError("Harness inference requires a full-method bundle prepared for formal test evaluation")
        expected = {'planSha256': file_hash(self.bundle / 'plan.json'), 'mode': 'stream'}
        if json.loads((source / 'run-spec.json').read_text()) != expected:
            raise ValueError("Source run does not belong to this full-method bundle")
        adapters = {}
        for task in self.plan['tasks']:
            stage = task['stage']
            directory = source / f'stage-{stage:02d}'
            record = json.loads((directory / 'stage.json').read_text())
            adapter = directory / 'adapter'
            digest = adapter_sha256(adapter)
            if digest != record['adapterSha256']:
                raise ValueError(f'Stage {stage} adapter differs from its record')
            self.model = PeftModel.from_pretrained(self.model, adapter, autocast_adapter_dtype=False).merge_and_unload()
            adapters[str(stage)] = digest
        harness = hero.Harness(tuple(json.loads(record['harness'])['skills']))
        evaluation = self.evaluate_test(self.model, stage, system=harness.system, full_panel=True)
        write_json(self.output / 'evaluation.json', {'variant': 'hero-with-harness', 'sourceRun': str(source),
                   'adapters': adapters, 'harness': harness.serialize(), **evaluation})
        self.say(**evaluation)

    def run(self) -> None:
        import torch
        from peft import LoraConfig, PeftModel

        plan, tok = self.plan, self.tokenizer
        os.environ.setdefault("TRACE_GPT2_TOKENIZER", str(self.bundle / "assets/tokenizers/gpt2/tokenizer.json"))
        os.environ.setdefault("TRACE_PUNKT_TOKENIZER", str(self.bundle / "assets/nltk/punkt/english.pickle"))
        tasks = plan["tasks"]
        self.train = {t["stage"]: read_jsonl(self.bundle / "inputs/train" / f"{t['file']}.jsonl") for t in tasks}
        self.mistake_caps = {s: self.answer_cap(rows, 1.0, 2.0, 64, self.limits["mistakeNewTokens"])
                          for s, rows in self.train.items()}
        settings = plan["harness"]

        # Reserved general probes: theta_0's own greedy answers, scored once by theta_0 (the base anchor).
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
        lora_config = None
        for task in tasks[:max(self.test_stages)]:
            stage, directory = task["stage"], self.output / f"stage-{task['stage']:02d}"
            if lora_config is None:
                lora_config = LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                         bias="none", target_modules=text_backbone_lora_targets(self.model),
                                         task_type="CAUSAL_LM")
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
            stage_started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model = hero.initialize_adapter(self.model, lora_config, seed=plan["initializationSeed"], stage=stage)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            records = self.train[stage]

            # 1. Guarded harness evolution on bounded evidence A_t.
            harness, evidence, guard = self.evolve(harness, task, records, model)

            # 2. Residual retention under the committed harness (frozen checkpoint, no target in context).
            pool = {r["id"]: r for r in residual + evidence}
            previous_residual = residual
            memory_budget = settings["residualBudgetTokens"] - settings["generalReserveTokens"]
            if plan["variant"] == "uniform-residual":
                residual = hero.uniform_budget(list(pool.values()), memory_budget, plan["seed"])
            else:
                rows = [(hero.chat_ids(tok, r["prompt"], harness.system(r["prompt"])), self.tokens(r["answer"]))
                        for r in pool.values()]
                fits = [len(p) + len(c) <= self.limits["trainTokens"] for p, c in rows]
                values = iter(hero.target_nll(model, [row for row, ok in zip(rows, fits) if ok], self.head32))
                residuals = {key: next(values) if ok else math.inf for key, ok in zip(pool, fits)}
                residual = hero.top_budget(list(pool.values()), residuals, memory_budget)

            # 3. Historical probes from R_{t-1}: theta_{t-1}'s greedy answers on bare prompts.
            by_task: dict[str, list] = {}
            for r in previous_residual:
                ids = hero.chat_ids(tok, r["prompt"])
                if len(ids) <= 2048:
                    by_task.setdefault(r["taskId"], []).append(ids)
            # (task, prompt ids): a few probes for every retained task, so one task's group can show a label shift.
            historical_prompts = [(name, ids) for name in sorted(by_task)
                                  for ids in by_task[name][:self.limits["historicalProbesPerTask"]]]
            historical = []
            if historical_prompts:
                with hero.frozen(model):
                    outs = hero.generate(model, [p for _, p in historical_prompts],
                                         max_new_tokens=self.limits["probeTokens"], eos=self.eos, pad_id=self.pad)
                historical = [hero.Pair(p, o["tokens"], group=name) for (name, p), o in zip(historical_prompts, outs)]
                hero.anchor_targets(model, historical)

            # 4. Drift-constrained distillation on the current task's verified answers.
            training = []
            for r in records:
                teacher = (hero.teacher_ids(tok, r["prompt"], harness.route(r["prompt"]), r["answer"])
                           if self.training_config.objective == "distill" else None)
                gold = self.tokens(r["answer"])[:self.limits["targetTokens"]] + [self.end_of_turn]
                student = hero.chat_ids(tok, r["prompt"])
                if len(student) + len(gold) <= self.limits["trainTokens"]:  # the baselines' rule: prompt + target
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
            for parameter in model.parameters():  # bf16 adapter: half the storage, and replay merges identically
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(torch.bfloat16)
            model.save_pretrained(self.adapter_dir(stage).with_suffix(".tmp"), safe_serialization=True)
            self.model = model.merge_and_unload()
            self.model.requires_grad_(False)

            evaluation = self.evaluate_test(self.model, stage) if self.wants_test(stage) else None
            shutil.rmtree(self.adapter_dir(stage), ignore_errors=True)  # left by a job stopped before stage.json
            os.replace(self.adapter_dir(stage).with_suffix(".tmp"), self.adapter_dir(stage))
            adapter_digest = adapter_sha256(self.adapter_dir(stage))
            write_json(directory / "stage.json", {
                "stage": stage, "taskId": task["taskId"], "harness": harness.serialize(),
                "variant": plan["variant"], "initializationSeed": plan["initializationSeed"] + stage,
                "harnessTokens": harness.tokens(tok), "guard": guard,
                "residual": residual, "residualTasks": sorted({r["taskId"] for r in residual}),
                "trainingRecords": len(training), "excludedOverLength": len(records) - len(training),
                "drift": {"initial": initial, "final": final}, "training": summary,
                "test": evaluation, "adapterSha256": adapter_digest,
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
    import torch

    excluded = ("visual", "vision", "mtp", "lm_head")
    return sorted(name for name, module in model.named_modules()
                  if isinstance(module, torch.nn.Linear) and not any(part in name.lower() for part in excluded))


# ---------------------------------------------------------------------------
# report (local)


def score(args) -> None:
    """Score formal-schedule test predictions locally with the hidden gold; derive the paper's stream metrics."""
    from scoring import score_predictions

    inference_record = args.run / 'evaluation.json'
    if inference_record.is_file():
        inference = json.loads(inference_record.read_text())
        if inference.get('variant') == 'hero-with-harness' and (
                args.diagonal_run is None or args.diagonal_run.resolve() != Path(inference['sourceRun']).resolve()):
            raise ValueError('Harness inference must use --diagonal-run from its original full-method run')
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))  # TRACE ROUGE-L recurses on long outputs
    data = Path(args.repository).resolve() / "experiments/data"
    manifest = json.loads((data / "trace-plus-17/manifest.json").read_text())
    gold = {t["id"]: {r["id"]: r for r in read_jsonl(data / "trace-plus-17" / t["files"]["testGold"]["path"])}
            for t in manifest["tasks"]}
    metric = {t["id"]: t["metric"] for t in manifest["tasks"]}
    general = {r["id"]: r for r in read_jsonl(data / "general-test/v1/hidden-scorer/general-test.jsonl")}
    cells, general_cells = {}, []
    for path in sorted(args.run.resolve().glob("stage-*/test-predictions.jsonl")):
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
    learned = {}
    for j in range(1, last + 1):
        task = order[j - 1]
        if args.diagonal_run is not None:
            predictions = [r for r in read_jsonl(args.diagonal_run / f'stage-{j:02d}/test-predictions.jsonl')
                           if r['group'] == task]
            if {r['id'] for r in predictions} != set(gold[task]):
                raise ValueError(f'Incomplete source diagonal at stage {j}')
            learned[j] = score_predictions(metric[task], predictions, list(gold[task].values()))['score']
        else:
            if (j, task) not in cells:
                raise ValueError(f'Missing diagonal at stage {j}; harness inference requires --diagonal-run')
            learned[j] = cells[(j, task)]
    diagonal = list(learned.values())
    metrics: dict[str, Any] = {"completedStages": last, "acquisitionMeanDiagonal": sum(diagonal) / len(diagonal)}
    row = [cells.get((last, order[j - 1])) for j in range(1, last + 1)]
    if all(value is not None for value in row):
        metrics["seenMean"] = sum(row) / len(row)
        earlier = range(1, last)
        if earlier:
            metrics["forgetting"] = sum(max(0.0, learned[j] - cells[(last, order[j - 1])])
                                        for j in earlier) / len(earlier)
            metrics["signedBWT"] = sum(cells[(last, order[j - 1])] - learned[j]
                                       for j in earlier) / len(earlier)
    by_stage: dict[int, list] = {}
    for cell in general_cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    metrics["generalByStage"] = [{"stage": st, "panel": rows[0]["panel"], "macro": sum(c["score"] for c in rows) / len(rows)}
                                 for st, rows in sorted(by_stage.items()) if len(rows) == len(GENERAL_CATEGORIES)]
    report = {"taskCells": [{"stage": st, "task": t, "score": v} for (st, t), v in sorted(cells.items())],
              "generalCells": general_cells, "metrics": metrics}
    if args.diagonal_run is not None:
        report['diagonalRun'] = str(args.diagonal_run.resolve())
    write_json(args.output, report)
    print(json.dumps(metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--repository", default=".")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", type=int, default=25)
    p.add_argument("--train-limit", type=int)
    p.add_argument("--skill-tokens", type=int, default=768, help="cap on a skill's instructions")
    p.add_argument("--harness-tokens", type=int, default=24576, help="B_K: cap on the serialized skill registry")
    p.add_argument("--train-tokens", type=int, default=16384, help="largest prompt + target length trained on")
    p.add_argument("--target-tokens", type=int, default=16384, help="verified targets are clipped to this length")
    p.add_argument("--probes-per-task", type=int, default=8, help="checkpoint-drift probes per retained task")
    p.add_argument("--harm-tolerance", type=float, default=0.3,
                   help="largest rise in the teacher's NLL of A_t's verified targets that a skill may cause")
    p.add_argument("--updates", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--checkpoint-budget", type=float, default=0.05)
    p.add_argument("--general-budget", type=float, default=0.05)
    p.add_argument("--dual-rate", type=float, default=0.5)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init-seed", type=int,
                   help="LoRA initialization seed; defaults to --seed; stage t uses init-seed + t")
    p.add_argument("--variant", choices=VARIANTS, default="full")
    s = commands.add_parser("stream")
    s.add_argument("--bundle", type=Path, default=HERE)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--fetch", action="store_true")
    s.add_argument("--device", default="cuda")
    s.add_argument("--decode-tokens", type=int, default=98304,
                   help="evaluation decoding batch: rows x longest row within this many tokens (98,304 fits 40 GB)")
    s.add_argument("--test-stages", type=lambda v: {int(x) for x in v.split(",")}, default=set(range(1, 26)),
                   help="comma-separated stages whose test predictions this job writes (test split); the job stops "
                        "after the last one, so several jobs can share one output after the stages are trained")
    h = commands.add_parser("evaluate-harness", help="Final full-run weights with the final routed harness; no training")
    h.add_argument("--bundle", type=Path, required=True)
    h.add_argument("--run", type=Path, required=True, help="Completed full-method stream output")
    h.add_argument("--output", type=Path, required=True, help="Separate inference output directory")
    h.add_argument("--checkpoint", type=Path, required=True)
    h.add_argument("--fetch", action="store_true")
    h.add_argument("--device", default="cuda")
    h.add_argument("--decode-tokens", type=int, default=98304)
    h.set_defaults(test_stages=set(range(1, 26)))
    c = commands.add_parser("score")
    c.add_argument("run", type=Path)
    c.add_argument("--repository", default=".")
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--diagonal-run", type=Path,
                   help="Original full-run diagonals, required when scoring final-only harness inference")
    args = parser.parse_args()
    if args.command == "prepare" and args.variant in ("drift-budget-tight", "drift-budget-loose"):
        preset = {"drift-budget-tight": 0.02, "drift-budget-loose": 0.1}[args.variant]
        if args.checkpoint_budget not in (0.05, preset):
            parser.error("--checkpoint-budget conflicts with the named sweep variant")
    if args.command == "evaluate-harness":
        source, output = args.run.resolve(), args.output.resolve()
        if source == output or source in output.parents:
            parser.error("--output must be separate from the source training run")
    if args.command == "prepare":
        prepare(args)
    elif args.command == "stream":
        Stream(args).run()
    elif args.command == "evaluate-harness":
        Stream(args).evaluate_harness(args.run)
    elif args.command == "score":
        score(args)


if __name__ == "__main__":
    main()
