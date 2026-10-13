"""Stable Harness Continual Learning over the TRACE+17 stream on this Mac: prepare a bundle, run, score.

prepare   Build a self-contained bundle: plan, stage training records, the evaluator's validation gold
          (the current task's full validation set and each task's frozen 20-item anchor panel), prompt-only
          test/G inputs (no test gold), this directory's code and scoring assets.
run       The stream through stage N in one process and one MLX engine: stages train in order, and each
          sealed stage is evaluated alongside the next stage's training. Sealed units are skipped and a
          stage's evidence and evaluations persist, so an interrupted run resumes.
train     Stage N alone: restore the stage N-1 harness, collect interaction evidence, run the four sequential
          component updates with the stable B_n = 0 gate, seal the harness. Every model call is served by
          Qwen3.5-4B.
evaluate  Stage N alone: the sealed stage N harness on the base control's sparse schedule; per-item sealed.
score     Score sealed predictions with this directory's evaluator and derive stream metrics.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# The rouge package reconstructs the LCS recursively, one frame per token of prediction plus reference, so
# scoring a long ROUGE-L prediction (during training too, in worker threads) needs a deep recursion limit.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
threading.stack_size(64 << 20)

from hcl_stable import HarnessState, build_component_optimizer_prompt, evolve_stable_harness  # noqa: E402
from protocol import (  # noqa: E402
    BASE_MODEL_ID, EVAL_MAX_INPUT_TOKENS, EVAL_MAX_NEW_TOKENS, STAGE_COUNT, evaluated_task_indices,
    general_panel_kind, load_manifest, read_jsonl,
)

REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_TREE_SHA256 = "e2361c06ba172642a18b1472f4758449bcddf20072e371921642bfba28004dc5"
GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")
PLAN_VERSION = "hcl-stable-plan-v1"
ANCHOR_PANEL = "experiments/data/evaluation-panels/v1/validation.json"  # HERO's frozen 20-item backtest panel
ENGINE_BATCH = 16             # concurrent sequences in the MLX decode batch
EVIDENCE_CHUNK = 32           # training records executed per evidence round
HARNESS_TOKEN_BUDGET = 12000  # rendered harness, Qwen tokens; leaves room inside the 16,384 per-call ceiling
EXECUTION_MAX_TOKENS = EVAL_MAX_NEW_TOKENS  # the harness executor decodes as in test evaluation
PROPOSER_MAX_TOKENS = EVAL_MAX_NEW_TOKENS   # a replacement component may use the protocol's full output ceiling
PROPOSER_TEMPERATURE = 0.2
INVALID = "__hcl_invalid_proposal__"
BUNDLE_CODE = ("run.py", "hcl_stable.py", "local_backend.py", "protocol.py", "scoring.py", "scorer_assets.py",
               "pyproject.toml", "uv.lock")
BUNDLE_TREES = ("ifeval",)
SCORER_ASSETS = {"assets/gpt2-tokenizer.json": "experiments/data/tokenizers/gpt2/tokenizer.json",
                 "assets/punkt-english.pickle": "experiments/data/nltk/punkt/english.pickle"}
TRAIN_FIELDS = ("id", "prompt", "answer", "references", "metadata")


def render_harness(state: HarnessState) -> str:
    return "\n\n".join([
        "Use the following external harness while answering. Do not mention the harness.",
        f"Task Interface:\n{json.dumps(state.task_interface, ensure_ascii=False, sort_keys=True)}",
        f"Experience Memory:\n{json.dumps(state.experience_memory, ensure_ascii=False, sort_keys=True)}",
        f"Capability Map:\n{json.dumps(state.capability_map, ensure_ascii=False, sort_keys=True)}",
        f"Adaptive Router:\n{json.dumps(state.adaptive_router, ensure_ascii=False, sort_keys=True)}",
    ])


def harness_messages(state: HarnessState, prompt: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": render_harness(state)}, {"role": "user", "content": prompt}]


def state_from_json(text: str) -> HarnessState:
    value = json.loads(text)
    return HarnessState(value["taskInterface"], value["experienceMemory"], value["capabilityMap"], value["adaptiveRouter"])


def parse_component(text: str) -> Mapping[str, Any]:
    """The proposer's replacement component; anything but one JSON object is an invalid candidate."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return {INVALID: f"not JSON: {error}"}
    return parsed if isinstance(parsed, dict) else {INVALID: "not a JSON object"}


def is_invalid(state: HarnessState) -> bool:
    return any(isinstance(part, Mapping) and INVALID in part for part in
               (state.task_interface, state.experience_memory, state.capability_map, state.adaptive_router))


def pack_evidence(interactions: Sequence[Mapping[str, Any]],
                  fits: Callable[[list[Mapping[str, Any]]], bool]) -> list[Mapping[str, Any]]:
    """Interactions in their recorded (seeded) order, added while the complete optimizer prompt still fits."""
    packed: list[Mapping[str, Any]] = []
    for interaction in interactions:
        if fits(packed + [interaction]):
            packed.append(interaction)
    return packed


def evidence_order(count: int, seed: int) -> list[int]:
    order = list(range(count))
    random.Random(seed).shuffle(order)
    return order


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
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def group_file(group: str) -> str:
    return group.replace(":", "__") + ".jsonl"


def stage_dir(output: Path, position: int) -> Path:
    return output / f"stage-{position:02d}"


def scientific_plan_hash(plan: dict[str, Any]) -> str:
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


def build_plan(dataset: Path, general_test: Path, anchor_panel: Path, *, profile: str, stages: int,
               minimum_current_gain: float, seed: int = 42, train_limit: int | None, validation_limit: int | None,
               eval_limit: int | None, general_limit: int | None) -> tuple[dict[str, Any], dict[str, dict[str, list]]]:
    manifest = load_manifest(dataset)
    tasks = manifest["tasks"]
    panels = {panel["taskId"]: panel["sampleIds"] for panel in json.loads(anchor_panel.read_text())["panels"]}
    files: dict[str, dict[str, list]] = {"train": {}, "validation": {}, "anchors": {}, "eval": {}}
    for task in tasks:
        rows = verified_rows(dataset, task["files"]["testGold"])
        files["eval"][task["id"]] = [{"id": r["id"], "prompt": r["prompt"]} for r in rows]
    test_manifest = json.loads((general_test / "manifest.json").read_text())
    general_rows = verified_rows(general_test, manifest_file(test_manifest, "hidden-scorer/general-test.jsonl"))
    probe_ids = set(json.loads((general_test / "hidden-scorer/probe-ids.json").read_text()))
    for category in GENERAL_CATEGORIES:
        files["eval"]["general:" + category] = [{"id": r["id"], "prompt": r["prompt"]} for r in general_rows
                                                if r["category"] == category]
    plan_stages: list[dict[str, Any]] = []
    for position in range(1, stages + 1):
        task = tasks[position - 1]
        key = f"{position:02d}"
        records = verified_rows(dataset, task["files"]["train"])
        records = records[:train_limit] if train_limit else records
        files["train"][key] = [{field: r[field] for field in TRAIN_FIELDS if field in r} for r in records]
        # Validation gold is closed over by the HCL evaluator only: the current task's full validation set
        # scores the current gain, and each task's frozen panel is its anchor set once it is historical.
        validation = verified_rows(dataset, task["files"]["validationGold"])
        by_id = {row["id"]: row for row in validation}
        anchors = [by_id[identifier] for identifier in panels[task["id"]]]
        files["validation"][key] = validation[:validation_limit] if validation_limit else validation
        files["anchors"][key] = anchors[:validation_limit] if validation_limit else anchors
        evaluations = []
        for index in evaluated_task_indices(position):
            ids = [r["id"] for r in files["eval"][tasks[index]["id"]]]
            evaluations.append({"group": tasks[index]["id"], "ids": ids[:eval_limit] if eval_limit else ids})
        kind = general_panel_kind(position)
        for category in GENERAL_CATEGORIES:
            ids = [r["id"] for r in files["eval"]["general:" + category] if kind == "full" or r["id"] in probe_ids]
            evaluations.append({"group": "general:" + category, "panel": kind,
                                "ids": ids[:general_limit] if general_limit else ids})
        plan_stages.append({"stage": position, "taskId": task["id"], "metric": task["metric"],
                            "trainRecords": len(records), "validationRecords": len(files["validation"][key]),
                            "anchorRecords": len(files["anchors"][key]), "evaluations": evaluations})
    needed = {e["group"] for s in plan_stages for e in s["evaluations"]}
    files["eval"] = {group: rows for group, rows in files["eval"].items() if group in needed}
    plan = {
        "schemaVersion": PLAN_VERSION, "method": "hcl-stable", "paper": "arXiv:2608.19013",
        "profile": profile, "model": BASE_MODEL_ID, "revision": REVISION, "baseTreeSha256": BASE_TREE_SHA256,
        "dataset": {"name": manifest["name"], "seed": manifest["seed"], "manifestSha256": file_hash(dataset / "manifest.json")},
        "seed": seed,
        "generalTestManifestSha256": file_hash(general_test / "manifest.json"),
        "stages": plan_stages,
        "hcl": {"profile": "stable (B_n = 0)", "alternativesPerComponent": 1, "minimumCurrentGain": minimum_current_gain,
                "components": ["task_interface", "experience_memory", "capability_map", "adaptive_router"],
                "harnessTokenBudget": HARNESS_TOKEN_BUDGET, "budgetTokenizer": "Qwen3.5-4B, no special tokens",
                "evidence": "training records executed with the deployed harness in a seeded uniform order, "
                            f"{EVIDENCE_CHUNK} per round, until a round adds nothing to the proposer prompt; each "
                            "proposal packs interactions in that order while the complete prompt fits the per-call ceiling",
                "currentScore": "the current task's full validation set",
                "anchors": {"panel": ANCHOR_PANEL, "panelSha256": file_hash(anchor_panel),
                            "rule": "the frozen 20-item validation panel of each earlier task (HERO's backtest panel), "
                                    "an anchor counting as solved at full credit"},
                "executorMaxTokens": EXECUTION_MAX_TOKENS,
                "proposer": {"model": BASE_MODEL_ID, "temperature": PROPOSER_TEMPERATURE, "maxTokens": PROPOSER_MAX_TOKENS}},
        "serving": {"backend": "mlx-lm continuous batching, one engine per process",
                    "prefixCache": "cache snapshots at shared token prefixes (longest common prefix with recent "
                                   "prompts, 64-token margin, 256-token granularity, at least 1,024 new tokens)",
                    "decoding": "greedy at temperature 0, sampled otherwise", "thinking": False,
                    "maxNewTokens": EVAL_MAX_NEW_TOKENS, "maxInputTokensPerCall": EVAL_MAX_INPUT_TOKENS,
                    "batchSize": ENGINE_BATCH},
        "evaluation": {"decoding": "sealed harness as system prompt, greedy", "maxNewTokens": EVAL_MAX_NEW_TOKENS,
                       "maxInputTokensPerCall": EVAL_MAX_INPUT_TOKENS, "stageZero": "the base control's formal predictions"},
    }
    return plan, files


def prepare(args) -> None:
    root = Path(args.repository).resolve()
    bundle = args.output.resolve()
    if bundle.exists():
        raise SystemExit(f"Bundle exists: {bundle}; use a new path")
    plan, files = build_plan(root / "experiments/data/trace-plus-17", root / "experiments/data/general-test/v1",
                             root / ANCHOR_PANEL, profile=args.profile, stages=args.stages,
                             minimum_current_gain=args.minimum_current_gain, seed=args.seed, train_limit=args.train_limit,
                             validation_limit=args.validation_limit, eval_limit=args.eval_limit,
                             general_limit=args.general_limit)
    for kind, directory in (("train", "train"), ("validation", "validation-gold"), ("anchors", "anchor-gold")):
        for key, rows in files[kind].items():
            write_jsonl(bundle / "inputs" / directory / f"{key}.jsonl", rows)
    for group, rows in files["eval"].items():
        write_jsonl(bundle / "inputs/eval" / group_file(group), rows)
    for name in BUNDLE_CODE:
        shutil.copy2(HERE / name, bundle / name)
    for name in BUNDLE_TREES:
        shutil.copytree(HERE / name, bundle / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for target, source in SCORER_ASSETS.items():
        (bundle / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, bundle / target)
    plan["files"] = {path.relative_to(bundle).as_posix(): file_hash(path)
                     for path in sorted(bundle.rglob("*")) if path.is_file()}
    atomic_json(bundle / "plan.json", plan)
    print(json.dumps({"bundle": str(bundle), "stages": len(plan["stages"]),
                      "planSha256": file_hash(bundle / "plan.json")}))


# ---------------------------------------------------------------------------
# Local units


def call_workers(plan: dict[str, Any]) -> int:
    """Concurrent model calls: twice the engine batch, or more for a server that batches many calls."""
    return max(2 * plan["serving"]["batchSize"], int(os.environ.get("HERO_CALL_WORKERS", "0")))


ENDPOINT_VARIABLE = "HERO_MODEL_ENDPOINT"  # an OpenAI-compatible endpoint serving the pinned checkpoint


def serving_libraries() -> dict[str, str]:
    """The serving stack recorded in the contract: the in-process MLX engine or the endpoint's server."""
    endpoint = os.environ.get(ENDPOINT_VARIABLE)
    if endpoint:
        from local_backend import endpoint_version

        return {"server": endpoint_version(endpoint)}
    import mlx.core
    import mlx_lm

    return {"mlx": mlx.core.__version__, "mlx-lm": mlx_lm.__version__}


class Run:
    """Verified bundle and checkpoint, the recorded runtime contract and one MLX engine shared by every unit
    in this process."""

    def __init__(self, bundle: Path, output: Path, checkpoint: Path, amend: str | None = None):
        import transformers

        self.bundle, self.output, self.checkpoint = bundle.resolve(), output.resolve(), checkpoint.resolve()
        self.plan = json.loads((self.bundle / "plan.json").read_text())
        if self.plan["schemaVersion"] != PLAN_VERSION:
            raise ValueError("Unsupported plan")
        for relative, expected in self.plan["files"].items():
            if file_hash(self.bundle / relative) != expected:
                raise ValueError(f"Bundle file changed: {relative}")
        if tree_hash(self.checkpoint) != self.plan["baseTreeSha256"]:
            raise ValueError("Base checkpoint differs from the pinned tree hash")
        libraries = {**serving_libraries(), "transformers": transformers.__version__}
        contract = {"scientificPlanSha256": scientific_plan_hash(self.plan), "baseTreeSha256": self.plan["baseTreeSha256"],
                    "libraries": libraries}
        path = self.output / "contract.json"
        if path.is_file():
            previous = json.loads(path.read_text())
            if previous != contract:
                if not amend:
                    raise ValueError("Run continues with a different scientific plan, checkpoint or library set; "
                                     "pass --amend with the reason to continue the sealed stages under it")
                sealed = [s["stage"] for s in self.plan["stages"] if (stage_dir(self.output, s["stage"]) / "training.json").is_file()]
                with (self.output / "contract-amendments.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reason": amend,
                                          "previous": previous, "current": contract,
                                          "firstAmendedStage": max(sealed, default=0) + 1}, ensure_ascii=False) + "\n")
                atomic_json(path, contract)
        else:
            atomic_json(path, contract)
        for variable, relative in (("TRACE_GPT2_TOKENIZER", "assets/gpt2-tokenizer.json"),
                                   ("TRACE_PUNKT_TOKENIZER", "assets/punkt-english.pickle")):
            if (self.bundle / relative).is_file():
                os.environ.setdefault(variable, str(self.bundle / relative))
        self.host = platform.node()
        self._engine = None
        self._lock = threading.Lock()

    def stage(self, position: int) -> dict[str, Any]:
        if not 1 <= position <= len(self.plan["stages"]):
            raise ValueError(f"Stage {position} is outside the plan")
        return self.plan["stages"][position - 1]

    def engine(self):
        with self._lock:
            if self._engine is None:
                serving = self.plan["serving"]
                endpoint = os.environ.get(ENDPOINT_VARIABLE)
                if endpoint:
                    from local_backend import HttpEngine

                    self._engine = HttpEngine(self.checkpoint, endpoint=endpoint, max_new_tokens=serving["maxNewTokens"],
                                              max_input_tokens=serving["maxInputTokensPerCall"])
                else:
                    from local_backend import MlxEngine

                    self._engine = MlxEngine(self.checkpoint, max_new_tokens=serving["maxNewTokens"],
                                             max_input_tokens=serving["maxInputTokensPerCall"],
                                             batch_size=serving["batchSize"])
            return self._engine

    def client(self):
        from local_backend import ChatClient

        return ChatClient(self.engine())

    def sealed_harness(self, position: int) -> HarnessState:
        directory = stage_dir(self.output, position)
        record = directory / "training.json"
        if not record.is_file():
            raise ValueError(f"Stage {position} is not trained")
        if file_hash(directory / "harness.json") != json.loads(record.read_text())["harnessSha256"]:
            raise ValueError(f"Stage {position} harness hash mismatch")
        return state_from_json((directory / "harness.json").read_text())


def run_calls(client, calls: Sequence[list[dict[str, str]]], *, max_tokens: int, workers: int,
              done: Callable[[int, dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    """Concurrent greedy calls; an over-long prompt is a per-call failure with an empty output. `done` receives
    each result as it completes, so callers can persist progress."""
    from local_backend import InputTooLong

    def one(indexed):
        index, messages = indexed
        try:
            reply = client.chat.completions.create(model=BASE_MODEL_ID, messages=messages, temperature=0,
                                                   max_tokens=max_tokens)
            result = {"text": reply.choices[0].message.content or "", "status": "complete",
                      "finishReason": reply.choices[0].finish_reason}
        except InputTooLong as error:
            result = {"text": "", "status": "failed", "error": str(error)}
        if done is not None:
            done(index, result)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, enumerate(calls)))


def train(run: Run, position: int) -> None:
    from protocol import canonical_score

    directory, task, settings = stage_dir(run.output, position), run.stage(position), run.plan["hcl"]
    if (directory / "training.json").is_file():
        run.sealed_harness(position)
        print(json.dumps({"stage": position, "phase": "train", "status": "already sealed"}), flush=True)
        return
    deployed = run.sealed_harness(position - 1) if position > 1 else HarnessState()
    engine, client = run.engine(), run.client()
    workers = call_workers(run.plan)
    work = directory / "training.tmp"  # evidence and evaluation results persist across a restarted stage
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = read_jsonl(run.bundle / "inputs/train" / f"{position:02d}.jsonl")

    def optimizer_prompt(state: HarnessState, component: str, interactions: Sequence[Mapping[str, Any]],
                         candidate_index: int) -> str:
        return build_component_optimizer_prompt(deployed=state, component=component,
                                                interaction_evidence={"taskId": task["taskId"], "interactions": list(interactions)}) \
            + f"\n\nCandidate index: {candidate_index}. Return one JSON object only."

    def fits_for(state: HarnessState, component: str, candidate_index: int):
        return lambda subset: len(engine.encode([{"role": "user", "content": optimizer_prompt(
            state, component, subset, candidate_index)}])) <= engine.max_input_tokens

    evidence_path = work / "evidence.json"
    if evidence_path.is_file():
        evidence = json.loads(evidence_path.read_text())
    else:
        order = evidence_order(len(records), run.plan["seed"] * 1000 + position)
        fits = fits_for(deployed, "task_interface", 0)
        interactions: list[dict[str, Any]] = []
        packed: list[dict[str, Any]] = []
        for start in range(0, len(order), EVIDENCE_CHUNK):
            chunk = [records[index] for index in order[start:start + EVIDENCE_CHUNK]]
            outputs = run_calls(client, [harness_messages(deployed, r["prompt"]) for r in chunk],
                                max_tokens=settings["executorMaxTokens"], workers=workers)
            added = False
            for record, output in zip(chunk, outputs):
                item = canonical_score(task["metric"], [{"id": record["id"], "prediction": output["text"]}], [record])
                interaction = {"u": record["prompt"], "i": task["taskId"], "z": output["text"],
                               "y": record.get("answer") or record.get("references"), "f": float(item["score"])}
                interactions.append(interaction)
                if fits(packed + [interaction]):
                    packed.append(interaction)
                    added = True
            if not added:
                break
        evidence = {"taskId": task["taskId"], "interactions": interactions}
        atomic_json(evidence_path, evidence)

    validations = {p: read_jsonl(run.bundle / "inputs/validation-gold" / f"{p:02d}.jsonl") for p in range(1, position + 1)}
    anchor_rows = {p: read_jsonl(run.bundle / "inputs/anchor-gold" / f"{p:02d}.jsonl") for p in range(1, position)}
    metrics = {s["stage"]: s["metric"] for s in run.plan["stages"]}
    budget = settings["harnessTokenBudget"]
    cache_dir = work / "evaluations"
    anchor_ids = [f"{run.plan['stages'][p - 1]['taskId']}:{row['id']}" for p in range(1, position) for row in anchor_rows[p]]

    def evaluator(state: HarnessState):
        # An invalid candidate keeps the anchor key set (the gate compares identical sets) and solves none.
        if is_invalid(state):
            return 0.0, dict.fromkeys(anchor_ids, False), False
        key = hashlib.sha256(state.to_json().encode()).hexdigest()
        cached = cache_dir / f"{key}.json"
        if cached.is_file():
            value = json.loads(cached.read_text())
            return value["current"], value["anchors"], value["valid"]
        valid = engine.count(render_harness(state)) <= budget
        prompts = [(position, row) for row in validations[position]] + \
                  [(p, row) for p in range(1, position) for row in anchor_rows[p]]
        outputs: list[dict[str, Any]] = []
        if valid:
            # Each executed item persists as it completes, so a restarted stage continues the evaluation.
            items = work / "outputs" / key
            items.mkdir(parents=True, exist_ok=True)
            paths = [items / f"{p:02d}-{hashlib.sha256(row['id'].encode()).hexdigest()[:16]}.json" for p, row in prompts]
            outputs = [json.loads(path.read_text()) if path.is_file() else None for path in paths]
            missing = [i for i, output in enumerate(outputs) if output is None]

            def persist(index: int, result: dict[str, Any]) -> None:
                atomic_json(paths[missing[index]], result)

            fresh = run_calls(client, [harness_messages(state, prompts[i][1]["prompt"]) for i in missing],
                              max_tokens=settings["executorMaxTokens"], workers=workers, done=persist)
            for i, result in zip(missing, fresh):
                outputs[i] = result
        by_stage: dict[int, list] = {}
        for (p, row), output in zip(prompts, outputs):
            by_stage.setdefault(p, []).append((row, output))
        current, anchors = 0.0, dict.fromkeys(anchor_ids, False)
        for p, pairs in by_stage.items():
            summary = canonical_score(metrics[p], [{"id": row["id"], "prediction": out["text"]} for row, out in pairs],
                                      [row for row, _ in pairs])
            if p == position:
                current = float(summary["score"])
            else:
                for item in summary["itemScores"]:
                    anchors[f"{run.plan['stages'][p - 1]['taskId']}:{item['id']}"] = float(item["score"]) >= 1.0 - 1e-12
        atomic_json(cached, {"current": current, "anchors": anchors, "valid": valid})
        return current, anchors, valid

    proposals: list[dict[str, Any]] = []

    def proposer(state: HarnessState, component: str, prompt: str, candidate_index: int):
        from local_backend import InputTooLong

        # A proposal persists with its deployed state, so a restarted stage evaluates the same candidate.
        cached = work / "proposals" / (hashlib.sha256(f"{state.to_json()}|{component}|{candidate_index}".encode())
                                       .hexdigest() + ".json")
        if cached.is_file():
            record = json.loads(cached.read_text())
            proposals.append(record)
            return parse_component(record["response"])
        packed = pack_evidence(evidence["interactions"], fits_for(state, component, candidate_index))
        record = {"component": component, "candidateIndex": candidate_index, "evidenceRecords": len(packed)}
        try:
            reply = client.chat.completions.create(
                model=BASE_MODEL_ID, messages=[{"role": "user", "content": optimizer_prompt(state, component, packed, candidate_index)}],
                temperature=settings["proposer"]["temperature"], max_tokens=settings["proposer"]["maxTokens"])
            text = reply.choices[0].message.content or ""
        except InputTooLong as error:
            text = ""
            record["error"] = str(error)
        component_value = parse_component(text)
        record["valid"] = INVALID not in component_value
        record["response"] = text
        atomic_json(cached, record)
        proposals.append(record)
        return component_value

    state, audits = evolve_stable_harness(deployed=deployed, interaction_evidence=evidence, proposer=proposer,
                                          evaluator=evaluator, alternatives_per_component=settings["alternativesPerComponent"],
                                          minimum_current_gain=settings["minimumCurrentGain"],
                                          seed=run.plan["seed"] + position * 10)
    (work / "harness.json").write_text(state.to_json() + "\n", encoding="utf-8")
    write_jsonl(work / "proposals.jsonl", proposals)
    for name in ("harness.json", "proposals.jsonl", "evidence.json"):
        os.replace(work / name, directory / name)
    shutil.rmtree(work)
    seconds = time.perf_counter() - started
    atomic_json(directory / "training.json", {
        "stage": position, "trainRecords": len(records), "executedEvidenceRecords": len(evidence["interactions"]),
        "seconds": seconds, "audits": [asdict(audit) for audit in audits],
        "harnessQwenTokens": engine.count(render_harness(state)),
        "evidenceScore": sum(i["f"] for i in evidence["interactions"]) / max(1, len(evidence["interactions"])),
        **client.usage(), "host": run.host, "harnessSha256": file_hash(directory / "harness.json")})
    print(json.dumps({"stage": position, "phase": "train", "seconds": round(seconds, 1),
                      "committed": [a.committed_candidate_id for a in audits],
                      "executedEvidenceRecords": len(evidence["interactions"]),
                      "harnessQwenTokens": engine.count(render_harness(state)), **client.usage()}), flush=True)


def evaluate(run: Run, position: int, groups: list[str] | None = None) -> None:
    directory, stage = stage_dir(run.output, position), run.stage(position)
    planned = [e["group"] for e in stage["evaluations"]]
    selected = groups or planned
    if set(selected) - set(planned):
        raise ValueError(f"Groups outside stage {position}: {sorted(set(selected) - set(planned))}")
    if (directory / "evaluation.json").is_file():
        print(json.dumps({"stage": position, "phase": "evaluate", "status": "already sealed"}), flush=True)
        return
    harness = run.sealed_harness(position)
    client = run.client()
    workers = call_workers(run.plan)
    for evaluation in stage["evaluations"]:
        group = evaluation["group"]
        output = directory / "predictions" / group_file(group)
        if group not in selected or output.is_file():
            continue
        prompts = {r["id"]: r["prompt"] for r in read_jsonl(run.bundle / "inputs/eval" / group_file(group))}
        items = directory / "items" / group.replace(":", "__")
        items.mkdir(parents=True, exist_ok=True)
        pending = [(index, identifier) for index, identifier in enumerate(evaluation["ids"])
                   if not (items / f"{index:04d}.json").is_file()]
        for start in range(0, len(pending), workers):
            chunk = pending[start:start + workers]
            began = time.perf_counter()
            outputs = run_calls(client, [harness_messages(harness, prompts[i]) for _, i in chunk],
                                max_tokens=run.plan["evaluation"]["maxNewTokens"], workers=workers)
            for (index, identifier), out in zip(chunk, outputs):
                row = {"id": identifier, "group": group, "stage": position, "prediction": out["text"],
                       "itemStatus": out["status"], "chunkSeconds": time.perf_counter() - began}
                if "error" in out:
                    row["error"] = out["error"]
                if "finishReason" in out:
                    row["stopReason"] = "eos" if out["finishReason"] == "stop" else out["finishReason"]
                atomic_json(items / f"{index:04d}.json", row)
        rows = []
        for index, identifier in enumerate(evaluation["ids"]):
            row = json.loads((items / f"{index:04d}.json").read_text())
            if row["id"] != identifier:
                raise ValueError(f"Sealed prediction does not match the plan: {group} {index}")
            rows.append(row)
        write_jsonl(output, rows)
        print(json.dumps({"stage": position, "group": group, "items": len(rows),
                          "failed": sum(r["itemStatus"] != "complete" for r in rows)}), flush=True)
    paths = [directory / "predictions" / group_file(g) for g in planned]
    if all(path.is_file() for path in paths):
        atomic_json(directory / "evaluation.json", {
            "stage": position, "trainingSha256": file_hash(directory / "training.json"), **client.usage(),
            "predictions": {path.relative_to(directory).as_posix(): file_hash(path) for path in paths}})
        print(json.dumps({"stage": position, "phase": "evaluate", "status": "sealed", **client.usage()}), flush=True)


def run_stream(run: Run, through: int) -> None:
    """Train stages in order; evaluate each sealed stage alongside the next stage's training."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        evaluations = []
        for position in range(1, through + 1):
            for done in [future for future in evaluations if future.done()]:
                done.result()  # an evaluation failure stops the stream
            train(run, position)
            evaluations.append(pool.submit(evaluate, run, position))
        for future in evaluations:
            future.result()
    engine = run.engine()
    print(json.dumps({"phase": "run", "through": through, "engineCalls": engine.calls,
                      "promptTokens": engine.prompt_tokens, "reusedPromptTokens": engine.reused_prompt_tokens,
                      "completionTokens": engine.completion_tokens}), flush=True)


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
    cells, general_cells, failures, stages = [], [], 0, []
    for stage in plan["stages"]:
        directory = stage_dir(run, stage["stage"])
        if (directory / "training.json").is_file():
            record = json.loads((directory / "training.json").read_text())
            stages.append({"stage": record["stage"], "seconds": record["seconds"], "harnessQwenTokens": record["harnessQwenTokens"],
                           "committed": [a["committed_candidate_id"] for a in record["audits"]], "modelCalls": record["modelCalls"]})
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
    metrics = {**stream_metrics(cells, order), "itemFailures": failures}
    by_stage: dict[int, list] = {}
    for cell in general_cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    metrics["generalByStage"] = [{"stage": s, "panel": rows[0]["panel"],
                                  "macro": sum(r["score"] for r in rows) / len(rows)}
                                 for s, rows in sorted(by_stage.items()) if len(rows) == len(GENERAL_CATEGORIES)]
    report = {"method": "hcl-stable", "profile": plan["profile"], "run": str(run),
              "planSha256": file_hash(args.bundle.resolve() / "plan.json"), "stages": stages,
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
    p.add_argument("--profile", choices=("smoke", "full"), required=True)
    p.add_argument("--minimum-current-gain", type=float, required=True,
                   help="The commit gate's minimum current-task gain (no project default is invented)")
    p.add_argument("--seed", type=int, default=42, help="method seed (the paper reports seeds 42, 43 and 44)")
    p.add_argument("--stages", type=int, default=STAGE_COUNT)
    p.add_argument("--train-limit", type=int, help="Smoke only: first N training records per task")
    p.add_argument("--validation-limit", type=int, help="Smoke only: first N validation and anchor records per task")
    p.add_argument("--eval-limit", type=int, help="Smoke only: first N test items per task")
    p.add_argument("--general-limit", type=int, help="Smoke only: first N G items per category")
    for name in ("run", "train", "evaluate"):
        u = commands.add_parser(name)
        if name == "run":
            u.add_argument("--through", type=int, help="Last stage (default: the whole plan)")
        else:
            u.add_argument("--stage", type=int, required=True)
        u.add_argument("--bundle", type=Path, default=HERE)
        u.add_argument("--output", type=Path, required=True)
        u.add_argument("--checkpoint", type=Path, required=True)
        u.add_argument("--amend", help="Reason for continuing sealed stages under a changed contract (logged)")
        if name == "evaluate":
            u.add_argument("--groups", help="Comma-separated evaluation groups (default: all)")
    s = commands.add_parser("score")
    s.add_argument("--repository", default=".")
    s.add_argument("--bundle", type=Path, required=True)
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.profile == "full" and (args.stages != STAGE_COUNT or args.train_limit or args.validation_limit
                                       or args.eval_limit or args.general_limit):
            parser.error("The full profile uses the complete stream and data")
        prepare(args)
    elif args.command == "score":
        score(args)
    else:
        run = Run(args.bundle, args.output, args.checkpoint, args.amend)
        if args.command == "run":
            run_stream(run, args.through or len(run.plan["stages"]))
        elif args.command == "train":
            train(run, args.stage)
        else:
            evaluate(run, args.stage, args.groups.split(",") if args.groups else None)


if __name__ == "__main__":
    main()
