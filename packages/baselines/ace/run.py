"""ACE (arXiv:2510.04618, offline mode) over the TRACE+17 stream: prepare a bundle, run, score.

prepare   Build a self-contained bundle: plan, stage training records, validation gold (scorer only),
          prompt-only evaluation inputs (no test gold), this directory's code (the `upstream/` package,
          adapter, scorer) and scoring assets.
run       The stream through stage N in one process: stages train in order, and each sealed stage is
          evaluated alongside the next stage's training. Sealed units are skipped, so an interrupted run
          resumes after its last completed training sample.
train     Stage N alone: restore the stage N-1 final playbook, run upstream `ACE.run(mode="offline")` over
          the stage's training records (validation every 100 steps) with every role served by Qwen3.5-4B,
          seal the final and the best-on-validation playbooks.
evaluate  Stage N alone: ACE's Generator with the stage N best-on-validation playbook on the base control's
          sparse schedule; per-item sealed and resumable.
score     Score sealed predictions with this directory's evaluator and derive stream metrics.

Model calls use in-process PyTorch/MPS or the CUDA server at ACE_LOCAL_BASE_URL.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import threading
import time
from typing import Any
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# The rouge package reconstructs the LCS recursively, one frame per token of prediction plus reference, so
# scoring a long ROUGE-L prediction (in training and in evaluation worker threads) needs a deep recursion limit.
sys.setrecursionlimit(max(sys.getrecursionlimit(), 100_000))
threading.stack_size(64 << 20)

from ace_adapter import (  # noqa: E402
    MAX_MODEL_LEN, ROLE_MAX_TOKENS, TraceDataProcessor, ace_config, ace_system,
    activate_upstream, to_ace_sample,
)
from protocol import (  # noqa: E402
    BASE_MODEL_ID, EVAL_MAX_INPUT_TOKENS, STAGE_COUNT, evaluated_task_indices, general_panel_kind, load_manifest,
    read_jsonl,
)

UPSTREAM_COMMIT = "82709de050e1db6e6ef2f07bcb0393560b94992a"
VLLM_VERSION = "0.31.0"  # serve.sh; its context-length rejection is the message the skip recognises
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BASE_TREE_SHA256 = "e2361c06ba172642a18b1472f4758449bcddf20072e371921642bfba28004dc5"
GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")
PLAN_VERSION = "ace-official-plan-v3"
BUNDLE_CODE = ("run.py", "local_backend.py", "ace_adapter.py", "protocol.py", "scoring.py", "scorer_assets.py", "serve.sh",
               "pyproject.toml", "uv.lock")
BUNDLE_TREES = ("upstream", "ifeval")
SCORER_ASSETS = {"assets/gpt2-tokenizer.json": "experiments/data/tokenizers/gpt2/tokenizer.json",
                 "assets/punkt-english.pickle": "experiments/data/nltk/punkt/english.pickle"}
TRAIN_FIELDS = ("id", "prompt", "answer", "references", "metadata")  # training and validation records
TIKTOKEN_CACHE = "assets/tiktoken"


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


def build_plan(dataset: Path, general_test: Path, *, profile: str, stages: int, train_limit: int | None,
               eval_limit: int | None, general_limit: int | None, device: str = "cuda"
               ) -> tuple[dict[str, Any], dict[str, list], dict[str, list], dict[str, list]]:
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
    validation_files: dict[str, list] = {}
    plan_stages: list[dict[str, Any]] = []
    for position in range(1, stages + 1):
        task = tasks[position - 1]
        records = verified_rows(dataset, task["files"]["train"])
        records = records[:train_limit] if train_limit else records
        # Training gold is agent-visible: ACE's correctness feedback scores its answers against it.
        train_files[f"{position:02d}"] = [{key: r[key] for key in TRAIN_FIELDS if key in r} for r in records]
        # Validation gold reaches only the scorer: upstream scores the Generator's answers to the prompts.
        validation = verified_rows(dataset, task["files"]["validationGold"])
        validation_files[f"{position:02d}"] = [{key: r[key] for key in TRAIN_FIELDS if key in r} for r in validation]
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
                            "trainRecords": len(records), "validationRecords": len(validation),
                            "evaluations": evaluations})
    needed = {e["group"] for s in plan_stages for e in s["evaluations"]}
    eval_files = {group: rows for group, rows in eval_files.items() if group in needed}
    plan = {
        "schemaVersion": PLAN_VERSION, "method": "ace", "paper": "arXiv:2510.04618 (ICLR 2026)",
        "upstream": {"repository": "https://github.com/ace-agent/ace", "commit": UPSTREAM_COMMIT,
                     "trainer": "ace.ACE (sequential), ACE.run(mode='offline')"},
        "profile": profile, "model": BASE_MODEL_ID, "revision": REVISION, "baseTreeSha256": BASE_TREE_SHA256,
        "dataset": {"name": manifest["name"], "seed": manifest["seed"], "manifestSha256": file_hash(dataset / "manifest.json")},
        "generalTestManifestSha256": file_hash(general_test / "manifest.json"),
        "stages": plan_stages,
        "ace": {**{key: value for key, value in ace_config(save_dir="", task_name="").items()
                   if key not in ("save_dir", "task_name")},
                "roleMaxTokens": ROLE_MAX_TOKENS, "batchSize": 1,
                "validation": "the task's validation split, scored every eval_steps; the stage is evaluated with "
                              "upstream's best_playbook and the next stage continues from its final playbook",
                "roles": {"generator": BASE_MODEL_ID, "reflector": BASE_MODEL_ID, "curator": BASE_MODEL_ID},
                "trainRecordsPerStage": "every record of the task's training file, once, in the dataset's seeded "
                                        "order (upstream's default single epoch)",
                "carriedAcrossStages": "playbook and bullet id counter"},
        "serving": {"device": device, "server": ("in-process PyTorch MPS" if device == "mps" else
                    "vLLM OpenAI-compatible chat completions (serve.sh)"), "maxModelLen": MAX_MODEL_LEN,
                    "maxInputTokensPerCall": EVAL_MAX_INPUT_TOKENS, "decoding": "greedy at temperature 0",
                    "thinking": "off by the backend's chat-template arguments"},
        "evaluation": {"decoding": "ACE generator with the sealed playbook, as upstream evaluate_single_test_sample",
                       "maxNewTokens": ROLE_MAX_TOKENS, "answer": "upstream extract_answer",
                       "stageZero": "the base control's formal predictions"},
    }
    return plan, train_files, validation_files, eval_files


def prepare(args) -> None:
    root = Path(args.repository).resolve()
    bundle = args.output.resolve()
    if bundle.exists():
        raise SystemExit(f"Bundle exists: {bundle}; use a new path")
    plan, train_files, validation_files, eval_files = build_plan(
        root / "experiments/data/trace-plus-17", root / "experiments/data/general-test/v1", profile=args.profile,
        stages=args.stages, train_limit=args.train_limit, eval_limit=args.eval_limit, general_limit=args.general_limit,
        device=args.device)
    for key, rows in train_files.items():
        write_jsonl(bundle / "inputs/train" / f"{key}.jsonl", rows)
    for key, rows in validation_files.items():
        write_jsonl(bundle / "inputs/validation-gold" / f"{key}.jsonl", rows)
    for group, rows in eval_files.items():
        write_jsonl(bundle / "inputs/eval" / group_file(group), rows)
    for name in BUNDLE_CODE:
        shutil.copy2(HERE / name, bundle / name)
    for name in BUNDLE_TREES:
        shutil.copytree(HERE / name, bundle / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".venv"))
    for target, source in SCORER_ASSETS.items():
        (bundle / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / source, bundle / target)
    # Upstream utils.py loads tiktoken's cl100k_base at import; the bundle carries it for offline nodes.
    os.environ["TIKTOKEN_CACHE_DIR"] = str(bundle / TIKTOKEN_CACHE)
    import tiktoken

    tiktoken.get_encoding("cl100k_base")
    plan["files"] = {path.relative_to(bundle).as_posix(): file_hash(path)
                     for path in sorted(bundle.rglob("*")) if path.is_file()}
    atomic_json(bundle / "plan.json", plan)
    print(json.dumps({"bundle": str(bundle), "stages": len(plan["stages"]),
                      "planSha256": file_hash(bundle / "plan.json")}))


# ---------------------------------------------------------------------------
# Server checks


def server_contract(checkpoint: Path) -> dict[str, Any]:
    """The serving facts the run depends on, read from the server and checked before any unit runs.

    The server must serve `checkpoint` (the pinned tree) as BASE_MODEL_ID with a context of 16,384 + 4,096
    tokens, so that every ACE call (4,096 new tokens) is rejected when its prompt exceeds the input ceiling.
    """
    import openai

    if os.getenv("ACE_LOCAL_DEVICE") == "mps":
        from local_backend import get_client
        return get_client().contract

    base = os.environ["ACE_LOCAL_BASE_URL"].rstrip("/")
    with urllib.request.urlopen(base + "/models", timeout=60) as response:
        models = json.load(response)["data"]
    model = next((m for m in models if m["id"] == BASE_MODEL_ID), None)
    if model is None or model.get("max_model_len") != MAX_MODEL_LEN:
        raise ValueError(f"The server must serve {BASE_MODEL_ID} with max_model_len {MAX_MODEL_LEN}: {models}")
    if Path(str(model.get("root"))).resolve() != checkpoint:
        raise ValueError(f"The server does not serve {checkpoint}: {model.get('root')}")
    with urllib.request.urlopen(base.removesuffix("/v1") + "/version", timeout=60) as response:
        version = json.load(response).get("version")
    if version != VLLM_VERSION:
        raise ValueError(f"The server must be vLLM {VLLM_VERSION}: {version}")
    client = openai.OpenAI(base_url=base, api_key=os.getenv("ACE_LOCAL_API_KEY", "EMPTY"))
    def tokens(**extra) -> list[int]:
        request = urllib.request.Request(
            base.removesuffix("/v1") + "/tokenize", method="POST", headers={"Content-Type": "application/json"},
            data=json.dumps({"model": BASE_MODEL_ID, "messages": [{"role": "user", "content": "Hi"}],
                             "add_generation_prompt": True, **extra}).encode())
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)["tokens"]

    # Upstream sends no chat-template arguments: the server's defaults must render the non-thinking prompt.
    default = tokens()
    if default != tokens(chat_template_kwargs={"enable_thinking": False}) or default == tokens(
            chat_template_kwargs={"enable_thinking": True}):
        raise ValueError("The server's default chat template does not disable thinking (see serve.sh)")
    try:  # ~17,000 tokens: above the input ceiling
        client.chat.completions.create(model=BASE_MODEL_ID, messages=[{"role": "user", "content": "a " * 17000}],
                                       temperature=0.0, max_tokens=ROLE_MAX_TOKENS)
        raise ValueError("The server accepted a prompt above the 16,384-token input ceiling")
    except openai.BadRequestError as error:
        # The skip in upstream/ace/ace.py recognises exactly this rejection.
        if "maximum context length" not in str(error):
            raise ValueError(f"Unexpected rejection of an over-ceiling prompt: {error}") from error
    return {"model": BASE_MODEL_ID, "root": model.get("root"), "maxModelLen": model["max_model_len"],
            "server": version, "thinking": False}


# ---------------------------------------------------------------------------
# Local units


class Run:
    """Verified bundle and the recorded runtime contract."""

    def __init__(self, bundle: Path, output: Path, checkpoint: Path, amend: str | None = None, device: str | None = None):
        import openai

        self.bundle, self.output, checkpoint = bundle.resolve(), output.resolve(), checkpoint.resolve()
        if self.bundle != HERE:
            raise ValueError(f"Run the bundle's own run.py ({self.bundle / 'run.py'}), so the verified code is the code that runs")
        self.plan = json.loads((self.bundle / "plan.json").read_text())
        if self.plan["schemaVersion"] != PLAN_VERSION:
            raise ValueError("Unsupported plan")
        if device is not None and device != self.plan["serving"]["device"]:
            raise ValueError("Requested device differs from the prepared bundle")
        for relative, expected in self.plan["files"].items():
            if file_hash(self.bundle / relative) != expected:
                raise ValueError(f"Bundle file changed: {relative}")
        if tree_hash(checkpoint) != self.plan["baseTreeSha256"]:
            raise ValueError("Base checkpoint differs from the pinned tree hash")
        os.environ["ACE_LOCAL_DEVICE"] = self.plan["serving"]["device"]
        if self.plan["serving"]["device"] == "mps":
            from local_backend import configure
            configure(checkpoint, self.output / "model-calls")
        contract = {"scientificPlanSha256": scientific_plan_hash(self.plan), "baseTreeSha256": self.plan["baseTreeSha256"],
                    "serving": server_contract(checkpoint),
                    "libraries": {"openai": openai.__version__}}
        path = self.output / "contract.json"
        if path.is_file():
            previous = json.loads(path.read_text())
            if previous != contract:
                if not amend:
                    raise ValueError("Run continues with a different scientific plan, server or library set; "
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
        if (self.bundle / TIKTOKEN_CACHE).is_dir():
            os.environ["TIKTOKEN_CACHE_DIR"] = str(self.bundle / TIKTOKEN_CACHE)

    def stage(self, position: int) -> dict[str, Any]:
        if not 1 <= position <= len(self.plan["stages"]):
            raise ValueError(f"Stage {position} is outside the plan")
        return self.plan["stages"][position - 1]

    def sealed_state(self, position: int) -> dict[str, Any]:
        directory = stage_dir(self.output, position)
        record = directory / "training.json"
        if not record.is_file():
            raise ValueError(f"Stage {position} is not trained")
        if file_hash(directory / "ace-state.json") != json.loads(record.read_text())["stateSha256"]:
            raise ValueError(f"Stage {position} ACE state hash mismatch")
        return json.loads((directory / "ace-state.json").read_text())


def call_usage(log_dir: Path) -> dict[str, Any]:
    """Model calls and tokens from upstream's per-call logs. Calls and rejected calls (per role) are split
    into training (`train_*` call ids) and validation (`test_eval_*`)."""
    usage: dict[str, Any] = {"modelCalls": {"training": 0, "validation": 0},
                             "cachedCalls": {"training": 0, "validation": 0},
                             "rejectedCalls": {"training": {}, "validation": {}},
                             "promptTokens": 0, "completionTokens": 0}
    for path in log_dir.rglob("*.json"):
        call = json.loads(path.read_text())
        if "role" not in call:
            continue
        phase = "validation" if call["call_id"].startswith("test_") else "training"
        if call.get("cache_hit"):
            usage["cachedCalls"][phase] += 1
            continue
        usage["modelCalls"][phase] += 1
        if "error" in call:
            rejected = usage["rejectedCalls"][phase]
            rejected[call["role"]] = rejected.get(call["role"], 0) + 1
        usage["promptTokens"] += call.get("prompt_num_tokens", 0)
        usage["completionTokens"] += call.get("response_num_tokens", 0)
    return usage


def selected_step(validations: list[dict[str, Any]]) -> int:
    """The step whose playbook upstream keeps as best_playbook: the earliest check reaching the highest
    accuracy above 0 (upstream starts from 0.0 and replaces only on a strictly higher accuracy); 0 when no
    check exceeds 0, i.e. the stage's starting playbook."""
    best, selected = 0.0, 0
    for check in validations:
        if check["accuracy"] > best:
            best, selected = check["accuracy"], check["step"]
    return selected


def train(run: Run, position: int) -> None:
    directory, stage = stage_dir(run.output, position), run.stage(position)
    if (directory / "training.json").is_file():
        run.sealed_state(position)
        print(json.dumps({"stage": position, "phase": "train", "status": "already sealed"}), flush=True)
        return
    # Completed samples survive interruption. After entering stage 4, no earlier
    # unsealed stage may be started again, even if its work directory disappeared.
    progress_path = run.output / "training-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {"maxStartedStage": 0}
    highest = progress["maxStartedStage"]
    if highest > 3 and position < highest:
        raise ValueError("Retraining forbidden after stage 3: an earlier training seal is missing")
    work = directory / "training.tmp"
    resume_path = work / "resume.json"
    if work.exists() and not resume_path.is_file():
        raise ValueError("Interrupted training has no sample checkpoint; refusing to retrain")
    if highest > 3 and position == highest and not resume_path.is_file():
        raise ValueError("Retraining forbidden after stage 3: sample checkpoint is missing")
    records = read_jsonl(run.bundle / "inputs/train" / f"{position:02d}.jsonl")
    validation = read_jsonl(run.bundle / "inputs/validation-gold" / f"{position:02d}.jsonl")
    work.mkdir(parents=True, exist_ok=True)
    atomic_json(progress_path, {"maxStartedStage": max(highest, position), "noRetrainingAfterStage": 3})
    state = run.sealed_state(position - 1) if position > 1 else None
    ace = ace_system(state["playbook"] if state else None, state["nextGlobalId"] if state else None)
    ace.resume_save_path = str(work / "ace" / "offline")
    prior_seconds = json.loads(resume_path.read_text()).get("elapsedSeconds", 0.0) if resume_path.is_file() else 0.0
    started = time.perf_counter()
    results = ace.run(mode="offline", train_samples=[to_ace_sample(r) for r in records],
                      val_samples=[to_ace_sample(r) for r in validation],
                      data_processor=TraceDataProcessor(stage["metric"]),
                      config={**ace_config(save_dir=str(work / "ace"), task_name=stage["taskId"].replace(":", "__")),
                              "resume_path": str(resume_path)})
    seconds = prior_seconds + time.perf_counter() - started
    steps = json.loads(next((work / "ace").rglob("pre_train_post_train_results.json")).read_text())
    checks = json.loads(next((work / "ace").rglob("train_results.json")).read_text())["results"]
    validations = [{"step": c["step"], "accuracy": c["val_result"].get("accuracy"),
                    "scoredItems": c["val_result"].get("total")} for c in checks]
    selected = selected_step(validations)
    # Upstream saves the playbook every save_steps (50) steps, so each validation step has a saved copy.
    expected = (next((work / "ace").rglob(f"epoch_1_step_{selected}_playbook.txt")).read_text() if selected
                else (state["playbook"] if state else ace._initialize_empty_playbook()))
    if ace.best_playbook != expected:
        raise ValueError(f"Stage {position}: best_playbook is not the playbook of the reported step {selected}")
    usage = call_usage(work / "ace")
    activate_upstream()
    from utils import count_tokens

    atomic_json(work / "ace-state.json", {"playbook": ace.playbook, "nextGlobalId": ace.next_global_id,
                                          "bestPlaybook": ace.best_playbook})
    (work / "playbook.txt").write_text(ace.playbook, encoding="utf-8")
    (work / "best-playbook.txt").write_text(ace.best_playbook, encoding="utf-8")
    with tarfile.open(work / "ace-logs.tar.xz", "w:xz") as archive:
        archive.add(work / "ace", arcname="ace")
    atomic_json(work / "ace-results.json", results)
    for name in ("ace-state.json", "playbook.txt", "best-playbook.txt", "ace-logs.tar.xz", "ace-results.json"):
        os.replace(work / name, directory / name)
    shutil.rmtree(work)
    record = {"stage": position, "trainRecords": len(records), "validationRecords": len(validation),
              "skippedSamples": sum("api_error" in step for step in steps),
              "playbookTokens": count_tokens(ace.playbook), "nextGlobalId": ace.next_global_id,
              "validations": validations, "selectedStep": selected,
              "bestPlaybookTokens": count_tokens(ace.best_playbook),
              **usage, "seconds": seconds,
              "stateSha256": file_hash(directory / "ace-state.json"),
              "logsSha256": file_hash(directory / "ace-logs.tar.xz")}
    atomic_json(directory / "training.json", record)
    print(json.dumps({key: value for key, value in record.items() if key not in ("stateSha256", "logsSha256")}),
          flush=True)


def evaluate(run: Run, position: int, groups: list[str] | None = None) -> None:
    import openai

    directory, stage = stage_dir(run.output, position), run.stage(position)
    planned = [e["group"] for e in stage["evaluations"]]
    selected = groups or planned
    if set(selected) - set(planned):
        raise ValueError(f"Groups outside stage {position}: {sorted(set(selected) - set(planned))}")
    if (directory / "evaluation.json").is_file():
        print(json.dumps({"stage": position, "phase": "evaluate", "status": "already sealed"}), flush=True)
        return
    state = run.sealed_state(position)
    ace = ace_system(state["bestPlaybook"], state["nextGlobalId"])  # upstream tests with best_playbook
    activate_upstream()
    from utils import extract_answer

    def predict(index: int, identifier: str, prompt: str, group: str) -> dict[str, Any]:
        # Upstream evaluate_single_test_sample, without the target it would compare against.
        started = time.perf_counter()
        try:
            response, _, call = ace.generator.generate(question=prompt, playbook=ace.playbook, context="",
                                                       reflection="(empty)", use_json_mode=False,
                                                       call_id=f"test_eval_{index}", log_dir=None)
        except openai.BadRequestError as error:
            return {"id": identifier, "group": group, "stage": position, "prediction": "", "response": "",
                    "itemStatus": "failed", "error": str(error), "seconds": time.perf_counter() - started}
        return {"id": identifier, "group": group, "stage": position, "prediction": extract_answer(response),
                "response": response, "itemStatus": "complete", "promptTokens": call.get("prompt_num_tokens"),
                "completionTokens": call.get("response_num_tokens"), "seconds": time.perf_counter() - started}

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
        with ThreadPoolExecutor(max_workers=run.plan["ace"]["test_workers"]) as pool:
            futures = {pool.submit(predict, index, identifier, prompts[identifier], group): index
                       for index, identifier in pending}
            for future in as_completed(futures):
                atomic_json(items / f"{futures[future]:04d}.json", future.result())
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
            "stage": position, "trainingSha256": file_hash(directory / "training.json"),
            "predictions": {path.relative_to(directory).as_posix(): file_hash(path) for path in paths}})
        print(json.dumps({"stage": position, "phase": "evaluate", "status": "sealed"}), flush=True)


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
    general_test = root / "experiments/data/general-test/v1"
    if file_hash(general_test / "manifest.json") != plan["generalTestManifestSha256"]:
        raise ValueError("G-test manifest differs from the plan")
    general = {r["id"]: r for r in verified_rows(general_test, manifest_file(
        json.loads((general_test / "manifest.json").read_text()), "hidden-scorer/general-test.jsonl"))}
    cells, general_cells, failures, stages = [], [], 0, []
    for stage in plan["stages"]:
        directory = stage_dir(run, stage["stage"])
        if (directory / "training.json").is_file():
            record = json.loads((directory / "training.json").read_text())
            stages.append({key: record[key] for key in ("stage", "seconds", "playbookTokens", "bestPlaybookTokens",
                                                        "selectedStep", "skippedSamples", "rejectedCalls",
                                                        "modelCalls")})
        if not (directory / "evaluation.json").is_file():
            continue
        sealed = json.loads((directory / "evaluation.json").read_text())["predictions"]
        stage_failures = 0
        for evaluation in stage["evaluations"]:
            path = directory / "predictions" / group_file(evaluation["group"])
            if file_hash(path) != sealed[path.relative_to(directory).as_posix()]:
                raise ValueError(f"Prediction file changed: {path}")
            rows = read_jsonl(path)
            if [r["id"] for r in rows] != evaluation["ids"]:
                raise ValueError(f"Prediction coverage differs from the plan: {path}")
            stage_failures += sum(r["itemStatus"] != "complete" for r in rows)
            group = evaluation["group"]
            if group.startswith("general:"):
                references = [general[r["id"]] for r in rows]
                result = canonical_score(references[0]["metric"], rows, references)
                general_cells.append({"stage": stage["stage"], "category": group.split(":", 1)[1],
                                      "panel": evaluation["panel"], "items": len(rows), "score": result["score"]})
            else:
                result = canonical_score(metrics_by_task[group], rows, [gold[group][r["id"]] for r in rows])
                cells.append({"stage": stage["stage"], "task": group, "items": len(rows), "score": result["score"]})
        failures += stage_failures
        if stages and stages[-1]["stage"] == stage["stage"]:
            stages[-1]["evaluationFailures"] = stage_failures
    order = [t["id"] for t in manifest["tasks"]][:len(plan["stages"])]
    metrics = {**stream_metrics(cells, order), "itemFailures": failures}
    by_stage: dict[int, list] = {}
    for cell in general_cells:
        by_stage.setdefault(cell["stage"], []).append(cell)
    metrics["generalByStage"] = [{"stage": s, "panel": rows[0]["panel"],
                                  "macro": sum(r["score"] for r in rows) / len(rows)}
                                 for s, rows in sorted(by_stage.items()) if len(rows) == len(GENERAL_CATEGORIES)]
    report = {"method": "ace", "profile": plan["profile"], "run": str(run),
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
    p.add_argument("--device", choices=("mps", "cuda"), default="cuda", help="Model backend recorded in the bundle")
    p.add_argument("--stages", type=int, default=STAGE_COUNT)
    p.add_argument("--train-limit", type=int, help="Smoke only: first N training records per task (below 100 "
                   "no validation check runs, so each stage is evaluated with its starting playbook)")
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
        u.add_argument("--checkpoint", type=Path, required=True, help="The pinned checkpoint directory the server serves")
        u.add_argument("--device", choices=("mps", "cuda"), help="Must match the prepared bundle device")
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
        if args.profile == "full" and (args.stages != STAGE_COUNT or args.train_limit or args.eval_limit or args.general_limit):
            parser.error("The full profile uses the complete stream and data")
        prepare(args)
    elif args.command == "score":
        score(args)
    else:
        run = Run(args.bundle, args.output, args.checkpoint, args.amend, args.device)
        if args.command == "run":
            run_stream(run, args.through or len(run.plan["stages"]))
        elif args.command == "train":
            train(run, args.stage)
        else:
            evaluate(run, args.stage, args.groups.split(",") if args.groups else None)


if __name__ == "__main__":
    main()
