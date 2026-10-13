"""Frozen empty-harness control on Qwen3.5-4B + TRACE+17.

A constant predictor: the preregistered empty K0, its fixed system prompt and
sandboxed Python tool, executed by the unchanged Qwen3.5-4B behind an
OpenAI-compatible endpoint. One evaluation covers every TRACE+17 test item and
the 1,000-item G-test; every item is sealed as it completes, so an interrupted
run resumes where it stopped under an identical contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import uuid

from protocol import BASE_MODEL_ID, canonical_score, load_manifest, read_jsonl, task_specs


GENERAL_CATEGORIES = ("general-knowledge", "mathematical-reasoning", "commonsense-reasoning",
                      "instruction-following", "multilingual-understanding")
MAX_TOOL_TURNS = 32


# Exact frozen K0 value from the preregistered static control. It is embedded as
# data; this package does not import the harness implementation.
STATIC_K0 = {"schemaVersion": "hero-harness-spec-v1", "skills": []}

SYSTEM_PROMPT = "\n".join([
    "You are executing the committed HERO harness specification for one benchmark input.",
    "Use the registered benchmark tools when the specification requires them.",
    "Do not inspect hidden-scorer data or credentials. Finish with the requested answer.",
    "",
    "",
])

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Run Python 3 to calculate or check an answer. Standard library and files inside "
            "the current sample workspace only; no network or external files. Print the result. "
            "Maximum execution time: 10 seconds."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "code": {"type": "string", "description": "Python source code to execute."},
            },
            "required": ["code"],
        },
    },
}


def _sandbox_profile(directory: Path) -> str:
    escaped = json.dumps(str(directory.resolve()))
    return "\n".join([
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
        '(allow file-read* (literal "/"))',
        '(allow file-read* (subpath "/System") (subpath "/usr") (subpath "/bin") (subpath "/sbin")',
        '  (subpath "/Library/Developer") (subpath "/Applications/Xcode.app")',
        '  (subpath "/private/var/db/dyld") (literal "/dev/null") (literal "/dev/urandom") (subpath "/dev/fd"))',
        f"(allow file-read* file-write* (subpath {escaped}))",
        '(allow file-write-data (literal "/dev/null") (subpath "/dev/fd"))',
    ])


def run_sandbox_python(workspace: Path, code: str) -> dict[str, object]:
    """Reproduce the static control's validated macOS Python tool locally."""
    if sys.platform != "darwin":
        raise RuntimeError("Static harness Python-tool execution requires the validated macOS sandbox.")
    workspace.mkdir(parents=True, exist_ok=True)
    directory = workspace.resolve(strict=True)
    python = subprocess.check_output(
        ["/usr/bin/xcrun", "--find", "python3"], text=True
    ).strip()
    call_id = str(uuid.uuid4())
    profile_path = directory / f"sandbox-{call_id}.sb"
    profile_path.write_text(_sandbox_profile(directory), encoding="utf-8")
    process = subprocess.Popen(
        ["/usr/bin/sandbox-exec", "-f", str(profile_path), python, "-I", "-B", "-c", code],
        cwd=directory,
        env={"PATH": "/usr/bin:/bin", "LANG": "en_US.UTF-8", "TMPDIR": str(directory)},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    result = capture_process(process)
    (directory / f"python-{call_id}.json").write_text(
        json.dumps({"code": code, "result": result}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def capture_process(process, *, timeout: float = 10, output_limit: int = 65536) -> dict[str, object]:
    """Bound output while reading, and reap the sample's whole process group."""
    streams = selectors.DefaultSelector()
    for pipe, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
        streams.register(pipe, selectors.EVENT_READ, name)
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    remaining = output_limit
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        while streams.get_map() and remaining > 0:
            wait = deadline - time.monotonic()
            if wait <= 0:
                timed_out = True
                break
            for key, _ in streams.select(wait):
                chunk = os.read(key.fileobj.fileno(), min(8192, remaining))
                if not chunk:
                    streams.unregister(key.fileobj)
                    continue
                captured[key.data].extend(chunk)
                remaining -= len(chunk)
                if remaining == 0:
                    break
        if remaining > 0 and not timed_out:
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        streams.close()
        process.stdout.close()
        process.stderr.close()
    return {
        "stdout": captured["stdout"].decode("utf-8", errors="replace"),
        "stderr": captured["stderr"].decode("utf-8", errors="replace"),
        "exitCode": process.returncode if process.returncode is not None else -1,
        "timedOut": timed_out,
    }


class StaticPredictor:
    def __init__(self, endpoint: str, *, max_new_tokens: int, request_timeout: float = 7200) -> None:
        from openai import OpenAI

        self.max_new_tokens = max_new_tokens
        # Execution only: late turns of a long tool loop can exceed the SDK's 600 s default on one device.
        self.client = OpenAI(
            base_url=endpoint,
            api_key=os.getenv("STATIC_HARNESS_API_KEY", "local"),
            max_retries=0,
            timeout=request_timeout,
        )

    def generate(self, prompt: str, *, workspace: Path) -> str:
        """Run the frozen empty harness, including its fixed Python tool loop."""

        if STATIC_K0["skills"]:
            raise ValueError("The frozen static control requires an empty K0 skills list.")
        messages: list[dict[str, object]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        self.last_trace = {"turns": 0, "toolCalls": 0, "promptTokens": 0, "completionTokens": 0,
                           "finishReason": None}
        for _ in range(MAX_TOOL_TURNS):
            response = self.client.chat.completions.create(
                model=BASE_MODEL_ID,
                messages=messages,
                tools=[PYTHON_TOOL],
                tool_choice="auto",
                temperature=0,
                max_tokens=self.max_new_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            choice = response.choices[0]
            usage = getattr(response, "usage", None)
            self.last_trace["turns"] += 1
            self.last_trace["finishReason"] = choice.finish_reason
            self.last_trace["promptTokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.last_trace["completionTokens"] += getattr(usage, "completion_tokens", 0) or 0
            if choice.finish_reason not in {"stop", "tool_calls"}:
                raise RuntimeError(f"Incomplete static-harness response: {choice.finish_reason}")
            message = choice.message
            tool_calls = message.tool_calls or []
            assistant: dict[str, object] = {
                "role": "assistant",
                "content": message.content or "",
            }
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in tool_calls
                ]
            messages.append(assistant)
            if not tool_calls:
                if not (message.content or "").strip():
                    raise ValueError("Static harness produced no terminal answer.")
                return message.content or ""
            self.last_trace["toolCalls"] += len(tool_calls)
            for call in tool_calls:
                if call.function.name != "python":
                    raise ValueError(f"Unexpected static-harness tool: {call.function.name}")
                arguments = json.loads(call.function.arguments)
                if not isinstance(arguments, dict) or set(arguments) != {"code"} or not isinstance(arguments["code"], str):
                    raise ValueError("Python tool call must contain exactly one string 'code' argument.")
                result = run_sandbox_python(workspace, arguments["code"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        raise RuntimeError(f"Static harness exceeded {MAX_TOOL_TURNS} model/tool turns for one sample.")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluation_panel(dataset: Path, general: Path, *, max_stages: int | None, max_eval_samples: int | None,
                     general_limit: int | None) -> tuple[list[dict], dict]:
    """Every test item of the selected tasks, then the G-test by category; all files hash-verified."""
    manifest = load_manifest(dataset)
    tasks = task_specs(dataset)[:max_stages]
    panel, gold = [], {}
    for task, entry in zip(tasks, manifest["tasks"]):
        spec = entry["files"]["testGold"]
        if file_hash(dataset / spec["path"]) != spec["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {spec['path']}")
        rows = read_jsonl(dataset / spec["path"])[:max_eval_samples]
        gold[task.id] = {"metric": task.metric, "rows": rows}
        panel += [{"group": task.id, "id": row["id"], "prompt": row["prompt"]} for row in rows]
    general_manifest = json.loads((general / "manifest.json").read_text())
    if general_manifest["version"] != "hero-general-test-v1" or general_manifest["seed"] != 42:
        raise ValueError("Expected the pinned seed-42 general-test/v1 panel")
    specs = {entry["path"]: entry for entry in general_manifest["files"]}
    for name in ("hidden-scorer/general-test.jsonl", "hidden-scorer/probe-ids.json"):
        if file_hash(general / name) != specs[name]["sha256"]:
            raise ValueError(f"General panel hash mismatch: {name}")
    general_rows = read_jsonl(general / "hidden-scorer/general-test.jsonl")
    probe_ids = json.loads((general / "hidden-scorer/probe-ids.json").read_text())
    for category in GENERAL_CATEGORIES:
        rows = [row for row in general_rows if row["category"] == category][:general_limit]
        gold["general:" + category] = {"metric": rows[0]["metric"], "rows": rows}
        panel += [{"group": "general:" + category, "id": row["id"], "prompt": row["prompt"]} for row in rows]
    provenance = {"datasetManifestSha256": file_hash(dataset / "manifest.json"),
                  "generalManifestSha256": file_hash(general / "manifest.json"),
                  "tasks": [task.id for task in tasks], "maxEvalSamples": max_eval_samples,
                  "generalLimit": general_limit, "items": len(panel)}
    return panel, {"gold": gold, "probeIds": probe_ids, "provenance": provenance}


def endpoint_health(endpoint: str) -> dict | None:
    """The serving bridge's identity, when it exposes one; its generation ceiling must cover the run's."""
    import urllib.request

    root = endpoint.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    try:
        with urllib.request.urlopen(root + "/health", timeout=10) as response:
            health = json.loads(response.read())
    except (OSError, ValueError):
        return None
    return {key: health.get(key) for key in ("model", "versionId", "bridgeSha256", "maxNewTokens", "maxInputTokens",
                                             "inferenceProtocol")}


def item_path(output: Path, group: str, index: int) -> Path:
    return output / "items" / group.replace(":", "__") / f"{index:04d}.json"


def predict_item(predictor: StaticPredictor, item: dict, workspace: Path) -> dict:
    """One sealed item. Deterministic per-item failures score zero; transport or server faults abort the run."""
    import openai

    started = time.perf_counter()
    predictor.last_trace = {}
    try:
        prediction, status, error = predictor.generate(item["prompt"], workspace=workspace), "complete", None
    except openai.APIStatusError as exc:
        if exc.status_code >= 500:
            raise
        prediction, status, error = "", "failed", f"HTTP {exc.status_code}: {exc}"
    except (RuntimeError, ValueError) as exc:
        prediction, status, error = "", "failed", str(exc)
    record = {"id": item["id"], "group": item["group"], "prediction": prediction, "itemStatus": status,
              "seconds": time.perf_counter() - started, **predictor.last_trace}
    if error is not None:
        record["error"] = error
    return record


def score_run(output: Path, panel: list[dict], evaluation: dict) -> dict:
    by_group: dict[str, list[dict]] = {}
    for index, item in enumerate(panel):
        record = json.loads(item_path(output, item["group"], index).read_text())
        if record["id"] != item["id"]:
            raise ValueError(f"Sealed item does not match the panel: {item['id']}")
        by_group.setdefault(item["group"], []).append(record)
    scores, records = {}, []
    for group, rows in by_group.items():
        gold = evaluation["gold"][group]
        result = canonical_score(gold["metric"], [{"id": r["id"], "prediction": r["prediction"]} for r in rows],
                                 gold["rows"])
        scores[group] = result
        records += rows
        (output / "predictions").mkdir(exist_ok=True)
        (output / "predictions" / (group.replace(":", "__") + ".jsonl")).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tasks = [group for group in by_group if not group.startswith("general:")]
    probes = set(evaluation["probeIds"])
    general = {}
    for category in GENERAL_CATEGORIES:
        result = scores["general:" + category]
        probe_items = [row for row in result["itemScores"] if row["id"] in probes]
        general[category] = {"score": result["score"],
                             "probeScore": (sum(row["score"] for row in probe_items) / len(probe_items)
                                            if probe_items else None)}
    metrics = {
        "AA": sum(scores[t]["score"] for t in tasks) / len(tasks),
        "taskScores": {t: scores[t]["score"] for t in tasks},
        "forgetting": 0.0, "signedBWT": 0.0, "note": "constant predictor: no parameter or harness update",
        "generalMacro": sum(v["score"] for v in general.values()) / len(general),
        "generalCategoryScores": general,
        "itemFailures": sum(r["itemStatus"] != "complete" for r in records),
        "items": len(records), "modelTurns": sum(r.get("turns", 0) for r in records),
        "toolCalls": sum(r.get("toolCalls", 0) for r in records),
        "promptTokens": sum(r.get("promptTokens", 0) for r in records),
        "completionTokens": sum(r.get("completionTokens", 0) for r in records),
        "seconds": sum(r["seconds"] for r in records),
    }
    if all(v["probeScore"] is not None for v in general.values()):
        metrics["generalProbeMacro"] = sum(v["probeScore"] for v in general.values()) / len(general)
    return {"scores": scores, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--general-dataset", type=Path, required=True, help="experiments/data/general-test/v1")
    parser.add_argument("--output", type=Path, required=True, help="New directory, or one to resume")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--max-stages", type=int, help="Smoke only: first N tasks")
    parser.add_argument("--max-eval-samples", type=int, help="Smoke only: first N test items per task")
    parser.add_argument("--general-limit", type=int, help="Smoke only: first N G items per category")
    parser.add_argument("--max-new-tokens", type=int, default=5120)
    parser.add_argument("--max-input-tokens", type=int, default=16384,
                        help="Protocol input ceiling per model call; the endpoint must enforce exactly this value")
    parser.add_argument("--request-timeout", type=float, default=7200,
                        help="Seconds per model request (execution only; not part of the contract)")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    for name in ("max_stages", "max_eval_samples", "general_limit"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if sys.platform != "darwin":
        parser.error("Static harness requires macOS for its sandboxed Python tool")

    panel, evaluation = evaluation_panel(args.dataset, args.general_dataset, max_stages=args.max_stages,
                                         max_eval_samples=args.max_eval_samples, general_limit=args.general_limit)
    health = endpoint_health(args.endpoint)
    if health and health.get("maxNewTokens") is not None and health["maxNewTokens"] < args.max_new_tokens:
        parser.error(f"The endpoint caps generation at {health['maxNewTokens']} tokens, below --max-new-tokens")
    if health and health.get("maxInputTokens") is not None and health["maxInputTokens"] != args.max_input_tokens:
        parser.error(f"The endpoint enforces {health['maxInputTokens']} prompt tokens per call, not --max-input-tokens")
    contract = {
        "method": "static-harness", "model": BASE_MODEL_ID, "staticHarness": STATIC_K0,
        "systemPrompt": SYSTEM_PROMPT, "tools": [PYTHON_TOOL], "maxToolTurns": MAX_TOOL_TURNS,
        "decoding": {"temperature": 0, "thinking": False, "maxNewTokensPerTurn": args.max_new_tokens,
                     "maxInputTokensPerCall": args.max_input_tokens},
        "training": False, "endpoint": health, "runnerSha256": file_hash(Path(__file__)),
        **evaluation["provenance"],
    }
    contract_path = args.output / "contract.json"
    if contract_path.is_file():
        stored = json.loads(contract_path.read_text())
        science = lambda value: {key: item for key, item in value.items() if key != "runnerSha256"}
        if science(stored) != science(contract):
            parser.error("The output was produced under a different contract; use a new directory")
        if stored["runnerSha256"] != contract["runnerSha256"]:
            # Same panel, prompt, tool, decoding and endpoint; only execution code changed. Audit every such resume.
            with (args.output / "resume-log.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"runnerSha256": contract["runnerSha256"],
                                      "originalRunnerSha256": stored["runnerSha256"],
                                      "resumedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic_json(contract_path, contract)

    predictor = StaticPredictor(args.endpoint, max_new_tokens=args.max_new_tokens, request_timeout=args.request_timeout)
    for index, item in enumerate(panel):
        path = item_path(args.output, item["group"], index)
        if path.is_file():
            if json.loads(path.read_text())["id"] != item["id"]:
                raise ValueError(f"Sealed item does not match the panel: {path}")
            continue
        workspace = args.output / "workspaces" / item["group"].replace(":", "__") / f"{index:04d}"
        if workspace.exists():
            # An interrupted attempt left files here; the rerun starts from an empty workspace.
            aborted = args.output / "workspaces-aborted" / workspace.parent.name / f"{index:04d}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
            aborted.parent.mkdir(parents=True, exist_ok=True)
            os.replace(workspace, aborted)
        record = predict_item(predictor, item, workspace)
        atomic_json(path, record)
        print(json.dumps({"item": index + 1, "of": len(panel), "id": item["id"], "status": record["itemStatus"],
                          "turns": record.get("turns"), "seconds": round(record["seconds"], 2)}), flush=True)

    result = score_run(args.output, panel, evaluation)
    atomic_json(args.output / "scores.json", result["scores"])
    atomic_json(args.output / "metrics.json", result["metrics"])
    atomic_json(args.output / "status.json", {
        "status": "complete_with_item_failures" if result["metrics"]["itemFailures"] else "complete",
        "items": len(panel), "itemFailures": result["metrics"]["itemFailures"],
        "contractSha256": file_hash(contract_path),
        "outputHashes": {name: file_hash(args.output / name) for name in ("scores.json", "metrics.json")}})
    print(json.dumps({key: result["metrics"][key] for key in ("AA", "generalMacro", "itemFailures", "items")}))


if __name__ == "__main__":
    main()
