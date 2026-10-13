"""Verify the sealed static-harness run independently of the runner's completion flag.

The audit rebuilds the pinned panel from hash-verified data, checks the contract's
science fields and serving endpoint, every sealed item and its sandbox workspace,
the prediction files, a full score recomputation, the aggregate metrics and the
recorded output hashes. With --output the verified result becomes the sealed report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from protocol import DEFAULT_DATASET_PATH, EVAL_MAX_INPUT_TOKENS, EVAL_MAX_NEW_TOKENS, STAGE_COUNT, canonical_score
import run as static


DEFAULT_GENERAL_PATH = Path("experiments/data/general-test/v1")
COUNTS = ("turns", "toolCalls", "promptTokens", "completionTokens")
# Error markers of the deterministic per-item failures the runner seals; each scores the empty prediction.
FAILURE_CLASSES = (
    ("malformed tool call", "malformed tool-call markup"),
    ("input ceiling", "Prompt contains"),
    ("output ceiling", "Incomplete static-harness response: length"),
    ("tool-turn limit", "model/tool turns for one sample"),
    ("unregistered tool", "Unexpected static-harness tool"),
)


def failure_class(error: str) -> str:
    return next((name for name, marker in FAILURE_CLASSES if marker in error), "other")


def expected_contract(provenance: dict) -> dict:
    """Science fields of the formal control; the endpoint identity and runner hash are checked separately."""
    return {"method": "static-harness", "model": static.BASE_MODEL_ID, "staticHarness": static.STATIC_K0,
            "systemPrompt": static.SYSTEM_PROMPT, "tools": [static.PYTHON_TOOL], "maxToolTurns": static.MAX_TOOL_TURNS,
            "decoding": {"temperature": 0, "thinking": False, "maxNewTokensPerTurn": EVAL_MAX_NEW_TOKENS,
                         "maxInputTokensPerCall": EVAL_MAX_INPUT_TOKENS},
            "training": False, **provenance}


def verify_contract(contract: dict, provenance: dict) -> None:
    expected = expected_contract(provenance)
    if set(contract) != set(expected) | {"endpoint", "runnerSha256"}:
        raise ValueError("Contract fields differ from the formal static control")
    if any(contract[key] != value for key, value in expected.items()):
        raise ValueError("Contract science fields differ from the formal static control")
    if len(provenance["tasks"]) != STAGE_COUNT or provenance["maxEvalSamples"] or provenance["generalLimit"]:
        raise ValueError("Expected the complete 25-task panel and G-test")
    endpoint = contract["endpoint"] or {}
    if (endpoint.get("model") != static.BASE_MODEL_ID or endpoint.get("versionId") != "base"
            or not endpoint.get("bridgeSha256") or not endpoint.get("inferenceProtocol")
            or (endpoint.get("maxNewTokens") or 0) < EVAL_MAX_NEW_TOKENS
            or endpoint.get("maxInputTokens") != EVAL_MAX_INPUT_TOKENS):
        raise ValueError("Endpoint is not the unchanged base model under the protocol ceilings")


def check_outcome(row: dict, workspace: Path) -> None:
    """An outcome the runner can legitimately seal, consistent with the sample's sandbox tool log."""
    for key in COUNTS:
        if not isinstance(row.get(key), int) or row[key] < 0:
            raise ValueError(f"Invalid {key}: {row['id']}")
    if not isinstance(row.get("seconds"), (int, float)) or not math.isfinite(row["seconds"]) or row["seconds"] < 0:
        raise ValueError(f"Invalid operation cost: {row['id']}")
    if row["turns"] > static.MAX_TOOL_TURNS:
        raise ValueError(f"Tool-turn limit exceeded: {row['id']}")
    executed = len(list(workspace.glob("python-*.json")))
    if row["itemStatus"] == "complete":
        if ("error" in row or not isinstance(row["prediction"], str) or not row["prediction"].strip()
                or row["finishReason"] != "stop" or row["turns"] < 1):
            raise ValueError(f"Invalid successful outcome: {row['id']}")
        if executed != row["toolCalls"]:
            raise ValueError(f"Tool calls disagree with the sample workspace: {row['id']}")
    elif row["itemStatus"] == "failed":
        if row["prediction"] != "" or not row.get("error"):
            raise ValueError(f"Invalid failed outcome: {row['id']}")
        if executed > row["toolCalls"]:
            raise ValueError(f"Sample workspace holds unrecorded tool calls: {row['id']}")
    else:
        raise ValueError(f"Unknown item status: {row['id']}")


def verify_items(output: Path, panel: list[dict]) -> list[dict]:
    """Every panel item sealed exactly once, in panel order; no workspace outside the panel."""
    expected = {static.item_path(output, item["group"], index) for index, item in enumerate(panel)}
    actual = set((output / "items").glob("*/*.json"))
    if actual - expected:
        raise ValueError("Unexpected prediction files outside the pinned panel")
    if expected - actual:
        raise ValueError(f"Incomplete prediction coverage: {len(actual & expected)}/{len(panel)}")
    workspaces = {path.relative_to(output / "workspaces") for path in (output / "workspaces").glob("*/*")}
    if workspaces - {path.relative_to(output / "items").with_suffix("") for path in expected}:
        raise ValueError("Sandbox workspace outside the pinned panel")
    rows = []
    for index, item in enumerate(panel):
        path = static.item_path(output, item["group"], index)
        row = json.loads(path.read_text())
        if row["id"] != item["id"] or row["group"] != item["group"]:
            raise ValueError(f"Sealed item does not match the panel: {item['id']}")
        check_outcome(row, output / "workspaces" / path.parent.name / path.stem)
        rows.append(row)
    return rows


def constant_metrics(scores: dict, rows: list[dict], probe_ids: set[str]) -> dict:
    """The runner's aggregates, recomputed from recomputed scores and sealed items in panel order."""
    tasks = [group for group in scores if not group.startswith("general:")]
    general = {}
    for category in static.GENERAL_CATEGORIES:
        result = scores["general:" + category]
        probe = [row["score"] for row in result["itemScores"] if row["id"] in probe_ids]
        general[category] = {"score": result["score"], "probeScore": sum(probe) / len(probe) if probe else None}
    metrics = {
        "AA": sum(scores[t]["score"] for t in tasks) / len(tasks),
        "taskScores": {t: scores[t]["score"] for t in tasks},
        "forgetting": 0.0, "signedBWT": 0.0, "note": "constant predictor: no parameter or harness update",
        "generalMacro": sum(v["score"] for v in general.values()) / len(general),
        "generalCategoryScores": general,
        "itemFailures": sum(row["itemStatus"] != "complete" for row in rows), "items": len(rows),
        "modelTurns": sum(row["turns"] for row in rows), "toolCalls": sum(row["toolCalls"] for row in rows),
        "promptTokens": sum(row["promptTokens"] for row in rows),
        "completionTokens": sum(row["completionTokens"] for row in rows),
        "seconds": sum(row["seconds"] for row in rows),
    }
    if all(v["probeScore"] is not None for v in general.values()):
        metrics["generalProbeMacro"] = sum(v["probeScore"] for v in general.values()) / len(general)
    return metrics


def verify(output: Path, dataset: Path, general_dataset: Path) -> dict:
    panel, evaluation = static.evaluation_panel(dataset, general_dataset, max_stages=None, max_eval_samples=None,
                                                general_limit=None)
    contract = json.loads((output / "contract.json").read_text())
    verify_contract(contract, evaluation["provenance"])
    rows = verify_items(output, panel)
    groups = list(dict.fromkeys(item["group"] for item in panel))
    scores = json.loads((output / "scores.json").read_text())
    if list(scores) != groups:
        raise ValueError("Score coverage mismatch")
    for group in groups:
        members = [row for row in rows if row["group"] == group]
        prediction_file = output / "predictions" / (group.replace(":", "__") + ".jsonl")
        if [json.loads(line) for line in prediction_file.read_text().splitlines()] != members:
            raise ValueError(f"Prediction file disagrees with sealed items: {group}")
        gold = evaluation["gold"][group]
        recomputed = canonical_score(gold["metric"], [{"id": r["id"], "prediction": r["prediction"]} for r in members],
                                     gold["rows"])
        if recomputed != scores[group] or not math.isfinite(recomputed["score"]) or not 0 <= recomputed["score"] <= 1:
            raise ValueError(f"Score recomputation mismatch: {group}")
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics != constant_metrics(scores, rows, set(evaluation["probeIds"])):
        raise ValueError("Aggregate metrics mismatch")
    failures = [row for row in rows if row["itemStatus"] == "failed"]
    status = json.loads((output / "status.json").read_text())
    expected_status = {"status": "complete_with_item_failures" if failures else "complete", "items": len(panel),
                       "itemFailures": len(failures), "contractSha256": static.file_hash(output / "contract.json"),
                       "outputHashes": {name: static.file_hash(output / name) for name in ("scores.json", "metrics.json")}}
    if status != expected_status:
        raise ValueError("Completion status disagrees with sealed results")
    return {
        "status": "verified", "scope": "25-task TRACE+17 static-harness control plus five-category G-test",
        "run": str(output), "contractSha256": expected_status["contractSha256"],
        "harness": {"staticHarness": contract["staticHarness"], "tools": [t["function"]["name"] for t in contract["tools"]],
                    "maxToolTurns": contract["maxToolTurns"]},
        "decoding": {"model": contract["model"], **contract["decoding"]}, "endpoint": contract["endpoint"],
        "predictions": len(rows), "itemFailures": len(failures),
        "failures": {"scoring": "the scorer-defined outcome for an empty prediction",
                     "byClass": dict(Counter(failure_class(row["error"]) for row in failures).most_common()),
                     "byGroup": dict(Counter(row["group"] for row in failures).most_common())},
        "metrics": metrics,
        "taskScores": {g: scores[g] for g in groups if not g.startswith("general:")},
        "generalScores": {g: scores[g] for g in groups if g.startswith("general:")},
        "cost": {key: metrics[key] for key in ("modelTurns", "toolCalls", "promptTokens", "completionTokens")}
                | {"inferenceSeconds": metrics["seconds"]},
        "sourceStatusHash": static.file_hash(output / "status.json"),
        "predictionChecks": "source hashes, contract science fields, endpoint identity, coverage, IDs, groups, outcomes, "
                            "sandbox tool logs, prediction files, score recomputation, metrics, output hashes",
        "limitations": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--general-dataset", type=Path, default=DEFAULT_GENERAL_PATH)
    parser.add_argument("--output", type=Path, help="Write the verified result as the sealed report")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("Report exists; choose a new filename")
    report = verify(args.run, args.dataset, args.general_dataset)
    if args.output:
        static.atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "predictions", "itemFailures")}
                     | {"AA": report["metrics"]["AA"], "generalMacro": report["metrics"]["generalMacro"]}))


if __name__ == "__main__":
    main()
