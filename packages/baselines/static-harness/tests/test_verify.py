import json

import pytest
import verify


PANEL = [{"group": "trace:T", "id": "t0", "prompt": "p"}, {"group": "general:g", "id": "g0", "prompt": "q"}]


def complete(item, **change):
    row = {"id": item["id"], "group": item["group"], "prediction": "answer", "itemStatus": "complete", "seconds": 1.5,
           "turns": 1, "toolCalls": 0, "promptTokens": 10, "completionTokens": 2, "finishReason": "stop"}
    return row | change


def seal(output, index, row):
    path = verify.static.item_path(output, PANEL[index]["group"], index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    return path


def test_verifier_requires_exact_panel_coverage(tmp_path):
    seal(tmp_path, 0, complete(PANEL[0]))
    with pytest.raises(ValueError, match="Incomplete"):
        verify.verify_items(tmp_path, PANEL)
    seal(tmp_path, 1, complete(PANEL[1]))
    assert [row["id"] for row in verify.verify_items(tmp_path, PANEL)] == ["t0", "g0"]
    (tmp_path / "items/trace__T/0009.json").write_text("{}")
    with pytest.raises(ValueError, match="Unexpected"):
        verify.verify_items(tmp_path, PANEL)


@pytest.mark.parametrize("change", [
    {"id": "other"}, {"group": "trace:U"}, {"prediction": " "}, {"finishReason": "length"}, {"turns": 0},
    {"turns": 33}, {"seconds": float("nan")}, {"promptTokens": -1}, {"error": "late"}, {"itemStatus": "partial"},
    {"itemStatus": "failed", "error": "HTTP 422"}, {"itemStatus": "failed", "prediction": ""},
])
def test_verifier_rejects_outcomes_the_runner_cannot_seal(tmp_path, change):
    seal(tmp_path, 0, complete(PANEL[0], **change))
    seal(tmp_path, 1, complete(PANEL[1]))
    with pytest.raises(ValueError):
        verify.verify_items(tmp_path, PANEL)


def test_tool_calls_must_match_the_sample_workspace(tmp_path):
    item = seal(tmp_path, 0, complete(PANEL[0], toolCalls=1, turns=2))
    seal(tmp_path, 1, complete(PANEL[1]))
    with pytest.raises(ValueError, match="workspace"):
        verify.verify_items(tmp_path, PANEL)
    workspace = tmp_path / "workspaces" / item.parent.name / item.stem
    workspace.mkdir(parents=True)
    (workspace / "python-1.json").write_text("{}")
    assert len(verify.verify_items(tmp_path, PANEL)) == 2
    (tmp_path / "workspaces/trace__T/0009").mkdir()
    with pytest.raises(ValueError, match="outside the pinned panel"):
        verify.verify_items(tmp_path, PANEL)


def test_failed_items_keep_an_empty_prediction_and_their_error(tmp_path):
    failed = complete(PANEL[0], prediction="", itemStatus="failed", finishReason=None, turns=0,
                      error="HTTP 422: Error code: 422 - {'detail': 'Qwen emitted malformed tool-call markup.'}")
    seal(tmp_path, 0, failed)
    seal(tmp_path, 1, complete(PANEL[1]))
    assert verify.verify_items(tmp_path, PANEL)[0]["itemStatus"] == "failed"
    assert verify.failure_class(failed["error"]) == "malformed tool call"
    assert verify.failure_class("Static harness exceeded 32 model/tool turns for one sample.") == "tool-turn limit"
    assert verify.failure_class("something new") == "other"


def test_contract_must_be_the_formal_control_on_the_base_endpoint():
    provenance = {"datasetManifestSha256": "d", "generalManifestSha256": "g", "tasks": [f"t{i}" for i in range(25)],
                  "maxEvalSamples": None, "generalLimit": None, "items": 3500}
    endpoint = {"model": verify.static.BASE_MODEL_ID, "versionId": "base", "bridgeSha256": "b",
                "maxNewTokens": 5120, "maxInputTokens": 16384, "inferenceProtocol": "hero-qwen-chat-eos-v1"}
    contract = verify.expected_contract(provenance) | {"endpoint": endpoint, "runnerSha256": "r"}
    verify.verify_contract(contract, provenance)
    for change in ({"maxToolTurns": 8}, {"staticHarness": {"schemaVersion": "hero-harness-spec-v1", "skills": [1]}},
                   {"endpoint": endpoint | {"versionId": "stage-3"}}, {"endpoint": endpoint | {"maxInputTokens": None}},
                   {"endpoint": None}):
        with pytest.raises(ValueError):
            verify.verify_contract(contract | change, provenance)
    with pytest.raises(ValueError, match="complete 25-task"):
        smoke = provenance | {"maxEvalSamples": 5}
        verify.verify_contract(verify.expected_contract(smoke) | {"endpoint": endpoint, "runnerSha256": "r"}, smoke)
