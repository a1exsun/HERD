"""OPHSD's evaluation: resumable vLLM decoding of a stage's panels, then the stage seal."""
import json
import types

import run
import vllm_decode


def test_evaluate_groups_decodes_missing_items_and_finalize_seals_the_stage(tmp_path, monkeypatch):
    bundle, stage_dir = tmp_path / "bundle", tmp_path / "output" / "stage-01"
    (bundle / "inputs/eval").mkdir(parents=True)
    stage_dir.mkdir(parents=True)
    (stage_dir / "training.json").write_text("{}")
    rows = {"a": [{"id": "a0", "prompt": "p0"}, {"id": "a1", "prompt": "p1"}], "b": [{"id": "b0", "prompt": "q0"}]}
    for group, records in rows.items():
        (bundle / "inputs/eval" / run.group_file(group)).write_text("".join(json.dumps(r) + "\n" for r in records))
    stage = {"stage": 1, "evaluations": [{"group": "a", "ids": ["a0", "a1"]}, {"group": "b", "ids": ["b0"]}]}
    decoded = []

    def evaluate_items(model, checkpoint, tokenizer, eos, pending, write, *, max_input_tokens, max_new_tokens):
        assert model == "merged" and checkpoint == tmp_path / "qwen" and eos == [2]
        for path, row, prompt in pending:
            decoded.append(prompt)
            write(path, {**row, "prediction": prompt.upper(), "seconds": 0.5})

    monkeypatch.setattr(vllm_decode, "evaluate_items", evaluate_items)
    model = types.SimpleNamespace(eval=lambda: "merged")
    run.atomic_json(stage_dir / "items" / "a" / "0000.json", {"id": "a0", "group": "a", "stage": 1, "prediction": "cached"})
    run.evaluate_groups(model, tmp_path / "qwen", object(), bundle, stage_dir, stage, stage["evaluations"][:1], [2],
                        "shard-0-of-2")
    assert decoded == ["p1"] and not run.finalize_stage(stage_dir, stage)  # panel b is still missing
    run.evaluate_groups(model, tmp_path / "qwen", object(), bundle, stage_dir, stage, stage["evaluations"][1:], [2],
                        "shard-1-of-2")
    assert decoded == ["p1", "q0"] and run.finalize_stage(stage_dir, stage)
    complete = json.loads((stage_dir / "complete.json").read_text())
    assert complete["engine"] == vllm_decode.ENGINE and set(complete["predictions"]) == {"predictions/a.jsonl", "predictions/b.jsonl"}
    assert complete["evaluationSeconds"] >= 0
