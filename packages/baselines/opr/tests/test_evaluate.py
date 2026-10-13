"""OPR's evaluate unit: resumable vLLM decoding of the stage's merged checkpoint, sealed per stage."""
import json
import types

import run
import vllm_decode


def test_evaluate_unit_decodes_only_missing_items_and_seals_the_stage(tmp_path, monkeypatch):
    bundle, predictions, trained = tmp_path / "bundle", tmp_path / "eval" / "stage-01", tmp_path / "run" / "stage-01"
    (bundle / "inputs/eval").mkdir(parents=True)
    trained.mkdir(parents=True)
    (trained / "training.json").write_text("{}")
    rows = {"a": [{"id": "a0", "prompt": "p0"}, {"id": "a1", "prompt": "p1"}], "b": [{"id": "b0", "prompt": "q0"}]}
    for group, records in rows.items():
        (bundle / "inputs/eval" / run.group_file(group)).write_text("".join(json.dumps(r) + "\n" for r in records))
    stage = {"stage": 1, "evaluations": [{"group": "a", "ids": ["a0", "a1"]}, {"group": "b", "ids": ["b0"]}]}
    built = []

    class Unit:
        def __init__(self, args):
            self.predictions, self.stage, self.bundle, self.dir = predictions, stage, bundle, trained
            self.checkpoint, self.tokenizer, self.eos = tmp_path / "qwen", object(), [2]

        def model(self, adapter, trainable):
            assert not trainable
            built.append(adapter)
            return types.SimpleNamespace(eval=lambda: "merged")

        def sealed_adapter(self, position):
            return f"adapter-{position}"

    decoded = []

    def evaluate_items(model, checkpoint, tokenizer, eos, pending, write, *, max_input_tokens, max_new_tokens):
        assert model == "merged" and max_input_tokens == 16384 and max_new_tokens == 5120
        for path, row, prompt in pending:
            decoded.append(prompt)
            write(path, {**row, "prediction": prompt.upper(), "seconds": 0.5})

    monkeypatch.setattr(run, "Unit", Unit)
    monkeypatch.setattr(vllm_decode, "evaluate_items", evaluate_items)
    run.atomic_json(predictions / "items" / "a" / "0000.json",
                    {"id": "a0", "group": "a", "stage": 1, "prediction": "cached", "seconds": 1.0})
    run.evaluate_unit(types.SimpleNamespace(groups="a"))
    assert decoded == ["p1"] and built == ["adapter-1"]
    assert [r["prediction"] for r in run.read_jsonl(predictions / "predictions" / "a.jsonl")] == ["cached", "P1"]
    assert not (predictions / "evaluation.json").exists()
    run.evaluate_unit(types.SimpleNamespace(groups=None))
    assert decoded == ["p1", "q0"]
    sealed = json.loads((predictions / "evaluation.json").read_text())
    assert sealed["engine"] == vllm_decode.ENGINE and sealed["trainingSha256"] == run.file_hash(trained / "training.json")
    assert set(sealed["predictions"]) == {"predictions/a.jsonl", "predictions/b.jsonl"}
