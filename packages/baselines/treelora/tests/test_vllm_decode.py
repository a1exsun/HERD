"""vllm_decode against a stand-in `vllm` module: inputs, stops, failures, cost shares and the export lifecycle."""
import json
import sys
import types
from pathlib import Path

import pytest

import vllm_decode


class Completion:
    def __init__(self, token_ids, finish_reason):
        self.token_ids, self.finish_reason = token_ids, finish_reason


class FakeLLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs, self.calls, self.closed = kwargs, [], False
        self.llm_engine = types.SimpleNamespace(engine_core=types.SimpleNamespace(shutdown=self.shutdown))
        FakeLLM.instances.append(self)

    def shutdown(self):
        self.closed = True

    def generate(self, prompts, params, use_tqdm):
        self.calls.append((prompts, params))
        # Echo: one token per prompt token, last token the first stop id when the prompt is even-length.
        outputs = []
        for prompt in prompts:
            ids = prompt["prompt_token_ids"]
            stop = len(ids) % 2 == 0
            tokens = [7] * len(ids) + ([params.stop_token_ids[0]] if stop else [])
            outputs.append(types.SimpleNamespace(outputs=[Completion(tokens, "stop" if stop else "length")]))
        return outputs


@pytest.fixture
def fake_vllm(monkeypatch):
    module = types.ModuleType("vllm")
    module.__version__ = vllm_decode.VLLM_VERSION
    module.LLM = FakeLLM
    module.SamplingParams = lambda **kwargs: types.SimpleNamespace(**kwargs)
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)
    monkeypatch.setattr(vllm_decode, "release_cuda", lambda: None)
    FakeLLM.instances.clear()
    return module


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False, "return_dict": True}
        return {"input_ids": [1] * len(messages[-1]["content"])}

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return "".join("x" for token in tokens if token != 2)


def test_engine_is_greedy_bf16_and_maps_stops_failures_and_costs(fake_vllm, tmp_path):
    engine = vllm_decode.GreedyEngine(tmp_path, max_input_tokens=4, max_new_tokens=9)
    (llm,) = FakeLLM.instances
    assert llm.kwargs["dtype"] == "bfloat16" and llm.kwargs["max_model_len"] == 13
    assert llm.kwargs["generation_config"] == "vllm" and llm.kwargs["seed"] == 42
    results = engine.decode(Tokenizer(), [[1, 1], [1, 1, 1], [1] * 5], eos=[2, 3])
    (prompts, params), = llm.calls
    assert params.temperature == 0.0 and params.max_tokens == 9 and params.stop_token_ids == [2, 3]
    assert [p["prompt_token_ids"] for p in prompts] == [[1, 1], [1, 1, 1]]  # the over-limit input is never sent
    first, second, failed = results
    assert first["outputTokenIds"] == [7, 7, 2] and first["stopReason"] == "eos" and first["prediction"] == "xx"
    assert second["stopReason"] == "length" and second["outputTokens"] == 3
    assert first["seconds"] == second["seconds"] and first["batchSize"] == 2 and first["engine"] == vllm_decode.ENGINE
    assert failed == {"prediction": "", "outputTokenIds": [], "promptTokens": 5, "outputTokens": 0, "seconds": 0.0,
                      "stopReason": "input_limit", "itemStatus": "failed", "engine": vllm_decode.ENGINE}
    engine.close()
    assert llm.closed


def test_engine_refuses_another_vllm_version(fake_vllm, tmp_path):
    fake_vllm.__version__ = "0.0.1"
    with pytest.raises(RuntimeError, match="requires vllm=="):
        vllm_decode.GreedyEngine(tmp_path, max_input_tokens=4, max_new_tokens=9)


class Model:
    def save_pretrained(self, target, safe_serialization):
        assert safe_serialization
        Path(target).mkdir(parents=True)
        (Path(target) / "config.json").write_text("{}")


def test_evaluate_items_exports_seals_each_item_and_removes_the_export(fake_vllm, tmp_path, monkeypatch):
    checkpoint = tmp_path / "qwen"
    checkpoint.mkdir()
    for name in vllm_decode.SIDE_FILES:
        (checkpoint / name).write_text(name)
    monkeypatch.setattr(vllm_decode, "CALL_ITEMS", 2)
    pending = [(tmp_path / f"{i}.json", {"id": str(i)}, "p" * (i + 1)) for i in range(3)]
    written = {}
    vllm_decode.evaluate_items(Model(), checkpoint, Tokenizer(), [2], pending, lambda path, row: written.update({path: row}),
                               max_input_tokens=8, max_new_tokens=9)
    (llm,) = FakeLLM.instances
    assert len(llm.calls) == 2 and llm.closed  # two calls of at most CALL_ITEMS, then shutdown
    assert llm.kwargs["model"] == str(checkpoint.parent / "qwen-evaluated")
    assert not (checkpoint.parent / "qwen-evaluated").exists()
    assert [written[path]["id"] for path, _, _ in pending] == ["0", "1", "2"]
    assert written[pending[1][0]]["stopReason"] == "eos" and written[pending[0][0]]["stopReason"] == "length"


def test_evaluate_unit_decodes_only_missing_items_and_seals_the_stage(tmp_path, monkeypatch):
    import run

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
            built.append(adapter)
            return types.SimpleNamespace(eval=lambda: "merged")

        def sealed_adapter(self, position):
            return f"adapter-{position}"

        def merged_model(self, through):  # SDFT rebuilds the merged chain instead
            built.append(through)
            return types.SimpleNamespace(eval=lambda: "merged")

    decoded = []

    def evaluate_items(model, checkpoint, tokenizer, eos, pending, write, *, max_input_tokens, max_new_tokens):
        assert model == "merged" and max_input_tokens == 16384 and max_new_tokens == 5120
        for path, row, prompt in pending:
            decoded.append(prompt)
            write(path, {**row, "prediction": prompt.upper(), "seconds": 0.5})

    monkeypatch.setattr(run, "Unit", Unit)
    monkeypatch.setattr(vllm_decode, "evaluate_items", evaluate_items)
    # A cached item from an interrupted job is kept, not decoded again.
    (predictions / "items" / "a").mkdir(parents=True)
    run.atomic_json(predictions / "items" / "a" / "0000.json", {"id": "a0", "group": "a", "stage": 1, "prediction": "cached", "seconds": 1.0})
    run.evaluate_unit(types.SimpleNamespace(groups="a"))
    assert decoded == ["p1"] and built in (["adapter-1"], [1])  # the model is built once, for stage 1
    assert [r["prediction"] for r in run.read_jsonl(predictions / "predictions" / "a.jsonl")] == ["cached", "P1"]
    assert not (predictions / "evaluation.json").exists()  # group b is still missing
    run.evaluate_unit(types.SimpleNamespace(groups=None))
    assert decoded == ["p1", "q0"]
    sealed = json.loads((predictions / "evaluation.json").read_text())
    assert sealed["engine"] == vllm_decode.ENGINE and sealed["trainingSha256"] == run.file_hash(trained / "training.json")
    assert set(sealed["predictions"]) == {"predictions/a.jsonl", "predictions/b.jsonl"}
