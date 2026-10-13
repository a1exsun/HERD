from __future__ import annotations

import json

import pytest
from hero_qwen.bridge import _as_sse, normalize_messages, parse_qwen_tool_calls, chat_eos_token_ids


def test_chat_eos_includes_im_end_without_losing_model_endoftext():
    assert chat_eos_token_ids(248044, 248046) == [248044, 248046]
    assert chat_eos_token_ids([248046, 248044], 248046) == [248044, 248046]
    with pytest.raises(ValueError):
        chat_eos_token_ids(248044, None)


def test_chat_turn_stops_at_tokenizer_eos_and_preserves_raw_tokens():
    import threading
    from types import SimpleNamespace
    import torch
    from transformers import BatchEncoding
    from hero_qwen.bridge import QwenBridge, INFERENCE_PROTOCOL

    class Tokenizer:
        pad_token_id = 248044
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)})
        def decode(self, ids, **kwargs):
            assert ids.tolist() == [34, 248046]
            return "C"

    def generate(**kwargs):
        assert kwargs["eos_token_id"] == [248044, 248046]
        assert kwargs["pad_token_id"] == 248044
        return torch.tensor([[1, 2, 34, 248046]])

    bridge = object.__new__(QwenBridge)
    bridge.torch, bridge.device, bridge.version_id = torch, "cpu", "test"
    bridge.tokenizer = Tokenizer()
    bridge.model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=248044))
    bridge.lock, bridge.eos_token_ids, bridge.inference_protocol = threading.Lock(), [248044, 248046], INFERENCE_PROTOCOL
    result = bridge.complete({"messages": [{"role": "user", "content": "question"}]})
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 2
    assert result["hero"]["completionTokenIds"] == [34, 248046]


def test_qwen_tool_call_parser_preserves_typed_arguments() -> None:
    content, calls = parse_qwen_tool_calls(
        "checking\n<tool_call>\n<function=read_file>\n"
        '<parameter=path>\n"notes.txt"\n</parameter>\n'
        "<parameter=limit>\n4\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert content == "checking"
    assert calls[0]["function"]["name"] == "read_file"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "path": "notes.txt",
        "limit": 4,
    }


def test_malformed_tool_call_fails_the_session() -> None:
    with pytest.raises(ValueError, match="malformed"):
        parse_qwen_tool_calls("<tool_call>broken</tool_call>")


def test_openai_text_parts_are_normalized_without_images() -> None:
    assert normalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": " two"},
                ],
            }
        ]
    ) == [{"role": "user", "content": "one two"}]


def test_historical_openai_tool_calls_become_qwen_template_arguments() -> None:
    messages = normalize_messages(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": '{"path":"notes.txt"}',
                        },
                    }
                ],
            }
        ]
    )
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {
        "path": "notes.txt"
    }


def test_streamed_tool_calls_carry_openai_indexes() -> None:
    response = {
        "id": "chatcmpl-test",
        "created": 0,
        "model": "Qwen/Qwen3.5-4B",
        "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-test",
                            "type": "function",
                            "function": {"name": "read", "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    first = next(_as_sse(response)).removeprefix("data: ").strip()
    assert json.loads(first)["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    chunks = list(_as_sse(response))
    assert json.loads(chunks[1].removeprefix("data: ").strip())["usage"] == response["usage"]


def test_target_nll_uses_completion_relative_causal_positions():
    import threading
    from types import SimpleNamespace
    import torch
    from transformers import BatchEncoding
    from hero_qwen.bridge import QwenBridge

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["return_dict"] and not kwargs["enable_thinking"]
            return BatchEncoding({"input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)})

        def encode(self, *args, **kwargs):
            return [2, 1]

    logits = torch.tensor([[[0., 0., 2.], [0., 3., 0.], [10., 0., 0.]]])

    def model(**kwargs):
        assert kwargs["input_ids"].tolist() == [[0, 1, 2, 1]]
        assert kwargs["logits_to_keep"] == 3 and kwargs["use_cache"] is False
        return SimpleNamespace(logits=logits)

    bridge = object.__new__(QwenBridge)
    bridge.torch, bridge.device, bridge.version_id = torch, "cpu", "test"
    bridge.tokenizer, bridge.model, bridge.lock = Tokenizer(), model, threading.Lock()
    result = bridge.target_nll({"messages": [{"role": "user", "content": "question"}], "target": "answer"})
    expected = -(logits[0, :2].log_softmax(-1)[torch.arange(2), torch.tensor([2, 1])]).mean()
    assert result["nll"] == pytest.approx(expected.item())
    assert result["promptTokens"] == result["targetTokens"] == 2


def test_target_nll_clips_the_target_and_appends_the_end_of_turn():
    import threading
    from types import SimpleNamespace
    import torch
    from transformers import BatchEncoding
    from hero_qwen.bridge import QwenBridge

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[0]]), "attention_mask": torch.ones(1, 1, dtype=torch.long)})

        def encode(self, *args, **kwargs):
            return [1, 2, 1]

        def convert_tokens_to_ids(self, token):
            assert token == "<|im_end|>"
            return 2

    seen = []

    def model(**kwargs):
        seen.append(kwargs["input_ids"].tolist())
        return SimpleNamespace(logits=torch.zeros(1, kwargs["logits_to_keep"], 3))

    bridge = object.__new__(QwenBridge)
    bridge.torch, bridge.device, bridge.version_id = torch, "cpu", "test"
    bridge.tokenizer, bridge.model, bridge.lock = Tokenizer(), model, threading.Lock()
    result = bridge.target_nll({"messages": [{"role": "user", "content": "q"}], "target": "a",
                                "maxTargetTokens": 2, "endOfTurn": True})
    assert seen == [[[0, 1, 2, 2]]] and result["targetTokens"] == 3


def test_clip_marks_cut_text_like_the_stream_runner():
    from hero_qwen.bridge import QwenBridge

    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(len(text.split())))

        def decode(self, ids, **kwargs):
            return " ".join("w" for _ in ids)

    bridge = object.__new__(QwenBridge)
    bridge.tokenizer = Tokenizer()
    assert bridge.clip("a b c", 2) == "w w ..."
    assert bridge.clip("a b", 2) == "a b"
    assert bridge.clip("a b c", 2, ellipsis=False) == "w w"
    assert bridge.clip("a b c d e", 4, tail=1) == "w w w ... w"


def test_generation_ceiling_defaults_to_1024_and_is_configurable():
    import threading
    from types import SimpleNamespace
    import torch
    from transformers import BatchEncoding
    from hero_qwen.bridge import QwenBridge, INFERENCE_PROTOCOL

    requested = []

    class Tokenizer:
        pad_token_id = 248044
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncoding({"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)})
        def decode(self, ids, **kwargs):
            return "C"

    def generate(**kwargs):
        requested.append(kwargs["max_new_tokens"])
        return torch.tensor([[1, 2, 34, 248046]])

    def bridge_with(ceiling=None):
        bridge = object.__new__(QwenBridge)
        bridge.torch, bridge.device, bridge.version_id = torch, "cpu", "test"
        bridge.tokenizer = Tokenizer()
        bridge.model = SimpleNamespace(generate=generate, generation_config=SimpleNamespace(eos_token_id=248044))
        bridge.lock, bridge.eos_token_ids, bridge.inference_protocol = threading.Lock(), [248044, 248046], INFERENCE_PROTOCOL
        if ceiling is not None:
            bridge.max_new_tokens = ceiling
        return bridge

    request = {"messages": [{"role": "user", "content": "q"}], "max_tokens": 5120}
    bridge_with().complete(request)
    bridge_with(5120).complete(request)
    assert requested == [1024, 5120]


def test_prompt_ceiling_is_configurable_and_rejects_longer_prompts():
    import threading
    from types import SimpleNamespace
    import pytest
    import torch
    from transformers import BatchEncoding
    from hero_qwen.bridge import QwenBridge, INFERENCE_PROTOCOL

    class Tokenizer:
        pad_token_id = 248044
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncoding({"input_ids": torch.ones(1, 20, dtype=torch.long),
                                  "attention_mask": torch.ones(1, 20, dtype=torch.long)})
        def decode(self, ids, **kwargs):
            return "C"

    bridge = object.__new__(QwenBridge)
    bridge.torch, bridge.device, bridge.version_id = torch, "cpu", "test"
    bridge.tokenizer = Tokenizer()
    bridge.model = SimpleNamespace(generate=lambda **kwargs: torch.ones(1, 22, dtype=torch.long),
                                   generation_config=SimpleNamespace(eos_token_id=248044))
    bridge.lock, bridge.eos_token_ids, bridge.inference_protocol = threading.Lock(), [248044, 248046], INFERENCE_PROTOCOL
    assert QwenBridge.max_input_tokens == 131072
    bridge.max_input_tokens = 16
    with pytest.raises(ValueError, match="permits at most 16"):
        bridge.complete({"messages": [{"role": "user", "content": "q"}]})
