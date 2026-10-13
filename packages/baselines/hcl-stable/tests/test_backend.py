from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import pytest

from local_backend import ChatClient, Completion, InputTooLong, PrefixStore, common_prefix_length, snapshot_boundary


class FakeEngine:
    """Resolves each submitted request at once; one token id per character."""

    max_input_tokens = 12
    max_new_tokens = 5

    def __init__(self):
        self.submitted = []

    def encode(self, messages):
        return [ord(c) for m in messages for c in m["content"]]

    def submit(self, request):
        self.submitted.append(request)
        future = Future()
        future.set_result(Completion("x" * request.max_new_tokens, len(request.ids), request.max_new_tokens, "stop",
                                     reused_prompt_tokens=len(request.ids) // 2))
        return future


@pytest.mark.parametrize("prompt_length", [13, 429])
def test_calls_are_capped_counted_and_long_prompts_fail_per_call(prompt_length):
    engine = FakeEngine()
    client = ChatClient(engine)
    with ThreadPoolExecutor(max_workers=4) as pool:
        replies = list(pool.map(lambda p: client.chat.completions.create(
            model="Qwen/Qwen3.5-4B", messages=[{"role": "user", "content": p}], temperature=0, max_tokens=4096),
            ["ab", "abcd", "abcdef"]))
    assert [r.usage.prompt_tokens for r in replies] == [2, 4, 6]
    assert all(r.choices[0].message.content == "xxxxx" and r.choices[0].finish_reason == "stop" for r in replies)
    assert client.usage() == {"modelCalls": 3, "promptTokens": 12, "reusedPromptTokens": 6, "completionTokens": 15}
    with pytest.raises(InputTooLong, match="per-call input ceiling") as failure:
        client.chat.completions.create(messages=[{"role": "user", "content": "x" * prompt_length}])
    assert str(prompt_length) not in str(failure.value)  # keep the local backend's numeric-free error contract
    assert len(engine.submitted) == 3  # the over-long call never reached the engine


def test_snapshot_boundaries_sit_inside_the_shared_prefix_on_a_stable_grid():
    a, b = np.arange(5000), np.concatenate([np.arange(3000), np.arange(9000, 11000)])
    assert common_prefix_length(a, b) == 3000
    assert snapshot_boundary(3000, 5000, granularity=256, margin=64) == 2816
    assert snapshot_boundary(3040, 5000, granularity=256, margin=64) == 2816  # small shifts keep the boundary
    assert snapshot_boundary(3000, 2817, granularity=256, margin=64) == 2560  # two prompt tokens stay after it


def test_prefix_store_returns_the_longest_exact_prefix_and_evicts_the_oldest():
    store = PrefixStore(max_entries=2, max_bytes=10**9)
    store.insert(np.arange(256), "short", 1)
    store.insert(np.arange(512), "long", 1)
    assert store.longest(np.arange(1000)) == (512, "long")
    assert store.longest(np.concatenate([np.arange(300), [-1] * 400])) == (256, "short")
    assert store.longest(np.arange(512)) == (256, "short")  # a snapshot must leave prompt tokens to process
    store.insert(np.arange(768), "longer", 1)  # evicts the least recently used entry ("long")
    assert store.longest(np.arange(1000)) == (768, "longer")
    assert len(store.entries) == 2


def test_stream_metrics_follow_the_paper_definitions():
    from run import stream_metrics

    cells = [{"stage": 1, "task": "a", "score": 0.8}, {"stage": 2, "task": "a", "score": 0.5},
             {"stage": 2, "task": "b", "score": 0.6}]
    metrics = stream_metrics(cells, ["a", "b"])
    assert metrics["AA"] == pytest.approx(0.55) and metrics["forgetting"] == pytest.approx(0.3)
