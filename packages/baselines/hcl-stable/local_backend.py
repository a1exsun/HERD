"""In-process Qwen3.5-4B under MLX behind an OpenAI-shaped chat client.

The method code talks to `client.chat.completions.create(...)`. This module
serves those calls from the pinned checkpoint on this Mac's GPU with mlx-lm's
continuous batching: concurrent calls from the method's worker threads join
the running decode batch as slots free up, so parallel generator, reflector
and evaluation calls share the device.

Calls in this study share long prefixes (a playbook or a harness followed by a
task input). The engine snapshots the model cache at a shared token prefix and
starts later calls from a copy, so each shared prefix is computed once. A
snapshot boundary comes from the longest common token prefix with recent
prompts, rounded down to a fixed granularity; the hybrid attention cache of
Qwen3.5 cannot be trimmed, so a snapshot is reused only when it is an exact
prefix of the new prompt. Reuse changes where the computation happens, not
what is computed.

Memory: sequences in a decode batch share one left-padded cache length, so the
worst case grows with batch size times the longest sequence (16,384 input plus
5,120 output tokens); the defaults keep two engines in parallel processes within
the GPU working set.

Every call is held to the protocol ceilings: at most `max_input_tokens`
prompt tokens (a longer prompt raises `InputTooLong`) and at most
`max_new_tokens` generated tokens. Decoding is greedy at temperature 0 and
sampled otherwise; thinking is disabled by the chat template.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import Future
import copy
from dataclasses import dataclass, field
import json
from pathlib import Path
import queue
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np


class InputTooLong(ValueError):
    """The chat prompt exceeds the per-call input ceiling (a deterministic per-call failure)."""


@dataclass
class Request:
    ids: list[int]
    temperature: float
    max_new_tokens: int
    future: Future = field(default_factory=Future)


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    reused_prompt_tokens: int = 0


def common_prefix_length(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    mismatch = np.flatnonzero(a[:n] != b[:n])
    return int(mismatch[0]) if len(mismatch) else n


def snapshot_boundary(shared: int, prompt_length: int, *, granularity: int, margin: int) -> int:
    """A snapshot position inside a shared prefix, stable across prompts that diverge slightly later.

    At least two prompt tokens stay after the boundary (the generator needs a non-empty remainder
    besides the final token).
    """
    boundary = max(0, shared - margin) // granularity * granularity
    return min(boundary, max(0, (prompt_length - 2) // granularity * granularity))


class PrefixStore:
    """Cache snapshots keyed by their exact token prefix, least recently used first out."""

    def __init__(self, *, max_entries: int, max_bytes: int):
        self.entries: OrderedDict[bytes, tuple[np.ndarray, Any, int]] = OrderedDict()
        self.max_entries, self.max_bytes = max_entries, max_bytes

    def longest(self, ids: np.ndarray) -> tuple[int, Any]:
        best_key, best_length = None, 0
        for key, (tokens, _, _) in self.entries.items():
            length = len(tokens)
            if best_length < length < len(ids) and np.array_equal(ids[:length], tokens):
                best_key, best_length = key, length
        if best_key is None:
            return 0, None
        self.entries.move_to_end(best_key)
        return best_length, self.entries[best_key][1]

    def insert(self, tokens: np.ndarray, snapshot: Any, nbytes: int) -> None:
        self.entries[tokens.tobytes()] = (tokens, snapshot, nbytes)
        self.entries.move_to_end(tokens.tobytes())
        while len(self.entries) > self.max_entries or sum(e[2] for e in self.entries.values()) > self.max_bytes:
            self.entries.popitem(last=False)


class MlxEngine:
    """The pinned checkpoint under MLX; `submit` returns a future resolved by the serving thread."""

    def __init__(self, checkpoint, *, max_new_tokens: int = 5120, max_input_tokens: int = 16384,
                 batch_size: int = 16, prefill_batch_size: int = 4, prefill_step_size: int = 1024,
                 prefix_min_tokens: int = 1024, prefix_granularity: int = 256, prefix_margin: int = 64,
                 prefix_entries: int = 32, prefix_bytes: int = 4 << 30, recent_prompts: int = 64,
                 cache_limit_bytes: int = 512 << 20, clear_every: int = 256, seed: int = 42):
        from transformers import AutoTokenizer

        self.checkpoint = Path(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        config = json.loads((self.checkpoint / "config.json").read_text())
        configured = config.get("text_config", config).get("eos_token_id")  # as the HF generation config derives it
        self.eos = sorted(set(([configured] if isinstance(configured, int) else list(configured or []))
                              + [self.tokenizer.eos_token_id]))
        self.max_new_tokens, self.max_input_tokens, self.batch_size = max_new_tokens, max_input_tokens, batch_size
        self.prefill_batch_size, self.prefill_step_size, self.seed = prefill_batch_size, prefill_step_size, seed
        self.cache_limit_bytes, self.clear_every = cache_limit_bytes, clear_every
        self.prefix_min_tokens, self.prefix_granularity, self.prefix_margin = prefix_min_tokens, prefix_granularity, prefix_margin
        self.store = PrefixStore(max_entries=prefix_entries, max_bytes=prefix_bytes)
        self.recent: deque[np.ndarray] = deque(maxlen=recent_prompts)
        self.calls = self.prompt_tokens = self.completion_tokens = self.reused_prompt_tokens = 0
        self._queue: queue.Queue[Request] = queue.Queue()
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._worker = threading.Thread(target=self._serve, name="mlx-engine", daemon=True)
        self._worker.start()
        self._ready.wait()
        if self._failure is not None:
            raise self._failure

    # -- caller side -----------------------------------------------------

    def encode(self, messages: list[dict[str, Any]]) -> list[int]:
        return list(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       enable_thinking=False, return_dict=True)["input_ids"])

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def submit(self, request: Request) -> Future:
        if self._failure is not None:
            raise RuntimeError("The MLX engine stopped") from self._failure
        self._queue.put(request)
        return request.future

    # -- serving thread --------------------------------------------------

    def _serve(self) -> None:
        try:
            import mlx.core as mx
            from mlx_lm.generate import BatchGenerator
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.utils import load_model

            mx.random.seed(self.seed)
            # Freed buffers return to the system beyond this size and the cache is emptied periodically, so
            # small per-step buffers never pile up toward Metal's per-process buffer-count limit and engines in
            # parallel processes do not hoard GPU memory.
            mx.set_cache_limit(self.cache_limit_bytes)
            self._clear_cache = mx.clear_cache
            self._mx = mx
            self.model, _ = load_model(self.checkpoint)
            self._sampler = make_sampler
            self._generator = BatchGenerator(
                self.model, max_tokens=self.max_new_tokens, stop_tokens=[[token] for token in self.eos],
                completion_batch_size=self.batch_size, prefill_batch_size=self.prefill_batch_size,
                prefill_step_size=self.prefill_step_size)
        except BaseException as error:  # reported to the constructor
            self._failure = error
            self._ready.set()
            return
        self._ready.set()
        self._active: dict[int, tuple[Request, list[int], int]] = {}
        self._snapshots: dict[int, np.ndarray] = {}      # uid -> prefix being computed for the store
        self._waiting: list[tuple[np.ndarray, Request, np.ndarray]] = []  # (prefix, request, ids)
        iterations = 0
        try:
            while True:
                iterations += 1
                if iterations % self.clear_every == 0:
                    self._clear_cache()
                block = not self._active
                while True:
                    try:
                        request = self._queue.get(block=block)
                    except queue.Empty:
                        break
                    self._admit(request)
                    block = False
                prompt_responses, generation_responses = self._generator.next()
                for response in prompt_responses:
                    if response.end_of_segment and not response.end_of_prompt and response.uid in self._snapshots:
                        self._store_snapshot(response.uid)
                for response in generation_responses:
                    request, tokens, reused = self._active[response.uid]
                    tokens.append(response.token)
                    if response.finish_reason is not None:
                        del self._active[response.uid]
                        self._finish(request, tokens, response.finish_reason, reused)
                        if response.uid in self._snapshots:  # finished before its prefix was stored
                            self._release(self._snapshots.pop(response.uid))
        except BaseException as error:  # every caller sees the device failure
            self._failure = error
            for request, _, _ in self._active.values():
                request.future.set_exception(error)
            for _, request, _ in self._waiting:
                request.future.set_exception(error)
            while not self._queue.empty():
                self._queue.get().future.set_exception(error)

    def _admit(self, request: Request) -> None:
        ids = np.asarray(request.ids, dtype=np.int64)
        stored, snapshot = self.store.longest(ids)
        pending = max((prefix for prefix in self._snapshots.values()
                       if len(prefix) > stored and len(prefix) < len(ids) and np.array_equal(ids[:len(prefix)], prefix)),
                      key=len, default=None)
        if pending is not None and len(pending) - stored >= self.prefix_min_tokens:
            self._waiting.append((pending, request, ids))  # admitted when that snapshot is stored
            return
        shared = max((common_prefix_length(ids, other) for other in self.recent), default=0)
        self.recent.append(ids)
        boundary = snapshot_boundary(shared, len(ids), granularity=self.prefix_granularity, margin=self.prefix_margin)
        segments = [request.ids[stored:]]
        if boundary - stored >= self.prefix_min_tokens:
            segments = [request.ids[stored:boundary], request.ids[boundary:]]
        cache = copy.deepcopy(snapshot) if snapshot is not None else None
        (uid,) = self._generator.insert_segments(
            [segments], max_tokens=[request.max_new_tokens], caches=[cache], all_tokens=[request.ids[:stored]],
            samplers=[self._sampler(request.temperature) if request.temperature > 0 else None])
        if len(segments) == 2:
            self._snapshots[uid] = ids[:boundary]
        self._active[uid] = (request, [], stored)
        self.calls += 1
        self.prompt_tokens += len(ids)
        self.reused_prompt_tokens += stored

    def _store_snapshot(self, uid: int) -> None:
        prefix = self._snapshots.pop(uid)
        (snapshot, _), = self._generator.extract_cache([uid]).values()
        # The extracted cache slices the running batch's buffers; a contiguous copy lets the snapshot hold only
        # its own prefix instead of pinning every batch buffer it was cut from.
        for layer in snapshot:
            layer.state = [None if array is None else self._mx.contiguous(array) for array in layer.state]
        self._mx.eval([array for layer in snapshot for array in layer.state if array is not None])
        self.store.insert(prefix, snapshot, sum(layer.nbytes for layer in snapshot))
        self._release(prefix)

    def _release(self, prefix: np.ndarray) -> None:
        waiting, self._waiting = self._waiting, []
        for pending, request, ids in waiting:
            if np.array_equal(pending, prefix):
                self._admit(request)
            else:
                self._waiting.append((pending, request, ids))

    def _finish(self, request: Request, tokens: list[int], reason: str, reused: int) -> None:
        stop = next((position for position, token in enumerate(tokens) if token in self.eos), None)
        kept = tokens[:stop] if stop is not None else tokens
        text = self.tokenizer.decode(kept, skip_special_tokens=True)
        self.completion_tokens += len(tokens)
        request.future.set_result(Completion(text, len(request.ids), len(tokens),
                                             "stop" if reason == "stop" else "length", reused))


def endpoint_version(endpoint: str) -> str:
    """The serving library version reported by an OpenAI-compatible vLLM endpoint."""
    import urllib.request

    root = endpoint.rstrip("/").removesuffix("/v1")
    with urllib.request.urlopen(root + "/version", timeout=30) as response:
        return "vllm " + json.loads(response.read())["version"]


class HttpEngine:
    """The pinned checkpoint behind an OpenAI-compatible completions endpoint (vLLM), with the same interface
    and the same per-call contract as `MlxEngine`: the chat-templated prompt token ids, thinking disabled,
    greedy decoding at temperature 0, the EOS union and the input and output ceilings. The server batches
    concurrent calls and caches shared prefixes itself."""

    def __init__(self, checkpoint, *, endpoint: str, model: str = "Qwen/Qwen3.5-4B", max_new_tokens: int = 5120,
                 max_input_tokens: int = 16384, workers: int = 192, seed: int = 42):
        from concurrent.futures import ThreadPoolExecutor
        from urllib.parse import urlparse

        from transformers import AutoTokenizer

        self.checkpoint = Path(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        config = json.loads((self.checkpoint / "config.json").read_text())
        configured = config.get("text_config", config).get("eos_token_id")
        self.eos = sorted(set(([configured] if isinstance(configured, int) else list(configured or []))
                              + [self.tokenizer.eos_token_id]))
        self.max_new_tokens, self.max_input_tokens, self.seed, self.model = max_new_tokens, max_input_tokens, seed, model
        parsed = urlparse(endpoint)
        self.host, self.port, self.path = parsed.hostname, parsed.port or 80, parsed.path.rstrip("/") + "/completions"
        self.calls = self.prompt_tokens = self.completion_tokens = self.reused_prompt_tokens = 0
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="http-engine")

    def encode(self, messages: list[dict[str, Any]]) -> list[int]:
        return list(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       enable_thinking=False, return_dict=True)["input_ids"])

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def submit(self, request: Request) -> Future:
        self._pool.submit(self._resolve, request)
        return request.future

    def _resolve(self, request: Request) -> None:
        try:
            request.future.set_result(self._complete(request))
        except BaseException as error:  # delivered to the waiting caller
            request.future.set_exception(error)

    def _complete(self, request: Request) -> Completion:
        import http.client
        import time

        body = {"model": self.model, "prompt": request.ids, "max_tokens": request.max_new_tokens,
                "temperature": request.temperature, "top_p": 1.0, "stop_token_ids": self.eos,
                "skip_special_tokens": True, "return_token_ids": True}
        if request.temperature > 0:
            body["seed"] = self.seed
        error: BaseException | None = None
        for attempt in range(8):
            try:
                connection = http.client.HTTPConnection(self.host, self.port, timeout=7200)
                connection.request("POST", self.path, json.dumps(body), {"Content-Type": "application/json"})
                response = connection.getresponse()
                payload = response.read().decode()
                connection.close()
                if response.status == 200:
                    break
                error = RuntimeError(f"HTTP {response.status}: {payload[:300]}")
            except OSError as failure:
                error = failure
            time.sleep(min(60, 5 * (attempt + 1)))
        else:
            raise RuntimeError("The model endpoint failed") from error
        data = json.loads(payload)
        choice = data["choices"][0]
        tokens = list(choice.get("token_ids") or [])
        stop = next((position for position, token in enumerate(tokens) if token in self.eos), None)
        text = self.tokenizer.decode(tokens[:stop] if stop is not None else tokens, skip_special_tokens=True)
        cached = int(((data.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        with self._lock:
            self.calls += 1
            self.prompt_tokens += len(request.ids)
            self.completion_tokens += len(tokens)
            self.reused_prompt_tokens += cached
        return Completion(text, len(request.ids), len(tokens),
                          "stop" if choice.get("finish_reason") == "stop" else "length", cached)


class ChatClient:
    """`client.chat.completions.create(...)` served by an engine; concurrent calls share its batch.

    Each client counts its own calls, so units sharing one engine keep separate usage records.
    """

    def __init__(self, engine):
        self.engine = engine
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.calls = self.prompt_tokens = self.completion_tokens = self.reused_prompt_tokens = 0
        self._lock = threading.Lock()

    def usage(self) -> dict[str, int]:
        return {"modelCalls": self.calls, "promptTokens": self.prompt_tokens,
                "reusedPromptTokens": self.reused_prompt_tokens, "completionTokens": self.completion_tokens}

    def create(self, *, model: str = "", messages: list[dict[str, Any]], temperature: float = 0.0,
               max_tokens: int | None = None, max_completion_tokens: int | None = None, **ignored: Any):
        ids = self.engine.encode([{"role": m["role"], "content": m.get("content") or ""} for m in messages])
        if len(ids) > self.engine.max_input_tokens:
            # No digits in the message: a status-like number could be misread as a retryable rate limit.
            raise InputTooLong("The prompt exceeds the per-call input ceiling.")
        requested = max_tokens or max_completion_tokens or self.engine.max_new_tokens
        request = Request(ids, float(temperature or 0.0), min(int(requested), self.engine.max_new_tokens))
        completion: Completion = self.engine.submit(request).result()
        with self._lock:
            self.calls += 1
            self.prompt_tokens += completion.prompt_tokens
            self.completion_tokens += completion.completion_tokens
            self.reused_prompt_tokens += completion.reused_prompt_tokens
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=completion.text, tool_calls=None),
                                     finish_reason=completion.finish_reason)],
            usage=SimpleNamespace(prompt_tokens=completion.prompt_tokens,
                                  completion_tokens=completion.completion_tokens))
