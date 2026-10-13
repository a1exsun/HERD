"""Greedy evaluation decoding with vLLM for this baseline's evaluated checkpoint.

The evaluated weights (the base checkpoint, or the base with a method's adapters merged in) are exported to a
node-local directory together with the base checkpoint's tokenizer and processor files, and decoded by an
in-process vLLM engine: bf16, greedy, the same chat-template token ids as the transformers path (thinking
disabled), the model/tokenizer EOS union as stop tokens, at most 5,120 new tokens. Prompts over the 16,384-token
input ceiling are recorded as `input_limit` failures and never sent. vLLM schedules every pending item of a call
together; each item's `seconds` is its share of that call's wall time.

Only the standard library is imported at module level, so training environments import this file safely.
"""
from __future__ import annotations

import gc
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

VLLM_VERSION = "0.31.0"
ENGINE = f"vllm=={VLLM_VERSION}"
# Base-checkpoint files the exported directory needs besides the weights and config.json.
SIDE_FILES = ("merges.txt", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
              "video_preprocessor_config.json", "vocab.json")
CALL_ITEMS = 512  # items per engine call; items are sealed after every call


def prompt_ids(tokenizer, prompt: str) -> list[int]:
    """The evaluation input: one user turn through the chat template, thinking disabled."""
    return list(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False,
                                              return_dict=True)["input_ids"])


def export_checkpoint(model, checkpoint: Path, target: Path) -> Path:
    """Write the (merged) transformers model as a checkpoint vLLM loads, with the base tokenizer files."""
    shutil.rmtree(target, ignore_errors=True)
    model.save_pretrained(target, safe_serialization=True)
    for name in SIDE_FILES:
        shutil.copy2(checkpoint / name, target / name)
    return target


def release_cuda() -> None:
    """Return the freed transformers model's GPU memory before the engine claims its share."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class GreedyEngine:
    def __init__(self, model_dir: Path, *, max_input_tokens: int, max_new_tokens: int, seed: int = 42,
                 gpu_memory_utilization: float = 0.85):
        import vllm
        from vllm import LLM

        if vllm.__version__ != VLLM_VERSION:
            raise RuntimeError(f"Evaluation requires {ENGINE}, found vllm=={vllm.__version__}")
        self.max_input_tokens, self.max_new_tokens = max_input_tokens, max_new_tokens
        # generation_config="vllm" ignores any sampling defaults saved with the checkpoint; the vision tower is
        # never used, so no image or video budget is reserved.
        self.llm = LLM(model=str(model_dir), tokenizer=str(model_dir), dtype="bfloat16", seed=seed,
                       max_model_len=max_input_tokens + max_new_tokens, gpu_memory_utilization=gpu_memory_utilization,
                       generation_config="vllm", limit_mm_per_prompt={"image": 0, "video": 0})

    def close(self) -> None:
        """Stop the engine process and free its GPU memory, so a later stage can start its own engine."""
        self.llm.llm_engine.engine_core.shutdown()
        del self.llm
        release_cuda()

    def __enter__(self) -> "GreedyEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def decode(self, tokenizer, encoded: Sequence[Sequence[int]], eos: Sequence[int]) -> list[dict[str, Any]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        results: list[dict[str, Any] | None] = [None] * len(encoded)
        for index, ids in enumerate(encoded):
            if len(ids) > self.max_input_tokens:
                results[index] = {"prediction": "", "outputTokenIds": [], "promptTokens": len(ids), "outputTokens": 0,
                                  "seconds": 0.0, "stopReason": "input_limit", "itemStatus": "failed", "engine": ENGINE}
        todo = [index for index in range(len(encoded)) if results[index] is None]
        if todo:
            params = SamplingParams(temperature=0.0, max_tokens=self.max_new_tokens, stop_token_ids=list(eos),
                                    detokenize=False)
            started = time.perf_counter()
            outputs = self.llm.generate([TokensPrompt(prompt_token_ids=list(encoded[index])) for index in todo],
                                        params, use_tqdm=False)
            share = (time.perf_counter() - started) / len(todo)
            for index, output in zip(todo, outputs):
                completion = output.outputs[0]
                tokens = list(completion.token_ids)
                results[index] = {"prediction": tokenizer.decode(tokens, skip_special_tokens=True),
                                  "outputTokenIds": tokens, "promptTokens": len(encoded[index]),
                                  "outputTokens": len(tokens), "seconds": share,
                                  "stopReason": "eos" if completion.finish_reason == "stop" else "length",
                                  "itemStatus": "complete", "engine": ENGINE, "batchSize": len(todo)}
        return [result for result in results if result is not None]


def evaluate_items(model, checkpoint: Path, tokenizer, eos: Sequence[int],
                   pending: Sequence[tuple[Path, dict[str, Any], str]], write, *, max_input_tokens: int,
                   max_new_tokens: int) -> None:
    """Decode `(path, row, prompt)` items with the weights of the transformers `model`.

    The model is exported next to `checkpoint` (node-local scratch in a job), released, and served by one engine;
    items are decoded in calls of CALL_ITEMS and `write(path, row | result)` seals each one. The export is removed
    afterwards."""
    export = checkpoint.parent / f"{checkpoint.name}-evaluated"
    export_checkpoint(model, checkpoint, export)
    del model
    release_cuda()
    try:
        with GreedyEngine(export, max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens) as engine:
            for start in range(0, len(pending), CALL_ITEMS):
                part = pending[start:start + CALL_ITEMS]
                results = engine.decode(tokenizer, [prompt_ids(tokenizer, prompt) for _, _, prompt in part], eos)
                for (path, row, _), result in zip(part, results):
                    write(path, {**row, **result})
    finally:
        shutil.rmtree(export, ignore_errors=True)
