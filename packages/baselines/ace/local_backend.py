"""Pinned BF16 Qwen on PyTorch MPS, exposing upstream ACE's client interface.

Completed greedy calls are sealed before returning, so recovery of an interrupted
sample reuses its completed calls instead of generating them again.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace as NS

from protocol import BASE_MODEL_ID, EVAL_MAX_INPUT_TOKENS

_client = None


def configure(checkpoint: Path, cache: Path):
    global _client
    if _client is None:
        _client = MPSClient(checkpoint, cache)
    elif _client.checkpoint != checkpoint.resolve() or _client.cache != cache.resolve():
        raise ValueError("MPS backend already configured for a different run")
    return _client


def get_client():
    if _client is None:
        raise RuntimeError("Configure the MPS backend through run.py first")
    return _client


class MPSClient:
    def __init__(self, checkpoint: Path, cache: Path):
        import torch
        import transformers
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no device fallback")
        if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Disable PYTORCH_ENABLE_MPS_FALLBACK for the MPS experiment")
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.checkpoint, self.cache = checkpoint.resolve(), cache.resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.torch, self.lock = torch, threading.Lock()
        torch.manual_seed(42)
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.checkpoint, dtype=torch.bfloat16, local_files_only=True).to("mps").eval()
        self.model.requires_grad_(False)
        eos = self.model.generation_config.eos_token_id
        self.eos = sorted(set(([eos] if isinstance(eos, int) else eos or []) + [self.tokenizer.eos_token_id]))
        self.chat = NS(completions=NS(create=self.create))
        self.contract = {"model": BASE_MODEL_ID, "root": str(self.checkpoint), "maxModelLen": 20480,
                         "server": "in-process", "device": "mps", "precision": "bfloat16",
                         "thinking": False, "torch": torch.__version__, "transformers": transformers.__version__}
        print(json.dumps({"MPS": self.contract}), flush=True)

    def create(self, *, model, messages, temperature, max_tokens):
        if model != BASE_MODEL_ID or temperature != 0 or not 1 <= max_tokens <= 4096:
            raise ValueError("MPS ACE requires the pinned model, greedy decoding and at most 4096 output tokens")
        # Serialise the shared model; training and evaluation use separate upstream threads.
        with self.lock:
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_dict=True, return_tensors="pt")
            count = inputs["input_ids"].shape[-1]
            if count > EVAL_MAX_INPUT_TOKENS:
                import httpx
                import openai

                request = httpx.Request("POST", "http://localhost/mps/chat/completions")
                raise openai.BadRequestError(
                    f"maximum context length: input {count} exceeds {EVAL_MAX_INPUT_TOKENS} tokens",
                    response=httpx.Response(400, request=request), body=None)
            key = hashlib.sha256(json.dumps({"contract": self.contract, "messages": messages,
                                             "maxTokens": max_tokens}, sort_keys=True).encode()).hexdigest()
            path = self.cache / key[:2] / (key + ".json")
            cached = path.is_file()
            if cached:
                record = json.loads(path.read_text())
            else:
                started = time.perf_counter()
                print(json.dumps({"phase": "mps-generate", "call": key, "promptTokens": count}), flush=True)
                with self.torch.inference_mode():
                    output = self.model.generate(
                        **inputs.to("mps"), max_new_tokens=max_tokens, do_sample=False,
                        eos_token_id=self.eos, pad_token_id=self.tokenizer.pad_token_id)
                tokens = output[0, count:].tolist()
                record = {"call": key, "content": self.tokenizer.decode(tokens, skip_special_tokens=True),
                          "promptTokens": count, "outputTokenIds": tokens,
                          "completionTokens": len(tokens), "seconds": time.perf_counter() - started}
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n")
                os.replace(temporary, path)
                print(json.dumps({"phase": "mps-complete", "call": key, "completionTokens": len(tokens),
                                  "seconds": record["seconds"]}), flush=True)
            return NS(choices=[NS(message=NS(content=record["content"]))], cache_hit=cached,
                      usage=NS(prompt_tokens=record["promptTokens"], completion_tokens=record["completionTokens"]))
