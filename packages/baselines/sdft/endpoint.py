"""On-policy rollouts from an OpenAI-compatible vLLM server.

The student of stage N is the base model with the sealed adapters of stages 1..N-1 merged in, plus the live
adapter of stage N. All adapters share one rank and alpha, so their sum is a single LoRA whose factors are
the stacked factors: rank r x N and alpha alpha x N keep every adapter's scaling alpha / r. Before each update
the live factors are stacked with the sealed ones and served as that LoRA. Completions are sampled one per
prompt with the method's sampling parameters (temperature, top-p 1, top-k -1, min-p 0, repetition penalty 1,
`logprobs=0`, prompt truncation at the configured length), stopping at the model's EOS ids, and return the
server's log-probability of every sampled token for the trainer's importance-sampling correction.
Requests carry no per-request seed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
import json
from pathlib import Path
import shutil
import time
from typing import Any, Sequence
from urllib.parse import urlparse


def _post(host: str, port: int, path: str, body: dict[str, Any]) -> dict[str, Any] | str:
    error: BaseException | None = None
    for attempt in range(8):
        try:
            connection = http.client.HTTPConnection(host, port, timeout=3600)
            connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            payload = response.read().decode()
            connection.close()
            if response.status == 200:
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    return payload
            error = RuntimeError(f"HTTP {response.status}: {payload[:300]}")
        except OSError as failure:
            error = failure
        time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError(f"The rollout endpoint failed on {path}") from error


def _get(host: str, port: int, path: str) -> dict[str, Any] | str:
    connection = http.client.HTTPConnection(host, port, timeout=60)
    connection.request("GET", path)
    response = connection.getresponse()
    payload = response.read().decode()
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status} on {path}: {payload[:300]}")
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload


def write_stacked_adapter(parts: Sequence[dict[str, Any]], config: dict[str, Any], directory: Path) -> None:
    """One PEFT LoRA directory whose factors stack `parts` (saved-format state dicts of equal rank)."""
    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key in parts[-1]:
        factors = [part[key].to(torch.float32) for part in parts]
        tensors[key] = torch.cat(factors, dim=0 if ".lora_A." in key else 1).to(torch.bfloat16).contiguous()
    save_file(tensors, str(directory / "adapter_model.safetensors"))
    stacked = dict(config)
    stacked["r"] = int(config["r"]) * len(parts)
    stacked["lora_alpha"] = int(config["lora_alpha"]) * len(parts)
    (directory / "adapter_config.json").write_text(json.dumps(stacked, indent=2) + "\n")


def sealed_parts(adapters: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    from safetensors.torch import load_file

    parts = [load_file(str(path / "adapter_model.safetensors")) for path in adapters]
    config = json.loads((adapters[0] / "adapter_config.json").read_text()) if adapters else None
    return parts, config


class EndpointSampler:
    """`sample_rollouts` served by the endpoint for a student with sealed adapters `sealed` and a live adapter."""

    def __init__(self, endpoint: str, sealed: Sequence[Path], scratch: Path, tag: str, workers: int = 64):
        parsed = urlparse(endpoint)
        self.host, self.port = parsed.hostname, parsed.port or 80
        self.parts, self.config = sealed_parts(sealed)
        self.scratch, self.tag, self.count = scratch, tag, 0
        self.pool = ThreadPoolExecutor(workers)
        # The serving stack, recorded with the stage: vLLM version and the served base model.
        self.server = {"version": _get(self.host, self.port, "/version"),
                       "models": [model.get("id") for model in _get(self.host, self.port, "/v1/models")["data"]]}

    def _live(self, model, directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch
        from peft.utils import get_peft_model_state_dict

        state = {key: value.detach().to("cpu", torch.float32) for key, value in get_peft_model_state_dict(model).items()}
        if self.config is None:
            model.peft_config["default"].save_pretrained(str(directory / "live-config"))
            self.config = json.loads((directory / "live-config" / "adapter_config.json").read_text())
        return state, self.config

    def __call__(self, model, tokenizer, encoded: Sequence[Sequence[int]], *, config, eos: Sequence[int]):
        from sdft import Rollout

        self.count += 1
        name = f"{self.tag}-{self.count}"
        directory = self.scratch / name
        live, adapter_config = self._live(model, self.scratch / f"{self.tag}-config")
        write_stacked_adapter(self.parts + [live], adapter_config, directory)
        _post(self.host, self.port, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(directory)})
        stops = set(int(token) for token in eos)

        def sample(index: int):
            body = {"model": name, "prompt": list(encoded[index]), "n": 1, "max_tokens": config.rollout_max_new_tokens,
                    "temperature": config.rollout_temperature, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
                    "repetition_penalty": 1.0, "truncate_prompt_tokens": config.max_prompt_length, "logprobs": 0,
                    "stop_token_ids": sorted(stops), "skip_special_tokens": True, "return_token_ids": True}
            choice = _post(self.host, self.port, "/v1/completions", body)["choices"][0]
            tokens = list(choice.get("token_ids") or [])
            logprobs = list((choice.get("logprobs") or {}).get("token_logprobs") or [])
            if len(logprobs) < len(tokens) or any(value is None for value in logprobs[:len(tokens)]):
                raise RuntimeError("The rollout endpoint returned no log-probability for some sampled tokens.")
            stop = next((position for position, token in enumerate(tokens) if token in stops), None)
            tokens = tokens[:stop + 1] if stop is not None else tokens
            return Rollout(tokenizer.decode(tokens, skip_special_tokens=True), tuple(tokens),
                           "eos" if stop is not None else "length", tuple(float(v) for v in logprobs[:len(tokens)]))

        try:
            return list(self.pool.map(sample, range(len(encoded))))
        finally:
            _post(self.host, self.port, "/v1/unload_lora_adapter", {"lora_name": name})
            shutil.rmtree(directory, ignore_errors=True)
