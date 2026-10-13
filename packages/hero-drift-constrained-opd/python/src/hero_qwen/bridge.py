"""Loopback-only OpenAI-compatible Qwen bridge for local DeepSeek Harness runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import threading
from collections.abc import Iterator
from typing import Any

MODEL_ID = "Qwen/Qwen3.5-4B"
INFERENCE_PROTOCOL = "hero-qwen-chat-eos-v1"
_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
_PARAMETER = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.DOTALL)


def chat_eos_token_ids(model_eos: int | list[int] | None, tokenizer_eos: int) -> list[int]:
    """A chat turn ends at the tokenizer EOS or the model's configured EOS."""
    values = ([] if model_eos is None else [model_eos] if isinstance(model_eos, int) else list(model_eos)) + [tokenizer_eos]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values):
        raise ValueError("Qwen chat EOS identifiers must be nonnegative integers.")
    return sorted(set(values))


def parse_qwen_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    for index, match in enumerate(_TOOL_CALL.finditer(text)):
        name = match.group(1).strip()
        if not name:
            raise ValueError("Qwen emitted a tool call without a function name.")
        arguments: dict[str, Any] = {}
        for parameter in _PARAMETER.finditer(match.group(2)):
            key = parameter.group(1).strip()
            raw = parameter.group(2).strip()
            try:
                arguments[key] = json.loads(raw)
            except json.JSONDecodeError:
                arguments[key] = raw
        call_digest = hashlib.sha256(f"{index}\0{match.group(0)}".encode()).hexdigest()[
            :24
        ]
        calls.append(
            {
                "id": f"call_{call_digest}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, separators=(",", ":")),
                },
            }
        )
    if "<tool_call>" in text and not calls:
        raise ValueError("Qwen emitted malformed tool-call markup.")
    content = _TOOL_CALL.sub("", text).strip()
    return content, calls


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {role!r}")
        content = message.get("content", "")
        if content is None:
            content = ""
        if isinstance(content, list):
            pieces = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            content = "".join(pieces)
        if not isinstance(content, str):
            raise TypeError("Message content must be text.")
        value: dict[str, Any] = {"role": role, "content": content}
        if role == "assistant" and message.get("tool_calls"):
            value["tool_calls"] = [
                _normalize_tool_call(call) for call in message["tool_calls"]
            ]
        if role == "tool":
            value["tool_call_id"] = message.get("tool_call_id")
        normalized.append(value)
    return normalized


def _normalize_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function", call)
    name = function.get("name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        raise ValueError(
            "Assistant tool calls require a function name and JSON-object arguments."
        )
    return {"type": "function", "function": {"name": name, "arguments": arguments}}


class QwenBridge:
    # Per-request ceilings. The DSH route keeps 1024 generated and 131072 prompt tokens; baseline runs use the
    # protocol's 5120 generated and 16384 prompt tokens per call.
    max_new_tokens = 1024
    max_input_tokens = 131072

    def __init__(self, checkpoint: str, version_id: str, *, max_new_tokens: int = 1024,
                 max_input_tokens: int = 131072) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.version_id = version_id
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, dtype=torch.bfloat16
        )
        self.model.to(self.device).eval()
        self.eos_token_ids = chat_eos_token_ids(self.model.generation_config.eos_token_id, self.tokenizer.eos_token_id)
        self.inference_protocol = INFERENCE_PROTOCOL
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.lock = threading.Lock()
        if max_new_tokens < 1 or max_input_tokens < 1:
            raise ValueError("Token ceilings must be positive.")
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens

    def count_tokens(self, text: str) -> int:
        if not isinstance(text, str):
            raise TypeError("Token counting requires text.")
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def clip(self, text: str, tokens: int, ellipsis: bool = True, tail: int = 0) -> str:
        """At most `tokens` tokens of `text`: the head, and the last `tail` of them taken from the end, joined and
        marked with " ..." when cut (the stream runner's `hero.clip`)."""
        if not isinstance(text, str) or tokens < 1 or not 0 <= tail < tokens:
            raise ValueError("Clipping requires text, a positive token count and a shorter tail.")
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ellipsis:
            return self.tokenizer.decode(ids[:tokens]).strip()
        if len(ids) <= tokens:
            return text
        head = self.tokenizer.decode(ids[:tokens - tail])
        return head + " ... " + self.tokenizer.decode(ids[-tail:]) if tail else head + " ..."

    def target_nll(self, request: dict[str, Any]) -> dict[str, Any]:
        """Mean NLL of `target` after the chat prompt. `maxTargetTokens` clips the target and `endOfTurn` appends
        <|im_end|>, as the stream runner scores verified answers."""
        messages = normalize_messages(request.get("messages", []))
        target = request.get("target")
        if not messages or not isinstance(target, str) or not target:
            raise ValueError("Residual scoring requires messages and a non-empty target.")
        with self.lock, self.torch.inference_mode():
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt",
                add_generation_prompt=True, enable_thinking=False,
            ).to(self.device)
            targets = self.tokenizer.encode(target, add_special_tokens=False)
            if request.get("maxTargetTokens") is not None:
                targets = targets[:int(request["maxTargetTokens"])]
            if request.get("endOfTurn"):
                targets = targets + [self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
            if not targets or prompt["input_ids"].shape[-1] > 131072:
                raise ValueError("Residual scoring exceeds prompt limits or has no target tokens.")
            target_ids = self.torch.tensor([targets], device=self.device)
            ids = self.torch.cat([prompt["input_ids"], target_ids], dim=-1)
            mask = self.torch.cat([prompt["attention_mask"], self.torch.ones_like(target_ids)], dim=-1)
            output = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=len(targets) + 1, use_cache=False)
            # The first retained logit predicts the first target token; the last
            # retained logit predicts beyond the target and must be excluded.
            loss = self.torch.nn.functional.cross_entropy(output.logits[0, :-1].float(), target_ids[0])
            return {"nll": float(loss.item()), "promptTokens": int(prompt["input_ids"].shape[-1]),
                    "targetTokens": len(targets), "model": MODEL_ID, "versionId": self.version_id}

    def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = normalize_messages(request.get("messages", []))
        if not messages:
            raise ValueError("messages must be non-empty.")
        tools = request.get("tools")
        max_tokens = min(
            int(request.get("max_tokens", request.get("max_completion_tokens", 1024))),
            self.max_new_tokens,
        )
        temperature = float(request.get("temperature", 0))
        with self.lock, self.torch.inference_mode():
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(self.device)
            prompt_tokens = inputs["input_ids"].shape[-1]
            if prompt_tokens > self.max_input_tokens:
                raise ValueError(
                    f"Prompt contains {prompt_tokens} tokens; the bridge permits at most {self.max_input_tokens}."
                )
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                eos_token_id=self.eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=temperature > 0,
                **(
                    {
                        "temperature": temperature,
                        "top_p": float(request.get("top_p", 1)),
                    }
                    if temperature > 0
                    else {}
                ),
            )
            output_ids = generated[0, prompt_tokens:]
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        content, tool_calls = parse_qwen_tool_calls(text)
        message: dict[str, Any] = {"role": "assistant", "content": content or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        response_id = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
            + output_ids.cpu().numpy().tobytes()
        ).hexdigest()[:24]
        reached_eos = output_ids.numel() > 0 and int(output_ids[-1]) in self.eos_token_ids
        finish_reason = "tool_calls" if tool_calls else ("stop" if reached_eos else "length")
        return {
            "id": f"chatcmpl-{response_id}",
            "object": "chat.completion",
            "created": 0,
            "model": MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(output_ids.numel()),
                "total_tokens": prompt_tokens + int(output_ids.numel()),
            },
            "hero": {
                "inferenceProtocol": self.inference_protocol,
                "eosTokenIds": self.eos_token_ids,
                "completionTokenIds": output_ids.cpu().tolist(),
            },
        }


def create_app(checkpoint: str, version_id: str, *, max_new_tokens: int = 1024, max_input_tokens: int = 131072):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    bridge = QwenBridge(checkpoint, version_id, max_new_tokens=max_new_tokens, max_input_tokens=max_input_tokens)
    app = FastAPI(title="HERO local Qwen bridge", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ready", "model": MODEL_ID, "versionId": bridge.version_id,
                "inferenceProtocol": bridge.inference_protocol, "eosTokenIds": bridge.eos_token_ids,
                "padTokenId": bridge.tokenizer.pad_token_id, "bridgeSha256": bridge.source_sha256,
                "maxNewTokens": bridge.max_new_tokens, "maxInputTokens": bridge.max_input_tokens}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "owned_by": "hero-local"}],
        }

    @app.post("/v1/chat/completions")
    def chat(request: dict[str, Any]):
        try:
            response = bridge.complete(request)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not request.get("stream", False):
            return response
        return StreamingResponse(_as_sse(response), media_type="text/event-stream")

    @app.post("/hero/token-counts")
    def token_counts(request: dict[str, Any]):
        texts = request.get("texts")
        if not isinstance(texts, list):
            raise HTTPException(status_code=422, detail="texts must be a list")
        return {"counts": [bridge.count_tokens(text) for text in texts], "versionId": bridge.version_id}

    @app.post("/hero/clip")
    def clip(request: dict[str, Any]):
        texts, tokens = request.get("texts"), request.get("tokens")
        if not isinstance(texts, list) or not isinstance(tokens, int):
            raise HTTPException(status_code=422, detail="texts must be a list and tokens an integer")
        try:
            return {"texts": [bridge.clip(text, tokens, bool(request.get("ellipsis", True)), int(request.get("tail", 0)))
                              for text in texts]}
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/hero/target-nll")
    def target_nll(request: dict[str, Any]):
        try:
            return bridge.target_nll(request)
        except (TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    return app


def _as_sse(response: dict[str, Any]) -> Iterator[str]:
    choice = response["choices"][0]
    message = choice["message"]
    delta = {key: value for key, value in message.items() if value is not None}
    if "tool_calls" in delta:
        delta["tool_calls"] = [
            {"index": index, **call} for index, call in enumerate(delta["tool_calls"])
        ]
    chunk = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    final = {
        **chunk,
        "usage": response["usage"],
        "choices": [
            {"index": 0, "delta": {}, "finish_reason": choice["finish_reason"]}
        ],
    }
    yield f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"
    yield f"data: {json.dumps(final, separators=(',', ':'))}\n\n"
    yield "data: [DONE]\n\n"


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(prog="hero-qwen-bridge")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--version-id", required=True)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Per-request generation ceiling (DSH default 1024; baseline protocol 5120)")
    parser.add_argument("--max-input-tokens", type=int, default=131072,
                        help="Per-request prompt ceiling (DSH default 131072; baseline protocol 16384)")
    arguments = parser.parse_args()
    if arguments.max_new_tokens < 1 or arguments.max_input_tokens < 1:
        parser.error("Token ceilings must be positive")
    uvicorn.run(
        create_app(arguments.checkpoint, arguments.version_id, max_new_tokens=arguments.max_new_tokens,
                   max_input_tokens=arguments.max_input_tokens),
        host=arguments.host,
        port=arguments.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
