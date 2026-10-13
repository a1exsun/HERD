"""On-Policy Harness Self-Distillation (OPHSD, arXiv:2605.08741) on the TRACE+17 stream.

The method code lives in `hero_baselines.ophsd_official` (harnesses, memory bank,
teacher-context builder, divergence losses). This module ports the parts of the
verl trainer and FSDP worker that cannot run on one GPU without verl, vLLM
and FSDP: the per-step loop of `OPHSDTrainer.fit`, the harness dispatch of
`_run_harness_inference`, the memory-bank update of `_update_memory_bank` and
the micro-batched update of `OPSDWorker._opsd_training_step`.

Per update: the student samples one rollout per prompt from
the bare prompt; the harness runs on the current student (draft-verify on
closed-label tasks, plan-solve otherwise); the teacher, the stage-initial
weights (LoRA disabled), reads the harness's final-call messages; the loss is
`reverse_kl`, KL(p_student || p_teacher) over the full vocabulary,
token-mean per micro-batch of two sequences; then the batch's labelled
examples enter the memory bank.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

from .protocol import EVAL_MAX_INPUT_TOKENS, EVAL_MAX_NEW_TOKENS


DRAFT_VERIFY = "draft-verify"
PLAN_SOLVE = "plan-solve"
MAX_CLOSED_LABELS = 10
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
EMBEDDING_TREE_SHA256 = "b38884accb3083dcd2e5868733dd7fb59f839ad7e1592975cbd93c3d58d21eba"
UPSTREAM_COMMIT = "36cc494a3d4672f4bac18c7e11374c8dd4b50eed"
HARNESS_MODEL_NAME = "student"  # the name the harness passes to its chat client; ignored locally


@dataclass(frozen=True)
class OphsdConfig:
    # Method defaults (ophsd_trainer.yaml and scripts/train_ophsd_*.sh).
    loss_type: str = "reverse_kl"
    loss_chunk_size: int = 256
    micro_batch_size: int = 2
    weight_decay: float = 0.1
    rollout_temperature: float = 1.0
    draft_k: int = 5
    confirm_k: int = 5
    challenge_k: int = 5
    cold_start_threshold: int = 10
    use_gt: bool = True
    # Project protocol: shared output and input ceilings; their sum is both the
    # teacher/student sequence limit (ophsd.max_length) and the harness
    # server's context length (rollout.max_model_len).
    max_new_tokens: int = EVAL_MAX_NEW_TOKENS
    max_input_tokens: int = EVAL_MAX_INPUT_TOKENS
    max_length: int = EVAL_MAX_INPUT_TOKENS + EVAL_MAX_NEW_TOKENS
    # Execution only; no effect on the objective.
    generation_batch_size: int = 32
    generation_token_budget: int = 450_000
    generation_prefill_budget: int = 65_536
    kl_chunk_tokens: int = 1024


# Fields that change speed or memory but not the objective; runs may resume across them.
EXECUTION_ONLY_FIELDS = frozenset({"generation_batch_size", "generation_token_budget",
                                   "generation_prefill_budget", "kl_chunk_tokens"})


# ---------------------------------------------------------------------------
# Harness selection and task fields


def closed_label_set(records: Sequence[Mapping[str, Any]]) -> list[str] | None:
    labels = sorted({str(record["answer"]).strip() for record in records})
    return labels if len(labels) <= MAX_CLOSED_LABELS else None


def select_harness(records: Sequence[Mapping[str, Any]]) -> str:
    """Draft-verify for closed-label (classification) tasks, plan-solve otherwise."""
    if not records:
        raise ValueError("Harness selection needs the stage's training records.")
    return DRAFT_VERIFY if closed_label_set(records) is not None else PLAN_SOLVE


def task_inputs(prompts: Sequence[str]) -> list[str]:
    """Each prompt without the instruction prefix and answer-cue suffix that every prompt
    of the task shares, cut at line boundaries: the analogue of the `fact` field
    (the LawBench case facts, the USPTO reaction SMILES) used for retrieval and examples."""
    prefix = os.path.commonprefix(list(prompts))
    prefix = prefix[:prefix.rfind("\n") + 1]
    suffix = os.path.commonprefix([p[::-1] for p in prompts])[::-1]
    suffix = suffix[suffix.find("\n"):] if "\n" in suffix else ""
    result = []
    for prompt in prompts:
        core = prompt[len(prefix):len(prompt) - len(suffix)] if len(prefix) + len(suffix) < len(prompt) else ""
        result.append(core.strip() or prompt)
    return result


def reference_answer(record: Mapping[str, Any]) -> str:
    answer = record.get("answer")
    if isinstance(answer, str) and answer:
        return answer
    references = [value for value in record.get("references", []) if isinstance(value, str) and value]
    if references:
        return references[0]
    raise ValueError(f"Record {record.get('id')} has no reference answer.")


# ---------------------------------------------------------------------------
# Generation


@dataclass(frozen=True)
class Generation:
    text: str
    token_ids: tuple[int, ...]
    prompt_tokens: int
    stop_reason: str


def chat_ids(tokenizer, prompt: str | Sequence[Mapping[str, str]]) -> list[int]:
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    return list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              enable_thinking=False, return_dict=True)["input_ids"])


def generation_chunks(lengths: Sequence[int], *, max_new_tokens: int, batch_size: int,
                      token_budget: int, prefill_budget: int) -> list[list[int]]:
    """Group prompt indices, shortest first, so each padded chunk fits both budgets.

    `token_budget` bounds the worst-case KV cache (prompt plus generated tokens);
    `prefill_budget` bounds the padded prompt tokens prefilled at once, whose
    linear-attention chunk states dominate memory for long prompts.
    """
    order = sorted(range(len(lengths)), key=lambda index: lengths[index])
    chunks: list[list[int]] = []
    for index in order:
        chunk = chunks[-1] if chunks else None
        if chunk is not None and len(chunk) < batch_size and \
                (len(chunk) + 1) * (lengths[index] + max_new_tokens) <= token_budget and \
                (len(chunk) + 1) * lengths[index] <= prefill_budget:
            chunk.append(index)
        else:
            chunks.append([index])
    return chunks


def sample_batch(model, tokenizer, prompts: list, *, temperature: float, max_new_tokens: int,
                 seed: int, eos: Sequence[int], batch_size: int, token_budget: int,
                 prefill_budget: int, top_p: float = 1.0) -> list[Generation]:
    """Left-padded batched sampling (top-k off) from the model's current adapter state.

    `prompts` are user-message strings or chat message lists.
    """
    import torch

    device = next(model.parameters()).device
    encoded = [chat_ids(tokenizer, prompt) for prompt in prompts]
    results: list[Generation | None] = [None] * len(prompts)
    pad = tokenizer.pad_token_id
    chunks = generation_chunks([len(tokens) for tokens in encoded], max_new_tokens=max_new_tokens,
                               batch_size=batch_size, token_budget=token_budget, prefill_budget=prefill_budget)
    for start, chunk in enumerate(chunks):
        width = max(len(encoded[index]) for index in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, index in enumerate(chunk):
            tokens = encoded[index]
            ids[row, width - len(tokens):] = torch.tensor(tokens)
            mask[row, width - len(tokens):] = 1
        torch.manual_seed(seed + start)
        with torch.inference_mode():
            output = model.generate(input_ids=ids.to(device), attention_mask=mask.to(device), do_sample=True,
                                    temperature=temperature, top_p=top_p, top_k=0, max_new_tokens=max_new_tokens,
                                    eos_token_id=list(eos), pad_token_id=pad)
        for row, index in enumerate(chunk):
            tokens = output[row, width:].tolist()
            stop = next((position for position, token in enumerate(tokens) if token in eos), None)
            tokens = tokens[:stop + 1] if stop is not None else tokens
            results[index] = Generation(tokenizer.decode(tokens, skip_special_tokens=True), tuple(tokens),
                                        len(encoded[index]), "eos" if stop is not None else "length")
    return [result for result in results if result is not None]


class LocalChatClient:
    """Stand-in for the colocated vLLM OpenAI server that the harnesses call.

    The harness code runs in one thread per sample, as in the trainer;
    each `chat.completions.create` blocks until every live sample thread
    is waiting, and the waiting requests are then served together by batched
    sampling from the current student. Requests are ordered by content, so the
    batches and seeds do not depend on thread timing. `max_tokens` is capped at the
    protocol output ceiling; as in vLLM, a request whose prompt plus `max_tokens`
    exceeds `max_model_len` is rejected with HTTP 400, and the harness's own
    `call_api` then halves `max_tokens` or gives up with an empty reply.
    """

    def __init__(self, generate: Callable[[list, float, float, int, int], list[Generation]],
                 count_tokens: Callable[[list], int], *, max_model_len: int, max_new_tokens: int):
        self._generate, self._count = generate, count_tokens
        self.max_model_len, self.max_new_tokens = max_model_len, max_new_tokens
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self._condition = threading.Condition()
        self._pending: list[dict[str, Any]] = []
        self._live = 0
        self.requests: list[dict[str, Any]] = []

    def _create(self, *, model: str, messages: list, temperature: float, max_tokens: int,
                top_p: float | None = None, timeout: Any = None, extra_body: Any = None, **_):
        request = {"messages": messages, "temperature": float(temperature),
                   "top_p": 1.0 if top_p is None else float(top_p),
                   "max_tokens": min(int(max_tokens), self.max_new_tokens), "done": False}
        with self._condition:
            self._pending.append(request)
            self._condition.notify_all()
            self._condition.wait_for(lambda: request["done"])
        if "error" in request:
            raise RuntimeError(request["error"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=request["text"]))])

    def run(self, jobs: Sequence[Callable[[], Any]], seed: int) -> list[Any]:
        results: list[Any] = [None] * len(jobs)

        def worker(index: int) -> None:
            try:
                results[index] = jobs[index]()
            finally:
                with self._condition:
                    self._live -= 1
                    self._condition.notify_all()

        with self._condition:
            self._live = len(jobs)
        threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(len(jobs))]
        for thread in threads:
            thread.start()
        round_ = 0
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._live == 0 or (self._pending and len(self._pending) == self._live))
                if not self._pending:
                    break
                batch, self._pending = self._pending, []
            self._serve(batch, seed + 7919 * round_)
            round_ += 1
            with self._condition:
                self._condition.notify_all()
        for thread in threads:
            thread.join()
        return results

    def _serve(self, batch: list[dict[str, Any]], seed: int) -> None:
        batch.sort(key=lambda r: (r["temperature"], r["top_p"], r["max_tokens"],
                                  json.dumps(r["messages"], ensure_ascii=False, sort_keys=True)))
        for request in batch:
            request["promptTokens"] = self._count(request["messages"])
            if request["promptTokens"] + request["max_tokens"] > self.max_model_len:
                request["error"] = (f"Error code: 400 - max_tokens={request['max_tokens']} is too large: "
                                    f"input_tokens={request['promptTokens']}, max_model_len={self.max_model_len}")
        ready = [r for r in batch if "error" not in r]
        groups: dict[tuple, list[dict[str, Any]]] = {}
        for request in ready:
            groups.setdefault((request["temperature"], request["top_p"], request["max_tokens"]), []).append(request)
        for offset, (key, requests) in enumerate(sorted(groups.items())):
            temperature, top_p, max_tokens = key
            outputs = self._generate([r["messages"] for r in requests], temperature, top_p, max_tokens,
                                     seed + 104_729 * offset)
            for request, output in zip(requests, outputs):
                request["text"] = output.text.strip()
                request["stopReason"], request["outputTokens"] = output.stop_reason, len(output.token_ids)
        with self._condition:
            for request in batch:
                self.requests.append({k: request[k] for k in ("temperature", "top_p", "max_tokens", "promptTokens")
                                      } | {k: request[k] for k in ("stopReason", "outputTokens", "error") if k in request})
                request["done"] = True
            self._condition.notify_all()


# ---------------------------------------------------------------------------
# Harness dispatch and memory bank (OPHSDTrainer._run_harness_inference / _update_memory_bank)


_current_query = threading.local()  # the training-item index each harness thread is serving


def query_excluding_bank():
    """The `MemoryBank` whose retrievals never return the query's own entry.

    A stage draws 32 x 32 = 1,024 samples from 500 records, so from its second epoch
    the query is already in the bank and would be retrieved with its gold label.
    Each retrieval asks for one extra entry and drops the query's own.
    """
    from .ophsd_official.harnesses._memory_bank import MemoryBank

    class QueryExcludingMemoryBank(MemoryBank):
        def retrieve(self, query_emb, k=5, filter_labels=None, same_label=True):
            query = getattr(_current_query, "index", None)
            found = super().retrieve(query_emb, k=k + 1, filter_labels=filter_labels, same_label=same_label)
            return [entry for entry in found if entry.get("index") != query][:k]

    return QueryExcludingMemoryBank()


class StageHarness:
    """One stage's harness, its memory bank and its training items."""

    def __init__(self, kind: str, records: Sequence[Mapping[str, Any]], client: LocalChatClient,
                 embed: Callable[[list[str]], list[list[float]]], config: OphsdConfig):
        import numpy as np

        from .ophsd_official.harnesses.math.plan_solve import PlanSolveHarness
        from .ophsd_official.harnesses.uspto.draft_verify import USPTODraftVerificationHarness

        self.kind, self.client = kind, client
        prompts = [str(r["prompt"]) for r in records]
        if kind == DRAFT_VERIFY:
            facts = task_inputs(prompts)
            self.items = [{"instruction": "", "question": prompt, "fact": fact, "answer": str(r["answer"]).strip(),
                           "label_str": str(r["answer"]).strip(), "accusations": [str(r["answer"]).strip()]}
                          for r, prompt, fact in zip(records, prompts, facts)]
            self.embeddings = np.asarray(embed(facts), dtype=np.float32)
            self.memory_bank = query_excluding_bank()
            self.added: set[int] = set()
            self.harness = USPTODraftVerificationHarness(
                client=client, model=HARNESS_MODEL_NAME, memory_bank=self.memory_bank,
                embed_fn=lambda text: np.asarray(embed([text])[0], dtype=np.float32),
                draft_k=config.draft_k, confirm_k=config.confirm_k, challenge_k=config.challenge_k,
                cold_start_threshold=config.cold_start_threshold, labels=closed_label_set(records))
        else:
            self.items = [{"question": prompt, "answer": reference_answer(r)} for r, prompt in zip(records, prompts)]
            self.memory_bank = None
            self.harness = PlanSolveHarness(client=client, model=HARNESS_MODEL_NAME, use_gt=config.use_gt)

    def run(self, indices: Sequence[int], seed: int) -> list[dict | None]:
        def job(index: int):
            item = self.items[index]
            _current_query.index = index
            try:
                if self.kind == DRAFT_VERIFY:
                    result = self.harness.predict(item["instruction"], item["question"], item["fact"],
                                                  precomputed_emb=self.embeddings[index].copy())
                    result.pop("query_emb", None)
                    return result
                # PlanSolveHarness.predict_batch's per-item call; the trainer forwards no solution.
                answer = item["answer"] if self.harness.use_gt else ""
                return self.harness.predict(item["question"], answer=answer, solution="")
            except Exception:
                return None

        return self.client.run([lambda index=index: job(index) for index in indices], seed)

    def observe(self, indices: Sequence[int]) -> None:
        """Add the batch's labelled examples (true labels) to the memory bank, once each."""
        if self.memory_bank is None:
            return
        for index in indices:
            if index in self.added:
                continue
            item = self.items[index]
            self.memory_bank.add({"fact": item["fact"], "accusations": item["accusations"],
                                  "label_str": item["label_str"], "index": index}, self.embeddings[index].copy())
            self.added.add(index)


# ---------------------------------------------------------------------------
# Divergence (OPSDWorker._opsd_training_step, method loss functions)


def response_hidden(model, seq: Mapping[str, Any]):
    """Final-norm hidden states at the positions whose logits predict the response tokens
    (the `loss_mask[:, 1:]` positions of the padded forward)."""
    import torch

    length = int(seq["attention_mask"].sum())
    ids = seq["input_ids"][:length]
    prompt_length = int((seq["loss_mask"][:length] == 0).sum())
    decoder = model.get_base_model().model
    device = next(model.parameters()).device
    # Text-only mRoPE positions are plain arange, as in the method's position_ids.
    positions = torch.arange(length - 1, device=device)[None]
    hidden = decoder(input_ids=ids[:-1][None].to(device), position_ids=positions, use_cache=False)[0]
    return hidden[0, prompt_length - 1:]


def _chunk_divergence_sum(student_hidden, teacher_hidden, weight, loss_type: str, chunk_size: int):
    from .ophsd_official.src.ophsd.losses import LOSS_FN_MAP

    loss, n_tokens = LOSS_FN_MAP[loss_type](teacher_hidden @ weight.T, student_hidden @ weight.T, chunk_size=chunk_size)
    return loss * n_tokens


def sequence_divergence_sum(model, student_seq, teacher_seq, config: OphsdConfig):
    """Sum over response tokens of the divergence; logits are materialised in
    recomputed chunks, the teacher is the stage-initial weights (LoRA disabled)."""
    import torch
    from torch.utils.checkpoint import checkpoint

    weight = model.get_base_model().lm_head.weight
    if weight.requires_grad:
        raise ValueError("The LM head must stay frozen.")
    with torch.no_grad(), model.disable_adapter():
        teacher_hidden = response_hidden(model, teacher_seq).detach()
    student_hidden = response_hidden(model, student_seq)
    if teacher_hidden.shape[0] != student_hidden.shape[0]:
        raise RuntimeError("Teacher and student response token counts diverged. "
                           "This usually means prompt truncation dropped response tokens.")
    total = student_hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, student_hidden.shape[0], config.kl_chunk_tokens):
        stop = start + config.kl_chunk_tokens
        total = total + checkpoint(_chunk_divergence_sum, student_hidden[start:stop], teacher_hidden[start:stop],
                                   weight, config.loss_type, config.loss_chunk_size, use_reentrant=False)
    return total, int(student_hidden.shape[0])


def opsd_update(model, optimizer, trainable, student_seqs, teacher_seqs, *, gradient_clip: float,
                config: OphsdConfig, on_sample: Callable[[int, float, int], None]) -> dict[str, Any]:
    """One optimizer step: micro-batches of `micro_batch_size` rows, each a token mean,
    averaged over the active micro-batches; clip, and skip non-finite steps."""
    import torch

    micro_batches = [list(range(start, min(start + config.micro_batch_size, len(student_seqs))))
                     for start in range(0, len(student_seqs), config.micro_batch_size)]
    grad_accum = max(1, len(micro_batches))
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_tokens = 0.0, 0
    for rows in micro_batches:
        n_tokens = sum(int(student_seqs[row]["loss_mask"].sum()) for row in rows)
        if n_tokens == 0:
            continue
        micro_sum = 0.0
        for row in rows:  # backward per row; the gradient equals that of the micro-batch mean
            divergence, tokens = sequence_divergence_sum(model, student_seqs[row], teacher_seqs[row], config)
            (divergence / n_tokens / grad_accum).backward()
            micro_sum += float(divergence.detach())
            on_sample(row, float(divergence.detach()) / tokens, tokens)
        total_loss += micro_sum / n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        optimizer.zero_grad(set_to_none=True)
        return {"loss": 0.0, "gradNorm": 0.0, "numTokens": 0, "stepped": False}
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
    stepped = bool(torch.isfinite(grad_norm))
    if stepped:
        optimizer.step()
    else:
        optimizer.zero_grad(set_to_none=True)
    return {"loss": total_loss / grad_accum, "gradNorm": float(grad_norm), "numTokens": total_tokens,
            "stepped": stepped}


# ---------------------------------------------------------------------------
# Stage training (OPHSDTrainer.fit, one stage)


@dataclass(frozen=True)
class StageSummary:
    stage: int
    task_id: str
    harness: str
    updates: int
    batch_size: int
    mean_loss: float
    skipped_samples: int
    skipped_steps: int
    excluded_long_prompts: list[str]
    seconds: float


@dataclass(frozen=True)
class StageProgress:
    """Resumable position inside a stage: the next update to run and what came before it."""
    next_update: int
    losses: list[float]
    seconds: float
    skipped_samples: int = 0
    skipped_steps: int = 0


def eligible_records(tokenizer, records: Sequence[Mapping[str, Any]], config: OphsdConfig):
    eligible, excluded = [], []
    for record in records:
        (eligible if len(chat_ids(tokenizer, str(record["prompt"]))) <= config.max_input_tokens else excluded).append(record)
    return eligible, excluded


def update_schedule(count: int, *, updates: int, batch_size: int, seed: int) -> list[list[int]]:
    """Record indices per update in dataloader order: verl's train dataloader, a RandomSampler
    drawing one `torch.randperm` per epoch from a generator (seeded here) with `drop_last=True`, so each epoch's incomplete
    last batch is dropped."""
    import torch

    if count < batch_size:
        raise ValueError("A stage needs at least one full batch of records.")
    generator = torch.Generator()
    generator.manual_seed(seed)
    schedule: list[list[int]] = []
    while len(schedule) < updates:
        order = torch.randperm(count, generator=generator).tolist()
        for start in range(0, count - batch_size + 1, batch_size):
            if len(schedule) == updates:
                break
            schedule.append(order[start:start + batch_size])
    return schedule


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=float))


def train_stage(model, tokenizer, *, stage: int, task_id: str, records: Sequence[Mapping[str, Any]], kind: str,
                embed: Callable[[list[str]], list[list[float]]], updates: int, batch_size: int,
                learning_rate: float, gradient_clip: float, seed: int, eos: Sequence[int], config: OphsdConfig,
                log: Callable[[dict[str, Any]], None],
                progress: StageProgress | None = None, optimizer_state: Any = None,
                save_checkpoint: Callable[[StageProgress, Any], None] | None = None,
                checkpoint_every: int = 4, deadline: float | None = None) -> StageSummary | None:
    """Fixed-update OPHSD on one stage; the caller merges the LoRA afterwards.

    Checkpoints every `checkpoint_every` updates and before `deadline` (epoch seconds);
    returns None when it stops early at the deadline. Resuming replays the memory-bank
    updates of completed steps, so the draft-verify bank is reconstructed exactly.
    """
    import torch

    from .ophsd_official.src.ophsd.batch_builder import build_ophsd_pairs

    started = time.perf_counter()
    eligible, excluded = eligible_records(tokenizer, records, config)
    if not eligible:
        raise ValueError(f"Stage {stage} has no training prompt within the input ceiling.")

    def generate(prompts, temperature, top_p, max_new_tokens, generation_seed):
        return sample_batch(model, tokenizer, prompts, temperature=temperature, top_p=top_p,
                            max_new_tokens=max_new_tokens, seed=generation_seed, eos=eos,
                            batch_size=config.generation_batch_size, token_budget=config.generation_token_budget,
                            prefill_budget=config.generation_prefill_budget)

    client = LocalChatClient(generate, lambda messages: len(chat_ids(tokenizer, messages)),
                             max_model_len=config.max_length, max_new_tokens=config.max_new_tokens)
    harness = StageHarness(kind, eligible, client, embed, config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=config.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    first = progress.next_update if progress else 0
    losses = list(progress.losses) if progress else []
    skipped_samples = progress.skipped_samples if progress else 0
    skipped_steps = progress.skipped_steps if progress else 0
    prior = progress.seconds if progress else 0.0
    schedule = update_schedule(len(eligible), updates=updates, batch_size=batch_size, seed=seed + stage)
    for update in range(first):
        harness.observe(schedule[update])
    longest = 0.0
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def encode(messages):
        return chat_ids(tokenizer, messages)

    def checkpoint(next_update: int) -> None:
        if save_checkpoint is not None:
            save_checkpoint(StageProgress(next_update, list(losses), prior + time.perf_counter() - started,
                                          skipped_samples, skipped_steps), optimizer.state_dict())

    for update in range(first, updates):
        if deadline is not None and update > first and time.time() + 1.5 * longest + 600 > deadline:
            checkpoint(update)
            return None
        indices = schedule[update]
        batch = [eligible[index] for index in indices]
        update_seed = seed + stage * 1_000_003 + update * 10_007
        update_started = time.perf_counter()
        model.eval()
        # 1. Student rollout from the bare prompt.
        rollouts = generate([str(r["prompt"]) for r in batch], config.rollout_temperature, 1.0,
                            config.max_new_tokens, update_seed)
        # 2. Harness on the current student.
        client.requests.clear()
        outputs = harness.run(indices, update_seed + 1)
        harness_seconds = time.perf_counter() - update_started
        # 4. Teacher/student pairs from the harness's final-call messages.
        student_seqs, teacher_seqs, kept = build_ophsd_pairs(
            [[{"role": "user", "content": str(r["prompt"])}] for r in batch], [list(g.token_ids) for g in rollouts],
            encode, config.max_length, pad, harness_outputs=outputs)
        skipped_samples += len(batch) - len(kept)
        # 5. Divergence, backward, step.
        model.train()
        sample_losses: dict[int, tuple[float, int]] = {}
        result = opsd_update(model, optimizer, trainable, student_seqs, teacher_seqs, gradient_clip=gradient_clip,
                             config=config, on_sample=lambda row, loss, tokens: sample_losses.__setitem__(row, (loss, tokens)))
        skipped_steps += not result["stepped"]
        # 6. Memory bank.
        harness.observe(indices)
        for position, (record, rollout, output) in enumerate(zip(batch, rollouts, outputs)):
            row = kept.index(position) if position in kept else None
            log({"kind": "sample", "stage": stage, "update": update, "id": str(record["id"]), "harness": kind,
                 "kept": row is not None, "loss": sample_losses[row][0] if row is not None else None,
                 "rolloutTokens": len(rollout.token_ids), "rolloutStopReason": rollout.stop_reason,
                 "rollout": rollout.text,
                 "teacherPromptTokens": int((teacher_seqs[row]["loss_mask"] == 0).sum()
                                            - (teacher_seqs[row]["attention_mask"] == 0).sum()) if row is not None else None,
                 "harnessOutput": _json_safe(output)})
        losses.append(result["loss"])
        seconds = time.perf_counter() - update_started
        longest = max(longest, seconds)
        log({"kind": "update", "stage": stage, "update": update, **result, "keptSamples": len(kept),
             "harnessCalls": list(client.requests), "harnessSeconds": harness_seconds, "seconds": seconds,
             "cudaPeakBytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None})
        if (update + 1) % checkpoint_every == 0 and update + 1 < updates:
            checkpoint(update + 1)
    return StageSummary(stage, task_id, kind, updates, batch_size, sum(losses) / len(losses), skipped_samples,
                        skipped_steps, [str(r["id"]) for r in excluded], prior + time.perf_counter() - started)


def summary_dict(summary: StageSummary) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(summary)))
