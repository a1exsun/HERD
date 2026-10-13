import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from hero_baselines.ophsd import (
    DRAFT_VERIFY, PLAN_SOLVE, Generation, LocalChatClient, OphsdConfig, StageHarness, select_harness, task_inputs,
)
from hero_baselines.protocol import evaluated_task_indices, general_panel_kind

ROOT = Path(__file__).resolve().parents[4]
UPSTREAM = ROOT / "packages/baselines/ophsd/upstream"
ADAPTED = ROOT / "packages/baselines/python/src/hero_baselines/ophsd_official"


def records(labels, prefix="x"):
    return [{"id": f"{prefix}{i}", "prompt": f"Classify.\nInput:\n{'a' * i} q{i}\nOutput:", "answer": label}
            for i, label in enumerate(labels)]


def fake_embed(texts):
    # Two-dimensional embedding: the number of "a" characters versus a constant.
    return [[text.count("a") + 0.01, 1.0] for text in texts]


class FakeGenerate:
    """Replies with `reply(messages)`; records each batched call."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, prompts, temperature, top_p, max_new_tokens, seed):
        self.calls.append((prompts, temperature, top_p, max_new_tokens, seed))
        return [Generation(self.reply(p), (1, 2), 3, "eos") for p in prompts]


def client_for(reply, max_model_len=21504):
    generate = FakeGenerate(reply)
    count = lambda messages: sum(len(m["content"]) for m in messages)
    return LocalChatClient(generate, count, max_model_len=max_model_len, max_new_tokens=5120), generate


def test_upstream_matches_its_manifest():
    manifest = json.loads((UPSTREAM / "UPSTREAM.json").read_text())
    assert manifest["commit"] == "36cc494a3d4672f4bac18c7e11374c8dd4b50eed"
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((UPSTREAM / relative).read_bytes()).hexdigest() == digest, relative


def test_only_the_listed_files_are_modified():
    unchanged = ["harnesses/__init__.py", "harnesses/_api.py", "harnesses/_memory_bank.py",
                 "harnesses/math/config.py", "src/__init__.py", "src/ophsd/__init__.py", "src/ophsd/losses.py"]
    modified = ["harnesses/uspto/__init__.py", "harnesses/uspto/config.py", "harnesses/uspto/draft_verify.py",
                "harnesses/math/__init__.py", "harnesses/math/plan_solve.py", "src/ophsd/batch_builder.py"]
    for relative in unchanged:
        assert (ADAPTED / relative).read_bytes() == (UPSTREAM / "ophsd_train" / relative).read_bytes(), relative
    for relative in modified:
        assert (ADAPTED / relative).read_bytes() != (UPSTREAM / "ophsd_train" / relative).read_bytes(), relative
    present = sorted(p.relative_to(ADAPTED).as_posix() for p in ADAPTED.rglob("*.py") if "__pycache__" not in p.parts)
    assert present == sorted(["__init__.py"] + unchanged + modified)


def test_harness_selection_follows_closed_label_sets():
    assert select_harness(records(["A", "B", "C"] * 5)) == DRAFT_VERIFY
    assert select_harness(records([f"summary {i}" for i in range(20)])) == PLAN_SOLVE


def test_registered_stream_assigns_draft_verify_to_the_six_closed_label_tasks():
    from hero_baselines.protocol import load_manifest, read_jsonl

    data = ROOT / "experiments/data/trace-plus-17"
    chosen = {task["id"]: select_harness(read_jsonl(data / task["files"]["train"]["path"]))
              for task in load_manifest(data)["tasks"]}
    assert sorted(t for t, h in chosen.items() if h == DRAFT_VERIFY) == sorted(
        ["trace:C-STANCE", "trace:FOMC", "superni:195", "superni:346", "superni:108", "superni:828"])


def test_task_inputs_drop_the_shared_instruction_and_answer_cue():
    prompts = ["Classify the tweet.\n\nInput:\nfirst tweet\n\nOutput:", "Classify the tweet.\n\nInput:\nsecond\n\nOutput:"]
    assert task_inputs(prompts) == ["first tweet", "second"]
    assert task_inputs(["same", "same"]) == ["same", "same"]


def test_task_inputs_of_the_draft_verify_tasks_carry_no_instruction():
    from hero_baselines.protocol import load_manifest, read_jsonl

    data = ROOT / "experiments/data/trace-plus-17"
    for task in load_manifest(data)["tasks"]:
        rows = read_jsonl(data / task["files"]["train"]["path"])
        if select_harness(rows) != DRAFT_VERIFY:
            continue
        prompts = [r["prompt"] for r in rows]
        facts = task_inputs(prompts)
        instruction = prompts[0].split("\n")[0]
        assert all(fact and fact != prompt for fact, prompt in zip(facts, prompts)), task["id"]
        assert sum(instruction in fact for fact in facts) <= len(facts) // 50, task["id"]


def test_label_extraction_scans_for_the_earliest_whole_word_label():
    from hero_baselines.ophsd_official.harnesses.uspto.draft_verify import extract_class

    assert extract_class("B", ["A", "B", "C"]) == "B"
    assert extract_class("Answer:\nB\n\nA is wrong", ["A", "B", "C"]) == "B"
    assert extract_class("A.支持", ["A", "B", "C"]) == "A"
    assert extract_class("a b c", ["A", "B", "C"]) is None
    assert extract_class("No, it is not.", ["no", "yes"]) == "no"
    assert extract_class("Bravo", ["B"]) is None
    assert extract_class("<think>A</think> C", ["A", "B", "C"]) == "C"


def test_local_client_batches_waiting_requests_deterministically():
    client, generate = client_for(lambda messages: messages[-1]["content"].upper())
    from hero_baselines.ophsd_official.harnesses._api import call_api

    jobs = [lambda text=text: call_api(client, "m", [{"role": "user", "content": text}]) for text in ("b", "a", "c")]
    assert client.run(jobs, seed=5) == ["B", "A", "C"]
    (prompts, temperature, top_p, max_tokens, seed), = generate.calls
    assert [p[0]["content"] for p in prompts] == ["a", "b", "c"]
    assert (temperature, top_p, max_tokens, seed) == (0.1, 1.0, 5120, 5)


def test_local_client_rejects_over_length_requests_like_vllm():
    from hero_baselines.ophsd_official.harnesses._api import call_api

    client, generate = client_for(lambda messages: "ok", max_model_len=5500)
    # 1,000 prompt tokens + min(8,192, 5,120) > 5,500: the harness halves its 8,192 to 4,096, which fits.
    assert client.run([lambda: call_api(client, "m", [{"role": "user", "content": "x" * 1000}])], seed=0) == ["ok"]
    assert [r["max_tokens"] for r in client.requests] == [5120, 4096] and generate.calls[0][3] == 4096
    client, generate = client_for(lambda messages: "ok", max_model_len=600)
    # Even 512 new tokens do not fit: the harness gives up with an empty reply.
    assert client.run([lambda: call_api(client, "m", [{"role": "user", "content": "x" * 1000}])], seed=0) == [""]
    assert generate.calls == [] and all("error" in r for r in client.requests)


def stage_harness(kind, rows, reply):
    client, generate = client_for(reply)
    return StageHarness(kind, rows, client, fake_embed, OphsdConfig()), generate


def test_draft_verify_teacher_is_the_cold_start_prompt_until_the_bank_warms_up():
    from hero_baselines.ophsd_official.src.ophsd.batch_builder import _get_teacher_messages_from_harness

    rows = records(["A", "B"] * 10)
    harness, _ = stage_harness(DRAFT_VERIFY, rows, lambda messages: "A")
    harness.observe(range(9))
    (output,) = harness.run([12], seed=0)
    assert output["mode"] == "cold_start"
    student = [{"role": "user", "content": rows[12]["prompt"]}]
    assert _get_teacher_messages_from_harness(output, student) == student


def test_draft_verify_teacher_is_the_verify_call_with_label_confirmers_and_challengers():
    from hero_baselines.ophsd_official.src.ophsd.batch_builder import _get_teacher_messages_from_harness

    rows = records(["A", "B"] * 10)
    harness, generate = stage_harness(DRAFT_VERIFY, rows, lambda messages: "B, because of the draft reasoning")
    harness.observe(range(16))
    (output,) = harness.run([3], seed=0)
    assert output["mode"] == "full" and output["draft_labels"] == ["B"]
    trace = output["trace"]
    assert len(trace["draft"]["retrieved"]) == 5
    assert {e["label_str"] for e in trace["verify"]["confirmers"]} == {"B"}
    assert {e["label_str"] for e in trace["verify"]["challengers"]} == {"A"}
    retrieved = trace["draft"]["retrieved"] + trace["verify"]["confirmers"] + trace["verify"]["challengers"]
    assert 3 in harness.added and all(e["index"] != 3 for e in retrieved)  # x3 is in the bank but never retrieved
    teacher = _get_teacher_messages_from_harness(output, [{"role": "user", "content": rows[3]["prompt"]}])
    assert teacher == trace["verify"]["messages"]
    assert "as class B (B)" in teacher[1]["content"] and "draft reasoning" not in teacher[1]["content"]
    assert [call[1] for call in generate.calls] == [0.1, 0.1]  # draft, verify


def test_bank_retrieval_excludes_only_the_query_itself():
    import numpy as np

    rows = records(["A", "B"] * 10)
    harness, _ = stage_harness(DRAFT_VERIFY, rows, lambda messages: "A")
    harness.observe(range(20))
    from hero_baselines import ophsd

    query = harness.embeddings[4]
    ophsd._current_query.index = 4
    found = harness.memory_bank.retrieve(query, k=5)
    ophsd._current_query.index = None
    assert len(found) == 5 and 4 not in [e["index"] for e in found]
    assert harness.memory_bank.retrieve(query, k=5)[0]["index"] == 4  # without a current query nothing is excluded
    assert np.isclose(found[0]["similarity"], max(e["similarity"] for e in found))


def test_unparsed_draft_falls_back_to_the_draft_call_messages():
    from hero_baselines.ophsd_official.src.ophsd.batch_builder import _get_teacher_messages_from_harness

    rows = records(["A", "B"] * 10)
    harness, _ = stage_harness(DRAFT_VERIFY, rows, lambda messages: "I cannot decide")
    harness.observe(range(16))
    (output,) = harness.run([3], seed=0)
    assert output["mode"] == "draft_only"
    assert _get_teacher_messages_from_harness(output, []) == output["trace"]["draft"]["messages"]


def test_plan_solve_plans_without_the_reference_and_verifies_with_the_answer():
    from hero_baselines.ophsd_official.src.ophsd.batch_builder import _get_teacher_messages_from_harness

    rows = [{"id": "m1", "prompt": "Summarise the meeting.", "answer": "SECRET REFERENCE"}]
    harness, generate = stage_harness(PLAN_SOLVE, rows, lambda messages: "1. Find the claim.")
    (output,) = harness.run([0], seed=0)
    assert output["mode"] == "plan_solve"
    plan_messages, solve_messages = output["trace"]["plan"]["messages"], output["trace"]["solve"]["messages"]
    assert "SECRET REFERENCE" not in json.dumps(plan_messages) and plan_messages[1]["content"].endswith("Provide a solution plan.")
    assert "Reference answer (for verification only): SECRET REFERENCE" in solve_messages[1]["content"]
    assert "boxed" not in json.dumps(solve_messages) and "math" not in json.dumps(plan_messages + solve_messages)
    assert _get_teacher_messages_from_harness(output, []) == solve_messages
    assert [(call[1], call[2], call[3]) for call in generate.calls] == [(0.6, 1.0, 4096), (0.6, 1.0, 5120)]


def test_pairs_skip_samples_without_a_final_harness_response():
    torch = pytest.importorskip("torch")
    from hero_baselines.ophsd_official.src.ophsd.batch_builder import build_ophsd_pairs

    encode = lambda messages: [7] * sum(len(m["content"]) for m in messages)
    outputs = [{"final_response": "x", "mode": "cold_start", "trace": {"cold_start": {"messages": [{"role": "user", "content": "abcd"}]}}},
               {"final_response": "", "mode": "cold_start", "trace": {}}]
    students, teachers, kept = build_ophsd_pairs([[{"role": "user", "content": "ab"}]] * 2, [[1, 2, 3], [4]], encode,
                                                 max_length=16, pad_token_id=0, harness_outputs=outputs)
    assert kept == [0]
    assert students[0]["loss_mask"].tolist() == [0, 0, 1, 1, 1] + [0] * 11
    assert teachers[0]["input_ids"][:7].tolist() == [7, 7, 7, 7, 1, 2, 3]


def test_chunked_divergence_equals_the_reference_loss_and_gradient():
    torch = pytest.importorskip("torch")
    from hero_baselines.ophsd import _chunk_divergence_sum
    from hero_baselines.ophsd_official.src.ophsd.losses import compute_reverse_kl_loss

    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(11, 4, generator=generator)
    student = torch.randn(7, 4, generator=generator, requires_grad=True)
    teacher = torch.randn(7, 4, generator=generator)
    chunked = sum(_chunk_divergence_sum(student[i:i + 3], teacher[i:i + 3], weight, "reverse_kl", 2)
                  for i in range(0, 7, 3)) / 7
    reference, tokens = compute_reverse_kl_loss(teacher @ weight.T, student @ weight.T, chunk_size=256)
    assert tokens == 7 and torch.allclose(chunked, reference, atol=1e-6)
    (grad_chunked,) = torch.autograd.grad(chunked, student)
    (grad_reference,) = torch.autograd.grad(compute_reverse_kl_loss(teacher @ weight.T, student @ weight.T)[0], student)
    assert torch.allclose(grad_chunked, grad_reference, atol=1e-6)
    p_s = torch.softmax(student @ weight.T, -1)
    direct = (p_s * (p_s.log() - torch.log_softmax(teacher @ weight.T, -1))).sum(-1).mean()
    assert torch.allclose(reference, direct, atol=1e-5)  # KL(student || teacher)


def test_update_averages_micro_batch_token_means(monkeypatch):
    torch = pytest.importorskip("torch")
    import hero_baselines.ophsd as ophsd

    parameter = torch.nn.Parameter(torch.zeros(()))
    values = {0: (2.0, 1), 1: (6.0, 3), 2: (5.0, 5)}  # row -> (divergence sum, response tokens)
    monkeypatch.setattr(ophsd, "sequence_divergence_sum",
                        lambda model, s, t, config: (parameter + values[s["row"]][0], values[s["row"]][1]))
    seqs = [{"row": row, "loss_mask": torch.ones(values[row][1])} for row in range(3)]
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    result = ophsd.opsd_update(None, optimizer, [parameter], seqs, seqs, gradient_clip=10.0,
                               config=OphsdConfig(), on_sample=lambda *args: None)
    assert result["loss"] == pytest.approx(((2 + 6) / 4 + 5 / 5) / 2)
    assert parameter.item() == pytest.approx(-((2 / 4) + 1 / 5) / 2)  # d/dparam of the mean of micro-batch means


def test_evaluation_schedule_matches_the_base_control():
    for stage in range(1, 26):
        seen = range(stage) if stage in (4, 8, 16, 25) else range(8) if stage in (12, 20) else [stage - 1]
        assert evaluated_task_indices(stage) == sorted(set(seen) | {stage - 1})
    assert [general_panel_kind(s) for s in (0, 1, 4, 5, 8, 16, 25)] == ["full", "probe", "full", "probe", "full", "full", "full"]
    assert sum(len(evaluated_task_indices(s)) for s in range(1, 26)) == 90


def load_runner():
    spec = importlib.util.spec_from_file_location("ophsd_run", ROOT / "packages/baselines/ophsd/run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_plan_covers_the_registered_panels_without_gold():
    runner = load_runner()
    plan, train_files, eval_files = runner.build_plan(
        ROOT / "experiments/data/trace-plus-17", ROOT / "experiments/data/general-test/v1",
        profile="full", stages=25, updates=32, batch_size=32, eval_limit=None, general_limit=None)
    assert len(plan["stages"]) == 25 and len(train_files) == 25
    assert plan["upstream"]["commit"].startswith("36cc494") and plan["training"]["weightDecay"] == 0.1
    assert "randperm per epoch" in plan["training"]["records"] and "set_seed" in plan["training"]["loraInit"]
    assert all(set(row) == {"id", "prompt"} for rows in eval_files.values() for row in rows)
    task_items = sum(len(e["ids"]) for s in plan["stages"] for e in s["evaluations"] if not e["group"].startswith("general:"))
    general_items = sum(len(e["ids"]) for s in plan["stages"] for e in s["evaluations"] if e["group"].startswith("general:"))
    assert task_items == 9000 and general_items == 4 * 1000 + 21 * 200


def test_generation_chunks_respect_the_token_budget_and_batch_limit():
    from hero_baselines.ophsd import generation_chunks

    short = generation_chunks([100] * 32, max_new_tokens=5120, batch_size=32, token_budget=450_000,
                              prefill_budget=65_536)
    assert [len(c) for c in short] == [32]
    lengths = [16000] * 32 + [200]
    long = generation_chunks(lengths, max_new_tokens=4096, batch_size=32, token_budget=450_000,
                             prefill_budget=65_536)
    widest = [max(lengths[i] for i in c) for c in long]
    assert all(len(c) * (w + 4096) <= 450_000 and (len(c) == 1 or len(c) * w <= 65_536) for c, w in zip(long, widest))
    assert max(len(c) for c in long) == 4 and sorted(i for c in long for i in c) == list(range(33))


def test_update_schedule_reshuffles_each_epoch_and_drops_the_incomplete_batch():
    torch = pytest.importorskip("torch")
    from hero_baselines.ophsd import update_schedule

    schedule = update_schedule(500, updates=32, batch_size=32, seed=43)
    assert len(schedule) == 32 and all(len(batch) == 32 for batch in schedule)
    generator = torch.Generator()
    generator.manual_seed(43)
    epochs = [torch.randperm(500, generator=generator).tolist() for _ in range(3)]
    assert schedule[:15] == [epochs[0][i:i + 32] for i in range(0, 480, 32)]  # 15 full batches, 20 dropped
    assert schedule[15:30] == [epochs[1][i:i + 32] for i in range(0, 480, 32)]
    assert schedule[30:] == [epochs[2][0:32], epochs[2][32:64]]
    assert len({i for batch in schedule[:15] for i in batch}) == 480
    assert update_schedule(500, updates=32, batch_size=32, seed=43) == schedule
    assert update_schedule(500, updates=32, batch_size=32, seed=44) != schedule
    with pytest.raises(ValueError):
        update_schedule(31, updates=1, batch_size=32, seed=0)


def test_execution_only_changes_keep_the_scientific_plan(tmp_path):
    runner = load_runner()
    plan, _, _ = runner.build_plan(
        ROOT / "experiments/data/trace-plus-17", ROOT / "experiments/data/general-test/v1",
        profile="full", stages=25, updates=32, batch_size=32, eval_limit=None, general_limit=None)
    rescheduled = json.loads(json.dumps(plan))
    rescheduled["ophsd"]["generation_batch_size"] = 16
    rescheduled["ophsd"]["generation_token_budget"] += 1
    rescheduled["files"] = {"run.py": "0" * 64}
    assert runner.semantic_plan_digest(rescheduled) == runner.semantic_plan_digest(plan)
    changed = json.loads(json.dumps(plan))
    changed["ophsd"]["loss_type"] = "forward_kl"
    assert runner.semantic_plan_digest(changed) != runner.semantic_plan_digest(plan)


def test_resumed_logs_drop_rows_from_rerun_updates(tmp_path):
    runner = load_runner()
    log = tmp_path / "training-log.jsonl"
    log.write_text("".join(json.dumps({"kind": "sample", "update": u}) + "\n" for u in (0, 1, 2, 3)))
    runner.truncate_log(log, 2)
    assert [json.loads(line)["update"] for line in log.read_text().splitlines()] == [0, 1]
