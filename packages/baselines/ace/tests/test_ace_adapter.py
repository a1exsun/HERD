import json
import pickle
from types import SimpleNamespace as NS

import httpx
import openai
import pytest

from ace_adapter import UPSTREAM, TraceDataProcessor, TraceTarget, ace_config, activate_upstream, to_ace_sample

activate_upstream()


def response(content):
    return NS(choices=[NS(message=NS(content=content))], usage=NS(prompt_tokens=4, completion_tokens=1))


def rejected(message="This model's maximum context length is 20480 tokens."):
    request = httpx.Request("POST", "http://127.0.0.1:8000/v1/chat/completions")
    return openai.BadRequestError(message, response=httpx.Response(400, request=request), body=None)


RECORD = {"id": "x", "prompt": "Choose A or B", "answer": "A", "references": ["A"], "metadata": {"source": "trace"}}


def test_upstream_is_the_complete_repository():
    assert "Apache License" in (UPSTREAM / "LICENSE.txt").read_text()
    for path in ("ace/ace.py", "ace/ace_batch.py", "eval/finance/run.py", "EXTENDING_ACE.md"):
        assert (UPSTREAM / path).is_file()


def test_local_provider_sends_upstream_requests_unchanged():
    from llm import timed_llm_call

    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return response("A")

    client = NS(chat=NS(completions=NS(create=create)))
    assert timed_llm_call(client, "local", "Qwen/Qwen3.5-4B", "q", "generator", "test_local")[0] == "A"
    assert requests == [{"model": "Qwen/Qwen3.5-4B", "messages": [{"role": "user", "content": "q"}],
                         "temperature": 0.0, "max_tokens": 4096}]


def test_local_provider_reads_the_server_url(monkeypatch):
    from utils import initialize_clients

    monkeypatch.setenv("ACE_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
    clients = initialize_clients("local")
    assert all(str(client.base_url).startswith("http://127.0.0.1:8000/v1") for client in clients)


def test_target_is_the_reference_string_everywhere_upstream_uses_it():
    target = to_ace_sample(RECORD)["target"]
    assert target == "A" and isinstance(target, str)
    assert json.loads(json.dumps({"target": target})) == {"target": "A"}
    assert "Ground truth: {}".format(target) == "Ground truth: A"
    restored = pickle.loads(pickle.dumps(target))
    assert restored == "A" and restored.record == RECORD


def test_correctness_is_full_credit_on_the_canonical_item_score():
    target = TraceTarget(RECORD)
    processor = TraceDataProcessor("leading_choice_accuracy")
    assert processor.answer_is_correct("A extra", target)
    assert not processor.answer_is_correct("B", target)
    assert processor.evaluate_accuracy(["A", "B"], [target, target]) == 0.5
    summary = TraceTarget({"id": "y", "prompt": "Text", "answer": "the cat sat", "metadata": {"source": "trace"}})
    assert not TraceDataProcessor("rouge_l").answer_is_correct("the cat", summary)


def scripted_ace(monkeypatch, curator_failures=(), error=rejected):
    """Upstream ACE whose three roles answer from scripts; the curator raises 400 on the listed steps."""
    monkeypatch.setenv("ACE_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1")
    from ace_adapter import ace_system

    ace = ace_system()
    generator = NS(chat=NS(completions=NS(create=lambda **_: response(
        json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": "B"})))))
    reflector = NS(chat=NS(completions=NS(create=lambda **_: response(json.dumps(
        {"reasoning": "r", "error_identification": "e", "root_cause_analysis": "c", "correct_approach": "a",
         "key_insight": "k", "bullet_tags": []})))))
    calls = []

    def curate(**_):
        calls.append(1)
        if len(calls) in curator_failures:
            raise error()
        return response(json.dumps({"reasoning": "r", "operations": [
            {"type": "ADD", "section": "strategies_and_insights", "content": f"insight {len(calls)}"}]}))

    ace.generator.api_client, ace.reflector.api_client = generator, reflector
    ace.curator.api_client = NS(chat=NS(completions=NS(create=curate)))
    return ace


def train_three(ace, tmp_path, **overrides):
    samples = [to_ace_sample({**RECORD, "id": str(i)}) for i in range(3)]
    validation = overrides.pop("val_samples", [])
    config = {**ace_config(save_dir=str(tmp_path), task_name="t"), **overrides}
    return ace.run(mode="offline", train_samples=samples, val_samples=validation,
                   data_processor=TraceDataProcessor("leading_choice_accuracy"), config=config)


def test_offline_trainer_adds_one_curated_bullet_per_sample(monkeypatch, tmp_path):
    ace = scripted_ace(monkeypatch)
    train_three(ace, tmp_path)
    assert [line.split(" :: ")[1] for line in ace.playbook.splitlines() if " :: " in line] == [
        "insight 1", "insight 2", "insight 3"]
    assert ace.next_global_id == 4


def test_rejected_call_skips_its_sample_and_keeps_the_playbook(monkeypatch, tmp_path):
    ace = scripted_ace(monkeypatch, curator_failures=(2,))
    train_three(ace, tmp_path)
    assert [line.split(" :: ")[1] for line in ace.playbook.splitlines() if " :: " in line] == [
        "insight 1", "insight 3"]
    steps = json.loads(next(tmp_path.rglob("pre_train_post_train_results.json")).read_text())
    assert ["api_error" in step for step in steps] == [False, True, False]


def test_without_the_setting_a_rejected_call_stops_training_as_upstream(monkeypatch, tmp_path):
    ace = scripted_ace(monkeypatch, curator_failures=(2,))
    with pytest.raises(openai.BadRequestError):
        train_three(ace, tmp_path, continue_on_llm_error=False)


def test_any_other_rejection_stops_training_as_upstream(monkeypatch, tmp_path):
    ace = scripted_ace(monkeypatch, curator_failures=(2,), error=lambda: rejected("Invalid request"))
    with pytest.raises(openai.BadRequestError):
        train_three(ace, tmp_path)


def test_settings_are_upstream_defaults_except_the_protocol_ones():
    config = ace_config(save_dir="s", task_name="t")
    assert (config["eval_steps"], config["playbook_token_budget"]) == (100, 80000)
    assert (config["num_epochs"], config["max_num_rounds"], config["curator_frequency"]) == (1, 3, 1)
    assert (config["save_steps"], config["test_workers"]) == (50, 20)


def test_validation_selects_upstreams_best_playbook(monkeypatch, tmp_path):
    ace = scripted_ace(monkeypatch)
    answers = iter(["A", "B", "B", "B"])  # validation at step 1 is right, at steps 2 and 3 wrong
    generate = ace.generator.api_client.chat.completions.create

    def validation_aware(**kwargs):
        if "Choose" in kwargs["messages"][0]["content"] and "validation" in kwargs["messages"][0]["content"]:
            return response(json.dumps({"reasoning": "r", "bullet_ids": [], "final_answer": next(answers)}))
        return generate(**kwargs)

    ace.generator.api_client = NS(chat=NS(completions=NS(create=validation_aware)))
    validation = [to_ace_sample({**RECORD, "id": "v", "prompt": "Choose A or B (validation)"})]
    train_three(ace, tmp_path, eval_steps=1, val_samples=validation)
    assert ace.best_playbook.count(" :: ") == 1 and ace.playbook.count(" :: ") == 3


def test_validation_with_every_call_rejected_scores_zero_instead_of_crashing(monkeypatch, tmp_path):
    ace = scripted_ace(monkeypatch)
    generate = ace.generator.api_client.chat.completions.create

    def reject_validation(**kwargs):
        if "(validation)" in kwargs["messages"][0]["content"]:
            raise rejected()
        return generate(**kwargs)

    ace.generator.api_client = NS(chat=NS(completions=NS(create=reject_validation)))
    validation = [to_ace_sample({**RECORD, "id": "v", "prompt": "Choose A or B (validation)"})]
    train_three(ace, tmp_path, eval_steps=1, val_samples=validation)
    assert ace.best_playbook.count(" :: ") == 0 and ace.playbook.count(" :: ") == 3


def test_selected_step_follows_upstreams_strict_improvement_rule():
    from run import selected_step

    assert selected_step([]) == 0
    assert selected_step([{"step": 100, "accuracy": 0.0}]) == 0
    assert selected_step([{"step": 100, "accuracy": 0.4}, {"step": 200, "accuracy": 0.6},
                          {"step": 300, "accuracy": 0.6}, {"step": 400, "accuracy": 0.5}]) == 200


def test_recovery_does_not_train_completed_samples_again(monkeypatch, tmp_path):
    ace = scripted_ace(monkeypatch)
    ace.resume_save_path = str(tmp_path / 'offline')
    checkpoint = tmp_path / 'resume.json'
    original = ace._train_single_sample_or_skip
    called = []

    def interrupted(**kwargs):
        called.append(kwargs['step'])
        if kwargs['step'] == 2:
            raise RuntimeError('interrupted')
        return original(**kwargs)

    ace._train_single_sample_or_skip = interrupted
    with pytest.raises(RuntimeError, match='interrupted'):
        train_three(ace, tmp_path, resume_path=str(checkpoint))
    assert json.loads(checkpoint.read_text())['step'] == 1
    resumed = scripted_ace(monkeypatch)
    resumed.resume_save_path = ace.resume_save_path
    resume_original = resumed._train_single_sample_or_skip
    resumed_steps = []

    def tracked(**kwargs):
        resumed_steps.append(kwargs['step'])
        return resume_original(**kwargs)

    resumed._train_single_sample_or_skip = tracked
    train_three(resumed, tmp_path, resume_path=str(checkpoint))
    assert resumed_steps == [2, 3]
    assert resumed.next_global_id == 4
    assert resumed.playbook.count(' :: ') == 3
    assert json.loads(checkpoint.read_text())['step'] == 3


def test_recovery_during_validation_keeps_the_completed_update(monkeypatch, tmp_path):
    import ace.ace as upstream

    ace = scripted_ace(monkeypatch)
    ace.resume_save_path = str(tmp_path / 'offline')
    checkpoint = tmp_path / 'resume.json'
    validation = [to_ace_sample(RECORD)]
    evaluate = upstream.evaluate_test_set

    def fail_validation(*args, **kwargs):
        raise RuntimeError('validation interrupted')

    monkeypatch.setattr(upstream, 'evaluate_test_set', fail_validation)
    with pytest.raises(RuntimeError, match='validation interrupted'):
        train_three(ace, tmp_path, resume_path=str(checkpoint), eval_steps=1, save_steps=1, val_samples=validation)
    assert json.loads(checkpoint.read_text())['pendingValidation'] is True
    monkeypatch.setattr(upstream, 'evaluate_test_set', evaluate)
    resumed = scripted_ace(monkeypatch)
    resumed.resume_save_path = ace.resume_save_path
    original = resumed._train_single_sample_or_skip
    steps = []

    def tracked(**kwargs):
        steps.append(kwargs['step'])
        return original(**kwargs)

    resumed._train_single_sample_or_skip = tracked
    train_three(resumed, tmp_path, resume_path=str(checkpoint), eval_steps=1, save_steps=1, val_samples=validation)
    assert steps == [2, 3]
    assert resumed.playbook.count(' :: ') == 3
    saved = json.loads(checkpoint.read_text())
    assert [row['step'] for row in saved['results']] == [1, 2, 3]
    assert saved['pendingValidation'] is False


def test_retraining_is_blocked_after_stage_three(tmp_path):
    from run import atomic_json, train

    atomic_json(tmp_path / 'training-progress.json', {'maxStartedStage': 4})
    run = NS(output=tmp_path, stage=lambda n: {'stage': n})
    with pytest.raises(ValueError, match='Retraining forbidden'):
        train(run, 3)
    with pytest.raises(ValueError, match='checkpoint is missing'):
        train(run, 4)
