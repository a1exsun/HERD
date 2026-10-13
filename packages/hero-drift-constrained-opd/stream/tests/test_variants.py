"""CPU gradient and execution checks for the paper's ablations."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hero
import run


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = Decoder()
        self.lm_head = torch.nn.Linear(8, 32, bias=False)


def model_and_loss():
    torch.manual_seed(1)
    model = hero.initialize_adapter(Model(), LoraConfig(r=2, target_modules=['proj']), seed=42, stage=1)
    head = hero.backbone_and_head(model)[1].weight.detach().float()
    return model, hero.Divergence(head, token_budget=5, chunk=1)


def test_sft_loss_and_gradients_match_full_vocabulary_oracle():
    model, divergence = model_and_loss()
    pairs = [hero.Pair([1, 2], [3, 4, 5]), hero.Pair([6], [7])]
    values = divergence.cross_entropy(model, pairs, 1.0)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad()
    losses = []
    for pair in pairs:
        hidden = hero.completion_hidden(model, [(pair.student, pair.completion)])
        losses.append(F.cross_entropy(hidden @ divergence.head32.t(), torch.tensor(pair.completion)))
    expected = torch.stack(losses)
    expected.mean().backward()
    assert torch.allclose(torch.tensor(values), expected.detach(), atol=1e-6)
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(grads[n], p.grad, atol=1e-6), n


@pytest.mark.parametrize('objective,checkpoint,base', [
    ('distill', True, True), ('distill', False, True), ('distill', True, False),
    ('distill', False, False), ('sft', True, True)])
def test_anchor_switches_keep_diagnostics_but_disable_penalties(objective, checkpoint, base, monkeypatch):
    model, divergence = model_and_loss()
    historical = [hero.Pair([1, 2], [3, 4], group='task')]
    general = [hero.Pair([1, 2], [3, 4], group='source')]
    hero.anchor_targets(model, historical + general)
    teacher_calls = []
    attach = divergence.attach_teacher
    monkeypatch.setattr(divergence, 'attach_teacher', lambda m, p: (teacher_calls.append(1), attach(m, p)))
    logs = []
    config = hero.StageConfig(updates=3, batch_size=1, learning_rate=0.01,
                             checkpoint_budget=1e-9 if checkpoint else None, general_budget=1e-9 if base else None,
                             objective=objective)
    result = hero.train_stage(model, [{'studentIds': [1, 2], 'teacherIds': [5, 6], 'goldIds': [3, 4]}],
                              divergence=divergence, historical=historical, general=general, config=config, log=logs.append)
    assert result['objective'] == objective
    assert len(teacher_calls) == (3 if objective == 'distill' else 0)
    for name, active in [('checkpoint', checkpoint), ('general', base)]:
        assert max(row[f'{name}Drift'] for row in logs) > 0
        multipliers = [row[f'{name}Multiplier'] for row in logs]
        assert max(multipliers) > 0 if active else multipliers == [0.0] * 3


def test_no_harness_keeps_evidence_without_any_model_calls():
    stream = run.Stream.__new__(run.Stream)
    stream.plan = {'seed': 42, 'variant': 'no-harness',
                   'harness': {'evidenceBudgetTokens': 8192, 'proposer': {}}}
    stream.tokenizer = None
    stream.tokens = lambda text: text.split()
    old = hero.Harness(({'id': 'old', 'promptPrefix': '', 'instructions': 'old skill'},))
    harness, evidence, guard = stream.evolve(old, {'stage': 1, 'taskId': 'task'},
                                            [{'id': 'x', 'prompt': 'p', 'answer': 'a'}], None)
    assert not harness.skills and evidence[0]['id'] == 'x'
    assert guard['enabled'] is False and guard['gain'] is None


def test_uniform_memory_is_deterministic_and_obeys_token_budget():
    records = [{'id': str(i), 'tokens': i % 5 + 1} for i in range(30)]
    selected = hero.uniform_budget(records, 24, 42)
    assert selected == hero.uniform_budget(list(reversed(records)), 24, 42)
    assert sum(r['tokens'] for r in selected) <= 24
    assert selected != hero.uniform_budget(records, 24, 43)


def test_output_binding_rejects_cross_variant_or_seed_resume(tmp_path):
    plan = tmp_path / 'plan.json'
    plan.write_text('{"variant":"full","seed":42}')
    output = tmp_path / 'output'
    run.bind_output(output, plan, 'stream')
    run.bind_output(output, plan, 'stream')
    plan.write_text('{"variant":"sft","seed":43}')
    with pytest.raises(ValueError, match='different bundle'):
        run.bind_output(output, plan, 'stream')


@pytest.mark.parametrize('variant', run.VARIANTS)
def test_cli_variant_selects_the_training_mechanism(variant, monkeypatch, tmp_path):
    captured = []
    monkeypatch.setattr(run, 'prepare', lambda args: captured.append(args))
    monkeypatch.setattr(sys, 'argv', ['run.py', 'prepare', '--output', str(tmp_path), '--variant', variant,
                                     '--seed', '43', '--init-seed', '99'])
    run.main()
    args = captured[0]
    config = run.training_config(args)
    assert args.seed == 43 and args.init_seed == 99
    assert config.objective == ('sft' if variant == 'sft' else 'distill')
    assert (config.checkpoint_budget is not None) == (variant not in ('no-checkpoint-anchor', 'no-anchors'))
    assert (config.general_budget is not None) == (variant not in ('no-base-anchor', 'no-anchors'))
    if config.checkpoint_budget is not None:
        assert config.checkpoint_budget == {'drift-budget-tight': 0.02, 'drift-budget-loose': 0.1}.get(variant, 0.05)


def test_prepare_serializes_variant_and_seeds_without_test_gold(tmp_path, monkeypatch):
    dataset = tmp_path / 'experiments/data/trace-plus-17'
    records = [{'id': 'x', 'prompt': 'p', 'answer': 'SECRET'}]
    for name in ('train', 'testGold'):
        run.write_jsonl(dataset / f'{name}.jsonl', records)
    run.write_json(dataset / 'manifest.json', {'tasks': [{'id': 'task', 'position': 1, 'metric': 'exact_match',
        'files': {name: {'path': f'{name}.jsonl'} for name in ('train', 'testGold')}}]})
    general = tmp_path / 'experiments/data/general-test/v1'
    run.write_jsonl(general / 'hidden-scorer/general-test.jsonl', [
        {'id': 'g', 'prompt': 'gp', 'answer': 'GENERAL_SECRET', 'category': 'knowledge'}])
    run.write_json(general / 'hidden-scorer/probe-ids.json', ['g'])
    monkeypatch.setattr(run, 'ASSETS', ())
    monkeypatch.setattr(run, 'general_probes', lambda *args: [])
    bundle = tmp_path / 'bundle'
    monkeypatch.setattr(sys, 'argv', ['run.py', 'prepare', '--repository', str(tmp_path), '--output', str(bundle),
                                    '--variant', 'sft', '--seed', '43', '--init-seed', '99'])
    run.main()
    plan = json.loads((bundle / 'plan.json').read_text())
    assert plan['variant'] == 'sft' and plan['training']['objective'] == 'sft'
    assert plan['seed'] == 43 and plan['initializationSeed'] == 99
    assert not (bundle / 'inputs/validation').exists()
    assert not (bundle / 'inputs/general-dev.jsonl').exists()
    assert 'answer' not in run.read_jsonl(bundle / 'inputs/test/01.jsonl')[0]
    assert 'answer' not in run.read_jsonl(bundle / 'inputs/general-test.jsonl')[0]
    assert all(run.file_hash(bundle / name) == digest for name, digest in plan['files'].items())


def test_harness_inference_replays_only_and_preserves_source(tmp_path, monkeypatch):
    import peft

    source, bundle, output = (tmp_path / n for n in ('source', 'bundle', 'output'))
    plan = {'variant': 'full', 'evaluation': {'split': 'test'}, 'tasks': [{'stage': 1}, {'stage': 2}]}
    run.write_json(bundle / 'plan.json', plan)
    run.bind_output(source, bundle / 'plan.json', 'stream')
    harness = hero.Harness(({'id': 't', 'promptPrefix': 'Solve', 'instructions': 'one letter'},))
    for stage in (1, 2):
        directory = source / f'stage-{stage:02d}'
        adapter = directory / 'adapter'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_config.json').write_text('{}')
        (adapter / 'adapter_model.safetensors').write_bytes(bytes([stage]))
        run.write_json(directory / 'stage.json', {'harness': harness.serialize(), 'adapterSha256': run.adapter_sha256(adapter)})
    before = {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}
    restored = []
    def load(model, adapter, **kwargs):
        restored.append(adapter.parent.name)
        return SimpleNamespace(merge_and_unload=lambda: model)
    monkeypatch.setattr(peft.PeftModel, 'from_pretrained', load)
    stream = run.Stream.__new__(run.Stream)
    stream.plan, stream.bundle, stream.output, stream.model = plan, bundle, output, object()
    stream.say = lambda **kwargs: None
    def evaluate(model, stage, *, system, full_panel):
        assert stage == 2 and full_panel
        assert system('Solve x') == hero.HARNESS_SYSTEM.format(instructions='one letter')
        return {'stage': stage, 'testItems': 2}
    stream.evaluate_test = evaluate
    stream.evaluate_harness(source)
    assert restored == ['stage-01', 'stage-02']
    assert json.loads((output / 'evaluation.json').read_text())['variant'] == 'hero-with-harness'
    assert before == {p: p.read_bytes() for p in source.rglob('*') if p.is_file()}


def test_final_only_score_uses_original_diagonal(tmp_path, monkeypatch):
    # Test source-panel wiring and metrics; the unchanged scorers have their own suite.
    def score_predictions(metric, predictions, gold):
        predicted = {r['id']: r['prediction'] for r in predictions}
        assert set(predicted) == {r['id'] for r in gold}
        return {'score': sum(predicted[r['id']] == r['answer'] for r in gold) / len(gold)}
    monkeypatch.setitem(sys.modules, 'scoring', SimpleNamespace(score_predictions=score_predictions))
    dataset = tmp_path / 'experiments/data/trace-plus-17'
    tasks = []
    for stage, task in enumerate(('a', 'b'), 1):
        run.write_jsonl(dataset / f'{task}.jsonl', [{'id': task, 'prompt': 'p', 'answer': 'yes'}])
        tasks.append({'id': task, 'metric': 'exact_match', 'files': {'testGold': {'path': f'{task}.jsonl'}}})
    run.write_json(dataset / 'manifest.json', {'tasks': tasks})
    run.write_jsonl(tmp_path / 'experiments/data/general-test/v1/hidden-scorer/general-test.jsonl', [])
    original, inferred = tmp_path / 'original', tmp_path / 'inferred'
    for stage, task in enumerate(('a', 'b'), 1):
        run.write_jsonl(original / f'stage-{stage:02d}/test-predictions.jsonl',
                        [{'id': task, 'group': task, 'prediction': 'yes'}])
    run.write_jsonl(inferred / 'stage-02/test-predictions.jsonl', [
        {'id': 'a', 'group': 'a', 'prediction': 'no'}, {'id': 'b', 'group': 'b', 'prediction': 'yes'}])
    destination = tmp_path / 'score.json'
    run.score(SimpleNamespace(repository=tmp_path, run=inferred, output=destination, diagonal_run=original))
    metrics = json.loads(destination.read_text())['metrics']
    assert metrics['acquisitionMeanDiagonal'] == 1
    assert metrics['seenMean'] == 0.5 and metrics['forgetting'] == 1 and metrics['signedBWT'] == -1
    with pytest.raises(ValueError, match='Missing diagonal'):
        run.score(SimpleNamespace(repository=tmp_path, run=inferred, output=destination, diagonal_run=None))
