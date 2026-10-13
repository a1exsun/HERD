import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('base_run', Path(__file__).parents[1] / 'run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def test_resume_rejects_tampering_and_different_contract(tmp_path):
    (tmp_path / 'items').mkdir()
    item = {'id': 'example', 'prompt': 'Only ordinary input'}
    record = dict(id=item['id'], promptSha256=base.digest(item['prompt']), contractSha256='contract', prediction='answer')
    record['sha256'] = base.digest(record)
    path = tmp_path / 'items/0000.json'
    base.atomic_json(path, record)
    assert len(base.load_completed(tmp_path, [item], 'contract')) == 1
    with pytest.raises(ValueError):
        base.load_completed(tmp_path, [item], 'different')
    record['prediction'] = 'tampered'
    base.atomic_json(path, record)
    with pytest.raises(ValueError):
        base.load_completed(tmp_path, [item], 'contract')


def test_panel_shape_and_labels_stay_out_of_predictor(tmp_path):
    tasks = []
    for task_id in ['trace:C-STANCE', 'trace:FOMC']:
        files = {}
        for split, size in [('train', 32), ('validationGold', 16), ('testGold', 16)]:
            path = tmp_path / f'{task_id}-{split}.jsonl'
            path.write_text('\n'.join(json.dumps({'id': f'{task_id}/{split}/{i}', 'prompt': f'question {i}', 'answer': 'secret'}) for i in range(size)))
            files[split] = {'path': path.name, 'sha256': base.file_hash(path)}
        tasks.append({'id': task_id, 'metric': 'strict_exact_match', 'files': files})
    (tmp_path / 'manifest.json').write_text(json.dumps({'seed': 42, 'tasks': tasks}))
    panel, provenance = base.smoke_panel(tmp_path)
    assert len(panel) == 85
    assert all(len(ids) == 32 for ids in provenance['selectedTrainIds'].values())
    assert len([r for r in panel if r['group'].startswith('synthetic-G/')]) == 20
    assert not any('/train/' in r['id'] for r in panel)
    # Only prompt text crosses the model boundary; metadata/references never do.
    import inspect
    assert list(inspect.signature(base.Predictor.predict).parameters) == ['self', 'prompt']
    path = tmp_path / tasks[0]['files']['train']['path']
    path.write_text('corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        base.smoke_panel(tmp_path)


def test_tree_hash_tracks_weights_and_names(tmp_path):
    (tmp_path / 'model').write_bytes(b'weights')
    original = base.tree_hash(tmp_path)
    (tmp_path / 'model').rename(tmp_path / 'renamed')
    assert base.tree_hash(tmp_path) != original


def test_full_panel_includes_all_available_test_rows_and_no_training(tmp_path):
    tasks = []
    for i in range(25):
        path = tmp_path / f'{i}.jsonl'
        path.write_text(json.dumps({'id': f'test:{i}', 'prompt': 'question', 'answer': 'answer'}) + '\n')
        tasks.append({'position': i + 1, 'id': f'task:{i}', 'metric': 'exact_match', 'files': {'testGold': {'path': path.name, 'sha256': base.file_hash(path), 'records': 1}}})
    (tmp_path / 'manifest.json').write_text(json.dumps({'seed': 42, 'tasks': tasks}))
    panel, provenance = base.full_panel(tmp_path)
    assert len(panel) == 25
    assert len(provenance['countDeviations']) == 25
    assert all(not r['id'].startswith('synthetic:') for r in panel)
    scores = {f'task:{i}': {'score': i / 25} for i in range(25)}
    metrics = base.constant_metrics(scores, provenance['taskOrder'])
    assert len(metrics['sparseMatrix']) == 90
    assert metrics['forgetting'] == metrics['signedBWT'] == 0
    assert len([r for r in metrics['sparseMatrix'] if r['stage'] == 25]) == 25
    assert metrics['generalChange'] is None
    general = tmp_path / 'general'; general.mkdir()
    make_general_fixture(general)
    combined, joined = base.full_panel(tmp_path, general)
    assert combined[:25] == panel and len(combined) == 1025
    assert joined['generalTestStatus'].startswith('included')


def test_resume_preserves_cache_after_wrapper_host_and_scorer_changes(tmp_path):
    import hashlib
    old_source = 'class Predictor:\n    def predict(self, prompt): return prompt\nclass InputLimitExceeded(ValueError): pass\n'
    new_source = old_source + '\ndef updated_monitor(): return 1\n'
    old = dict(model='pinned', panelSha256='panel', device='mps', runnerSha256=hashlib.sha256(old_source.encode()).hexdigest(), scorerSha256='old', host='old-host')
    new = dict(old, runnerSha256=hashlib.sha256(new_source.encode()).hexdigest(), scorerSha256='new', host='new-host')
    original_hash = base.prepare_contract(tmp_path, old, old_source)
    original_manifest = (tmp_path / 'manifest.json').read_bytes()
    (tmp_path / 'scores.json').write_text('{"oldScore": 1}')
    (tmp_path / 'items').mkdir()
    item = dict(id='item', prompt='prompt')
    record = dict(id='item', prediction='cached result', promptSha256=base.digest('prompt'), contractSha256=original_hash)
    record['sha256'] = base.digest(record)
    base.atomic_json(tmp_path / 'items/0000.json', record)
    original_item = (tmp_path / 'items/0000.json').read_bytes()
    assert base.prepare_contract(tmp_path, new, new_source) == original_hash
    assert (tmp_path / 'manifest.json').read_bytes() == original_manifest
    assert (tmp_path / 'items/0000.json').read_bytes() == original_item
    assert base.load_completed(tmp_path, [item], original_hash)[0]['prediction'] == 'cached result'
    assert len(list((tmp_path / 'executions').glob('*/previous-scores.json'))) == 1


@pytest.mark.parametrize('field', ['model', 'checkpointSha256', 'panelSha256', 'recordIds', 'chatTemplateSha256', 'device', 'precision', 'reasoning', 'maxNewTokens', 'libraries'])
def test_resume_rejects_changed_prediction_contract(field):
    old = {field: 'old'}
    with pytest.raises(base.ResumeMismatch, match=field):
        base.compatible_resume(old, {field: 'new'})


def test_resume_rejects_changed_model_code_and_unverified_source():
    import hashlib
    old = 'class Predictor:\n    def predict(self, prompt): return prompt\nclass InputLimitExceeded(ValueError): pass\n'
    new = old.replace('return prompt', 'return "different"')
    a = {'runnerSha256': hashlib.sha256(old.encode()).hexdigest()}
    b = {'runnerSha256': hashlib.sha256(new.encode()).hexdigest()}
    with pytest.raises(base.ResumeMismatch, match='source snapshot'):
        base.compatible_resume(a, b)
    with pytest.raises(base.ResumeMismatch, match='Inference implementation'):
        base.compatible_resume(a, b, old, new)


def make_general_fixture(tmp_path):
    rows, probes, categories = [], [], []
    for c in base.GENERAL_CATEGORIES:
        categories.append({'category': c, 'metric': 'exact_match'})
        for i in range(200):
            row = {'id': f'{c}:{i}', 'category': c, 'metric': 'exact_match', 'prompt': 'prompt', 'answer': 'answer'}
            rows.append(row)
            if i < 40:
                probes.append(row['id'])
    (tmp_path / 'hidden-scorer').mkdir()
    test = tmp_path / 'hidden-scorer/general-test.jsonl'
    test.write_text('\n'.join(json.dumps(r) for r in rows))
    probe = tmp_path / 'hidden-scorer/probe-ids.json'
    probe.write_text(json.dumps(probes))
    manifest = {'version': 'hero-general-test-v1', 'seed': 42, 'categories': categories,
                'files': [{'path': str(p.relative_to(tmp_path)), 'sha256': base.file_hash(p)} for p in (test, probe)]}
    base.atomic_json(tmp_path / 'manifest.json', manifest)
    return manifest


def test_general_panel_hash_counts_probe_and_constant_metrics(tmp_path):
    make_general_fixture(tmp_path)
    panel, provenance = base.general_panel(tmp_path)
    assert len(panel) == 1000 and len(provenance['generalProbeIds']) == 200
    scores = {f'general:{c}': {'sampleSize': 200, 'score': .2, 'itemScores': [{'id': f'{c}:{i}', 'score': float(i < 40)} for i in range(200)]} for c in base.GENERAL_CATEGORIES}
    result = base.general_metrics(scores, provenance)
    assert result['generalMacro'] == .2 and result['generalProbeMacro'] == 1
    assert result['generalChange'] == 0 and len(result['generalByStage']) == 26
    assert result['generalByStage'][0]['sampleSize'] == 1000
    assert result['generalByStage'][1]['sampleSize'] == 200
    (tmp_path / 'hidden-scorer/probe-ids.json').write_text('[]')
    with pytest.raises(ValueError, match='hash mismatch'):
        base.general_panel(tmp_path)


def test_general_probe_must_belong_to_test(tmp_path):
    manifest = make_general_fixture(tmp_path)
    path = tmp_path / 'hidden-scorer/probe-ids.json'
    probes = json.loads(path.read_text()); probes[0] = 'not-in-test'
    path.write_text(json.dumps(probes))
    manifest['files'][1]['sha256'] = base.file_hash(path)
    base.atomic_json(tmp_path / 'manifest.json', manifest)
    with pytest.raises(ValueError, match='not a subset'):
        base.general_panel(tmp_path)


def test_resume_fingerprints_the_engine_the_run_uses():
    import hashlib
    old = ('class Predictor:\n    def predict(self, prompt): return prompt\n'
           'class VllmPredictor:\n    def predict_many(self, prompts): return prompts\n'
           'class InputLimitExceeded(ValueError): pass\n')
    changed_transformers = old.replace('return prompt\n', 'return "different"\n')
    changed_vllm = old.replace('return prompts', 'return []')
    digest = lambda source: hashlib.sha256(source.encode()).hexdigest()
    vllm = {'engine': 'vllm==0.31.0'}
    # A vLLM run's cache survives edits to the transformers Predictor, not to VllmPredictor.
    base.compatible_resume({**vllm, 'runnerSha256': digest(old)}, {**vllm, 'runnerSha256': digest(changed_transformers)},
                           old, changed_transformers)
    with pytest.raises(base.ResumeMismatch, match='Inference implementation'):
        base.compatible_resume({**vllm, 'runnerSha256': digest(old)}, {**vllm, 'runnerSha256': digest(changed_vllm)},
                               old, changed_vllm)


def test_vllm_engine_seals_batches_failures_and_pauses(tmp_path, monkeypatch):
    import types
    import vllm_decode
    panel = [{'id': f'r{i}', 'group': 'g', 'metric': 'exact_match', 'prompt': 'p' * (i + 1), 'answer': 'P' * (i + 1),
              'references': ['P' * (i + 1)]} for i in range(5)]
    contract = {'profile': 'full', 'engine': vllm_decode.ENGINE}
    monkeypatch.setattr(base, 'build_contract', lambda args: (contract, panel, {'taskOrder': ['g']}, tmp_path / 'ckpt', 'w'))
    monkeypatch.setattr(base, 'tree_hash', lambda path: 'w')
    monkeypatch.setattr(vllm_decode, 'CALL_ITEMS', 2)
    calls = []

    class FakeVllm:
        def __init__(self, checkpoint, max_new_tokens):
            assert max_new_tokens == 5120

        def predict_many(self, prompts):
            calls.append(len(prompts))
            return [base.InputLimitExceeded(16385) if prompt == 'ppp' else
                    {'prediction': prompt.upper(), 'outputTokenIds': [1, 2], 'promptTokens': 3, 'outputTokens': 2,
                     'seconds': 0.25, 'stopReason': 'eos', 'tokensPerSecond': 8.0, 'processPeakRssBytes': 1,
                     'deviceAllocatedBytes': None, 'engine': vllm_decode.ENGINE, 'batchSize': len(prompts)}
                    for prompt in prompts]

    monkeypatch.setattr(base, 'VllmPredictor', FakeVllm)
    args = types.SimpleNamespace(profile='full', output=tmp_path / 'run', device='cuda', max_new_tokens=5120,
                                 verify_resume=False, stop_after=3)
    base.run(args)
    assert calls == [2, 1] and json.loads((args.output / 'status.json').read_text())['status'] == 'paused'
    failed = json.loads((args.output / 'items/0002.json').read_text())
    assert failed['stopReason'] == 'input_limit' and failed['itemStatus'] == 'failed' and failed['seconds'] == 0.0
    args.stop_after = None
    base.run(args)  # resumes with the two remaining items in one call
    assert calls == [2, 1, 2]
    status = json.loads((args.output / 'status.json').read_text())
    assert status['status'] == 'complete_with_item_failures' and status['completed'] == 5
