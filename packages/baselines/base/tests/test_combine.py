import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import combine


def test_combination_preserves_task_results_and_requires_predictor_identity(tmp_path, monkeypatch):
    tasks, general = tmp_path / 'tasks', tmp_path / 'general'
    source = 'class Predictor: pass\nclass InputLimitExceeded(ValueError): pass\n'
    for path, profile in ((tasks, 'full'), (general, 'general')):
        path.mkdir()
        (path / 'runner-source.py').write_text(source)
        contract = {key: 'same' for key in combine.PREDICTOR_FIELDS}
        contract.update(profile=profile, runnerSha256=combine.base.file_hash(path / 'runner-source.py'))
        (path / 'manifest.json').write_text(json.dumps(contract))
        (path / 'status.json').write_text('{"status":"complete"}')
    original = (tasks / 'manifest.json').read_bytes()
    monkeypatch.setattr(combine.base, 'ROOT', tmp_path)
    monkeypatch.setattr(combine, 'length_stops', lambda run: [])
    monkeypatch.setattr(combine, 'verify', lambda p: {'metrics': {'AA': .4, 'generalChange': None} if p == tasks else {'generalChange': 0., 'generalMacro': .6}, 'verifiedItems': 2477 if p == tasks else 1000, 'itemFailures': 0, 'scores': {}, 'inferenceSeconds': 1})
    result = combine.combine(tasks, general)
    assert result['predictions'] == 3477
    assert result['metrics']['generalChange'] == 0 and result['metrics']['AA'] == .4
    assert result['inferenceSeconds'] == {'tasks': 1, 'general': 1}
    assert (tasks / 'manifest.json').read_bytes() == original
    contract = json.loads((general / 'manifest.json').read_text()); contract['precision'] = 'different'
    (general / 'manifest.json').write_text(json.dumps(contract))
    with pytest.raises(ValueError, match='precision'):
        combine.combine(tasks, general)


def test_repetition_period_detects_only_exact_closing_loops():
    assert combine.repetition_period([9, 1, 2, 3, 1, 2, 3, 1, 2, 3]) == 3
    assert combine.repetition_period([1, 2, 3, 4, 5, 6, 7]) == 0
