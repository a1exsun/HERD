import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import verify


def fixture(tmp_path):
    (tmp_path / 'items').mkdir()
    panel = [{'id': 'a', 'prompt': 'prompt', 'group': 'task'}]
    row = dict(id='a', group='task', contractSha256='contract', promptSha256=verify.base.digest('prompt'), prediction='a', promptTokens=5, outputTokenIds=[1, 2], outputTokens=2, seconds=1, stopReason='eos')
    return panel, row


def seal(tmp_path, row):
    row['sha256'] = verify.base.digest(row)
    verify.base.atomic_json(tmp_path / 'items/0000.json', row)


def test_verifier_rejects_incomplete_and_out_of_contract_files(tmp_path):
    panel, row = fixture(tmp_path)
    with pytest.raises(ValueError, match='Incomplete'):
        verify.verify_items(tmp_path, panel, 'contract', True, 5120)
    assert verify.verify_items(tmp_path, panel, 'contract', False, 5120) == []
    seal(tmp_path, row)
    assert len(verify.verify_items(tmp_path, panel, 'contract', True, 5120)) == 1
    (tmp_path / 'items/9999.json').write_text('{}')
    with pytest.raises(ValueError, match='Unexpected'):
        verify.verify_items(tmp_path, panel, 'contract', True, 5120)


@pytest.mark.parametrize('change', [{'group': 'wrong'}, {'outputTokens': 1}, {'seconds': float('nan')}, {'stopReason': 'length'}, {'stopReason': 'input_limit'}])
def test_verifier_rejects_invalid_results_even_with_valid_hash(tmp_path, change):
    panel, row = fixture(tmp_path)
    row.update(change)
    # For nonfinite costs write directly, since the production writer rejects NaN.
    import json
    row['sha256'] = verify.base.digest(row)
    (tmp_path / 'items/0000.json').write_text(json.dumps(row))
    with pytest.raises(ValueError):
        verify.verify_items(tmp_path, panel, 'contract', True, 5120)

