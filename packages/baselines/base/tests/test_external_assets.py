from pathlib import Path
from types import SimpleNamespace

import run as base


def test_contract_supports_explicit_checkpoint_and_records_actual_scorer_assets(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'weights'
    checkpoint.mkdir()
    (checkpoint / 'tokenizer_config.json').write_text('{}')
    registry = tmp_path / 'registry.json'
    base.atomic_json(registry, {'model': base.MODEL_ID, 'modelRevision': base.REVISION,
                               'base': {'sha256': base.tree_hash(checkpoint)}})
    monkeypatch.setattr(base, 'general_panel', lambda _: ([{'id': 'a', 'prompt': 'p'}], {'generalCategories': ['test']}))
    args = SimpleNamespace(checkpoint=checkpoint, registry=registry, profile='general',
                           general_dataset=tmp_path, device='cpu', max_new_tokens=5120)
    contract, _, provenance, actual, _ = base.build_contract(args)
    assert actual == checkpoint
    assert provenance['checkpointPath'] == str(checkpoint)
    assert contract['scorerSha256'] == base.file_hash(Path(base.__file__).with_name('scoring.py'))
    assert all(Path(path).is_absolute() for path in contract['scorerAssets'])
    assert all(base.file_hash(Path(path)) == value for path, value in contract['scorerAssets'].items())
