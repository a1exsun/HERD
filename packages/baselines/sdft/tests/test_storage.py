"""Scratch adapter identity and persistent checkpoint capacity boundaries."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from run import Unit, persist_checkpoint, tree_hash


def test_sealed_adapter_reads_scratch_and_rejects_changed_state(tmp_path):
    output, scratch = tmp_path / 'output', tmp_path / 'scratch'
    adapter = scratch / 'stage-01/adapter'; adapter.mkdir(parents=True)
    (adapter / 'weights').write_bytes(b'exact sealed state')
    seal = output / 'stage-01/training.json'; seal.parent.mkdir(parents=True)
    seal.write_text(json.dumps({'adapterSha256': tree_hash(adapter)}))
    unit = object.__new__(Unit); unit.output = output; unit.adapter_root = scratch
    assert unit.sealed_adapter(1) == adapter
    (adapter / 'weights').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        unit.sealed_adapter(1)


def test_checkpoint_capacity_failure_preserves_old_checkpoint(tmp_path, monkeypatch):
    old = tmp_path / 'work/checkpoint'; old.mkdir(parents=True)
    (old / 'state').write_bytes(b'old verified state')
    staging = tmp_path / 'scratch'; staging.mkdir(); (staging / 'state').write_bytes(b'new')
    monkeypatch.setattr(shutil, 'disk_usage', lambda path: SimpleNamespace(free=512 * 1024 ** 2 + 2))
    with pytest.raises(RuntimeError, match='capacity'):
        persist_checkpoint(staging, old)
    assert (old / 'state').read_bytes() == b'old verified state'
    assert not (old.parent / 'checkpoint.new').exists()


def test_checkpoint_copy_is_verified_before_old_retirement(tmp_path, monkeypatch):
    old = tmp_path / 'work/checkpoint'; old.mkdir(parents=True); (old / 'state').write_bytes(b'old')
    staging = tmp_path / 'scratch'; staging.mkdir(); (staging / 'state').write_bytes(b'new')
    copy = shutil.copytree
    def corrupt(source, destination):
        copy(source, destination); (destination / 'state').write_bytes(b'corrupt')
    monkeypatch.setattr(shutil, 'copytree', corrupt)
    with pytest.raises(RuntimeError, match='verification'):
        persist_checkpoint(staging, old)
    assert (old / 'state').read_bytes() == b'old'


def test_checkpoint_exact_copy_replaces_previous(tmp_path):
    old = tmp_path / 'work/checkpoint'; old.mkdir(parents=True); (old / 'state').write_bytes(b'old')
    staging = tmp_path / 'scratch'; staging.mkdir(); (staging / 'state').write_bytes(b'new exact bytes')
    persist_checkpoint(staging, old)
    assert tree_hash(old) == tree_hash(staging)


def test_verified_pending_checkpoint_recovers_same_run(tmp_path):
    from run import CHECKPOINT_ENCODING, file_hash, recover_pending_checkpoint
    work = tmp_path / 'stage-04/training.tmp'; pending = work / 'checkpoint.new'
    (pending / 'adapter').mkdir(parents=True); (pending / 'adapter/weights').write_bytes(b'adapter')
    for name in ['teacher.xor.gz', 'optimizer.pt', 'rng.pt']:
        (pending / name).write_bytes(name.encode())
    saved = {'nextUpdate': 12, 'adapterSha256': tree_hash(pending / 'adapter'),
             'teacherEncoding': CHECKPOINT_ENCODING, 'teacherSha256': file_hash(pending / 'teacher.xor.gz'),
             'optimizerSha256': file_hash(pending / 'optimizer.pt'), 'rngSha256': file_hash(pending / 'rng.pt')}
    (pending / 'progress.json').write_text(json.dumps(saved))
    expected = tree_hash(pending)
    recover_pending_checkpoint(work)
    assert tree_hash(work / 'checkpoint') == expected
    assert not pending.exists()


def test_incomplete_pending_preserves_original_state(tmp_path):
    from run import recover_pending_checkpoint
    work = tmp_path / 'stage-04/training.tmp'
    (work / 'checkpoint').mkdir(parents=True); (work / 'checkpoint/old').write_bytes(b'old')
    (work / 'checkpoint.new').mkdir(); (work / 'checkpoint.new/partial').write_bytes(b'partial')
    with pytest.raises(RuntimeError, match='verification'):
        recover_pending_checkpoint(work)
    assert (work / 'checkpoint/old').read_bytes() == b'old'
    assert (work / 'checkpoint.new/partial').read_bytes() == b'partial'
