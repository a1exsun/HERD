from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from hero_qwen.config import MODEL_ID, MODEL_REVISION, sha256_file, sha256_path


def test_pins_the_qwen_revision() -> None:
    assert MODEL_ID == "Qwen/Qwen3.5-4B"
    assert len(MODEL_REVISION) == 40


def test_file_digest_is_plain_sha256(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"weights")
    assert sha256_file(path) == sha256_path(path) == hashlib.sha256(b"weights").hexdigest()


def test_directory_digest_binds_relative_paths_and_contents(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"x")
    (tmp_path / "y.bin").write_bytes(b"y")
    digest = sha256_path(tmp_path)
    (tmp_path / "y.bin").rename(tmp_path / "z.bin")
    assert sha256_path(tmp_path) != digest
    (tmp_path / "z.bin").rename(tmp_path / "y.bin")
    assert sha256_path(tmp_path) == digest
    (tmp_path / "y.bin").write_bytes(b"Y")
    assert sha256_path(tmp_path) != digest


def test_missing_artifact_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither a file nor directory"):
        sha256_path(tmp_path / "missing")
