from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from hero_qwen.config import sha256_file
from hero_qwen.model_registry import (
    activate_version,
    download_base,
    initialize_base,
    install_adapter,
    install_stream_stage,
)


def _checkpoint(path: Path, marker: str) -> Path:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
    )
    (path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (path / "weights.bin").write_text(marker, encoding="utf-8")
    return path


def _adapter(path: Path, marker: str, **config: object) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", **config}), encoding="utf-8"
    )
    (path / "adapter_model.safetensors").write_text(marker, encoding="utf-8")
    return path


def _materialize(source: Path, adapters: list[Path], destination: Path) -> None:
    shutil.copytree(source, destination)
    applied = [Path(path).name for path in adapters]
    (destination / "applied.json").write_text(json.dumps(applied), encoding="utf-8")


def _stream_run(path: Path, run_marker: str, stages: int, *, rank: int = 16) -> Path:
    """A stream/run.py output: bundle/plan.json and output/stage-NN/{adapter,stage.json}."""
    tasks = [{"stage": n, "taskId": f"trace:T{n}", "file": f"t{n}"} for n in range(1, 4)]
    plan = {
        "schemaVersion": "hero-stream-plan-v2",
        "model": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "tasks": tasks,
        "lora": {"rank": 16, "alpha": 32, "dropout": 0.0},
    }
    (path / "bundle").mkdir(parents=True)
    (path / "bundle" / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    for stage in range(1, stages + 1):
        directory = path / "output" / f"stage-{stage:02d}"
        _adapter(directory / "adapter", f"{run_marker}-{stage}", r=rank, lora_alpha=2 * rank)
        digest = hashlib.sha256()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            digest.update(name.encode() + (directory / "adapter" / name).read_bytes())
        (directory / "stage.json").write_text(json.dumps({
            "stage": stage,
            "taskId": f"trace:T{stage}",
            "adapterSha256": digest.hexdigest(),
            "weightsFingerprint": f"{stage:x}" * 64,
            "seconds": run_marker,
            "job": {"slurmJobId": "170603"},
        }), encoding="utf-8")
    return path


def test_installs_lora_into_rolling_active_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "models"
    base = _checkpoint(tmp_path / "base", "base")
    registry = initialize_base(root, base)
    assert registry["activeVersionId"] == "base"
    registry = install_adapter(
        root,
        _adapter(tmp_path / "adapter-1", "one"),
        1,
        source_job_id="42",
        materializer=_materialize,
    )
    version = registry["versions"][-1]
    assert registry["activeVersionId"] == version["id"]
    assert Path(registry["activeCheckpointPath"]) == root / "active"
    assert version["sourceJobId"] == "42" and "stream" not in version
    assert json.loads((root / "active" / "applied.json").read_text()) == [version["id"]]


def test_installs_stream_stages_along_the_run_parent_chain(tmp_path: Path) -> None:
    root = tmp_path / "models"
    initialize_base(root, _checkpoint(tmp_path / "base", "base"))
    run = _stream_run(tmp_path / "run", "a", 2)
    first = install_stream_stage(root, run, 1, origin="remote:/r/a", materializer=_materialize)
    second = install_stream_stage(root, run, 2, materializer=_materialize)
    one, two = first["versions"][-1], second["versions"][-1]
    assert one["stream"] == {
        "planSha256": sha256_file(run / "bundle" / "plan.json"),
        "taskId": "trace:T1",
        "stageJsonSha256": sha256_file(run / "output" / "stage-01" / "stage.json"),
        "parentStageJsonSha256": None,
        "weightsFingerprint": "1" * 64,
        "adapterSha256": json.loads((run / "output" / "stage-01" / "stage.json").read_text())["adapterSha256"],
        "origin": "remote:/r/a",
    }
    assert one["sourceJobId"] == "170603"
    assert two["parentVersionId"] == one["id"]
    assert two["stream"]["parentStageJsonSha256"] == one["stream"]["stageJsonSha256"]
    assert two["stream"]["origin"] == str(run)
    applied = json.loads((root / "active" / "applied.json").read_text())
    assert applied == [two["id"]]  # merged onto the active stage-1 checkpoint


def test_stream_stage_requires_its_own_runs_preceding_stage(tmp_path: Path) -> None:
    root = tmp_path / "models"
    initialize_base(root, _checkpoint(tmp_path / "base", "base"))
    run_a = _stream_run(tmp_path / "run-a", "a", 2)
    run_b = _stream_run(tmp_path / "run-b", "b", 2)  # same plan, different stage-01 result
    with pytest.raises(ValueError, match="Adapter stage must be 1"):
        install_stream_stage(root, run_a, 2, materializer=_materialize)
    install_stream_stage(root, run_b, 1, materializer=_materialize)
    with pytest.raises(ValueError, match="is not stage 1 of this stream run"):
        install_stream_stage(root, run_a, 2, materializer=_materialize)
    activate_version(root, "base", materializer=_materialize)
    install_adapter(root, _adapter(tmp_path / "manual", "m"), 1, materializer=_materialize)
    with pytest.raises(ValueError, match="is not stage 1 of this stream run"):
        install_stream_stage(root, run_a, 2, materializer=_materialize)
    assert not list((root / "adapters").glob("stage-2-*"))
    assert [v["stage"] for v in json.loads((root / "registry.json").read_text())["versions"]] == [0, 1, 1]


def test_stream_stage_is_verified_against_stage_json_and_plan(tmp_path: Path) -> None:
    root = tmp_path / "models"
    initialize_base(root, _checkpoint(tmp_path / "base", "base"))
    run = _stream_run(tmp_path / "run", "a", 1)
    with pytest.raises(ValueError, match="stage 2 has not completed"):
        install_stream_stage(root, run, 2, materializer=_materialize)
    with pytest.raises(ValueError, match="no stage 4"):
        install_stream_stage(root, run, 4, materializer=_materialize)
    record = run / "output" / "stage-01" / "stage.json"
    record.write_text(json.dumps({**json.loads(record.read_text()), "taskId": "trace:T2"}))
    with pytest.raises(ValueError, match=r"is not stage 1 \(trace:T1\)"):
        install_stream_stage(root, run, 1, materializer=_materialize)
    wrong_rank = _stream_run(tmp_path / "rank", "a", 1, rank=8)
    with pytest.raises(ValueError, match="rank or alpha"):
        install_stream_stage(root, wrong_rank, 1, materializer=_materialize)
    assert json.loads((root / "registry.json").read_text())["activeVersionId"] == "base"


def test_reconstructs_versions_from_immutable_base_and_adapter_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models"
    base = _checkpoint(tmp_path / "base", "base")
    initialize_base(root, base)
    first = install_adapter(
        root, _adapter(tmp_path / "adapter-1", "one"), 1, materializer=_materialize
    )
    first_id = first["activeVersionId"]
    second = install_adapter(
        root, _adapter(tmp_path / "adapter-2", "two"), 2, materializer=_materialize
    )
    second_id = second["activeVersionId"]
    base_active = activate_version(root, "base", materializer=_materialize)
    assert base_active["activeCheckpointPath"] == str(base)
    restored = activate_version(root, first_id, materializer=_materialize)
    assert restored["activeVersionId"] == first_id
    assert second_id in {version["id"] for version in restored["versions"]}


def test_stream_adapter_is_bound_by_stage_json_adapter_sha256(tmp_path: Path) -> None:
    root = tmp_path / "models"
    initialize_base(root, _checkpoint(tmp_path / "base", "base"))
    run = _stream_run(tmp_path / "run", "a", 1)
    adapter = run / "output" / "stage-01" / "adapter"
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        digest.update(name.encode() + (adapter / name).read_bytes())
    record_path = run / "output" / "stage-01" / "stage.json"
    record = json.loads(record_path.read_text())
    record_path.write_text(json.dumps({**record, "adapterSha256": "0" * 64}))
    with pytest.raises(ValueError, match="differs from the adapterSha256"):
        install_stream_stage(root, run, 1, materializer=_materialize)
    record_path.write_text(json.dumps({**record, "adapterSha256": digest.hexdigest()}))
    version = install_stream_stage(root, run, 1, materializer=_materialize)["versions"][-1]
    assert version["stream"]["adapterSha256"] == digest.hexdigest()


def test_failed_materialization_leaves_no_pending_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "models"
    initialize_base(root, _checkpoint(tmp_path / "base", "base"))

    def incomplete(source: Path, adapters: list[Path], destination: Path) -> None:
        _materialize(source, adapters, destination)
        (destination / "model.safetensors.index.json").unlink()

    with pytest.raises(ValueError, match="Checkpoint is incomplete"):
        install_adapter(root, _adapter(tmp_path / "a", "one"), 1, materializer=incomplete)
    assert not (root / "active-next").exists() and not list((root / "adapters").iterdir())

    def single_file(source: Path, adapters: list[Path], destination: Path) -> None:
        incomplete(source, adapters, destination)
        (destination / "model.safetensors").write_text("merged", encoding="utf-8")

    registry = install_adapter(root, _adapter(tmp_path / "b", "two"), 1, materializer=single_file)
    assert registry["activeVersionId"].startswith("stage-1-")


def test_download_is_pinned_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        base = _checkpoint(root / "base", "base")
        (base / ".cache").mkdir()
        return str(base)

    first = download_base(root, downloader=downloader)

    def repeated_download(**_: object) -> str:
        raise AssertionError("download repeated")

    second = download_base(root, downloader=repeated_download)
    assert first == second
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "local_dir": root / "base",
            "allow_patterns": ["*.json", "*.safetensors", "*.model", "*.txt"],
        }
    ]
    assert not (root / "base" / ".cache").exists()
