"""Versioned local Qwen base, LoRA chain, and rolling active checkpoint.

Stream-run stages (stream/run.py output) install with their provenance: stage N's adapter
was trained on stage N-1's merged weights of the same run, so it may only be installed on
top of base (N = 1) or of the version installed from that run's stage N-1.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MODEL_ID, MODEL_REVISION, sha256_file, sha256_path

REGISTRY_VERSION = "hero-local-model-registry-v1"
STREAM_PLAN_VERSION = "hero-stream-plan-v2"
Materializer = Callable[[Path, list[Path], Path], None]
Downloader = Callable[..., str]


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m hero_qwen.model_registry")
    parser.add_argument("--root", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init-base")
    initialize.add_argument("--base", required=True)
    subparsers.add_parser("download-base")
    subparsers.add_parser("show")
    install = subparsers.add_parser("install-adapter")
    source = install.add_mutually_exclusive_group(required=True)
    source.add_argument("--adapter")
    source.add_argument(
        "--stream-run",
        help="Stream run directory holding bundle/plan.json and output/stage-NN/",
    )
    install.add_argument("--stage", required=True, type=int)
    install.add_argument("--source-job-id")
    install.add_argument(
        "--stream-origin", help="Where the stream run came from (recorded only)"
    )
    activate = subparsers.add_parser("activate")
    activate.add_argument("--version", required=True)
    arguments = parser.parse_args()
    root = Path(arguments.root).resolve()
    if arguments.command == "init-base":
        value = initialize_base(root, Path(arguments.base).resolve())
    elif arguments.command == "download-base":
        value = download_base(root)
    elif arguments.command == "show":
        value = load_registry(root)
    elif arguments.command == "install-adapter" and arguments.stream_run is not None:
        if arguments.source_job_id is not None:
            parser.error("--source-job-id is read from stage.json for --stream-run")
        value = install_stream_stage(
            root,
            Path(arguments.stream_run).resolve(),
            arguments.stage,
            origin=arguments.stream_origin,
        )
    elif arguments.command == "install-adapter":
        if arguments.stream_origin is not None:
            parser.error("--stream-origin requires --stream-run")
        value = install_adapter(
            root,
            Path(arguments.adapter).resolve(),
            arguments.stage,
            source_job_id=arguments.source_job_id,
        )
    else:
        value = activate_version(root, arguments.version)
    print(json.dumps(value, indent=2))


def download_base(
    root: Path, *, downloader: Downloader | None = None
) -> dict[str, Any]:
    registry_path = root / "registry.json"
    if registry_path.exists():
        return load_registry(root)
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    base_path = root / "base"
    downloaded = Path(
        downloader(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=base_path,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        )
    ).resolve()
    if downloaded != base_path.resolve():
        raise ValueError("Hugging Face downloaded Qwen outside the requested base path.")
    download_metadata = base_path / ".cache"
    if download_metadata.exists():
        shutil.rmtree(download_metadata)
    return initialize_base(root, base_path.resolve())


def initialize_base(root: Path, base_path: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    _validate_checkpoint(base_path)
    with _registry_lock(root):
        registry_path = root / "registry.json"
        base_digest = sha256_path(base_path)
        if registry_path.exists():
            current = load_registry(root)
            if (
                current["base"]["sha256"] != base_digest
                or Path(current["base"]["path"]) != base_path
            ):
                raise ValueError(
                    "Existing model registry refers to a different immutable base."
                )
            return current
        value = {
            "schemaVersion": REGISTRY_VERSION,
            "model": MODEL_ID,
            "modelRevision": MODEL_REVISION,
            "base": {
                "path": str(base_path),
                "sha256": base_digest,
                "bytes": _directory_bytes(base_path),
            },
            "activeVersionId": "base",
            "activeCheckpointPath": str(base_path),
            "versions": [
                {
                    "id": "base",
                    "stage": 0,
                    "parentVersionId": None,
                    "adapter": None,
                    "materializedSha256": base_digest,
                    "installedAt": datetime.now(timezone.utc).isoformat(),
                }
            ],
        }
        _write_registry(registry_path, value)
        return value


def load_registry(root: Path) -> dict[str, Any]:
    registry_path = root / "registry.json"
    value = json.loads(registry_path.read_text(encoding="utf-8"))
    _validate_registry(value)
    return value


def install_adapter(
    root: Path,
    adapter_path: Path,
    stage: int,
    *,
    source_job_id: str | None = None,
    stream: dict[str, Any] | None = None,
    materializer: Materializer | None = None,
) -> dict[str, Any]:
    _validate_adapter(adapter_path)
    with _registry_lock(root):
        registry = load_registry(root)
        parent = _version(registry, registry["activeVersionId"])
        if stage != parent["stage"] + 1:
            raise ValueError(
                f"Adapter stage must be {parent['stage'] + 1} for the active parent."
            )
        if stream is not None:
            _check_stream_parent(parent, stream)
        adapter_digest = sha256_path(adapter_path)
        version_id = f"stage-{stage}-{adapter_digest[:12]}"
        if any(item["id"] == version_id for item in registry["versions"]):
            raise ValueError(f"Adapter version already exists: {version_id}")
        stored_adapter = root / "adapters" / version_id
        stored_adapter.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(adapter_path, stored_adapter)
        if sha256_path(stored_adapter) != adapter_digest:
            raise ValueError("Stored adapter failed digest verification.")
        try:
            active_source = Path(registry["activeCheckpointPath"])
            materialized_digest = _replace_active(
                root,
                active_source,
                [stored_adapter],
                materializer or _materialize,
            )
        except Exception:
            shutil.rmtree(stored_adapter, ignore_errors=True)
            raise
        version = {
            "id": version_id,
            "stage": stage,
            "parentVersionId": parent["id"],
            "adapter": {
                "path": str(stored_adapter),
                "sha256": adapter_digest,
                "bytes": _directory_bytes(stored_adapter),
            },
            "materializedSha256": materialized_digest,
            "installedAt": datetime.now(timezone.utc).isoformat(),
            **({} if source_job_id is None else {"sourceJobId": source_job_id}),
            **({} if stream is None else {"stream": stream}),
        }
        registry["versions"].append(version)
        registry["activeVersionId"] = version_id
        registry["activeCheckpointPath"] = str(root / "active")
        _write_registry(root / "registry.json", registry)
        _discard_previous(root)
        return registry


def install_stream_stage(
    root: Path,
    run: Path,
    stage: int,
    *,
    origin: str | None = None,
    materializer: Materializer | None = None,
) -> dict[str, Any]:
    """Install output/stage-NN/adapter of a stream run, verified against its stage.json and plan."""
    plan_path = run / "bundle" / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("schemaVersion") != STREAM_PLAN_VERSION
        or plan.get("model") != MODEL_ID
        or plan.get("revision") != MODEL_REVISION
    ):
        raise ValueError(
            f"Stream plan is not a {STREAM_PLAN_VERSION} plan for pinned Qwen3.5-4B."
        )
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not 1 <= stage <= len(tasks):
        raise ValueError(f"Stream plan has no stage {stage}.")
    directory = run / "output" / f"stage-{stage:02d}"
    record_path = directory / "stage.json"
    if not record_path.is_file():
        raise ValueError(f"Stream stage {stage} has not completed: {record_path} is missing.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("stage") != stage or record.get("taskId") != tasks[stage - 1]["taskId"]:
        raise ValueError(
            f"{record_path} is not stage {stage} ({tasks[stage - 1]['taskId']}) of the plan."
        )
    fingerprint = record.get("weightsFingerprint")
    if not isinstance(fingerprint, str) or not _is_sha256(fingerprint):
        raise ValueError(f"{record_path} has no weightsFingerprint.")
    adapter = directory / "adapter"
    _validate_adapter(adapter)
    adapter_digest = record.get("adapterSha256")
    if adapter_digest != _stream_adapter_sha256(adapter):
        raise ValueError(f"{adapter} differs from the adapterSha256 in {record_path}.")
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    lora = plan["lora"]
    if config.get("r") != lora["rank"] or config.get("lora_alpha") != lora["alpha"]:
        raise ValueError("Stream adapter rank or alpha differs from the plan's LoRA.")
    parent_record = None
    if stage > 1:
        parent_path = run / "output" / f"stage-{stage - 1:02d}" / "stage.json"
        if not parent_path.is_file():
            raise ValueError(f"Stream parent record {parent_path} is missing.")
        if json.loads(parent_path.read_text(encoding="utf-8")).get("stage") != stage - 1:
            raise ValueError(f"{parent_path} is not stage {stage - 1}.")
        parent_record = sha256_file(parent_path)
    job = record.get("job") or {}
    return install_adapter(
        root,
        adapter,
        stage,
        source_job_id=job.get("slurmJobId"),
        stream={
            "planSha256": sha256_file(plan_path),
            "taskId": record["taskId"],
            "stageJsonSha256": sha256_file(record_path),
            "parentStageJsonSha256": parent_record,
            "weightsFingerprint": fingerprint,
            **({} if adapter_digest is None else {"adapterSha256": adapter_digest}),
            "origin": origin if origin is not None else str(run),
        },
        materializer=materializer,
    )


def _stream_adapter_sha256(adapter: Path) -> str:
    """stream/run.py adapter_sha256: each file's name, then its bytes."""
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        digest.update(name.encode())
        with (adapter / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _check_stream_parent(parent: dict[str, Any], stream: dict[str, Any]) -> None:
    """Stage N > 1 must sit on the version installed from stage N-1 of the same run."""
    if stream["parentStageJsonSha256"] is None:
        return  # stage 1: install_adapter has already required the base as parent
    link = parent.get("stream")
    if (
        link is None
        or link["planSha256"] != stream["planSha256"]
        or link["stageJsonSha256"] != stream["parentStageJsonSha256"]
    ):
        raise ValueError(
            f"Active version {parent['id']} is not stage {parent['stage']} of this stream run; "
            "its adapter was trained on that run's merged weights. Activate the run's "
            "preceding stage (or base) first."
        )


def activate_version(
    root: Path,
    version_id: str,
    *,
    materializer: Materializer | None = None,
) -> dict[str, Any]:
    with _registry_lock(root):
        registry = load_registry(root)
        target = _version(registry, version_id)
        if target["id"] == registry["activeVersionId"]:
            return registry
        if target["id"] == "base":
            _retire_active(root)
            registry["activeVersionId"] = "base"
            registry["activeCheckpointPath"] = registry["base"]["path"]
            _write_registry(root / "registry.json", registry)
            _discard_previous(root)
            return registry
        chain = _adapter_chain(registry, version_id)
        digest = _replace_active(
            root,
            Path(registry["base"]["path"]),
            [Path(item["adapter"]["path"]) for item in chain],
            materializer or _materialize,
        )
        if digest != target["materializedSha256"]:
            raise ValueError(
                "Reconstructed checkpoint differs from the originally installed version."
            )
        registry["activeVersionId"] = version_id
        registry["activeCheckpointPath"] = str(root / "active")
        _write_registry(root / "registry.json", registry)
        _discard_previous(root)
        return registry


def _replace_active(
    root: Path, source: Path, adapters: list[Path], materializer: Materializer
) -> str:
    pending = root / "active-next"
    if pending.exists():
        raise FileExistsError(
            "active-next exists from an earlier interrupted materialization."
        )
    try:
        materializer(source, adapters, pending)
        _validate_checkpoint(pending)
        digest = sha256_path(pending)
    except Exception:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    _retire_active(root)
    pending.replace(root / "active")
    return digest


def _retire_active(root: Path) -> None:
    active = root / "active"
    previous = root / "active-previous"
    if previous.exists():
        raise FileExistsError(
            "active-previous exists from an earlier interrupted activation."
        )
    if active.exists():
        active.replace(previous)


def _discard_previous(root: Path) -> None:
    previous = root / "active-previous"
    if previous.exists():
        shutil.rmtree(previous)


def _materialize(source: Path, adapters: list[Path], destination: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    model = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, device_map="cpu"
    )
    for adapter in adapters:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload(
            safe_merge=True
        )
    model.save_pretrained(destination, safe_serialization=True)
    AutoTokenizer.from_pretrained(source).save_pretrained(destination)


def _adapter_chain(registry: dict[str, Any], version_id: str) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current = _version(registry, version_id)
    while current["id"] != "base":
        chain.append(current)
        current = _version(registry, current["parentVersionId"])
    chain.reverse()
    return chain


def _version(registry: dict[str, Any], version_id: str) -> dict[str, Any]:
    for version in registry["versions"]:
        if version["id"] == version_id:
            return version
    raise ValueError(f"Unknown local model version: {version_id}")


def _validate_registry(value: dict[str, Any]) -> None:
    if value.get("schemaVersion") != REGISTRY_VERSION:
        raise ValueError("Unsupported local model registry schema.")
    if value.get("model") != MODEL_ID or value.get("modelRevision") != MODEL_REVISION:
        raise ValueError("Local registry model identity is not pinned Qwen3.5-4B.")
    versions = value.get("versions")
    if (
        not isinstance(versions, list)
        or not versions
        or versions[0].get("id") != "base"
    ):
        raise ValueError("Local model registry has no base version.")
    if value.get("activeVersionId") not in {version.get("id") for version in versions}:
        raise ValueError("Local model registry active version does not exist.")
    for version in versions:
        stream = version.get("stream")
        if stream is not None and not all(
            isinstance(stream.get(key), str) and _is_sha256(stream[key])
            for key in ("planSha256", "stageJsonSha256", "weightsFingerprint")
        ):
            raise ValueError("Local model registry contains invalid stream provenance.")


def _validate_checkpoint(path: Path) -> None:
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (path / name).is_file()]
    # The pinned base is sharded; a local merge is saved as one file (Transformers 5's default shard size).
    if not any((path / name).is_file() for name in ("model.safetensors.index.json", "model.safetensors")):
        missing.append("model.safetensors(.index.json)")
    if missing:
        raise ValueError(f"Checkpoint is incomplete: missing {', '.join(missing)}")
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError("Checkpoint is not Qwen3.5.")


def _validate_adapter(path: Path) -> None:
    config_path = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights.is_file():
        raise ValueError(
            "LoRA adapter requires adapter_config.json and adapter_model.safetensors."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise ValueError("Only LoRA adapters can be installed into HERO Qwen.")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _write_registry(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    # Atomic replace pattern: write temporary file then rename onto destination registry.
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _registry_lock(root: Path):
    lock_path = root / "registry.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
        fcntl.flock(handle, fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
