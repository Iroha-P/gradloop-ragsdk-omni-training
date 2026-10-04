"""One-update SWIFT LoRA feasibility gate for Ascend NPU."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

_PINNED_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_DATA_ARTIFACTS = (
    "train.jsonl",
    "synthetic-grid.png",
    "synthetic-tone.wav",
)


class LoraRunError(ValueError):
    """Stable, non-sensitive failure from the LoRA feasibility gate."""


ProcessRunner = Callable[[Sequence[str], Mapping[str, str], Path, Path], int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, reason: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LoraRunError(reason) from exc
    if type(value) is not dict:
        raise LoraRunError(reason)
    return value


def _validate_dataset(dataset_root: Path) -> dict[str, object]:
    manifest = _read_json(
        dataset_root / "lora_manifest.json", "lora_dataset_invalid"
    )
    artifacts = manifest.get("artifacts")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("sample_count") != 10
        or manifest.get("private_data_allowed") is not False
        or type(artifacts) is not dict
        or set(artifacts) != set(_REQUIRED_DATA_ARTIFACTS)
    ):
        raise LoraRunError("lora_dataset_invalid")
    for name in _REQUIRED_DATA_ARTIFACTS:
        expected = artifacts.get(name)
        path = dataset_root / name
        if (
            type(expected) is not str
            or _SHA256.fullmatch(expected) is None
            or not path.is_file()
            or _sha256(path) != expected
        ):
            raise LoraRunError("lora_dataset_invalid")
    return manifest


def _find_adapter(output_root: Path) -> tuple[Path, dict[str, object]]:
    adapter_paths = tuple(output_root.rglob("adapter_model.safetensors"))
    candidates = tuple(
        path.parent
        for path in adapter_paths
        if (path.parent / "adapter_config.json").is_file()
    )
    if len(candidates) != 1:
        raise LoraRunError("adapter_artifact_invalid")
    adapter_root = candidates[0]
    full_model_paths = (
        *output_root.rglob("model.safetensors"),
        *output_root.rglob("model-*.safetensors"),
    )
    if full_model_paths:
        raise LoraRunError("full_model_artifact_forbidden")

    trainer_states = tuple(output_root.rglob("trainer_state.json"))
    global_steps: list[int] = []
    for path in trainer_states:
        value = _read_json(path, "trainer_state_invalid")
        step = value.get("global_step")
        if type(step) is int:
            global_steps.append(step)
    if not global_steps or max(global_steps) < 1:
        raise LoraRunError("optimizer_update_unproven")

    artifacts = {
        name: _sha256(adapter_root / name)
        for name in ("adapter_model.safetensors", "adapter_config.json")
    }
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "peft_lora_adapter",
        "optimizer_updates": max(global_steps),
        "private_data_allowed": False,
        "full_model_weights_included": False,
        "artifacts": artifacts,
    }
    return adapter_root, manifest


def build_swift_command(
    *,
    swift_executable: Path,
    config_path: Path,
    model_cache_root: Path,
    dataset_root: Path,
    output_root: Path,
) -> tuple[str, ...]:
    """Build a shell-free SWIFT invocation from validated local inputs."""

    if not swift_executable.is_file() or not config_path.is_file():
        raise LoraRunError("swift_runtime_unavailable")
    return (
        str(swift_executable.resolve()),
        "sft",
        str(config_path.resolve()),
        "--model",
        str(model_cache_root.resolve()),
        "--dataset",
        str((dataset_root / "train.jsonl").resolve()),
        "--output_dir",
        str(output_root.resolve()),
    )


def _default_runner(
    command: Sequence[str],
    environment: Mapping[str, str],
    working_directory: Path,
    log_path: Path,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        completed = subprocess.run(
            tuple(command),
            cwd=working_directory,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def run_lora_feasibility(
    *,
    swift_executable: Path,
    config_path: Path,
    model_cache_root: Path,
    model_revision: str,
    dataset_root: Path,
    output_root: Path,
    log_path: Path,
    code_commit: str,
    device_index: int,
    process_runner: ProcessRunner = _default_runner,
) -> tuple[dict[str, object], dict[str, object]]:
    """Execute one SWIFT update and prove a LoRA-only artifact was saved."""

    if (
        type(model_revision) is not str
        or _PINNED_REVISION.fullmatch(model_revision) is None
        or type(code_commit) is not str
        or not code_commit
        or type(device_index) is not int
        or device_index < 0
        or output_root.exists()
        or log_path.exists()
    ):
        raise LoraRunError("lora_run_identity_invalid")
    dataset_manifest = _validate_dataset(dataset_root)
    command = build_swift_command(
        swift_executable=swift_executable,
        config_path=config_path,
        model_cache_root=model_cache_root,
        dataset_root=dataset_root,
        output_root=output_root,
    )
    environment = dict(os.environ)
    environment.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": str(device_index),
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "SWIFT_SINGLE_DEVICE_MODE": "1",
            "ROOT_IMAGE_DIR": str(dataset_root),
        }
    )
    try:
        return_code = process_runner(command, environment, dataset_root, log_path)
    except Exception as exc:
        raise LoraRunError("lora_process_failed") from exc
    if return_code != 0:
        raise LoraRunError("lora_process_failed")

    _, adapter_manifest = _find_adapter(output_root)
    report = {
        "schema_version": "1.0",
        "status": "trained",
        "model": {
            "id": "openbmb/MiniCPM-o-4_5",
            "revision": model_revision,
            "dtype": "bfloat16",
            "local_files_only": True,
        },
        "run_identity": {
            "code_commit": code_commit,
            "config_sha256": _sha256(config_path),
            "dataset_manifest_sha256": _sha256(
                dataset_root / "lora_manifest.json"
            ),
        },
        "training": {
            "framework": "ms-swift",
            "tuner": "lora",
            "optimizer_updates": adapter_manifest["optimizer_updates"],
            "sample_count": dataset_manifest["sample_count"],
        },
        "privacy": {
            "private_data_allowed": False,
            "raw_dataset_retained_in_report": False,
            "paths_retained": False,
            "hub_push_enabled": False,
        },
        "adapter": adapter_manifest,
    }
    return report, adapter_manifest
