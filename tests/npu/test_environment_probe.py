from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.environment import collect_environment


class _ReadyNpu:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(index: int) -> str:
        assert index == 0
        return "Ascend Test Device"


class _ReadyTorch:
    npu = _ReadyNpu()


def test_ready_probe_reports_only_non_sensitive_runtime_facts() -> None:
    versions = {
        "torch": "2.6.0",
        "torch-npu": "2.6.0",
        "transformers": "4.51.0",
        "accelerate": "1.5.0",
        "torchaudio": "2.6.0",
        "timm": "1.0.15",
        "decord": "0.6.0",
        "soundfile": "0.13.1",
        "minicpmo-utils": "1.0.6",
    }

    def importer(name: str):
        if name == "torch":
            return _ReadyTorch()
        if name == "torch_npu":
            return object()
        raise ImportError(name)

    report = collect_environment(
        importer=importer,
        version_resolver=versions.__getitem__,
        python_version=(3, 11, 9),
        machine="aarch64",
    )

    assert report["status"] == "ready"
    assert report["privacy"] == {
        "private_data_allowed": False,
        "prompts_retained": False,
        "outputs_retained": False,
        "paths_retained": False,
    }
    assert report["python_version"] == "3.11.9"
    assert report["machine"] == "aarch64"
    assert report["packages"] == versions
    assert report["npu"] == {
        "available": True,
        "device_count": 1,
        "device_name": "Ascend Test Device",
    }
    assert report["blockers"] == []


def test_matching_torch_2_10_runtime_is_supported() -> None:
    versions = {
        "torch": "2.10.0",
        "torch-npu": "2.10.0.post2",
        "transformers": "4.51.3",
        "accelerate": "1.10.0",
        "torchaudio": "2.10.0",
        "timm": "1.0.15",
        "decord": "0.6.0",
        "soundfile": "0.13.1",
        "minicpmo-utils": "1.0.6",
    }

    def importer(name: str):
        if name == "torch":
            return _ReadyTorch()
        if name == "torch_npu":
            return object()
        raise ImportError(name)

    report = collect_environment(
        importer=importer,
        version_resolver=versions.__getitem__,
        python_version=(3, 11, 15),
        machine="aarch64",
    )

    assert report["status"] == "ready"
    assert report["blockers"] == []


def test_linux_arm64_decord2_distribution_is_supported() -> None:
    versions = {
        "torch": "2.10.0",
        "torch-npu": "2.10.0",
        "transformers": "4.51.3",
        "accelerate": "1.14.0",
        "torchaudio": "2.10.0",
        "timm": "1.0.28",
        "decord2": "3.4.0",
        "soundfile": "0.14.0",
        "minicpmo-utils": "1.0.6",
    }

    def importer(name: str):
        if name == "torch":
            return _ReadyTorch()
        if name == "torch_npu":
            return object()
        raise ImportError(name)

    report = collect_environment(
        importer=importer,
        version_resolver=versions.__getitem__,
        python_version=(3, 11, 15),
        machine="aarch64",
    )

    assert report["status"] == "ready"
    assert report["packages"]["decord"] == "3.4.0"
    assert report["blockers"] == []


def test_missing_npu_fails_closed_without_echoing_import_details() -> None:
    def importer(name: str):
        raise RuntimeError(f"sensitive detail from {name}")

    def missing_version(name: str) -> str:
        raise LookupError(name)

    report = collect_environment(
        importer=importer,
        version_resolver=missing_version,
        python_version=(3, 10, 14),
        machine="x86_64",
    )

    assert report["status"] == "blocked"
    assert report["npu"] == {
        "available": False,
        "device_count": 0,
        "device_name": "unavailable",
    }
    assert set(report["blockers"]) == {
        "python_version_unsupported",
        "torch_missing",
        "torch_npu_missing",
        "transformers_missing",
        "accelerate_missing",
        "torchaudio_missing",
        "timm_missing",
        "decord_missing",
        "soundfile_missing",
        "minicpmo_utils_missing",
        "npu_unavailable",
    }
    assert "sensitive detail" not in repr(report)


def test_incompatible_official_versions_are_blocked() -> None:
    versions = {
        "torch": "2.8.0",
        "torch-npu": "2.6.0",
        "transformers": "4.52.0",
        "accelerate": "1.5.0",
        "torchaudio": "2.8.0",
        "timm": "1.0.15",
        "decord": "0.6.0",
        "soundfile": "0.13.1",
        "minicpmo-utils": "1.0.4",
    }

    def importer(name: str):
        if name == "torch":
            return _ReadyTorch()
        if name == "torch_npu":
            return object()
        raise ImportError(name)

    report = collect_environment(
        importer=importer,
        version_resolver=versions.__getitem__,
        python_version=(3, 11, 9),
        machine="aarch64",
    )

    assert report["status"] == "blocked"
    assert set(report["blockers"]) == {
        "torch_npu_version_mismatch",
        "transformers_version_unsupported",
        "minicpmo_utils_version_unsupported",
    }
