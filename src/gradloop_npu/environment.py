"""Read-only, privacy-safe Ascend runtime probing."""

from __future__ import annotations

import platform
import re
import sys
from importlib import import_module
from importlib.metadata import version
from typing import Callable

_REQUIRED_PACKAGES = {
    "torch": ("torch",),
    "torch-npu": ("torch-npu",),
    "transformers": ("transformers",),
    "accelerate": ("accelerate",),
    "torchaudio": ("torchaudio",),
    "timm": ("timm",),
    # The upstream decord project does not publish Linux ARM64 wheels.
    # decord2 is an API-compatible maintained distribution that still imports
    # as ``decord`` and provides ARM64 wheels.
    "decord": ("decord", "decord2"),
    "soundfile": ("soundfile",),
    "minicpmo-utils": ("minicpmo-utils",),
}
_SAFE_DEVICE_NAME = re.compile(r"^[A-Za-z0-9_.() +:-]{1,80}$")
_VERSION_PREFIX = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")


def _safe_package_version(
    distribution: str, resolver: Callable[[str], str]
) -> str:
    try:
        resolved = resolver(distribution)
    except Exception:  # noqa: BLE001 - metadata backends must fail closed.
        return "missing"
    if type(resolved) is not str or not resolved or len(resolved) > 80:
        return "unknown"
    return resolved


def _safe_package_version_from_candidates(
    distributions: tuple[str, ...], resolver: Callable[[str], str]
) -> str:
    for distribution in distributions:
        resolved = _safe_package_version(distribution, resolver)
        if resolved != "missing":
            return resolved
    return "missing"


def _safe_device_name(value: object) -> str:
    if type(value) is str and _SAFE_DEVICE_NAME.fullmatch(value):
        return value
    return "unknown"


def _version_tuple(value: str) -> tuple[int, int, int] | None:
    match = _VERSION_PREFIX.match(value)
    if match is None:
        return None
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3) or 0),
    )


def collect_environment(
    *,
    importer: Callable[[str], object] = import_module,
    version_resolver: Callable[[str], str] = version,
    python_version: tuple[int, int, int] | None = None,
    machine: str | None = None,
) -> dict[str, object]:
    """Collect only facts needed for a non-invasive NPU readiness decision."""

    current_python = python_version or (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    current_machine = machine or platform.machine() or "unknown"
    packages = {
        name: _safe_package_version_from_candidates(
            distributions, version_resolver
        )
        for name, distributions in _REQUIRED_PACKAGES.items()
    }
    blockers: list[str] = []

    if current_python[:2] != (3, 11):
        blockers.append("python_version_unsupported")
    for name in _REQUIRED_PACKAGES:
        if packages[name] == "missing":
            blockers.append(f"{name.replace('-', '_')}_missing")

    torch_version = _version_tuple(packages["torch"])
    torch_npu_version = _version_tuple(packages["torch-npu"])
    transformers_version = _version_tuple(packages["transformers"])
    utilities_version = _version_tuple(packages["minicpmo-utils"])
    if torch_version is not None and not ((2, 3, 0) <= torch_version <= (2, 10, 99)):
        blockers.append("torch_version_unsupported")
    if (
        torch_version is not None
        and torch_npu_version is not None
        and torch_version[:2] != torch_npu_version[:2]
    ):
        blockers.append("torch_npu_version_mismatch")
    if (
        transformers_version is not None
        and transformers_version[:2] != (4, 51)
    ):
        blockers.append("transformers_version_unsupported")
    if utilities_version is not None and utilities_version < (1, 0, 5):
        blockers.append("minicpmo_utils_version_unsupported")

    try:
        torch_module = importer("torch")
        importer("torch_npu")
        npu_module = torch_module.npu
        available = bool(npu_module.is_available())
        device_count = int(npu_module.device_count()) if available else 0
        device_name = (
            _safe_device_name(npu_module.get_device_name(0))
            if device_count > 0
            else "unavailable"
        )
    except Exception:  # noqa: BLE001 - runtime probes must fail closed.
        available = False
        device_count = 0
        device_name = "unavailable"

    if not available or device_count < 1:
        blockers.append("npu_unavailable")

    return {
        "schema_version": "1.0",
        "status": "ready" if not blockers else "blocked",
        "privacy": {
            "private_data_allowed": False,
            "prompts_retained": False,
            "outputs_retained": False,
            "paths_retained": False,
        },
        "python_version": ".".join(str(part) for part in current_python),
        "machine": current_machine,
        "packages": packages,
        "npu": {
            "available": available,
            "device_count": device_count,
            "device_name": device_name,
        },
        "blockers": blockers,
    }
