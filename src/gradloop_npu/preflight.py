"""Offline-only MiniCPM-o baseline preflight validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path

MODEL_ID = "openbmb/MiniCPM-o-4_5"
ARTIFACT_NAMES = (
    "samples.jsonl",
    "evidence.json",
    "source_manifest.json",
    "asset_manifest.json",
    "split_manifest.json",
    "release_report.json",
    "evaluation_cases.jsonl",
    "summary.json",
)
_PINNED_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class PreflightError(ValueError):
    """Stable, non-sensitive failure from an offline preflight check."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_summary(bundle_root: Path) -> Mapping[str, object]:
    try:
        value = json.loads((bundle_root / "summary.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError("bundle_invalid") from exc
    if type(value) is not dict:
        raise PreflightError("bundle_invalid")
    return value


def _validate_bundle(bundle_root: Path) -> tuple[Mapping[str, object], dict[str, str]]:
    if not bundle_root.is_dir():
        raise PreflightError("bundle_invalid")
    paths = {name: bundle_root / name for name in ARTIFACT_NAMES}
    if any(not path.is_file() for path in paths.values()):
        raise PreflightError("bundle_invalid")

    summary = _load_summary(bundle_root)
    support = summary.get("refusal_evaluation_support")
    is_releasable = (
        summary.get("allowed") is True
        and type(summary.get("sample_count")) is int
        and int(summary["sample_count"]) > 0
        and type(summary.get("evaluation_case_count")) is int
        and int(summary["evaluation_case_count"]) > 0
        and summary.get("finding_count") == 0
        and summary.get("source_leakage_count") == 0
        and type(support) is dict
        and type(support.get("positive")) is int
        and int(support["positive"]) > 0
        and type(support.get("negative")) is int
        and int(support["negative"]) > 0
    )
    if not is_releasable:
        raise PreflightError("bundle_not_releasable")

    return summary, {name: _sha256(path) for name, path in paths.items()}


def _validate_model_cache(model_cache_root: Path) -> dict[str, object]:
    if not model_cache_root.is_dir():
        raise PreflightError("model_cache_incomplete")
    required = ("config.json", "model.safetensors.index.json")
    if any(not (model_cache_root / name).is_file() for name in required):
        raise PreflightError("model_cache_incomplete")

    index_path = model_cache_root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PreflightError("model_cache_incomplete") from exc
    if type(index) is not dict or type(index.get("weight_map")) is not dict:
        raise PreflightError("model_cache_incomplete")
    weight_map = index["weight_map"]
    if not weight_map:
        raise PreflightError("model_cache_incomplete")
    shard_names = tuple(sorted(set(weight_map.values())))
    if any(
        type(name) is not str
        or Path(name).name != name
        or Path(name).suffix != ".safetensors"
        for name in shard_names
    ):
        raise PreflightError("model_cache_incomplete")
    if any(not (model_cache_root / name).is_file() for name in shard_names):
        raise PreflightError("model_cache_incomplete")
    return {
        "required_files": [*required, *shard_names],
        "shard_count": len(shard_names),
        "index_sha256": _sha256(index_path),
    }


def prepare_preflight(
    *,
    bundle_root: Path,
    model_cache_root: Path,
    model_revision: str,
) -> dict[str, object]:
    """Verify local-only model and data inputs without importing the model."""

    if type(model_revision) is not str or _PINNED_REVISION.fullmatch(model_revision) is None:
        raise PreflightError("model_revision_unpinned")

    summary, artifact_hashes = _validate_bundle(bundle_root)
    model_cache = _validate_model_cache(model_cache_root)
    return {
        "schema_version": "1.0",
        "status": "ready",
        "privacy": {
            "private_data_allowed": False,
            "prompts_retained": False,
            "outputs_retained": False,
            "paths_retained": False,
        },
        "model": {
            "id": MODEL_ID,
            "revision": model_revision,
            "dtype": "bfloat16",
            "local_files_only": True,
            "trust_remote_code": True,
        },
        "model_cache": model_cache,
        "bundle": {
            "sample_count": summary["sample_count"],
            "evaluation_case_count": summary["evaluation_case_count"],
            "artifact_sha256": artifact_hashes,
        },
    }
