from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.preflight import PreflightError, prepare_preflight

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


def _write_bundle(root: Path) -> None:
    root.mkdir()
    summary = {
        "allowed": True,
        "sample_count": 60,
        "evaluation_case_count": 5,
        "finding_count": 0,
        "source_leakage_count": 0,
        "refusal_evaluation_support": {"positive": 3, "negative": 2},
    }
    for name in ARTIFACT_NAMES:
        content = json.dumps(summary if name == "summary.json" else {})
        (root / name).write_text(content + "\n", encoding="utf-8")


def _write_model_cache(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text("{}\n", encoding="utf-8")
    index = {
        "weight_map": {
            "model.layer": "model-00001-of-00001.safetensors",
        }
    }
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index) + "\n", encoding="utf-8"
    )
    (root / "model-00001-of-00001.safetensors").write_bytes(b"fixture")


def test_preflight_is_local_only_and_does_not_expose_paths(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    model_cache = tmp_path / "model-cache"
    _write_bundle(bundle)
    _write_model_cache(model_cache)

    report = prepare_preflight(
        bundle_root=bundle,
        model_cache_root=model_cache,
        model_revision="a" * 40,
    )

    assert report["status"] == "ready"
    assert report["privacy"] == {
        "private_data_allowed": False,
        "prompts_retained": False,
        "outputs_retained": False,
        "paths_retained": False,
    }
    assert report["model"] == {
        "id": "openbmb/MiniCPM-o-4_5",
        "revision": "a" * 40,
        "dtype": "bfloat16",
        "local_files_only": True,
        "trust_remote_code": True,
    }
    assert report["bundle"]["sample_count"] == 60
    assert report["bundle"]["evaluation_case_count"] == 5
    assert set(report["bundle"]["artifact_sha256"]) == set(ARTIFACT_NAMES)
    assert report["model_cache"]["shard_count"] == 1
    assert len(report["model_cache"]["index_sha256"]) == 64
    assert str(tmp_path) not in repr(report)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"model_revision": "main"}, "model_revision_unpinned"),
        ({"missing_model": True}, "model_cache_incomplete"),
        ({"unsafe_summary": True}, "bundle_not_releasable"),
    ],
)
def test_preflight_fails_closed(
    tmp_path: Path, change: dict[str, object], reason: str
) -> None:
    bundle = tmp_path / "bundle"
    model_cache = tmp_path / "model-cache"
    _write_bundle(bundle)
    _write_model_cache(model_cache)
    revision = str(change.get("model_revision", "b" * 40))

    if change.get("missing_model"):
        (model_cache / "model-00001-of-00001.safetensors").unlink()
    if change.get("unsafe_summary"):
        summary_path = bundle / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["finding_count"] = 1
        summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(PreflightError, match=reason):
        prepare_preflight(
            bundle_root=bundle,
            model_cache_root=model_cache,
            model_revision=revision,
        )
