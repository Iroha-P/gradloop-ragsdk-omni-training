from __future__ import annotations

import hashlib
import json
import sys
import wave
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.smoke_assets import AssetBuildError, build_smoke_assets


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_smoke_assets_are_deterministic_and_multimodal(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_manifest = build_smoke_assets(first)
    second_manifest = build_smoke_assets(second)

    assert first_manifest == second_manifest
    assert first_manifest["sample_count"] == 10
    assert first_manifest["modality_counts"] == {
        "audio": 5,
        "image": 5,
        "text": 10,
    }
    assert _tree_hashes(first) == _tree_hashes(second)

    assert (first / "synthetic-grid.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    with wave.open(str(first / "synthetic-tone.wav"), "rb") as audio:
        assert audio.getframerate() == 16_000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2

    cases = [
        json.loads(line)
        for line in (first / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(cases) == 10
    assert all("prompt" in case and "modalities" in case for case in cases)
    routes = {
        (
            "omni"
            if {"image", "audio"} <= set(case["modalities"])
            else "vision"
            if "image" in case["modalities"]
            else "audio"
            if "audio" in case["modalities"]
            else "text"
        )
        for case in cases
    }
    assert routes == {"text", "vision", "audio", "omni"}


def test_smoke_asset_builder_never_overwrites(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(AssetBuildError, match="output_exists"):
        build_smoke_assets(output)

    assert marker.read_text(encoding="utf-8") == "keep"
