from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.lora_assets import LoraAssetError, build_lora_assets
from gradloop_npu.smoke_assets import build_smoke_assets


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_lora_assets_are_deterministic_and_public_safe(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_smoke_assets(smoke)

    first_manifest = build_lora_assets(smoke_root=smoke, output_root=first)
    second_manifest = build_lora_assets(smoke_root=smoke, output_root=second)

    assert first_manifest == second_manifest
    assert first_manifest["sample_count"] == 10
    assert first_manifest["private_data_allowed"] is False
    assert _tree_hashes(first) == _tree_hashes(second)
    rows = [
        json.loads(line)
        for line in (first / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 10
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)
    assert sum("images" in row for row in rows) == 5
    assert sum("audios" in row for row in rows) == 5
    rendered = json.dumps(rows)
    assert "private record" in rendered
    assert str(tmp_path) not in rendered


def test_lora_asset_builder_never_overwrites(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    output = tmp_path / "existing"
    build_smoke_assets(smoke)
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(LoraAssetError, match="output_exists"):
        build_lora_assets(smoke_root=smoke, output_root=output)
    assert marker.read_text(encoding="utf-8") == "keep"
