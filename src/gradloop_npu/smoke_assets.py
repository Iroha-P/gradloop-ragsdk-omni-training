"""Deterministic, synthetic binary assets for the Stage B NPU smoke test."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
import tempfile
import wave
import zlib
from pathlib import Path


class AssetBuildError(ValueError):
    """Stable failure raised before a smoke bundle can be published."""


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return (
        struct.pack(">I", len(payload))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def _png_bytes() -> bytes:
    width = 16
    height = 16
    rows = []
    for y_value in range(height):
        row = bytearray([0])
        for x_value in range(width):
            row.extend(
                (
                    x_value * 16,
                    y_value * 16,
                    (x_value + y_value) * 8,
                )
            )
        rows.append(bytes(row))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _write_wave(path: Path) -> None:
    sample_rate = 16_000
    sample_count = 1_600
    frames = bytearray()
    for index in range(sample_count):
        value = int(6_000 * math.sin(2 * math.pi * 440 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(bytes(frames))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index in range(10):
        modalities = ["text"]
        assets: list[str] = []
        if 2 <= index <= 4 or index >= 8:
            modalities.append("image")
            assets.append("synthetic-grid.png")
        if 5 <= index <= 7 or index >= 8:
            modalities.append("audio")
            assets.append("synthetic-tone.wav")
        cases.append(
            {
                "case_id": f"npu-smoke-{index:02d}",
                "modalities": modalities,
                "assets": assets,
                "prompt": (
                    "Describe the supplied synthetic signal and state which "
                    "evidence supports the answer."
                ),
            }
        )
    return cases


def build_smoke_assets(output_root: Path) -> dict[str, object]:
    """Create ten deterministic multimodal cases without external data."""

    if output_root.exists():
        raise AssetBuildError("output_exists")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        image_path = temporary / "synthetic-grid.png"
        audio_path = temporary / "synthetic-tone.wav"
        cases_path = temporary / "cases.jsonl"
        image_path.write_bytes(_png_bytes())
        _write_wave(audio_path)
        cases = _cases()
        cases_path.write_text(
            "".join(
                json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
                for case in cases
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": "1.0",
            "sample_count": len(cases),
            "modality_counts": {"audio": 5, "image": 5, "text": 10},
            "artifacts": {
                "cases.jsonl": _sha256(cases_path),
                "synthetic-grid.png": _sha256(image_path),
                "synthetic-tone.wav": _sha256(audio_path),
            },
        }
        (temporary / "smoke_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
