"""Deterministic, public-safe LoRA feasibility dataset generation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from gradloop_npu.base_smoke import SmokeCase, load_smoke_cases


class LoraAssetError(ValueError):
    """Stable failure raised before publishing LoRA feasibility inputs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assistant_answer(case: SmokeCase) -> str:
    statements = ["The request uses deterministic synthetic evidence only."]
    if "image" in case.modalities:
        statements.append("The image is a small RGB gradient grid.")
    if "audio" in case.modalities:
        statements.append("The audio is a short synthetic 440 Hz tone without speech.")
    if case.modalities == ("text",):
        statements.append("No image or audio evidence is supplied.")
    statements.append("No claim about a real person or private record is supported.")
    return " ".join(statements)


def _training_row(case: SmokeCase) -> dict[str, object]:
    tags = ""
    row: dict[str, object] = {}
    if "image" in case.modalities:
        tags += "<image>"
        row["images"] = ["synthetic-grid.png"]
    if "audio" in case.modalities:
        tags += "<audio>"
        row["audios"] = ["synthetic-tone.wav"]
    row["messages"] = [
        {
            "role": "system",
            "content": (
                "Use only supplied synthetic evidence. Do not infer personal or "
                "real-world facts."
            ),
        },
        {
            "role": "user",
            "content": f"{tags}{case.prompt}",
        },
        {
            "role": "assistant",
            "content": _assistant_answer(case),
        },
    ]
    return row


def build_lora_assets(
    *,
    smoke_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Create a ten-sample SWIFT dataset without external or private content."""

    if output_root.exists():
        raise LoraAssetError("output_exists")
    _, cases = load_smoke_cases(smoke_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        for name in ("synthetic-grid.png", "synthetic-tone.wav"):
            shutil.copyfile(smoke_root / name, temporary / name)
        dataset_path = temporary / "train.jsonl"
        dataset_path.write_text(
            "".join(
                json.dumps(
                    _training_row(case),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for case in cases
            ),
            encoding="utf-8",
        )
        artifacts = {
            name: _sha256(temporary / name)
            for name in (
                "train.jsonl",
                "synthetic-grid.png",
                "synthetic-tone.wav",
            )
        }
        manifest = {
            "schema_version": "1.0",
            "sample_count": len(cases),
            "modality_counts": {"audio": 5, "image": 5, "text": 10},
            "source": "deterministic_synthetic_smoke_bundle",
            "private_data_allowed": False,
            "artifacts": artifacts,
        }
        (temporary / "lora_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
