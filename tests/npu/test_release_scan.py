from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.release_scan import ReleaseScanError, scan_stage_b_reports


def _write_report(path: Path, extra: dict[str, object] | None = None) -> None:
    value: dict[str, object] = {
        "status": "completed",
        "privacy": {
            "private_data_allowed": False,
            "prompts_retained": False,
            "outputs_retained": False,
            "paths_retained": False,
        },
        "records": [
            {
                "case_id": "synthetic-case",
                "output_sha256": "a" * 64,
                "output_char_count": 12,
            }
        ],
    }
    if extra:
        value.update(extra)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_release_scan_accepts_aggregate_evidence(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_report(first)
    _write_report(
        second,
        {
            "privacy": {
                "raw_prompts_included": False,
                "raw_outputs_included": False,
                "paths_included": False,
            }
        },
    )

    report = scan_stage_b_reports((first, second))

    assert report == {
        "schema_version": "1.0",
        "status": "passed",
        "scanned_report_count": 2,
        "finding_count": 0,
        "matched_content_retained": False,
        "paths_retained": False,
    }
    assert str(tmp_path) not in json.dumps(report)


def test_release_scan_accepts_lora_adapter_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "adapter-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "artifact_type": "peft_lora_adapter",
                "optimizer_updates": 1,
                "private_data_allowed": False,
                "full_model_weights_included": False,
                "artifacts": {
                    "adapter_model.safetensors": "a" * 64,
                    "adapter_config.json": "b" * 64,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = scan_stage_b_reports((manifest,))

    assert report["status"] == "passed"
    assert report["scanned_report_count"] == 1
    assert report["finding_count"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        {"raw_output": "private answer"},
        {
            "model_path": (
                "C" + ":" + chr(92) + "private" + chr(92) + "model"
            )
        },
        {"note": "contact test" + "@" + "example.com"},
        {"note": "phone " + "138" + "0013" + "8000"},
        {"note": "identity " + "110101" + "19900101" + "1234"},
        {"note": "api_" + "key" + "=" + "secret-value"},
    ],
)
def test_release_scan_fails_closed(
    tmp_path: Path, extra: dict[str, object]
) -> None:
    report_path = tmp_path / "report.json"
    _write_report(report_path, extra)

    with pytest.raises(ReleaseScanError, match="release_scan_blocked"):
        scan_stage_b_reports((report_path,))
