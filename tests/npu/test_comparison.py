from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.comparison import ComparisonError, compare_base_lora


def _write_report(
    path: Path,
    *,
    adapter: str | None,
    values: tuple[float, float, float],
    revision: str = "a" * 40,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "model": {
                    "revision": revision,
                    "adapter_manifest_sha256": adapter,
                },
                "run_identity": {"evaluation_cases_sha256": "b" * 64},
                "privacy": {
                    "private_data_allowed": False,
                    "prompts_retained": False,
                    "outputs_retained": False,
                },
                "metrics": {
                    "citation_precision": values[0],
                    "refusal_accuracy": values[1],
                    "structure_valid_rate": values[2],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_comparison_promotes_only_with_five_point_gain(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    lora = tmp_path / "lora.json"
    _write_report(base, adapter=None, values=(0.7, 0.7, 0.7))
    _write_report(lora, adapter="c" * 64, values=(0.8, 0.8, 0.8))

    report = compare_base_lora(
        base_report_path=base,
        lora_report_path=lora,
    )

    assert report["promotion_decision"] == "promote_lora"
    assert report["absolute_gain"] == pytest.approx(0.1)
    assert report["privacy"]["raw_outputs_included"] is False
    assert str(tmp_path) not in json.dumps(report)


def test_comparison_rejects_identity_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    lora = tmp_path / "lora.json"
    _write_report(base, adapter=None, values=(0.7, 0.7, 0.7))
    _write_report(
        lora,
        adapter="c" * 64,
        values=(0.8, 0.8, 0.8),
        revision="d" * 40,
    )

    with pytest.raises(ComparisonError, match="evaluation_identity_mismatch"):
        compare_base_lora(base_report_path=base, lora_report_path=lora)
