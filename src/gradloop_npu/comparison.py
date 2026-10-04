"""Deterministic Base-versus-LoRA comparison without raw model content."""

from __future__ import annotations

import json
from pathlib import Path

_SCORE_KEYS = ("citation_precision", "refusal_accuracy", "structure_valid_rate")


class ComparisonError(ValueError):
    """Stable comparison failure."""


def _read_report(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ComparisonError("evaluation_report_invalid") from exc
    if type(value) is not dict:
        raise ComparisonError("evaluation_report_invalid")
    return value


def _metrics(report: dict[str, object]) -> dict[str, float]:
    metrics = report.get("metrics")
    if type(metrics) is not dict:
        raise ComparisonError("evaluation_report_invalid")
    values: dict[str, float] = {}
    for key in _SCORE_KEYS:
        value = metrics.get(key)
        if type(value) not in {int, float} or not 0 <= float(value) <= 1:
            raise ComparisonError("evaluation_report_invalid")
        values[key] = float(value)
    return values


def compare_base_lora(
    *,
    base_report_path: Path,
    lora_report_path: Path,
    minimum_absolute_gain: float = 0.05,
) -> dict[str, object]:
    """Apply the frozen composite promotion rule."""

    if (
        type(minimum_absolute_gain) is not float
        or not 0 <= minimum_absolute_gain <= 1
    ):
        raise ComparisonError("comparison_config_invalid")
    base = _read_report(base_report_path)
    lora = _read_report(lora_report_path)
    for report in (base, lora):
        if (
            report.get("status") != "completed"
            or type(report.get("model")) is not dict
            or type(report.get("run_identity")) is not dict
            or type(report.get("privacy")) is not dict
            or report["privacy"].get("private_data_allowed") is not False
            or report["privacy"].get("prompts_retained") is not False
            or report["privacy"].get("outputs_retained") is not False
        ):
            raise ComparisonError("evaluation_report_invalid")
    if (
        base["model"].get("revision") != lora["model"].get("revision")
        or base["run_identity"].get("evaluation_cases_sha256")
        != lora["run_identity"].get("evaluation_cases_sha256")
        or base["model"].get("adapter_manifest_sha256") is not None
        or type(lora["model"].get("adapter_manifest_sha256")) is not str
    ):
        raise ComparisonError("evaluation_identity_mismatch")

    base_metrics = _metrics(base)
    lora_metrics = _metrics(lora)
    base_score = sum(base_metrics.values()) / len(_SCORE_KEYS)
    lora_score = sum(lora_metrics.values()) / len(_SCORE_KEYS)
    absolute_gain = lora_score - base_score
    relative_gain = absolute_gain / base_score if base_score else 0.0
    promote = absolute_gain >= minimum_absolute_gain
    return {
        "schema_version": "1.0",
        "status": "evaluated",
        "composite_definition": {
            "metrics": list(_SCORE_KEYS),
            "aggregation": "unweighted_mean",
            "minimum_absolute_gain": minimum_absolute_gain,
        },
        "base": {"metrics": base_metrics, "composite_score": base_score},
        "lora": {"metrics": lora_metrics, "composite_score": lora_score},
        "absolute_gain": absolute_gain,
        "relative_gain": relative_gain,
        "promotion_decision": "promote_lora" if promote else "retain_base",
        "privacy": {
            "raw_prompts_included": False,
            "raw_outputs_included": False,
            "paths_included": False,
        },
    }
