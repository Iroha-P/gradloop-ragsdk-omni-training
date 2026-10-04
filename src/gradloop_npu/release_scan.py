"""Fail-closed privacy scan for Stage B aggregate JSON reports."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
_POSIX_PATH = re.compile(r"(?<![A-Za-z0-9_:/.~-])/(?:home|root|workspace|mnt|tmp)/")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_CN_ID = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
_CREDENTIAL = re.compile(
    r"(?i)\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
    r"\b\s*[:=]\s*\S+"
)
_FORBIDDEN_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "raw_prompt",
        "raw_prompts",
        "raw_output",
        "raw_outputs",
        "output_text",
        "response_text",
        "answer_text",
        "command",
        "environment",
        "hostname",
        "username",
        "working_directory",
        "log_path",
        "model_path",
        "dataset_path",
    }
)


class ReleaseScanError(ValueError):
    """Stable report validation failure."""


def _walk(value: object) -> Iterable[tuple[str | None, object]]:
    if type(value) is dict:
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif type(value) is list:
        for item in value:
            yield None, item
            yield from _walk(item)


def _report_findings(value: object) -> set[str]:
    findings: set[str] = set()
    if type(value) is not dict:
        return {"invalid_report"}
    for key, item in _walk(value):
        if key in _FORBIDDEN_KEYS:
            findings.add("forbidden_field")
        if type(item) is str:
            if _WINDOWS_PATH.search(item) or _POSIX_PATH.search(item):
                findings.add("absolute_path")
            if _EMAIL.search(item):
                findings.add("email")
            if _PHONE.search(item):
                findings.add("phone")
            if _CN_ID.search(item):
                findings.add("identity_number")
            if _CREDENTIAL.search(item):
                findings.add("credential")
    privacy = value.get("privacy")
    adapter_attestation_valid = (
        value.get("artifact_type") == "peft_lora_adapter"
        and value.get("private_data_allowed") is False
        and value.get("full_model_weights_included") is False
    )
    if type(privacy) is not dict:
        if not adapter_attestation_valid:
            findings.add("privacy_attestation_missing")
    else:
        attestation_keys = (
            "private_data_allowed",
            "prompts_retained",
            "outputs_retained",
            "paths_retained",
            "raw_prompts_included",
            "raw_outputs_included",
            "paths_included",
        )
        present = tuple(key for key in attestation_keys if key in privacy)
        if not present or any(privacy[key] is not False for key in present):
            findings.add("privacy_attestation_invalid")
    return findings


def scan_stage_b_reports(report_paths: tuple[Path, ...]) -> dict[str, object]:
    """Scan aggregate reports without returning paths or matched content."""

    if (
        type(report_paths) is not tuple
        or not report_paths
        or any(not isinstance(path, Path) for path in report_paths)
    ):
        raise ReleaseScanError("report_set_invalid")
    findings: set[str] = set()
    for path in report_paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ReleaseScanError("report_invalid") from exc
        findings.update(_report_findings(value))
    if findings:
        raise ReleaseScanError("release_scan_blocked")
    return {
        "schema_version": "1.0",
        "status": "passed",
        "scanned_report_count": len(report_paths),
        "finding_count": 0,
        "matched_content_retained": False,
        "paths_retained": False,
    }
