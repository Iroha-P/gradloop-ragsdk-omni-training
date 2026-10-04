"""Fail-closed release checks for Stage A samples."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from .dataset_schema import TrainingSample

_PATTERNS = {
    "email": re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b"),
    "phone": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "identity_number": re.compile(
        r"(?<!\d)\d{6}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[0-9Xx](?!\d)"
    ),
    "absolute_path": re.compile(
        r"(?i)(?:\b[a-z]:[\\/]|\\\\[^\\\s]+\\[^\\\s]+|"
        r"(?:^|\s)/(?:home|root|users|workspace|mnt|tmp|var/tmp)/)"
    ),
    "credential": re.compile(
        r"(?i)(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|"
        r"auth[_ -]?token|password|passwd|"
        r"authorization\s*[:=]?\s*bearer)\s*[:=]?\s*\S+|"
        r"\bAKIA[A-Z0-9]{16}\b|"
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b|"
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
}
_LEXEME_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+")
_SAFE_FINDING_VALUE = re.compile(r"[a-z0-9][a-z0-9._:-]{0,127}")
_TOKEN_CHUNK_SIZE = 4096
_MAX_EVIDENCE_CHARS = 1_048_576


@dataclass(frozen=True)
class ReleaseFinding:
    sample_id: str
    field: str
    code: str


@dataclass(frozen=True)
class ReleaseGateReport:
    sample_id: str
    allowed: bool
    findings: tuple[ReleaseFinding, ...]


class ReleaseGateError(ValueError):
    """A stable, non-sensitive reason why a candidate bundle was blocked."""

    def __init__(self, message: str, findings: Sequence[ReleaseFinding] = ()) -> None:
        super().__init__(message)
        self.findings = tuple(findings)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(lexeme.casefold() for lexeme in _LEXEME_PATTERN.findall(text))


def _token_chunks(
    tokens: tuple[str, ...], *, overlap: int
) -> tuple[tuple[str, ...], ...]:
    if not tokens:
        return ()
    chunks: list[tuple[str, ...]] = []
    start = 0
    while start < len(tokens):
        chunks.append(tokens[start : start + _TOKEN_CHUNK_SIZE])
        if start + _TOKEN_CHUNK_SIZE >= len(tokens):
            break
        start += _TOKEN_CHUNK_SIZE - overlap
    return tuple(chunks)


def _has_near_copy(answer: str, evidence: str) -> bool:
    answer_tokens = _tokens(answer)
    if len(answer_tokens) < 12:
        return False
    evidence_tokens = _tokens(evidence)
    if not evidence_tokens:
        return False

    needed_for_coverage = (len(answer_tokens) * 3 + 4) // 5
    required_run = min(32, max(16, needed_for_coverage))
    answer_windows = {
        answer_tokens[index : index + required_run]
        for index in range(len(answer_tokens) - required_run + 1)
    }
    if not answer_windows:
        return False
    return any(
        chunk[index : index + required_run] in answer_windows
        for chunk in _token_chunks(evidence_tokens, overlap=required_run - 1)
        for index in range(len(chunk) - required_run + 1)
    )


def _scan_text(
    findings: list[ReleaseFinding],
    *,
    sample_id: str,
    field: str,
    value: str,
) -> None:
    for code in release_text_codes(value):
        _add_finding(findings, sample_id, field, code)


def release_text_codes(value: str) -> tuple[str, ...]:
    """Return stable privacy/credential/path codes without retaining text."""

    if type(value) is not str:
        return ("invalid_text",)
    return tuple(
        code for code, pattern in _PATTERNS.items() if pattern.search(value) is not None
    )


def _sample_text_fields(sample: TrainingSample) -> tuple[tuple[str, str], ...]:
    fields: list[tuple[str, str]] = [
        ("prompt", sample.prompt),
        ("reference_answer", sample.reference_answer),
        *(("follow_ups", value) for value in sample.follow_ups),
        *(("rubric.description", item.description) for item in sample.rubric),
        ("refusal_reason", sample.refusal_reason),
    ]
    for sidecar in sample.asset_sidecars:
        fields.extend(
            (
                ("asset_sidecars.ocr_text", sidecar.ocr_text),
                ("asset_sidecars.transcript", sidecar.transcript),
                ("asset_sidecars.metadata_text", sidecar.metadata_text),
            )
        )
    return tuple(fields)


def _add_finding(
    findings: list[ReleaseFinding], sample_id: str, field: str, code: str
) -> None:
    finding = ReleaseFinding(sample_id=sample_id, field=field, code=code)
    if finding not in findings:
        findings.append(finding)


def evaluate_release_candidate(
    sample: TrainingSample,
    evidence_text: Mapping[str, str],
    approved_sources: AbstractSet[str],
) -> ReleaseGateReport:
    """Evaluate one immutable sample without retaining sensitive source text."""

    findings: list[ReleaseFinding] = []
    for field, value in _sample_text_fields(sample):
        _scan_text(findings, sample_id=sample.sample_id, field=field, value=value)

    if sample.privacy_status != "unscanned":
        _add_finding(findings, sample.sample_id, "privacy_status", "unscanned_status")

    for source_id in sample.source_ids:
        if source_id not in approved_sources:
            _add_finding(findings, sample.sample_id, "source_ids", "unknown_source")

    for evidence_id in sample.evidence_ids:
        text = evidence_text.get(evidence_id)
        if type(text) is not str or not text:
            _add_finding(findings, sample.sample_id, "evidence_ids", "missing_evidence")
        elif len(text) > _MAX_EVIDENCE_CHARS:
            _add_finding(
                findings, sample.sample_id, "evidence_text", "invalid_evidence"
            )
        else:
            _scan_text(
                findings,
                sample_id=sample.sample_id,
                field="evidence_text",
                value=text,
            )
            if _has_near_copy(sample.reference_answer, text):
                _add_finding(
                    findings, sample.sample_id, "reference_answer", "near_copy"
                )

    frozen_findings = tuple(findings)
    return ReleaseGateReport(
        sample_id=sample.sample_id,
        allowed=not frozen_findings,
        findings=frozen_findings,
    )


def _safe_finding(finding: object, sample_id: str) -> bool:
    return (
        type(finding) is ReleaseFinding
        and finding.sample_id == sample_id
        and all(
            type(value) is str and _SAFE_FINDING_VALUE.fullmatch(value) is not None
            for value in (finding.sample_id, finding.field, finding.code)
        )
    )


def assert_bundle_releasable(reports: Sequence[ReleaseGateReport]) -> None:
    """Raise a stable error unless every report is allowed and ID is unique."""

    if (
        not isinstance(reports, Sequence)
        or isinstance(reports, (str, bytes))
        or not reports
    ):
        raise ReleaseGateError("bundle_blocked")

    sample_ids: set[str] = set()
    findings: list[ReleaseFinding] = []
    invalid_report = False
    blocked_report = False
    for report in reports:
        if type(report) is not ReleaseGateReport:
            invalid_report = True
            continue
        if (
            type(report.sample_id) is not str
            or _SAFE_FINDING_VALUE.fullmatch(report.sample_id) is None
            or type(report.allowed) is not bool
            or type(report.findings) is not tuple
            or any(
                not _safe_finding(finding, report.sample_id)
                for finding in report.findings
            )
            or report.allowed is bool(report.findings)
        ):
            invalid_report = True
            continue
        if not report.allowed or report.findings:
            blocked_report = True
            findings.extend(report.findings)
        if report.sample_id in sample_ids:
            _add_finding(findings, report.sample_id, "sample_id", "duplicate_id")
        sample_ids.add(report.sample_id)
    if invalid_report or blocked_report or findings:
        raise ReleaseGateError("bundle_blocked", findings)
