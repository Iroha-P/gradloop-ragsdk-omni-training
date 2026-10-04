from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_data.dataset_schema import sample_from_mapping
from gradloop_data.release_gate import (
    ReleaseFinding,
    ReleaseGateError,
    ReleaseGateReport,
    assert_bundle_releasable,
    evaluate_release_candidate,
)

VALID_SAMPLE_MAPPING = {
    "sample_id": "sample-000",
    "scenario": "course_learning",
    "task_type": "evidence_answer",
    "modalities": ["text"],
    "source_ids": ["public-source-001"],
    "evidence_ids": ["evidence-001"],
    "prompt": "Explain the relationship using the supplied evidence.",
    "reference_answer": "The supplied evidence describes a direct relationship.",
    "follow_ups": ["What assumption supports that conclusion?"],
    "rubric": [
        {
            "dimension": "evidence_use",
            "description": "Uses the supplied evidence.",
            "max_score": 5,
        }
    ],
    "refusal_reason": "",
    "provenance_ids": ["provenance-001"],
    "generation_method": "deterministic_template_v1",
    "quality_status": "candidate",
    "privacy_status": "unscanned",
    "asset_sidecars": [],
}


def evaluate(payload: dict[str, object]):
    return evaluate_release_candidate(
        sample_from_mapping(payload),
        evidence_text={"evidence-001": "Synthetic public evidence."},
        approved_sources={"public-source-001"},
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("prompt", "Contact learner@example.org", "email"),
        ("reference_answer", "Call 13800138000", "phone"),
        ("reference_answer", "Identity 11010519491231002X", "identity_number"),
        (
            "prompt",
            "Open " + "C" + ":" + "\\" + "private" + "\\" + "record.pdf",
            "absolute_path",
        ),
        ("prompt", "authorization bearer synthetic-secret-value", "credential"),
    ],
)
def test_sensitive_text_is_blocked_without_echoing_match(
    field: str, value: str, expected_code: str
) -> None:
    report = evaluate({**VALID_SAMPLE_MAPPING, field: value})

    assert not report.allowed
    assert expected_code in {finding.code for finding in report.findings}
    assert value not in repr(report)


def test_unknown_source_blocks_bundle() -> None:
    sample = sample_from_mapping(VALID_SAMPLE_MAPPING)
    report = evaluate_release_candidate(
        sample,
        evidence_text={"evidence-001": "Synthetic public evidence."},
        approved_sources=set(),
    )

    assert "unknown_source" in {finding.code for finding in report.findings}
    with pytest.raises(ReleaseGateError, match="bundle_blocked"):
        assert_bundle_releasable([report])


def test_long_evidence_copy_is_blocked() -> None:
    copied = " ".join(f"word{index}" for index in range(40))
    report = evaluate_release_candidate(
        sample_from_mapping({**VALID_SAMPLE_MAPPING, "reference_answer": copied}),
        evidence_text={"evidence-001": copied},
        approved_sources={"public-source-001"},
    )

    assert "near_copy" in {finding.code for finding in report.findings}


def test_short_technical_phrase_is_not_near_copy() -> None:
    report = evaluate_release_candidate(
        sample_from_mapping(
            {**VALID_SAMPLE_MAPPING, "reference_answer": "binary search tree"}
        ),
        evidence_text={"evidence-001": "A binary search tree stores ordered keys."},
        approved_sources={"public-source-001"},
    )

    assert "near_copy" not in {finding.code for finding in report.findings}
    assert report.allowed


@pytest.mark.parametrize(
    ("field", "value", "expected_code"),
    [
        ("follow_ups", ["Use learner@example.org for feedback."], "email"),
        (
            "rubric",
            [
                {
                    "dimension": "evidence_use",
                    "description": "Read " + "/" + "home/student/notes",
                    "max_score": 5,
                }
            ],
            "absolute_path",
        ),
        ("refusal_reason", "Call 13800138000", "phone"),
    ],
)
def test_sensitive_text_in_all_textual_sample_fields_is_blocked(
    field: str, value: object, expected_code: str
) -> None:
    payload = {
        **VALID_SAMPLE_MAPPING,
        field: value,
        **({"task_type": "refusal", "reference_answer": ""} if field == "refusal_reason" else {}),
    }

    assert expected_code in {finding.code for finding in evaluate(payload).findings}


def test_sensitive_text_in_asset_sidecar_is_blocked() -> None:
    report = evaluate(
        {
            **VALID_SAMPLE_MAPPING,
            "modalities": ["text", "image"],
            "asset_sidecars": [
                {
                    "asset_id": "asset-001",
                    "modality": "image",
                    "ocr_text": "",
                    "transcript": "",
                    "metadata_text": "authorization bearer synthetic-secret-value",
                }
            ],
        }
    )

    assert "credential" in {finding.code for finding in report.findings}


@pytest.mark.parametrize(
    "value",
    [
        "AK" + "IA" + "A" * 16,
        "gh" + "p_" + "A" * 36,
        "-----BEGIN " + "PRIVATE KEY-----",
        "Open " + "/" + "root/restricted/input.txt",
    ],
)
def test_common_credential_and_local_path_forms_are_blocked(value: str) -> None:
    report = evaluate({**VALID_SAMPLE_MAPPING, "prompt": value})

    assert not report.allowed
    assert {"credential", "absolute_path"}.intersection(
        finding.code for finding in report.findings
    )
    assert value not in repr(report)


def test_referenced_evidence_text_is_scanned_without_echoing_match() -> None:
    unsafe = "gh" + "p_" + "B" * 36
    report = evaluate_release_candidate(
        sample_from_mapping(VALID_SAMPLE_MAPPING),
        evidence_text={"evidence-001": unsafe},
        approved_sources={"public-source-001"},
    )

    assert not report.allowed
    assert ReleaseFinding("sample-000", "evidence_text", "credential") in report.findings
    assert unsafe not in repr(report)


def test_near_copy_after_first_scan_chunk_is_blocked() -> None:
    copied = " ".join(f"copied{index}" for index in range(40))
    late_evidence = " ".join(["padding"] * 4_200) + " " + copied
    report = evaluate_release_candidate(
        sample_from_mapping({**VALID_SAMPLE_MAPPING, "reference_answer": copied}),
        evidence_text={"evidence-001": late_evidence},
        approved_sources={"public-source-001"},
    )

    assert "near_copy" in {finding.code for finding in report.findings}


def test_oversized_evidence_is_rejected_instead_of_partially_scanned() -> None:
    report = evaluate_release_candidate(
        sample_from_mapping(VALID_SAMPLE_MAPPING),
        evidence_text={"evidence-001": "x" * (1_048_576 + 1)},
        approved_sources={"public-source-001"},
    )

    assert "invalid_evidence" in {finding.code for finding in report.findings}


def test_missing_evidence_is_blocked() -> None:
    report = evaluate_release_candidate(
        sample_from_mapping(VALID_SAMPLE_MAPPING),
        evidence_text={},
        approved_sources={"public-source-001"},
    )

    assert "missing_evidence" in {finding.code for finding in report.findings}


def test_repeated_sample_id_blocks_bundle() -> None:
    report = evaluate(VALID_SAMPLE_MAPPING)

    with pytest.raises(ReleaseGateError, match="bundle_blocked") as raised:
        assert_bundle_releasable([report, report])

    assert "duplicate_id" in {finding.code for finding in raised.value.findings}


def test_empty_bundle_report_set_is_blocked() -> None:
    with pytest.raises(ReleaseGateError, match="bundle_blocked"):
        assert_bundle_releasable(())


@pytest.mark.parametrize("reports", [None, (None,), "not-reports"])
def test_malformed_report_container_raises_stable_non_echoing_error(
    reports: object,
) -> None:
    with pytest.raises(ReleaseGateError, match="bundle_blocked") as raised:
        assert_bundle_releasable(reports)  # type: ignore[arg-type]

    assert raised.value.findings == ()
    assert "not-reports" not in repr(raised.value)


def test_malformed_findings_are_not_retained_in_bundle_error() -> None:
    raw_value = "synthetic-private-value"
    report = ReleaseGateReport(
        "sample-000",
        False,
        (raw_value,),  # type: ignore[arg-type]
    )

    with pytest.raises(ReleaseGateError, match="bundle_blocked") as raised:
        assert_bundle_releasable((report,))

    assert raised.value.findings == ()
    assert raw_value not in repr(raised.value)


def test_malformed_release_finding_fields_are_not_retained() -> None:
    raw_value = "synthetic-private-value"
    report = ReleaseGateReport(
        "sample-000",
        False,
        (ReleaseFinding(raw_value, raw_value, raw_value),),
    )

    with pytest.raises(ReleaseGateError, match="bundle_blocked") as raised:
        assert_bundle_releasable((report,))

    assert raised.value.findings == ()
    assert raw_value not in repr(raised.value)


@pytest.mark.parametrize(
    "report",
    [
        ReleaseGateReport("", True, ()),
        ReleaseGateReport("sample-000", 1, ()),  # type: ignore[arg-type]
        ReleaseGateReport(
            "sample-000",
            False,
            (ReleaseFinding("other-sample", "prompt", "email"),),
        ),
    ],
)
def test_structurally_invalid_bundle_reports_are_blocked(
    report: ReleaseGateReport,
) -> None:
    with pytest.raises(ReleaseGateError, match="bundle_blocked"):
        assert_bundle_releasable((report,))


@pytest.mark.parametrize(
    "report",
    [
        ReleaseGateReport("sample-000", False, ()),
        ReleaseGateReport(
            "sample-000",
            True,
            (ReleaseFinding("sample-000", "prompt", "email"),),
        ),
    ],
)
def test_inconsistent_report_cannot_bypass_bundle_gate(
    report: ReleaseGateReport,
) -> None:
    with pytest.raises(ReleaseGateError, match="bundle_blocked"):
        assert_bundle_releasable([report])


def test_non_unscanned_status_is_blocked() -> None:
    sample = sample_from_mapping(VALID_SAMPLE_MAPPING)
    object.__setattr__(sample, "privacy_status", "clean")

    report = evaluate_release_candidate(
        sample,
        evidence_text={"evidence-001": "Synthetic public evidence."},
        approved_sources={"public-source-001"},
    )

    assert "unscanned_status" in {finding.code for finding in report.findings}


def test_clean_candidate_is_allowed_and_not_mutated() -> None:
    sample = sample_from_mapping(VALID_SAMPLE_MAPPING)
    report = evaluate_release_candidate(
        sample,
        evidence_text={"evidence-001": "Synthetic public evidence."},
        approved_sources={"public-source-001"},
    )

    assert report.allowed
    assert sample.privacy_status == "unscanned"
