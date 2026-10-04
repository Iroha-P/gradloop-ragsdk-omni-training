"""Build a deterministic, fail-closed public Stage A release bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_data.dataset_schema import SCENARIOS, TrainingSample, sample_to_mapping
from gradloop_data.release_gate import (
    ReleaseGateError,
    ReleaseGateReport,
    assert_bundle_releasable,
    evaluate_release_candidate,
)
from gradloop_data.source_registry import (
    PublicationSource,
    SourceRegistryError,
    publication_source_manifest,
)
from gradloop_data.splits import (
    SplitAssignment,
    assert_no_source_leakage,
    assign_source_connected_splits,
)
from gradloop_data.synthetic import (
    build_synthetic_samples,
    synthetic_evidence,
    synthetic_publication_sources,
)


class ArgumentParseFailure(ValueError):
    """Internal fixed-output signal for unsafe command-line input."""


class SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser that never reflects raw argument values on errors."""

    def error(self, message: str) -> None:
        del message
        raise ArgumentParseFailure


def _build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--synthetic-per-scenario", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _is_child_of(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validated_output(
    output: Path, allowed_output_roots: tuple[Path, ...]
) -> Path:
    if not isinstance(output, Path) or not allowed_output_roots:
        raise ValueError("invalid_arguments")
    resolved_output = (
        output.resolve()
        if output.is_absolute()
        else (REPOSITORY_ROOT / output).resolve()
    )
    resolved_roots = tuple(root.resolve() for root in allowed_output_roots)
    if not any(_is_child_of(resolved_output, root) for root in resolved_roots):
        raise ValueError("invalid_arguments")
    if resolved_output.exists():
        raise ValueError("invalid_arguments")
    return resolved_output


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Sequence[object]) -> None:
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _bundle_summary(
    samples: Sequence[TrainingSample],
    assignments: Sequence[SplitAssignment],
    reports: Sequence[ReleaseGateReport],
    evaluation_cases: Sequence[dict[str, object]],
) -> dict[str, object]:
    scenario_counts: dict[str, int] = {}
    modality_counts: dict[str, int] = {}
    for sample in samples:
        scenario = sample.scenario
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        for modality in sample.modalities:
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
    source_splits: dict[str, str] = {}
    source_leakage_count = 0
    for assignment in assignments:
        for source_id in assignment.source_ids:
            previous = source_splits.setdefault(source_id, assignment.split)
            if previous != assignment.split:
                source_leakage_count += 1
    finding_count = sum(len(report.findings) for report in reports)
    refusal_cases = [
        case for case in evaluation_cases if case["requires_refusal"] is True
    ]
    return {
        "allowed": not finding_count and not source_leakage_count,
        "asset_count": sum(len(sample.asset_sidecars) for sample in samples),
        "evaluation_case_count": len(evaluation_cases),
        "finding_count": finding_count,
        "modality_counts": {
            modality: modality_counts[modality]
            for modality in sorted(modality_counts)
        },
        "refusal_count": sum(sample.task_type == "refusal" for sample in samples),
        "refusal_evaluation_support": {
            "negative": len(evaluation_cases) - len(refusal_cases),
            "positive": len(refusal_cases),
        },
        "sample_count": len(samples),
        "scenario_counts": {
            scenario: scenario_counts[scenario] for scenario in sorted(scenario_counts)
        },
        "source_leakage_count": source_leakage_count,
    }


def _validated_evidence(
    samples: Sequence[TrainingSample], evidence: Mapping[str, str]
) -> dict[str, str]:
    referenced_ids = {
        evidence_id for sample in samples for evidence_id in sample.evidence_ids
    }
    if (
        not isinstance(evidence, Mapping)
        or set(evidence) != referenced_ids
        or any(
            type(evidence_id) is not str
            or type(value) is not str
            or not value
            for evidence_id, value in evidence.items()
        )
    ):
        raise ReleaseGateError("bundle_blocked")
    return {evidence_id: evidence[evidence_id] for evidence_id in sorted(referenced_ids)}


def _validated_source_manifest(
    samples: Sequence[TrainingSample],
    sources: Mapping[str, PublicationSource],
) -> tuple[dict[str, str], ...]:
    referenced_ids = {source_id for sample in samples for source_id in sample.source_ids}
    if set(sources) != referenced_ids:
        raise SourceRegistryError("invalid_publication_source")
    manifest = publication_source_manifest(sources)
    for sample in samples:
        expected_provenance_ids = {
            sources[source_id].provenance_id for source_id in sample.source_ids
        }
        if (
            set(sample.provenance_ids) != expected_provenance_ids
            or len(expected_provenance_ids) != len(sample.source_ids)
        ):
            raise SourceRegistryError("invalid_publication_source")
        for source_id in sample.source_ids:
            if sources[source_id].provenance_id not in sample.provenance_ids:
                raise SourceRegistryError("invalid_publication_source")
    return manifest


def _evaluation_cases(
    samples: Sequence[TrainingSample],
    assignments: Sequence[SplitAssignment],
) -> tuple[dict[str, object], ...]:
    sample_by_id = {sample.sample_id: sample for sample in samples}
    assignment_by_id = {
        assignment.sample_id: assignment for assignment in assignments
    }
    if set(sample_by_id) != set(assignment_by_id):
        raise ReleaseGateError("bundle_blocked")
    cases = tuple(
        {
            "case_id": sample.sample_id,
            "relevant_evidence_ids": (
                [] if sample.task_type == "refusal" else list(sample.evidence_ids)
            ),
            "requires_refusal": sample.task_type == "refusal",
            "scenario": sample.scenario,
        }
        for sample in samples
        if assignment_by_id[sample.sample_id].split == "test"
    )
    evaluation_sources = {
        source_id
        for case in cases
        for source_id in sample_by_id[case["case_id"]].source_ids
    }
    non_test_sources = {
        source_id
        for assignment in assignments
        if assignment.split in {"train", "validation"}
        for source_id in assignment.source_ids
    }
    if evaluation_sources.intersection(non_test_sources):
        raise ReleaseGateError("bundle_blocked")
    return cases


def _asset_manifest(samples: Sequence[TrainingSample]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "asset_id": sidecar.asset_id,
            "modality": sidecar.modality,
            "sample_id": sample.sample_id,
            "sidecar_only": True,
        }
        for sample in sorted(samples, key=lambda item: item.sample_id)
        for sidecar in sorted(sample.asset_sidecars, key=lambda item: item.asset_id)
    )


def _assert_frozen_evaluation_contract(
    evaluation_cases: Sequence[dict[str, object]],
) -> None:
    scenarios = {case["scenario"] for case in evaluation_cases}
    positive_support = sum(
        case["requires_refusal"] is True for case in evaluation_cases
    )
    negative_support = len(evaluation_cases) - positive_support
    if (
        not evaluation_cases
        or scenarios != set(SCENARIOS)
        or not positive_support
        or not negative_support
    ):
        raise ReleaseGateError("bundle_blocked")


def _write_bundle_atomically(
    *,
    output: Path,
    allowed_output_roots: tuple[Path, ...],
    samples: Sequence[TrainingSample],
    evidence: dict[str, str],
    assignments: Sequence[SplitAssignment],
    reports: Sequence[ReleaseGateReport],
    source_manifest: Sequence[dict[str, str]],
    asset_manifest: Sequence[dict[str, object]],
    evaluation_cases: Sequence[dict[str, object]],
    summary: dict[str, object],
) -> None:
    destination = _validated_output(output, allowed_output_roots)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".stage-a-", dir=destination.parent))
    try:
        _write_jsonl(
            temporary / "samples.jsonl",
            [sample_to_mapping(sample) for sample in samples],
        )
        _write_json(
            temporary / "evidence.json",
            {evidence_id: evidence[evidence_id] for evidence_id in sorted(evidence)},
        )
        _write_json(
            temporary / "source_manifest.json",
            {"sources": list(source_manifest)},
        )
        _write_json(
            temporary / "asset_manifest.json",
            {"assets": list(asset_manifest)},
        )
        _write_json(
            temporary / "split_manifest.json",
            {"assignments": [asdict(assignment) for assignment in assignments]},
        )
        _write_json(
            temporary / "release_report.json",
            {
                "reports": [
                    {
                        "allowed": report.allowed,
                        "findings": [asdict(finding) for finding in report.findings],
                        "sample_id": report.sample_id,
                    }
                    for report in reports
                ]
            },
        )
        _write_jsonl(
            temporary / "evaluation_cases.jsonl",
            evaluation_cases,
        )
        _write_json(temporary / "summary.json", summary)
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_bundle(
    output: Path,
    synthetic_per_scenario: int,
    *,
    allowed_output_roots: tuple[Path, ...],
) -> dict[str, object]:
    """Build and atomically publish a verified deterministic synthetic bundle."""

    samples = build_synthetic_samples(synthetic_per_scenario)
    evidence = _validated_evidence(samples, synthetic_evidence(samples))
    try:
        publication_sources = synthetic_publication_sources(samples)
    except ValueError:
        raise ReleaseGateError("bundle_blocked") from None
    source_manifest = _validated_source_manifest(samples, publication_sources)
    approved_sources = frozenset(publication_sources)
    assignments = assign_source_connected_splits(samples, salt="stage-a-v1")
    assert_no_source_leakage(assignments)
    reports = tuple(
        evaluate_release_candidate(sample, evidence, approved_sources) for sample in samples
    )
    assert_bundle_releasable(reports)
    if {report.sample_id for report in reports} != {
        sample.sample_id for sample in samples
    }:
        raise ReleaseGateError("bundle_blocked")
    evaluation_cases = _evaluation_cases(samples, assignments)
    _assert_frozen_evaluation_contract(evaluation_cases)
    asset_manifest = _asset_manifest(samples)
    summary = _bundle_summary(samples, assignments, reports, evaluation_cases)
    _write_bundle_atomically(
        output=output,
        allowed_output_roots=allowed_output_roots,
        samples=samples,
        evidence=evidence,
        assignments=assignments,
        reports=reports,
        source_manifest=source_manifest,
        asset_manifest=asset_manifest,
        evaluation_cases=evaluation_cases,
        summary=summary,
    )
    return summary


def _emit(reason: str) -> int:
    print(json.dumps({"reason": reason}, sort_keys=True, separators=(",", ":")))
    return 2 if reason == "invalid_arguments" else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    allowed_output_roots: tuple[Path, ...] | None = None,
) -> int:
    try:
        args = _build_parser().parse_args(argv)
        if allowed_output_roots is None and args.output.is_absolute():
            raise ArgumentParseFailure
        roots = allowed_output_roots or (
            REPOSITORY_ROOT / "data",
            REPOSITORY_ROOT / "outputs",
            REPOSITORY_ROOT / "dist",
        )
        summary = build_bundle(
            args.output,
            args.synthetic_per_scenario,
            allowed_output_roots=roots,
        )
    except ArgumentParseFailure:
        return _emit("invalid_arguments")
    except (ReleaseGateError, SourceRegistryError):
        return _emit("bundle_blocked")
    except (ValueError, OSError):
        return _emit("invalid_arguments")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
