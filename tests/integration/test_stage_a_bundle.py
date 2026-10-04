from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
from pytest import CaptureFixture

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
BUILD = REPOSITORY_ROOT / "scripts" / "build_stage_a_bundle.py"
EVALUATE = REPOSITORY_ROOT / "scripts" / "evaluate_frozen_set.py"

from gradloop_data.provenance import ProvenanceRecord
from gradloop_data.source_registry import PublicPublicationSource, SourceRegistryEntry


def _load_script(module_name: str, script: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


BUNDLE_CLI = _load_script("stage_a_bundle_test_cli", BUILD)
EVALUATION_CLI = _load_script("frozen_evaluation_test_cli", EVALUATE)


def _write_synthetic_evaluation_inputs(tmp_path: Path) -> tuple[Path, Path]:
    cases = tmp_path / "cases.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    cases.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "case_id": "case-001",
                        "scenario": "course_learning",
                        "relevant_evidence_ids": ["evidence-a"],
                        "requires_refusal": False,
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case-002",
                        "scenario": "research_defense",
                        "relevant_evidence_ids": [],
                        "requires_refusal": True,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "case_id": "case-001",
                        "retrieved_evidence_ids": ["evidence-a"],
                        "cited_evidence_ids": ["evidence-a"],
                        "refused": False,
                        "structure_valid": True,
                        "latency_ms": 100,
                    }
                ),
                json.dumps(
                    {
                        "case_id": "case-002",
                        "retrieved_evidence_ids": [],
                        "cited_evidence_ids": [],
                        "refused": True,
                        "structure_valid": True,
                        "latency_ms": 80,
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return cases, predictions


def test_bundle_builder_creates_releasable_three_scenario_bundle(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    output = tmp_path / "bundle"
    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "20", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 0
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "allowed": True,
        "asset_count": 60,
        "evaluation_case_count": 5,
        "finding_count": 0,
        "modality_counts": {
            "audio": 20,
            "document": 20,
            "image": 20,
            "text": 60,
        },
        "refusal_count": 30,
        "refusal_evaluation_support": {
            "negative": 2,
            "positive": 3,
        },
        "sample_count": 60,
        "scenario_counts": {
            "course_learning": 20,
            "research_defense": 20,
            "technical_interview": 20,
        },
        "source_leakage_count": 0,
    }
    samples = [
        json.loads(line)
        for line in (output / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assignments = {
        item["sample_id"]: item
        for item in json.loads(
            (output / "split_manifest.json").read_text(encoding="utf-8")
        )["assignments"]
    }
    evaluation_cases = [
        json.loads(line)
        for line in (output / "evaluation_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    samples_by_id = {sample["sample_id"]: sample for sample in samples}
    evidence = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
    assert set(evidence) == {
        evidence_id for sample in samples for evidence_id in sample["evidence_ids"]
    }
    release_reports = json.loads(
        (output / "release_report.json").read_text(encoding="utf-8")
    )["reports"]
    assert {report["sample_id"] for report in release_reports} == set(samples_by_id)
    assert all(
        report["allowed"] is True and report["findings"] == []
        for report in release_reports
    )
    assert evaluation_cases
    assert all(assignments[case["case_id"]]["split"] == "test" for case in evaluation_cases)
    evaluation_source_ids = {
        source_id
        for case in evaluation_cases
        for source_id in samples_by_id[case["case_id"]]["source_ids"]
    }
    non_test_source_ids = {
        source_id
        for sample_id, assignment in assignments.items()
        if assignment["split"] in {"train", "validation"}
        for source_id in samples_by_id[sample_id]["source_ids"]
    }
    assert evaluation_source_ids.isdisjoint(non_test_source_ids)
    assets = json.loads(
        (output / "asset_manifest.json").read_text(encoding="utf-8")
    )["assets"]
    assert {asset["modality"] for asset in assets} == {"document", "image", "audio"}
    assert {
        (asset["sample_id"], asset["asset_id"], asset["modality"], asset["sidecar_only"])
        for asset in assets
    } == {
        (
            sample["sample_id"],
            sidecar["asset_id"],
            sidecar["modality"],
            True,
        )
        for sample in samples
        for sidecar in sample["asset_sidecars"]
    }
    source_manifest = json.loads(
        (output / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert len(source_manifest["sources"]) == 60
    assert {
        source["provenance_kind"] for source in source_manifest["sources"]
    } == {"synthetic_recipe"}
    sources_by_id = {
        source["source_id"]: source for source in source_manifest["sources"]
    }
    assert set(sources_by_id) == {
        source_id for sample in samples for source_id in sample["source_ids"]
    }
    assert all(
        sources_by_id[sample["source_ids"][0]]["provenance_id"]
        == sample["provenance_ids"][0]
        for sample in samples
    )
    assert "private" not in captured.out.casefold()


def test_bundle_builder_rejects_unsafe_output_without_echo(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    raw_value = str(tmp_path / "outside" / "sensitive-name")
    result = BUNDLE_CLI.main(
        ["--output", raw_value, "--synthetic-per-scenario", "1"]
    )
    captured = capsys.readouterr()

    assert result == 2
    assert raw_value not in captured.out + captured.err
    assert "invalid_arguments" in captured.out


def test_unsafe_referenced_evidence_cannot_reach_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BUNDLE_CLI.synthetic_evidence
    unsafe = "gh" + "p_" + "D" * 36

    def unsafe_evidence(samples: object) -> dict[str, str]:
        evidence = original(samples)
        evidence[next(iter(evidence))] = unsafe
        return evidence

    monkeypatch.setattr(BUNDLE_CLI, "synthetic_evidence", unsafe_evidence)
    output = tmp_path / "unsafe-bundle"

    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "2", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out) == {"reason": "bundle_blocked"}
    assert unsafe not in captured.out + captured.err
    assert not output.exists()


def test_unreferenced_evidence_is_rejected_before_publication(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BUNDLE_CLI.synthetic_evidence

    def evidence_with_orphan(samples: object) -> dict[str, str]:
        evidence = original(samples)
        evidence["synthetic-evidence-orphan"] = "Unused synthetic evidence."
        return evidence

    monkeypatch.setattr(BUNDLE_CLI, "synthetic_evidence", evidence_with_orphan)
    output = tmp_path / "orphan-bundle"

    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "1", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out) == {"reason": "bundle_blocked"}
    assert not output.exists()


def test_non_training_eligible_registry_source_blocks_publication(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def non_training_sources(samples: object) -> dict[str, PublicPublicationSource]:
        first = samples[0]
        source_id = first.source_ids[0]
        entry = SourceRegistryEntry(
            source_id=source_id,
            canonical_url="https://example.org/synthetic-source",
            license_id="CC-BY-SA-4.0",
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            attribution="Synthetic Example Author",
            allowed_uses=("research", "retrieval", "evaluation"),
            modality="text",
            login_required=False,
            paywalled=False,
            robots_allowed=True,
            terms_allowed=True,
            content_risks=(),
        )
        return {
            source_id: PublicPublicationSource(
                entry=entry,
                provenance_id=first.provenance_ids[0],
                provenance=ProvenanceRecord(
                    canonical_url=entry.canonical_url,
                    retrieved_at="2026-01-01T00:00:00Z",
                    content_sha256="e" * 64,
                    license_id=entry.license_id,
                    license_url=entry.license_url,
                    attribution=entry.attribution,
                    allowed_uses=entry.allowed_uses,
                ),
            )
        }

    monkeypatch.setattr(
        BUNDLE_CLI, "synthetic_publication_sources", non_training_sources, raising=False
    )
    output = tmp_path / "non-training-bundle"

    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "1", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out) == {"reason": "bundle_blocked"}
    assert not output.exists()


@pytest.mark.parametrize(
    ("unsafe_field", "unsafe_value"),
    [
        ("attribution", "Contact synthetic.person@example.org"),
        ("attribution", "authorization bearer synthetic-secret-value"),
        (
            "attribution",
            "Open " + "/" + "root" + "/" + "synthetic-review",
        ),
        (
            "retrieved_at",
            "C" + ":" + "\\" + "Users" + "\\" + "synthetic-review",
        ),
    ],
)
def test_unsafe_public_manifest_metadata_never_reaches_bundle(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    unsafe_field: str,
    unsafe_value: str,
) -> None:
    def unsafe_sources(samples: object) -> dict[str, PublicPublicationSource]:
        sources: dict[str, PublicPublicationSource] = {}
        for sample in samples:
            source_id = sample.source_ids[0]
            entry = SourceRegistryEntry(
                source_id=source_id,
                canonical_url="https://example.org/synthetic-source",
                license_id="CC0-1.0",
                license_url="https://creativecommons.org/publicdomain/zero/1.0/",
                attribution=(
                    unsafe_value
                    if unsafe_field == "attribution"
                    else "Synthetic Example Author"
                ),
                allowed_uses=("research", "model_training", "redistribution"),
                modality="text",
                login_required=False,
                paywalled=False,
                robots_allowed=True,
                terms_allowed=True,
                content_risks=(),
            )
            sources[source_id] = PublicPublicationSource(
                entry=entry,
                provenance_id=sample.provenance_ids[0],
                provenance=ProvenanceRecord(
                    canonical_url=entry.canonical_url,
                    retrieved_at=(
                        unsafe_value
                        if unsafe_field == "retrieved_at"
                        else "2026-01-01T00:00:00Z"
                    ),
                    content_sha256="f" * 64,
                    license_id=entry.license_id,
                    license_url=entry.license_url,
                    attribution=entry.attribution,
                    allowed_uses=entry.allowed_uses,
                ),
            )
        return sources

    monkeypatch.setattr(BUNDLE_CLI, "synthetic_publication_sources", unsafe_sources)
    output = tmp_path / "unsafe-manifest-bundle"

    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "12", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out) == {"reason": "bundle_blocked"}
    assert unsafe_value not in captured.out + captured.err
    assert not output.exists()


def test_orphan_sample_provenance_blocks_publication(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BUNDLE_CLI.build_synthetic_samples
    original_sources = BUNDLE_CLI.synthetic_publication_sources

    def samples_with_orphan(count: int) -> tuple[object, ...]:
        samples = original(count)
        return (
            replace(
                samples[0],
                provenance_ids=(
                    samples[0].provenance_ids[0],
                    "orphan-provenance",
                ),
            ),
            *samples[1:],
        )

    def sources_for_samples_with_orphan(samples: object) -> dict[str, object]:
        sanitized = (
            replace(samples[0], provenance_ids=(samples[0].provenance_ids[0],)),
            *samples[1:],
        )
        return original_sources(sanitized)

    monkeypatch.setattr(BUNDLE_CLI, "build_synthetic_samples", samples_with_orphan)
    monkeypatch.setattr(
        BUNDLE_CLI,
        "synthetic_publication_sources",
        sources_for_samples_with_orphan,
    )
    output = tmp_path / "orphan-provenance-bundle"

    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "12", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out) == {"reason": "bundle_blocked"}
    assert "orphan-provenance" not in captured.out + captured.err
    assert not output.exists()


def test_missing_registry_source_blocks_publication(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = BUNDLE_CLI.synthetic_publication_sources

    def sources_with_missing_record(samples: object) -> dict[str, object]:
        sources = original(samples)
        sources.pop(next(iter(sources)))
        return sources

    monkeypatch.setattr(
        BUNDLE_CLI, "synthetic_publication_sources", sources_with_missing_record
    )
    output = tmp_path / "missing-registry-source-bundle"

    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "17", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out) == {"reason": "bundle_blocked"}
    assert not output.exists()


@pytest.mark.parametrize("per_scenario", [1, 2, 3, 10, 11, 12, 16])
def test_small_counts_without_complete_frozen_refusal_support_fail_closed(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    per_scenario: int,
) -> None:
    output = tmp_path / f"small-count-{per_scenario}"

    result = BUNDLE_CLI.main(
        [
            "--synthetic-per-scenario",
            str(per_scenario),
            "--output",
            str(output),
        ],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out) == {"reason": "bundle_blocked"}
    assert str(output) not in captured.out + captured.err
    assert not output.exists()


def test_smallest_current_count_with_scenarios_and_refusal_support_builds(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    output = tmp_path / "minimum-supported-count"

    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "17", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert result == 0
    assert json.loads(captured.out) == summary
    assert summary["evaluation_case_count"] == 5
    assert summary["refusal_evaluation_support"] == {"negative": 2, "positive": 3}


def test_upper_supported_count_preserves_frozen_contract(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    output = tmp_path / "upper-supported-count"

    result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "100", "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert result == 0
    assert json.loads(captured.out) == summary
    assert summary["sample_count"] == 300
    assert summary["evaluation_case_count"] > 0
    assert summary["refusal_evaluation_support"]["positive"] > 0
    assert summary["refusal_evaluation_support"]["negative"] > 0


def test_clis_reject_existing_outputs_without_changing_sentinels(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    bundle_output = tmp_path / "existing-bundle"
    bundle_output.mkdir()
    bundle_sentinel = bundle_output / "sentinel.txt"
    bundle_sentinel.write_text("preserve", encoding="utf-8")
    bundle_result = BUNDLE_CLI.main(
        ["--synthetic-per-scenario", "17", "--output", str(bundle_output)],
        allowed_output_roots=(tmp_path,),
    )
    bundle_captured = capsys.readouterr()

    cases, predictions = _write_synthetic_evaluation_inputs(tmp_path)
    evaluation_output = tmp_path / "existing-evaluation.json"
    evaluation_output.write_text("preserve", encoding="utf-8")
    evaluation_result = EVALUATION_CLI.main(
        [
            "--cases",
            str(cases),
            "--predictions",
            str(predictions),
            "--output",
            str(evaluation_output),
        ],
        allowed_output_roots=(tmp_path,),
    )
    evaluation_captured = capsys.readouterr()

    assert bundle_result == 2
    assert evaluation_result == 2
    assert json.loads(bundle_captured.out) == {"reason": "invalid_arguments"}
    assert json.loads(evaluation_captured.out) == {"reason": "invalid_arguments"}
    assert bundle_sentinel.read_text(encoding="utf-8") == "preserve"
    assert evaluation_output.read_text(encoding="utf-8") == "preserve"


def test_frozen_evaluation_cli_writes_aggregate_metrics(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    cases, predictions = _write_synthetic_evaluation_inputs(tmp_path)
    output = tmp_path / "aggregate.json"
    result = EVALUATION_CLI.main(
        ["--cases", str(cases), "--predictions", str(predictions), "--output", str(output)],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == ""
    assert captured.err == ""
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "by_scenario": {
            "course_learning": {
                "case_count": 1.0,
                "citation_precision": 1.0,
                "citation_support": 1.0,
                "mean_latency_ms": 100.0,
                "recall_at_k": 1.0,
                "recall_support": 1.0,
                "refusal_accuracy": 0.0,
                "refusal_negative_support": 1.0,
                "refusal_positive_support": 0.0,
                "structure_valid_rate": 1.0,
            },
            "research_defense": {
                "case_count": 1.0,
                "citation_precision": 1.0,
                "citation_support": 1.0,
                "mean_latency_ms": 80.0,
                "recall_at_k": 0.0,
                "recall_support": 0.0,
                "refusal_accuracy": 1.0,
                "refusal_negative_support": 0.0,
                "refusal_positive_support": 1.0,
                "structure_valid_rate": 1.0,
            },
            "technical_interview": {
                "case_count": 0.0,
                "citation_precision": 0.0,
                "citation_support": 0.0,
                "mean_latency_ms": 0.0,
                "recall_at_k": 0.0,
                "recall_support": 0.0,
                "refusal_accuracy": 0.0,
                "refusal_negative_support": 0.0,
                "refusal_positive_support": 0.0,
                "structure_valid_rate": 0.0,
            },
        },
        "case_count": 2,
        "citation_support": 2,
        "citation_precision": 1.0,
        "mean_latency_ms": 90.0,
        "recall_support": 1,
        "recall_at_k": 1.0,
        "refusal_accuracy": 1.0,
        "refusal_negative_support": 1,
        "refusal_positive_support": 1,
        "structure_valid_rate": 1.0,
    }


@pytest.mark.parametrize(
    "invalid_line",
    [
        (
            '{"case_id":"case-001","case_id":"case-001",'
            '"scenario":"course_learning","relevant_evidence_ids":[],'
            '"requires_refusal":false}'
        ),
        (
            '{"case_id":"case-001","scenario":"course_learning",'
            '"relevant_evidence_ids":[],"requires_refusal":NaN}'
        ),
    ],
)
def test_evaluator_rejects_non_strict_json_without_raw_echo(
    tmp_path: Path,
    capsys: CaptureFixture[str],
    invalid_line: str,
) -> None:
    cases, predictions = _write_synthetic_evaluation_inputs(tmp_path)
    cases.write_text(invalid_line + "\n", encoding="utf-8")
    predictions.write_text(
        json.dumps(
            {
                "case_id": "case-001",
                "retrieved_evidence_ids": [],
                "cited_evidence_ids": [],
                "refused": False,
                "structure_valid": True,
                "latency_ms": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "invalid-output.json"

    result = EVALUATION_CLI.main(
        [
            "--cases",
            str(cases),
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ],
        allowed_output_roots=(tmp_path,),
    )
    captured = capsys.readouterr()

    assert result == 1
    assert json.loads(captured.out) == {"reason": "invalid_input"}
    assert invalid_line not in captured.out + captured.err
    assert not output.exists()
