from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.base_smoke import BackendResult
from gradloop_npu.frozen_eval import FrozenEvalError, run_frozen_eval


def _write_bundle(root: Path) -> None:
    root.mkdir()
    scenarios = (
        "course_learning",
        "technical_interview",
        "research_defense",
        "research_defense",
        "research_defense",
    )
    requires_refusal = (False, True, False, True, True)
    samples: list[dict[str, object]] = []
    cases: list[dict[str, object]] = []
    evidence: dict[str, str] = {}
    for index, (scenario, refusal) in enumerate(zip(scenarios, requires_refusal)):
        case_id = f"case-{index}"
        evidence_id = f"evidence-{index}"
        relevant = [] if refusal else [evidence_id]
        samples.append({"sample_id": case_id, "prompt": f"question {index}"})
        cases.append(
            {
                "case_id": case_id,
                "scenario": scenario,
                "relevant_evidence_ids": relevant,
                "requires_refusal": refusal,
            }
        )
        evidence[evidence_id] = f"synthetic evidence {index}"
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in samples), encoding="utf-8"
    )
    (root / "evaluation_cases.jsonl").write_text(
        "".join(json.dumps(value) + "\n" for value in cases), encoding="utf-8"
    )
    (root / "evidence.json").write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    (root / "summary.json").write_text(
        json.dumps(
            {
                "allowed": True,
                "evaluation_case_count": 5,
                "finding_count": 0,
                "source_leakage_count": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


class _FakeTextBackend:
    def __init__(self) -> None:
        self.closed = False
        self.index = 0

    def infer_text(self, prompt: str) -> BackendResult:
        refusal = "(no supporting evidence supplied)" in prompt
        cited = [] if refusal else [f"evidence-{self.index}"]
        text = json.dumps(
            {"answer": "", "cited_evidence_ids": cited, "refused": refusal}
        )
        self.index += 1
        return BackendResult(
            text=text,
            first_chunk_ms=5,
            end_to_end_ms=10,
            response_start_method="stream_observed",
            peak_memory_bytes=2048,
        )

    def close(self) -> None:
        self.closed = True


def test_frozen_eval_records_metrics_without_raw_content(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    backend = _FakeTextBackend()

    report = run_frozen_eval(
        bundle_root=bundle,
        model_revision="a" * 40,
        code_commit="test-commit",
        config_sha256="b" * 64,
        backend_factory=lambda: backend,
    )

    assert backend.closed is True
    assert report["status"] == "completed"
    assert report["model"]["adapter_manifest_sha256"] is None
    assert report["metrics"]["case_count"] == 5
    assert report["metrics"]["refusal_accuracy"] == 1.0
    assert report["metrics"]["structure_valid_rate"] == 1.0
    assert report["metrics"]["citation_precision"] == 1.0
    assert report["privacy"]["prompts_retained"] is False
    rendered = json.dumps(report)
    assert "question 0" not in rendered
    assert "synthetic evidence" not in rendered
    assert '"answer"' not in rendered
    assert str(tmp_path) not in rendered


def test_frozen_eval_rejects_unreleasable_bundle(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    _write_bundle(bundle)
    summary_path = bundle / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["finding_count"] = 1
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    with pytest.raises(FrozenEvalError, match="frozen_bundle_not_releasable"):
        run_frozen_eval(
            bundle_root=bundle,
            model_revision="a" * 40,
            code_commit="test-commit",
            config_sha256="b" * 64,
            backend_factory=_FakeTextBackend,
        )
