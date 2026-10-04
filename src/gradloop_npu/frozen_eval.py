"""Frozen, public-safe Base evaluation for the Stage A evidence set."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from gradloop_eval.metrics import compute_metrics
from gradloop_eval.schema import EvaluationCase, EvaluationPrediction
from gradloop_npu.base_smoke import BackendResult, BaseSmokeError

_PINNED_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCENARIOS = frozenset(
    {"course_learning", "technical_interview", "research_defense"}
)


class FrozenEvalError(ValueError):
    """Stable, non-sensitive failure from the frozen evaluation."""


@dataclass(frozen=True)
class FrozenCase:
    """Validated public-safe evaluation material."""

    case: EvaluationCase
    prompt: str
    evidence: tuple[tuple[str, str], ...]


class TextBackend(Protocol):
    """Text-only model contract."""

    def infer_text(self, prompt: str) -> BackendResult:
        """Run one text prompt."""

    def close(self) -> None:
        """Release model state."""


TextBackendFactory = Callable[[], TextBackend]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FrozenEvalError("frozen_bundle_invalid") from exc


def _read_jsonl(path: Path) -> tuple[object, ...]:
    try:
        return tuple(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    except Exception as exc:
        raise FrozenEvalError("frozen_bundle_invalid") from exc


def _validate_summary(bundle_root: Path) -> None:
    summary = _read_json(bundle_root / "summary.json")
    if (
        type(summary) is not dict
        or summary.get("allowed") is not True
        or summary.get("finding_count") != 0
        or summary.get("source_leakage_count") != 0
        or summary.get("evaluation_case_count") != 5
    ):
        raise FrozenEvalError("frozen_bundle_not_releasable")


def load_frozen_cases(bundle_root: Path) -> tuple[FrozenCase, ...]:
    """Load the immutable five-case Stage A evaluation set."""

    if not bundle_root.is_dir():
        raise FrozenEvalError("frozen_bundle_invalid")
    _validate_summary(bundle_root)
    samples_value = _read_jsonl(bundle_root / "samples.jsonl")
    evidence_value = _read_json(bundle_root / "evidence.json")
    cases_value = _read_jsonl(bundle_root / "evaluation_cases.jsonl")
    if (
        type(evidence_value) is not dict
        or any(type(key) is not str for key in evidence_value)
        or any(type(value) is not str for value in evidence_value.values())
    ):
        raise FrozenEvalError("frozen_bundle_invalid")

    prompts: dict[str, str] = {}
    for value in samples_value:
        if type(value) is not dict:
            raise FrozenEvalError("frozen_bundle_invalid")
        sample_id = value.get("sample_id")
        prompt = value.get("prompt")
        if type(sample_id) is not str or type(prompt) is not str or not prompt:
            raise FrozenEvalError("frozen_bundle_invalid")
        prompts[sample_id] = prompt

    frozen_cases: list[FrozenCase] = []
    for value in cases_value:
        if type(value) is not dict:
            raise FrozenEvalError("frozen_bundle_invalid")
        case_id = value.get("case_id")
        scenario = value.get("scenario")
        relevant = value.get("relevant_evidence_ids")
        requires_refusal = value.get("requires_refusal")
        if (
            type(case_id) is not str
            or case_id not in prompts
            or type(scenario) is not str
            or scenario not in _SCENARIOS
            or type(relevant) is not list
            or any(type(item) is not str for item in relevant)
            or len(relevant) != len(set(relevant))
            or type(requires_refusal) is not bool
            or (requires_refusal and relevant)
            or any(item not in evidence_value for item in relevant)
        ):
            raise FrozenEvalError("frozen_bundle_invalid")
        case = EvaluationCase(
            case_id=case_id,
            scenario=scenario,
            relevant_evidence_ids=tuple(relevant),
            requires_refusal=requires_refusal,
        )
        frozen_cases.append(
            FrozenCase(
                case=case,
                prompt=prompts[case_id],
                evidence=tuple((item, evidence_value[item]) for item in relevant),
            )
        )
    case_ids = tuple(item.case.case_id for item in frozen_cases)
    if len(frozen_cases) != 5 or len(case_ids) != len(set(case_ids)):
        raise FrozenEvalError("frozen_bundle_invalid")
    return tuple(frozen_cases)


def _model_prompt(item: FrozenCase) -> str:
    evidence_lines = (
        "\n".join(f"[{identifier}] {text}" for identifier, text in item.evidence)
        if item.evidence
        else "(no supporting evidence supplied)"
    )
    return (
        "Use only the supplied evidence. If it is insufficient, refuse. "
        "Return one JSON object with exactly these fields: "
        '"answer" (string), "cited_evidence_ids" (array of strings), '
        '"refused" (boolean).\n'
        f"Evidence:\n{evidence_lines}\nQuestion: {item.prompt}"
    )


def _parse_prediction(
    item: FrozenCase, result: BackendResult
) -> tuple[EvaluationPrediction, dict[str, object]]:
    structure_valid = True
    cited_ids: tuple[str, ...] = ()
    refused = False
    try:
        value = json.loads(result.text)
        if (
            type(value) is not dict
            or set(value) != {"answer", "cited_evidence_ids", "refused"}
            or type(value["answer"]) is not str
            or type(value["cited_evidence_ids"]) is not list
            or any(type(identifier) is not str for identifier in value["cited_evidence_ids"])
            or len(value["cited_evidence_ids"])
            != len(set(value["cited_evidence_ids"]))
            or type(value["refused"]) is not bool
        ):
            raise ValueError
        cited_ids = tuple(value["cited_evidence_ids"])
        refused = value["refused"]
    except (json.JSONDecodeError, TypeError, ValueError):
        structure_valid = False

    prediction = EvaluationPrediction(
        case_id=item.case.case_id,
        retrieved_evidence_ids=item.case.relevant_evidence_ids,
        cited_evidence_ids=cited_ids,
        refused=refused,
        structure_valid=structure_valid,
        latency_ms=result.end_to_end_ms,
    )
    record = {
        "case_id": item.case.case_id,
        "scenario": item.case.scenario,
        "requires_refusal": item.case.requires_refusal,
        "status": "completed",
        "output_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        "output_char_count": len(result.text),
        "structure_valid": structure_valid,
        "refused": refused,
        "cited_evidence_ids": list(cited_ids),
        "time_to_first_chunk_ms": result.first_chunk_ms,
        "response_start_method": result.response_start_method,
        "end_to_end_ms": result.end_to_end_ms,
        "peak_memory_bytes": result.peak_memory_bytes,
    }
    return prediction, record


def run_frozen_eval(
    *,
    bundle_root: Path,
    model_revision: str,
    code_commit: str,
    config_sha256: str,
    backend_factory: TextBackendFactory,
    adapter_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Run the five frozen cases and retain only public aggregate evidence."""

    if (
        type(model_revision) is not str
        or _PINNED_REVISION.fullmatch(model_revision) is None
        or type(code_commit) is not str
        or not code_commit
        or type(config_sha256) is not str
        or _SHA256.fullmatch(config_sha256) is None
        or (
            adapter_manifest_sha256 is not None
            and (
                type(adapter_manifest_sha256) is not str
                or _SHA256.fullmatch(adapter_manifest_sha256) is None
            )
        )
    ):
        raise FrozenEvalError("run_identity_invalid")
    cases = load_frozen_cases(bundle_root)
    predictions: list[EvaluationPrediction] = []
    records: list[dict[str, object]] = []
    backend: TextBackend | None = None
    try:
        backend = backend_factory()
        for item in cases:
            try:
                result = backend.infer_text(_model_prompt(item))
            except BaseSmokeError as exc:
                raise FrozenEvalError("backend_failed") from exc
            prediction, record = _parse_prediction(item, result)
            predictions.append(prediction)
            records.append(record)
    except FrozenEvalError:
        raise
    except Exception as exc:
        raise FrozenEvalError("backend_failed") from exc
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                raise FrozenEvalError("backend_close_failed") from exc

    metrics = compute_metrics(
        tuple(item.case for item in cases),
        tuple(predictions),
        recall_k=5,
    )
    peak_memory = max(int(record["peak_memory_bytes"]) for record in records)
    return {
        "schema_version": "1.0",
        "status": "completed",
        "model": {
            "id": "openbmb/MiniCPM-o-4_5",
            "revision": model_revision,
            "dtype": "bfloat16",
            "local_files_only": True,
            "adapter_manifest_sha256": adapter_manifest_sha256,
        },
        "run_identity": {
            "code_commit": code_commit,
            "config_sha256": config_sha256,
            "evaluation_cases_sha256": _sha256(
                bundle_root / "evaluation_cases.jsonl"
            ),
        },
        "privacy": {
            "private_data_allowed": False,
            "prompts_retained": False,
            "outputs_retained": False,
            "paths_retained": False,
        },
        "retrieval_note": "relevant_evidence_ids_are_frozen_inputs_not_model_output",
        "metrics": asdict(metrics),
        "peak_memory_bytes": peak_memory,
        "records": records,
    }


class TorchNpuTextBackend:
    """Text-only wrapper around the official MiniCPM-o Transformers route."""

    def __init__(
        self,
        *,
        model_cache_root: Path,
        device_index: int,
        max_new_tokens: int,
        adapter_root: Path | None = None,
    ) -> None:
        from gradloop_npu.base_smoke import (
            SmokeCase,
            TorchNpuMiniCPMBackend,
        )

        self._case_type = SmokeCase
        self._backend = TorchNpuMiniCPMBackend(
            route="text",
            model_cache_root=model_cache_root,
            device_index=device_index,
            max_new_tokens=max_new_tokens,
            adapter_root=adapter_root,
        )

    def infer_text(self, prompt: str) -> BackendResult:
        case = self._case_type(
            case_id="frozen-eval-runtime",
            modalities=("text",),
            assets=(),
            prompt=prompt,
        )
        return self._backend.infer(case, {})

    def close(self) -> None:
        self._backend.close()
