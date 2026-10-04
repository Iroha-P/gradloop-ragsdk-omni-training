# Stage A Public Data and Evaluation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fail-closed, public-safe Stage A pipeline that validates source licenses, represents three training scenarios, prevents source-level leakage, blocks privacy and near-copy risks, creates synthetic fixtures, and produces a frozen evaluation bundle without reading private data or making network calls.

**Architecture:** Extend the existing deny-by-default source policy with an immutable source registry and a standard-library-only dataset model. Candidate samples flow through source-connected split assignment and a release gate before a deterministic bundle builder writes ignored local artifacts; evaluation consumes the same schema and emits aggregate metrics without raw text. This plan changes only the training repository and produces no cloud upload, model training, or private-corpus access.

**Tech Stack:** Python 3.9+, standard library dataclasses/JSON/hashlib/re, pytest, existing Git staged-file scanner.

## Global Constraints

- The public product name is `GradLoop RAGSDK Omni`.
- The Chinese positioning is `证据约束的全模态学习与面试训练 Agent`.
- Supported scenarios are exactly `course_learning`, `technical_interview`, and `research_defense`.
- Stage A performs no implicit network request and accepts no login-required, paywalled, unknown-license, private, or machine-local source.
- Public-weight training eligibility is limited to `CC0-1.0`, `PDM-1.0`, and `CC-BY-4.0`.
- `CC-BY-SA-4.0` is retrieval/evaluation-only in Stage A and must not be marked model-training eligible.
- Real data, JSONL bundles, reports, model files, checkpoints, caches, and outputs stay in Git-ignored directories.
- Committed files are UTF-8 text, smaller than one mebibyte, and must pass `scripts/security/precommit_scan.py --staged`.
- No private corpus, user document, private workspace path, credential, remote, push, cloud upload, Qwen artifact, or MiniCPM weight is used.
- Every task uses TDD and ends with a local commit only.

## File Structure

### New source modules

- `src/gradloop_data/source_registry.py`: immutable source declarations and public-training eligibility.
- `src/gradloop_data/dataset_schema.py`: canonical sample, rubric, and asset-sidecar records.
- `src/gradloop_data/splits.py`: source-connected deterministic train/validation/test assignment.
- `src/gradloop_data/release_gate.py`: privacy, credential, path, license, duplicate, and near-copy findings.
- `src/gradloop_data/synthetic.py`: deterministic public-safe samples for all three scenarios.
- `src/gradloop_eval/schema.py`: frozen evaluation case and prediction records.
- `src/gradloop_eval/metrics.py`: aggregate retrieval, citation, refusal, and structure metrics.

### New command-line entry points

- `scripts/build_stage_a_bundle.py`: validate, split, gate, and write ignored Stage A artifacts.
- `scripts/evaluate_frozen_set.py`: evaluate prediction JSONL against frozen cases.

### New tests

- `tests/data/test_source_registry.py`
- `tests/data/test_dataset_schema.py`
- `tests/data/test_source_splits.py`
- `tests/data/test_release_gate.py`
- `tests/data/test_synthetic_factory.py`
- `tests/eval/test_metrics.py`
- `tests/integration/test_stage_a_bundle.py`

### Documentation

- `docs/data-card.md`
- `docs/stage-a-runbook.md`
- `README.md`
- `SECURITY.md`

---

### Task 1: Public-Training License Capabilities and Source Registry

**Files:**
- Modify: `configs/sources/license-allowlist.yaml`
- Create: `src/gradloop_data/source_registry.py`
- Create: `tests/data/test_source_registry.py`
- Modify: `tests/data/test_web_source_licensing.py`

**Interfaces:**
- Consumes: `gradloop_data.web_sources.SourceLicense`, `load_policy`, and `dry_run_source`.
- Produces: `SourceRegistryEntry`, `source_registry_entry(values)`, and `assert_public_training_eligible(entry, policy_path=None)`.

- [ ] **Step 1: Write failing tests for license capabilities**

```python
from pathlib import Path

import pytest

from gradloop_data.source_registry import (
    SourceRegistryEntry,
    SourceRegistryError,
    assert_public_training_eligible,
    source_registry_entry,
)


def entry(license_id: str, allowed_uses: tuple[str, ...]) -> SourceRegistryEntry:
    return SourceRegistryEntry(
        source_id="public-source-001",
        canonical_url="https://example.org/course/chapter-1",
        license_id=license_id,
        license_url={
            "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
            "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
            "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
        }[license_id],
        attribution="Example Author",
        allowed_uses=allowed_uses,
        modality="text",
        login_required=False,
        paywalled=False,
        robots_allowed=True,
        terms_allowed=True,
        content_risks=(),
    )


@pytest.mark.parametrize("license_id", ["CC0-1.0", "CC-BY-4.0"])
def test_public_training_accepts_permissive_licenses(license_id: str) -> None:
    assert_public_training_eligible(
        entry(license_id, ("research", "model_training", "redistribution"))
    )


def test_public_training_rejects_share_alike_source() -> None:
    with pytest.raises(SourceRegistryError, match="training_not_permitted"):
        assert_public_training_eligible(
            entry("CC-BY-SA-4.0", ("research", "retrieval", "evaluation"))
        )


def test_mapping_rejects_unknown_fields_without_echoing_values() -> None:
    with pytest.raises(SourceRegistryError, match="invalid_registry_entry"):
        source_registry_entry({"source_id": "safe", "unexpected": "private-value"})
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```text
python -m pytest tests/data/test_source_registry.py -q
```

Expected: collection fails because `gradloop_data.source_registry` does not exist.

- [ ] **Step 3: Make `CC-BY-SA-4.0` retrieval/evaluation-only**

Change its `permitted_uses` in `configs/sources/license-allowlist.yaml` to:

```json
[
  "research",
  "retrieval",
  "evaluation",
  "adaptation"
]
```

Keep `model_training` only for `CC0-1.0`, `PDM-1.0`, and `CC-BY-4.0`.

- [ ] **Step 4: Implement the immutable registry**

```python
"""Immutable, fail-closed declarations for public training sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

from .web_sources import SourceLicense, dry_run_source


_SOURCE_ID = re.compile(r"[a-z0-9][a-z0-9._:-]{2,127}")
_MODALITIES = frozenset({"text", "document", "image", "audio", "video"})
_FIELDS = frozenset(
    {
        "source_id",
        "canonical_url",
        "license_id",
        "license_url",
        "attribution",
        "allowed_uses",
        "modality",
        "login_required",
        "paywalled",
        "robots_allowed",
        "terms_allowed",
        "content_risks",
    }
)


class SourceRegistryError(ValueError):
    pass


@dataclass(frozen=True)
class SourceRegistryEntry:
    source_id: str
    canonical_url: str
    license_id: str
    license_url: str
    attribution: str
    allowed_uses: tuple[str, ...]
    modality: str
    login_required: bool
    paywalled: bool
    robots_allowed: bool
    terms_allowed: bool
    content_risks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _SOURCE_ID.fullmatch(self.source_id):
            raise SourceRegistryError("invalid_registry_entry")
        if self.modality not in _MODALITIES:
            raise SourceRegistryError("invalid_registry_entry")

    def as_source_license(self) -> SourceLicense:
        return SourceLicense(
            url=self.canonical_url,
            license_id=self.license_id,
            license_url=self.license_url,
            attribution=self.attribution,
            allowed_uses=self.allowed_uses,
            category="open_licensed",
            login_required=self.login_required,
            paywalled=self.paywalled,
            robots_allowed=self.robots_allowed,
            terms_allowed=self.terms_allowed,
            content_risks=self.content_risks,
        )


def source_registry_entry(values: Mapping[str, object]) -> SourceRegistryEntry:
    if set(values) != _FIELDS:
        raise SourceRegistryError("invalid_registry_entry")
    try:
        return SourceRegistryEntry(
            source_id=values["source_id"],
            canonical_url=values["canonical_url"],
            license_id=values["license_id"],
            license_url=values["license_url"],
            attribution=values["attribution"],
            allowed_uses=tuple(values["allowed_uses"]),
            modality=values["modality"],
            login_required=values["login_required"],
            paywalled=values["paywalled"],
            robots_allowed=values["robots_allowed"],
            terms_allowed=values["terms_allowed"],
            content_risks=tuple(values["content_risks"]),
        )
    except (KeyError, TypeError, ValueError):
        raise SourceRegistryError("invalid_registry_entry") from None


def assert_public_training_eligible(
    entry: SourceRegistryEntry,
    policy_path: Path | str | None = None,
) -> None:
    if "model_training" not in entry.allowed_uses:
        raise SourceRegistryError("training_not_permitted")
    decision = dry_run_source(entry.as_source_license(), policy_path=policy_path)
    if not decision.allowed:
        raise SourceRegistryError("source_not_permitted")
```

Add explicit type checks in `source_registry_entry` so booleans, strings, and tuples cannot be replaced by truthy values or iterated strings. Keep all exception messages stable and non-sensitive.

- [ ] **Step 5: Update existing policy assertions**

In `tests/data/test_web_source_licensing.py`, assert:

```python
assert policy.allowed_licenses["CC-BY-SA-4.0"].permitted_uses == frozenset(
    {"research", "retrieval", "evaluation", "adaptation"}
)
```

- [ ] **Step 6: Run focused and regression tests**

Run:

```text
python -m pytest tests/data/test_source_registry.py tests/data/test_web_source_licensing.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 1**

```text
git add configs/sources/license-allowlist.yaml src/gradloop_data/source_registry.py tests/data/test_source_registry.py tests/data/test_web_source_licensing.py
git commit -m "feat: add public training source registry"
```

---

### Task 2: Canonical Dataset Schema

**Files:**
- Create: `src/gradloop_data/dataset_schema.py`
- Create: `tests/data/test_dataset_schema.py`

**Interfaces:**
- Consumes: source IDs approved by Task 1.
- Produces: `RubricDimension`, `AssetSidecar`, `TrainingSample`, `sample_from_mapping(values)`, and `sample_to_mapping(sample)`.

- [ ] **Step 1: Write failing schema tests**

```python
import pytest

from gradloop_data.dataset_schema import (
    DatasetSchemaError,
    TrainingSample,
    sample_from_mapping,
    sample_to_mapping,
)


VALID = {
    "sample_id": "course-001",
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


def test_valid_sample_round_trips_without_type_drift() -> None:
    sample = sample_from_mapping(VALID)
    assert isinstance(sample, TrainingSample)
    assert sample.scenario == "course_learning"
    assert sample_to_mapping(sample) == VALID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario", "general_chat"),
        ("modalities", ["text", "unknown"]),
        ("source_ids", []),
        ("quality_status", "approved"),
        ("privacy_status", "clean"),
    ],
)
def test_invalid_or_premature_values_fail_closed(field: str, value: object) -> None:
    payload = {**VALID, field: value}
    with pytest.raises(DatasetSchemaError, match="invalid_sample"):
        sample_from_mapping(payload)
```

- [ ] **Step 2: Run tests and verify module-not-found failure**

Run:

```text
python -m pytest tests/data/test_dataset_schema.py -q
```

Expected: collection fails because `dataset_schema.py` does not exist.

- [ ] **Step 3: Implement exact schema types**

```python
"""Canonical Stage A training-sample records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Mapping


SCENARIOS = frozenset(
    {"course_learning", "technical_interview", "research_defense"}
)
MODALITIES = frozenset({"text", "document", "image", "audio", "video"})
TASK_TYPES = frozenset(
    {
        "evidence_answer",
        "follow_up",
        "response_grading",
        "weakness_summary",
        "multimodal_interpretation",
        "refusal",
    }
)
_ID = re.compile(r"[a-z0-9][a-z0-9._:-]{2,127}")
_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "scenario",
        "task_type",
        "modalities",
        "source_ids",
        "evidence_ids",
        "prompt",
        "reference_answer",
        "follow_ups",
        "rubric",
        "refusal_reason",
        "provenance_ids",
        "generation_method",
        "quality_status",
        "privacy_status",
        "asset_sidecars",
    }
)


class DatasetSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class RubricDimension:
    dimension: str
    description: str
    max_score: int


@dataclass(frozen=True)
class AssetSidecar:
    asset_id: str
    modality: str
    ocr_text: str
    transcript: str
    metadata_text: str


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    scenario: str
    task_type: str
    modalities: tuple[str, ...]
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    prompt: str
    reference_answer: str
    follow_ups: tuple[str, ...]
    rubric: tuple[RubricDimension, ...]
    refusal_reason: str
    provenance_ids: tuple[str, ...]
    generation_method: str
    quality_status: str
    privacy_status: str
    asset_sidecars: tuple[AssetSidecar, ...]
```

Implement strict constructors that:

- require exactly `_SAMPLE_FIELDS`;
- reject booleans where integers are expected;
- reject strings where lists are expected;
- require non-empty unique IDs for sources, evidence, and provenance;
- accept only `quality_status="candidate"` and `privacy_status="unscanned"` at ingestion;
- limit each text field to 32,768 characters and each collection to 64 items;
- return only `DatasetSchemaError("invalid_sample")` for malformed content.

`sample_to_mapping` must explicitly convert tuples to lists so the output exactly matches the JSON-compatible input structure.

- [ ] **Step 4: Add asset-sidecar and refusal invariants**

Add tests proving:

```python
def test_refusal_requires_reason_and_no_unsupported_answer() -> None:
    payload = {
        **VALID,
        "task_type": "refusal",
        "reference_answer": "",
        "refusal_reason": "insufficient_evidence",
    }
    assert sample_from_mapping(payload).refusal_reason == "insufficient_evidence"


def test_asset_modality_must_appear_in_sample_modalities() -> None:
    payload = {
        **VALID,
        "asset_sidecars": [
            {
                "asset_id": "asset-001",
                "modality": "image",
                "ocr_text": "synthetic chart",
                "transcript": "",
                "metadata_text": "",
            }
        ],
    }
    with pytest.raises(DatasetSchemaError, match="invalid_sample"):
        sample_from_mapping(payload)
```

- [ ] **Step 5: Run schema tests**

Run:

```text
python -m pytest tests/data/test_dataset_schema.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```text
git add src/gradloop_data/dataset_schema.py tests/data/test_dataset_schema.py
git commit -m "feat: define canonical training sample schema"
```

---

### Task 3: Source-Connected Deterministic Splits

**Files:**
- Create: `src/gradloop_data/splits.py`
- Create: `tests/data/test_source_splits.py`

**Interfaces:**
- Consumes: `Sequence[TrainingSample]`.
- Produces: `SplitAssignment`, `assign_source_connected_splits(samples, salt)`, and `assert_no_source_leakage(assignments)`.

- [ ] **Step 1: Write failing connected-component tests**

```python
from gradloop_data.dataset_schema import sample_from_mapping
from gradloop_data.splits import (
    assign_source_connected_splits,
    assert_no_source_leakage,
)


def make_sample(sample_id: str, source_ids: list[str]) -> object:
    payload = {
        **VALID_SAMPLE_MAPPING,
        "sample_id": sample_id,
        "source_ids": source_ids,
    }
    return sample_from_mapping(payload)


def test_shared_sources_force_samples_into_same_split() -> None:
    samples = [
        make_sample("sample-001", ["source-a", "source-b"]),
        make_sample("sample-002", ["source-b", "source-c"]),
        make_sample("sample-003", ["source-z"]),
    ]
    assignments = assign_source_connected_splits(samples, salt="stage-a-v1")
    by_sample = {item.sample_id: item.split for item in assignments}
    assert by_sample["sample-001"] == by_sample["sample-002"]
    assert_no_source_leakage(assignments)


def test_assignment_is_input_order_independent() -> None:
    samples = [
        make_sample("sample-001", ["source-a"]),
        make_sample("sample-002", ["source-b"]),
    ]
    first = assign_source_connected_splits(samples, salt="stage-a-v1")
    second = assign_source_connected_splits(list(reversed(samples)), salt="stage-a-v1")
    assert sorted(first, key=lambda item: item.sample_id) == sorted(
        second, key=lambda item: item.sample_id
    )
```

Define `VALID_SAMPLE_MAPPING` in the test module as a complete synthetic mapping; do not import fixtures from another test file.

- [ ] **Step 2: Run tests and verify failure**

Run:

```text
python -m pytest tests/data/test_source_splits.py -q
```

Expected: collection fails because `gradloop_data.splits` does not exist.

- [ ] **Step 3: Implement union-find source grouping**

```python
"""Deterministic source-connected dataset splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Sequence

from .dataset_schema import TrainingSample


class SplitError(ValueError):
    pass


@dataclass(frozen=True)
class SplitAssignment:
    sample_id: str
    source_ids: tuple[str, ...]
    split: str


def _bucket(component_sources: tuple[str, ...], salt: str) -> str:
    material = (salt + "\n" + "\n".join(component_sources)).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 100
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "test"
```

Implement a local union-find:

1. create one parent entry for every source ID;
2. union all source IDs appearing in the same sample;
3. materialize sorted source components;
4. hash the complete component with `_bucket`;
5. assign every sample to its component split;
6. return assignments sorted by `sample_id`.

Reject duplicate sample IDs, empty input, non-string salt, and samples whose sources cannot resolve to exactly one component with `SplitError("invalid_split_input")`.

- [ ] **Step 4: Implement explicit leakage assertion**

```python
def assert_no_source_leakage(assignments: Sequence[SplitAssignment]) -> None:
    seen: dict[str, str] = {}
    for assignment in assignments:
        for source_id in assignment.source_ids:
            previous = seen.setdefault(source_id, assignment.split)
            if previous != assignment.split:
                raise SplitError("source_leakage")
```

Add a test that manually creates two assignments with the same source in different splits and expects `SplitError("source_leakage")`.

- [ ] **Step 5: Run split tests**

Run:

```text
python -m pytest tests/data/test_source_splits.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```text
git add src/gradloop_data/splits.py tests/data/test_source_splits.py
git commit -m "feat: prevent source-level dataset leakage"
```

---

### Task 4: Privacy, Secret, Path, and Near-Copy Release Gate

**Files:**
- Create: `src/gradloop_data/release_gate.py`
- Create: `tests/data/test_release_gate.py`

**Interfaces:**
- Consumes: `TrainingSample`, evidence text keyed by evidence ID, and approved source IDs.
- Produces: `ReleaseFinding`, `ReleaseGateReport`, `evaluate_release_candidate(sample, evidence_text, approved_sources)`, and `assert_bundle_releasable(reports)`.

- [ ] **Step 1: Write failing privacy tests**

```python
import pytest

from gradloop_data.dataset_schema import sample_from_mapping
from gradloop_data.release_gate import (
    ReleaseGateError,
    assert_bundle_releasable,
    evaluate_release_candidate,
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
    payload = {**VALID_SAMPLE_MAPPING, field: value}
    sample = sample_from_mapping(payload)
    report = evaluate_release_candidate(
        sample,
        evidence_text={"evidence-001": "Synthetic public evidence."},
        approved_sources={"public-source-001"},
    )
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
    with pytest.raises(ReleaseGateError, match="bundle_blocked"):
        assert_bundle_releasable([report])
```

- [ ] **Step 2: Write failing near-copy and duplicate tests**

```python
def test_long_evidence_copy_is_blocked() -> None:
    copied = " ".join(f"word{index}" for index in range(40))
    payload = {**VALID_SAMPLE_MAPPING, "reference_answer": copied}
    report = evaluate_release_candidate(
        sample_from_mapping(payload),
        evidence_text={"evidence-001": copied},
        approved_sources={"public-source-001"},
    )
    assert "near_copy" in {finding.code for finding in report.findings}


def test_short_technical_phrase_is_not_near_copy() -> None:
    payload = {**VALID_SAMPLE_MAPPING, "reference_answer": "binary search tree"}
    report = evaluate_release_candidate(
        sample_from_mapping(payload),
        evidence_text={"evidence-001": "A binary search tree stores ordered keys."},
        approved_sources={"public-source-001"},
    )
    assert "near_copy" not in {finding.code for finding in report.findings}
```

- [ ] **Step 3: Run release-gate tests and verify failure**

Run:

```text
python -m pytest tests/data/test_release_gate.py -q
```

Expected: collection fails because `release_gate.py` does not exist.

- [ ] **Step 4: Implement stable, non-echoing findings**

```python
"""Fail-closed release checks for Stage A samples."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping, Sequence, AbstractSet

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
        r"(?:^|\s)/(?:home|users|workspace|mnt)/)"
    ),
    "credential": re.compile(
        r"(?i)(?:api[_ -]?key|access[_ -]?token|client[_ -]?secret|"
        r"authorization\s*[:=]?\s*bearer)\s*[:=]?\s*\S+"
    ),
}


class ReleaseGateError(ValueError):
    pass


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
```

Scan `prompt`, `reference_answer`, `follow_ups`, rubric descriptions, `refusal_reason`, and every asset sidecar text field. Do not store matched text, local paths, or raw exceptions in findings.

- [ ] **Step 5: Implement bounded near-copy detection**

Tokenize with:

```python
_LEXEME_PATTERN = re.compile(r"[\u3400-\u9fff]|[A-Za-z0-9_]+")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        lexeme.casefold() for lexeme in _LEXEME_PATTERN.findall(text)[:4096]
    )
```

Implement a rolling dynamic-programming longest common contiguous run using only the previous row. Emit `near_copy` when:

- the longest shared run is at least 32 tokens; or
- the run is at least 16 tokens and covers at least 60% of the answer tokens.

Do not apply near-copy rejection to answers shorter than 12 tokens.

- [ ] **Step 6: Implement source and bundle checks**

`evaluate_release_candidate` must also emit:

- `unknown_source` when a sample source is not approved;
- `missing_evidence` when an evidence ID has no text;
- `duplicate_id` through the bundle-level function when sample IDs repeat;
- `unscanned_status` unless the incoming status is exactly `unscanned`, so malformed pre-approved values cannot bypass scanning.

After all checks pass, return `allowed=True` without mutating the immutable sample. `assert_bundle_releasable` raises only `ReleaseGateError("bundle_blocked")`.

- [ ] **Step 7: Run release-gate tests**

Run:

```text
python -m pytest tests/data/test_release_gate.py -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 4**

```text
git add src/gradloop_data/release_gate.py tests/data/test_release_gate.py
git commit -m "feat: add fail-closed dataset release gate"
```

---

### Task 5: Deterministic Three-Scenario Synthetic Factory

**Files:**
- Create: `src/gradloop_data/synthetic.py`
- Create: `tests/data/test_synthetic_factory.py`

**Interfaces:**
- Consumes: an integer `per_scenario` between 1 and 100 and a fixed recipe version.
- Produces: `build_synthetic_samples(per_scenario, recipe_version="stage-a-v1") -> tuple[TrainingSample, ...]` and `synthetic_evidence(samples) -> dict[str, str]`.

- [ ] **Step 1: Write failing scenario and determinism tests**

```python
from gradloop_data.synthetic import (
    build_synthetic_samples,
    synthetic_evidence,
)


def test_factory_balances_all_three_scenarios() -> None:
    samples = build_synthetic_samples(per_scenario=4)
    counts = {
        scenario: sum(sample.scenario == scenario for sample in samples)
        for scenario in {
            "course_learning",
            "technical_interview",
            "research_defense",
        }
    }
    assert counts == {
        "course_learning": 4,
        "technical_interview": 4,
        "research_defense": 4,
    }


def test_factory_is_byte_stable_for_same_recipe() -> None:
    first = build_synthetic_samples(3, recipe_version="stage-a-v1")
    second = build_synthetic_samples(3, recipe_version="stage-a-v1")
    assert first == second
    assert synthetic_evidence(first) == synthetic_evidence(second)


def test_factory_contains_no_person_or_institution_identity_fields() -> None:
    rendered = repr(build_synthetic_samples(5)).casefold()
    for forbidden in ("email", "phone", "student_id", "identity_card", "school_name"):
        assert forbidden not in rendered
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```text
python -m pytest tests/data/test_synthetic_factory.py -q
```

Expected: collection fails because `synthetic.py` does not exist.

- [ ] **Step 3: Implement deterministic templates**

Use fictional, non-personal evidence families:

```python
_SCENARIO_RECIPES = {
    "course_learning": {
        "task_type": "evidence_answer",
        "evidence": (
            "A synthetic process has input {index}, applies transformation "
            "{factor}, and produces output {result}."
        ),
        "prompt": "Explain how the output follows from the supplied process.",
        "follow_up": "Which transformation is essential to the conclusion?",
    },
    "technical_interview": {
        "task_type": "follow_up",
        "evidence": (
            "A synthetic service accepts {index} requests and uses a bounded "
            "queue of size {result} with deterministic retry."
        ),
        "prompt": "Identify the main reliability trade-off in this design.",
        "follow_up": "How would the conclusion change if the queue were unbounded?",
    },
    "research_defense": {
        "task_type": "multimodal_interpretation",
        "evidence": (
            "A synthetic experiment reports baseline {index}, treatment "
            "{result}, and one controlled variable."
        ),
        "prompt": "State the supported conclusion and one limitation.",
        "follow_up": "Which additional control would strengthen the claim?",
    },
}
```

Generate stable IDs from `sha256(recipe_version + scenario + index)`. Each sample has:

- one synthetic source ID;
- one evidence ID;
- a three-dimension rubric (`evidence_use`, `reasoning`, `communication`);
- `quality_status="candidate"` and `privacy_status="unscanned"`;
- no file, URL, user, school, person, or organization identity.

- [ ] **Step 4: Pass generated samples through the release gate**

Add:

```python
def test_generated_samples_pass_release_gate() -> None:
    samples = build_synthetic_samples(10)
    evidence = synthetic_evidence(samples)
    approved = {source_id for sample in samples for source_id in sample.source_ids}
    reports = [
        evaluate_release_candidate(sample, evidence, approved)
        for sample in samples
    ]
    assert all(report.allowed for report in reports)
    assert_bundle_releasable(reports)
```

- [ ] **Step 5: Run synthetic tests**

Run:

```text
python -m pytest tests/data/test_synthetic_factory.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 5**

```text
git add src/gradloop_data/synthetic.py tests/data/test_synthetic_factory.py
git commit -m "feat: generate public-safe scenario fixtures"
```

---

### Task 6: Frozen Evaluation Schema and Metrics

**Files:**
- Create: `src/gradloop_eval/schema.py`
- Create: `src/gradloop_eval/metrics.py`
- Create: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: `EvaluationCase` and `EvaluationPrediction`.
- Produces: `EvaluationMetrics` and `compute_metrics(cases, predictions)`.

- [ ] **Step 1: Write failing metric tests**

```python
from gradloop_eval.metrics import compute_metrics
from gradloop_eval.schema import EvaluationCase, EvaluationPrediction


def test_metrics_report_retrieval_citation_refusal_and_structure() -> None:
    cases = (
        EvaluationCase(
            case_id="case-001",
            scenario="course_learning",
            relevant_evidence_ids=("evidence-a",),
            requires_refusal=False,
        ),
        EvaluationCase(
            case_id="case-002",
            scenario="research_defense",
            relevant_evidence_ids=(),
            requires_refusal=True,
        ),
    )
    predictions = (
        EvaluationPrediction(
            case_id="case-001",
            retrieved_evidence_ids=("evidence-a", "evidence-x"),
            cited_evidence_ids=("evidence-a",),
            refused=False,
            structure_valid=True,
            latency_ms=120,
        ),
        EvaluationPrediction(
            case_id="case-002",
            retrieved_evidence_ids=(),
            cited_evidence_ids=(),
            refused=True,
            structure_valid=True,
            latency_ms=80,
        ),
    )
    metrics = compute_metrics(cases, predictions, recall_k=5)
    assert metrics.recall_at_k == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.refusal_accuracy == 1.0
    assert metrics.structure_valid_rate == 1.0
    assert metrics.mean_latency_ms == 100.0
```

Add failures for duplicate case IDs, missing predictions, extra predictions, negative latency, invalid scenario, and `recall_k < 1`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```text
python -m pytest tests/eval/test_metrics.py -q
```

Expected: collection fails because `gradloop_eval` does not exist.

- [ ] **Step 3: Implement frozen records**

```python
from dataclasses import dataclass

from gradloop_data.dataset_schema import SCENARIOS


class EvaluationSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    scenario: str
    relevant_evidence_ids: tuple[str, ...]
    requires_refusal: bool

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS or type(self.requires_refusal) is not bool:
            raise EvaluationSchemaError("invalid_evaluation_case")


@dataclass(frozen=True)
class EvaluationPrediction:
    case_id: str
    retrieved_evidence_ids: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...]
    refused: bool
    structure_valid: bool
    latency_ms: int
```

Require unique non-empty IDs, tuple-only collections, booleans of exact type, and non-negative integer latency.

- [ ] **Step 4: Implement aggregate metrics**

```python
@dataclass(frozen=True)
class EvaluationMetrics:
    case_count: int
    recall_at_k: float
    citation_precision: float
    refusal_accuracy: float
    structure_valid_rate: float
    mean_latency_ms: float
    by_scenario: dict[str, dict[str, float]]
```

Metric definitions:

- Recall@k: average fraction of relevant evidence found in the first `k` retrieved IDs; cases with no relevant evidence are excluded.
- Citation precision: supported citations divided by all citations; a case with no citations contributes 1 only when it requires refusal and refused correctly, otherwise 0.
- Refusal accuracy: exact agreement between `requires_refusal` and `refused`.
- Structure-valid rate: fraction with `structure_valid=True`.
- Mean latency: arithmetic mean in milliseconds.
- `by_scenario`: repeat refusal accuracy, structure-valid rate, and case count separately for all three scenarios.

- [ ] **Step 5: Run evaluation tests**

Run:

```text
python -m pytest tests/eval/test_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 6**

```text
git add src/gradloop_eval/schema.py src/gradloop_eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: add frozen evaluation metrics"
```

---

### Task 7: Fail-Closed Stage A Bundle and Evaluation CLIs

**Files:**
- Create: `scripts/build_stage_a_bundle.py`
- Create: `scripts/evaluate_frozen_set.py`
- Create: `tests/integration/test_stage_a_bundle.py`

**Interfaces:**
- Consumes: CLI arguments containing only counts and ignored relative output paths.
- Produces: ignored JSONL samples, evidence JSON, split manifest JSON, release report JSON, evaluation cases JSONL, and aggregate metrics JSON.

- [ ] **Step 1: Write failing end-to-end bundle test**

```python
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BUILD = REPOSITORY_ROOT / "scripts" / "build_stage_a_bundle.py"


def test_bundle_builder_creates_releasable_three_scenario_bundle(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD),
            "--synthetic-per-scenario",
            "6",
            "--output",
            str(output),
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary == {
        "allowed": True,
        "finding_count": 0,
        "sample_count": 18,
        "scenario_counts": {
            "course_learning": 6,
            "research_defense": 6,
            "technical_interview": 6,
        },
        "source_leakage_count": 0,
    }
    assert "private" not in result.stdout.casefold()
```

- [ ] **Step 2: Write failing invalid-argument and no-echo tests**

```python
def test_bundle_builder_rejects_unsafe_output_without_echo(tmp_path: Path) -> None:
    raw_value = str(tmp_path / "outside" / "sensitive-name")
    result = subprocess.run(
        [sys.executable, str(BUILD), "--output", raw_value, "--synthetic-per-scenario", "0"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode != 0
    assert raw_value not in result.stdout + result.stderr
    assert "invalid_arguments" in result.stdout
```

- [ ] **Step 3: Run integration tests and verify failure**

Run:

```text
python -m pytest tests/integration/test_stage_a_bundle.py -q
```

Expected: test fails because the bundle script does not exist.

- [ ] **Step 4: Implement the builder**

The builder must:

1. use `SafeArgumentParser` behavior equivalent to `collect_open_sources.py`;
2. require `1 <= synthetic_per_scenario <= 100`;
3. allow output only below repository-local ignored roots `data/`, `outputs/`, or `dist/`, except tests may pass an injected temporary root through `main(argv, allowed_output_roots=(tmp_path,))`;
4. build synthetic samples and evidence;
5. approve only the synthetic source IDs created in memory;
6. assign source-connected splits;
7. evaluate every release candidate and block the complete bundle on any finding;
8. write UTF-8 JSON with sorted keys and deterministic ordering;
9. write files through a temporary sibling directory and atomically rename only after all checks pass;
10. emit only the fixed summary or a stable reason code, never raw arguments or text.

Use this public function:

```python
def build_bundle(
    output: Path,
    synthetic_per_scenario: int,
    *,
    allowed_output_roots: tuple[Path, ...],
) -> dict[str, object]:
    samples = build_synthetic_samples(synthetic_per_scenario)
    evidence = synthetic_evidence(samples)
    approved_sources = {
        source_id for sample in samples for source_id in sample.source_ids
    }
    assignments = assign_source_connected_splits(samples, salt="stage-a-v1")
    assert_no_source_leakage(assignments)
    reports = tuple(
        evaluate_release_candidate(sample, evidence, approved_sources)
        for sample in samples
    )
    assert_bundle_releasable(reports)
    summary = _bundle_summary(samples, assignments, reports)
    _write_bundle_atomically(
        output=output,
        allowed_output_roots=allowed_output_roots,
        samples=samples,
        evidence=evidence,
        assignments=assignments,
        reports=reports,
        summary=summary,
    )
    return summary
```

Implement `_bundle_summary` with sorted scenario keys and `_write_bundle_atomically` with `tempfile.mkdtemp`, UTF-8 JSON writers, and `Path.replace`. Reject an existing output directory rather than merging or overwriting it.

- [ ] **Step 5: Implement the frozen evaluation CLI**

`scripts/evaluate_frozen_set.py` must:

- accept `--cases`, `--predictions`, and `--output`;
- enforce a 16 MiB limit per input;
- parse one strict JSON object per line;
- call the Task 6 schema constructors and `compute_metrics`;
- atomically write aggregate-only JSON;
- never echo a path, raw prediction, answer, evidence, or exception;
- return `2` for invalid arguments and `1` for invalid input.

Add an integration test with two fully synthetic cases and predictions and assert the aggregate output.

- [ ] **Step 6: Run integration and complete tests**

Run:

```text
python -m pytest tests/integration/test_stage_a_bundle.py -q
python -m pytest -q
```

Expected: integration tests pass, followed by the complete suite passing.

- [ ] **Step 7: Commit Task 7**

```text
git add scripts/build_stage_a_bundle.py scripts/evaluate_frozen_set.py tests/integration/test_stage_a_bundle.py
git commit -m "feat: build and evaluate Stage A bundles"
```

---

### Task 8: Data Card, Runbook, Security Documentation, and Final Verification

**Files:**
- Create: `docs/data-card.md`
- Create: `docs/stage-a-runbook.md`
- Modify: `README.md`
- Modify: `SECURITY.md`

**Interfaces:**
- Consumes: the Task 7 CLI contracts.
- Produces: a reproducible local Stage A workflow and explicit public-release boundary.

- [ ] **Step 1: Write the data card**

Document these exact sections:

```markdown
# GradLoop Stage A Data Card

## Purpose
## Supported Scenarios
## Allowed Licenses
## Excluded Sources
## Modalities
## Sample Schema
## Source-Level Split Policy
## Privacy and Near-Copy Gate
## Known Limitations
## Redistribution Rules
## Reproduction Evidence
```

State that the repository contains no real dataset and that generated bundles remain ignored. Record the three scenario percentages from the design specification and explain that the current deterministic fixtures validate the pipeline rather than claiming production training quality.

- [ ] **Step 2: Write the runbook**

Include copy-ready commands using the existing Anaconda interpreter:

```text
python -m pytest -q
python scripts/build_stage_a_bundle.py --synthetic-per-scenario 20 --output dist/stage-a-public
python scripts/security/precommit_scan.py --staged
```

Explain each generated file, the stable failure codes, and the rule that no file under ignored data/output roots is staged.

- [ ] **Step 3: Update README and SECURITY**

README must show:

- the formal product title;
- Stage A scope;
- the three supported scenarios;
- the safe builder and evaluation commands;
- a warning that Stage B Ascend training is not yet implemented.

SECURITY must add:

- private-corpus and private-workspace exclusion;
- source-license and near-copy gates;
- OCR/transcript/metadata scanning;
- fail-closed bundle semantics;
- a statement that deleting cloud data cannot untrain a model, so privacy rejection occurs before training.

- [ ] **Step 4: Run the complete verification matrix**

Run:

```text
python -m pytest -q
python scripts/build_stage_a_bundle.py --synthetic-per-scenario 20 --output dist/stage-a-public
git status --short
```

Verify:

- the complete test suite has zero failures;
- the bundle summary reports 60 samples, 20 per scenario, zero findings, and zero source leakage;
- `dist/stage-a-public` does not appear in `git status`;
- only intended source and documentation files are uncommitted.

- [ ] **Step 5: Stage intended files and run the repository gate**

```text
git add README.md SECURITY.md docs/data-card.md docs/stage-a-runbook.md
python scripts/security/precommit_scan.py --staged
git diff --cached --name-only
```

Expected: the scanner passes and the staged list contains only the four documentation files.

- [ ] **Step 6: Commit Task 8**

```text
git commit -m "docs: document Stage A public data pipeline"
```

- [ ] **Step 7: Verify the final branch state**

Run:

```text
python -m pytest -q
git status --short
git log -8 --oneline
git remote
```

Expected:

- all tests pass;
- worktree is clean;
- Task 1 through Task 8 commits are present;
- `git remote` produces no entries.

## Plan Self-Review Checklist

- [ ] Every Stage A requirement in the approved design maps to at least one task.
- [ ] No task reads private data, local private paths, user documents, or cloud credentials.
- [ ] No task makes a network call or uploads an artifact.
- [ ] Source IDs, sample IDs, split names, scenario names, and function signatures are consistent across tasks.
- [ ] The license policy excludes CC-BY-SA from public model training.
- [ ] Source-connected splitting happens before bundle creation.
- [ ] Every release finding omits raw matched content.
- [ ] Synthetic fixtures test all three scenarios but are not represented as production-quality training data.
- [ ] Frozen metrics report each scenario separately.
- [ ] The final verification checks tests, ignored artifacts, staged security, clean status, and zero remotes.
