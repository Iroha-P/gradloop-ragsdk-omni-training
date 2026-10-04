"""Deterministic, public-safe synthetic samples for Stage A evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256

from .dataset_schema import AssetSidecar, RubricDimension, TrainingSample
from .source_registry import SyntheticPublicationSource

_RECIPE_VERSION = "stage-a-v1"
_SCENARIO_ORDER = (
    "course_learning",
    "technical_interview",
    "research_defense",
)
_SCENARIO_RECIPES: Mapping[str, Mapping[str, str]] = {
    "course_learning": {
        "modality": "document",
        "sidecar_text": "Synthetic document page with a bounded process diagram.",
        "task_type": "evidence_answer",
        "evidence": (
            "A synthetic process has input {index}, applies transformation "
            "{factor}, and produces output {result}."
        ),
        "prompt": "Explain how the output follows from the supplied process.",
        "follow_up": "Which transformation is essential to the conclusion?",
        "reference_answer": (
            "The output follows from multiplying the stated input by the "
            "listed transformation factor."
        ),
    },
    "technical_interview": {
        "modality": "audio",
        "sidecar_text": "Synthetic audio transcript describing a bounded queue.",
        "task_type": "follow_up",
        "evidence": (
            "A synthetic service accepts {index} requests and uses a bounded "
            "queue of size {result} with deterministic retry."
        ),
        "prompt": "Identify the main reliability trade-off in this design.",
        "follow_up": "How would the conclusion change if the queue were unbounded?",
        "reference_answer": (
            "The bounded queue limits stored work while deterministic retry "
            "keeps handling predictable, at the cost of delaying excess work."
        ),
    },
    "research_defense": {
        "modality": "image",
        "sidecar_text": "Synthetic image OCR describing a controlled comparison.",
        "task_type": "multimodal_interpretation",
        "evidence": (
            "A synthetic experiment reports baseline {index}, treatment "
            "{result}, and one controlled variable."
        ),
        "prompt": "State the supported conclusion and one limitation.",
        "follow_up": "Which additional control would strengthen the claim?",
        "reference_answer": (
            "The treatment can be compared with baseline under the stated "
            "control, but one observation cannot establish a broad causal claim."
        ),
    },
}
_RUBRIC = (
    RubricDimension(
        dimension="evidence_use",
        description="Connects the conclusion to the supplied synthetic evidence.",
        max_score=5,
    ),
    RubricDimension(
        dimension="reasoning",
        description="States a valid conclusion and any relevant limitation.",
        max_score=5,
    ),
    RubricDimension(
        dimension="communication",
        description="Uses clear, direct, and self-contained language.",
        max_score=5,
    ),
)


def _stable_digest(recipe_version: str, scenario: str, index: int) -> str:
    return sha256(f"{recipe_version}:{scenario}:{index}".encode()).hexdigest()


def _identifier(prefix: str, digest: str) -> str:
    return f"{prefix}-{digest}"


def _validate_recipe_inputs(per_scenario: int, recipe_version: str) -> None:
    if type(per_scenario) is not int or not 1 <= per_scenario <= 100:
        raise ValueError("invalid_per_scenario")
    if recipe_version != _RECIPE_VERSION:
        raise ValueError("unsupported_recipe_version")


def _sample_for(
    scenario: str, index: int, recipe_version: str
) -> TrainingSample:
    recipe = _SCENARIO_RECIPES[scenario]
    digest = _stable_digest(recipe_version, scenario, index)
    source_id = _identifier("synthetic-source", digest)
    evidence_id = _identifier("synthetic-evidence", digest)
    provenance_id = _identifier("synthetic-provenance", digest)
    asset_id = _identifier("synthetic-asset", digest)
    modality = recipe["modality"]
    requires_refusal = index % 2 == 0
    sidecar_text = recipe["sidecar_text"]
    sidecar = AssetSidecar(
        asset_id=asset_id,
        modality=modality,
        ocr_text=sidecar_text if modality in {"document", "image"} else "",
        transcript=sidecar_text if modality == "audio" else "",
        metadata_text="Deterministic synthetic sidecar with no binary media.",
    )

    return TrainingSample(
        sample_id=_identifier("synthetic-sample", digest),
        scenario=scenario,
        task_type="refusal" if requires_refusal else recipe["task_type"],
        modalities=("text", modality),
        source_ids=(source_id,),
        evidence_ids=(evidence_id,),
        prompt=(
            "Determine whether the supplied evidence supports the requested claim."
            if requires_refusal
            else recipe["prompt"]
        ),
        reference_answer="" if requires_refusal else recipe["reference_answer"],
        follow_ups=(recipe["follow_up"],),
        rubric=_RUBRIC,
        refusal_reason="insufficient_evidence" if requires_refusal else "",
        provenance_ids=(provenance_id,),
        generation_method="deterministic_synthetic_template",
        quality_status="candidate",
        privacy_status="unscanned",
        asset_sidecars=(sidecar,),
    )


def build_synthetic_samples(
    per_scenario: int, recipe_version: str = _RECIPE_VERSION
) -> tuple[TrainingSample, ...]:
    """Build a balanced, repeatable set of non-personal synthetic samples."""

    _validate_recipe_inputs(per_scenario, recipe_version)
    return tuple(
        _sample_for(scenario, index, recipe_version)
        for scenario in _SCENARIO_ORDER
        for index in range(1, per_scenario + 1)
    )


def _evidence_for_sample(sample: TrainingSample) -> str:
    recipe = _SCENARIO_RECIPES[sample.scenario]
    evidence_id = sample.evidence_ids[0]
    index = next(
        (
            candidate
            for candidate in range(1, 101)
            if evidence_id
            == _identifier(
                "synthetic-evidence",
                _stable_digest(_RECIPE_VERSION, sample.scenario, candidate),
            )
        ),
        None,
    )
    if index is None:
        raise ValueError("invalid_synthetic_sample")
    factor = index + 1
    return recipe["evidence"].format(index=index, factor=factor, result=index * factor)


def synthetic_evidence(samples: Sequence[TrainingSample]) -> dict[str, str]:
    """Return deterministic evidence text keyed by each synthetic evidence ID."""

    evidence: dict[str, str] = {}
    for sample in samples:
        if type(sample) is not TrainingSample or len(sample.evidence_ids) != 1:
            raise ValueError("invalid_synthetic_sample")
        evidence_id = sample.evidence_ids[0]
        if not evidence_id.startswith("synthetic-evidence-"):
            raise ValueError("invalid_synthetic_sample")
        evidence[evidence_id] = _evidence_for_sample(sample)
    return evidence


def synthetic_publication_sources(
    samples: Sequence[TrainingSample],
    recipe_version: str = _RECIPE_VERSION,
) -> dict[str, SyntheticPublicationSource]:
    """Return stable recipe provenance for every generated synthetic source."""

    _validate_recipe_inputs(1, recipe_version)
    sources: dict[str, SyntheticPublicationSource] = {}
    for sample in samples:
        if (
            type(sample) is not TrainingSample
            or len(sample.source_ids) != 1
            or len(sample.provenance_ids) != 1
            or not sample.source_ids[0].startswith("synthetic-source-")
            or not sample.provenance_ids[0].startswith("synthetic-provenance-")
        ):
            raise ValueError("invalid_synthetic_sample")
        recipe_sha256 = sha256(
            (
                f"stage-a-scenario-fixtures:{recipe_version}:"
                f"{sample.scenario}:{sample.generation_method}"
            ).encode()
        ).hexdigest()
        source_id = sample.source_ids[0]
        sources[source_id] = SyntheticPublicationSource(
            source_id=source_id,
            provenance_id=sample.provenance_ids[0],
            recipe_name="stage-a-scenario-fixtures",
            recipe_version=recipe_version,
            recipe_sha256=recipe_sha256,
        )
    if len(sources) != len(samples):
        raise ValueError("invalid_synthetic_sample")
    return sources
