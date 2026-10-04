from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_data.provenance import ProvenanceRecord
from gradloop_data.source_registry import (
    PublicPublicationSource,
    SourceRegistryEntry,
    SourceRegistryError,
    SyntheticPublicationSource,
    assert_public_training_eligible,
    publication_source_manifest,
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
    with pytest.raises(SourceRegistryError, match="invalid_registry_entry") as exc_info:
        source_registry_entry({"source_id": "safe", "unexpected": "private-value"})

    assert "private-value" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", 1),
        ("allowed_uses", "model_training"),
        ("login_required", 0),
        ("content_risks", ["faces"]),
    ],
)
def test_mapping_rejects_noncanonical_value_types(field: str, value: object) -> None:
    values: dict[str, object] = {
        "source_id": "public-source-001",
        "canonical_url": "https://example.org/course/chapter-1",
        "license_id": "CC0-1.0",
        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
        "attribution": "Example Author",
        "allowed_uses": ("research", "model_training"),
        "modality": "text",
        "login_required": False,
        "paywalled": False,
        "robots_allowed": True,
        "terms_allowed": True,
        "content_risks": (),
    }
    values[field] = value

    with pytest.raises(SourceRegistryError, match="invalid_registry_entry"):
        source_registry_entry(values)


def test_synthetic_recipe_manifest_has_stable_privacy_safe_provenance() -> None:
    source = SyntheticPublicationSource(
        source_id="synthetic-source-001",
        provenance_id="synthetic-provenance-001",
        recipe_name="stage-a-scenario-fixtures",
        recipe_version="stage-a-v1",
        recipe_sha256="a" * 64,
    )

    assert publication_source_manifest({"synthetic-source-001": source}) == (
        {
            "attribution_obligations": "none",
            "license_decision": "synthetic_original",
            "provenance_id": "synthetic-provenance-001",
            "provenance_kind": "synthetic_recipe",
            "recipe_name": "stage-a-scenario-fixtures",
            "recipe_sha256": "a" * 64,
            "recipe_version": "stage-a-v1",
            "source_id": "synthetic-source-001",
        },
    )


def test_public_manifest_revalidates_training_license_and_provenance() -> None:
    registry_entry = entry(
        "CC-BY-4.0", ("research", "model_training", "redistribution")
    )
    source = PublicPublicationSource(
        entry=registry_entry,
        provenance_id="provenance-001",
        provenance=ProvenanceRecord(
            canonical_url=registry_entry.canonical_url,
            retrieved_at="2026-01-01T00:00:00Z",
            content_sha256="b" * 64,
            license_id=registry_entry.license_id,
            license_url=registry_entry.license_url,
            attribution=registry_entry.attribution,
            allowed_uses=registry_entry.allowed_uses,
        ),
    )

    manifest = publication_source_manifest({registry_entry.source_id: source})

    assert manifest == (
        {
            "attribution_obligations": "Example Author",
            "content_sha256": "b" * 64,
            "license_decision": "public_training_eligible",
            "license_id": "CC-BY-4.0",
            "provenance_id": "provenance-001",
            "provenance_kind": "public_content",
            "retrieved_at": "2026-01-01T00:00:00Z",
            "source_id": "public-source-001",
        },
    )


def test_publication_manifest_rejects_non_training_eligible_source() -> None:
    registry_entry = entry(
        "CC-BY-SA-4.0", ("research", "retrieval", "evaluation")
    )
    source = PublicPublicationSource(
        entry=registry_entry,
        provenance_id="provenance-001",
        provenance=ProvenanceRecord(
            canonical_url=registry_entry.canonical_url,
            retrieved_at="2026-01-01T00:00:00Z",
            content_sha256="c" * 64,
            license_id=registry_entry.license_id,
            license_url=registry_entry.license_url,
            attribution=registry_entry.attribution,
            allowed_uses=registry_entry.allowed_uses,
        ),
    )

    with pytest.raises(SourceRegistryError, match="training_not_permitted"):
        publication_source_manifest({registry_entry.source_id: source})


@pytest.mark.parametrize(
    ("field", "unsafe"),
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
def test_public_manifest_rejects_unsafe_free_text_without_echo(
    field: str,
    unsafe: str,
) -> None:
    registry_entry = entry(
        "CC0-1.0", ("research", "model_training", "redistribution")
    )
    if field == "attribution":
        object.__setattr__(registry_entry, "attribution", unsafe)
    source = PublicPublicationSource(
        entry=registry_entry,
        provenance_id="provenance-001",
        provenance=ProvenanceRecord(
            canonical_url=registry_entry.canonical_url,
            retrieved_at=(
                unsafe if field == "retrieved_at" else "2026-01-01T00:00:00Z"
            ),
            content_sha256="d" * 64,
            license_id=registry_entry.license_id,
            license_url=registry_entry.license_url,
            attribution=registry_entry.attribution,
            allowed_uses=registry_entry.allowed_uses,
        ),
    )

    with pytest.raises(SourceRegistryError) as raised:
        publication_source_manifest({registry_entry.source_id: source})

    assert str(raised.value) in {
        "invalid_publication_source",
        "unsafe_publication_source",
    }
    assert unsafe not in repr(raised.value)


@pytest.mark.parametrize(
    "retrieved_at",
    [
        "2026-02-30T00:00:00Z",
        "2026-01-01T00:00:00+00:00",
        "2026-01-01",
    ],
)
def test_public_manifest_requires_canonical_utc_retrieval_timestamp(
    retrieved_at: str,
) -> None:
    registry_entry = entry(
        "CC0-1.0", ("research", "model_training", "redistribution")
    )
    source = PublicPublicationSource(
        entry=registry_entry,
        provenance_id="provenance-001",
        provenance=ProvenanceRecord(
            canonical_url=registry_entry.canonical_url,
            retrieved_at=retrieved_at,
            content_sha256="d" * 64,
            license_id=registry_entry.license_id,
            license_url=registry_entry.license_url,
            attribution=registry_entry.attribution,
            allowed_uses=registry_entry.allowed_uses,
        ),
    )

    with pytest.raises(SourceRegistryError, match="invalid_publication_source"):
        publication_source_manifest({registry_entry.source_id: source})


def test_publication_manifest_rejects_duplicate_provenance_ownership() -> None:
    shared_provenance_id = "synthetic-provenance-shared"
    sources = {
        "synthetic-source-001": SyntheticPublicationSource(
            source_id="synthetic-source-001",
            provenance_id=shared_provenance_id,
            recipe_name="stage-a-scenario-fixtures",
            recipe_version="stage-a-v1",
            recipe_sha256="a" * 64,
        ),
        "synthetic-source-002": SyntheticPublicationSource(
            source_id="synthetic-source-002",
            provenance_id=shared_provenance_id,
            recipe_name="stage-a-scenario-fixtures",
            recipe_version="stage-a-v1",
            recipe_sha256="b" * 64,
        ),
    }

    with pytest.raises(SourceRegistryError, match="invalid_publication_source"):
        publication_source_manifest(sources)


@pytest.mark.parametrize(
    "updates",
    [
        {"source_id": 1},
        {"provenance_id": 1},
        {"recipe_sha256": "not-a-digest"},
    ],
)
def test_synthetic_publication_source_rejects_invalid_structure(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "source_id": "synthetic-source-001",
        "provenance_id": "synthetic-provenance-001",
        "recipe_name": "stage-a-scenario-fixtures",
        "recipe_version": "stage-a-v1",
        "recipe_sha256": "a" * 64,
    }
    values.update(updates)

    with pytest.raises(SourceRegistryError, match="invalid_publication_source"):
        SyntheticPublicationSource(**values)  # type: ignore[arg-type]
