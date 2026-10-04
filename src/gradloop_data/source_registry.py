"""Immutable, fail-closed declarations for public training sources."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from .provenance import ProvenanceRecord
from .release_gate import release_text_codes
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
_TEXT_FIELDS = frozenset(
    {
        "source_id",
        "canonical_url",
        "license_id",
        "license_url",
        "attribution",
        "modality",
    }
)
_BOOLEAN_FIELDS = frozenset(
    {"login_required", "paywalled", "robots_allowed", "terms_allowed"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RETRIEVED_AT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_MANIFEST_FREE_TEXT_FIELDS = frozenset({"attribution_obligations"})


class SourceRegistryError(ValueError):
    """A stable, non-sensitive reason why a registry entry was rejected."""


def _invalid_registry_entry() -> None:
    raise SourceRegistryError("invalid_registry_entry")


def _validate_values(
    *,
    source_id: object,
    canonical_url: object,
    license_id: object,
    license_url: object,
    attribution: object,
    allowed_uses: object,
    modality: object,
    login_required: object,
    paywalled: object,
    robots_allowed: object,
    terms_allowed: object,
    content_risks: object,
) -> None:
    if any(
        type(value) is not str
        for value in (
            source_id,
            canonical_url,
            license_id,
            license_url,
            attribution,
            modality,
        )
    ):
        _invalid_registry_entry()
    if any(
        type(value) is not bool
        for value in (login_required, paywalled, robots_allowed, terms_allowed)
    ):
        _invalid_registry_entry()
    if (
        type(allowed_uses) is not tuple
        or any(type(value) is not str for value in allowed_uses)
        or type(content_risks) is not tuple
        or any(type(value) is not str for value in content_risks)
    ):
        _invalid_registry_entry()
    if _SOURCE_ID.fullmatch(source_id) is None or modality not in _MODALITIES:
        _invalid_registry_entry()


@dataclass(frozen=True)
class SourceRegistryEntry:
    """A typed declaration of a public source eligible for policy evaluation."""

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
        _validate_values(
            source_id=self.source_id,
            canonical_url=self.canonical_url,
            license_id=self.license_id,
            license_url=self.license_url,
            attribution=self.attribution,
            allowed_uses=self.allowed_uses,
            modality=self.modality,
            login_required=self.login_required,
            paywalled=self.paywalled,
            robots_allowed=self.robots_allowed,
            terms_allowed=self.terms_allowed,
            content_risks=self.content_risks,
        )

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


@dataclass(frozen=True)
class SyntheticPublicationSource:
    """Validated provenance identity for one deterministic synthetic source."""

    source_id: str
    provenance_id: str
    recipe_name: str
    recipe_version: str
    recipe_sha256: str

    def __post_init__(self) -> None:
        if (
            any(
                type(value) is not str
                for value in (
                    self.source_id,
                    self.provenance_id,
                    self.recipe_name,
                    self.recipe_version,
                    self.recipe_sha256,
                )
            )
            or _SOURCE_ID.fullmatch(self.source_id) is None
            or _SOURCE_ID.fullmatch(self.provenance_id) is None
            or _SOURCE_ID.fullmatch(self.recipe_name) is None
            or _SOURCE_ID.fullmatch(self.recipe_version) is None
            or _SHA256.fullmatch(self.recipe_sha256) is None
        ):
            raise SourceRegistryError("invalid_publication_source")


@dataclass(frozen=True)
class PublicPublicationSource:
    """A registry entry joined to immutable retrieval provenance."""

    entry: SourceRegistryEntry
    provenance_id: str
    provenance: ProvenanceRecord

    def __post_init__(self) -> None:
        if (
            type(self.entry) is not SourceRegistryEntry
            or type(self.provenance_id) is not str
            or _SOURCE_ID.fullmatch(self.provenance_id) is None
            or type(self.provenance) is not ProvenanceRecord
        ):
            raise SourceRegistryError("invalid_publication_source")


PublicationSource = Union[SyntheticPublicationSource, PublicPublicationSource]


def source_registry_entry(values: Mapping[str, object]) -> SourceRegistryEntry:
    """Build an immutable entry from a complete, exactly typed mapping."""

    try:
        if set(values) != _FIELDS:
            _invalid_registry_entry()
        if any(type(values[name]) is not str for name in _TEXT_FIELDS):
            _invalid_registry_entry()
        if any(type(values[name]) is not bool for name in _BOOLEAN_FIELDS):
            _invalid_registry_entry()
        if (
            type(values["allowed_uses"]) is not tuple
            or any(type(value) is not str for value in values["allowed_uses"])
            or type(values["content_risks"]) is not tuple
            or any(type(value) is not str for value in values["content_risks"])
        ):
            _invalid_registry_entry()
        return SourceRegistryEntry(
            source_id=values["source_id"],
            canonical_url=values["canonical_url"],
            license_id=values["license_id"],
            license_url=values["license_url"],
            attribution=values["attribution"],
            allowed_uses=values["allowed_uses"],
            modality=values["modality"],
            login_required=values["login_required"],
            paywalled=values["paywalled"],
            robots_allowed=values["robots_allowed"],
            terms_allowed=values["terms_allowed"],
            content_risks=values["content_risks"],
        )
    except (KeyError, TypeError, ValueError):
        _invalid_registry_entry()


def assert_public_training_eligible(
    entry: SourceRegistryEntry,
    policy_path: Path | str | None = None,
) -> None:
    """Raise unless an entry declares and passes public model-training policy."""

    if "model_training" not in entry.allowed_uses:
        raise SourceRegistryError("training_not_permitted")
    decision = dry_run_source(entry.as_source_license(), policy_path=policy_path)
    if not decision.allowed:
        raise SourceRegistryError("source_not_permitted")


def _public_manifest_record(source: PublicPublicationSource) -> dict[str, str]:
    entry = source.entry
    provenance = source.provenance
    assert_public_training_eligible(entry)
    if (
        provenance.canonical_url != entry.canonical_url
        or provenance.license_id != entry.license_id
        or provenance.license_url != entry.license_url
        or provenance.attribution != entry.attribution
        or provenance.allowed_uses != entry.allowed_uses
        or type(provenance.retrieved_at) is not str
        or _RETRIEVED_AT.fullmatch(provenance.retrieved_at) is None
        or type(provenance.content_sha256) is not str
        or _SHA256.fullmatch(provenance.content_sha256) is None
    ):
        raise SourceRegistryError("invalid_publication_source")
    try:
        datetime.strptime(provenance.retrieved_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        raise SourceRegistryError("invalid_publication_source") from None
    return {
        "attribution_obligations": entry.attribution,
        "content_sha256": provenance.content_sha256,
        "license_decision": "public_training_eligible",
        "license_id": entry.license_id,
        "provenance_id": source.provenance_id,
        "provenance_kind": "public_content",
        "retrieved_at": provenance.retrieved_at,
        "source_id": entry.source_id,
    }


def _synthetic_manifest_record(
    source: SyntheticPublicationSource,
) -> dict[str, str]:
    return {
        "attribution_obligations": "none",
        "license_decision": "synthetic_original",
        "provenance_id": source.provenance_id,
        "provenance_kind": "synthetic_recipe",
        "recipe_name": source.recipe_name,
        "recipe_sha256": source.recipe_sha256,
        "recipe_version": source.recipe_version,
        "source_id": source.source_id,
    }


def publication_source_manifest(
    sources: Mapping[str, PublicationSource],
) -> tuple[dict[str, str], ...]:
    """Revalidate publication sources and return a privacy-safe manifest."""

    if (
        not isinstance(sources, Mapping)
        or not sources
        or any(type(source_id) is not str for source_id in sources)
    ):
        raise SourceRegistryError("invalid_publication_source")
    records: list[dict[str, str]] = []
    provenance_ids: set[str] = set()
    for source_id in sorted(sources):
        source = sources[source_id]
        if type(source) is SyntheticPublicationSource:
            if source.source_id != source_id:
                raise SourceRegistryError("invalid_publication_source")
            records.append(_synthetic_manifest_record(source))
        elif type(source) is PublicPublicationSource:
            if source.entry.source_id != source_id:
                raise SourceRegistryError("invalid_publication_source")
            records.append(_public_manifest_record(source))
        else:
            raise SourceRegistryError("invalid_publication_source")
        if source.provenance_id in provenance_ids:
            raise SourceRegistryError("invalid_publication_source")
        provenance_ids.add(source.provenance_id)
        if any(
            release_text_codes(value)
            for field, value in records[-1].items()
            if field in _MANIFEST_FREE_TEXT_FIELDS
        ):
            raise SourceRegistryError("unsafe_publication_source")
    return tuple(records)
