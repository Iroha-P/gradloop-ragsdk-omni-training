"""Immutable provenance metadata for explicitly licensed public content."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProvenanceRecord:
    """Audit record emitted after an approved source is retrieved."""

    canonical_url: str
    retrieved_at: str
    content_sha256: str
    license_id: str
    license_url: str
    attribution: str
    allowed_uses: tuple[str, ...]
