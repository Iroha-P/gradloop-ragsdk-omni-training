"""Deterministic source-connected dataset splits."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

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


def assign_source_connected_splits(
    samples: Sequence[TrainingSample], salt: str
) -> list[SplitAssignment]:
    """Assign each connected source component to one deterministic split."""

    if (
        not isinstance(samples, Sequence)
        or not samples
        or type(salt) is not str
        or any(type(sample) is not TrainingSample for sample in samples)
    ):
        raise SplitError("invalid_split_input")

    sample_ids = [sample.sample_id for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise SplitError("invalid_split_input")

    parent: dict[str, str] = {}

    def find(source_id: str) -> str:
        root = source_id
        while parent[root] != root:
            root = parent[root]
        while parent[source_id] != source_id:
            next_source_id = parent[source_id]
            parent[source_id] = root
            source_id = next_source_id
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for sample in samples:
        if (
            type(sample.source_ids) is not tuple
            or not sample.source_ids
            or any(type(source_id) is not str for source_id in sample.source_ids)
        ):
            raise SplitError("invalid_split_input")
        for source_id in sample.source_ids:
            parent.setdefault(source_id, source_id)
        first_source_id = sample.source_ids[0]
        for source_id in sample.source_ids[1:]:
            union(first_source_id, source_id)

    components: dict[str, list[str]] = {}
    for source_id in parent:
        components.setdefault(find(source_id), []).append(source_id)
    component_splits = {
        root: _bucket(tuple(sorted(source_ids)), salt)
        for root, source_ids in components.items()
    }

    assignments: list[SplitAssignment] = []
    for sample in samples:
        roots = {find(source_id) for source_id in sample.source_ids}
        if len(roots) != 1:
            raise SplitError("invalid_split_input")
        root = roots.pop()
        split = component_splits.get(root)
        if split is None:
            raise SplitError("invalid_split_input")
        assignments.append(
            SplitAssignment(sample.sample_id, tuple(sorted(sample.source_ids)), split)
        )

    return sorted(assignments, key=lambda assignment: assignment.sample_id)


def assert_no_source_leakage(assignments: Sequence[SplitAssignment]) -> None:
    """Raise when a source ID appears in more than one split."""

    seen: dict[str, str] = {}
    for assignment in assignments:
        for source_id in assignment.source_ids:
            previous = seen.setdefault(source_id, assignment.split)
            if previous != assignment.split:
                raise SplitError("source_leakage")
