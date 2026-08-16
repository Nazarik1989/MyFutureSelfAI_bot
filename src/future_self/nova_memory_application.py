from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from .nova_memory import NOVA_MEMORY_CATEGORIES, NovaMemoryCategory

NOVA_MEMORY_APPLICATION_MAX_ITEMS = 12
NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES = 8 * 1024

_CATEGORY_PRIORITY: tuple[NovaMemoryCategory, ...] = (
    "interaction",
    "orientation",
    "about_me",
)
_CATEGORY_RANK = {category: rank for rank, category in enumerate(_CATEGORY_PRIORITY)}


class NovaMemoryApplicationItem(Protocol):
    """The storage-independent fields required to build an answer projection."""

    public_id: str
    category: NovaMemoryCategory
    content: str
    important: bool
    updated_at: datetime


class NovaMemoryProjectionError(ValueError):
    """A privacy-safe failure to construct a confirmed-memory projection."""


@dataclass(frozen=True, slots=True)
class ConfirmedMemoryRecord:
    """One immutable record whose public fields exactly match the provider contract."""

    category: NovaMemoryCategory
    important: bool
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.category, str)
            or self.category not in NOVA_MEMORY_CATEGORIES
            or not isinstance(self.important, bool)
            or not isinstance(self.content, str)
            or not self.content
        ):
            raise NovaMemoryProjectionError("Invalid confirmed-memory record.")
        try:
            self.content.encode("utf-8")
        except UnicodeError:
            raise NovaMemoryProjectionError("Invalid confirmed-memory record.") from None

    def provider_payload(self) -> dict[str, object]:
        return {
            "category": self.category,
            "important": self.important,
            "content": self.content,
        }


def serialize_confirmed_memory(records: Iterable[ConfirmedMemoryRecord]) -> str:
    """Serialize the exact provider array with the adapter's compact JSON settings."""
    return json.dumps(
        [record.provider_payload() for record in records],
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class NovaMemoryProjection:
    """A bounded transient projection; its repr contains privacy-safe metrics only."""

    records: tuple[ConfirmedMemoryRecord, ...] = field(repr=False)
    selected_count: int
    omitted_count: int
    important_count: int
    payload_bytes: int
    collection_revision: str = field(repr=False)

    def __post_init__(self) -> None:
        safe_integer_metrics = all(
            type(value) is int
            for value in (
                self.selected_count,
                self.omitted_count,
                self.important_count,
                self.payload_bytes,
            )
        )
        if (
            not isinstance(self.records, tuple)
            or any(not isinstance(record, ConfirmedMemoryRecord) for record in self.records)
            or not isinstance(self.collection_revision, str)
            or not self.collection_revision
            or not safe_integer_metrics
            or self.selected_count != len(self.records)
            or self.omitted_count < 0
            or self.important_count != sum(record.important for record in self.records)
            or self.payload_bytes != len(serialize_confirmed_memory(self.records).encode("utf-8"))
            or self.selected_count > NOVA_MEMORY_APPLICATION_MAX_ITEMS
            or self.payload_bytes > NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES
        ):
            raise NovaMemoryProjectionError("Invalid confirmed-memory projection.")

    def __bool__(self) -> bool:
        return bool(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def provider_payload(self) -> list[dict[str, object]]:
        return [record.provider_payload() for record in self.records]

    def provider_json(self) -> str:
        return serialize_confirmed_memory(self.records)


@dataclass(frozen=True, slots=True)
class _Candidate:
    public_id: str = field(repr=False)
    category: NovaMemoryCategory
    content: str = field(repr=False)
    important: bool
    updated_at: datetime = field(repr=False)

    def record(self) -> ConfirmedMemoryRecord:
        return ConfirmedMemoryRecord(
            category=self.category,
            important=self.important,
            content=self.content,
        )


def build_nova_memory_projection(
    items: Iterable[NovaMemoryApplicationItem],
    *,
    collection_revision: str,
) -> NovaMemoryProjection:
    """Select and serialize a stable, whole-record confirmed-memory projection."""
    if not isinstance(collection_revision, str) or not collection_revision:
        raise NovaMemoryProjectionError("Invalid collection revision.")

    candidates = tuple(_candidate(item) for item in items)
    if len({candidate.public_id for candidate in candidates}) != len(candidates):
        raise NovaMemoryProjectionError("Invalid confirmed-memory collection.")

    priority_order = _selection_order(candidates)
    selected: list[ConfirmedMemoryRecord] = []
    payload_bytes = len(serialize_confirmed_memory(selected).encode("utf-8"))
    for candidate in priority_order:
        if len(selected) >= NOVA_MEMORY_APPLICATION_MAX_ITEMS:
            break
        next_records = (*selected, candidate.record())
        next_bytes = len(serialize_confirmed_memory(next_records).encode("utf-8"))
        if next_bytes > NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES:
            break
        selected.append(next_records[-1])
        payload_bytes = next_bytes

    records = tuple(selected)
    return NovaMemoryProjection(
        records=records,
        selected_count=len(records),
        omitted_count=len(candidates) - len(records),
        important_count=sum(record.important for record in records),
        payload_bytes=payload_bytes,
        collection_revision=collection_revision,
    )


def _candidate(item: NovaMemoryApplicationItem) -> _Candidate:
    try:
        public_id = item.public_id
        category = item.category
        content = item.content
        important = item.important
        updated_at = item.updated_at
    except (AttributeError, TypeError):
        raise NovaMemoryProjectionError("Invalid confirmed-memory item.") from None

    if (
        not isinstance(public_id, str)
        or not public_id
        or not isinstance(category, str)
        or category not in NOVA_MEMORY_CATEGORIES
        or not isinstance(content, str)
        or not content
        or not isinstance(important, bool)
        or not isinstance(updated_at, datetime)
    ):
        raise NovaMemoryProjectionError("Invalid confirmed-memory item.")
    try:
        content.encode("utf-8")
    except UnicodeError:
        raise NovaMemoryProjectionError("Invalid confirmed-memory item.") from None
    normalized_updated_at = (
        updated_at.replace(tzinfo=UTC) if updated_at.tzinfo is None else updated_at.astimezone(UTC)
    )
    return _Candidate(
        public_id=public_id,
        category=category,
        content=content,
        important=important,
        updated_at=normalized_updated_at,
    )


def _selection_order(candidates: Sequence[_Candidate]) -> tuple[_Candidate, ...]:
    stable = sorted(candidates, key=lambda item: item.public_id)
    stable.sort(key=lambda item: _CATEGORY_RANK[item.category])
    stable.sort(key=lambda item: item.updated_at, reverse=True)

    important = [item for item in stable if item.important]
    selected_ids = {item.public_id for item in important}
    represented = {item.category for item in important}
    seeds: list[_Candidate] = []
    for category in _CATEGORY_PRIORITY:
        if category in represented:
            continue
        seed = next(
            (
                item
                for item in stable
                if item.category == category and item.public_id not in selected_ids
            ),
            None,
        )
        if seed is not None:
            seeds.append(seed)
            selected_ids.add(seed.public_id)

    remainder = [item for item in stable if item.public_id not in selected_ids]
    return tuple((*important, *seeds, *remainder))
