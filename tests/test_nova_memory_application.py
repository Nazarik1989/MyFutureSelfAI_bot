from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime, timedelta

import pytest

from future_self.nova_memory import NovaMemoryCategory
from future_self.nova_memory_application import (
    NOVA_MEMORY_APPLICATION_MAX_ITEMS,
    NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES,
    ConfirmedMemoryRecord,
    NovaMemoryProjection,
    NovaMemoryProjectionError,
    build_nova_memory_projection,
    serialize_confirmed_memory,
)


@dataclass(frozen=True)
class MemoryItem:
    public_id: str
    category: NovaMemoryCategory
    content: str
    important: bool
    updated_at: datetime


MOMENT = datetime(2026, 8, 13, 12, tzinfo=UTC)
REVISION = "opaque-collection-revision"


def item(
    public_id: str,
    category: NovaMemoryCategory,
    *,
    content: str | None = None,
    important: bool = False,
    seconds_ago: int = 0,
) -> MemoryItem:
    return MemoryItem(
        public_id=public_id,
        category=category,
        content=content or f"memory-{public_id}",
        important=important,
        updated_at=MOMENT - timedelta(seconds=seconds_ago),
    )


def contents(projection: NovaMemoryProjection) -> list[str]:
    return [record["content"] for record in projection.provider_payload()]


def test_exact_selection_order_puts_all_important_first_then_category_seeds_and_remainder():
    items = [
        item("imp-about-new", "about_me", important=True, seconds_ago=1),
        item("imp-interaction-old", "interaction", important=True, seconds_ago=2),
        item("imp-orientation-oldest", "orientation", important=True, seconds_ago=3),
        item("normal-about", "about_me", seconds_ago=4),
        item("normal-interaction", "interaction", seconds_ago=5),
        item("normal-orientation", "orientation", seconds_ago=6),
    ]

    projection = build_nova_memory_projection(reversed(items), collection_revision=REVISION)

    assert contents(projection) == [
        "memory-imp-about-new",
        "memory-imp-interaction-old",
        "memory-imp-orientation-oldest",
        "memory-normal-about",
        "memory-normal-interaction",
        "memory-normal-orientation",
    ]
    assert projection.important_count == 3


def test_missing_category_seeds_follow_category_priority_before_stable_remainder():
    items = [
        item("about-new", "about_me", seconds_ago=1),
        item("interaction-old", "interaction", seconds_ago=30),
        item("orientation-mid", "orientation", seconds_ago=20),
        item("about-old", "about_me", seconds_ago=40),
        item("interaction-older", "interaction", seconds_ago=50),
    ]

    projection = build_nova_memory_projection(items, collection_revision=REVISION)

    assert contents(projection) == [
        "memory-interaction-old",
        "memory-orientation-mid",
        "memory-about-new",
        "memory-about-old",
        "memory-interaction-older",
    ]


def test_stable_ties_use_category_priority_then_public_id_and_ignore_input_order():
    items = [
        item("z", "about_me"),
        item("b", "interaction"),
        item("a", "interaction"),
        item("o", "orientation"),
    ]

    forward = build_nova_memory_projection(items, collection_revision=REVISION)
    backward = build_nova_memory_projection(reversed(items), collection_revision=REVISION)

    assert contents(forward) == ["memory-a", "memory-o", "memory-z", "memory-b"]
    assert forward.provider_json() == backward.provider_json()


def test_projection_selects_at_most_twelve_records_and_counts_omitted():
    projection = build_nova_memory_projection(
        [item(f"{index:02}", "about_me", seconds_ago=index) for index in range(20)],
        collection_revision=REVISION,
    )

    assert len(projection) == NOVA_MEMORY_APPLICATION_MAX_ITEMS
    assert projection.selected_count == 12
    assert projection.omitted_count == 8


def test_payload_bytes_use_exact_compact_utf8_json_including_four_byte_unicode():
    projection = build_nova_memory_projection(
        [item("emoji", "interaction", content="Привет 🧬🌍")],
        collection_revision=REVISION,
    )
    payload = projection.provider_json()

    assert payload == '[{"category":"interaction","important":false,"content":"Привет 🧬🌍"}]'
    assert projection.payload_bytes == len(payload.encode("utf-8"))
    assert projection.payload_bytes > len(payload)
    assert projection.payload_bytes <= NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES


def test_whole_record_cutoff_stops_instead_of_truncating_content():
    first = item("first", "interaction", content="A" * 200)
    second = item("second", "orientation", content="🧬" * 2000, seconds_ago=1)
    third = item("third", "about_me", content="short", seconds_ago=2)

    projection = build_nova_memory_projection([third, second, first], collection_revision=REVISION)

    assert contents(projection) == [first.content]
    assert projection.omitted_count == 2
    assert first.content in projection.provider_json()
    assert second.content not in projection.provider_json()
    assert third.content not in projection.provider_json()


def test_exact_8_kibibyte_boundary_is_accepted_and_one_more_byte_is_rejected():
    one_byte = serialize_confirmed_memory(
        (ConfirmedMemoryRecord("interaction", False, "x"),)
    ).encode("utf-8")
    overhead = len(one_byte) - 1
    exact_content = "x" * (NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES - overhead)
    exact = build_nova_memory_projection(
        [item("exact", "interaction", content=exact_content)],
        collection_revision=REVISION,
    )
    too_large = build_nova_memory_projection(
        [item("large", "interaction", content=f"{exact_content}x")],
        collection_revision=REVISION,
    )

    assert exact.payload_bytes == NOVA_MEMORY_APPLICATION_MAX_PAYLOAD_BYTES
    assert exact.selected_count == 1
    assert too_large.provider_json() == "[]"
    assert too_large.selected_count == 0
    assert too_large.omitted_count == 1


def test_provider_payload_contains_only_allowed_fields_and_returns_fresh_containers():
    projection = build_nova_memory_projection(
        [item("private-public-id", "about_me", content="private-content", important=True)],
        collection_revision="private-revision",
    )

    first = projection.provider_payload()
    second = projection.provider_payload()

    assert first == [{"category": "about_me", "important": True, "content": "private-content"}]
    assert set(first[0]) == {"category", "important", "content"}
    assert first is not second
    assert first[0] is not second[0]
    assert "private-public-id" not in projection.provider_json()
    assert "private-revision" not in projection.provider_json()


def test_projection_and_records_are_immutable_and_repr_safe():
    secret = "SECRET-memory-content"
    revision = "SECRET-revision"
    record = ConfirmedMemoryRecord("orientation", True, secret)
    projection = build_nova_memory_projection(
        [item("SECRET-public-id", "orientation", content=secret, important=True)],
        collection_revision=revision,
    )

    assert secret not in repr(record)
    assert secret not in repr(projection)
    assert revision not in repr(projection)
    assert "SECRET-public-id" not in repr(projection)
    assert "records=" not in repr(projection)
    with pytest.raises(FrozenInstanceError):
        record.content = "replacement"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        projection.records = ()  # type: ignore[misc]


def test_empty_projection_is_false_and_serializes_to_exact_empty_array():
    projection = build_nova_memory_projection([], collection_revision=REVISION)

    assert not projection
    assert len(projection) == 0
    assert projection.provider_payload() == []
    assert projection.provider_json() == "[]"
    assert projection.payload_bytes == 2
    assert projection.omitted_count == 0


@pytest.mark.parametrize(
    "invalid",
    [
        MemoryItem("", "about_me", "safe", False, MOMENT),
        MemoryItem("id", "invalid", "safe", False, MOMENT),  # type: ignore[arg-type]
        MemoryItem("id", "about_me", "", False, MOMENT),
        MemoryItem("id", "about_me", "safe", 1, MOMENT),  # type: ignore[arg-type]
        MemoryItem("id", "about_me", "safe", False, "today"),  # type: ignore[arg-type]
    ],
)
def test_invalid_items_fail_with_privacy_safe_error(invalid: MemoryItem):
    with pytest.raises(NovaMemoryProjectionError) as error:
        build_nova_memory_projection([invalid], collection_revision=REVISION)

    assert repr(error.value) == "NovaMemoryProjectionError('Invalid confirmed-memory item.')"


def test_duplicate_internal_ids_fail_closed_without_exposing_data():
    secret = "duplicate-secret"

    with pytest.raises(NovaMemoryProjectionError) as error:
        build_nova_memory_projection(
            [
                item("same-id", "about_me", content=secret),
                item("same-id", "interaction", content="other"),
            ],
            collection_revision=REVISION,
        )

    assert secret not in repr(error.value)
    assert "same-id" not in repr(error.value)


def test_unhashable_category_fails_with_privacy_safe_error():
    invalid = MemoryItem(
        "private-public-id",
        ["about_me"],  # type: ignore[arg-type]
        "private-content",
        False,
        MOMENT,
    )

    with pytest.raises(NovaMemoryProjectionError) as error:
        build_nova_memory_projection([invalid], collection_revision=REVISION)

    assert repr(error.value) == "NovaMemoryProjectionError('Invalid confirmed-memory item.')"


def test_structural_input_has_no_owner_or_cross_owner_selection_channel():
    projection = build_nova_memory_projection(
        [item("owner-scoped", "about_me", content="only supplied snapshot")],
        collection_revision=REVISION,
    )
    serialized = json.loads(projection.provider_json())

    assert serialized == projection.provider_payload()
    assert all(set(record) == {"category", "important", "content"} for record in serialized)
