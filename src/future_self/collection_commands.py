from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .collections_service import CollectionKind

CollectionCommandAction = Literal["create", "add", "show", "move_last", "add_more"]


@dataclass(frozen=True, slots=True)
class CollectionCommand:
    action: CollectionCommandAction
    target: str | None = None
    content: str | None = None
    kind: CollectionKind | None = None
    forced_item_kind: str | None = None


class CollectionCommandRouter:
    """Conservative Russian routing before date parsing and any LLM call."""

    _TYPE_MAP: dict[str, CollectionKind] = {
        "тему": "topic",
        "тема": "topic",
        "проект": "project",
        "список": "list",
        "раздел": "topic",
    }

    def route(self, text: str) -> CollectionCommand | None:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return None

        create = re.fullmatch(
            r"(?:создай|создать)\s+(тему|тема|проект|список|раздел)\s+(.+?)[.!]?",
            compact,
            flags=re.IGNORECASE,
        )
        if create:
            return CollectionCommand(
                "create",
                target=create.group(2).strip(),
                kind=self._TYPE_MAP[create.group(1).lower()],
            )

        save = re.fullmatch(
            r"(?:сохрани|сохранить|запиши|записать)\s+в\s+"
            r"(?:(проект|тему|тема|список|раздел)\s+)?(.+?)\s*:\s*(.+)",
            compact,
            flags=re.IGNORECASE,
        )
        if save:
            kind = self._TYPE_MAP.get((save.group(1) or "").lower())
            return CollectionCommand(
                "add",
                target=save.group(2).strip(),
                content=save.group(3).strip(),
                kind=kind,
            )

        idea = re.fullmatch(
            r"(?:запиши|сохрани)\s+идею\s+(?:для|в)\s+(.+?)\s*:\s*(.+)",
            compact,
            flags=re.IGNORECASE,
        )
        if idea:
            return CollectionCommand(
                "add",
                target=idea.group(1).strip(),
                content=idea.group(2).strip(),
                forced_item_kind="idea",
            )

        add = re.fullmatch(r"(?:добавь|добавить)\s+в\s+(.+)", compact, flags=re.IGNORECASE)
        if add:
            return CollectionCommand("add", target=add.group(1).strip())

        show = re.fullmatch(
            r"(?:покажи|открой)\s+(?:(?:проект|тему|тема|список|раздел)\s+)?(.+?)[?!.]?",
            compact,
            flags=re.IGNORECASE,
        )
        if show:
            return CollectionCommand("show", target=show.group(1).strip().rstrip("?!."))

        contains = re.fullmatch(
            r"(?:что\s+(?:находится|лежит)\s+в|что\s+в)\s+(.+?)[?!.]?",
            compact,
            flags=re.IGNORECASE,
        )
        if contains:
            return CollectionCommand("show", target=contains.group(1).strip().rstrip("?!."))

        move = re.fullmatch(
            r"(?:перенеси|перемести)\s+(?:это|запись)\s+в\s+(.+?)[.!]?",
            compact,
            flags=re.IGNORECASE,
        )
        if move:
            return CollectionCommand("move_last", target=move.group(1).strip().rstrip("?!."))

        more = re.fullmatch(
            r"(?:еще|ещё)\s+(?:добавь|добавить)\s+(.+)", compact, flags=re.IGNORECASE
        )
        if more:
            return CollectionCommand("add_more", content=more.group(1).strip())
        return None
