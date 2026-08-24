import re
from dataclasses import dataclass
from typing import Literal

SystemAction = Literal[
    "list_drafts",
    "discard_one_draft",
    "discard_selected_drafts",
    "discard_all_active_drafts",
    "show_last_saved",
    "archive_overdue_tasks",
    "trash_inbox_items",
    "trash_inbox_commands",
    "cancel_system_action",
]


@dataclass(slots=True, frozen=True)
class SystemActionRoute:
    kind: Literal["none", "action", "confirm", "cancel", "pending", "clarify"]
    action: SystemAction | None = None


class SystemActionRouter:
    """Deterministic destructive/system intent routing before content routing."""

    DELETE_FORMS = frozenset(
        {
            "удали",
            "удалить",
            "удалите",
            "удаляй",
            "удаляйте",
            "удалять",
            "убери",
            "убрать",
            "уберите",
            "убирай",
            "убирать",
            "очисти",
            "очистить",
            "очистите",
            "очищай",
            "очищать",
            "сотри",
            "стереть",
            "стирать",
        }
    )
    EXPLICIT_DELETE_FORMS = frozenset(
        {
            "удали",
            "удалите",
            "удаляй",
            "удаляйте",
            "убери",
            "уберите",
            "убирай",
            "очисти",
            "очистите",
            "очищай",
            "сотри",
        }
    )
    EXPLICIT_REMINDER_DISABLE_FORMS = frozenset(
        {
            "деактивируй",
            "деактивируйте",
            "отключи",
            "отключите",
            "отмени",
            "отмените",
            "останови",
            "остановите",
            "сними",
            "снимите",
            "выключи",
            "выключите",
            "выруби",
        }
    )
    CLEANUP_FORMS = frozenset({"очисти", "очистить", "очистите", "очищай", "очищать"})
    ALL_MARKERS = frozenset({"все", "всех", "целиком", "полностью"})
    COMMAND_FILLERS = frozenset({"ну", "пожалуйста", "прошу", "давай", "давайте", "же", "ка"})
    QUESTION_PREFIXES = (
        "как удалить",
        "как мне удалить",
        "как убрать",
        "как очистить",
        "можно ли удалить",
        "можно ли убрать",
        "можно ли очистить",
        "подскажи как удалить",
        "расскажи как удалить",
        "почему удалить",
        "зачем удалять",
        "что будет если удалить",
    )
    CAPTURE_PREFIXES = (
        "создай задачу",
        "добавь задачу",
        "запиши задачу",
        "запиши как задачу",
        "создай заметку",
        "добавь заметку",
        "напомни",
        "не забудь",
        "сохрани фразу",
        "запиши фразу",
        "запомни фразу",
        "запланируй",
        "запланировать",
    )
    REMINDER_DISABLE_FORMS = frozenset(
        {
            "деактивируй",
            "деактивировать",
            "деактивируйте",
            "отключи",
            "отключить",
            "отключите",
            "отмени",
            "отменить",
            "отмените",
            "останови",
            "остановить",
            "остановите",
            "сними",
            "снимите",
            "снять",
            "выключи",
            "выключить",
            "выключите",
            "выруби",
            "вырубить",
        }
    )
    PENDING_PASSTHROUGH = frozenset(
        {
            "меню",
            "главное меню",
            "открой меню",
            "покажи меню",
            "помощь",
            "покажи помощь",
            "какие есть команды",
            "какие у тебя есть команды",
            "какие у тебя команды",
            "какие команды у тебя есть",
            "что ты умеешь",
            "как пользоваться ботом",
        }
    )

    LAST_SAVED = (
        "напомни что ты сохранил",
        "что сохранилось последним",
        "что ты сохранил",
        "покажи последнюю сохраненную запись",
        "последняя сохраненная запись",
    )
    LIST_DRAFTS = (
        "покажи что не сохранено",
        "покажи черновики",
        "список черновиков",
        "покажи drafts",
    )
    DELETE_ALL = (
        "удали все черновики",
        "очисти черновики",
        "удали все несохраненные карточки",
        "убери этот мусор из drafts",
        "все несохраненные задачи удалить",
        "из черновиков хочу все удалить",
    )
    ARCHIVE_OVERDUE_TASKS = (
        "удали все неактуальные задачи",
        "удалить все неактуальные задачи",
        "убери все неактуальные задачи",
        "очисти неактуальные задачи",
        "убери просроченные задачи",
        "удали просроченные задачи",
        "удали все неактуальное",
        "убери все неактуальное",
    )
    KEEP_LAST = ("оставь только последнюю", "оставь только самый новый")
    DELETE_CONFIRM = (
        "да удалить",
        "удаляй все",
        "подтверждаю удаление",
        "да да удалить",
        "да все удалить",
    )

    def route(self, text: str, *, pending_action: str | None) -> SystemActionRoute:
        normalized = self._normalize(text)
        if pending_action and normalized in self.PENDING_PASSTHROUGH:
            return SystemActionRoute(kind="none")
        token_list = normalized.split()
        tokens = set(token_list)
        delete_positions = [
            index for index, token in enumerate(token_list) if token in self.DELETE_FORMS
        ]
        delete_mentioned = bool(delete_positions)
        negated_keep = self._has_adjacent_prefixed_sequence(
            token_list,
            "не",
            ("остав",),
        )
        keep_requested = "оставь" in tokens and not negated_keep
        question_or_capture = delete_mentioned and self._is_question_or_capture(normalized)
        negated_delete = delete_mentioned and self._is_negated_delete(
            normalized, token_list, delete_positions
        )
        excluded_stale_selector = delete_mentioned and self._has_excluded_stale_selector(normalized)
        targets = self._targets(token_list)
        mixed_targets = len(targets) > 1
        overdue_marker = any(token.startswith("просроченн") for token in token_list)
        stale_marker = (
            overdue_marker
            or any(token.startswith(("неактуальн", "устаревш", "устарел")) for token in token_list)
            or any(
                token == "не"
                and index + 1 < len(token_list)
                and token_list[index + 1].startswith("актуальн")
                for index, token in enumerate(token_list)
            )
        )
        vague_marker = any(
            token.startswith(("ненужн", "стар")) or token == "мусор" for token in token_list
        )
        temporal_marker = "прошл" in normalized and any(
            marker in normalized for marker in ("недел", "месяц", "год")
        )
        delete_requested = delete_mentioned and not negated_delete
        all_requested = bool(tokens & self.ALL_MARKERS)
        task_target = targets == {"tasks"}
        draft_target = targets == {"drafts"}
        reminder_target = targets == {"reminders"}
        task_bulk_scope = all_requested or any(token in {"задачи", "задач"} for token in token_list)
        reminder_control_mentioned = (
            bool(tokens & self.REMINDER_DISABLE_FORMS)
            or self._has_adjacent_prefixed_sequence(
                token_list,
                "не",
                ("нужн", "присыл"),
            )
            or self._has_negated_modal_control(
                token_list,
                ("надо", "нужно"),
                ("присыл",),
            )
        )
        unsupported_reminder_cleanup = reminder_target and (
            delete_mentioned or reminder_control_mentioned
        )
        safe_nounless_task_cleanup = (
            delete_requested
            and not targets
            and stale_marker
            and self._is_nounless_task_cleanup(token_list)
        )
        draft_bulk_requested = draft_target and (
            all_requested
            or stale_marker
            or vague_marker
            or bool(tokens & self.CLEANUP_FORMS)
            or "несохраненн" in normalized
        )
        desired_action: SystemAction | None = None
        if keep_requested and self._contains(normalized, self.KEEP_LAST):
            desired_action = "discard_selected_drafts"
        elif delete_requested and draft_bulk_requested:
            desired_action = "discard_all_active_drafts"
        elif delete_requested and (
            (task_target and task_bulk_scope and stale_marker) or safe_nounless_task_cleanup
        ):
            desired_action = "archive_overdue_tasks"
        cleanup_shaped = delete_mentioned and (
            all_requested
            or stale_marker
            or vague_marker
            or temporal_marker
            or bool(tokens & self.CLEANUP_FORMS)
            or any(target in targets for target in ("tasks", "drafts", "inbox", "reminders"))
        )
        destructive_shaped = cleanup_shaped or keep_requested or reminder_control_mentioned
        if pending_action:
            if question_or_capture:
                return SystemActionRoute(kind="pending")
            if negated_delete or negated_keep:
                return SystemActionRoute(kind="cancel", action="cancel_system_action")
            if (
                (mixed_targets and destructive_shaped)
                or excluded_stale_selector
                or unsupported_reminder_cleanup
            ):
                return SystemActionRoute(kind="clarify")
            # A clearly named new target replaces the pending preview; it never
            # confirms the previous one. This supports corrections such as
            # "нет, я имел в виду ... черновики" without mutating either set.
            current_target = (
                "tasks"
                if pending_action == "archive_overdue_tasks"
                else "inbox"
                if pending_action in {"trash_inbox_items", "trash_inbox_commands"}
                else "drafts"
            )
            mentioned_target = next(iter(targets)) if len(targets) == 1 else None
            confirm_requested = self._contains(normalized, self.DELETE_CONFIRM) or (
                "да" in tokens and "удал" in normalized
            )
            if desired_action is not None:
                if desired_action != pending_action or not confirm_requested:
                    return SystemActionRoute(kind="action", action=desired_action)
            elif mentioned_target is not None and mentioned_target != current_target:
                return SystemActionRoute(kind="cancel", action="cancel_system_action")

            negated_confirmation = self._has_adjacent_prefixed_sequence(
                token_list,
                "не",
                ("подтвержд",),
            )
            if (
                "отмена" in tokens
                or "нет" in tokens
                or "отмени" in tokens
                or negated_delete
                or negated_confirmation
            ):
                return SystemActionRoute(kind="cancel", action="cancel_system_action")
            if confirm_requested:
                return SystemActionRoute(kind="confirm")
            if cleanup_shaped:
                return SystemActionRoute(kind="clarify")
            return SystemActionRoute(kind="pending")
        if question_or_capture:
            return SystemActionRoute(kind="none")
        if negated_delete or negated_keep:
            return SystemActionRoute(kind="clarify")
        if (
            (mixed_targets and destructive_shaped)
            or excluded_stale_selector
            or unsupported_reminder_cleanup
        ):
            return SystemActionRoute(kind="clarify")
        if self._contains(normalized, self.LAST_SAVED):
            return SystemActionRoute(kind="action", action="show_last_saved")
        if self._contains(normalized, self.LIST_DRAFTS):
            return SystemActionRoute(kind="action", action="list_drafts")
        if desired_action is not None:
            return SystemActionRoute(kind="action", action=desired_action)
        if delete_requested and self._contains(normalized, self.DELETE_CONFIRM):
            return SystemActionRoute(kind="pending")
        if cleanup_shaped:
            return SystemActionRoute(kind="clarify")
        return SystemActionRoute(kind="none")

    @classmethod
    def is_explicit_cleanup_command(cls, text: str) -> bool:
        """Distinguish a direct control from narrative onboarding prose.

        Imperative forms are explicit anywhere because users often correct a
        previous voice transcription with a short preface. Infinitives are
        explicit only at the start of the command (after harmless fillers).
        """

        tokens = cls._normalize(text).split()
        if not tokens:
            return False
        if (
            any(
                token in cls.EXPLICIT_DELETE_FORMS | cls.EXPLICIT_REMINDER_DISABLE_FORMS
                for token in tokens
            )
            or "оставь" in tokens
        ):
            return True
        first_control = next(
            (token for token in tokens if token not in cls.COMMAND_FILLERS),
            None,
        )
        if first_control in cls.DELETE_FORMS | cls.REMINDER_DISABLE_FORMS:
            return True
        return any(
            tokens[index : index + 2] == ["не", "присылай"] for index in range(len(tokens) - 1)
        ) and any(token.startswith("напомин") for token in tokens)

    @classmethod
    def _targets(cls, tokens: list[str]) -> set[str]:
        targets: set[str] = set()
        unsaved_target = any(token.startswith("несохраненн") for token in tokens)
        if any(token.startswith("задач") for token in tokens):
            targets.add("tasks")
        if any(token.startswith("напомин") for token in tokens):
            targets.add("reminders")
        if unsaved_target:
            # "Несохранённые задачи" is the established user-facing name for drafts.
            targets.discard("tasks")
            targets.add("drafts")
        if any(token.startswith("чернов") or token == "drafts" for token in tokens):
            targets.add("drafts")
        if any(token in {"inbox", "инбокс", "инбок"} for token in tokens):
            targets.add("inbox")
        return targets

    @classmethod
    def _is_question_or_capture(cls, normalized: str) -> bool:
        if any(normalized.startswith(prefix) for prefix in cls.CAPTURE_PREFIXES):
            return True
        if any(normalized.startswith(prefix) for prefix in cls.QUESTION_PREFIXES):
            return True
        if re.match(
            r"^(?:создай|добавь|запиши|поставь)(?:\s+мне)?\s+"
            r"(?:задачу|заметку|напоминание)\b",
            normalized,
        ):
            return True
        if re.match(
            r"^(?:пожалуйста\s+)?(?:(?:можешь|можете)\s+)?"
            r"(?:создай|создать|добавь|добавить|запиши|записать|поставь|поставить|"
            r"запланируй|запланировать)(?:\s+пожалуйста)?(?:\s+мне)?\s+"
            r"(?:задачу|заметку|напоминание)\b",
            normalized,
        ):
            return True
        if re.match(
            r"^(?:мне\s+)?(?:нужно|надо)\s+"
            r"(?:создать|добавить|записать|поставить|запланировать)\s+"
            r"(?:задачу|заметку|напоминание)\b",
            normalized,
        ):
            return True
        if re.match(
            r"^(?:я\s+)?хочу\s+чтобы\s+(?:ты\s+)?"
            r"(?:создал(?:а)?|добавил(?:а)?|записал(?:а)?|поставил(?:а)?)\s+"
            r"(?:задачу|заметку|напоминание)\b",
            normalized,
        ):
            return True
        if re.match(r"^(?:моя\s+)?(?:задача|заметка|напоминание)\b", normalized):
            return True
        if re.match(
            r"^(?:(?:я|мы)\s+)?"
            r"(?:хочу|хотим|хотел(?:а|и)?|планирую|планируем)\s+"
            r"(?:создать|добавить|записать|поставить|запланировать)\s+"
            r"(?:мне\s+)?(?:задачу|заметку|напоминание)\b",
            normalized,
        ):
            return True
        if re.match(
            r"^(?:(?:я|мы)\s+)?"
            r"(?:хочу|хотим|хотел(?:а|и)?|планирую|планируем)\s+"
            r"(?:создать|добавить|записать|поставить|запланировать)\b",
            normalized,
        ):
            return True
        if re.match(
            r"^(?:как|можно\s+ли|стоит\s+ли|надо\s+ли|нужно\s+ли|следует\s+ли|"
            r"почему|зачем|что\s+будет\s+если)\b",
            normalized,
        ):
            return True
        if re.search(r"\b(?:стоит|надо|нужно|следует)\s+ли\s+(?:удал|убир|очищ|стер)", normalized):
            return True
        if re.search(r"\b(?:удал|убир|очищ|стер)\w*\s+ли\b", normalized):
            return True
        if re.match(
            r"^(?:что\s+(?:ты\s+)?(?:думаешь|скажешь)|как\s+(?:ты\s+)?считаешь)\b",
            normalized,
        ):
            return True
        if re.match(r"^не\s+уверен(?:а)?\b.*\b(?:надо|нужно|стоит|следует)\s+ли\b", normalized):
            return True
        if re.match(r"^(?:команда|фраза)\b", normalized):
            return True
        if re.match(r"^(?:когда|если)\b.*\b(?:скажу|говорю|услышишь)\b", normalized):
            return True
        if re.search(r"\b(?:команд\w*|фраз\w*|пример\w*)\b", normalized):
            return True
        if re.match(r"^(?:проверь|проверить|проверим|проверка|тест|тестирую)\b", normalized):
            return True
        return bool(re.search(r"(?:^|\s)(?:обсуждали|расскажи|подскажи)\s+как\s+", normalized))

    @classmethod
    def _is_negated_delete(
        cls, normalized: str, tokens: list[str], delete_positions: list[int]
    ) -> bool:
        modal_negations = {"надо", "нужно", "хочу", "просил", "следует"}
        delete_prefixes = ("удал", "убир", "очищ", "стер")
        if cls._has_negated_modal_control(tokens, tuple(modal_negations), delete_prefixes):
            return True
        if re.search(
            r"\bне\s+(?:планирую|планировал(?:а)?|собираюсь|собирался|собиралась|"
            r"стану|буду|намерен(?:а)?|хочу|просил(?:а)?)\b"
            r"(?:\s+\w+){0,3}\s+(?:удал|убир|очищ|стер)",
            normalized,
        ):
            return True
        if re.search(
            r"\bпередумал(?:а|и)?\b(?:\s+\w+){0,3}\s+(?:удал|убир|очищ|стер)",
            normalized,
        ):
            return True
        if any(
            phrase in normalized for phrase in ("не делай этого", "не выполняй это", "не трогай их")
        ):
            return True
        for position in delete_positions:
            preceding = tokens[max(0, position - 2) : position]
            if (preceding and preceding[-1] == "не") or preceding == ["не", "да"]:
                return True
        return False

    @staticmethod
    def _has_excluded_stale_selector(normalized: str) -> bool:
        stale = r"(?:просроченн\w*|неактуальн\w*|устаревш\w*|устарел\w*)"
        modifiers = r"(?:(?:всех|сам\w*|уже|особенно)\s+)*"
        return bool(
            re.search(rf"\bне\s+(?:сам\w*\s+)?{stale}\b", normalized)
            or re.search(rf"\bкроме\s+{modifiers}{stale}\b", normalized)
            or re.search(rf"\bза\s+исключением\s+{modifiers}{stale}\b", normalized)
            or re.search(rf"\bисключая\s+{modifiers}{stale}\b", normalized)
            or re.search(rf"\bне\s+трогая\s+{stale}\b", normalized)
            or re.search(rf"\b{stale}\s+(?:остав\w*|не\s+трог\w*)\b", normalized)
        )

    @classmethod
    def _is_nounless_task_cleanup(cls, tokens: list[str]) -> bool:
        allowed = (
            cls.DELETE_FORMS
            | cls.ALL_MARKERS
            | cls.COMMAND_FILLERS
            | {
                "давно",
                "их",
                "не",
            }
        )
        plural_stale = any(
            token
            in {
                "просроченные",
                "просроченных",
                "неактуальные",
                "неактуальных",
                "устаревшие",
                "устаревших",
            }
            for token in tokens
        )
        explicit_bulk = bool(set(tokens) & cls.ALL_MARKERS) or plural_stale
        return explicit_bulk and all(
            token in allowed
            or token.startswith(("просроченн", "неактуальн", "устаревш", "устарел", "актуальн"))
            for token in tokens
        )

    @staticmethod
    def _contains(value: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in value for pattern in patterns)

    @staticmethod
    def _has_adjacent_prefixed_sequence(
        tokens: list[str],
        first: str,
        following_prefixes: tuple[str, ...],
    ) -> bool:
        """Match grammatical neighbours, never character substrings across words.

        Russian ``мне`` ends with the characters ``не``.  Treating normalized
        text as an arbitrary substring therefore turns positive phrases such as
        ``мне нужно`` into destructive negations.  All negative control markers
        pass through this token boundary helper instead.
        """

        return any(
            tokens[index] == first and tokens[index + 1].startswith(following_prefixes)
            for index in range(max(0, len(tokens) - 1))
        )

    @staticmethod
    def _has_negated_modal_control(
        tokens: list[str],
        modals: tuple[str, ...],
        control_prefixes: tuple[str, ...],
    ) -> bool:
        return any(
            tokens[index] == "не"
            and tokens[index + 1] in modals
            and any(token.startswith(control_prefixes) for token in tokens[index + 2 : index + 6])
            for index in range(max(0, len(tokens) - 2))
        )

    @staticmethod
    def _normalize(text: str) -> str:
        lowered = text.lower().replace("ё", "е")
        lowered = re.sub(r"[^a-zа-я0-9]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered).strip()
