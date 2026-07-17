from future_self.schemas import IntentResult

from .harness import (
    DraftState,
    ExpectedState,
    InboxState,
    LLMStub,
    Scenario,
    ScenarioStep,
)


def capture(text: str, *, intent: str, kind: str, title: str) -> LLMStub:
    return LLMStub(
        text,
        IntentResult(
            intent=intent,
            confidence=0.99,
            inbox_kind=kind,
            title=title,
        ),
    )


def sorted_drafts(*drafts: DraftState) -> tuple[DraftState, ...]:
    return tuple(sorted(drafts))


def focused_save_scenario(
    index: int,
    command: str,
    source: str,
    *,
    known_defect: str | None = None,
) -> Scenario:
    raw_text = f"Подготовить проверку сохранения {index:02d}"
    title = f"Проверка сохранения {index:02d}"
    channel = "голосовой" if source == "voice" else "текстовой"
    return Scenario(
        name=f"focused-save-{index:02d}-{source}",
        llm_stubs=(capture(raw_text, intent="inbox_task", kind="task", title=title),),
        steps=(
            ScenarioStep("text", raw_text, reply_contains=("Заголовок", title)),
            ScenarioStep(
                source,
                command,
                reply_contains=(f"Сохранено в inbox по {channel} команде",),
                reply_excludes=("Не сохраняю",),
            ),
        ),
        expected=ExpectedState(
            drafts=(DraftState(title, "task", "confirmed", "text"),),
            inbox=(InboxState(title, "task", "text"),),
            llm_inputs=(raw_text,),
        ),
        known_defect=known_defect,
    )


def no_draft_scenario(index: int, command: str, source: str) -> Scenario:
    return Scenario(
        name=f"no-draft-save-{index:02d}-{source}",
        steps=(
            ScenarioStep(
                source,
                command,
                reply_contains=("Нет одной актуальной",),
            ),
        ),
        expected=ExpectedState(),
    )


def negative_scenario(
    index: int,
    command: str,
    source: str,
    *,
    known_defect: str | None = None,
) -> Scenario:
    raw_text = f"Идея для отрицательной команды {index:02d}"
    title = f"Отрицательная команда {index:02d}"
    return Scenario(
        name=f"negative-command-{index:02d}-{source}",
        llm_stubs=(capture(raw_text, intent="inbox_idea", kind="idea", title=title),),
        steps=(
            ScenarioStep("text", raw_text, reply_contains=("Заголовок", title)),
            ScenarioStep(
                source,
                command,
                reply_contains=("удалена без сохранения",),
            ),
        ),
        expected=ExpectedState(
            drafts=(DraftState(title, "idea", "discarded", "text"),),
            llm_inputs=(raw_text,),
        ),
        known_defect=known_defect,
    )


def ordinary_content_scenario(index: int, text: str, source: str) -> Scenario:
    title = f"Контент про inbox {index:02d}"
    return Scenario(
        name=f"ordinary-content-{index:02d}-{source}",
        llm_stubs=(capture(text, intent="inbox_note", kind="note", title=title),),
        steps=(ScenarioStep(source, text, reply_contains=("Заголовок", title)),),
        expected=ExpectedState(
            drafts=(DraftState(title, "note", "preview", source),),
            llm_inputs=(text,),
        ),
    )


def ambiguous_scenario(index: int, command: str, source: str) -> Scenario:
    first_raw = f"Первая идея неоднозначности {index:02d}"
    second_raw = f"Вторая идея неоднозначности {index:02d}"
    first_title = f"Первая неоднозначная {index:02d}"
    second_title = f"Вторая неоднозначная {index:02d}"
    return Scenario(
        name=f"lost-focus-{index:02d}-{source}",
        llm_stubs=(
            capture(first_raw, intent="inbox_idea", kind="idea", title=first_title),
            capture(second_raw, intent="inbox_idea", kind="idea", title=second_title),
        ),
        steps=(
            ScenarioStep("text", first_raw, reply_contains=("Заголовок", first_title)),
            ScenarioStep("voice", second_raw, reply_contains=("Заголовок", second_title)),
            ScenarioStep("setup_clear_focus"),
            ScenarioStep(
                source,
                command,
                reply_contains=("К какой карточке применить команду?",),
            ),
        ),
        expected=ExpectedState(
            drafts=sorted_drafts(
                DraftState(first_title, "idea", "preview", "text"),
                DraftState(second_title, "idea", "preview", "voice"),
            ),
            llm_inputs=(first_raw, second_raw),
        ),
    )


SAVE_COMMANDS = (
    "Сохрани инбокс",
    "сохрани в инбокс",
    "СОХРАНИ В ИНБОКС",
    "Сохрани inbox",
    "сохрани в INBOX",
    "Сохрани это в инбокс",
    "Сохрани это инбокс",
    "СОХРАНИ ЭТО В INBOX",
    "Сохраним инбокс",
    "сохраним в инбокс",
    "СОХРАНИМ В INBOX",
    "Сохраним это в инбокс",
    "сохраним это inbox",
    "  Сохрани   это   в   инбокс  ",
    "Сохраним это в inbox?!",
    "Сохрани... в... инбокс!!!",
)

NEGATIVE_COMMANDS = (
    "Не сохраняй",
    "НЕ СОХРАНЯЙ!",
    "не сохранять.",
    "Не сохраняй в инбокс",
    "НЕ СОХРАНЯЙ В INBOX?!",
    "Не надо сохранять",
    "не надо сохранять в инбокс",
    "НЕ НАДО СОХРАНЯТЬ В INBOX!",
    "  не   сохраняй   в   инбокс   ",
    "Не надо сохранять, в inbox",
)

ORDINARY_CONTENT = (
    "Хочу понять, стоит ли сохранять полезные статьи в инбокс для чтения",
    "Как лучше сохранять статьи в inbox для чтения",
    "Напиши инструкцию, как сохранить заметки в инбокс",
    "Обсудим привычку сохранять идеи в инбокс по пятницам",
    "Я хочу сохранять ссылки в inbox, но не уверен",
    "Почему полезно сохранять мысли в инбокс",
    "Стоит ли вообще сохранять всё подряд в inbox",
    "Сравни способы сохранить материалы в инбокс",
    "План статьи: как сохранять идеи в inbox",
    "Не надо сохранять каждую случайную мысль в инбокс автоматически",
)

NOISY_SAVE_DEFECTS = (
    "Ну сохрани в инбокс",
    "Пожалуйста, сохрани в inbox",
    "Сохрани это в инбокс, пожалуйста",
    "Короче сохраним это в инбокс",
    "Эээ сохрани в инбокс",
    "Сохрани, пожалуйста, это в inbox",
)

CLIPPED_SAVE_DEFECTS = (
    "Это в инбокс",
    "Это в inbox",
    "В инбокс",
    "Сохрани в инбок",
)

NEGATIVE_DEFECTS = (
    "Не сохраняй это в инбокс",
    "Пожалуйста, не сохраняй в inbox",
    "Не надо это сохранять в инбокс",
    "Не сохраняй в инбок",
)


GENERATED_SAVE_SCENARIOS = tuple(
    focused_save_scenario(
        index,
        command,
        "voice" if index % 2 else "text",
    )
    for index, command in enumerate(SAVE_COMMANDS, start=1)
)

GENERATED_NO_DRAFT_SCENARIOS = tuple(
    no_draft_scenario(
        index,
        command,
        "text" if index % 2 else "voice",
    )
    for index, command in enumerate(SAVE_COMMANDS[:8], start=1)
)

GENERATED_NEGATIVE_SCENARIOS = tuple(
    negative_scenario(
        index,
        command,
        "voice" if index % 2 else "text",
    )
    for index, command in enumerate(NEGATIVE_COMMANDS, start=1)
)

GENERATED_CONTENT_SCENARIOS = tuple(
    ordinary_content_scenario(
        index,
        text,
        "text" if index % 2 else "voice",
    )
    for index, text in enumerate(ORDINARY_CONTENT, start=1)
)

GENERATED_AMBIGUOUS_SCENARIOS = tuple(
    ambiguous_scenario(
        index,
        SAVE_COMMANDS[index - 1],
        "voice" if index % 2 else "text",
    )
    for index in range(1, 5)
)

KNOWN_SAVE_DEFECT_SCENARIOS = tuple(
    focused_save_scenario(
        100 + index,
        command,
        "voice" if index % 2 else "text",
        known_defect="AUTOTEST-D001: save command with filler or politeness reaches LLM",
    )
    for index, command in enumerate(NOISY_SAVE_DEFECTS, start=1)
) + tuple(
    focused_save_scenario(
        110 + index,
        command,
        "voice" if index % 2 else "text",
        known_defect="AUTOTEST-D002: clipped save transcription reaches LLM",
    )
    for index, command in enumerate(CLIPPED_SAVE_DEFECTS, start=1)
)

KNOWN_NEGATIVE_DEFECT_SCENARIOS = tuple(
    negative_scenario(
        120 + index,
        command,
        "voice" if index % 2 else "text",
        known_defect="AUTOTEST-D003: extended negative command reaches LLM",
    )
    for index, command in enumerate(NEGATIVE_DEFECTS, start=1)
)


THERAPIST = "Записаться к терапевту и разобраться с причиной слабости"

CORE_SCENARIOS = (
    Scenario(
        name="production-sequence-mixed-text-voice-repeat",
        llm_stubs=(capture(THERAPIST, intent="inbox_task", kind="task", title=THERAPIST),),
        steps=(
            ScenarioStep("text", THERAPIST, reply_contains=("Заголовок", THERAPIST)),
            ScenarioStep(
                "voice",
                "Сохраним это в inbox?!",
                reply_contains=("Сохранено в inbox по голосовой команде",),
                reply_excludes=("Не сохраняю",),
            ),
            ScenarioStep("text", "Сохрани в инбокс", reply_contains=("Нет одной актуальной",)),
            ScenarioStep(
                "voice",
                "Сохрани инбокс",
                reply_contains=("Нет одной актуальной",),
            ),
        ),
        expected=ExpectedState(
            drafts=(DraftState(THERAPIST, "task", "confirmed", "text"),),
            inbox=(InboxState(THERAPIST, "task", "text"),),
            llm_inputs=(THERAPIST,),
        ),
    ),
    Scenario(
        name="focused-draft-wins-when-an-older-preview-exists",
        llm_stubs=(
            capture(
                "Старая карточка",
                intent="inbox_idea",
                kind="idea",
                title="Старая карточка",
            ),
            capture(
                "Новая focused карточка",
                intent="inbox_task",
                kind="task",
                title="Новая focused карточка",
            ),
        ),
        steps=(
            ScenarioStep("text", "Старая карточка", reply_contains=("Заголовок",)),
            ScenarioStep("voice", "Новая focused карточка", reply_contains=("Заголовок",)),
            ScenarioStep(
                "voice",
                "Сохрани в инбокс",
                reply_contains=("Сохранено в inbox по голосовой команде",),
            ),
        ),
        expected=ExpectedState(
            drafts=sorted_drafts(
                DraftState("Старая карточка", "idea", "preview", "text"),
                DraftState("Новая focused карточка", "task", "confirmed", "voice"),
            ),
            inbox=(InboxState("Новая focused карточка", "task", "voice"),),
            llm_inputs=("Старая карточка", "Новая focused карточка"),
        ),
    ),
    Scenario(
        name="natural-read-command-remains-read-only",
        steps=(
            ScenarioStep("text", "Что у меня сохранено?", reply_contains=("Inbox пока пуст",)),
            ScenarioStep("voice", "ЧТО У МЕНЯ СОХРАНЕНО?!", reply_contains=("Inbox пока пуст",)),
        ),
        expected=ExpectedState(),
    ),
)


CALLBACK_SCENARIOS = (
    Scenario(
        name="callback-save-repeat-is-idempotent",
        llm_stubs=(
            capture("Callback save", intent="inbox_note", kind="note", title="Callback save"),
        ),
        steps=(
            ScenarioStep("text", "Callback save", reply_contains=("Заголовок",)),
            ScenarioStep("callback", "save", reply_contains=("Сохранено в inbox",)),
            ScenarioStep("callback", "save", reply_contains=("уже неактуальна",)),
        ),
        expected=ExpectedState(
            drafts=(DraftState("Callback save", "note", "confirmed", "text"),),
            inbox=(InboxState("Callback save", "note", "text"),),
            llm_inputs=("Callback save",),
        ),
    ),
    Scenario(
        name="callback-drop-repeat-never-saves",
        llm_stubs=(
            capture("Callback drop", intent="inbox_note", kind="note", title="Callback drop"),
        ),
        steps=(
            ScenarioStep("text", "Callback drop", reply_contains=("Заголовок",)),
            ScenarioStep("callback", "drop", reply_contains=("Не сохраняю",)),
            ScenarioStep("callback", "drop", reply_contains=("уже неактуальна",)),
        ),
        expected=ExpectedState(
            drafts=(DraftState("Callback drop", "note", "discarded", "text"),),
            llm_inputs=("Callback drop",),
        ),
    ),
    Scenario(
        name="callback-save-then-voice-repeat-is-idempotent",
        llm_stubs=(
            capture(
                "Callback then voice",
                intent="inbox_note",
                kind="note",
                title="Callback then voice",
            ),
        ),
        steps=(
            ScenarioStep("text", "Callback then voice", reply_contains=("Заголовок",)),
            ScenarioStep("callback", "save", reply_contains=("Сохранено в inbox",)),
            ScenarioStep(
                "voice",
                "Сохрани это в инбокс",
                reply_contains=("Нет одной актуальной",),
            ),
        ),
        expected=ExpectedState(
            drafts=(DraftState("Callback then voice", "note", "confirmed", "text"),),
            inbox=(InboxState("Callback then voice", "note", "text"),),
            llm_inputs=("Callback then voice",),
        ),
    ),
    Scenario(
        name="voice-save-then-stale-callback-is-idempotent",
        llm_stubs=(
            capture(
                "Voice then callback",
                intent="inbox_task",
                kind="task",
                title="Voice then callback",
            ),
        ),
        steps=(
            ScenarioStep("text", "Voice then callback", reply_contains=("Заголовок",)),
            ScenarioStep(
                "voice",
                "Сохрани в inbox",
                reply_contains=("Сохранено в inbox по голосовой команде",),
            ),
            ScenarioStep("callback", "save", reply_contains=("уже неактуальна",)),
        ),
        expected=ExpectedState(
            drafts=(DraftState("Voice then callback", "task", "confirmed", "text"),),
            inbox=(InboxState("Voice then callback", "task", "text"),),
            llm_inputs=("Voice then callback",),
        ),
    ),
    Scenario(
        name="callback-edit-repeat-is-idempotent",
        llm_stubs=(
            capture("Callback edit", intent="inbox_note", kind="note", title="Callback edit"),
        ),
        steps=(
            ScenarioStep("text", "Callback edit", reply_contains=("Заголовок",)),
            ScenarioStep("callback", "edit"),
            ScenarioStep("callback", "edit", reply_contains=("уже неактуальна",)),
        ),
        expected=ExpectedState(
            drafts=(DraftState("Callback edit", "note", "editing", "text"),),
            llm_inputs=("Callback edit",),
        ),
    ),
)


SCENARIOS = (
    *CORE_SCENARIOS,
    *GENERATED_SAVE_SCENARIOS,
    *GENERATED_NO_DRAFT_SCENARIOS,
    *GENERATED_NEGATIVE_SCENARIOS,
    *GENERATED_CONTENT_SCENARIOS,
    *GENERATED_AMBIGUOUS_SCENARIOS,
    *CALLBACK_SCENARIOS,
    *KNOWN_SAVE_DEFECT_SCENARIOS,
    *KNOWN_NEGATIVE_DEFECT_SCENARIOS,
)
