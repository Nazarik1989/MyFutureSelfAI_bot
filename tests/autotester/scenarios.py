from future_self.schemas import IntentResult

from .harness import (
    DraftState,
    ExpectedState,
    InboxState,
    LLMStub,
    Scenario,
    ScenarioStep,
    VisionState,
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


def doctor_steps(
    *,
    reason: str = "Слабость",
    duration: str = "Три недели",
    symptoms: str = "Быстро устаю к вечеру",
    medications: str = "нет",
    questions: str = "Что важно наблюдать дальше?",
) -> tuple[ScenarioStep, ...]:
    return (
        ScenarioStep("command", "/doctor_prepare", reply_contains=("причина обращения",)),
        ScenarioStep("doctor_answer", reason, reply_contains=("Как долго",)),
        ScenarioStep("doctor_answer", duration, reply_contains=("Перечисли симптомы",)),
        ScenarioStep("doctor_answer", symptoms, reply_contains=("лекарства",)),
        ScenarioStep("doctor_answer", medications, reply_contains=("вопросы",)),
        ScenarioStep(
            "doctor_answer",
            questions,
            reply_contains=(
                "Краткое фактическое резюме",
                "не медицинский диагноз",
            ),
        ),
    )


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

NOISY_SAVE_VARIANTS = (
    "Ну сохрани в инбокс",
    "Пожалуйста, сохрани в inbox",
    "Сохрани это в инбокс, пожалуйста",
    "Короче сохраним это в инбокс",
    "Эээ сохрани в инбокс",
    "Сохрани, пожалуйста, это в inbox",
)

CLIPPED_SAVE_VARIANTS = (
    "Это в инбокс",
    "Это в inbox",
    "В инбокс",
    "Сохрани в инбок",
)

EXTENDED_NEGATIVE_VARIANTS = (
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

RESOLVED_SAVE_REGRESSION_SCENARIOS = tuple(
    focused_save_scenario(
        100 + index,
        command,
        "voice" if index % 2 else "text",
    )
    for index, command in enumerate(NOISY_SAVE_VARIANTS, start=1)
) + tuple(
    focused_save_scenario(
        110 + index,
        command,
        "voice" if index % 2 else "text",
    )
    for index, command in enumerate(CLIPPED_SAVE_VARIANTS, start=1)
)

RESOLVED_NEGATIVE_REGRESSION_SCENARIOS = tuple(
    negative_scenario(
        120 + index,
        command,
        "voice" if index % 2 else "text",
    )
    for index, command in enumerate(EXTENDED_NEGATIVE_VARIANTS, start=1)
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


HEALTH_SCENARIOS = (
    Scenario(
        name="health-empty-is-private-and-read-only",
        steps=(
            ScenarioStep(
                "command",
                "/health",
                reply_contains=("история пока пуста", "не является медицинским диагнозом"),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="health-checkin-one-question-at-a-time",
        steps=(
            ScenarioStep("command", "/checkin", reply_contains=("Энергия",)),
            ScenarioStep("health_answer", "7", reply_contains=("Сон",)),
            ScenarioStep("health_answer", "6", reply_contains=("Настроение",)),
            ScenarioStep("health_answer", "8", reply_contains=("Стресс",)),
            ScenarioStep("health_answer", "3", reply_contains=("Физическое",)),
            ScenarioStep("health_answer", "7", reply_contains=("симптомы",)),
            ScenarioStep(
                "health_answer",
                "нет",
                reply_contains=("70/100", "не медицинский диагноз"),
            ),
        ),
        expected=ExpectedState(health_scores=(70,)),
    ),
    Scenario(
        name="health-checkin-minimum-boundary",
        steps=(
            ScenarioStep("command", "/checkin", reply_contains=("Энергия",)),
            ScenarioStep("health_answer", "0", reply_contains=("Сон",)),
            ScenarioStep("health_answer", "0", reply_contains=("Настроение",)),
            ScenarioStep("health_answer", "0", reply_contains=("Стресс",)),
            ScenarioStep("health_answer", "10", reply_contains=("Физическое",)),
            ScenarioStep("health_answer", "0", reply_contains=("симптомы",)),
            ScenarioStep("health_answer", "нет", reply_contains=("0/100",)),
        ),
        expected=ExpectedState(health_scores=(0,)),
    ),
    Scenario(
        name="health-checkin-maximum-boundary",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "10"),
            ScenarioStep("health_answer", "10"),
            ScenarioStep("health_answer", "10"),
            ScenarioStep("health_answer", "0"),
            ScenarioStep("health_answer", "10"),
            ScenarioStep("health_answer", "нет", reply_contains=("100/100",)),
            ScenarioStep(
                "command",
                "/health",
                reply_contains=("Линейка: 100/100", "Неделя: 1 check-in"),
            ),
        ),
        expected=ExpectedState(health_scores=(100,)),
    ),
    Scenario(
        name="health-invalid-ratings-stay-on-current-question",
        steps=(
            ScenarioStep("command", "/checkin", reply_contains=("Энергия",)),
            ScenarioStep("health_answer", "-1", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "11", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "7.5", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "много", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "0", reply_contains=("Сон",)),
            ScenarioStep("health_answer", "10", reply_contains=("Настроение",)),
            ScenarioStep("health_answer", "5", reply_contains=("Стресс",)),
            ScenarioStep("health_answer", "10", reply_contains=("Физическое",)),
            ScenarioStep("health_answer", "0", reply_contains=("симптомы",)),
            ScenarioStep("health_answer", "нет", reply_contains=("30/100",)),
        ),
        expected=ExpectedState(health_scores=(30,)),
    ),
    Scenario(
        name="health-cancel-discards-partial-and-new-checkin-resumes-cleanly",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "9"),
            ScenarioStep("command", "/cancel", reply_contains=("отменён", "ничего не сохранено")),
            ScenarioStep("command", "/health", reply_contains=("история пока пуста",)),
            ScenarioStep("command", "/checkin", reply_contains=("Энергия",)),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "нет", reply_contains=("50/100",)),
        ),
        expected=ExpectedState(health_scores=(50,)),
    ),
    Scenario(
        name="health-invalid-answer-at-every-rating-step-does-not-advance",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "один", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "7", reply_contains=("Сон",)),
            ScenarioStep("health_answer", "6 часов", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "6", reply_contains=("Настроение",)),
            ScenarioStep("health_answer", "11", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "8", reply_contains=("Стресс",)),
            ScenarioStep("health_answer", "-1", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "3", reply_contains=("Физическое",)),
            ScenarioStep("health_answer", "7.0", reply_contains=("от 0 до 10",)),
            ScenarioStep("health_answer", "7", reply_contains=("симптомы",)),
            ScenarioStep("health_answer", "нет", reply_contains=("70/100",)),
        ),
        expected=ExpectedState(health_scores=(70,)),
    ),
    Scenario(
        name="health-edit-display-and-delete-real-command-paths",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "6"),
            ScenarioStep("health_answer", "8"),
            ScenarioStep("health_answer", "3"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "нет", reply_contains=("70/100",)),
            ScenarioStep("command", "/health_edit 1", reply_contains=("Исправляем запись",)),
            ScenarioStep("health_answer", "10"),
            ScenarioStep("health_answer", "10"),
            ScenarioStep("health_answer", "10"),
            ScenarioStep("health_answer", "0"),
            ScenarioStep("health_answer", "10"),
            ScenarioStep("health_answer", "нет", reply_contains=("100/100",)),
            ScenarioStep("command", "/health", reply_contains=("Линейка: 100/100", "#1")),
            ScenarioStep("command", "/health_delete 1", reply_contains=("удалена",)),
            ScenarioStep("command", "/health", reply_contains=("история пока пуста",)),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="health-records-are-private-between-users",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "6"),
            ScenarioStep("health_answer", "8"),
            ScenarioStep("health_answer", "3"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "личное наблюдение"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep("command", "/health", reply_contains=("история пока пуста",)),
            ScenarioStep("command", "/health_edit 1", reply_contains=("нет",)),
            ScenarioStep("command", "/health_delete 1", reply_contains=("не найдена",)),
            ScenarioStep("switch_user", "900001:910001"),
            ScenarioStep(
                "command",
                "/health",
                reply_contains=("70/100", "личное наблюдение"),
            ),
        ),
        expected=ExpectedState(health_scores=(70,)),
    ),
    Scenario(
        name="health-red-flag-stays-out-of-llm-and-recommends-urgent-help",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep(
                "health_answer",
                "Сильная боль в груди",
                reply_contains=("экстренную медицинскую службу", "не ставит диагноз"),
                reply_excludes=("лекарство", "анализ"),
            ),
        ),
        expected=ExpectedState(health_scores=(50,)),
    ),
    Scenario(
        name="health-negated-red-flag-does-not-escalate",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep(
                "health_answer",
                "Нет боли в груди, просто устал",
                reply_contains=("50/100",),
                reply_excludes=("экстренную медицинскую службу",),
            ),
        ),
        expected=ExpectedState(health_scores=(50,)),
    ),
    Scenario(
        name="health-colloquial-breathing-red-flag-is-escalated",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "4"),
            ScenarioStep("health_answer", "4"),
            ScenarioStep("health_answer", "4"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "2"),
            ScenarioStep(
                "health_answer",
                "Я задыхаюсь, мне не хватает воздуха",
                reply_contains=("экстренную медицинскую службу", "Не оставайтесь"),
            ),
        ),
        expected=ExpectedState(health_scores=(34,)),
    ),
    Scenario(
        name="health-prolonged-weakness-recommends-doctor-and-observations",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "3"),
            ScenarioStep("health_answer", "4"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("health_answer", "6"),
            ScenarioStep("health_answer", "3"),
            ScenarioStep(
                "health_answer",
                "Слабость уже несколько недель",
                reply_contains=("записаться", "наблюдения", "не диагноз"),
                reply_excludes=("назначить", "принимайте"),
            ),
        ),
        expected=ExpectedState(health_scores=(38,)),
    ),
    Scenario(
        name="health-reminder-explicit-opt-in-and-opt-out",
        steps=(
            ScenarioStep(
                "command",
                "/health_reminder_on 20:15",
                reply_contains=("добровольное напоминание включено", "20:15"),
            ),
            ScenarioStep(
                "command",
                "/health_reminder_off",
                reply_contains=("отключено",),
            ),
        ),
        expected=ExpectedState(
            health_reminder_enabled=False,
            health_reminder_time="20:15",
            health_reminder_schedules=("20:15",),
            health_reminder_removals=1,
        ),
    ),
    Scenario(
        name="health-reminder-default-invalid-update-and-repeat-off",
        steps=(
            ScenarioStep(
                "command",
                "/health_reminder_on 24:00",
                reply_contains=("формате HH:MM",),
            ),
            ScenarioStep(
                "command",
                "/health_reminder_on",
                reply_contains=("20:00", "добровольное"),
            ),
            ScenarioStep(
                "command",
                "/health_reminder_on 07:05",
                reply_contains=("07:05",),
            ),
            ScenarioStep("command", "/health_reminder_off", reply_contains=("отключено",)),
            ScenarioStep(
                "command",
                "/health_reminder_off",
                reply_contains=("отключено",),
            ),
        ),
        expected=ExpectedState(
            health_reminder_enabled=False,
            health_reminder_time="07:05",
            health_reminder_schedules=("20:00", "07:05"),
            health_reminder_removals=2,
        ),
    ),
)


DOCTOR_PREP_SCENARIOS = (
    Scenario(
        name="doctor-prep-full-factual-flow-without-health-data",
        steps=doctor_steps(),
        expected=ExpectedState(doctor_prep_count=1),
    ),
    Scenario(
        name="doctor-prep-includes-owner-health-track-dynamics",
        steps=(
            ScenarioStep("command", "/checkin"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "6"),
            ScenarioStep("health_answer", "8"),
            ScenarioStep("health_answer", "3"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "нет"),
            *doctor_steps(),
            ScenarioStep(
                "command",
                "/doctor_prepare_show 1",
                reply_contains=("Health Track: 1 check-in", "линейка 70/100"),
            ),
        ),
        expected=ExpectedState(health_scores=(70,), doctor_prep_count=1),
    ),
    Scenario(
        name="doctor-prep-cancel-and-resume-does-not-save-partial-record",
        steps=(
            ScenarioStep("command", "/doctor_prepare"),
            ScenarioStep("doctor_answer", "Головная боль"),
            ScenarioStep(
                "command",
                "/cancel",
                reply_contains=("отменена", "не создана"),
            ),
            ScenarioStep(
                "command",
                "/doctor_preparations",
                reply_contains=("пока нет",),
            ),
            *doctor_steps(reason="Головная боль"),
        ),
        expected=ExpectedState(doctor_prep_count=1),
    ),
    Scenario(
        name="doctor-prep-required-blank-answers-do-not-advance",
        steps=(
            ScenarioStep("command", "/doctor_prepare"),
            ScenarioStep("doctor_answer", "   ", reply_contains=("не должна быть пустой",)),
            ScenarioStep("doctor_answer", "Слабость", reply_contains=("Как долго",)),
            ScenarioStep("doctor_answer", " ", reply_contains=("не должна быть пустой",)),
            ScenarioStep("doctor_answer", "Две недели", reply_contains=("симптомы",)),
            ScenarioStep("doctor_answer", "  ", reply_contains=("не должны быть пустыми",)),
            ScenarioStep("doctor_answer", "Усталость к вечеру", reply_contains=("лекарства",)),
            ScenarioStep("doctor_answer", "нет", reply_contains=("вопросы",)),
            ScenarioStep("doctor_answer", "нет", reply_contains=("сохранена",)),
        ),
        expected=ExpectedState(doctor_prep_count=1),
    ),
    Scenario(
        name="doctor-prep-edit-show-and-delete-owner-record",
        steps=(
            *doctor_steps(),
            ScenarioStep(
                "command",
                "/doctor_prepare_edit 1",
                reply_contains=("Исправляем",),
            ),
            ScenarioStep("doctor_answer", "Обновлённая причина"),
            ScenarioStep("doctor_answer", "Пять дней"),
            ScenarioStep("doctor_answer", "Наблюдение изменилось"),
            ScenarioStep("doctor_answer", "витамин D"),
            ScenarioStep(
                "doctor_answer",
                "Нужна ли повторная консультация?",
                reply_contains=("Обновлённая причина",),
            ),
            ScenarioStep(
                "command",
                "/doctor_prepare_show 1",
                reply_contains=("Обновлённая причина", "витамин D"),
            ),
            ScenarioStep("command", "/doctor_prepare_delete 1", reply_contains=("удалена",)),
            ScenarioStep("command", "/doctor_preparations", reply_contains=("пока нет",)),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="doctor-prep-owner-isolation-for-list-show-edit-delete",
        steps=(
            *doctor_steps(reason="Приватная причина 7c2"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep("command", "/doctor_preparations", reply_contains=("пока нет",)),
            ScenarioStep("command", "/doctor_prepare_show 1", reply_contains=("не найдена",)),
            ScenarioStep("command", "/doctor_prepare_edit 1", reply_contains=("не найдена",)),
            ScenarioStep("command", "/doctor_prepare_delete 1", reply_contains=("не найдена",)),
            ScenarioStep("switch_user", "900001:910001"),
            ScenarioStep(
                "command",
                "/doctor_prepare_show 1",
                reply_contains=("Приватная причина 7c2",),
            ),
        ),
        expected=ExpectedState(doctor_prep_count=1),
    ),
    Scenario(
        name="doctor-prep-red-flag-recommends-urgent-help-not-routine-delay",
        steps=(
            ScenarioStep("command", "/doctor_prepare"),
            ScenarioStep(
                "doctor_answer",
                "Боль в груди",
                reply_contains=(
                    "экстренную медицинскую службу",
                    "Не жди завершения опроса",
                ),
            ),
            ScenarioStep("doctor_answer", "Началось сегодня"),
            ScenarioStep(
                "doctor_answer",
                "Сильная боль в груди и трудно дышать",
                reply_contains=(
                    "экстренную медицинскую службу",
                    "Не жди завершения опроса",
                ),
            ),
            ScenarioStep("doctor_answer", "нет"),
            ScenarioStep(
                "doctor_answer",
                "Что делать?",
                reply_contains=(
                    "экстренную медицинскую службу",
                    "не заменяют срочную помощь",
                ),
                reply_excludes=("принимайте", "диагноз:"),
            ),
        ),
        expected=ExpectedState(doctor_prep_count=1),
    ),
    Scenario(
        name="doctor-prep-prolonged-weakness-suggests-visit-observation-list",
        steps=doctor_steps(
            reason="Слабость",
            duration="Несколько недель",
            symptoms="Слабость усиливается к вечеру",
        )[:-1]
        + (
            ScenarioStep(
                "doctor_answer",
                "Что важно наблюдать?",
                reply_contains=("записаться", "подготовить наблюдения", "не диагноз"),
                reply_excludes=("назначить лечение",),
            ),
        ),
        expected=ExpectedState(doctor_prep_count=1),
    ),
    Scenario(
        name="doctor-prep-creates-one-generic-reminder-task-idempotently",
        steps=(
            *doctor_steps(reason="Чувствительная причина не для reminder"),
            ScenarioStep(
                "command",
                "/doctor_prepare_task 1 через 2 часа",
                reply_contains=("Задача «Записаться к врачу» создана", "Reminder"),
                reply_excludes=("Чувствительная причина",),
            ),
            ScenarioStep(
                "command",
                "/doctor_prepare_task 1 через 2 часа",
                reply_contains=("дубликат не добавлен",),
            ),
        ),
        expected=ExpectedState(
            inbox=(InboxState("Записаться к врачу", "task", "doctor_prepare"),),
            doctor_prep_count=1,
            task_reminder_count=1,
        ),
    ),
    Scenario(
        name="doctor-prep-task-invalid-time-and-foreign-record-create-nothing",
        steps=(
            *doctor_steps(),
            ScenarioStep(
                "command",
                "/doctor_prepare_task 1 когда-нибудь",
                reply_contains=("Не понял будущее время",),
            ),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep(
                "command",
                "/doctor_prepare_task 1 через 2 часа",
                reply_contains=("не найдена",),
            ),
        ),
        expected=ExpectedState(doctor_prep_count=1),
    ),
)

DOCTOR_SEARCH_SCENARIOS = (
    Scenario(
        name="doctor-search-official-svetogorsk-then-vyborg-is-read-only",
        steps=(
            ScenarioStep(
                "command",
                "/location Светогорск → Выборг",
                reply_contains=("Локация сохранена", "Светогорск → Выборг"),
            ),
            ScenarioStep(
                "command",
                "/doctor_find",
                reply_contains=(
                    "Светогорск",
                    "Выборг",
                    "Терапевт",
                    "122",
                    "https://zdrav.lenreg.ru/",
                    "+7 (81378) 36-268",
                    "+7 (81378) 2-83-46",
                ),
                reply_excludes=("диагноз", "частная клиника"),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="doctor-search-task-and-reminder-are-idempotent",
        steps=(
            ScenarioStep("command", "/location Саратов", reply_contains=("Саратов",)),
            ScenarioStep(
                "command",
                "/doctor_find_task через 2 часа",
                reply_contains=("Записаться к терапевту", "Reminder"),
            ),
            ScenarioStep(
                "command",
                "/doctor_find_task через 2 часа",
                reply_contains=("дубликат не добавлен",),
            ),
        ),
        expected=ExpectedState(
            inbox=(
                InboxState(
                    "Записаться к терапевту: Саратов",
                    "task",
                    "doctor_search",
                ),
            ),
            task_reminder_count=1,
        ),
    ),
    Scenario(
        name="doctor-search-invalid-time-and-owner-isolation",
        steps=(
            ScenarioStep(
                "command",
                "/location Светогорск → Выборг",
                reply_contains=("Светогорск → Выборг",),
            ),
            ScenarioStep(
                "command",
                "/doctor_find_task когда-нибудь",
                reply_contains=("Не понял будущее время",),
            ),
            ScenarioStep(
                "command",
                "/doctor_find_task через 3 часа",
                reply_contains=("Записаться к терапевту",),
            ),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep("command", "/location Саратов", reply_contains=("Саратов",)),
            ScenarioStep(
                "command",
                "/doctor_find_task через 3 часа",
                reply_contains=("Записаться к терапевту",),
            ),
        ),
        expected=ExpectedState(
            inbox=(
                InboxState(
                    "Записаться к терапевту: Саратов",
                    "task",
                    "doctor_search",
                ),
                InboxState(
                    "Записаться к терапевту: Светогорск → Выборг",
                    "task",
                    "doctor_search",
                ),
            ),
            task_reminder_count=2,
        ),
    ),
)

VISION_PAGINATION_ITEMS = (
    ("money", "Денежная цель 1"),
    ("money", "Денежная цель 2"),
    ("money", "Денежная цель 3"),
    ("travel", "Путешествие 1"),
    ("travel", "Путешествие 2"),
    ("travel", "Путешествие 3"),
)

VISION_PAGINATION_STEPS = tuple(
    step
    for category, wish in VISION_PAGINATION_ITEMS
    for step in (
        ScenarioStep("command", "/vision"),
        ScenarioStep("vision_callback", "add"),
        ScenarioStep("vision_callback", category),
        ScenarioStep("text", wish),
        ScenarioStep("vision_callback", "skip"),
        ScenarioStep("vision_callback", "skip"),
        ScenarioStep("vision_callback", "skip"),
        ScenarioStep("vision_callback", "confirm"),
    )
)


def vision_image_item_steps(category: str, wish: str) -> tuple[ScenarioStep, ...]:
    return (
        ScenarioStep("command", "/vision"),
        ScenarioStep("vision_callback", "add"),
        ScenarioStep("vision_callback", category),
        ScenarioStep("text", wish),
        ScenarioStep("vision_callback", "skip"),
        ScenarioStep("vision_callback", "skip"),
        ScenarioStep("vision_callback", "skip"),
        ScenarioStep("vision_callback", "confirm"),
    )


VISION_IMAGE_CREATE_STEPS = (
    ScenarioStep("command", "/vision"),
    ScenarioStep("vision_callback", "add"),
    ScenarioStep("vision_callback", "travel"),
    ScenarioStep("text", "Личная мечта с фотографией"),
    ScenarioStep("vision_callback", "skip"),
    ScenarioStep("vision_callback", "skip"),
    ScenarioStep("vision_callback", "skip"),
    ScenarioStep("vision_callback", "confirm"),
)

VISION_SCENARIOS = (
    Scenario(
        name="vision-image-jpeg-photo-confirm-repeat-render-and-download",
        steps=(
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd", reply_contains=("Отправь одно фото",)),
            ScenarioStep("vision_photo", "jpeg:red", reply_contains=("Preview",)),
            ScenarioStep("vision_capture_callback", "imageconfirm"),
            ScenarioStep(
                "vision_replay_callback",
                "imageconfirm",
                reply_contains=("Фото сохранено", "Личное фото"),
            ),
            ScenarioStep(
                "vision_replay_callback",
                "imageconfirm",
                reply_contains=("устарело",),
            ),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "render"),
            ScenarioStep("vision_callback", "renderall", reply_contains=("Активных желаний: 1",)),
            ScenarioStep("vision_callback", "download", reply_contains=("Активных желаний: 1",)),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState(
                    "travel",
                    "Личная мечта с фотографией",
                    "active",
                    has_image=True,
                ),
            ),
            vision_image_count=1,
        ),
    ),
    Scenario(
        name="vision-image-png-cancel-then-webp-document-confirm",
        steps=(
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_document", "png:blue", reply_contains=("Preview",)),
            ScenarioStep("vision_callback", "imagecancel", reply_contains=("отменено",)),
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_document", "webp:green", reply_contains=("Preview",)),
            ScenarioStep("vision_callback", "imageconfirm", reply_contains=("Фото сохранено",)),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState(
                    "travel",
                    "Личная мечта с фотографией",
                    "active",
                    has_image=True,
                ),
            ),
            vision_image_count=1,
        ),
    ),
    Scenario(
        name="vision-image-replace-cancel-keeps-old-then-atomic-replace",
        steps=(
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_photo", "jpeg:red"),
            ScenarioStep("vision_callback", "imageconfirm"),
            ScenarioStep("vision_callback", "imagereplace"),
            ScenarioStep("vision_document", "png:blue"),
            ScenarioStep("vision_callback", "imagecancel", reply_contains=("отменено",)),
            ScenarioStep("vision_callback", "imagereplace"),
            ScenarioStep("vision_document", "png:blue"),
            ScenarioStep("vision_callback", "imageconfirm", reply_contains=("Фото сохранено",)),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState(
                    "travel",
                    "Личная мечта с фотографией",
                    "active",
                    has_image=True,
                ),
            ),
            vision_image_count=1,
        ),
    ),
    Scenario(
        name="vision-image-delete-owner-bound-and-single-use",
        steps=(
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_photo", "jpeg:red"),
            ScenarioStep("vision_callback", "imageconfirm"),
            ScenarioStep("vision_callback", "imagedeleteask"),
            ScenarioStep("vision_capture_callback", "imagedelete"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep(
                "vision_replay_callback",
                "imagedelete",
                reply_contains=("устарело",),
            ),
            ScenarioStep("switch_user", "900001:910001"),
            ScenarioStep(
                "vision_replay_callback",
                "imagedelete",
                reply_contains=("Фото удалено",),
            ),
            ScenarioStep(
                "vision_replay_callback",
                "imagedelete",
                reply_contains=("устарело",),
            ),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("travel", "Личная мечта с фотографией", "active"),),
        ),
    ),
    Scenario(
        name="vision-image-invalid-magic-mime-animation-size-and-pixels-recover",
        steps=(
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_document", "mismatch", reply_contains=("Файл отклонён",)),
            ScenarioStep("vision_document", "corrupt", reply_contains=("Файл отклонён",)),
            ScenarioStep(
                "vision_document",
                "animated-gif",
                reply_contains=("GIF", "не поддерживаются"),
            ),
            ScenarioStep("vision_photo", "oversize-meta", reply_contains=("Файл отклонён",)),
            ScenarioStep("vision_photo", "multipixel-meta", reply_contains=("Файл отклонён",)),
            ScenarioStep("vision_document", "png:purple", reply_contains=("Preview",)),
            ScenarioStep("vision_callback", "imageconfirm"),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState(
                    "travel",
                    "Личная мечта с фотографией",
                    "active",
                    has_image=True,
                ),
            ),
            vision_image_count=1,
        ),
    ),
    Scenario(
        name="vision-image-restart-drops-unconfirmed-upload-capability",
        steps=(
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_capture_callback", "imagecancel"),
            ScenarioStep("restart"),
            ScenarioStep("vision_photo", "jpeg:red"),
            ScenarioStep(
                "vision_replay_callback",
                "imagecancel",
                reply_contains=("устарело",),
            ),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("travel", "Личная мечта с фотографией", "active"),),
        ),
    ),
    Scenario(
        name="vision-image-identical-photos-remain-owner-isolated",
        steps=(
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_photo", "jpeg:red"),
            ScenarioStep("vision_callback", "imageconfirm"),
            ScenarioStep("switch_user", "900002:910002"),
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_photo", "jpeg:red"),
            ScenarioStep("vision_callback", "imageconfirm"),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState(
                    "travel",
                    "Личная мечта с фотографией",
                    "active",
                    has_image=True,
                ),
                VisionState(
                    "travel",
                    "Личная мечта с фотографией",
                    "active",
                    has_image=True,
                ),
            ),
            vision_image_count=2,
        ),
    ),
    Scenario(
        name="vision-image-concurrent-begin-is-rejected-without-overwrite",
        steps=(
            *VISION_IMAGE_CREATE_STEPS,
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep(
                "vision_raw_callback",
                "vision:imageadd:1",
                reply_contains=("текущую операцию",),
            ),
            ScenarioStep("vision_photo", "jpeg:red"),
            ScenarioStep("vision_callback", "imageconfirm"),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState(
                    "travel",
                    "Личная мечта с фотографией",
                    "active",
                    has_image=True,
                ),
            ),
            vision_image_count=1,
        ),
    ),
    Scenario(
        name="vision-image-outside-active-flow-is-never-attached",
        steps=(ScenarioStep("vision_photo", "jpeg:red"),),
        expected=ExpectedState(),
    ),
    Scenario(
        name="vision-render-empty-map-is-safe-and-read-only",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep(
                "vision_callback",
                "render",
                reply_contains=("Сначала добавь желание",),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="vision-render-voice-photo-download-and-repeat-are-private",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "travel"),
            ScenarioStep("voice", "Увидеть северное сияние ✨"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "render", reply_contains=("Что визуализировать",)),
            ScenarioStep("vision_capture_callback", "renderall"),
            ScenarioStep(
                "vision_replay_callback",
                "renderall",
                reply_contains=("Активных желаний: 1", "Вся карта"),
            ),
            ScenarioStep("vision_capture_callback", "download"),
            ScenarioStep(
                "vision_replay_callback",
                "download",
                reply_contains=("Активных желаний: 1",),
            ),
            ScenarioStep(
                "vision_replay_callback",
                "download",
                reply_contains=("устарел",),
            ),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("travel", "Увидеть северное сияние ✨", "active"),),
        ),
    ),
    Scenario(
        name="vision-render-category-callback-is-owner-bound",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "travel"),
            ScenarioStep("text", "Побывать в Карелии"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "money"),
            ScenarioStep("text", "Создать финансовую подушку"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "render"),
            ScenarioStep("vision_capture_callback", "rendertravel"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep(
                "vision_replay_callback",
                "rendertravel",
                reply_contains=("устарел",),
            ),
            ScenarioStep("switch_user", "900001:910001"),
            ScenarioStep(
                "vision_replay_callback",
                "rendertravel",
                reply_contains=("Активных желаний: 1", "Путешествия"),
            ),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState("money", "Создать финансовую подушку", "active"),
                VisionState("travel", "Побывать в Карелии", "active"),
            ),
        ),
    ),
    Scenario(
        name="vision-render-multiple-pages-has-stable-page-captions",
        steps=(
            *VISION_PAGINATION_STEPS,
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "render"),
            ScenarioStep(
                "vision_callback",
                "renderall",
                reply_contains=("страница 1/2", "страница 2/2"),
            ),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState("money", "Денежная цель 1", "active"),
                VisionState("money", "Денежная цель 2", "active"),
                VisionState("money", "Денежная цель 3", "active"),
                VisionState("travel", "Путешествие 1", "active"),
                VisionState("travel", "Путешествие 2", "active"),
                VisionState("travel", "Путешествие 3", "active"),
            ),
        ),
    ),
    Scenario(
        name="vision-render-concurrent-owner-request-is-rejected",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "home"),
            ScenarioStep("text", "Создать спокойное пространство"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "render"),
            ScenarioStep("vision_capture_callback", "renderall"),
            ScenarioStep("vision_hold_render"),
            ScenarioStep(
                "vision_replay_callback",
                "renderall",
                reply_contains=("уже создаётся",),
            ),
            ScenarioStep("vision_release_render"),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("home", "Создать спокойное пространство", "active"),),
        ),
    ),
    Scenario(
        name="vision-render-selection-cancel-is-single-use",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "other"),
            ScenarioStep("text", "Сохранить важную мечту"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "render"),
            ScenarioStep("vision_capture_callback", "rendercancel"),
            ScenarioStep(
                "vision_replay_callback",
                "rendercancel",
                reply_contains=("отменена",),
            ),
            ScenarioStep(
                "vision_replay_callback",
                "rendercancel",
                reply_contains=("устарел",),
            ),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("other", "Сохранить важную мечту", "active"),),
        ),
    ),
    Scenario(
        name="vision-full-voice-skip-confirm-task-idempotent-without-llm",
        steps=(
            ScenarioStep("command", "/vision", reply_contains=("Карта желаний",)),
            ScenarioStep("vision_callback", "add", reply_contains=("Выбери категорию",)),
            ScenarioStep(
                "vision_callback",
                "travel",
                reply_contains=("Сформулируй желание",),
            ),
            ScenarioStep(
                "voice",
                "Увидеть северное сияние",
                reply_contains=("Почему это важно",),
            ),
            ScenarioStep("vision_callback", "skip", reply_contains=("Желаемая дата",)),
            ScenarioStep("vision_callback", "skip", reply_contains=("первый небольшой шаг",)),
            ScenarioStep(
                "text",
                "Выбрать месяц поездки",
                reply_contains=("Preview карточки", "Увидеть северное сияние"),
            ),
            ScenarioStep("vision_callback", "confirm", reply_contains=("Желание сохранено",)),
            ScenarioStep(
                "vision_callback",
                "confirm",
                reply_contains=("действие устарело",),
            ),
            ScenarioStep("vision_callback", "task", reply_contains=("Задача создана",)),
            ScenarioStep(
                "vision_callback",
                "task",
                reply_contains=("дубликат не добавлен",),
            ),
        ),
        expected=ExpectedState(
            inbox=(
                InboxState(
                    "Шаг к желанию: Увидеть северное сияние",
                    "task",
                    "vision",
                ),
            ),
            vision_items=(
                VisionState(
                    "travel",
                    "Увидеть северное сияние",
                    "active",
                    True,
                ),
            ),
        ),
    ),
    Scenario(
        name="vision-cancel-removes-persistent-partial-draft",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "home"),
            ScenarioStep("text", "Создать уютный дом"),
            ScenarioStep(
                "command",
                "/cancel",
                reply_contains=("ничего не сохранено",),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="vision-persistent-draft-resumes-after-process-restart",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "home"),
            ScenarioStep("text", "Дом у озера"),
            ScenarioStep("restart"),
            ScenarioStep(
                "command",
                "/vision",
                reply_contains=("незавершённая", "Почему это важно"),
            ),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("home", "Дом у озера", "active"),),
        ),
    ),
    Scenario(
        name="vision-group-chat-is-blocked-before-private-content-handler",
        steps=(
            ScenarioStep(
                "group_command",
                "/vision",
                reply_contains=("только в личном чате",),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="vision-category-groups-counts-and-pagination",
        steps=(
            *VISION_PAGINATION_STEPS,
            ScenarioStep("command", "/vision"),
            ScenarioStep(
                "vision_callback",
                "list:active",
                reply_contains=("Моя карта — 6", "Деньги (3)", "Путешествия (3)"),
            ),
            ScenarioStep(
                "vision_callback",
                "list:active:1",
                reply_contains=("Моя карта — 6", "Путешествия (3)"),
            ),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState("money", "Денежная цель 1", "active"),
                VisionState("money", "Денежная цель 2", "active"),
                VisionState("money", "Денежная цель 3", "active"),
                VisionState("travel", "Путешествие 1", "active"),
                VisionState("travel", "Путешествие 2", "active"),
                VisionState("travel", "Путешествие 3", "active"),
            ),
        ),
    ),
    Scenario(
        name="vision-manage-edit-status-archive-restore-delete",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "money"),
            ScenarioStep("text", "Создать финансовую подушку"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
            ScenarioStep("vision_callback", "status", reply_contains=("Статус",)),
            ScenarioStep("vision_callback", "status", reply_contains=("Статус",)),
            ScenarioStep("vision_callback", "edit", reply_contains=("Что изменить",)),
            ScenarioStep("vision_callback", "editwish", reply_contains=("Пришли желание",)),
            ScenarioStep("text", "Обновлённая финансовая цель"),
            ScenarioStep("vision_callback", "archive", reply_contains=("архивирована",)),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "list:active", reply_contains=("пока нет",)),
            ScenarioStep("vision_callback", "list:archived", reply_contains=("Архив",)),
            ScenarioStep("vision_callback", "view", reply_contains=("в архиве",)),
            ScenarioStep("vision_callback", "status", reply_contains=("Статус",)),
            ScenarioStep(
                "vision_callback",
                "deleteask",
                reply_contains=("подтверждения",),
            ),
            ScenarioStep("vision_callback", "delete", reply_contains=("удалена",)),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="vision-identical-wishes-remain-owner-isolated",
        steps=(
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "growth_creativity"),
            ScenarioStep("text", "Научиться рисовать"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep(
                "vision_raw_callback",
                "vision:view:1",
                reply_contains=("недоступна",),
            ),
            ScenarioStep(
                "vision_raw_callback",
                "vision:status:1:achieved",
                reply_contains=("недоступна",),
            ),
            ScenarioStep(
                "vision_raw_callback",
                "vision:task:1",
                reply_contains=("недоступна",),
            ),
            ScenarioStep(
                "vision_raw_callback",
                "vision:delete:1:999999:1",
                reply_contains=("недоступна",),
            ),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "add"),
            ScenarioStep("vision_callback", "growth_creativity"),
            ScenarioStep("voice", "Научиться рисовать"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "skip"),
            ScenarioStep("vision_callback", "confirm"),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState(
                    "growth_creativity",
                    "Научиться рисовать",
                    "active",
                ),
                VisionState(
                    "growth_creativity",
                    "Научиться рисовать",
                    "active",
                ),
            )
        ),
    ),
    Scenario(
        name="vision-personal-jpeg-photo-confirm-render-download-is-idempotent",
        steps=(
            *vision_image_item_steps("travel", "Увидеть океан"),
            ScenarioStep("vision_callback", "imageadd", reply_contains=("Отправь одно фото",)),
            ScenarioStep("vision_photo", "jpeg", reply_contains=("Preview",)),
            ScenarioStep("vision_capture_callback", "imageconfirm"),
            ScenarioStep("vision_replay_callback", "imageconfirm", reply_contains=("сохранено",)),
            ScenarioStep("vision_replay_callback", "imageconfirm", reply_contains=("устарело",)),
            ScenarioStep("command", "/vision"),
            ScenarioStep("vision_callback", "render"),
            ScenarioStep("vision_callback", "renderall", reply_contains=("Активных желаний: 1",)),
            ScenarioStep("vision_callback", "download", reply_contains=("Активных желаний: 1",)),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("travel", "Увидеть океан", "active", has_image=True),),
            vision_image_count=1,
        ),
    ),
    Scenario(
        name="vision-personal-png-document-preview-cancel-does-not-attach",
        steps=(
            *vision_image_item_steps("home", "Создать светлый дом"),
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_document", "png", reply_contains=("Preview",)),
            ScenarioStep("vision_callback", "imagecancel", reply_contains=("отменено",)),
            ScenarioStep("vision_document", "png"),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("home", "Создать светлый дом", "active"),),
        ),
    ),
    Scenario(
        name="vision-personal-webp-replace-delete-and-callbacks-are-owner-bound",
        steps=(
            *vision_image_item_steps("growth_creativity", "Научиться фотографии"),
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_document", "webp"),
            ScenarioStep("vision_callback", "imageconfirm"),
            ScenarioStep("vision_callback", "imagereplace"),
            ScenarioStep("vision_document", "png"),
            ScenarioStep("vision_callback", "imageconfirm"),
            ScenarioStep("vision_callback", "imagedeleteask"),
            ScenarioStep("vision_capture_callback", "imagedelete"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep("vision_replay_callback", "imagedelete", reply_contains=("недоступна",)),
            ScenarioStep("switch_user", "900001:910001"),
            ScenarioStep("vision_replay_callback", "imagedelete", reply_contains=("Фото удалено",)),
            ScenarioStep("vision_replay_callback", "imagedelete", reply_contains=("недоступна",)),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("growth_creativity", "Научиться фотографии", "active"),),
        ),
    ),
    Scenario(
        name="vision-personal-invalid-media-is-rejected-before-valid-retry",
        steps=(
            *vision_image_item_steps("other", "Сохранить личный образ"),
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_document", "mismatch", reply_contains=("отклонён",)),
            ScenarioStep("vision_document", "corrupt", reply_contains=("отклонён",)),
            ScenarioStep("vision_document", "animated-gif", reply_contains=("отклонён",)),
            ScenarioStep("vision_document", "pdf", reply_contains=("отклонён",)),
            ScenarioStep("vision_photo", "oversize-meta", reply_contains=("отклонён",)),
            ScenarioStep("vision_photo", "multipixel-meta", reply_contains=("отклонён",)),
            ScenarioStep("vision_photo", "jpeg", reply_contains=("Preview",)),
            ScenarioStep("vision_callback", "imageconfirm"),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState("other", "Сохранить личный образ", "active", has_image=True),
            ),
            vision_image_count=1,
        ),
    ),
    Scenario(
        name="vision-personal-upload-restart-and-stale-callback-save-nothing",
        steps=(
            *vision_image_item_steps("money", "Финансовая свобода"),
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_capture_callback", "imagecancel"),
            ScenarioStep("restart"),
            ScenarioStep("vision_photo", "jpeg"),
            ScenarioStep("vision_replay_callback", "imagecancel", reply_contains=("недоступна",)),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("money", "Финансовая свобода", "active"),),
        ),
    ),
    Scenario(
        name="vision-personal-identical-images-remain-owner-isolated",
        steps=(
            *vision_image_item_steps("health_energy", "Больше энергии"),
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_photo", "jpeg"),
            ScenarioStep("vision_callback", "imageconfirm"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep(
                "vision_raw_callback", "vision:imageadd:1", reply_contains=("недоступна",)
            ),
            *vision_image_item_steps("health_energy", "Больше энергии"),
            ScenarioStep("vision_callback", "imageadd"),
            ScenarioStep("vision_photo", "jpeg"),
            ScenarioStep("vision_callback", "imageconfirm"),
        ),
        expected=ExpectedState(
            vision_items=(
                VisionState("health_energy", "Больше энергии", "active", has_image=True),
                VisionState("health_energy", "Больше энергии", "active", has_image=True),
            ),
            vision_image_count=2,
        ),
    ),
    Scenario(
        name="vision-personal-concurrent-add-flow-has-one-owner-session",
        steps=(
            *vision_image_item_steps("travel", "Поехать в горы"),
            ScenarioStep("vision_capture_callback", "imageadd"),
            ScenarioStep(
                "vision_replay_callback", "imageadd", reply_contains=("Отправь одно фото",)
            ),
            ScenarioStep(
                "vision_replay_callback",
                "imageadd",
                reply_contains=("Сначала заверши",),
            ),
            ScenarioStep("vision_photo", "jpeg-second"),
            ScenarioStep("vision_callback", "imageconfirm"),
        ),
        expected=ExpectedState(
            vision_items=(VisionState("travel", "Поехать в горы", "active", has_image=True),),
            vision_image_count=1,
        ),
    ),
)

TIMEZONE_REGRESSION_SCENARIOS = tuple(
    Scenario(
        name=f"timezone-onboarding-{name}",
        steps=(
            ScenarioStep(
                "timezone_onboarding",
                f"{answer}=>{expected}",
                reply_contains=(f"timezone={expected}",),
            ),
        ),
        expected=ExpectedState(),
    )
    for name, answer, expected in (
        ("moscow-en", "Moscow", "Europe/Moscow"),
        ("moscow-ru", "Москва", "Europe/Moscow"),
        ("moscow-short", "МСК", "Europe/Moscow"),
        ("moscow-gmt3", "GMT+3", "Europe/Moscow"),
        ("saratov-ru", "Саратов", "Europe/Saratov"),
        ("saratov-gmt4", "GMT+4", "Europe/Saratov"),
    )
)

NAVIGATION_SCENARIOS = (
    Scenario(
        name="navigation-menu-help-quick-start-and-back-have-no-dead-end",
        steps=(
            ScenarioStep("command", "/menu", reply_contains=("Главное меню",)),
            ScenarioStep("navigation_callback", "help", reply_contains=("Помощь",)),
            ScenarioStep("navigation_callback", "quick", reply_contains=("Быстрый старт",)),
            ScenarioStep("navigation_callback", "help", reply_contains=("Помощь",)),
            ScenarioStep("navigation_callback", "root", reply_contains=("Главное меню",)),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-every-main-section-has-real-back-path",
        steps=(
            ScenarioStep("command", "/menu"),
            ScenarioStep("navigation_callback", "ideas", reply_contains=("Идеи и заметки",)),
            ScenarioStep("navigation_callback", "root"),
            ScenarioStep("navigation_callback", "tasks", reply_contains=("Задачи",)),
            ScenarioStep("navigation_callback", "root"),
            ScenarioStep("navigation_callback", "vision", reply_contains=("Карта желаний",)),
            ScenarioStep("navigation_callback", "root"),
            ScenarioStep("navigation_callback", "health", reply_contains=("Здоровье",)),
            ScenarioStep("navigation_callback", "root"),
            ScenarioStep("navigation_callback", "doctor", reply_contains=("Врач",)),
            ScenarioStep("navigation_callback", "root"),
            ScenarioStep("navigation_callback", "profile", reply_contains=("Профиль",)),
            ScenarioStep("navigation_callback", "root"),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-natural-text-menu-is-deterministic-and-read-only",
        steps=(ScenarioStep("text", "ГЛАВНОЕ МЕНЮ!!!", reply_contains=("Главное меню",)),),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-natural-voice-help-is-deterministic-and-read-only",
        steps=(ScenarioStep("voice", "Как пользоваться ботом?", reply_contains=("Помощь",)),),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-unified-doctor-section-preserves-legacy-actions",
        steps=(
            ScenarioStep("command", "/doctor", reply_contains=("Врач",)),
            ScenarioStep(
                "navigation_callback",
                "doctor_find",
                reply_contains=("настрой локацию", "Навигация"),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-health-flow-can-continue-without-losing-answers",
        steps=(
            ScenarioStep("command", "/menu"),
            ScenarioStep("navigation_callback", "health"),
            ScenarioStep("navigation_callback", "checkin", reply_contains=("Энергия",)),
            ScenarioStep("health_answer", "7", reply_contains=("Сон",)),
            ScenarioStep("text", "Меню", reply_contains=("не завершён сценарий",)),
            ScenarioStep("navigation_callback", "continue", reply_contains=("Продолжаем",)),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "3"),
            ScenarioStep("health_answer", "7"),
            ScenarioStep("health_answer", "нет", reply_contains=("70/100",)),
        ),
        expected=ExpectedState(health_scores=(70,)),
    ),
    Scenario(
        name="navigation-health-flow-explicit-exit-clears-only-current-state",
        steps=(
            ScenarioStep("command", "/menu"),
            ScenarioStep("navigation_callback", "health"),
            ScenarioStep("navigation_callback", "checkin"),
            ScenarioStep("health_answer", "5"),
            ScenarioStep("text", "Главное меню", reply_contains=("не завершён сценарий",)),
            ScenarioStep(
                "navigation_callback", "exit", reply_contains=("остановлен", "Главное меню")
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-flow-callback-is-owner-bound-stale-and-single-use",
        steps=(
            ScenarioStep("command", "/menu"),
            ScenarioStep("navigation_callback", "health"),
            ScenarioStep("navigation_callback", "checkin"),
            ScenarioStep("text", "Меню"),
            ScenarioStep("navigation_capture_callback", "exit"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep("navigation_replay_callback", "exit", reply_contains=("устарела",)),
            ScenarioStep("switch_user", "900001:910001"),
            ScenarioStep("navigation_replay_callback", "exit", reply_contains=("остановлен",)),
            ScenarioStep("navigation_replay_callback", "exit", reply_contains=("устарела",)),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-doctor-preparation-entry-and-exit-save-no-medical-record",
        steps=(
            ScenarioStep("command", "/doctor"),
            ScenarioStep(
                "navigation_callback", "doctor_prepare", reply_contains=("причина обращения",)
            ),
            ScenarioStep("text", "Меню", reply_contains=("не завершён сценарий",)),
            ScenarioStep("navigation_callback", "exit", reply_contains=("остановлен",)),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-ordinary-menu-word-is-content-not-command",
        llm_stubs=(
            capture(
                "Добавь меню ужина в заметки",
                intent="inbox_note",
                kind="note",
                title="Меню ужина",
            ),
        ),
        steps=(
            ScenarioStep(
                "text",
                "Добавь меню ужина в заметки",
                reply_contains=("Меню ужина",),
                reply_excludes=("Главное меню",),
            ),
        ),
        expected=ExpectedState(
            drafts=(DraftState("Меню ужина", "note", "preview", "text"),),
            llm_inputs=("Добавь меню ужина в заметки",),
        ),
    ),
    Scenario(
        name="navigation-forged-unknown-callback-fails-closed",
        steps=(
            ScenarioStep(
                "navigation_raw_callback",
                "nav:section:medical-private-record",
                reply_contains=("устарела",),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-group-chat-is-blocked-before-menu-handler",
        steps=(
            ScenarioStep(
                "group_command",
                "/menu",
                reply_contains=("только в личном чате",),
                reply_excludes=("Главное меню",),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="navigation-identical-menus-remain-user-isolated",
        steps=(
            ScenarioStep("command", "/menu"),
            ScenarioStep("switch_user", "900002:910002"),
            ScenarioStep("command", "/menu"),
            ScenarioStep("navigation_callback", "ideas"),
            ScenarioStep("navigation_callback", "inbox", reply_contains=("Inbox",)),
        ),
        expected=ExpectedState(),
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
    *RESOLVED_SAVE_REGRESSION_SCENARIOS,
    *RESOLVED_NEGATIVE_REGRESSION_SCENARIOS,
    *HEALTH_SCENARIOS,
    *DOCTOR_PREP_SCENARIOS,
    *DOCTOR_SEARCH_SCENARIOS,
    *TIMEZONE_REGRESSION_SCENARIOS,
    *VISION_SCENARIOS,
    *NAVIGATION_SCENARIOS,
)
