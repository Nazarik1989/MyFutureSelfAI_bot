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


THERAPIST = "Записаться к терапевту и разобраться с причиной слабости"
ORDINARY_SAVE_TEXT = "Хочу понять, стоит ли сохранять полезные статьи в инбокс для чтения"


SCENARIOS = (
    Scenario(
        name="focused draft saves through voice and repeat is idempotent",
        llm_stubs=(capture(THERAPIST, intent="inbox_task", kind="task", title=THERAPIST),),
        steps=(
            ScenarioStep("text", THERAPIST, reply_contains=("Заголовок", THERAPIST)),
            ScenarioStep(
                "voice",
                "Сохраним это в inbox?!",
                reply_contains=("Сохранено в inbox по голосовой команде",),
                reply_excludes=("Не сохраняю",),
            ),
            ScenarioStep(
                "text",
                "Сохрани в инбокс",
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
        name="negative text and voice commands discard without saving",
        llm_stubs=(
            capture(
                "Идея: вечерняя прогулка",
                intent="inbox_idea",
                kind="idea",
                title="Вечерняя прогулка",
            ),
            capture(
                "Идея: читать перед сном",
                intent="inbox_idea",
                kind="idea",
                title="Читать перед сном",
            ),
        ),
        steps=(
            ScenarioStep("text", "Идея: вечерняя прогулка", reply_contains=("Заголовок",)),
            ScenarioStep(
                "text",
                "Не сохраняй в инбокс",
                reply_contains=("удалена без сохранения",),
            ),
            ScenarioStep("voice", "Идея: читать перед сном", reply_contains=("Заголовок",)),
            ScenarioStep(
                "voice",
                "Не надо сохранять",
                reply_contains=("удалена без сохранения",),
            ),
        ),
        expected=ExpectedState(
            drafts=(
                DraftState("Вечерняя прогулка", "idea", "discarded", "text"),
                DraftState("Читать перед сном", "idea", "discarded", "voice"),
            ),
            llm_inputs=("Идея: вечерняя прогулка", "Идея: читать перед сном"),
        ),
    ),
    Scenario(
        name="multiple drafts without focus require explicit choice",
        llm_stubs=(
            capture("Первая идея", intent="inbox_idea", kind="idea", title="Первая"),
            capture("Вторая идея", intent="inbox_idea", kind="idea", title="Вторая"),
        ),
        steps=(
            ScenarioStep("text", "Первая идея", reply_contains=("Заголовок",)),
            ScenarioStep("voice", "Вторая идея", reply_contains=("Заголовок",)),
            ScenarioStep("setup_clear_focus"),
            ScenarioStep(
                "voice",
                "Сохрани это в инбокс!",
                reply_contains=("К какой карточке применить команду?",),
            ),
        ),
        expected=ExpectedState(
            drafts=(
                DraftState("Вторая", "idea", "preview", "voice"),
                DraftState("Первая", "idea", "preview", "text"),
            ),
            llm_inputs=("Первая идея", "Вторая идея"),
        ),
    ),
    Scenario(
        name="save commands without drafts never call LLM",
        steps=(
            ScenarioStep("text", "Сохрани инбокс", reply_contains=("Нет одной актуальной",)),
            ScenarioStep(
                "voice",
                "Сохраним это в inbox?!",
                reply_contains=("Нет одной актуальной",),
            ),
        ),
        expected=ExpectedState(),
    ),
    Scenario(
        name="ordinary text mentioning save and inbox reaches LLM",
        llm_stubs=(
            capture(
                ORDINARY_SAVE_TEXT,
                intent="inbox_note",
                kind="note",
                title="Разбор полезных статей",
            ),
        ),
        steps=(ScenarioStep("text", ORDINARY_SAVE_TEXT, reply_contains=("Заголовок",)),),
        expected=ExpectedState(
            drafts=(DraftState("Разбор полезных статей", "note", "preview", "text"),),
            llm_inputs=(ORDINARY_SAVE_TEXT,),
        ),
    ),
    Scenario(
        name="ordinary voice mentioning save and inbox reaches LLM",
        llm_stubs=(
            capture(
                ORDINARY_SAVE_TEXT,
                intent="inbox_note",
                kind="note",
                title="Разбор полезных статей",
            ),
        ),
        steps=(ScenarioStep("voice", ORDINARY_SAVE_TEXT, reply_contains=("Заголовок",)),),
        expected=ExpectedState(
            drafts=(DraftState("Разбор полезных статей", "note", "preview", "voice"),),
            llm_inputs=(ORDINARY_SAVE_TEXT,),
        ),
    ),
    Scenario(
        name="callback and commands use the same persistent save path",
        llm_stubs=(
            capture(
                "Заметка для callback",
                intent="inbox_note",
                kind="note",
                title="Callback-проверка",
            ),
        ),
        steps=(
            ScenarioStep("text", "Заметка для callback", reply_contains=("Заголовок",)),
            ScenarioStep("callback", "save", reply_contains=("Сохранено в inbox",)),
        ),
        expected=ExpectedState(
            drafts=(DraftState("Callback-проверка", "note", "confirmed", "text"),),
            inbox=(InboxState("Callback-проверка", "note", "text"),),
            llm_inputs=("Заметка для callback",),
        ),
    ),
    Scenario(
        name="natural read command stays read only for text and voice",
        steps=(
            ScenarioStep("text", "Что у меня сохранено?", reply_contains=("Inbox пока пуст",)),
            ScenarioStep("voice", "Что у меня сохранено?!", reply_contains=("Inbox пока пуст",)),
        ),
        expected=ExpectedState(),
    ),
)
