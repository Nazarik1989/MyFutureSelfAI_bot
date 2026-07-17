import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from sqlalchemy import select
from sqlalchemy.engine import make_url

from future_self.bot import (
    DOCTOR_DURATION,
    DOCTOR_MEDICATIONS,
    DOCTOR_QUESTIONS,
    DOCTOR_REASON,
    DOCTOR_SYMPTOMS,
    HEALTH_ENERGY,
    HEALTH_MOOD,
    HEALTH_PHYSICAL,
    HEALTH_SLEEP,
    HEALTH_STRESS,
    HEALTH_SYMPTOMS,
    FutureSelfBot,
)
from future_self.config import Settings
from future_self.db import Database
from future_self.models import (
    DoctorVisitPrep,
    DraftInboxItem,
    HealthCheckIn,
    HealthReminderPreference,
    InboxItem,
    TaskReminder,
)
from future_self.schemas import IntentResult

from .fakes import (
    FakeBot,
    FakeCallbackQuery,
    FakeMessage,
    FakeVoice,
    ScriptedTranscription,
    StrictAI,
)

AUTOTEST_TELEGRAM_TOKEN = "000000:AUTOTEST_ONLY"
AUTOTEST_AI_KEY = "autotest-key"
AUTOTEST_BASE_URL = "https://invalid.autotest"

StepKind = Literal[
    "text",
    "voice",
    "callback",
    "command",
    "doctor_answer",
    "health_answer",
    "switch_user",
    "setup_clear_focus",
]


class UnsafeAutotestConfiguration(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LLMStub:
    text: str
    response: IntentResult


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    kind: StepKind
    value: str = ""
    reply_contains: tuple[str, ...] = ()
    reply_excludes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, order=True)
class DraftState:
    title: str
    kind: str
    status: str
    source: str


@dataclass(frozen=True, slots=True, order=True)
class InboxState:
    title: str
    kind: str
    source: str


@dataclass(frozen=True, slots=True)
class ExpectedState:
    drafts: tuple[DraftState, ...] = ()
    inbox: tuple[InboxState, ...] = ()
    llm_inputs: tuple[str, ...] = ()
    health_scores: tuple[int, ...] = ()
    health_reminder_enabled: bool = False
    health_reminder_time: str | None = None
    health_reminder_schedules: tuple[str, ...] = ()
    health_reminder_removals: int = 0
    doctor_prep_count: int = 0
    task_reminder_count: int = 0


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    steps: tuple[ScenarioStep, ...]
    expected: ExpectedState
    llm_stubs: tuple[LLMStub, ...] = ()
    known_defect: str | None = None


@dataclass(frozen=True, slots=True)
class StepResult:
    kind: StepKind
    value: str
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    steps: tuple[StepResult, ...]
    state: ExpectedState


class RecordingHealthScheduler:
    def __init__(self) -> None:
        self.scheduled_times: list[str] = []
        self.removed_users: list[int] = []

    def schedule_health_reminder(self, **kwargs: object) -> None:
        local_time = kwargs["local_time"]
        self.scheduled_times.append(local_time.strftime("%H:%M"))

    def remove_health_reminder(self, user_id: int) -> None:
        self.removed_users.append(user_id)


def build_autotest_settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        telegram_bot_token=AUTOTEST_TELEGRAM_TOKEN,
        ai_provider="openrouter",
        ai_api_key=AUTOTEST_AI_KEY,
        ai_base_url=AUTOTEST_BASE_URL,
        ai_model="autotest-model",
        transcription_provider="disabled",
        transcription_api_key=None,
        transcription_base_url=AUTOTEST_BASE_URL,
        openai_api_key=None,
        database_url=database_url,
    )


def assert_safe_runtime(settings: Settings, database_url: str, sandbox: Path) -> None:
    if settings.telegram_bot_token != AUTOTEST_TELEGRAM_TOKEN:
        raise UnsafeAutotestConfiguration("Autotester requires its Telegram sentinel token")
    if settings.ai_api_key != AUTOTEST_AI_KEY:
        raise UnsafeAutotestConfiguration("Autotester requires its AI sentinel key")
    if settings.ai_base_url != AUTOTEST_BASE_URL:
        raise UnsafeAutotestConfiguration("Autotester requires its non-routable AI base URL")
    if settings.transcription_base_url != AUTOTEST_BASE_URL:
        raise UnsafeAutotestConfiguration(
            "Autotester requires its non-routable transcription base URL"
        )
    if settings.database_url != database_url:
        raise UnsafeAutotestConfiguration("Settings and harness database URLs differ")

    url = make_url(database_url)
    if url.get_backend_name() != "sqlite" or not url.drivername.endswith("aiosqlite"):
        raise UnsafeAutotestConfiguration("Autotester only accepts sqlite+aiosqlite")
    if not url.database or url.database == ":memory:":
        raise UnsafeAutotestConfiguration("Autotester requires a temporary SQLite file")

    database_path = Path(url.database).resolve()
    sandbox_path = sandbox.resolve()
    if not database_path.is_relative_to(sandbox_path):
        raise UnsafeAutotestConfiguration("Autotest database must stay inside pytest tmp_path")


class BotAutotester:
    def __init__(
        self,
        *,
        database: Database,
        bot: FutureSelfBot,
        ai: StrictAI,
        transcription: ScriptedTranscription,
        context: SimpleNamespace,
        telegram_user_id: int = 900_001,
        chat_id: int = 910_001,
    ):
        self.database = database
        self.bot = bot
        self.ai = ai
        self.transcription = transcription
        self.context = context
        self.telegram_user_id = telegram_user_id
        self.chat_id = chat_id
        self.messages: list[FakeMessage] = []
        self.health_state: int | None = None
        self.doctor_state: int | None = None
        self.contexts = {telegram_user_id: context}

    @classmethod
    async def create(cls, sandbox: Path, stubs: tuple[LLMStub, ...]) -> "BotAutotester":
        database_path = sandbox / "autotester.db"
        database_url = f"sqlite+aiosqlite:///{database_path.as_posix()}"
        settings = build_autotest_settings(database_url)
        assert_safe_runtime(settings, database_url, sandbox)

        responses = {stub.text: stub.response for stub in stubs}
        if len(responses) != len(stubs):
            raise AssertionError("Each LLM stub input must be unique")
        database = Database(database_url)
        await database.create_all_for_tests()
        ai = StrictAI(responses)
        transcription = ScriptedTranscription()
        context = SimpleNamespace(user_data={}, bot=FakeBot())
        bot = FutureSelfBot(settings, database, ai, transcription)
        scheduler = RecordingHealthScheduler()
        bot.scheduler = scheduler
        return cls(
            database=database,
            bot=bot,
            ai=ai,
            transcription=transcription,
            context=context,
        )

    async def close(self) -> None:
        await self.database.dispose()

    async def run(self, scenario: Scenario) -> ScenarioResult:
        results: list[StepResult] = []
        for step in scenario.steps:
            result = await self._run_step(step)
            output = "\n".join(result.outputs)
            for expected in step.reply_contains:
                if expected not in output:
                    raise AssertionError(
                        f"{scenario.name}: {step.kind} {step.value!r} did not output {expected!r}; "
                        f"outputs={result.outputs!r}"
                    )
            for forbidden in step.reply_excludes:
                if forbidden in output:
                    raise AssertionError(
                        f"{scenario.name}: {step.kind} {step.value!r} output forbidden "
                        f"{forbidden!r}; outputs={result.outputs!r}"
                    )
            results.append(result)

        actual = await self.snapshot()
        if actual != scenario.expected:
            raise AssertionError(
                f"{scenario.name}: final state differs\n"
                f"expected={scenario.expected!r}\n"
                f"actual={actual!r}"
            )
        return ScenarioResult(tuple(results), actual)

    async def snapshot(self) -> ExpectedState:
        async with self.database.sessions() as session:
            drafts = list((await session.scalars(select(DraftInboxItem))).all())
            inbox_items = list((await session.scalars(select(InboxItem))).all())
            health_records = list((await session.scalars(select(HealthCheckIn))).all())
            health_preference = await session.scalar(select(HealthReminderPreference))
            doctor_preps = list((await session.scalars(select(DoctorVisitPrep))).all())
            task_reminders = list((await session.scalars(select(TaskReminder))).all())
        return ExpectedState(
            drafts=tuple(
                sorted(
                    DraftState(draft.title, draft.kind, draft.status, draft.source)
                    for draft in drafts
                )
            ),
            inbox=tuple(
                sorted(InboxState(item.title, item.kind, item.source) for item in inbox_items)
            ),
            llm_inputs=tuple(self.ai.route_calls),
            health_scores=tuple(sorted(record.state_score for record in health_records)),
            health_reminder_enabled=bool(health_preference and health_preference.enabled),
            health_reminder_time=(
                health_preference.local_time.strftime("%H:%M") if health_preference else None
            ),
            health_reminder_schedules=tuple(self.bot.scheduler.scheduled_times),
            health_reminder_removals=len(self.bot.scheduler.removed_users),
            doctor_prep_count=len(doctor_preps),
            task_reminder_count=len(task_reminders),
        )

    async def _run_step(self, step: ScenarioStep) -> StepResult:
        if step.kind == "switch_user":
            return self._switch_user(step.value)
        if step.kind == "setup_clear_focus":
            await self.bot.conversation.clear_focus(self.telegram_user_id, self.chat_id)
            return StepResult(step.kind, step.value, ())
        if step.kind == "callback":
            return await self._run_callback(step.value)
        if step.kind == "command":
            return await self._run_command(step.value)
        if step.kind == "health_answer":
            return await self._run_health_answer(step.value)
        if step.kind == "doctor_answer":
            return await self._run_doctor_answer(step.value)

        if step.kind == "voice":
            self.transcription.queue(step.value)
            message = FakeMessage(voice=FakeVoice())
            route = self.bot.voice
        else:
            message = FakeMessage(step.value)
            route = self.bot.text
        self.messages.append(message)
        await route(self._update_for(message), self.context)
        outputs = tuple(str(reply["text"]) for reply in message.replies) + tuple(message.edits)
        return StepResult(step.kind, step.value, outputs)

    async def _run_command(self, value: str) -> StepResult:
        parts = value.split()
        command = parts[0].removeprefix("/")
        self.context.args = parts[1:]
        message = FakeMessage(value)
        self.messages.append(message)
        update = self._update_for(message)
        handlers = {
            "health": self.bot.health_command,
            "checkin": self.bot.health_checkin_start,
            "health_edit": self.bot.health_checkin_start,
            "health_delete": self.bot.health_delete_command,
            "health_reminder_on": self.bot.health_reminder_on,
            "health_reminder_off": self.bot.health_reminder_off,
            "cancel": self.bot.cancel_health_checkin,
            "doctor_prepare": self.bot.doctor_prepare_start,
            "doctor_prepare_edit": self.bot.doctor_prepare_start,
            "doctor_preparations": self.bot.doctor_preparations,
            "doctor_prepare_show": self.bot.doctor_prepare_show,
            "doctor_prepare_delete": self.bot.doctor_prepare_delete,
            "doctor_prepare_task": self.bot.doctor_prepare_task,
            "doctor_find": self.bot.doctor_find,
            "doctor_find_task": self.bot.doctor_find_task,
        }
        if command == "cancel" and self.doctor_state in {
            DOCTOR_REASON,
            DOCTOR_DURATION,
            DOCTOR_SYMPTOMS,
            DOCTOR_MEDICATIONS,
            DOCTOR_QUESTIONS,
        }:
            result = await self.bot.cancel_doctor_prepare(update, self.context)
        else:
            result = await handlers[command](update, self.context)
        if command in {"checkin", "health_edit", "cancel"}:
            self.health_state = result
        if command in {"doctor_prepare", "doctor_prepare_edit", "cancel"}:
            self.doctor_state = result
        outputs = tuple(str(reply["text"]) for reply in message.replies)
        return StepResult("command", value, outputs)

    async def _run_health_answer(self, value: str) -> StepResult:
        handlers = {
            HEALTH_ENERGY: self.bot.health_energy,
            HEALTH_SLEEP: self.bot.health_sleep,
            HEALTH_MOOD: self.bot.health_mood,
            HEALTH_STRESS: self.bot.health_stress,
            HEALTH_PHYSICAL: self.bot.health_physical,
            HEALTH_SYMPTOMS: self.bot.health_symptoms,
        }
        if self.health_state not in handlers:
            raise AssertionError("No active health check-in state")
        message = FakeMessage(value)
        self.messages.append(message)
        if self.health_state != HEALTH_SYMPTOMS and re.fullmatch(r"(?:10|[0-9])", value) is None:
            await self.bot.health_invalid_rating(self._update_for(message), self.context)
        else:
            self.health_state = await handlers[self.health_state](
                self._update_for(message),
                self.context,
            )
        outputs = tuple(str(reply["text"]) for reply in message.replies)
        return StepResult("health_answer", value, outputs)

    async def _run_doctor_answer(self, value: str) -> StepResult:
        handlers = {
            DOCTOR_REASON: self.bot.doctor_prepare_reason,
            DOCTOR_DURATION: self.bot.doctor_prepare_duration,
            DOCTOR_SYMPTOMS: self.bot.doctor_prepare_symptoms,
            DOCTOR_MEDICATIONS: self.bot.doctor_prepare_medications,
            DOCTOR_QUESTIONS: self.bot.doctor_prepare_questions,
        }
        if self.doctor_state not in handlers:
            raise AssertionError("No active doctor preparation state")
        message = FakeMessage(value)
        self.messages.append(message)
        self.doctor_state = await handlers[self.doctor_state](
            self._update_for(message),
            self.context,
        )
        outputs = tuple(str(reply["text"]) for reply in message.replies)
        return StepResult("doctor_answer", value, outputs)

    def _switch_user(self, value: str) -> StepResult:
        parts = value.split(":", maxsplit=1)
        self.telegram_user_id = int(parts[0])
        self.chat_id = int(parts[1]) if len(parts) == 2 else self.telegram_user_id + 10_000
        self.context = self.contexts.setdefault(
            self.telegram_user_id,
            SimpleNamespace(user_data={}, bot=FakeBot()),
        )
        self.health_state = None
        self.doctor_state = None
        return StepResult("switch_user", value, ())

    async def _run_callback(self, action: str) -> StepResult:
        data, message = self._latest_preview_callback(action)
        query = FakeCallbackQuery(data, message)
        update = SimpleNamespace(
            callback_query=query,
            effective_user=SimpleNamespace(id=self.telegram_user_id),
            effective_chat=SimpleNamespace(id=self.chat_id),
        )
        await self.bot.inbox_action(update, self.context)
        outputs = tuple(query.edits) + tuple(
            answer for answer, _show_alert in query.answers if answer is not None
        )
        return StepResult("callback", action, outputs)

    def _latest_preview_callback(self, action: str) -> tuple[str, FakeMessage]:
        prefix = f"inbox:{action}:"
        for message in reversed(self.messages):
            for reply in reversed(message.replies):
                markup = reply.get("reply_markup")
                if markup is None:
                    continue
                for row in markup.inline_keyboard:
                    for button in row:
                        if button.callback_data and button.callback_data.startswith(prefix):
                            return button.callback_data, message
        raise AssertionError(f"No active preview callback found for {action!r}")

    def _update_for(self, message: FakeMessage) -> SimpleNamespace:
        return SimpleNamespace(
            effective_message=message,
            message=message,
            effective_user=SimpleNamespace(id=self.telegram_user_id),
            effective_chat=SimpleNamespace(id=self.chat_id),
        )
