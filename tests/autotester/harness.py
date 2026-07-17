from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from sqlalchemy import select
from sqlalchemy.engine import make_url

from future_self.bot import FutureSelfBot
from future_self.config import Settings
from future_self.db import Database
from future_self.models import DraftInboxItem, InboxItem
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

StepKind = Literal["text", "voice", "callback", "setup_clear_focus"]


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


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    steps: tuple[ScenarioStep, ...]
    expected: ExpectedState
    llm_stubs: tuple[LLMStub, ...] = ()


@dataclass(frozen=True, slots=True)
class StepResult:
    kind: StepKind
    value: str
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    steps: tuple[StepResult, ...]
    state: ExpectedState


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
        return cls(
            database=database,
            bot=FutureSelfBot(settings, database, ai, transcription),
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
        )

    async def _run_step(self, step: ScenarioStep) -> StepResult:
        if step.kind == "setup_clear_focus":
            await self.bot.conversation.clear_focus(self.telegram_user_id, self.chat_id)
            return StepResult(step.kind, step.value, ())
        if step.kind == "callback":
            return await self._run_callback(step.value)

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
