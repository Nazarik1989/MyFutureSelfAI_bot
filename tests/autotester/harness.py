import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from PIL import Image
from pypdf import PdfWriter
from sqlalchemy import func, select
from sqlalchemy.engine import make_url
from telegram.ext import ApplicationHandlerStop, ConversationHandler

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
from future_self.lab_media import MAX_LAB_INPUT_BYTES, MAX_PDF_PAGES
from future_self.labs import LabUploadSessionStore
from future_self.models import (
    DoctorVisitPrep,
    DraftInboxItem,
    HealthCheckIn,
    HealthReminderPreference,
    InboxItem,
    LabDocument,
    LifeCollection,
    LifeCollectionLink,
    LifeCollectionPreference,
    TaskReminder,
    VisionDraft,
    VisionItem,
    VisionItemImage,
)
from future_self.repositories import OnboardingRepository
from future_self.schemas import IntentResult
from future_self.vision import CATEGORY_META
from future_self.vision_images import MAX_IMAGE_INPUT_BYTES, MAX_IMAGE_PIXELS

from .fakes import (
    FakeBot,
    FakeCallbackQuery,
    FakeImageMedia,
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
    "vision_callback",
    "vision_raw_callback",
    "vision_capture_callback",
    "vision_replay_callback",
    "vision_hold_render",
    "vision_release_render",
    "vision_photo",
    "vision_document",
    "navigation_callback",
    "navigation_raw_callback",
    "navigation_capture_callback",
    "navigation_replay_callback",
    "lab_callback",
    "lab_raw_callback",
    "lab_capture_callback",
    "lab_replay_callback",
    "lab_photo",
    "lab_document",
    "lab_text",
    "task_callback",
    "task_raw_callback",
    "task_capture_callback",
    "task_replay_callback",
    "collection_callback",
    "collection_raw_callback",
    "collection_capture_callback",
    "collection_replay_callback",
    "restart",
    "group_command",
    "timezone_onboarding",
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


@dataclass(frozen=True, slots=True, order=True)
class VisionState:
    category: str
    wish_text: str
    status: str
    linked_task: bool = False
    has_image: bool = False


@dataclass(frozen=True, slots=True, order=True)
class LabState:
    title: str
    document_date: str | None
    page_count: int


@dataclass(frozen=True, slots=True, order=True)
class CollectionState:
    name: str
    kind: str
    status: str
    item_count: int


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
    vision_items: tuple[VisionState, ...] = ()
    vision_draft_count: int = 0
    vision_image_count: int = 0
    lab_documents: tuple[LabState, ...] = ()
    collections: tuple[CollectionState, ...] = ()
    collection_link_count: int = 0
    collection_onboarded: bool = False


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
        database_path: Path,
        bot: FutureSelfBot,
        ai: StrictAI,
        transcription: ScriptedTranscription,
        context: SimpleNamespace,
        telegram_user_id: int = 900_001,
        chat_id: int = 910_001,
    ):
        self.database = database
        self.database_path = database_path
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
        self.saved_vision_callbacks: dict[str, tuple[str, FakeMessage]] = {}
        self.saved_navigation_callbacks: dict[str, tuple[str, FakeMessage]] = {}
        self.saved_lab_callbacks: dict[str, tuple[str, FakeMessage]] = {}
        self.saved_task_callbacks: dict[str, tuple[str, FakeMessage]] = {}
        self.saved_collection_callbacks: dict[str, tuple[str, FakeMessage]] = {}

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
        bot.lab_uploads = LabUploadSessionStore(root=sandbox / "lab-temp")
        scheduler = RecordingHealthScheduler()
        bot.scheduler = scheduler
        return cls(
            database=database,
            database_path=database_path,
            bot=bot,
            ai=ai,
            transcription=transcription,
            context=context,
        )

    async def close(self) -> None:
        await self.database.dispose()
        for suffix in ("", "-wal", "-shm"):
            self.database_path.with_name(self.database_path.name + suffix).unlink(missing_ok=True)

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
            vision_items = list((await session.scalars(select(VisionItem))).all())
            vision_images = list((await session.scalars(select(VisionItemImage))).all())
            vision_drafts = list((await session.scalars(select(VisionDraft))).all())
            lab_documents = list((await session.scalars(select(LabDocument))).all())
            collection_rows = (
                await session.execute(
                    select(LifeCollection, func.count(LifeCollectionLink.id))
                    .outerjoin(
                        LifeCollectionLink,
                        LifeCollectionLink.collection_id == LifeCollection.id,
                    )
                    .group_by(LifeCollection.id)
                )
            ).all()
            collection_link_count = int(
                await session.scalar(select(func.count(LifeCollectionLink.id))) or 0
            )
            collection_onboarded = bool(
                await session.scalar(
                    select(func.count(LifeCollectionPreference.owner_id)).where(
                        LifeCollectionPreference.onboarding_completed.is_(True)
                    )
                )
            )
        image_item_ids = {image.vision_item_id for image in vision_images}
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
            vision_items=tuple(
                sorted(
                    VisionState(
                        item.category,
                        item.wish_text,
                        item.status,
                        item.linked_task_id is not None,
                        item.id in image_item_ids,
                    )
                    for item in vision_items
                )
            ),
            vision_draft_count=len(vision_drafts),
            vision_image_count=len(vision_images),
            lab_documents=tuple(
                sorted(
                    LabState(
                        item.title,
                        item.document_date.isoformat() if item.document_date else None,
                        item.page_count,
                    )
                    for item in lab_documents
                )
            ),
            collections=tuple(
                sorted(
                    CollectionState(
                        collection.name,
                        collection.kind,
                        collection.status,
                        int(item_count or 0),
                    )
                    for collection, item_count in collection_rows
                )
            ),
            collection_link_count=collection_link_count,
            collection_onboarded=collection_onboarded,
        )

    async def _run_step(self, step: ScenarioStep) -> StepResult:
        if step.kind == "switch_user":
            return self._switch_user(step.value)
        if step.kind == "setup_clear_focus":
            await self.bot.conversation.clear_focus(self.telegram_user_id, self.chat_id)
            return StepResult(step.kind, step.value, ())
        if step.kind == "callback":
            return await self._run_callback(step.value)
        if step.kind == "vision_callback":
            return await self._run_vision_callback(step.value)
        if step.kind == "vision_raw_callback":
            return await self._run_raw_vision_callback(step.value)
        if step.kind == "vision_capture_callback":
            data, message = self._latest_vision_callback(step.value)
            self.saved_vision_callbacks[step.value] = (data, message)
            return StepResult(step.kind, step.value, ("callback captured",))
        if step.kind == "vision_replay_callback":
            try:
                data, message = self.saved_vision_callbacks[step.value]
            except KeyError as exc:
                raise AssertionError(f"No saved vision callback for {step.value!r}") from exc
            return await self._dispatch_vision_callback(step.kind, step.value, data, message)
        if step.kind == "navigation_callback":
            return await self._run_navigation_callback(step.value)
        if step.kind == "navigation_raw_callback":
            message = FakeMessage()
            self.messages.append(message)
            return await self._dispatch_navigation_callback(
                step.kind, step.value, step.value, message
            )
        if step.kind == "navigation_capture_callback":
            data, message = self._latest_navigation_callback(step.value)
            self.saved_navigation_callbacks[step.value] = (data, message)
            return StepResult(step.kind, step.value, ("callback captured",))
        if step.kind == "navigation_replay_callback":
            try:
                data, message = self.saved_navigation_callbacks[step.value]
            except KeyError as exc:
                raise AssertionError(f"No saved navigation callback for {step.value!r}") from exc
            return await self._dispatch_navigation_callback(step.kind, step.value, data, message)
        if step.kind == "lab_callback":
            return await self._run_lab_callback(step.value)
        if step.kind == "lab_raw_callback":
            message = FakeMessage()
            self.messages.append(message)
            return await self._dispatch_lab_callback(step.kind, step.value, step.value, message)
        if step.kind == "lab_capture_callback":
            data, message = self._latest_lab_callback(step.value)
            self.saved_lab_callbacks[step.value] = (data, message)
            return StepResult(step.kind, step.value, ("callback captured",))
        if step.kind == "lab_replay_callback":
            try:
                data, message = self.saved_lab_callbacks[step.value]
            except KeyError as exc:
                raise AssertionError(f"No saved lab callback for {step.value!r}") from exc
            return await self._dispatch_lab_callback(step.kind, step.value, data, message)
        if step.kind == "task_callback":
            return await self._run_task_callback(step.value)
        if step.kind == "task_raw_callback":
            message = FakeMessage()
            self.messages.append(message)
            return await self._dispatch_task_callback(step.kind, step.value, step.value, message)
        if step.kind == "task_capture_callback":
            data, message = self._latest_task_callback(step.value)
            self.saved_task_callbacks[step.value] = (data, message)
            return StepResult(step.kind, step.value, ("callback captured",))
        if step.kind == "task_replay_callback":
            try:
                data, message = self.saved_task_callbacks[step.value]
            except KeyError as exc:
                raise AssertionError(f"No saved task callback for {step.value!r}") from exc
            return await self._dispatch_task_callback(step.kind, step.value, data, message)
        if step.kind == "collection_callback":
            return await self._run_collection_callback(step.value)
        if step.kind == "collection_raw_callback":
            message = FakeMessage()
            self.messages.append(message)
            return await self._dispatch_collection_callback(
                step.kind, step.value, step.value, message
            )
        if step.kind == "collection_capture_callback":
            data, message = self._latest_collection_callback(step.value)
            self.saved_collection_callbacks[step.value] = (data, message)
            return StepResult(step.kind, step.value, ("callback captured",))
        if step.kind == "collection_replay_callback":
            try:
                data, message = self.saved_collection_callbacks[step.value]
            except KeyError as exc:
                raise AssertionError(f"No saved collection callback for {step.value!r}") from exc
            return await self._dispatch_collection_callback(step.kind, step.value, data, message)
        if step.kind == "vision_hold_render":
            user = await self.bot._user(self.telegram_user_id)
            if not await self.bot.vision_render_limiter.acquire(user.id):
                raise AssertionError("Could not hold owner render slot")
            return StepResult(step.kind, step.value, ("render slot held",))
        if step.kind == "vision_release_render":
            user = await self.bot._user(self.telegram_user_id)
            await self.bot.vision_render_limiter.release(user.id)
            return StepResult(step.kind, step.value, ("render slot released",))
        if step.kind in {"vision_photo", "vision_document"}:
            return await self._run_vision_media(step.kind, step.value)
        if step.kind in {"lab_photo", "lab_document"}:
            return await self._run_lab_media(step.kind, step.value)
        if step.kind == "lab_text":
            return await self._run_lab_text(step.value)
        if step.kind == "restart":
            scheduler = self.bot.scheduler
            lab_root = self.bot.lab_uploads.root
            self.bot = FutureSelfBot(
                self.bot.settings,
                self.database,
                self.ai,
                self.transcription,
            )
            self.bot.lab_uploads = LabUploadSessionStore(root=lab_root)
            self.bot.scheduler = scheduler
            return StepResult(step.kind, step.value, ("bot restarted",))
        if step.kind == "group_command":
            return await self._run_group_command(step.value)
        if step.kind == "timezone_onboarding":
            return await self._run_timezone_onboarding(step.value)
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

    async def _run_group_command(self, value: str) -> StepResult:
        message = FakeMessage(value)
        self.messages.append(message)
        update = SimpleNamespace(
            effective_message=message,
            message=message,
            callback_query=None,
            effective_user=SimpleNamespace(id=self.telegram_user_id),
            effective_chat=SimpleNamespace(id=self.chat_id, type="group"),
        )
        try:
            await self.bot.private_chat_guard(update, self.context)
        except ApplicationHandlerStop:
            pass
        else:
            raise AssertionError("Group update was not stopped before feature handlers")
        return StepResult(
            "group_command",
            value,
            tuple(str(reply["text"]) for reply in message.replies),
        )

    async def _run_vision_media(self, kind: StepKind, value: str) -> StepResult:
        payload, mime_type, width, height, declared_size = self._vision_media_fixture(value)
        media = FakeImageMedia(
            payload,
            mime_type=mime_type if kind == "vision_document" else None,
            width=width if kind == "vision_photo" else None,
            height=height if kind == "vision_photo" else None,
            file_size=declared_size,
        )
        message = FakeMessage(
            photo=[media] if kind == "vision_photo" else None,
            document=media if kind == "vision_document" else None,
        )
        self.messages.append(message)
        try:
            await self.bot.vision_image_gate(self._update_for(message), self.context)
        except ApplicationHandlerStop:
            pass
        outputs = tuple(str(reply["text"]) for reply in message.replies)
        return StepResult(kind, value, outputs)

    async def _run_lab_media(self, kind: StepKind, value: str) -> StepResult:
        payload, mime_type, width, height, declared_size = self._lab_media_fixture(value)
        media = FakeImageMedia(
            payload,
            mime_type=mime_type if kind == "lab_document" else None,
            width=width if kind == "lab_photo" else None,
            height=height if kind == "lab_photo" else None,
            file_size=declared_size,
        )
        message = FakeMessage(
            photo=[media] if kind == "lab_photo" else None,
            document=media if kind == "lab_document" else None,
        )
        self.messages.append(message)
        try:
            await self.bot.labs_media_gate(self._update_for(message), self.context)
        except ApplicationHandlerStop:
            pass
        return StepResult(kind, value, tuple(str(reply["text"]) for reply in message.replies))

    async def _run_lab_text(self, value: str) -> StepResult:
        message = FakeMessage(value)
        self.messages.append(message)
        try:
            await self.bot.labs_text_gate(self._update_for(message), self.context)
        except ApplicationHandlerStop:
            pass
        return StepResult(
            "lab_text",
            value,
            tuple(str(reply["text"]) for reply in message.replies) + tuple(message.edits),
        )

    @staticmethod
    def _lab_media_fixture(
        value: str,
    ) -> tuple[bytes, str, int, int, int | None]:
        if value in {"jpeg", "png", "webp"}:
            image_format = value.upper()
            mime_type = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }[image_format]
            return BotAutotester._encoded_image(image_format, "teal"), mime_type, 120, 80, None
        if value.startswith("pdf"):
            encrypted = value == "pdf-encrypted"
            javascript = value == "pdf-js"
            attachment = value == "pdf-attachment"
            page_count = {
                "pdf1": 1,
                "pdf3": 3,
                "pdf-too-many": MAX_PDF_PAGES + 1,
            }.get(value, 1)
            writer = PdfWriter()
            for _ in range(page_count):
                writer.add_blank_page(width=612, height=792)
            if encrypted:
                writer.encrypt("password")
            if javascript:
                writer.add_js("app.alert('blocked')")
            if attachment:
                writer.add_attachment("blocked.txt", b"blocked")
            output = BytesIO()
            writer.write(output)
            return output.getvalue(), "application/pdf", 0, 0, None
        if value == "corrupt":
            return b"corrupt-lab-document", "image/jpeg", 10, 10, None
        if value == "mismatch":
            return BotAutotester._encoded_image("PNG", "purple"), "image/jpeg", 120, 80, None
        if value == "oversize-meta":
            return (
                BotAutotester._encoded_image("JPEG", "red"),
                "image/jpeg",
                120,
                80,
                MAX_LAB_INPUT_BYTES + 1,
            )
        raise AssertionError(f"Unknown lab media fixture {value!r}")

    @staticmethod
    def _vision_media_fixture(
        value: str,
    ) -> tuple[bytes, str, int, int, int | None]:
        presets = {
            "jpeg": ("JPEG", "red"),
            "jpeg-second": ("JPEG", "blue"),
            "png": ("PNG", "green"),
            "webp": ("WEBP", "orange"),
        }
        if value in presets:
            image_format, color = presets[value]
            mime_type = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
                "WEBP": "image/webp",
            }[image_format]
            return (
                BotAutotester._encoded_image(image_format, color),
                mime_type,
                120,
                80,
                None,
            )
        if value == "pdf":
            return b"%PDF-1.4\n% autotest only", "application/pdf", 0, 0, None
        if value == "corrupt":
            return b"corrupt-autotest-image", "image/jpeg", 10, 10, None
        if value == "mismatch":
            payload = BotAutotester._encoded_image("PNG", "purple")
            return payload, "image/jpeg", 120, 80, None
        if value == "oversize-meta":
            payload = BotAutotester._encoded_image("JPEG", "red")
            return payload, "image/jpeg", 120, 80, MAX_IMAGE_INPUT_BYTES + 1
        if value == "multipixel-meta":
            payload = BotAutotester._encoded_image("JPEG", "red")
            return payload, "image/jpeg", MAX_IMAGE_PIXELS, 2, None
        if value == "animated-gif":
            first = Image.new("RGB", (20, 20), "red")
            second = Image.new("RGB", (20, 20), "blue")
            output = BytesIO()
            first.save(
                output,
                format="GIF",
                save_all=True,
                append_images=[second],
                duration=100,
                loop=0,
            )
            first.close()
            second.close()
            return output.getvalue(), "image/gif", 20, 20, None
        image_format, color = value.split(":", maxsplit=1)
        format_name = image_format.upper()
        mime_type = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }[format_name]
        return BotAutotester._encoded_image(format_name, color), mime_type, 120, 80, None

    @staticmethod
    def _encoded_image(image_format: str, color: str) -> bytes:
        image = Image.new("RGB", (120, 80), color)
        output = BytesIO()
        image.save(output, format=image_format)
        image.close()
        return output.getvalue()

    async def _run_timezone_onboarding(self, value: str) -> StepResult:
        answer, expected = value.split("=>", maxsplit=1)
        user = await self.bot._user(self.telegram_user_id)
        async with self.database.session() as session:
            state = await OnboardingRepository(session).get_or_create(user.id)
            state.current_step = 1
            state.answers = {"display_name": "Автотест"}
        message = FakeMessage(answer)
        self.messages.append(message)
        update = self._update_for(message)
        context = SimpleNamespace(user_data={"onboarding_user_id": user.id})
        await self.bot.onboarding_answer(update, context)
        async with self.database.sessions() as session:
            state = await OnboardingRepository(session).get_or_create(user.id)
            saved = state.answers.get("timezone")
        if saved != expected:
            raise AssertionError(f"Timezone onboarding saved {saved!r}, expected {expected!r}")
        outputs = tuple(str(reply["text"]) for reply in message.replies) + (f"timezone={saved}",)
        return StepResult("timezone_onboarding", value, outputs)

    async def _run_command(self, value: str) -> StepResult:
        parts = value.split()
        command = parts[0].removeprefix("/")
        self.context.args = parts[1:]
        message = FakeMessage(value)
        self.messages.append(message)
        update = self._update_for(message)
        handlers = {
            "menu": self.bot.menu_command,
            "help": self.bot.help_command,
            "doctor": self.bot.doctor_command,
            "inbox": self.bot.inbox,
            "tasks": self.bot.tasks_command,
            "collections": self.bot.collections_command,
            "drafts": self.bot.drafts_command,
            "vision": self.bot.vision_command,
            "location": self.bot.location_command,
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
            "labs": self.bot.labs_command,
        }
        current_user = await self.bot._user(self.telegram_user_id)
        vision_draft = await self.bot.vision_service.draft(current_user.id, self.chat_id)
        if command == "cancel" and vision_draft is not None:
            result = await self.bot.cancel_draft_edit(update, self.context)
        elif command == "cancel" and self.doctor_state in {
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

    async def _run_task_callback(self, action: str) -> StepResult:
        data, message = self._latest_task_callback(action)
        return await self._dispatch_task_callback("task_callback", action, data, message)

    async def _dispatch_task_callback(
        self,
        kind: StepKind,
        value: str,
        data: str,
        message: FakeMessage,
    ) -> StepResult:
        previous_replies = len(message.replies)
        query = FakeCallbackQuery(data, message)
        update = SimpleNamespace(
            effective_message=message,
            message=message,
            callback_query=query,
            effective_user=SimpleNamespace(id=self.telegram_user_id),
            effective_chat=SimpleNamespace(id=self.chat_id, type="private"),
        )
        await self.bot.task_callback(update, self.context)
        outputs = (
            tuple(query.edits)
            + tuple(answer for answer, _show_alert in query.answers if answer is not None)
            + tuple(str(reply["text"]) for reply in message.replies[previous_replies:])
        )
        return StepResult(kind, value, outputs)

    def _latest_task_callback(self, action: str) -> tuple[str, FakeMessage]:
        exact = {
            "hub": "task:hub",
            "today": "task:list:today:0",
            "upcoming": "task:list:upcoming:0",
            "overdue": "task:list:overdue:0",
            "no-due": "task:list:no_due:0",
            "completed": "task:list:completed:0",
        }.get(action)
        labels = {
            "open": "Открыть",
            "complete": "Выполнено",
            "reopen": "Вернуть в активные",
            "reschedule": "Перенести",
            "30m": "Через 30 минут",
            "1h": "Через 1 час",
            "tomorrow": "Завтра в 09:00",
            "custom": "Другая дата и время",
            "preserve": "Сохранить прежний интервал",
            "new-reminder": "Выбрать новое напоминание",
            "reminder-edit": "Изменить напоминание",
            "reminder-off": "Отключить напоминание",
            "delete": "Удалить",
            "delete-confirm": "Да, удалить",
            "cancel": "Отмена",
            "collection-view": "Открыть в Task Hub",
        }
        label = labels.get(action)
        for message in reversed(self.messages):
            for reply in reversed(message.replies):
                markup = reply.get("reply_markup")
                if markup is None:
                    continue
                for row in markup.inline_keyboard:
                    for button in row:
                        data = button.callback_data
                        if data is None:
                            continue
                        if exact is not None and data == exact:
                            return data, message
                        if label is not None and (
                            button.text == label or button.text.startswith(label)
                        ):
                            return data, message
        raise AssertionError(f"No task callback found for {action!r}")

    async def _run_collection_callback(self, action: str) -> StepResult:
        data, message = self._latest_collection_callback(action)
        return await self._dispatch_collection_callback(
            "collection_callback", action, data, message
        )

    async def _dispatch_collection_callback(
        self,
        kind: StepKind,
        value: str,
        data: str,
        message: FakeMessage,
    ) -> StepResult:
        previous_replies = len(message.replies)
        query = FakeCallbackQuery(data, message)
        update = SimpleNamespace(
            effective_message=message,
            message=message,
            callback_query=query,
            effective_user=SimpleNamespace(id=self.telegram_user_id),
            effective_chat=SimpleNamespace(id=self.chat_id, type="private"),
        )
        await self.bot.collection_callback(update, self.context)
        outputs = (
            tuple(query.edits)
            + tuple(answer for answer, _show_alert in query.answers if answer is not None)
            + tuple(str(reply["text"]) for reply in message.replies[previous_replies:])
        )
        return StepResult(kind, value, outputs)

    def _latest_collection_callback(self, action: str) -> tuple[str, FakeMessage]:
        labels = {
            "onboard-all": "Создать все",
            "onboard-select": "Выбрать разделы",
            "onboard-empty": "Начать с пустого",
            "onboard-custom": "Создать свой",
            "confirm-create": "Создать",
            "starter-shopping": "Покупки",
            "starter-travel": "Отдых и путешествия",
            "starter-confirm": "Создать выбранные",
            "all": "Все активные",
            "topics": "Темы",
            "projects": "Проекты",
            "lists": "Списки",
            "archived-list": "Архив",
            "create-topic": "+ Тема",
            "create-project": "+ Проект",
            "create-list": "+ Список",
            "open": "Открыть",
            "content": "Открыть содержимое",
            "add": "Добавить запись",
            "rename": "Переименовать",
            "alias": "Добавить alias",
            "archive": "Архивировать",
            "restore": "Восстановить",
            "delete": "Удалить",
            "delete-links": "Удалить только связи",
            "item-open": "Открыть 1",
            "task-hub": "Открыть в Task Hub",
            "move": "Переместить",
            "link": "Связать ещё",
            "unlink": "Убрать из раздела",
            "delete-item": "Удалить запись",
            "save": "Сохранить",
        }
        target_label = action.removeprefix("target:") if action.startswith("target:") else None
        collection_label = (
            action.removeprefix("collection:") if action.startswith("collection:") else None
        )
        expected_label = labels.get(action)
        for message in reversed(self.messages):
            for reply in reversed(message.replies):
                markup = reply.get("reply_markup")
                if markup is None:
                    continue
                for row in markup.inline_keyboard:
                    for button in row:
                        data = button.callback_data
                        if data is None or not data.startswith("collection:"):
                            continue
                        if target_label is not None and button.text == target_label:
                            return data, message
                        if collection_label is not None and button.text.startswith(
                            collection_label
                        ):
                            return data, message
                        if expected_label is not None and (
                            button.text == expected_label or button.text.startswith(expected_label)
                        ):
                            return data, message
        raise AssertionError(f"No collection callback found for {action!r}")

    async def _run_lab_callback(self, action: str) -> StepResult:
        data, message = self._latest_lab_callback(action)
        return await self._dispatch_lab_callback("lab_callback", action, data, message)

    async def _dispatch_lab_callback(
        self,
        kind: StepKind,
        value: str,
        data: str,
        message: FakeMessage,
    ) -> StepResult:
        previous_replies = len(message.replies)
        query = FakeCallbackQuery(data, message)
        update = SimpleNamespace(
            effective_message=message,
            message=message,
            callback_query=query,
            effective_user=SimpleNamespace(id=self.telegram_user_id),
            effective_chat=SimpleNamespace(id=self.chat_id, type="private"),
        )
        await self.bot.labs_action(update, self.context)
        outputs = (
            tuple(query.edits)
            + tuple(answer for answer, _show_alert in query.answers if answer is not None)
            + tuple(str(reply["text"]) for reply in message.replies[previous_replies:])
        )
        return StepResult(kind, value, outputs)

    def _latest_lab_callback(self, action: str) -> tuple[str, FakeMessage]:
        exact = {
            "menu": "labs:menu",
            "add": "labs:add",
            "help": "labs:help",
            "list": "labs:list:0",
        }.get(action)
        prefixes = {
            "cancel": "labs:cancel:",
            "draft-title": "labs:draft:title:",
            "draft-date": "labs:draft:date:",
            "draft-save": "labs:draft:save:",
            "open": "labs:open:",
            "view": "labs:view:",
            "rename": "labs:rename:",
            "date": "labs:date:",
            "delete": "labs:delete:",
            "delete-confirm": "labs:deleteconfirm:",
            "list-next": "labs:list:",
        }
        prefix = prefixes.get(action)
        for message in reversed(self.messages):
            for reply in reversed(message.replies):
                markup = reply.get("reply_markup")
                if markup is None:
                    continue
                for row in markup.inline_keyboard:
                    for button in row:
                        data = button.callback_data
                        if data is None:
                            continue
                        if exact is not None and data == exact:
                            return data, message
                        if prefix is not None and data.startswith(prefix):
                            return data, message
        raise AssertionError(f"No lab callback found for {action!r}")

    async def _run_navigation_callback(self, action: str) -> StepResult:
        data, message = self._latest_navigation_callback(action)
        return await self._dispatch_navigation_callback(
            "navigation_callback", action, data, message
        )

    async def _dispatch_navigation_callback(
        self,
        kind: StepKind,
        value: str,
        data: str,
        message: FakeMessage,
    ) -> StepResult:
        previous_replies = len(message.replies)
        query = FakeCallbackQuery(data, message)
        update = SimpleNamespace(
            effective_message=message,
            message=message,
            callback_query=query,
            effective_user=SimpleNamespace(id=self.telegram_user_id),
            effective_chat=SimpleNamespace(id=self.chat_id, type="private"),
        )
        if data == "nav:action:checkin":
            self.health_state = await self.bot.navigation_health_entry(update, self.context)
        elif data == "nav:action:doctor_prepare":
            self.doctor_state = await self.bot.navigation_doctor_entry(update, self.context)
        elif data == "nav:action:onboarding":
            await self.bot.navigation_onboarding_entry(update, self.context)
        else:
            result = await self.bot.navigation_action(update, self.context)
            if result == ConversationHandler.END:
                if "health_checkin" not in self.context.user_data:
                    self.health_state = None
                if "doctor_prepare" not in self.context.user_data:
                    self.doctor_state = None
        outputs = (
            tuple(query.edits)
            + tuple(answer for answer, _show_alert in query.answers if answer is not None)
            + tuple(str(reply["text"]) for reply in message.replies[previous_replies:])
        )
        return StepResult(kind, value, outputs)

    def _latest_navigation_callback(self, action: str) -> tuple[str, FakeMessage]:
        expected = {
            "root": "nav:root",
            "help": "nav:help",
        }.get(action)
        for message in reversed(self.messages):
            for reply in reversed(message.replies):
                markup = reply.get("reply_markup")
                if markup is None:
                    continue
                for row in markup.inline_keyboard:
                    for button in row:
                        data = button.callback_data
                        if data is None:
                            continue
                        if expected is not None and data == expected:
                            return data, message
                        if data == f"nav:section:{action}":
                            return data, message
                        if data == f"nav:action:{action}":
                            return data, message
                        if data == f"nav:help:{action}":
                            return data, message
                        if data.startswith(f"nav:flow:{action}:"):
                            return data, message
        raise AssertionError(f"No navigation callback for action={action!r}")

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

    async def _run_vision_callback(self, action: str) -> StepResult:
        data, message = self._latest_vision_callback(action)
        return await self._dispatch_vision_callback("vision_callback", action, data, message)

    async def _run_raw_vision_callback(self, data: str) -> StepResult:
        message = FakeMessage()
        self.messages.append(message)
        return await self._dispatch_vision_callback("vision_raw_callback", data, data, message)

    async def _dispatch_vision_callback(
        self,
        kind: StepKind,
        value: str,
        data: str,
        message: FakeMessage,
    ) -> StepResult:
        previous_replies = len(message.replies)
        query = FakeCallbackQuery(data, message)
        update = SimpleNamespace(
            callback_query=query,
            effective_user=SimpleNamespace(id=self.telegram_user_id),
            effective_chat=SimpleNamespace(id=self.chat_id),
        )
        await self.bot.vision_action(update, self.context)
        outputs = (
            tuple(query.edits)
            + tuple(answer for answer, _show_alert in query.answers if answer is not None)
            + tuple(str(reply["text"]) for reply in message.replies[previous_replies:])
        )
        return StepResult(kind, value, outputs)

    def _latest_vision_callback(self, action: str) -> tuple[str, FakeMessage]:
        for message in reversed(self.messages):
            for reply in reversed(message.replies):
                markup = reply.get("reply_markup")
                if markup is None:
                    continue
                for row in markup.inline_keyboard:
                    for button in row:
                        data = button.callback_data
                        if data is None:
                            continue
                        if action in CATEGORY_META:
                            matches = data.startswith("vision:cat:") and data.endswith(f":{action}")
                        elif action == "editwish":
                            matches = data.startswith("vision:editfield:") and data.endswith(
                                ":wish"
                            )
                        elif action == "renderall":
                            matches = data.startswith("vision:renderpick:") and data.endswith(
                                ":all"
                            )
                        elif action == "rendertravel":
                            matches = data.startswith("vision:renderpick:") and data.endswith(
                                ":travel"
                            )
                        elif action == "rendermoney":
                            matches = data.startswith("vision:renderpick:") and data.endswith(
                                ":money"
                            )
                        elif action == "download":
                            matches = data.startswith("vision:renderdownload:")
                        else:
                            matches = data.startswith(f"vision:{action}")
                        if matches:
                            return data, message
        raise AssertionError(f"No vision callback found for {action!r}")

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
