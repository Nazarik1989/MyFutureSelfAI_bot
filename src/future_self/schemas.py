import unicodedata
from datetime import date, datetime, time
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

GuestAction = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]

NOVA_HELP_MAX_PAYLOAD_BYTES = 4 * 1024
NovaHelpStep = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
WeeklyReviewSmallStep = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]


class VisionSummary(BaseModel):
    summary: str
    values: list[str] = Field(default_factory=list)
    desired_identity: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    motivation_style: str | None = None


class TimezoneResolution(BaseModel):
    timezone: str | None = Field(default=None, max_length=64)
    city: str | None = Field(default=None, max_length=120)
    country: str | None = Field(default=None, max_length=120)
    ambiguous: bool = False


class ReminderTimezoneResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["resolved", "not_mentioned", "ambiguous", "insufficient"]
    timezone: str | None = Field(default=None, max_length=64)
    matched_text: str | None = Field(default=None, max_length=120)
    city: str | None = Field(default=None, max_length=120)
    country: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        if self.status == "resolved":
            if not self.timezone or not self.matched_text:
                raise ValueError("resolved reminder timezone requires timezone and matched_text")
        elif self.status == "ambiguous":
            if self.timezone is not None or not self.matched_text:
                raise ValueError("ambiguous reminder timezone requires matched_text only")
        elif self.timezone is not None or self.matched_text is not None:
            raise ValueError("unresolved reminder timezone must not include resolution fields")
        return self


class WeeklyReviewReminderCandidate(BaseModel):
    """Untrusted provider output retained only until evidence verification."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200)
    schedule_wording: str = Field(min_length=1, max_length=200)
    evidence: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def fields_are_grounded_in_evidence(self) -> Self:
        if self.schedule_wording not in self.evidence:
            raise ValueError("weekly reminder schedule must be copied from its evidence")
        title = unicodedata.normalize("NFKC", self.title).casefold()
        evidence = unicodedata.normalize("NFKC", self.evidence).casefold()
        if title not in evidence:
            raise ValueError("weekly reminder title must be copied from its evidence")
        return self


class WeeklyReviewExtraction(BaseModel):
    """Strict, transient result of weekly-review input extraction."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    focus: str = Field(min_length=1, max_length=300)
    approach: str | None = Field(default=None, min_length=1, max_length=500)
    small_steps: list[WeeklyReviewSmallStep] = Field(
        default_factory=list,
        min_length=0,
        max_length=3,
    )
    reminder_candidates: list[WeeklyReviewReminderCandidate] = Field(
        default_factory=list,
        min_length=0,
        max_length=5,
    )

    @model_validator(mode="after")
    def candidate_evidence_is_not_reused(self) -> Self:
        evidence = [candidate.evidence for candidate in self.reminder_candidates]
        if len(evidence) != len(set(evidence)):
            raise ValueError("weekly reminder evidence must not be reused")
        return self


class GoalProposal(BaseModel):
    life_area: str
    title: str
    outcome: str
    progress_criterion: str
    horizon: str
    priority: int = Field(ge=1, le=5)
    vision_link: str


class GoalProposals(BaseModel):
    goals: list[GoalProposal] = Field(min_length=3, max_length=5)


class RoutineProposal(BaseModel):
    goal_title: str
    frequency: str
    minimum_version: str
    normal_version: str
    preferred_time: str | None = None


class RoutineProposals(BaseModel):
    routines: list[RoutineProposal] = Field(max_length=3)


class TemporalResolution(BaseModel):
    resolved_at: datetime
    remind_at: datetime | None = None
    timezone: str
    resolved_local_date: date
    resolved_local_time: time | None = None
    precision: Literal["date", "datetime"]
    original_expression: str
    resolution_status: Literal["resolved"] = "resolved"


class ParsedThought(BaseModel):
    kind: Literal["idea", "task", "desire", "note"]
    title: str = Field(min_length=1, max_length=200)
    description: str | None = None
    next_step: str | None = None
    resolved_date: date | None = None
    temporal_resolution: TemporalResolution | None = None


class GuestThoughtBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: Literal["idea", "task", "desire", "note"]
    title: str = Field(min_length=1, max_length=120)
    essence: str = Field(min_length=1, max_length=500)
    next_step: str = Field(min_length=1, max_length=300)


class GuestFirstStep(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    focus: str = Field(min_length=1, max_length=300)
    first_step: str = Field(min_length=1, max_length=300)
    actions: list[GuestAction] = Field(min_length=0, max_length=3)


class NovaHelpPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    response: str = Field(min_length=1, max_length=800)
    steps: list[NovaHelpStep] = Field(default_factory=list, min_length=0, max_length=3)
    action_id: str | None = Field(default=None, min_length=1, max_length=100)
    kind: Literal["guide", "clarify", "unsupported"]

    @model_validator(mode="after")
    def serialized_payload_fits_limit(self) -> "NovaHelpPlan":
        if len(self.model_dump_json().encode("utf-8")) > NOVA_HELP_MAX_PAYLOAD_BYTES:
            raise ValueError("Nova help payload exceeds the 4 KiB limit")
        return self


MessageIntent = Literal[
    "conversation",
    "question",
    "inbox_idea",
    "inbox_task",
    "inbox_desire",
    "inbox_note",
    "reflection",
    "explicit_capture",
    "unknown",
    "shared_idea",
]


class IntentResult(BaseModel):
    intent: MessageIntent
    confidence: float = Field(ge=0, le=1)
    inbox_kind: Literal["idea", "task", "desire", "note"] | None = None
    title: str | None = Field(default=None, max_length=200)
    next_step: str | None = None
    answer: str | None = None
    topic: str | None = Field(default=None, max_length=200)


class AssistantAnswer(BaseModel):
    answer: str = Field(min_length=1, max_length=2000)


class NovaCompanionCapture(BaseModel):
    """Grounded suggestion exposed to UI code after evidence has been discarded."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: Literal["idea", "task", "desire", "note"]
    title: str = Field(min_length=1, max_length=200, repr=False)
    next_step: str | None = Field(default=None, min_length=1, max_length=300, repr=False)


class NovaCompanionProviderCapture(NovaCompanionCapture):
    """Untrusted provider suggestion with validation-only current-message evidence."""

    evidence: str = Field(min_length=1, max_length=500, repr=False, exclude=True)


class NovaCompanionReminderOffer(BaseModel):
    """Server-validated, non-mutating reminder proposal for the UI layer."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=200, repr=False)
    schedule_wording: str | None = Field(default=None, min_length=1, max_length=160, repr=False)
    evidence: str = Field(min_length=1, max_length=600, repr=False, exclude=True)


class NovaCompanionProviderReminderOffer(NovaCompanionReminderOffer):
    """Untrusted provider proposal; evidence is retained only for server validation."""


NovaDialogueAction = Literal["capture", "reminder", "plan", "memory", "clarify"]
NovaDialogueOfferKind = Literal["method", "exercise", "reminder_setup", "plan"]
NovaMemoryCategory = Literal["fact", "preference", "orientation", "theme", "identity"]
NovaObservedMemoryKey = Literal[
    "identity",
    "response_length",
    "tone",
    "reminder_style",
]
NovaCompanionDiagnosticCode = Literal[
    "invalid_answer",
    "invalid_capture",
    "invalid_reminder_offer",
    "invalid_dialogue_state",
    "invalid_memory_candidate",
    "conflicting_actions",
]


class NovaCompanionDialogueStateUpdate(BaseModel):
    """Untrusted bounded working-state proposal; never an execution capability."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    active_topic: str | None = Field(default=None, min_length=1, max_length=200, repr=False)
    current_user_goal: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        repr=False,
    )
    last_assistant_offer: str | None = Field(
        default=None,
        min_length=1,
        max_length=600,
        repr=False,
    )
    last_assistant_offer_kinds: list[NovaDialogueOfferKind] = Field(
        default_factory=list,
        max_length=4,
        repr=False,
    )
    unresolved_question: str | None = Field(
        default=None,
        min_length=1,
        max_length=300,
        repr=False,
    )
    requested_action: NovaDialogueAction | None = None
    open_loops: list[Annotated[str, StringConstraints(min_length=1, max_length=200)]] = Field(
        default_factory=list,
        max_length=5,
        repr=False,
    )
    clear_fields: list[
        Literal[
            "active_topic",
            "current_user_goal",
            "last_assistant_offer",
            "unresolved_question",
            "requested_action",
            "open_loops",
        ]
    ] = Field(default_factory=list, max_length=6, repr=False)

    @model_validator(mode="after")
    def validate_offer_and_clear_fields(self) -> Self:
        if bool(self.last_assistant_offer) != bool(self.last_assistant_offer_kinds):
            raise ValueError("assistant offer text and kinds must be supplied together")
        if len(set(self.last_assistant_offer_kinds)) != len(self.last_assistant_offer_kinds):
            raise ValueError("assistant offer kinds must be unique")
        if len(set(self.clear_fields)) != len(self.clear_fields):
            raise ValueError("dialogue clear fields must be unique")
        supplied = []
        for name in self.clear_fields:
            value = getattr(self, name, None)
            if (name == "open_loops" and bool(value)) or (
                name != "open_loops" and value is not None
            ):
                supplied.append(name)
        if supplied:
            raise ValueError("dialogue field cannot be updated and cleared together")
        return self


class NovaCompanionMemoryCandidate(BaseModel):
    """One untrusted structured-memory proposal grounded in the current user turn."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: NovaMemoryCategory
    key: NovaObservedMemoryKey | None = None
    value: str = Field(min_length=1, max_length=500, repr=False)
    evidence: str = Field(min_length=1, max_length=600, repr=False, exclude=True)
    salience: int = Field(default=3, ge=1, le=5)
    supersedes_value: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        repr=False,
    )


class NovaCompanionProviderResponse(BaseModel):
    """Private structured-output shape used only at the provider boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=2000, repr=False)
    capture: NovaCompanionProviderCapture | None = Field(default=None, repr=False)
    reminder_offer: NovaCompanionProviderReminderOffer | None = Field(default=None, repr=False)
    dialogue_state_update: NovaCompanionDialogueStateUpdate | None = Field(
        default=None,
        repr=False,
    )
    memory_candidate: NovaCompanionMemoryCandidate | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def one_optional_offer(self) -> Self:
        if self.capture is not None and self.reminder_offer is not None:
            raise ValueError("capture and reminder_offer are mutually exclusive")
        return self


class NovaCompanionProviderTransport(BaseModel):
    """Shallow provider envelope; proposals are independently encoded JSON values."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=2000, repr=False)
    capture: str | None = Field(default=None, repr=False)
    reminder_offer: str | None = Field(default=None, repr=False)
    dialogue_state_update: str | None = Field(default=None, repr=False)
    memory_candidate: str | None = Field(default=None, repr=False)


class NovaCompanionResponse(BaseModel):
    """One conversation-first answer with at most one non-mutating capture offer."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=2000, repr=False)
    capture: NovaCompanionCapture | None = Field(default=None, repr=False)
    reminder_offer: NovaCompanionReminderOffer | None = Field(default=None, repr=False)
    dialogue_state_update: NovaCompanionDialogueStateUpdate | None = Field(
        default=None,
        repr=False,
    )
    memory_candidate: NovaCompanionMemoryCandidate | None = Field(default=None, repr=False)
    memory_rejected: bool = Field(default=False, exclude=True)
    diagnostic_codes: tuple[NovaCompanionDiagnosticCode, ...] = Field(
        default=(),
        max_length=5,
        repr=False,
        exclude=True,
    )

    @model_validator(mode="after")
    def one_optional_offer(self) -> Self:
        if self.capture is not None and self.reminder_offer is not None:
            raise ValueError("capture and reminder_offer are mutually exclusive")
        return self


class TodayPlan(BaseModel):
    vision_reminder: str
    main_focus: str
    actions: list[str] = Field(max_length=3)
    hard_day_minimum: str
