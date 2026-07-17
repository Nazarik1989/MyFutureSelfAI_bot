from datetime import date, datetime, time
from typing import Literal

from pydantic import BaseModel, Field


class VisionSummary(BaseModel):
    summary: str
    values: list[str] = Field(default_factory=list)
    desired_identity: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    motivation_style: str | None = None


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


class TodayPlan(BaseModel):
    vision_reminder: str
    main_focus: str
    actions: list[str] = Field(max_length=3)
    hard_day_minimum: str
