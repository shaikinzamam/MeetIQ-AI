"""Pydantic models = the contract for MeetIQ's structured output.

The LLM is asked to produce `MeetingAnalysis`. We validate it with Pydantic so the
rest of the app can trust the shape of the data. Validators below are forgiving about
small LLM quirks (e.g. "High" vs "high", null instead of an empty list).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Level = Literal["high", "medium", "low"]


def _normalize_level(value: object, default: str = "medium") -> str:
    if value is None:
        return default
    value = str(value).strip().lower()
    return value if value in {"high", "medium", "low"} else default


class Decision(BaseModel):
    decision: str
    context: str | None = None


class ActionItem(BaseModel):
    task: str
    owner: str = "Unassigned"
    deadline: str | None = None
    priority: Level = "medium"

    @field_validator("owner", mode="before")
    @classmethod
    def _owner_default(cls, v):
        return "Unassigned" if v is None or not str(v).strip() else str(v).strip()

    @field_validator("deadline", mode="before")
    @classmethod
    def _deadline_clean(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return None if v.lower() in {"", "none", "null", "n/a", "not specified", "tbd"} else v

    @field_validator("priority", mode="before")
    @classmethod
    def _priority_norm(cls, v):
        return _normalize_level(v)


class RiskOrBlocker(BaseModel):
    description: str
    type: Literal["risk", "blocker"] = "risk"
    severity: Level = "medium"
    mitigation: str | None = None

    @field_validator("severity", mode="before")
    @classmethod
    def _severity_norm(cls, v):
        return _normalize_level(v)

    @field_validator("type", mode="before")
    @classmethod
    def _type_norm(cls, v):
        v = str(v).strip().lower() if v else "risk"
        return v if v in {"risk", "blocker"} else "risk"


class MeetingAnalysis(BaseModel):
    """What the LLM produces."""

    title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    risks_blockers: list[RiskOrBlocker] = Field(default_factory=list)

    @field_validator("key_points", "decisions", "action_items", "risks_blockers", mode="before")
    @classmethod
    def _none_to_list(cls, v):
        return [] if v is None else v


class EfficiencyMetrics(BaseModel):
    """Computed by our own code (NOT the LLM), so the numbers are reproducible."""

    transcript_words: int
    est_read_minutes_transcript: float
    est_read_minutes_report: float
    est_minutes_saved_per_reader: float
    est_seconds_saved: int = 0
    action_items_total: int
    action_items_with_owner: int
    action_items_with_deadline: int
    accountability_score: float | None = Field(
        default=None,
        description="% of action items that have BOTH an owner and a deadline (0-100).",
    )


class MeetingReport(BaseModel):
    """Final API/UI response."""

    analysis: MeetingAnalysis
    metrics: EfficiencyMetrics
    model_used: str