"""Data models for FinBench-50 and automated evaluation.

Defines Pydantic v2 immutable models for evaluation questions, system answers,
and evaluation scores.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from finsage.knowledge.temporal import FinanceDomain

__all__ = [
    "BenchmarkQuestion",
    "EvalResult",
    "QuestionType",
    "SystemAnswer",
]


class QuestionType(str, Enum):
    """Classification of benchmark questions."""

    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    TEMPORAL = "temporal"


class BenchmarkQuestion(BaseModel):
    """An evaluation question in FinBench-50.

    Each question is tied to a specific paper and finance domain.
    Factual questions have verifiable ground truths; temporal questions
    contain point-in-time constraints; analytical questions require
    multi-fact synthesis without a single ground-truth string.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    paper_title: str
    domain: FinanceDomain
    question_text: str
    question_type: QuestionType
    ground_truth: str | None = None
    temporal_constraint: datetime | None = None
    difficulty: str = Field(default="medium", pattern=r"^(easy|medium|hard)$")
    expected_reasoning: str | None = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @field_validator("domain", mode="before")
    @classmethod
    def _validate_domain(cls, v: object) -> object:
        if isinstance(v, str):
            for d in FinanceDomain:
                if d.value == v.lower() or d.name == v.upper():
                    return d
        return v


class SystemAnswer(BaseModel):
    """An answer produced by one of the evaluation systems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    system_name: str
    answer_text: str
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    temporal_scope: str | None = None
    has_conflict_note: bool = False
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class EvalResult(BaseModel):
    """Evaluation result for one system on one benchmark question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question_id: str
    system_name: str
    score: float = Field(ge=0.0, le=1.0)
    scorer: str
    notes: str | None = None
