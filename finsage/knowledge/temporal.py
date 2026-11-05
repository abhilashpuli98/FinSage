"""Temporal Knowledge Unit (TKU) — time-bounded financial facts.

Novel contribution C1: every extracted fact gets explicit validity windows
(``valid_from``, ``valid_until``) so retrieval can filter stale knowledge and
research can measure temporal decay.

Design:

- ``FinanceDomain`` and ``KnowledgeType`` classify each unit.
- ``TemporalKnowledgeUnit`` is a frozen Pydantic v2 model carrying one
  atomic factual claim with its validity window and confidence scores.
- ``TKUCollection`` groups units extracted from a single document.

All models are frozen Pydantic v2 with ``extra="forbid"`` — no LLM, no
external service imports.  ``knowledge/`` is a dependency leaf.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Enums ────────────────────────────────────────────────────────────


class FinanceDomain(str, Enum):
    """Primary domain tag for a temporal fact."""

    FACTOR_INVESTING = "factor_investing"
    OPTIONS_PRICING = "options_pricing"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    CREDIT_RISK = "credit_risk"
    ML_FINANCE = "ml_finance"
    BEHAVIORAL_FINANCE = "behavioral_finance"
    MACRO_ECONOMICS = "macro_economics"


class KnowledgeType(str, Enum):
    """Semantic category of the extracted knowledge unit."""

    FACT = "fact"
    DEFINITION = "definition"
    CLAIM = "claim"
    FORMULA = "formula"
    STATISTIC = "statistic"
    METHODOLOGY = "methodology"


class TemporalDecay(str, Enum):
    """Staleness classification for a temporal fact.

    Used by downstream retrieval to weight or filter facts.
    """

    CURRENT = "current"
    STALE = "stale"
    EXPIRED = "expired"
    UNDATED = "undated"


# ── Core model ───────────────────────────────────────────────────────


def _compute_content_hash(source_id: str, content: str) -> str:
    """SHA-256 hash of ``source_id::content[:100]``, truncated to 16 hex."""
    payload = f"{source_id}::{content[:100]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class TemporalKnowledgeUnit(BaseModel):
    """A single atomic factual claim with temporal validity.

    This is the core model of Contribution C1 (TKU).  Every extracted
    fact carries an explicit ``[valid_from, valid_until)`` window,
    extraction and temporal confidence scores, and a deterministic
    ``content_hash`` for deduplication.

    The model is frozen (immutable) and forbids extra fields, consistent
    with all knowledge-layer types in finsage.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    tku_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this TKU.",
    )
    content: str = Field(
        ...,
        min_length=1,
        description=(
            "The atomic factual claim, e.g. "
            "'Momentum factor earns 1.2% monthly premium (1963-2022).'"
        ),
    )
    content_type: KnowledgeType = Field(
        default=KnowledgeType.FACT,
        description="Semantic category of this knowledge unit.",
    )
    source_id: str = Field(
        ...,
        min_length=1,
        description="ID of the originating document.",
    )
    source_title: str = Field(
        ...,
        min_length=1,
        description="Title of the originating document.",
    )
    extraction_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this unit was extracted.",
    )

    # ── Temporal validity window [valid_from, valid_until) ────────────
    valid_from: datetime | None = Field(
        default=None,
        description="Start of the fact's evidence period (inclusive).",
    )
    valid_until: datetime | None = Field(
        default=None,
        description="End of the fact's evidence period (exclusive, None = assumed current).",
    )

    # ── Confidence scores ────────────────────────────────────────────
    temporal_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in the extracted validity window.",
    )
    extraction_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in the content extraction itself.",
    )

    # ── Optional metadata ────────────────────────────────────────────
    market: str | None = Field(
        default=None,
        description='Market scope, e.g. "US Equities", "China A-Share".',
    )
    domain: FinanceDomain | None = Field(
        default=None,
        description="Primary finance domain.",
    )
    tags: tuple[str, ...] = Field(
        default=(),
        description="Free-form tags for classification.",
    )
    page_references: tuple[int, ...] = Field(
        default=(),
        description="Page numbers in the source document.",
    )
    content_hash: str = Field(
        default="",
        description=(
            "SHA-256 hash of source_id::content[:100] (truncated to 16 hex). "
            "Computed automatically."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _set_content_hash(cls, data: dict) -> dict:  # type: ignore[type-arg]
        """Compute content_hash from source_id and content if not set."""
        if isinstance(data, dict):
            content = data.get("content", "")
            source_id = data.get("source_id", "")
            existing = data.get("content_hash", "")
            if not existing and content and source_id:
                data["content_hash"] = _compute_content_hash(
                    str(source_id), str(content)
                )
        return data

    # ── Pure query methods (no side effects, no LLM calls) ───────────

    def is_valid_at(self, query_date: datetime) -> bool:
        """Point-in-time check. True if evidence is valid at *query_date*."""
        if self.valid_from is not None and query_date < self.valid_from:
            return False
        if self.valid_until is not None and query_date >= self.valid_until:
            return False
        return True

    def composite_confidence(self) -> float:
        """Combined confidence: extraction_confidence × temporal_confidence."""
        return self.extraction_confidence * self.temporal_confidence

    def temporal_scope(self) -> str:
        """Human-readable validity scope, e.g. ``'1990–2022 (US Equities)'``."""
        start = self.valid_from.strftime("%Y") if self.valid_from else "?"
        end = self.valid_until.strftime("%Y") if self.valid_until else "present"
        scope = f"{start}\u2013{end}"
        if self.market:
            scope = f"{scope} ({self.market})"
        return scope

    def decay_status(self, now: datetime | None = None) -> TemporalDecay:
        """Classify staleness relative to *now* (defaults to UTC now)."""
        reference = now or datetime.now(timezone.utc)
        if self.valid_from is None and self.valid_until is None:
            return TemporalDecay.UNDATED
        if self.valid_until is not None and reference >= self.valid_until:
            return TemporalDecay.EXPIRED
        if self.valid_from is not None and reference >= self.valid_from:
            return TemporalDecay.CURRENT
        return TemporalDecay.CURRENT


# ── Collection model ─────────────────────────────────────────────────


class TKUCollection(BaseModel):
    """A set of TKUs extracted from a single document.

    Provides filtering and aggregation helpers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    collection_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this collection.",
    )
    document_id: str = Field(
        ...,
        min_length=1,
        description="ID of the source document.",
    )
    source_title: str = Field(
        ...,
        min_length=1,
        description="Title of the source document.",
    )
    tkus: tuple[TemporalKnowledgeUnit, ...] = Field(
        default=(),
        description="The temporal knowledge units in this collection.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this collection was created.",
    )

    # ── Query helpers ─────────────────────────────────────────────────

    def filter_by_date(
        self, query_date: datetime
    ) -> tuple[TemporalKnowledgeUnit, ...]:
        """Return TKUs whose validity window includes *query_date*."""
        return tuple(t for t in self.tkus if t.is_valid_at(query_date))

    def by_domain(
        self, domain: FinanceDomain
    ) -> tuple[TemporalKnowledgeUnit, ...]:
        """Return TKUs matching the given finance domain."""
        return tuple(t for t in self.tkus if t.domain == domain)

    def stats(self) -> dict[str, int]:
        """Count TKUs per ``KnowledgeType``."""
        counts: dict[str, int] = {}
        for tku in self.tkus:
            key = tku.content_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts
