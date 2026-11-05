"""Cross-Source Conflict Record — data models for C2 (CSCG).

When two ``TemporalKnowledgeUnit`` items from different sources make
contradictory claims about the same entity during overlapping time
windows, a ``ConflictRecord`` captures the disagreement.

Detection logic lives in ``finsage.mind.conflict`` (layer 4); this
module is pure data models in the knowledge leaf (layer 1). No LLM,
no external service imports.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# ── Enums ────────────────────────────────────────────────────────────


class ConflictType(str, Enum):
    """Category of disagreement between two TKUs."""

    VALUE_MISMATCH = "value_mismatch"
    DIRECTION_MISMATCH = "direction_mismatch"
    SIGNIFICANCE_MISMATCH = "significance_mismatch"


class ResolutionStatus(str, Enum):
    """How (if at all) a conflict has been resolved."""

    UNRESOLVED = "unresolved"
    RESOLVED_BY_MARKET = "resolved_by_market"
    RESOLVED_BY_PERIOD = "resolved_by_period"
    RESOLVED_BY_METHODOLOGY = "resolved_by_methodology"
    CONTEXTUAL_SYNTHESIS = "contextual_synthesis"


# ── Core model ───────────────────────────────────────────────────────


class ConflictRecord(BaseModel):
    """A detected contradiction between two temporal knowledge units.

    All TKU data is stored inline (content, source titles, validity
    windows) so the record is self-contained — no store lookups are
    needed to display or evaluate a conflict.

    The ``synthesis_text`` and ``resolution_explanation`` fields are
    populated by the ``mind`` layer (LLM-powered), not here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    conflict_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this conflict record.",
    )

    # ── TKU A ────────────────────────────────────────────────────────
    tku_a_id: str = Field(..., description="ID of the first conflicting TKU.")
    tku_b_id: str = Field(..., description="ID of the second conflicting TKU.")
    tku_a_content: str = Field(
        ..., min_length=1, description="Claim text of TKU A."
    )
    tku_b_content: str = Field(
        ..., min_length=1, description="Claim text of TKU B."
    )
    tku_a_source: str = Field(
        ..., min_length=1, description="Paper title of TKU A."
    )
    tku_b_source: str = Field(
        ..., min_length=1, description="Paper title of TKU B."
    )

    # ── Temporal windows (inline copies from TKUs) ───────────────────
    tku_a_valid_from: datetime | None = None
    tku_a_valid_until: datetime | None = None
    tku_b_valid_from: datetime | None = None
    tku_b_valid_until: datetime | None = None

    # ── Conflict metadata ────────────────────────────────────────────
    conflict_type: ConflictType = Field(
        ..., description="Category of the disagreement."
    )
    entity_name: str = Field(
        ...,
        min_length=1,
        description=(
            "What the two TKUs disagree about, "
            'e.g. "momentum factor", "Fed funds rate".'
        ),
    )

    # ── Temporal overlap (computed by detection logic) ────────────────
    temporal_overlap_start: datetime | None = Field(
        default=None,
        description="Start of the overlapping validity period.",
    )
    temporal_overlap_end: datetime | None = Field(
        default=None,
        description="End of the overlapping validity period.",
    )

    # ── Resolution ───────────────────────────────────────────────────
    resolution_status: ResolutionStatus = Field(
        default=ResolutionStatus.UNRESOLVED,
        description="How this conflict has been resolved (if at all).",
    )
    resolution_explanation: str | None = Field(
        default=None,
        description="Human/LLM explanation of the resolution.",
    )
    synthesis_text: str | None = Field(
        default=None,
        description=(
            "LLM-generated synthesis reconciling the two claims. "
            "Set by the mind layer, not here."
        ),
    )

    # ── Confidence & timestamp ───────────────────────────────────────
    detection_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence that this is a genuine conflict.",
    )
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this conflict was detected.",
    )

    # ── Convenience methods ──────────────────────────────────────────

    def is_resolved(self) -> bool:
        """True iff the conflict has any resolution status other than UNRESOLVED."""
        return self.resolution_status != ResolutionStatus.UNRESOLVED

    def user_facing_note(self) -> str:
        """Return the synthesis text if available, else a formatted description."""
        if self.synthesis_text:
            return self.synthesis_text
        return (
            f"Conflict ({self.conflict_type.value}): "
            f'"{self.tku_a_content}" vs "{self.tku_b_content}" '
            f"on entity '{self.entity_name}'."
        )

    def pair_hash(self) -> str:
        """Deterministic hash of the TKU pair, order-independent.

        Useful for deduplication: the same two TKUs always produce the
        same pair hash regardless of which is A vs B.
        """
        ids = sorted([self.tku_a_id, self.tku_b_id])
        payload = f"{ids[0]}::{ids[1]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
