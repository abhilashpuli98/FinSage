"""Cross-Source Conflict Detection and Resolution (C2: CSCG).

Contribution C2 detects when two ``TemporalKnowledgeUnit`` items from
different sources make contradictory claims about the same entity during
overlapping time windows, then attempts resolution via four deterministic
and LLM-assisted strategies.

Architecture
~~~~~~~~~~~~

This module lives in ``finsage.mind`` (layer 4).  It imports from:

- ``finsage.knowledge.temporal`` — ``TemporalKnowledgeUnit``, ``TKUCollection``, ``FinanceDomain``
- ``finsage.knowledge.conflict`` — ``ConflictRecord``, ``ConflictType``, ``ResolutionStatus``
- ``finsage.configs.ollama`` — ``TaskType``, ``ModelTier``
- ``finsage.mind.router`` — ``LLMRouter``, ``safe_json_parse``

No imports from ``preprocess``, ``rag``, ``flows``, or ``etl``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np

from finsage.configs.ollama import ModelTier, TaskType
from finsage.knowledge.conflict import (
    ConflictRecord,
    ConflictType,
    ResolutionStatus,
)
from finsage.knowledge.temporal import (
    TemporalKnowledgeUnit,
    TKUCollection,
)
from finsage.mind.router import LLMRouter, safe_json_parse

_LOGGER = logging.getLogger(__name__)

# Maximum number of candidate pairs to evaluate with LLM calls.
# Guards against quadratic cost explosions on large TKU sets.
_MAX_LLM_CANDIDATES = 1000

#  Embedding similarity


def compute_cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1].  Returns 0.0 when either vector
        has zero norm.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    denominator = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denominator)


#  Temporal overlap helper


def _temporal_overlap(
    a: TemporalKnowledgeUnit,
    b: TemporalKnowledgeUnit,
) -> tuple[datetime | None, datetime | None] | None:
    """Return the overlap interval of two TKUs, or ``None`` if disjoint.

    Both open-ended (``None``) boundaries are treated as unbounded —
    two TKUs with ``valid_until=None`` and ``valid_from=None`` always
    overlap.  A result with both entries ``None`` means "fully
    unbounded overlap".
    """
    start_a = a.valid_from
    end_a = a.valid_until
    start_b = b.valid_from
    end_b = b.valid_until

    # Compute the later start and earlier end.
    if start_a is not None and start_b is not None:
        overlap_start = max(start_a, start_b)
    else:
        overlap_start = start_a or start_b  # whichever is set, or None

    if end_a is not None and end_b is not None:
        overlap_end = min(end_a, end_b)
    else:
        overlap_end = end_a or end_b  # whichever is set, or None

    # Check for disjoint windows.
    if overlap_start is not None and overlap_end is not None:
        if overlap_start >= overlap_end:
            return None

    return overlap_start, overlap_end


#  Conflict-detection prompt

_CONFLICT_DETECTION_PROMPT = """\
Are these two financial research claims about the same phenomenon?
If yes, do they agree or disagree on the key finding?

Claim A: {content_a}
Source A: {source_a} | Period: {period_a} | Market: {market_a}

Claim B: {content_b}
Source B: {source_b} | Period: {period_b} | Market: {market_b}

Return ONLY valid JSON, no other text:
{{"same_claim": true/false, "agreement": "agree"|"disagree"|"partial"|"different_scope", \
"conflict_type": "VALUE_MISMATCH"|"DIRECTION_MISMATCH"|"SIGNIFICANCE_MISMATCH"|null, \
"detection_confidence": 0.0-1.0, "reason": "one sentence"}}"""

_METHODOLOGY_PROMPT = """\
Could different methodologies explain the following conflict between two \
financial research claims?

Claim A: {content_a}
Tags A: {tags_a}

Claim B: {content_b}
Tags B: {tags_b}

Return ONLY valid JSON, no other text:
{{"methodology_explains": true/false, "explanation": "one sentence", \
"confidence": 0.0-1.0}}"""

_SYNTHESIS_PROMPT = """\
Synthesize the following conflicting financial research claims into one \
balanced, objective sentence suitable for a research survey.

Claim A: {content_a} (Source: {source_a}, Period: {period_a})
Claim B: {content_b} (Source: {source_b}, Period: {period_b})

Return ONLY the synthesis sentence, nothing else."""


#  ConflictDetector ─


def _period_str(tku: TemporalKnowledgeUnit) -> str:
    """Format validity window as a human-readable string."""
    start = tku.valid_from.strftime("%Y") if tku.valid_from else "?"
    end = tku.valid_until.strftime("%Y") if tku.valid_until else "present"
    return f"{start}–{end}"


class ConflictDetector:
    """Detect contradictions between TKUs across different sources.

    Algorithm (Algorithm 2 in the paper):

    1. Flatten all TKU collections into a single list.
    2. Pre-filter candidate pairs: skip same-source, skip different
       domains, skip temporally disjoint windows.
    3. For each candidate pair, compute embedding cosine similarity via
       the router.  Only proceed to LLM confirmation if similarity
       exceeds ``semantic_threshold``.
    4. Use an LLM (EXTRACTION task) to confirm whether the pair is a
       genuine conflict.  Parse the structured JSON response.
    5. Return all confirmed ``ConflictRecord`` objects.
    """

    def __init__(
        self,
        router: LLMRouter,
        semantic_threshold: float = 0.82,
        max_candidates: int = _MAX_LLM_CANDIDATES,
    ) -> None:
        """Bind router and detection policy.

        Args:
            router: LLM router for embeddings and conflict confirmation.
            semantic_threshold: Minimum cosine similarity to proceed to
                LLM confirmation.
            max_candidates: Maximum number of pairs to evaluate with
                LLM calls.
        """
        self._router = router
        self._semantic_threshold = semantic_threshold
        self._max_candidates = max_candidates

    async def detect(
        self,
        tku_collections: list[TKUCollection],
    ) -> list[ConflictRecord]:
        """Run the full detection pipeline on a set of TKU collections.

        Args:
            tku_collections: One or more TKU collections from different
                source documents.

        Returns:
            A list of confirmed ``ConflictRecord`` objects.
        """
        # Step 1 — Flatten.
        all_tkus = [t for coll in tku_collections for t in coll.tkus]
        if len(all_tkus) < 2:
            return []

        # Step 2 — Pre-filter candidate pairs.
        candidates = self._prefilter(all_tkus)

        # Step 3 — Embed and filter by semantic similarity.
        embeddings: dict[str, list[float]] = {}
        semantic_candidates: list[
            tuple[
                TemporalKnowledgeUnit,
                TemporalKnowledgeUnit,
                datetime | None,
                datetime | None,
            ]
        ] = []

        for tku_a, tku_b, overlap_start, overlap_end in candidates:
            if len(semantic_candidates) >= self._max_candidates:
                _LOGGER.info(
                    "Reached max candidate limit (%d), stopping pair "
                    "evaluation.",
                    self._max_candidates,
                )
                break

            # Get embeddings (cached per TKU ID to avoid redundant calls).
            try:
                if tku_a.tku_id not in embeddings:
                    embeddings[tku_a.tku_id] = await self._router.embed(
                        tku_a.content
                    )
                if tku_b.tku_id not in embeddings:
                    embeddings[tku_b.tku_id] = await self._router.embed(
                        tku_b.content
                    )
            except RuntimeError:
                # Embedding providers all failed — skip this pair.
                _LOGGER.warning(
                    "Embedding failed for pair (%s, %s), skipping.",
                    tku_a.tku_id,
                    tku_b.tku_id,
                )
                continue

            similarity = compute_cosine_similarity(
                embeddings[tku_a.tku_id],
                embeddings[tku_b.tku_id],
            )
            if similarity >= self._semantic_threshold:
                semantic_candidates.append(
                    (tku_a, tku_b, overlap_start, overlap_end)
                )

        # Step 4 — LLM confirmation.
        conflicts: list[ConflictRecord] = []
        seen_pairs: set[str] = set()

        for tku_a, tku_b, overlap_start, overlap_end in semantic_candidates:
            # Deduplicate — order-independent pair key.
            pair_key = "::".join(sorted([tku_a.tku_id, tku_b.tku_id]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            record = await self._confirm_conflict(
                tku_a, tku_b, overlap_start, overlap_end
            )
            if record is not None:
                conflicts.append(record)

        _LOGGER.info(
            "Conflict detection complete: %d confirmed from %d semantic "
            "candidates (%d pre-filtered pairs).",
            len(conflicts),
            len(semantic_candidates),
            len(candidates),
        )
        return conflicts

    def _prefilter(
        self,
        all_tkus: list[TemporalKnowledgeUnit],
    ) -> list[
        tuple[
            TemporalKnowledgeUnit,
            TemporalKnowledgeUnit,
            datetime | None,
            datetime | None,
        ]
    ]:
        """Pre-filter pairs by source, domain, and temporal overlap."""
        candidates: list[
            tuple[
                TemporalKnowledgeUnit,
                TemporalKnowledgeUnit,
                datetime | None,
                datetime | None,
            ]
        ] = []
        for i, tku_a in enumerate(all_tkus):
            for tku_b in all_tkus[i + 1 :]:
                # Skip same source.
                if tku_a.source_id == tku_b.source_id:
                    continue

                # Skip different domains (must be same to conflict).
                if (
                    tku_a.domain is not None
                    and tku_b.domain is not None
                    and tku_a.domain != tku_b.domain
                ):
                    continue

                # Skip temporally disjoint.
                overlap = _temporal_overlap(tku_a, tku_b)
                if overlap is None:
                    continue

                candidates.append((tku_a, tku_b, overlap[0], overlap[1]))

        return candidates

    async def _confirm_conflict(
        self,
        tku_a: TemporalKnowledgeUnit,
        tku_b: TemporalKnowledgeUnit,
        overlap_start: datetime | None,
        overlap_end: datetime | None,
    ) -> ConflictRecord | None:
        """Use LLM to confirm whether a candidate pair is a real conflict."""
        prompt = _CONFLICT_DETECTION_PROMPT.format(
            content_a=tku_a.content,
            source_a=tku_a.source_title,
            period_a=_period_str(tku_a),
            market_a=tku_a.market or "unspecified",
            content_b=tku_b.content,
            source_b=tku_b.source_title,
            period_b=_period_str(tku_b),
            market_b=tku_b.market or "unspecified",
        )

        try:
            response = await self._router.generate(
                TaskType.EXTRACTION,
                prompt,
                force_tier=ModelTier.API_CHEAP,
            )
        except RuntimeError:
            _LOGGER.warning(
                "LLM generation failed for conflict check (%s, %s).",
                tku_a.tku_id,
                tku_b.tku_id,
            )
            return None

        parsed = safe_json_parse(response.text)
        if not isinstance(parsed, dict):
            _LOGGER.warning(
                "LLM returned unparseable response for conflict check: %s",
                response.text[:200],
            )
            return None

        return _parsed_to_conflict_record(
            parsed, tku_a, tku_b, overlap_start, overlap_end
        )


def _parsed_to_conflict_record(
    parsed: dict[str, Any],
    tku_a: TemporalKnowledgeUnit,
    tku_b: TemporalKnowledgeUnit,
    overlap_start: datetime | None,
    overlap_end: datetime | None,
) -> ConflictRecord | None:
    """Convert a parsed LLM JSON response into a ``ConflictRecord``.

    Returns ``None`` if the LLM response indicates no conflict.
    """
    same_claim = parsed.get("same_claim", False)
    agreement = parsed.get("agreement", "agree")

    if not same_claim or agreement == "agree":
        return None

    conflict_type_str = parsed.get("conflict_type")
    try:
        if not isinstance(conflict_type_str, str):
            raise ValueError("missing conflict_type")
        conflict_type = ConflictType(conflict_type_str.lower())
    except (ValueError, AttributeError):
        # Default to VALUE_MISMATCH when the LLM returns unexpected type.
        conflict_type = ConflictType.VALUE_MISMATCH

    detection_confidence = parsed.get("detection_confidence", 0.5)
    try:
        detection_confidence = float(detection_confidence)
        detection_confidence = max(0.0, min(1.0, detection_confidence))
    except (TypeError, ValueError):
        detection_confidence = 0.5

    reason = parsed.get("reason", "LLM-detected conflict")

    return ConflictRecord(
        tku_a_id=tku_a.tku_id,
        tku_b_id=tku_b.tku_id,
        tku_a_content=tku_a.content,
        tku_b_content=tku_b.content,
        tku_a_source=tku_a.source_title,
        tku_b_source=tku_b.source_title,
        tku_a_valid_from=tku_a.valid_from,
        tku_a_valid_until=tku_a.valid_until,
        tku_b_valid_from=tku_b.valid_from,
        tku_b_valid_until=tku_b.valid_until,
        conflict_type=conflict_type,
        entity_name=reason,
        temporal_overlap_start=overlap_start,
        temporal_overlap_end=overlap_end,
        detection_confidence=detection_confidence,
    )


#  ConflictResolver ─


class ConflictResolver:
    """Attempt resolution of detected conflicts via four strategies.

    Strategies are applied in order — the first one that succeeds wins:

    1. **Market Scope** (no LLM): different markets explain the conflict.
    2. **Temporal Non-Overlap** (no LLM): recheck confirms windows are
       disjoint.
    3. **Methodological** (LLM, API_CHEAP): different research methods
       could explain the disagreement.
    4. **Contextual Synthesis** (LLM, always succeeds): generate one
       balanced sentence.

    Since ``ConflictRecord`` is frozen, every resolution creates a new
    instance via ``model_copy(update=...)``.
    """

    def __init__(self, router: LLMRouter) -> None:
        """Bind router for LLM-assisted resolution strategies.

        Args:
            router: LLM router for methodology and synthesis calls.
        """
        self._router = router

    async def resolve(
        self,
        conflicts: list[ConflictRecord],
        tku_index: dict[str, TemporalKnowledgeUnit],
    ) -> list[ConflictRecord]:
        """Attempt resolution for each conflict.

        Args:
            conflicts: List of detected conflicts.
            tku_index: Map from ``tku_id`` to ``TemporalKnowledgeUnit``
                for fast metadata lookup.

        Returns:
            A new list of ``ConflictRecord`` objects with resolution
            fields populated.
        """
        resolved: list[ConflictRecord] = []
        for conflict in conflicts:
            tku_a = tku_index.get(conflict.tku_a_id)
            tku_b = tku_index.get(conflict.tku_b_id)

            if tku_a is None or tku_b is None:
                _LOGGER.warning(
                    "TKU lookup failed for conflict %s — keeping unresolved.",
                    conflict.conflict_id,
                )
                resolved.append(conflict)
                continue

            result = self._try_market_scope(conflict, tku_a, tku_b)
            if result is not None:
                resolved.append(result)
                continue

            result = self._try_temporal_period(conflict, tku_a, tku_b)
            if result is not None:
                resolved.append(result)
                continue

            result = await self._try_methodology(conflict, tku_a, tku_b)
            if result is not None:
                resolved.append(result)
                continue

            result = await self._synthesize(conflict, tku_a, tku_b)
            resolved.append(result)

        return resolved

    @staticmethod
    def _try_market_scope(
        conflict: ConflictRecord,
        tku_a: TemporalKnowledgeUnit,
        tku_b: TemporalKnowledgeUnit,
    ) -> ConflictRecord | None:
        """Resolve by different market scopes (no LLM needed)."""
        if tku_a.market and tku_b.market and tku_a.market != tku_b.market:
            explanation = f"Different markets: {tku_a.market} vs {tku_b.market}"
            return conflict.model_copy(
                update={
                    "resolution_status": ResolutionStatus.RESOLVED_BY_MARKET,
                    "resolution_explanation": explanation,
                }
            )
        return None

    @staticmethod
    def _try_temporal_period(
        conflict: ConflictRecord,
        tku_a: TemporalKnowledgeUnit,
        tku_b: TemporalKnowledgeUnit,
    ) -> ConflictRecord | None:
        """Resolve by non-overlapping time periods (no LLM needed)."""
        overlap = _temporal_overlap(tku_a, tku_b)
        if overlap is None:
            explanation = (
                f"Non-overlapping periods: "
                f"{_period_str(tku_a)} vs {_period_str(tku_b)}"
            )
            return conflict.model_copy(
                update={
                    "resolution_status": ResolutionStatus.RESOLVED_BY_PERIOD,
                    "resolution_explanation": explanation,
                }
            )
        return None

    async def _try_methodology(
        self,
        conflict: ConflictRecord,
        tku_a: TemporalKnowledgeUnit,
        tku_b: TemporalKnowledgeUnit,
    ) -> ConflictRecord | None:
        """Resolve by methodological differences (LLM, API_CHEAP)."""
        prompt = _METHODOLOGY_PROMPT.format(
            content_a=tku_a.content,
            tags_a=", ".join(tku_a.tags) if tku_a.tags else "none",
            content_b=tku_b.content,
            tags_b=", ".join(tku_b.tags) if tku_b.tags else "none",
        )

        try:
            response = await self._router.generate(
                TaskType.VERIFICATION,
                prompt,
                force_tier=ModelTier.API_CHEAP,
            )
        except RuntimeError:
            _LOGGER.warning(
                "LLM methodology check failed for conflict %s.",
                conflict.conflict_id,
            )
            return None

        parsed = safe_json_parse(response.text)
        if not isinstance(parsed, dict):
            return None

        methodology_explains = parsed.get("methodology_explains", False)
        confidence = parsed.get("confidence", 0.0)

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        if methodology_explains and confidence > 0.7:
            explanation = parsed.get(
                "explanation", "Different methodologies explain the conflict."
            )
            return conflict.model_copy(
                update={
                    "resolution_status": (
                        ResolutionStatus.RESOLVED_BY_METHODOLOGY
                    ),
                    "resolution_explanation": str(explanation),
                }
            )
        return None

    async def _synthesize(
        self,
        conflict: ConflictRecord,
        tku_a: TemporalKnowledgeUnit,
        tku_b: TemporalKnowledgeUnit,
    ) -> ConflictRecord:
        """Generate a balanced synthesis sentence (LLM, always succeeds)."""
        prompt = _SYNTHESIS_PROMPT.format(
            content_a=tku_a.content,
            source_a=tku_a.source_title,
            period_a=_period_str(tku_a),
            content_b=tku_b.content,
            source_b=tku_b.source_title,
            period_b=_period_str(tku_b),
        )

        try:
            response = await self._router.generate(
                TaskType.GENERATION,
                prompt,
                force_tier=ModelTier.API_CHEAP,
            )
            synthesis = response.text.strip()
        except RuntimeError:
            synthesis = (
                f"Research is mixed on this topic: "
                f'"{tku_a.content}" ({tku_a.source_title}, '
                f"{_period_str(tku_a)}) vs "
                f'"{tku_b.content}" ({tku_b.source_title}, '
                f"{_period_str(tku_b)})."
            )
            _LOGGER.warning(
                "LLM synthesis failed for conflict %s — using template.",
                conflict.conflict_id,
            )

        return conflict.model_copy(
            update={
                "resolution_status": ResolutionStatus.CONTEXTUAL_SYNTHESIS,
                "synthesis_text": synthesis,
            }
        )
