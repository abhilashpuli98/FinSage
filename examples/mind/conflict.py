"""Conflict detection and resolution example.

Demonstrates the C2 (CSCG) pipeline:

1. Build two ``TKUCollection``s with overlapping temporal windows.
2. Run ``ConflictDetector.detect`` to find contradictions.
3. Run ``ConflictResolver.resolve`` to attempt resolution.

This example prints the routing and resolution steps to stdout.
LLM calls are mocked so no API keys or Ollama server are required.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from finsage.configs.ollama import OllamaModelCfg, RouterCfg
from finsage.knowledge.temporal import (
    FinanceDomain,
    TemporalKnowledgeUnit,
    TKUCollection,
)
from finsage.mind.conflict import ConflictDetector, ConflictResolver
from finsage.mind.router import LLMRouter


def _utc(y: int) -> datetime:
    return datetime(y, 1, 1, tzinfo=timezone.utc)


def _make_collections() -> list[TKUCollection]:
    """Build two collections with a known conflict on momentum factor."""
    tku_a = TemporalKnowledgeUnit(
        content="Momentum factor earns 1.2% monthly premium (1963-1990)",
        source_id="paper-001",
        source_title="Jegadeesh & Titman (1993)",
        domain=FinanceDomain.FACTOR_INVESTING,
        valid_from=_utc(1963),
        valid_until=_utc(1990),
        market="US Equities",
        temporal_confidence=0.95,
        extraction_confidence=0.90,
    )
    tku_b = TemporalKnowledgeUnit(
        content="Momentum premium is statistically insignificant post-1990",
        source_id="paper-002",
        source_title="Schwert (2003)",
        domain=FinanceDomain.FACTOR_INVESTING,
        valid_from=_utc(1980),
        valid_until=_utc(2002),
        market="US Equities",
        temporal_confidence=0.88,
        extraction_confidence=0.85,
    )
    coll_a = TKUCollection(
        document_id="doc-001",
        source_title="Jegadeesh & Titman (1993)",
        tkus=(tku_a,),
    )
    coll_b = TKUCollection(
        document_id="doc-002",
        source_title="Schwert (2003)",
        tkus=(tku_b,),
    )
    return [coll_a, coll_b]


async def main() -> None:
    """Run the conflict detection and resolution pipeline with mocks."""
    collections = _make_collections()
    router = LLMRouter(RouterCfg(ollama=OllamaModelCfg(max_retries=1)))

    # Mock embedding to return similar vectors (above threshold).
    embed_response = {"data": [{"embedding": [0.9, 0.1, 0.05]}]}
    embed_mock = AsyncMock(return_value=embed_response)

    # Mock LLM conflict confirmation response.
    conflict_json = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"same_claim": true, "agreement": "disagree", '
                        '"conflict_type": "SIGNIFICANCE_MISMATCH", '
                        '"detection_confidence": 0.85, '
                        '"reason": "momentum factor significance"}'
                    )
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20),
    )

    # Mock LLM methodology check: says methodology does NOT explain.
    methodology_json = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"methodology_explains": false, '
                        '"explanation": "similar methods", '
                        '"confidence": 0.3}'
                    )
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=40, completion_tokens=15),
    )

    # Mock LLM synthesis.
    synthesis_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "Research is mixed on the momentum premium: "
                        "Jegadeesh & Titman (1993) find a 1.2% monthly "
                        "premium in 1963-1990, while Schwert (2003) reports "
                        "statistical insignificance in 1980-2002."
                    )
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=45, completion_tokens=30),
    )

    generate_mock = AsyncMock(
        side_effect=[conflict_json, methodology_json, synthesis_response]
    )

    with (
        patch.object(OllamaModelCfg, "is_available", return_value=False),
        patch("finsage.mind.router.acompletion", new=generate_mock),
        patch("finsage.mind.router.aembedding", new=embed_mock),
    ):
        # Step 1: Detect conflicts.
        detector = ConflictDetector(router, semantic_threshold=0.5)
        conflicts = await detector.detect(collections)

        print(f"Detected {len(conflicts)} conflict(s):")
        for c in conflicts:
            print(f"  - {c.conflict_type.value}: {c.entity_name}")
            print(f"    A: {c.tku_a_content}")
            print(f"    B: {c.tku_b_content}")
            print(f"    Confidence: {c.detection_confidence}")
            print()

        # Step 2: Resolve conflicts.
        all_tkus = [t for coll in collections for t in coll.tkus]
        tku_index = {t.tku_id: t for t in all_tkus}

        resolver = ConflictResolver(router)
        resolved = await resolver.resolve(conflicts, tku_index)

        print(f"Resolution results ({len(resolved)} conflict(s)):")
        for r in resolved:
            print(f"  - Status: {r.resolution_status.value}")
            print(f"    Note: {r.user_facing_note()}")
            print()

    # Cost report.
    report = router.cost_tracker.get_report()
    print("Cost report:")
    for task_type, stats in report.items():
        print(f"  {task_type}: {stats}")


if __name__ == "__main__":
    asyncio.run(main())
