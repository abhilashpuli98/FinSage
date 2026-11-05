"""Tests for mind.conflict (C2: CSCG detection and resolution)."""

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from finsage.configs.ollama import (
    OllamaModelCfg,
    RouterCfg,
)
from finsage.knowledge.conflict import (
    ConflictRecord,
    ConflictType,
    ResolutionStatus,
)
from finsage.knowledge.temporal import (
    FinanceDomain,
    TemporalKnowledgeUnit,
    TKUCollection,
)
from finsage.mind.conflict import (
    ConflictDetector,
    ConflictResolver,
    _temporal_overlap,
    compute_cosine_similarity,
)
from finsage.mind.router import LLMRouter


def _utc(y: int, m: int = 1, d: int = 1) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _make_tku(
    content: str = "Momentum premium is 1.2% monthly",
    source_id: str = "paper-001",
    source_title: str = "Paper A",
    **overrides,
) -> TemporalKnowledgeUnit:
    """Factory for test TKUs with sensible defaults."""
    defaults = {
        "content": content,
        "source_id": source_id,
        "source_title": source_title,
        "domain": FinanceDomain.FACTOR_INVESTING,
        "valid_from": _utc(1990),
        "valid_until": _utc(2020),
        "temporal_confidence": 0.9,
        "extraction_confidence": 0.85,
        "market": "US Equities",
    }
    defaults.update(overrides)
    return TemporalKnowledgeUnit(**defaults)


def _make_collection(
    tkus: list[TemporalKnowledgeUnit],
    document_id: str = "doc-001",
    source_title: str = "Paper A",
) -> TKUCollection:
    """Wrap TKUs into a collection."""
    return TKUCollection(
        document_id=document_id,
        source_title=source_title,
        tkus=tuple(tkus),
    )


def _completion(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build the LiteLLM-shaped object needed by the router boundary."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _make_router() -> LLMRouter:
    """Create a router with default config for testing."""
    return LLMRouter(RouterCfg(ollama=OllamaModelCfg(max_retries=1)))


#  Cosine similarity tests ─


class CosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors(self) -> None:
        self.assertAlmostEqual(
            compute_cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0
        )

    def test_orthogonal_vectors(self) -> None:
        self.assertAlmostEqual(
            compute_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0
        )

    def test_opposite_vectors(self) -> None:
        self.assertAlmostEqual(
            compute_cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0
        )

    def test_zero_vector_returns_zero(self) -> None:
        self.assertAlmostEqual(
            compute_cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0
        )

    def test_high_dimensional(self) -> None:
        a = [0.5] * 384
        b = [0.5] * 384
        self.assertAlmostEqual(compute_cosine_similarity(a, b), 1.0)


#  Temporal overlap tests ─


class TemporalOverlapTests(unittest.TestCase):
    def test_overlapping_windows(self) -> None:
        a = _make_tku(valid_from=_utc(1990), valid_until=_utc(2010))
        b = _make_tku(valid_from=_utc(2005), valid_until=_utc(2020))
        result = _temporal_overlap(a, b)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], _utc(2005))
        self.assertEqual(result[1], _utc(2010))

    def test_disjoint_windows(self) -> None:
        a = _make_tku(valid_from=_utc(1990), valid_until=_utc(2000))
        b = _make_tku(valid_from=_utc(2005), valid_until=_utc(2020))
        result = _temporal_overlap(a, b)
        self.assertIsNone(result)

    def test_open_ended_a(self) -> None:
        a = _make_tku(valid_from=_utc(1990), valid_until=None)
        b = _make_tku(valid_from=_utc(2005), valid_until=_utc(2020))
        result = _temporal_overlap(a, b)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], _utc(2005))
        self.assertEqual(result[1], _utc(2020))

    def test_both_fully_unbounded(self) -> None:
        a = _make_tku(valid_from=None, valid_until=None)
        b = _make_tku(valid_from=None, valid_until=None)
        result = _temporal_overlap(a, b)
        self.assertIsNotNone(result)

    def test_touching_but_not_overlapping(self) -> None:
        a = _make_tku(valid_from=_utc(1990), valid_until=_utc(2000))
        b = _make_tku(valid_from=_utc(2000), valid_until=_utc(2020))
        result = _temporal_overlap(a, b)
        self.assertIsNone(result)


#  ConflictDetector tests ─


class DetectorPrefilterTests(unittest.IsolatedAsyncioTestCase):
    async def test_skips_same_source_pairs(self) -> None:
        tku_a = _make_tku(source_id="paper-001", content="Claim A")
        tku_b = _make_tku(source_id="paper-001", content="Claim B")
        coll = _make_collection([tku_a, tku_b])

        router = _make_router()
        detector = ConflictDetector(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=AsyncMock()),
            patch("finsage.mind.router.aembedding", new=AsyncMock()),
        ):
            conflicts = await detector.detect([coll])

        self.assertEqual(len(conflicts), 0)

    async def test_skips_different_domain_pairs(self) -> None:
        tku_a = _make_tku(
            source_id="paper-001",
            content="Claim A about factors",
            domain=FinanceDomain.FACTOR_INVESTING,
        )
        tku_b = _make_tku(
            source_id="paper-002",
            content="Claim B about credit",
            domain=FinanceDomain.CREDIT_RISK,
        )
        coll_a = _make_collection([tku_a], document_id="doc-001")
        coll_b = _make_collection(
            [tku_b], document_id="doc-002", source_title="Paper B"
        )

        router = _make_router()
        detector = ConflictDetector(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=AsyncMock()),
            patch("finsage.mind.router.aembedding", new=AsyncMock()),
        ):
            conflicts = await detector.detect([coll_a, coll_b])

        self.assertEqual(len(conflicts), 0)

    async def test_skips_temporally_disjoint_pairs(self) -> None:
        tku_a = _make_tku(
            source_id="paper-001",
            content="Claim A",
            valid_from=_utc(1990),
            valid_until=_utc(2000),
        )
        tku_b = _make_tku(
            source_id="paper-002",
            content="Claim B",
            valid_from=_utc(2005),
            valid_until=_utc(2020),
        )
        coll_a = _make_collection([tku_a], document_id="doc-001")
        coll_b = _make_collection(
            [tku_b], document_id="doc-002", source_title="Paper B"
        )

        router = _make_router()
        detector = ConflictDetector(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=AsyncMock()),
            patch("finsage.mind.router.aembedding", new=AsyncMock()),
        ):
            conflicts = await detector.detect([coll_a, coll_b])

        self.assertEqual(len(conflicts), 0)


class DetectorLLMConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirms_conflict_from_llm_response(self) -> None:
        """Full pipeline: embedding similarity > threshold → LLM confirms."""
        tku_a = _make_tku(
            source_id="paper-001",
            source_title="Jegadeesh & Titman (1993)",
            content="Momentum premium is 1.2% monthly",
            valid_from=_utc(1965),
            valid_until=_utc(1990),
        )
        tku_b = _make_tku(
            source_id="paper-002",
            source_title="Schwert (2003)",
            content="Momentum premium is statistically insignificant",
            valid_from=_utc(1980),
            valid_until=_utc(2002),
        )
        coll_a = _make_collection(
            [tku_a], document_id="doc-001", source_title="Paper A"
        )
        coll_b = _make_collection(
            [tku_b], document_id="doc-002", source_title="Paper B"
        )

        # Mock embeddings: similar vectors so they pass threshold.
        embed_mock = AsyncMock(
            return_value={"data": [{"embedding": [0.9, 0.1, 0.05]}]}
        )
        # Mock LLM confirmation response.
        llm_response = _completion(
            '{"same_claim": true, "agreement": "disagree", '
            '"conflict_type": "SIGNIFICANCE_MISMATCH", '
            '"detection_confidence": 0.85, '
            '"reason": "momentum factor significance"}'
        )
        generate_mock = AsyncMock(return_value=llm_response)

        router = _make_router()

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=generate_mock),
            patch("finsage.mind.router.aembedding", new=embed_mock),
        ):
            detector = ConflictDetector(router, semantic_threshold=0.5)
            conflicts = await detector.detect([coll_a, coll_b])

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(
            conflicts[0].conflict_type, ConflictType.SIGNIFICANCE_MISMATCH
        )
        self.assertEqual(conflicts[0].tku_a_id, tku_a.tku_id)
        self.assertEqual(conflicts[0].tku_b_id, tku_b.tku_id)
        self.assertAlmostEqual(conflicts[0].detection_confidence, 0.85)

    async def test_no_conflict_when_llm_says_agree(self) -> None:
        tku_a = _make_tku(
            source_id="paper-001",
            content="Momentum factor works",
            valid_from=_utc(1990),
            valid_until=_utc(2020),
        )
        tku_b = _make_tku(
            source_id="paper-002",
            content="Momentum factor also works",
            valid_from=_utc(1990),
            valid_until=_utc(2020),
        )
        coll_a = _make_collection([tku_a], document_id="doc-001")
        coll_b = _make_collection(
            [tku_b], document_id="doc-002", source_title="Paper B"
        )

        embed_mock = AsyncMock(
            return_value={"data": [{"embedding": [0.9, 0.1]}]}
        )
        llm_response = _completion(
            '{"same_claim": true, "agreement": "agree", '
            '"conflict_type": null, "detection_confidence": 0.9, '
            '"reason": "They agree"}'
        )
        generate_mock = AsyncMock(return_value=llm_response)

        router = _make_router()

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=generate_mock),
            patch("finsage.mind.router.aembedding", new=embed_mock),
        ):
            detector = ConflictDetector(router, semantic_threshold=0.5)
            conflicts = await detector.detect([coll_a, coll_b])

        self.assertEqual(len(conflicts), 0)

    async def test_below_similarity_threshold_skips_llm(self) -> None:
        """When embedding similarity is below threshold, no LLM call."""
        tku_a = _make_tku(
            source_id="paper-001",
            content="Momentum premium is 1.2%",
            valid_from=_utc(1990),
            valid_until=_utc(2020),
        )
        tku_b = _make_tku(
            source_id="paper-002",
            content="Credit risk spreads widen",
            domain=FinanceDomain.FACTOR_INVESTING,
            valid_from=_utc(1990),
            valid_until=_utc(2020),
        )
        coll_a = _make_collection([tku_a], document_id="doc-001")
        coll_b = _make_collection(
            [tku_b], document_id="doc-002", source_title="Paper B"
        )

        # Return dissimilar embeddings.
        call_count = 0

        async def embed_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {"data": [{"embedding": [1.0, 0.0, 0.0]}]}
            return {"data": [{"embedding": [0.0, 0.0, 1.0]}]}

        embed_mock = AsyncMock(side_effect=embed_side_effect)
        generate_mock = AsyncMock()

        router = _make_router()

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=generate_mock),
            patch("finsage.mind.router.aembedding", new=embed_mock),
        ):
            detector = ConflictDetector(router, semantic_threshold=0.82)
            conflicts = await detector.detect([coll_a, coll_b])

        self.assertEqual(len(conflicts), 0)
        # acompletion should NOT have been called for conflict confirmation.
        generate_mock.assert_not_awaited()

    async def test_empty_collections_return_empty(self) -> None:
        router = _make_router()
        detector = ConflictDetector(router)
        conflicts = await detector.detect([])
        self.assertEqual(len(conflicts), 0)

    async def test_single_tku_returns_empty(self) -> None:
        tku = _make_tku()
        coll = _make_collection([tku])
        router = _make_router()
        detector = ConflictDetector(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=AsyncMock()),
            patch("finsage.mind.router.aembedding", new=AsyncMock()),
        ):
            conflicts = await detector.detect([coll])

        self.assertEqual(len(conflicts), 0)


#  ConflictResolver tests


def _make_conflict_record(**overrides) -> ConflictRecord:
    """Factory with sensible defaults for testing."""
    defaults = {
        "tku_a_id": "tku-001",
        "tku_b_id": "tku-002",
        "tku_a_content": "Momentum premium is 1.2% monthly",
        "tku_b_content": "Momentum premium is statistically insignificant",
        "tku_a_source": "Jegadeesh & Titman (1993)",
        "tku_b_source": "Schwert (2003)",
        "tku_a_valid_from": _utc(1965),
        "tku_a_valid_until": _utc(1990),
        "tku_b_valid_from": _utc(1980),
        "tku_b_valid_until": _utc(2002),
        "conflict_type": ConflictType.SIGNIFICANCE_MISMATCH,
        "entity_name": "momentum factor",
        "temporal_overlap_start": _utc(1980),
        "temporal_overlap_end": _utc(1990),
        "detection_confidence": 0.85,
    }
    defaults.update(overrides)
    return ConflictRecord(**defaults)


class ResolverMarketScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_by_market_when_markets_differ(self) -> None:
        tku_a = _make_tku(
            tku_id="tku-001",
            source_id="paper-001",
            market="US Equities",
        )
        tku_b = _make_tku(
            tku_id="tku-002",
            source_id="paper-002",
            market="China A-Share",
        )
        conflict = _make_conflict_record(
            tku_a_id=tku_a.tku_id, tku_b_id=tku_b.tku_id
        )
        tku_index = {tku_a.tku_id: tku_a, tku_b.tku_id: tku_b}

        router = _make_router()
        resolver = ConflictResolver(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=AsyncMock()),
        ):
            results = await resolver.resolve([conflict], tku_index)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].resolution_status, ResolutionStatus.RESOLVED_BY_MARKET
        )
        self.assertIn("US Equities", results[0].resolution_explanation or "")
        self.assertIn("China A-Share", results[0].resolution_explanation or "")

    async def test_does_not_resolve_by_market_when_same(self) -> None:
        tku_a = _make_tku(
            tku_id="tku-001", source_id="paper-001", market="US Equities"
        )
        tku_b = _make_tku(
            tku_id="tku-002", source_id="paper-002", market="US Equities"
        )
        conflict = _make_conflict_record(
            tku_a_id=tku_a.tku_id, tku_b_id=tku_b.tku_id
        )
        tku_index = {tku_a.tku_id: tku_a, tku_b.tku_id: tku_b}

        # LLM mocks for later strategies.
        methodology_response = _completion(
            '{"methodology_explains": false, "explanation": "", "confidence": 0.0}'
        )
        synthesis_response = _completion(
            "Research is mixed on momentum factor."
        )
        generate_mock = AsyncMock(
            side_effect=[methodology_response, synthesis_response]
        )

        router = _make_router()
        resolver = ConflictResolver(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=generate_mock),
        ):
            results = await resolver.resolve([conflict], tku_index)

        self.assertEqual(len(results), 1)
        # Should NOT be resolved by market.
        self.assertNotEqual(
            results[0].resolution_status, ResolutionStatus.RESOLVED_BY_MARKET
        )


class ResolverPeriodTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_by_period_when_disjoint(self) -> None:
        tku_a = _make_tku(
            tku_id="tku-001",
            source_id="paper-001",
            market="US Equities",
            valid_from=_utc(1990),
            valid_until=_utc(2000),
        )
        tku_b = _make_tku(
            tku_id="tku-002",
            source_id="paper-002",
            market="US Equities",
            valid_from=_utc(2005),
            valid_until=_utc(2020),
        )
        conflict = _make_conflict_record(
            tku_a_id=tku_a.tku_id, tku_b_id=tku_b.tku_id
        )
        tku_index = {tku_a.tku_id: tku_a, tku_b.tku_id: tku_b}

        router = _make_router()
        resolver = ConflictResolver(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=AsyncMock()),
        ):
            results = await resolver.resolve([conflict], tku_index)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].resolution_status, ResolutionStatus.RESOLVED_BY_PERIOD
        )


class ResolverMethodologyTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolves_by_methodology_when_llm_confirms(self) -> None:
        tku_a = _make_tku(
            tku_id="tku-001",
            source_id="paper-001",
            market="US Equities",
            tags=("OLS", "monthly"),
        )
        tku_b = _make_tku(
            tku_id="tku-002",
            source_id="paper-002",
            market="US Equities",
            tags=("Fama-MacBeth", "daily"),
        )
        conflict = _make_conflict_record(
            tku_a_id=tku_a.tku_id, tku_b_id=tku_b.tku_id
        )
        tku_index = {tku_a.tku_id: tku_a, tku_b.tku_id: tku_b}

        methodology_response = _completion(
            '{"methodology_explains": true, '
            '"explanation": "OLS vs Fama-MacBeth produce different results", '
            '"confidence": 0.9}'
        )
        generate_mock = AsyncMock(return_value=methodology_response)

        router = _make_router()
        resolver = ConflictResolver(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=generate_mock),
        ):
            results = await resolver.resolve([conflict], tku_index)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].resolution_status,
            ResolutionStatus.RESOLVED_BY_METHODOLOGY,
        )
        self.assertIn("Fama-MacBeth", results[0].resolution_explanation or "")


class ResolverSynthesisTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_when_no_other_strategy_works(self) -> None:
        tku_a = _make_tku(
            tku_id="tku-001",
            source_id="paper-001",
            market="US Equities",
        )
        tku_b = _make_tku(
            tku_id="tku-002",
            source_id="paper-002",
            market="US Equities",
        )
        conflict = _make_conflict_record(
            tku_a_id=tku_a.tku_id, tku_b_id=tku_b.tku_id
        )
        tku_index = {tku_a.tku_id: tku_a, tku_b.tku_id: tku_b}

        # Methodology says no.
        methodology_response = _completion(
            '{"methodology_explains": false, '
            '"explanation": "same methods", "confidence": 0.2}'
        )
        synthesis_response = _completion(
            "Research is mixed on momentum factor: Jegadeesh & Titman find "
            "a 1.2% premium while Schwert finds it insignificant."
        )
        generate_mock = AsyncMock(
            side_effect=[methodology_response, synthesis_response]
        )

        router = _make_router()
        resolver = ConflictResolver(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=generate_mock),
        ):
            results = await resolver.resolve([conflict], tku_index)

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0].resolution_status,
            ResolutionStatus.CONTEXTUAL_SYNTHESIS,
        )
        self.assertIn("mixed", results[0].synthesis_text or "")


class ResolverFrozenModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_creates_new_instance(self) -> None:
        """Resolution does not mutate the original ConflictRecord."""
        tku_a = _make_tku(
            tku_id="tku-001",
            source_id="paper-001",
            market="US Equities",
        )
        tku_b = _make_tku(
            tku_id="tku-002",
            source_id="paper-002",
            market="China A-Share",
        )
        conflict = _make_conflict_record(
            tku_a_id=tku_a.tku_id, tku_b_id=tku_b.tku_id
        )
        tku_index = {tku_a.tku_id: tku_a, tku_b.tku_id: tku_b}

        router = _make_router()
        resolver = ConflictResolver(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=AsyncMock()),
        ):
            results = await resolver.resolve([conflict], tku_index)

        # Original is still unresolved.
        self.assertFalse(conflict.is_resolved())
        # Resolved copy has a different status.
        self.assertTrue(results[0].is_resolved())
        # They share the same IDs.
        self.assertEqual(results[0].tku_a_id, conflict.tku_a_id)
        self.assertEqual(results[0].tku_b_id, conflict.tku_b_id)

    async def test_missing_tku_in_index_kept_unresolved(self) -> None:
        conflict = _make_conflict_record()
        tku_index: dict[str, TemporalKnowledgeUnit] = {}

        router = _make_router()
        resolver = ConflictResolver(router)

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=AsyncMock()),
        ):
            results = await resolver.resolve([conflict], tku_index)

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].is_resolved())


#  Export tests ─


class ExportTests(unittest.TestCase):
    def test_module_exports_available(self) -> None:
        from finsage.mind import (
            ConflictDetector,
            ConflictResolver,
            compute_cosine_similarity,
        )

        self.assertIsNotNone(ConflictDetector)
        self.assertIsNotNone(ConflictResolver)
        self.assertIsNotNone(compute_cosine_similarity)


if __name__ == "__main__":
    unittest.main()
