"""Offline tests for ``TemporalPaperFlow`` (Phase 6).

All LLM and network calls are mocked.  Tests verify the extraction
pipeline, temporal confidence assignment, deduplication, and config
propagation.
"""

import json
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from finsage.configs.paper import ArxivIdentifier, RawText
from finsage.configs.temporal import TemporalPaperCfg
from finsage.flows.temporal_paper import (
    TemporalPaperFlow,
    _build_tkus_from_parsed,
    _chunk_text,
    _detect_domain,
)
from finsage.knowledge.temporal import (
    FinanceDomain,
    KnowledgeType,
    TKUCollection,
)
from finsage.preprocess.time import DateRange

#  Fixtures ─

_SAMPLE_LLM_RESPONSE = json.dumps(
    [
        {
            "content": "Momentum factor earns 1.2% monthly premium in US equities.",
            "content_type": "STATISTIC",
            "temporal_reference": "1963-2022",
            "market": "US Equities",
            "extraction_confidence": 0.92,
        },
        {
            "content": "The Fama-French three-factor model explains 90% of cross-sectional return variation.",
            "content_type": "CLAIM",
            "temporal_reference": None,
            "market": None,
            "extraction_confidence": 0.85,
        },
        {
            "content": "Value premium is defined as the return differential between high and low book-to-market portfolios.",
            "content_type": "DEFINITION",
            "temporal_reference": None,
            "market": None,
            "extraction_confidence": 0.95,
        },
    ]
)

_PAPER_TEXT = (
    "Abstract: We study the momentum factor premium in US equities "
    "from January 1963 to December 2022. Our analysis shows the Fama-French "
    "three-factor model explains 90% of cross-sectional return variation. "
    "We use machine learning techniques including gradient boosting and "
    "neural networks to predict factor returns. The value premium is defined "
    "as the return differential between high and low book-to-market portfolios."
)

_LONG_SECTION = "A" * 200
_SHORT_SECTION = "Too short"


#  Test: TemporalPaperCfg ─


class TestTemporalPaperCfg(unittest.TestCase):
    """Configuration creation and defaults."""

    def test_default_cfg(self) -> None:
        cfg = TemporalPaperCfg()
        self.assertEqual(cfg.model, "gpt-4o-mini")
        self.assertEqual(cfg.local_model, "llama3.2")
        self.assertTrue(cfg.use_local_if_available)
        self.assertEqual(cfg.max_tkus_per_section, 20)
        self.assertAlmostEqual(cfg.semantic_similarity_threshold, 0.92)
        self.assertEqual(cfg.min_section_length, 150)
        self.assertEqual(cfg.chunk_chars, 1800)

    def test_custom_cfg(self) -> None:
        cfg = TemporalPaperCfg(
            model="gpt-4o",
            local_model="llama3.1",
            max_tkus_per_section=10,
            chunk_chars=2000,
        )
        self.assertEqual(cfg.model, "gpt-4o")
        self.assertEqual(cfg.local_model, "llama3.1")
        self.assertEqual(cfg.max_tkus_per_section, 10)
        self.assertEqual(cfg.chunk_chars, 2000)

    def test_cfg_extra_forbid(self) -> None:
        """Extra fields are rejected (inherited from BaseFlowCfg)."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            TemporalPaperCfg(bogus_field="x")  # type: ignore[call-arg]


#  Test: chunk_text helper


class TestChunkText(unittest.TestCase):
    """Internal text chunking helper."""

    def test_short_text_single_chunk(self) -> None:
        result = _chunk_text("Hello world.", 100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Hello world.")

    def test_long_text_multiple_chunks(self) -> None:
        # Build text with sentence boundaries.
        sentences = ["Sentence number %d. " % i for i in range(100)]
        text = "".join(sentences)
        result = _chunk_text(text, 400)
        self.assertGreater(len(result), 1)
        # All text should be covered.
        self.assertEqual("".join(result), text)

    def test_empty_text(self) -> None:
        result = _chunk_text("", 100)
        self.assertEqual(result, [""])


#  Test: domain detection ─


class TestDomainDetection(unittest.TestCase):
    """Heuristic domain detection."""

    def test_factor_investing(self) -> None:
        text = "The momentum factor and Fama-French alpha are significant."
        self.assertEqual(_detect_domain(text), FinanceDomain.FACTOR_INVESTING)

    def test_ml_finance(self) -> None:
        text = "We use machine learning with deep learning and neural network approaches."
        self.assertEqual(_detect_domain(text), FinanceDomain.ML_FINANCE)

    def test_no_domain(self) -> None:
        text = "This is a paper about butterflies."
        self.assertIsNone(_detect_domain(text))


#  Test: _build_tkus_from_parsed


class TestBuildTkusFromParsed(unittest.TestCase):
    """Unit tests for the LLM response → TKU conversion."""

    def test_basic_parsing(self) -> None:
        items = json.loads(_SAMPLE_LLM_RESPONSE)
        tkus = _build_tkus_from_parsed(
            items,
            source_id="paper-001",
            source_title="Test Paper",
            sample_period=None,
            domain=FinanceDomain.FACTOR_INVESTING,
            max_per_section=20,
        )
        self.assertEqual(len(tkus), 3)
        self.assertEqual(tkus[0].content_type, KnowledgeType.STATISTIC)
        self.assertEqual(tkus[0].source_id, "paper-001")
        self.assertEqual(tkus[0].domain, FinanceDomain.FACTOR_INVESTING)

    def test_explicit_temporal_reference_high_confidence(self) -> None:
        """TKUs with explicit temporal_reference get ≥0.85 confidence."""
        items = [
            {
                "content": "Factor X earns 2% during 1990-2020.",
                "content_type": "STATISTIC",
                "temporal_reference": "1990-2020",
                "market": "US Equities",
                "extraction_confidence": 0.9,
            }
        ]
        tkus = _build_tkus_from_parsed(
            items,
            source_id="p1",
            source_title="Paper 1",
            sample_period=None,
            domain=None,
            max_per_section=20,
        )
        self.assertEqual(len(tkus), 1)
        self.assertGreaterEqual(tkus[0].temporal_confidence, 0.85)
        self.assertIsNotNone(tkus[0].valid_from)
        self.assertIsNotNone(tkus[0].valid_until)

    def test_no_temporal_reference_inherits_sample_period(self) -> None:
        """TKUs without temporal_reference inherit document-level period."""
        sample = DateRange(
            start=datetime(1990, 1, 1, tzinfo=timezone.utc),
            end=datetime(2020, 12, 31, tzinfo=timezone.utc),
            confidence=0.90,
            raw_text="1990-2020",
        )
        items = [
            {
                "content": "Markets are efficient.",
                "content_type": "CLAIM",
                "temporal_reference": None,
                "market": None,
                "extraction_confidence": 0.8,
            }
        ]
        tkus = _build_tkus_from_parsed(
            items,
            source_id="p2",
            source_title="Paper 2",
            sample_period=sample,
            domain=None,
            max_per_section=20,
        )
        self.assertEqual(len(tkus), 1)
        self.assertLessEqual(tkus[0].temporal_confidence, 0.55)
        self.assertIsNotNone(tkus[0].valid_from)

    def test_max_per_section_cap(self) -> None:
        """Respects max_per_section limit."""
        items = [
            {
                "content": f"Fact {i}.",
                "content_type": "FACT",
                "temporal_reference": None,
                "market": None,
                "extraction_confidence": 0.8,
            }
            for i in range(50)
        ]
        tkus = _build_tkus_from_parsed(
            items,
            source_id="p3",
            source_title="Paper 3",
            sample_period=None,
            domain=None,
            max_per_section=5,
        )
        self.assertEqual(len(tkus), 5)

    def test_invalid_items_skipped(self) -> None:
        """Non-dict and empty-content items are skipped."""
        items = [
            "not a dict",
            {"content": "", "content_type": "FACT"},
            {"no_content_key": True},
        ]
        tkus = _build_tkus_from_parsed(
            items,
            source_id="p4",
            source_title="Paper 4",
            sample_period=None,
            domain=None,
            max_per_section=20,
        )
        self.assertEqual(len(tkus), 0)


#  Test: TemporalPaperFlow ─


def _mock_parsed_document() -> MagicMock:
    """Create a mock ParsedDocument with pages."""
    page1 = MagicMock()
    page1.text = _PAPER_TEXT
    page2 = MagicMock()
    page2.text = _LONG_SECTION
    doc = MagicMock()
    doc.pages = [page1, page2]
    return doc


def _mock_raw_paper() -> MagicMock:
    """Create a mock RawPaper for fetch_arxiv."""
    raw = MagicMock()
    raw.title = "Test: Momentum Factor Analysis"
    raw.arxiv_id = "2509.21507"
    raw.content_type = "application/pdf"
    raw.bytes = b"%PDF-1.4 fake"
    return raw


class TestTemporalPaperFlowConstruction(unittest.IsolatedAsyncioTestCase):
    """Flow construction and config binding."""

    def test_creates_with_default_cfg(self) -> None:
        cfg = TemporalPaperCfg()
        flow = TemporalPaperFlow(cfg)
        self.assertIsNotNone(flow.cfg)
        self.assertEqual(flow.cfg.model, "gpt-4o-mini")

    def test_cfg_is_copied(self) -> None:
        """The flow stores a deep copy — mutating original has no effect."""
        cfg = TemporalPaperCfg()
        flow = TemporalPaperFlow(cfg)
        # Pydantic v2 frozen model can't be mutated, but verify identity.
        self.assertIsNot(flow.cfg, cfg)

    async def test_rejects_raw_text(self) -> None:
        """RawText input raises ValueError."""
        cfg = TemporalPaperCfg()
        flow = TemporalPaperFlow(cfg)
        with self.assertRaises(ValueError):
            await flow.build(RawText(text="hello"))


class TestTemporalPaperFlowBuild(unittest.IsolatedAsyncioTestCase):
    """Full build pipeline with mocked LLM and network calls."""

    @patch("finsage.flows.temporal_paper.fetch_arxiv", new_callable=AsyncMock)
    @patch("finsage.flows.temporal_paper.parse_pdf", new_callable=AsyncMock)
    async def test_build_returns_tku_collection(
        self, mock_parse: AsyncMock, mock_fetch: AsyncMock
    ) -> None:
        """build() returns a TKUCollection with correct metadata."""
        mock_fetch.return_value = _mock_raw_paper()
        mock_parse.return_value = _mock_parsed_document()

        # Mock the router's generate and embed methods.
        cfg = TemporalPaperCfg()
        flow = TemporalPaperFlow(cfg)

        async def mock_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
            response = MagicMock()
            response.text = _SAMPLE_LLM_RESPONSE
            return response

        async def mock_embed(text: str) -> list[float]:
            # Return deterministic but distinct embeddings per content hash.
            seed = hash(text) % 100
            return [float(seed + i) for i in range(10)]

        flow._router.generate = mock_generate  # type: ignore[assignment]
        flow._router.embed = mock_embed  # type: ignore[assignment]

        result = await flow.build(ArxivIdentifier(id="2509.21507"))

        self.assertIsInstance(result, TKUCollection)
        self.assertEqual(result.source_title, "Test: Momentum Factor Analysis")
        self.assertEqual(result.document_id, "2509.21507")
        self.assertGreater(len(result.tkus), 0)

    @patch("finsage.flows.temporal_paper.fetch_arxiv", new_callable=AsyncMock)
    @patch("finsage.flows.temporal_paper.parse_pdf", new_callable=AsyncMock)
    async def test_source_metadata_populated(
        self, mock_parse: AsyncMock, mock_fetch: AsyncMock
    ) -> None:
        """All TKUs carry source_id and source_title."""
        mock_fetch.return_value = _mock_raw_paper()
        mock_parse.return_value = _mock_parsed_document()

        cfg = TemporalPaperCfg()
        flow = TemporalPaperFlow(cfg)

        async def mock_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
            response = MagicMock()
            response.text = _SAMPLE_LLM_RESPONSE
            return response

        async def mock_embed(text: str) -> list[float]:
            seed = hash(text) % 100
            return [float(seed + i) for i in range(10)]

        flow._router.generate = mock_generate  # type: ignore[assignment]
        flow._router.embed = mock_embed  # type: ignore[assignment]

        result = await flow.build(ArxivIdentifier(id="2509.21507"))

        for tku in result.tkus:
            self.assertEqual(tku.source_id, "2509.21507")
            self.assertEqual(tku.source_title, "Test: Momentum Factor Analysis")
            self.assertGreater(len(tku.content), 0)

    @patch("finsage.flows.temporal_paper.fetch_arxiv", new_callable=AsyncMock)
    @patch("finsage.flows.temporal_paper.parse_pdf", new_callable=AsyncMock)
    async def test_short_sections_skipped(
        self, mock_parse: AsyncMock, mock_fetch: AsyncMock
    ) -> None:
        """Sections shorter than min_section_length are skipped."""
        mock_fetch.return_value = _mock_raw_paper()
        # Only short sections.
        page = MagicMock()
        page.text = _SHORT_SECTION
        doc = MagicMock()
        doc.pages = [page]
        mock_parse.return_value = doc

        cfg = TemporalPaperCfg(min_section_length=100)
        flow = TemporalPaperFlow(cfg)

        generate_calls: list[object] = []

        async def mock_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
            generate_calls.append(args)
            response = MagicMock()
            response.text = "[]"
            return response

        flow._router.generate = mock_generate  # type: ignore[assignment]
        flow._router.embed = AsyncMock(return_value=[0.1] * 10)  # type: ignore[assignment]

        result = await flow.build(ArxivIdentifier(id="2509.21507"))

        # No LLM calls should have been made — section too short.
        self.assertEqual(len(generate_calls), 0)
        self.assertEqual(len(result.tkus), 0)

    @patch("finsage.flows.temporal_paper.fetch_arxiv", new_callable=AsyncMock)
    @patch("finsage.flows.temporal_paper.parse_pdf", new_callable=AsyncMock)
    async def test_deduplication_removes_near_identical(
        self, mock_parse: AsyncMock, mock_fetch: AsyncMock
    ) -> None:
        """Near-identical TKUs from same source are deduplicated."""
        mock_fetch.return_value = _mock_raw_paper()
        mock_parse.return_value = _mock_parsed_document()

        # Return duplicate items.
        duplicate_response = json.dumps(
            [
                {
                    "content": "Momentum premium is 1.2% monthly.",
                    "content_type": "STATISTIC",
                    "temporal_reference": "1963-2022",
                    "market": "US Equities",
                    "extraction_confidence": 0.9,
                },
                {
                    "content": "Monthly momentum premium equals 1.2%.",
                    "content_type": "STATISTIC",
                    "temporal_reference": "1963-2022",
                    "market": "US Equities",
                    "extraction_confidence": 0.85,
                },
            ]
        )

        cfg = TemporalPaperCfg(semantic_similarity_threshold=0.8)
        flow = TemporalPaperFlow(cfg)

        async def mock_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
            response = MagicMock()
            response.text = duplicate_response
            return response

        # Return identical embeddings → similarity = 1.0 → deduplicated.
        async def mock_embed(text: str) -> list[float]:
            return [1.0, 0.0, 0.0, 0.0, 0.0]

        flow._router.generate = mock_generate  # type: ignore[assignment]
        flow._router.embed = mock_embed  # type: ignore[assignment]

        result = await flow.build(ArxivIdentifier(id="2509.21507"))

        # With identical embeddings and threshold 0.8, duplicates are removed.
        # Each section produces 2 items but they share identical embeddings,
        # so only 1 survives per section.  The two sections each produce one,
        # but since the content is identical across sections too, overall
        # the count is less than 2 * num_sections.
        unique_contents = {tku.content for tku in result.tkus}
        # We should have fewer TKUs than the raw total.
        self.assertLessEqual(len(result.tkus), len(unique_contents) + 1)

    @patch("finsage.flows.temporal_paper.fetch_arxiv", new_callable=AsyncMock)
    @patch("finsage.flows.temporal_paper.parse_pdf", new_callable=AsyncMock)
    async def test_llm_failure_handled_gracefully(
        self, mock_parse: AsyncMock, mock_fetch: AsyncMock
    ) -> None:
        """LLM failures produce an empty collection, not a crash."""
        mock_fetch.return_value = _mock_raw_paper()
        mock_parse.return_value = _mock_parsed_document()

        cfg = TemporalPaperCfg()
        flow = TemporalPaperFlow(cfg)

        async def mock_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("all LLM tiers failed for extraction")

        flow._router.generate = mock_generate  # type: ignore[assignment]
        flow._router.embed = AsyncMock(return_value=[0.1] * 10)  # type: ignore[assignment]

        result = await flow.build(ArxivIdentifier(id="2509.21507"))

        self.assertIsInstance(result, TKUCollection)
        self.assertEqual(len(result.tkus), 0)


class TestTemporalPaperFlowConfig(unittest.TestCase):
    """Config override propagation."""

    def test_custom_model_propagates_to_router(self) -> None:
        cfg = TemporalPaperCfg(model="gpt-4o")
        flow = TemporalPaperFlow(cfg)
        self.assertEqual(flow.router.cfg.api_cheap_model, "gpt-4o")

    def test_local_disabled_changes_routing(self) -> None:
        cfg = TemporalPaperCfg(use_local_if_available=False)
        flow = TemporalPaperFlow(cfg)
        # All tiers should be api_cheap when local is disabled.
        for tier_value in flow.router.cfg.default_routing.values():
            self.assertEqual(tier_value, "api_cheap")


if __name__ == "__main__":
    unittest.main()
