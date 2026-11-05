"""Tests for knowledge.temporal (TKU data models — C1)."""

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from finsage.knowledge.temporal import (
    FinanceDomain,
    KnowledgeType,
    TemporalDecay,
    TemporalKnowledgeUnit,
    TKUCollection,
)


def _utc(y: int, m: int = 1, d: int = 1) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


class TemporalKnowledgeUnitTests(unittest.TestCase):
    def _make(self, **overrides):
        defaults = {
            "content": "Momentum premium is 1.2% monthly (1963-2022)",
            "source_id": "arxiv-2024-001",
            "source_title": "Factor Investing Survey",
        }
        defaults.update(overrides)
        return TemporalKnowledgeUnit(**defaults)

    def test_minimal_creation(self):
        tku = self._make()
        self.assertEqual(tku.content_type, KnowledgeType.FACT)
        self.assertAlmostEqual(tku.extraction_confidence, 0.8)
        self.assertAlmostEqual(tku.temporal_confidence, 0.5)
        self.assertIsNotNone(tku.tku_id)

    def test_content_hash_auto_computed(self):
        tku = self._make()
        self.assertTrue(len(tku.content_hash) == 16)
        self.assertRegex(tku.content_hash, r"^[0-9a-f]{16}$")

    def test_content_hash_deterministic(self):
        tku1 = self._make()
        tku2 = self._make()
        self.assertEqual(tku1.content_hash, tku2.content_hash)

    def test_content_hash_differs_for_different_content(self):
        tku1 = self._make(content="fact A")
        tku2 = self._make(content="fact B")
        self.assertNotEqual(tku1.content_hash, tku2.content_hash)

    def test_is_valid_at(self):
        tku = self._make(
            valid_from=_utc(2020, 1, 1),
            valid_until=_utc(2023, 1, 1),
        )
        self.assertTrue(tku.is_valid_at(_utc(2021, 6, 15)))
        self.assertFalse(tku.is_valid_at(_utc(2019, 12, 31)))
        self.assertFalse(tku.is_valid_at(_utc(2023, 1, 1)))  # exclusive

    def test_is_valid_at_open_ended(self):
        tku = self._make(valid_from=_utc(2020, 1, 1))
        self.assertTrue(tku.is_valid_at(_utc(2099, 1, 1)))
        self.assertFalse(tku.is_valid_at(_utc(2019, 1, 1)))

    def test_is_valid_at_unbounded(self):
        tku = self._make()
        self.assertTrue(tku.is_valid_at(_utc(1900, 1, 1)))

    def test_composite_confidence(self):
        tku = self._make(
            extraction_confidence=0.9,
            temporal_confidence=0.8,
        )
        self.assertAlmostEqual(tku.composite_confidence(), 0.72)

    def test_temporal_scope_full(self):
        tku = self._make(
            valid_from=_utc(1990),
            valid_until=_utc(2022),
            market="US Equities",
        )
        self.assertEqual(tku.temporal_scope(), "1990\u20132022 (US Equities)")

    def test_temporal_scope_open_ended(self):
        tku = self._make(valid_from=_utc(2010))
        self.assertEqual(tku.temporal_scope(), "2010\u2013present")

    def test_temporal_scope_unknown(self):
        tku = self._make()
        self.assertEqual(tku.temporal_scope(), "?\u2013present")

    def test_decay_status_current(self):
        tku = self._make(
            valid_from=_utc(2024, 1, 1),
            valid_until=_utc(2026, 1, 1),
        )
        self.assertEqual(
            tku.decay_status(_utc(2025, 6, 1)), TemporalDecay.CURRENT
        )

    def test_decay_status_expired(self):
        tku = self._make(
            valid_from=_utc(2020, 1, 1),
            valid_until=_utc(2022, 1, 1),
        )
        self.assertEqual(
            tku.decay_status(_utc(2025, 1, 1)), TemporalDecay.EXPIRED
        )

    def test_decay_status_undated(self):
        tku = self._make()
        self.assertEqual(
            tku.decay_status(_utc(2025, 1, 1)), TemporalDecay.UNDATED
        )

    def test_domain_enum(self):
        tku = self._make(domain=FinanceDomain.FACTOR_INVESTING)
        self.assertEqual(tku.domain, FinanceDomain.FACTOR_INVESTING)

    def test_knowledge_type_enum(self):
        tku = self._make(content_type=KnowledgeType.FORMULA)
        self.assertEqual(tku.content_type, KnowledgeType.FORMULA)

    def test_tags_tuple(self):
        tku = self._make(tags=("momentum", "factor"))
        self.assertEqual(tku.tags, ("momentum", "factor"))

    def test_page_references(self):
        tku = self._make(page_references=(1, 5, 12))
        self.assertEqual(tku.page_references, (1, 5, 12))

    def test_empty_content_rejected(self):
        with self.assertRaises(ValidationError):
            self._make(content="")

    def test_confidence_out_of_range(self):
        with self.assertRaises(ValidationError):
            self._make(extraction_confidence=1.5)
        with self.assertRaises(ValidationError):
            self._make(temporal_confidence=-0.1)

    def test_frozen(self):
        tku = self._make()
        with self.assertRaises(ValidationError):
            tku.content = "changed"  # type: ignore[misc]

    def test_extra_forbid(self):
        with self.assertRaises(ValidationError):
            self._make(unknown_field="x")

    def test_roundtrip_json(self):
        tku = self._make(
            valid_from=_utc(1990),
            valid_until=_utc(2022),
            domain=FinanceDomain.ML_FINANCE,
            tags=("deep_learning",),
        )
        data = tku.model_dump_json()
        restored = TemporalKnowledgeUnit.model_validate_json(data)
        self.assertEqual(restored.content, tku.content)
        self.assertEqual(restored.content_hash, tku.content_hash)
        self.assertEqual(restored.domain, FinanceDomain.ML_FINANCE)
        self.assertEqual(restored.tags, ("deep_learning",))


class TKUCollectionTests(unittest.TestCase):
    def _make_collection(self):
        tku1 = TemporalKnowledgeUnit(
            content="Momentum premium is significant",
            source_id="doc-1",
            source_title="Paper A",
            valid_from=_utc(1990),
            valid_until=_utc(2020),
            domain=FinanceDomain.FACTOR_INVESTING,
            content_type=KnowledgeType.FACT,
        )
        tku2 = TemporalKnowledgeUnit(
            content="Black-Scholes formula",
            source_id="doc-1",
            source_title="Paper A",
            domain=FinanceDomain.OPTIONS_PRICING,
            content_type=KnowledgeType.FORMULA,
        )
        tku3 = TemporalKnowledgeUnit(
            content="Market efficiency claim",
            source_id="doc-1",
            source_title="Paper A",
            valid_from=_utc(2000),
            valid_until=_utc(2010),
            content_type=KnowledgeType.CLAIM,
        )
        return TKUCollection(
            document_id="doc-1",
            source_title="Paper A",
            tkus=(tku1, tku2, tku3),
        )

    def test_filter_by_date(self):
        col = self._make_collection()
        valid = col.filter_by_date(_utc(2005))
        # tku1 (1990-2020) and tku3 (2000-2010) are valid; tku2 (undated) valid
        self.assertEqual(len(valid), 3)

    def test_filter_by_date_excludes_expired(self):
        col = self._make_collection()
        valid = col.filter_by_date(_utc(2022))
        # tku1 expired (valid_until=2020), tku3 expired (valid_until=2010)
        # only tku2 (unbounded) still valid
        self.assertEqual(len(valid), 1)

    def test_by_domain(self):
        col = self._make_collection()
        result = col.by_domain(FinanceDomain.FACTOR_INVESTING)
        self.assertEqual(len(result), 1)
        self.assertIn("Momentum", result[0].content)

    def test_stats(self):
        col = self._make_collection()
        s = col.stats()
        self.assertEqual(s["fact"], 1)
        self.assertEqual(s["formula"], 1)
        self.assertEqual(s["claim"], 1)

    def test_empty_collection(self):
        col = TKUCollection(
            document_id="doc-2",
            source_title="Empty",
        )
        self.assertEqual(len(col.tkus), 0)
        self.assertEqual(col.stats(), {})

    def test_frozen(self):
        col = self._make_collection()
        with self.assertRaises(ValidationError):
            col.document_id = "changed"  # type: ignore[misc]

    def test_roundtrip_json(self):
        col = self._make_collection()
        data = col.model_dump_json()
        restored = TKUCollection.model_validate_json(data)
        self.assertEqual(len(restored.tkus), 3)
        self.assertEqual(restored.document_id, "doc-1")


class FinanceDomainTests(unittest.TestCase):
    def test_all_values(self):
        self.assertEqual(len(FinanceDomain), 7)
        self.assertIn(FinanceDomain.CREDIT_RISK, FinanceDomain)

    def test_string_value(self):
        self.assertEqual(FinanceDomain.ML_FINANCE.value, "ml_finance")


class KnowledgeTypeTests(unittest.TestCase):
    def test_all_values(self):
        self.assertEqual(len(KnowledgeType), 6)
        self.assertIn(KnowledgeType.METHODOLOGY, KnowledgeType)


if __name__ == "__main__":
    unittest.main()
