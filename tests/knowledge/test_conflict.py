"""Tests for knowledge.conflict (C2: CSCG data models)."""

import unittest
from datetime import datetime, timezone

from pydantic import ValidationError

from finsage.knowledge.conflict import (
    ConflictRecord,
    ConflictType,
    ResolutionStatus,
)


def _utc(y: int, m: int = 1, d: int = 1) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _make_conflict(**overrides) -> ConflictRecord:
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


class ConflictTypeTests(unittest.TestCase):
    def test_all_values(self):
        self.assertEqual(len(ConflictType), 3)

    def test_string_values(self):
        self.assertEqual(ConflictType.VALUE_MISMATCH.value, "value_mismatch")
        self.assertEqual(
            ConflictType.DIRECTION_MISMATCH.value, "direction_mismatch"
        )
        self.assertEqual(
            ConflictType.SIGNIFICANCE_MISMATCH.value,
            "significance_mismatch",
        )


class ResolutionStatusTests(unittest.TestCase):
    def test_all_values(self):
        self.assertEqual(len(ResolutionStatus), 5)

    def test_string_values(self):
        self.assertEqual(ResolutionStatus.UNRESOLVED.value, "unresolved")
        self.assertEqual(
            ResolutionStatus.CONTEXTUAL_SYNTHESIS.value,
            "contextual_synthesis",
        )


class ConflictRecordCreationTests(unittest.TestCase):
    def test_minimal_creation(self):
        cr = _make_conflict()
        self.assertEqual(cr.tku_a_id, "tku-001")
        self.assertEqual(cr.tku_b_id, "tku-002")
        self.assertEqual(cr.conflict_type, ConflictType.SIGNIFICANCE_MISMATCH)
        self.assertEqual(cr.entity_name, "momentum factor")
        self.assertAlmostEqual(cr.detection_confidence, 0.85)
        self.assertIsNotNone(cr.conflict_id)
        self.assertIsNotNone(cr.detected_at)

    def test_default_resolution_is_unresolved(self):
        cr = _make_conflict()
        self.assertEqual(cr.resolution_status, ResolutionStatus.UNRESOLVED)

    def test_default_synthesis_is_none(self):
        cr = _make_conflict()
        self.assertIsNone(cr.synthesis_text)
        self.assertIsNone(cr.resolution_explanation)

    def test_temporal_windows_stored(self):
        cr = _make_conflict()
        self.assertEqual(cr.tku_a_valid_from, _utc(1965))
        self.assertEqual(cr.tku_a_valid_until, _utc(1990))
        self.assertEqual(cr.tku_b_valid_from, _utc(1980))
        self.assertEqual(cr.tku_b_valid_until, _utc(2002))

    def test_temporal_overlap_stored(self):
        cr = _make_conflict()
        self.assertEqual(cr.temporal_overlap_start, _utc(1980))
        self.assertEqual(cr.temporal_overlap_end, _utc(1990))

    def test_none_temporal_windows_allowed(self):
        cr = _make_conflict(
            tku_a_valid_from=None,
            tku_a_valid_until=None,
            tku_b_valid_from=None,
            tku_b_valid_until=None,
            temporal_overlap_start=None,
            temporal_overlap_end=None,
        )
        self.assertIsNone(cr.tku_a_valid_from)


class ConflictRecordValidationTests(unittest.TestCase):
    def test_empty_content_rejected(self):
        with self.assertRaises(ValidationError):
            _make_conflict(tku_a_content="")

    def test_empty_source_rejected(self):
        with self.assertRaises(ValidationError):
            _make_conflict(tku_a_source="")

    def test_empty_entity_rejected(self):
        with self.assertRaises(ValidationError):
            _make_conflict(entity_name="")

    def test_confidence_out_of_range_high(self):
        with self.assertRaises(ValidationError):
            _make_conflict(detection_confidence=1.5)

    def test_confidence_out_of_range_low(self):
        with self.assertRaises(ValidationError):
            _make_conflict(detection_confidence=-0.1)

    def test_frozen(self):
        cr = _make_conflict()
        with self.assertRaises(ValidationError):
            cr.entity_name = "changed"  # type: ignore[misc]

    def test_extra_forbid(self):
        with self.assertRaises(ValidationError):
            _make_conflict(unknown_field="x")


class IsResolvedTests(unittest.TestCase):
    def test_unresolved_returns_false(self):
        cr = _make_conflict()
        self.assertFalse(cr.is_resolved())

    def test_resolved_by_market_returns_true(self):
        cr = _make_conflict(
            resolution_status=ResolutionStatus.RESOLVED_BY_MARKET
        )
        self.assertTrue(cr.is_resolved())

    def test_resolved_by_period_returns_true(self):
        cr = _make_conflict(
            resolution_status=ResolutionStatus.RESOLVED_BY_PERIOD
        )
        self.assertTrue(cr.is_resolved())

    def test_resolved_by_methodology_returns_true(self):
        cr = _make_conflict(
            resolution_status=ResolutionStatus.RESOLVED_BY_METHODOLOGY
        )
        self.assertTrue(cr.is_resolved())

    def test_contextual_synthesis_returns_true(self):
        cr = _make_conflict(
            resolution_status=ResolutionStatus.CONTEXTUAL_SYNTHESIS
        )
        self.assertTrue(cr.is_resolved())


class UserFacingNoteTests(unittest.TestCase):
    def test_returns_synthesis_when_set(self):
        cr = _make_conflict(
            synthesis_text="The discrepancy is explained by sample period."
        )
        self.assertEqual(
            cr.user_facing_note(),
            "The discrepancy is explained by sample period.",
        )

    def test_returns_description_when_no_synthesis(self):
        cr = _make_conflict()
        note = cr.user_facing_note()
        self.assertIn("significance_mismatch", note)
        self.assertIn("momentum factor", note)
        self.assertIn("1.2% monthly", note)
        self.assertIn("statistically insignificant", note)


class PairHashTests(unittest.TestCase):
    def test_pair_hash_is_hex_16(self):
        cr = _make_conflict()
        h = cr.pair_hash()
        self.assertEqual(len(h), 16)
        self.assertRegex(h, r"^[0-9a-f]{16}$")

    def test_pair_hash_order_independent(self):
        cr1 = _make_conflict(tku_a_id="aaa", tku_b_id="bbb")
        cr2 = _make_conflict(tku_a_id="bbb", tku_b_id="aaa")
        self.assertEqual(cr1.pair_hash(), cr2.pair_hash())

    def test_pair_hash_differs_for_different_pairs(self):
        cr1 = _make_conflict(tku_a_id="aaa", tku_b_id="bbb")
        cr2 = _make_conflict(tku_a_id="aaa", tku_b_id="ccc")
        self.assertNotEqual(cr1.pair_hash(), cr2.pair_hash())


class RoundtripTests(unittest.TestCase):
    def test_json_roundtrip(self):
        cr = _make_conflict(
            resolution_status=ResolutionStatus.CONTEXTUAL_SYNTHESIS,
            synthesis_text="Reconciled via methodological differences.",
        )
        data = cr.model_dump_json()
        restored = ConflictRecord.model_validate_json(data)
        self.assertEqual(restored.conflict_id, cr.conflict_id)
        self.assertEqual(restored.tku_a_content, cr.tku_a_content)
        self.assertEqual(
            restored.resolution_status,
            ResolutionStatus.CONTEXTUAL_SYNTHESIS,
        )
        self.assertEqual(
            restored.synthesis_text,
            "Reconciled via methodological differences.",
        )

    def test_dict_roundtrip(self):
        cr = _make_conflict()
        d = cr.model_dump()
        restored = ConflictRecord.model_validate(d)
        self.assertEqual(restored.entity_name, "momentum factor")


if __name__ == "__main__":
    unittest.main()
