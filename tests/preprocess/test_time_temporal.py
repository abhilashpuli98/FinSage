"""Tests for preprocess.time — temporal extraction helpers (C1: TKU)."""

import unittest
from datetime import datetime, timezone

from finsage.preprocess.time import (
    DateRange,
    DateWindow,
    extract_date_strings,
    fiscal_year_to_window,
    halfyear_to_window,
    infer_sample_period,
    infer_temporal_bounds,
    normalize_date_range,
    parse_validity_window,
    quarter_to_window,
)


def _utc(y: int, m: int = 1, d: int = 1) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


# ── quarter_to_window ────────────────────────────────────────────────


class QuarterToWindowTests(unittest.TestCase):
    def test_q1(self):
        w = quarter_to_window(2024, 1)
        self.assertEqual(w.start, _utc(2024, 1, 1))
        self.assertEqual(w.end, _utc(2024, 4, 1))

    def test_q4(self):
        w = quarter_to_window(2024, 4)
        self.assertEqual(w.start, _utc(2024, 10, 1))
        self.assertEqual(w.end, _utc(2025, 1, 1))

    def test_invalid_quarter(self):
        with self.assertRaises(ValueError):
            quarter_to_window(2024, 5)

    def test_quarter_zero(self):
        with self.assertRaises(ValueError):
            quarter_to_window(2024, 0)


# ── halfyear_to_window ───────────────────────────────────────────────


class HalfyearToWindowTests(unittest.TestCase):
    def test_h1(self):
        w = halfyear_to_window(2024, 1)
        self.assertEqual(w.start, _utc(2024, 1, 1))
        self.assertEqual(w.end, _utc(2024, 7, 1))

    def test_h2(self):
        w = halfyear_to_window(2024, 2)
        self.assertEqual(w.start, _utc(2024, 7, 1))
        self.assertEqual(w.end, _utc(2025, 1, 1))

    def test_invalid_half(self):
        with self.assertRaises(ValueError):
            halfyear_to_window(2024, 3)


# ── fiscal_year_to_window ────────────────────────────────────────────


class FiscalYearToWindowTests(unittest.TestCase):
    def test_fy2024(self):
        w = fiscal_year_to_window(2024)
        self.assertEqual(w.start, _utc(2024, 1, 1))
        self.assertEqual(w.end, _utc(2025, 1, 1))


# ── normalize_date_range ─────────────────────────────────────────────


class NormalizeDateRangeTests(unittest.TestCase):
    def test_q3_2020(self):
        r = normalize_date_range("Q3 2020")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(2020, 7, 1))
        self.assertEqual(r.end, _utc(2020, 9, 30))
        self.assertAlmostEqual(r.confidence, 0.95)

    def test_q1_2024(self):
        r = normalize_date_range("Q1 2024")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(2024, 1, 1))
        self.assertEqual(r.end, _utc(2024, 3, 31))

    def test_year_range_1990_2022(self):
        r = normalize_date_range("1990-2022")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(1990, 1, 1))
        self.assertEqual(r.end, _utc(2022, 12, 31))
        self.assertAlmostEqual(r.confidence, 0.90)

    def test_post_2008(self):
        r = normalize_date_range("post-2008")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(2009, 1, 1))
        self.assertIsNone(r.end)
        self.assertAlmostEqual(r.confidence, 0.65)

    def test_pre_2020(self):
        r = normalize_date_range("pre-2020")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertIsNone(r.start)
        self.assertEqual(r.end, _utc(2020, 12, 31))
        self.assertAlmostEqual(r.confidence, 0.65)

    def test_since_2010(self):
        r = normalize_date_range("since 2010")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(2010, 1, 1))
        self.assertIsNone(r.end)
        self.assertAlmostEqual(r.confidence, 0.70)

    def test_fiscal_year_2019(self):
        r = normalize_date_range("fiscal year 2019")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(2019, 4, 1))
        self.assertEqual(r.end, _utc(2020, 3, 31))
        self.assertAlmostEqual(r.confidence, 0.85)

    def test_fy2024(self):
        r = normalize_date_range("FY2024")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(2024, 4, 1))

    def test_single_year(self):
        r = normalize_date_range("2020")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(2020, 1, 1))
        self.assertEqual(r.end, _utc(2020, 12, 31))
        self.assertAlmostEqual(r.confidence, 0.50)

    def test_unknown_returns_none(self):
        r = normalize_date_range("not a date")
        self.assertIsNone(r)

    def test_month_year_range(self):
        r = normalize_date_range("January 1990 to December 2020")
        self.assertIsNotNone(r)
        assert r is not None
        self.assertEqual(r.start, _utc(1990, 1, 1))
        self.assertEqual(r.end, _utc(2020, 12, 31))
        self.assertAlmostEqual(r.confidence, 0.95)


# ── extract_date_strings ─────────────────────────────────────────────


class ExtractDateStringsTests(unittest.TestCase):
    def test_finds_dates_in_financial_text(self):
        text = (
            "We study momentum returns from 1963-2022. "
            "Q3 2020 was volatile. Post-2008 regulations changed markets."
        )
        results = extract_date_strings(text)
        self.assertIn("1963-2022", results)
        self.assertIn("Q3 2020", results)
        self.assertIn("Post-2008", results)

    def test_deduplication(self):
        text = "Returns in 2020 and again in 2020"
        results = extract_date_strings(text)
        self.assertEqual(results.count("2020"), 1)

    def test_empty_text(self):
        results = extract_date_strings("")
        self.assertEqual(results, [])

    def test_no_dates(self):
        results = extract_date_strings("no temporal info here at all")
        self.assertEqual(results, [])


# ── infer_sample_period ──────────────────────────────────────────────


class InferSamplePeriodTests(unittest.TestCase):
    def test_prefers_closed_range(self):
        closed = DateRange(
            start=_utc(1990),
            end=_utc(2022, 12, 31),
            confidence=0.9,
            raw_text="1990-2022",
        )
        open_ended = DateRange(
            start=_utc(2010),
            end=None,
            confidence=0.7,
            raw_text="since 2010",
        )
        result = infer_sample_period([open_ended, closed])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.raw_text, "1990-2022")

    def test_prefers_longer_span(self):
        short = DateRange(
            start=_utc(2020),
            end=_utc(2021, 12, 31),
            confidence=0.9,
            raw_text="2020-2021",
        )
        long = DateRange(
            start=_utc(1990),
            end=_utc(2022, 12, 31),
            confidence=0.9,
            raw_text="1990-2022",
        )
        result = infer_sample_period([short, long])
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.raw_text, "1990-2022")

    def test_empty_list_returns_none(self):
        self.assertIsNone(infer_sample_period([]))

    def test_single_candidate(self):
        r = DateRange(
            start=_utc(2020),
            end=_utc(2022, 12, 31),
            confidence=0.9,
            raw_text="2020-2022",
        )
        result = infer_sample_period([r])
        self.assertEqual(result, r)


# ── parse_validity_window (legacy wrapper) ───────────────────────────


class ParseValidityWindowTests(unittest.TestCase):
    def test_quarter_pattern(self):
        w = parse_validity_window("Revenue grew 5% in Q3 2024")
        self.assertIsNotNone(w)
        assert w is not None
        self.assertEqual(w.start, _utc(2024, 7, 1))
        self.assertEqual(w.end, _utc(2024, 10, 1))

    def test_quarter_lowercase(self):
        w = parse_validity_window("q2 2025 outlook")
        self.assertIsNotNone(w)
        assert w is not None
        self.assertEqual(w.start, _utc(2025, 4, 1))

    def test_halfyear_pattern(self):
        w = parse_validity_window("H1 2025 guidance raised")
        self.assertIsNotNone(w)
        assert w is not None
        self.assertEqual(w.start, _utc(2025, 1, 1))
        self.assertEqual(w.end, _utc(2025, 7, 1))

    def test_fy_pattern(self):
        w = parse_validity_window("FY2024 earnings beat estimate")
        self.assertIsNotNone(w)
        assert w is not None
        self.assertEqual(w.start, _utc(2024, 1, 1))
        self.assertEqual(w.end, _utc(2025, 1, 1))

    def test_year_range(self):
        w = parse_validity_window("Rate cuts expected 2024-2025")
        self.assertIsNotNone(w)
        assert w is not None
        self.assertEqual(w.start, _utc(2024, 1, 1))
        self.assertEqual(w.end, _utc(2026, 1, 1))

    def test_bare_year(self):
        w = parse_validity_window("S&P 500 returned 26% in 2023")
        self.assertIsNotNone(w)
        assert w is not None
        self.assertEqual(w.start, _utc(2023, 1, 1))
        self.assertEqual(w.end, _utc(2024, 1, 1))

    def test_no_period_found(self):
        w = parse_validity_window("no temporal info here")
        self.assertIsNone(w)

    def test_returns_date_window_namedtuple(self):
        w = parse_validity_window("Q2 2024")
        self.assertIsInstance(w, DateWindow)


# ── infer_temporal_bounds ────────────────────────────────────────────


class InferTemporalBoundsTests(unittest.TestCase):
    def test_with_period(self):
        vf, vu = infer_temporal_bounds("Q3 2024 earnings call")
        self.assertEqual(vf, _utc(2024, 7, 1))
        self.assertEqual(vu, _utc(2024, 10, 1))

    def test_fallback_to_as_of(self):
        vf, vu = infer_temporal_bounds(
            "rates unchanged",
            fallback_as_of=_utc(2024, 6, 12),
        )
        self.assertEqual(vf, _utc(2024, 6, 12))
        self.assertIsNone(vu)

    def test_no_info_no_fallback(self):
        vf, vu = infer_temporal_bounds("nothing here")
        self.assertIsNone(vf)
        self.assertIsNone(vu)


if __name__ == "__main__":
    unittest.main()
