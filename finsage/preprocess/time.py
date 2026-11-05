"""Time helpers used across preprocess + downstream flows.

Pure synchronous functions. ``zoneinfo`` is stdlib (Python 3.9+) so we
avoid pulling in pendulum/arrow as a dependency. All datetimes are
expected to be aware (tz-attached); naive ones are treated as UTC.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from typing import NamedTuple

from dateutil.relativedelta import relativedelta

# ISO 8601 + the common journal/news layouts. Order matters: more specific
# patterns first so ``2024-04-15T10:30:00Z`` doesn't get partial-matched
# as "2024-04-15".
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %b %Y",
    "%d %B %Y",
    "%b %d, %Y",
    "%B %d, %Y",
)


def to_utc(dt: datetime) -> datetime:
    """Return a UTC-aware datetime equivalent to ``dt``.

    Naive inputs are *interpreted* as UTC (we do not guess local time).
    Aware inputs are converted into the UTC zone.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_filing_date(value: str) -> datetime:
    """Parse a date string (filing dates, news timestamps) into UTC datetime.

    Accepts ISO-8601 with or without time component, ``YYYY-MM-DD``,
    ``YYYY/MM/DD``, journal long form (``Apr 15, 2024``), and a few
    related variants.

    Args:
        value: Date or datetime string.

    Returns:
        Aware UTC datetime.

    Raises:
        ValueError: If none of the accepted formats match.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty date string")

    last_error: Exception | None = None
    for fmt in _DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError as exc:
            last_error = exc
            continue
        return to_utc(parsed)

    raise ValueError(
        f"could not parse date {value!r}; tried {len(_DATE_FORMATS)} formats"
    ) from last_error


def parse_news_datetime(value: str) -> datetime:
    """Parse an RFC 822 or ISO-8601 news timestamp into UTC.

    Args:
        value: News timestamp string.

    Returns:
        A timezone-aware UTC datetime.

    Raises:
        ValueError: If the timestamp is empty or unsupported.
    """
    text = value.strip()
    if not text:
        raise ValueError("empty news datetime string")

    try:
        return parse_filing_date(text)
    except ValueError as iso_error:
        try:
            parsed = parsedate_to_datetime(text)
        except (TypeError, ValueError) as rfc_error:
            raise ValueError(
                f"could not parse news datetime {value!r}"
            ) from rfc_error
        if parsed is None:
            raise ValueError(
                f"could not parse news datetime {value!r}"
            ) from iso_error
        return to_utc(parsed)


def business_days_between(a: date, b: date) -> int:
    """Count weekdays (Mon-Fri) strictly between ``a`` and ``b``.

    The count is **inclusive of both endpoints** when they fall on
    weekdays, and direction-insensitive (``a > b`` returns the same value
    as the swap). No holiday calendar — that arrives in a follow-up issue.

    Examples:
        Mon -> Mon (same day) -> 1
        Mon -> Fri (same week) -> 5
        Fri -> Mon (over weekend) -> 2
    """
    if a > b:
        a, b = b, a
    total_days = (b - a).days + 1
    full_weeks, remainder = divmod(total_days, 7)
    weekdays = full_weeks * 5

    # Walk the leftover days, starting from a's weekday.
    start_dow = a.weekday()
    for offset in range(remainder):
        if (start_dow + offset) % 7 < 5:  # 0..4 are Mon..Fri
            weekdays += 1
    return weekdays


# ── Temporal extraction helpers (C1: TKU) ────────────────────────────
# Pure functions that parse date ranges from financial text.  Used by
# downstream extraction flows to populate TemporalKnowledgeUnit validity
# windows. Only stdlib + python-dateutil; no finsage imports beyond utils.


class DateWindow(NamedTuple):
    """A half-open ``[start, end)`` date window."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class DateRange:
    """A date range extracted from text, with confidence and raw source.

    Uses frozen dataclass (not Pydantic) following AGENTS.md convention
    for preprocess-layer data values.
    """

    start: datetime | None
    end: datetime | None
    confidence: float
    raw_text: str


# ── Quarter / half-year / fiscal-year helpers ────────────────────────

_QUARTER_START_MONTH: dict[int, int] = {1: 1, 2: 4, 3: 7, 4: 10}
_QUARTER_END: dict[int, tuple[int, int]] = {
    1: (3, 31),
    2: (6, 30),
    3: (9, 30),
    4: (12, 31),
}


def quarter_to_window(year: int, quarter: int) -> DateWindow:
    """Convert a calendar quarter (1-4) to a UTC ``DateWindow``.

    Args:
        year: Calendar year.
        quarter: Quarter number (1-4).

    Returns:
        Half-open ``[start_of_quarter, start_of_next_quarter)`` window.

    Raises:
        ValueError: If quarter is not in 1-4.
    """
    if quarter not in _QUARTER_START_MONTH:
        raise ValueError(f"quarter must be 1-4, got {quarter}")
    start_month = _QUARTER_START_MONTH[quarter]
    start = datetime(year, start_month, 1, tzinfo=timezone.utc)
    end = start + relativedelta(months=3)
    return DateWindow(start, end)


def halfyear_to_window(year: int, half: int) -> DateWindow:
    """Convert a half-year (1 or 2) to a UTC ``DateWindow``.

    Args:
        year: Calendar year.
        half: Half-year number (1 = Jan-Jun, 2 = Jul-Dec).

    Returns:
        Half-open ``[start_of_half, start_of_next_half)`` window.

    Raises:
        ValueError: If half is not 1 or 2.
    """
    if half not in (1, 2):
        raise ValueError(f"half must be 1 or 2, got {half}")
    start_month = 1 if half == 1 else 7
    start = datetime(year, start_month, 1, tzinfo=timezone.utc)
    end = start + relativedelta(months=6)
    return DateWindow(start, end)


def fiscal_year_to_window(year: int) -> DateWindow:
    """Convert a fiscal/calendar year to a UTC ``DateWindow``.

    Assumes calendar-year fiscal year (Jan 1 – Dec 31).

    Args:
        year: The fiscal year.

    Returns:
        Half-open ``[Jan 1 year, Jan 1 year+1)`` window.
    """
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    return DateWindow(start, end)


# ── Regex patterns for date extraction ───────────────────────────────

# Month name → month number mapping
_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_MONTH_NAMES = "|".join(_MONTH_MAP.keys())

# Ordered by priority — more specific patterns first.
_MONTH_YEAR_RANGE_RE = re.compile(
    rf"\b({_MONTH_NAMES})\s+(\d{{4}})\s+(?:to|-|–|—)\s+"
    rf"({_MONTH_NAMES})\s+(\d{{4}})\b",
    re.IGNORECASE,
)
_QUARTER_RE = re.compile(r"\bQ([1-4])\s*['\u2019]?(\d{4})\b", re.IGNORECASE)
_HALF_RE = re.compile(r"\bH([12])\s*['\u2019]?(\d{4})\b", re.IGNORECASE)
_FY_RE = re.compile(
    r"\b(?:fiscal\s+year|FY)\s*['\u2019]?(\d{4})\b", re.IGNORECASE
)
_POST_RE = re.compile(r"\bpost[- ](\d{4})\b", re.IGNORECASE)
_PRE_RE = re.compile(r"\bpre[- ](\d{4})\b", re.IGNORECASE)
_SINCE_RE = re.compile(r"\bsince\s+(\d{4})\b", re.IGNORECASE)
_YEAR_RANGE_RE = re.compile(r"\b(\d{4})\s*[-–—]\s*(\d{4})\b")
_BARE_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

_MIN_YEAR = 1800
_MAX_YEAR = 2030


def _valid_year(year: int) -> bool:
    return _MIN_YEAR <= year <= _MAX_YEAR


# ── Public functions ─────────────────────────────────────────────────


def extract_date_strings(text: str) -> list[str]:
    """Extract all date-like strings from *text* via regex.

    Returns a deduplicated list of raw matched strings, preserving the
    order of first occurrence.  Patterns recognised (in priority order):

    - ``"January 1990 to December 2020"``
    - ``"1990-2020"`` / ``"1990–2020"``
    - ``"Q1 2020"`` .. ``"Q4 2020"``
    - ``"fiscal year 2019"`` / ``"FY2019"``
    - ``"post-2008"`` / ``"pre-2020"`` / ``"since 2010"``
    - Standalone 4-digit years in 1900–2029

    Args:
        text: Free-form text to scan.

    Returns:
        Deduplicated list of matched raw date strings.
    """
    seen: set[str] = set()
    results: list[str] = []

    patterns = [
        _MONTH_YEAR_RANGE_RE,
        _YEAR_RANGE_RE,
        _QUARTER_RE,
        _HALF_RE,
        _FY_RE,
        _POST_RE,
        _PRE_RE,
        _SINCE_RE,
        _BARE_YEAR_RE,
    ]
    for pat in patterns:
        for m in pat.finditer(text):
            raw = m.group(0).strip()
            if raw not in seen:
                seen.add(raw)
                results.append(raw)
    return results


def normalize_date_range(raw: str) -> DateRange | None:
    """Convert a raw date string into a ``DateRange``.

    Handles quarters, fiscal years, pre/post/since, year ranges,
    month-year ranges, and standalone years.

    Args:
        raw: A raw date string (typically from :func:`extract_date_strings`).

    Returns:
        A ``DateRange`` if recognised, else ``None``.
    """
    text = raw.strip()

    # Q1-Q4 YYYY
    m = _QUARTER_RE.fullmatch(text) or _QUARTER_RE.search(text)
    if m and text.upper().startswith("Q"):
        q, y = int(m.group(1)), int(m.group(2))
        if _valid_year(y):
            end_m, end_d = _QUARTER_END[q]
            return DateRange(
                start=datetime(
                    y, _QUARTER_START_MONTH[q], 1, tzinfo=timezone.utc
                ),
                end=datetime(y, end_m, end_d, tzinfo=timezone.utc),
                confidence=0.95,
                raw_text=raw,
            )

    # fiscal year YYYY / FY YYYY
    m = _FY_RE.search(text)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return DateRange(
                start=datetime(y, 4, 1, tzinfo=timezone.utc),
                end=datetime(y + 1, 3, 31, tzinfo=timezone.utc),
                confidence=0.85,
                raw_text=raw,
            )

    # post-YYYY
    m = _POST_RE.search(text)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return DateRange(
                start=datetime(y + 1, 1, 1, tzinfo=timezone.utc),
                end=None,
                confidence=0.65,
                raw_text=raw,
            )

    # pre-YYYY
    m = _PRE_RE.search(text)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return DateRange(
                start=None,
                end=datetime(y, 12, 31, tzinfo=timezone.utc),
                confidence=0.65,
                raw_text=raw,
            )

    # since YYYY
    m = _SINCE_RE.search(text)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return DateRange(
                start=datetime(y, 1, 1, tzinfo=timezone.utc),
                end=None,
                confidence=0.70,
                raw_text=raw,
            )

    # Month YYYY to Month YYYY
    m = _MONTH_YEAR_RANGE_RE.search(text)
    if m:
        m1 = _MONTH_MAP.get(m.group(1).lower())
        y1 = int(m.group(2))
        m2 = _MONTH_MAP.get(m.group(3).lower())
        y2 = int(m.group(4))
        if m1 and m2 and _valid_year(y1) and _valid_year(y2):
            return DateRange(
                start=datetime(y1, m1, 1, tzinfo=timezone.utc),
                end=datetime(
                    y2,
                    m2,
                    _month_last_day(y2, m2),
                    tzinfo=timezone.utc,
                ),
                confidence=0.95,
                raw_text=raw,
            )

    # YYYY-YYYY range
    m = _YEAR_RANGE_RE.search(text)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if _valid_year(y1) and _valid_year(y2) and y1 < y2:
            return DateRange(
                start=datetime(y1, 1, 1, tzinfo=timezone.utc),
                end=datetime(y2, 12, 31, tzinfo=timezone.utc),
                confidence=0.90,
                raw_text=raw,
            )

    # Single YYYY
    m = _BARE_YEAR_RE.fullmatch(text)
    if m:
        y = int(m.group(1))
        if _valid_year(y):
            return DateRange(
                start=datetime(y, 1, 1, tzinfo=timezone.utc),
                end=datetime(y, 12, 31, tzinfo=timezone.utc),
                confidence=0.50,
                raw_text=raw,
            )

    return None


def _month_last_day(year: int, month: int) -> int:
    """Return the last day of the given month."""
    if month == 12:
        return 31
    next_month_start = datetime(year, month + 1, 1)
    last_day = next_month_start + relativedelta(days=-1)
    return last_day.day


def infer_sample_period(
    date_ranges: list[DateRange],
) -> DateRange | None:
    """Pick the most likely data sample period from extracted ranges.

    Strategy (deterministic — same input always same output):
    1. Prefer ranges with both ``start`` AND ``end`` (not open-ended).
    2. Among those, prefer longer spans (multi-year > single-year).
    3. Among ties, prefer higher confidence.
    4. Among remaining ties, prefer the latest ``end`` date.
    5. If empty or all ``None``: return ``None``.

    Args:
        date_ranges: List of ``DateRange`` objects from a document.

    Returns:
        The most likely sample period, or ``None``.
    """
    if not date_ranges:
        return None

    # Split into closed (both endpoints) and open-ended
    closed = [r for r in date_ranges if r.start and r.end]
    candidates = closed if closed else date_ranges

    # Filter out None-only entries
    candidates = [r for r in candidates if r.start or r.end]
    if not candidates:
        return None

    def sort_key(r: DateRange) -> tuple[float, float, float]:
        # Larger span first, then higher confidence, then later end
        span_days = 0.0
        if r.start and r.end:
            span_days = (r.end - r.start).total_seconds()
        end_ts = r.end.timestamp() if r.end else 0.0
        return (span_days, r.confidence, end_ts)

    candidates.sort(key=sort_key, reverse=True)
    return candidates[0]


# ── Legacy convenience wrappers (used by existing code) ──────────────


def parse_validity_window(text: str) -> DateWindow | None:
    """Extract the first recognisable period from *text*.

    Recognises (in priority order):
    - ``Q3 2024`` → calendar quarter
    - ``H1 2025`` → half-year
    - ``FY2024`` / ``FY 2024`` → fiscal year
    - ``2024-2025`` → multi-year range
    - ``2024`` → single calendar year

    Returns:
        A ``DateWindow`` if a period is found, else ``None``.
    """
    m = _QUARTER_RE.search(text)
    if m:
        return quarter_to_window(int(m.group(2)), int(m.group(1)))

    m = _HALF_RE.search(text)
    if m:
        return halfyear_to_window(int(m.group(2)), int(m.group(1)))

    m = _FY_RE.search(text)
    if m:
        return fiscal_year_to_window(int(m.group(1)))

    m = _YEAR_RANGE_RE.search(text)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        if y1 < y2:
            start = datetime(y1, 1, 1, tzinfo=timezone.utc)
            end = datetime(y2 + 1, 1, 1, tzinfo=timezone.utc)
            return DateWindow(start, end)

    m = _BARE_YEAR_RE.search(text)
    if m:
        return fiscal_year_to_window(int(m.group(1)))

    return None


def infer_temporal_bounds(
    text: str,
    fallback_as_of: datetime | None = None,
) -> tuple[datetime | None, datetime | None]:
    """Heuristic: extract ``(valid_from, valid_until)`` from *text*.

    Tries :func:`parse_validity_window` first.  Falls back to using
    *fallback_as_of* as ``valid_from`` with no ``valid_until``.

    Returns:
        Tuple of ``(valid_from, valid_until)``; either may be ``None``.
    """
    window = parse_validity_window(text)
    if window is not None:
        return (window.start, window.end)
    if fallback_as_of is not None:
        return (to_utc(fallback_as_of), None)
    return (None, None)
