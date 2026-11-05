"""TemporalPaperFlow — extract time-bounded TKUs from a paper.

Contribution C1 integration layer.  This flow combines:

- ``PaperFlow``-style fetch and parse (reusing ``preprocess.fetch`` +
  ``preprocess.format`` directly),
- deterministic temporal extraction (``preprocess.time``),
- LLM-routed knowledge extraction (``mind.router``), and
- semantic deduplication

to produce a self-contained ``TKUCollection`` from one paper input.

The flow is **pure processing** (``input → artifact``): it binds no
library, persists nothing, and retrieves nothing.  The caller decides
what to do with the returned ``TKUCollection``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from finsage.configs.ollama import OllamaModelCfg, RouterCfg, TaskType
from finsage.configs.paper import (
    ArxivIdentifier,
    DoiIdentifier,
    HttpUrl,
    LocalFilePath,
    PaperInput,
    RawText,
)
from finsage.configs.temporal import TemporalPaperCfg
from finsage.knowledge.temporal import (
    FinanceDomain,
    KnowledgeType,
    TemporalKnowledgeUnit,
    TKUCollection,
)
from finsage.mind.conflict import compute_cosine_similarity
from finsage.mind.router import LLMRouter, safe_json_parse
from finsage.preprocess.fetch import (
    Fetched,
    RawPaper,
    fetch_arxiv,
    fetch_url,
    read_local_file,
)
from finsage.preprocess.format import parse_pdf
from finsage.preprocess.time import (
    DateRange,
    extract_date_strings,
    infer_sample_period,
    normalize_date_range,
)

_LOGGER = logging.getLogger(__name__)

__all__ = ["TemporalPaperFlow"]


# ── Domain detection heuristic ───────────────────────────────────────

_DOMAIN_KEYWORDS: dict[FinanceDomain, tuple[str, ...]] = {
    FinanceDomain.FACTOR_INVESTING: (
        "factor",
        "momentum",
        "value premium",
        "size effect",
        "fama-french",
        "smart beta",
        "alpha",
    ),
    FinanceDomain.OPTIONS_PRICING: (
        "option",
        "black-scholes",
        "implied volatility",
        "greeks",
        "straddle",
        "call",
        "put",
        "derivatives",
    ),
    FinanceDomain.MARKET_MICROSTRUCTURE: (
        "bid-ask",
        "spread",
        "order book",
        "market maker",
        "liquidity",
        "high-frequency",
        "tick",
    ),
    FinanceDomain.CREDIT_RISK: (
        "credit risk",
        "default",
        "credit spread",
        "cds",
        "rating",
        "bankruptcy",
        "merton",
    ),
    FinanceDomain.ML_FINANCE: (
        "machine learning",
        "deep learning",
        "neural network",
        "random forest",
        "gradient boosting",
        "lstm",
        "transformer",
        "prediction",
    ),
    FinanceDomain.BEHAVIORAL_FINANCE: (
        "behavioral",
        "sentiment",
        "overconfidence",
        "herding",
        "anchoring",
        "prospect theory",
        "anomaly",
    ),
    FinanceDomain.MACRO_ECONOMICS: (
        "interest rate",
        "gdp",
        "inflation",
        "monetary policy",
        "yield curve",
        "central bank",
        "fiscal",
    ),
}


def _detect_domain(text: str) -> FinanceDomain | None:
    """Best-effort domain detection from full paper text."""
    lower = text.lower()
    scores: dict[FinanceDomain, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in lower)
    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] >= 2 else None


# ── Extraction prompts ───────────────────────────────────────────────

_EXTRACTION_SYSTEM = """\
You are a financial knowledge extraction specialist.
Extract distinct, atomic factual claims from financial research text.
Rules: atomic (one fact per claim), self-contained, verifiable, include \
numbers when present.
Return ONLY a valid JSON array. No other text."""

_EXTRACTION_USER = """\
Paper: {title}
Data sample period: {sample_start} to {sample_end}

Extract all factual claims from this text:
{chunk_text}

JSON format per claim:
[{{"content": "complete sentence", \
"content_type": "FACT|DEFINITION|CLAIM|FORMULA|STATISTIC|METHODOLOGY", \
"temporal_reference": "time period as written in text or null", \
"market": "market name or null", \
"extraction_confidence": 0.0-1.0}}]"""


# ── Helpers ──────────────────────────────────────────────────────────


def _chunk_text(text: str, chunk_chars: int) -> list[str]:
    """Split text into roughly equal chunks at sentence boundaries."""
    if len(text) <= chunk_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= chunk_chars:
            chunks.append(remaining)
            break
        # Find the last sentence-ending punctuation before the limit.
        cut = chunk_chars
        for sep in (".\n", ". ", ".\t"):
            idx = remaining.rfind(sep, 0, chunk_chars)
            if idx != -1:
                cut = idx + len(sep)
                break
        chunks.append(remaining[:cut])
        remaining = remaining[cut:]
    return chunks


def _aware_or_now(value: datetime | None) -> datetime:
    """Normalise an optional datetime to UTC-aware."""
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _require_pdf(raw: Fetched) -> None:
    """Raise if the fetched content is not a PDF."""
    media_type = (raw.content_type or "").lower()
    if not media_type.startswith("application/pdf"):
        raise ValueError(
            f"TemporalPaperFlow requires a PDF; got {media_type!r}"
        )


# ── Knowledge-type mapping ───────────────────────────────────────────

_CONTENT_TYPE_MAP: dict[str, KnowledgeType] = {
    "fact": KnowledgeType.FACT,
    "definition": KnowledgeType.DEFINITION,
    "claim": KnowledgeType.CLAIM,
    "formula": KnowledgeType.FORMULA,
    "statistic": KnowledgeType.STATISTIC,
    "methodology": KnowledgeType.METHODOLOGY,
}


# ── TemporalPaperFlow ───────────────────────────────────────────────


class TemporalPaperFlow:
    """Config-bound flow producing a ``TKUCollection`` from one paper.

    ``TemporalPaperFlow(cfg)`` binds an immutable copy of the extraction
    config; ``build(input)`` fetches, parses, and extracts temporal
    knowledge units from each call's input.

    The flow is pure: no persistence, no retrieval, no library binding.

    Example::

        flow = TemporalPaperFlow(TemporalPaperCfg(model="gpt-4o-mini"))
        collection = await flow.build(ArxivIdentifier(id="2509.21507"))
        print(f"Extracted {len(collection.tkus)} TKUs")
    """

    __slots__ = ("_cfg", "_router")

    def __init__(self, cfg: TemporalPaperCfg) -> None:
        """Bind an immutable copy of the extraction config.

        Args:
            cfg: Temporal extraction settings.  Its fields control model
                selection, chunking, and deduplication thresholds.
        """
        self._cfg: TemporalPaperCfg = cfg.model_copy(deep=True)
        self._router: LLMRouter = _build_router(cfg)

    @property
    def cfg(self) -> TemporalPaperCfg:
        """The immutable extraction config bound to this flow."""
        return self._cfg

    @property
    def router(self) -> LLMRouter:
        """The LLM router used by this flow."""
        return self._router

    async def build(self, input: PaperInput) -> TKUCollection:
        """Build a ``TKUCollection`` from one paper input.

        Pipeline:
        1. Fetch the PDF (arXiv / HTTP / local).
        2. Parse to extract page text.
        3. Extract temporal metadata (sample period, dates).
        4. Extract TKUs section by section via LLM.
        5. Deduplicate semantically similar TKUs.
        6. Return the self-contained ``TKUCollection``.

        Args:
            input: A typed paper source (arXiv ID, URL, or local path).

        Returns:
            A ``TKUCollection`` containing all extracted TKUs with
            temporal validity windows and confidence scores.

        Raises:
            ValueError: If the input is not a PDF.
            NotImplementedError: If the input type is unsupported
                (e.g. DOI or RawText).
            RuntimeError: If all LLM tiers fail during extraction.
        """
        # Step 1 — Fetch.
        title, source_id, full_text = await _fetch_and_parse(input)
        _LOGGER.info("Fetched paper %r (%d chars).", title[:60], len(full_text))

        # Step 2 — Temporal metadata.
        raw_dates = extract_date_strings(full_text)
        date_ranges = [
            r for s in raw_dates if (r := normalize_date_range(s)) is not None
        ]
        sample_period = infer_sample_period(date_ranges)
        sample_start = (
            sample_period.start.strftime("%Y")
            if sample_period and sample_period.start
            else "unknown"
        )
        sample_end = (
            sample_period.end.strftime("%Y")
            if sample_period and sample_period.end
            else "present"
        )
        _LOGGER.info(
            "Temporal metadata: %d date strings, sample period %s–%s.",
            len(raw_dates),
            sample_start,
            sample_end,
        )

        # Step 3 — Domain detection.
        domain = _detect_domain(full_text)

        # Step 4 — Section-by-section extraction.
        all_tkus = await self._extract_tkus(
            full_text=full_text,
            title=title,
            source_id=source_id,
            sample_start=sample_start,
            sample_end=sample_end,
            sample_period=sample_period,
            domain=domain,
        )

        # Step 5 — Deduplication.
        deduped = await self._deduplicate(all_tkus)
        _LOGGER.info(
            "Extraction complete: %d TKUs (%d before dedup).",
            len(deduped),
            len(all_tkus),
        )

        # Step 6 — Build collection.
        return TKUCollection(
            document_id=source_id,
            source_title=title,
            tkus=tuple(deduped),
        )

    async def _extract_tkus(
        self,
        *,
        full_text: str,
        title: str,
        source_id: str,
        sample_start: str,
        sample_end: str,
        sample_period: DateRange | None,
        domain: FinanceDomain | None,
    ) -> list[TemporalKnowledgeUnit]:
        """Extract TKUs from text sections via LLM calls."""
        cfg = self._cfg
        # Split the full text into section-sized chunks.
        sections = _chunk_text(full_text, cfg.chunk_chars)
        all_tkus: list[TemporalKnowledgeUnit] = []

        for section_idx, section in enumerate(sections):
            if len(section.strip()) < cfg.min_section_length:
                continue

            prompt = _EXTRACTION_USER.format(
                title=title,
                sample_start=sample_start,
                sample_end=sample_end,
                chunk_text=section,
            )

            try:
                response = await self._router.generate(
                    TaskType.EXTRACTION,
                    prompt,
                    system_prompt=_EXTRACTION_SYSTEM,
                )
            except RuntimeError:
                _LOGGER.warning(
                    "LLM extraction failed for section %d of %r.",
                    section_idx,
                    title[:40],
                )
                continue

            parsed = safe_json_parse(response.text)
            if not isinstance(parsed, list):
                _LOGGER.debug(
                    "Non-array LLM response for section %d, skipping.",
                    section_idx,
                )
                continue

            section_tkus = _build_tkus_from_parsed(
                parsed,
                source_id=source_id,
                source_title=title,
                sample_period=sample_period,
                domain=domain,
                max_per_section=cfg.max_tkus_per_section,
            )
            all_tkus.extend(section_tkus)

        return all_tkus

    async def _deduplicate(
        self,
        tkus: list[TemporalKnowledgeUnit],
    ) -> list[TemporalKnowledgeUnit]:
        """Remove near-duplicate TKUs from the same source via embeddings."""
        if len(tkus) <= 1:
            return tkus

        threshold = self._cfg.semantic_similarity_threshold
        embeddings: dict[str, list[float]] = {}

        # Attempt to embed each TKU; skip embedding failures.
        for tku in tkus:
            if tku.tku_id not in embeddings:
                try:
                    embeddings[tku.tku_id] = await self._router.embed(
                        tku.content
                    )
                except RuntimeError:
                    _LOGGER.debug(
                        "Embedding failed for TKU %s, keeping it.",
                        tku.tku_id,
                    )

        # If we couldn't embed anything, skip deduplication.
        if not embeddings:
            return tkus

        kept: list[TemporalKnowledgeUnit] = []
        removed_ids: set[str] = set()

        for i, tku_a in enumerate(tkus):
            if tku_a.tku_id in removed_ids:
                continue
            kept.append(tku_a)

            # Only dedup within same source.
            emb_a = embeddings.get(tku_a.tku_id)
            if emb_a is None:
                continue

            for tku_b in tkus[i + 1 :]:
                if tku_b.tku_id in removed_ids:
                    continue
                if tku_b.source_id != tku_a.source_id:
                    continue
                emb_b = embeddings.get(tku_b.tku_id)
                if emb_b is None:
                    continue

                similarity = compute_cosine_similarity(emb_a, emb_b)
                if similarity >= threshold:
                    # Keep the one with higher confidence.
                    if (
                        tku_b.extraction_confidence
                        > tku_a.extraction_confidence
                    ):
                        kept[-1] = tku_b
                        removed_ids.add(tku_a.tku_id)
                        break
                    else:
                        removed_ids.add(tku_b.tku_id)

        return kept


# ── Module-private helpers ───────────────────────────────────────────


def _build_router(cfg: TemporalPaperCfg) -> LLMRouter:
    """Create the LLMRouter from the flow config."""
    ollama_cfg = OllamaModelCfg(generation_model=cfg.local_model)
    router_cfg = RouterCfg(
        ollama=ollama_cfg,
        api_cheap_model=cfg.model,
        api_powerful_model=cfg.model,
    )
    # If the user doesn't want local fallback, override default routing
    # to use API_CHEAP for extraction and summarization.
    if not cfg.use_local_if_available:
        from finsage.configs.ollama import ModelTier

        router_cfg = RouterCfg(
            ollama=ollama_cfg,
            api_cheap_model=cfg.model,
            api_powerful_model=cfg.model,
            default_routing={
                k: ModelTier.API_CHEAP.value for k in router_cfg.default_routing
            },
        )
    return LLMRouter(router_cfg)


async def _fetch_and_parse(
    input: PaperInput,
) -> tuple[str, str, str]:
    """Fetch and parse a paper into (title, source_id, full_text).

    Reuses the preprocess fetch and format layers directly.  The flow
    does not depend on ``PaperFlow`` internals.

    Returns:
        A 3-tuple of ``(title, source_id, full_text)``.

    Raises:
        ValueError: If the fetched content is not a PDF.
        NotImplementedError: For DOI inputs (no resolver yet).
        TypeError: For unsupported input variants.
    """
    if isinstance(input, ArxivIdentifier):
        raw: RawPaper = await fetch_arxiv(input.id)
        _require_pdf(raw)
        title = raw.title or f"arXiv:{input.id}"
        source_id = raw.arxiv_id or input.id
        parsed = await parse_pdf(raw.bytes)
        full_text = "\n\n".join(page.text for page in parsed.pages)
        return title, source_id, full_text

    if isinstance(input, HttpUrl):
        raw_http = await fetch_url(input.url)
        _require_pdf(raw_http)
        parsed = await parse_pdf(raw_http.bytes)
        full_text = "\n\n".join(page.text for page in parsed.pages)
        return input.url, input.url, full_text

    if isinstance(input, LocalFilePath):
        raw_local = await read_local_file(input.path)
        _require_pdf(raw_local)
        parsed = await parse_pdf(raw_local.bytes)
        full_text = "\n\n".join(page.text for page in parsed.pages)
        source_id = str(input.path)
        return source_id, source_id, full_text

    if isinstance(input, RawText):
        raise ValueError(
            "TemporalPaperFlow requires a PDF; RawText has no physical "
            "page evidence"
        )

    if isinstance(input, DoiIdentifier):
        raise NotImplementedError(
            "DOI inputs require an exact open PDF resolver before they "
            "can produce a paper source revision"
        )

    raise TypeError(f"Unsupported PaperInput variant: {type(input)!r}")


def _build_tkus_from_parsed(
    parsed_items: list[Any],
    *,
    source_id: str,
    source_title: str,
    sample_period: DateRange | None,
    domain: FinanceDomain | None,
    max_per_section: int,
) -> list[TemporalKnowledgeUnit]:
    """Convert parsed LLM JSON array items into TKU instances."""
    tkus: list[TemporalKnowledgeUnit] = []

    for item in parsed_items[:max_per_section]:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if not content or not isinstance(content, str):
            continue

        # Resolve content type.
        raw_type = str(item.get("content_type", "fact")).lower()
        content_type = _CONTENT_TYPE_MAP.get(raw_type, KnowledgeType.FACT)

        # Resolve extraction confidence.
        try:
            extraction_confidence = float(
                item.get("extraction_confidence", 0.8)
            )
            extraction_confidence = max(0.0, min(1.0, extraction_confidence))
        except (TypeError, ValueError):
            extraction_confidence = 0.8

        # Resolve temporal window from the LLM's temporal_reference.
        temporal_ref = item.get("temporal_reference")
        valid_from: datetime | None = None
        valid_until: datetime | None = None
        temporal_confidence = 0.50  # default: inferred from document

        if temporal_ref and isinstance(temporal_ref, str):
            dr = normalize_date_range(temporal_ref)
            if dr is not None:
                valid_from = dr.start
                valid_until = dr.end
                temporal_confidence = max(0.85, dr.confidence)
            else:
                # LLM gave a reference we can't parse — moderate confidence.
                temporal_confidence = 0.60
        elif sample_period is not None:
            # Inherit document-level sample period.
            valid_from = sample_period.start
            valid_until = sample_period.end
            temporal_confidence = min(0.55, sample_period.confidence * 0.6)

        # Resolve market.
        market = item.get("market")
        if market and not isinstance(market, str):
            market = None

        try:
            tku = TemporalKnowledgeUnit(
                content=content,
                content_type=content_type,
                source_id=source_id,
                source_title=source_title,
                valid_from=valid_from,
                valid_until=valid_until,
                temporal_confidence=temporal_confidence,
                extraction_confidence=extraction_confidence,
                market=market,
                domain=domain,
            )
            tkus.append(tku)
        except Exception:
            _LOGGER.debug(
                "Failed to build TKU from LLM item: %s",
                str(item)[:200],
            )
            continue

    return tkus
