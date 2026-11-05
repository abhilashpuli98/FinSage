"""Minimal TemporalPaperFlow example — extract TKUs from a paper.

Demonstrates extracting a TKUCollection with temporal validity windows
and confidence metrics. Runs with live arXiv/LLM if configured, or with
mocked input for offline deterministic execution.
"""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from finsage.configs import TemporalPaperCfg
from finsage.configs.paper import ArxivIdentifier
from finsage.flows import TemporalPaperFlow


async def _run_live() -> None:
    """Run flow against live arXiv and LLM."""
    cfg = TemporalPaperCfg(model="gpt-4o-mini")
    flow = TemporalPaperFlow(cfg)
    collection = await flow.build(ArxivIdentifier(id="2509.21507"))
    _print_collection(collection, flow)


async def _run_mocked() -> None:
    """Run flow with mocked paper fetch and LLM responses."""
    print(
        "Running in demonstration mode (mocked arXiv fetch & LLM extraction)..."
    )
    mock_pdf = MagicMock()
    page1 = MagicMock()
    page1.text = (
        "Abstract: We study the momentum factor premium in US equities from "
        "January 1963 to December 2022. Our analysis shows the momentum factor "
        "earns 1.2% monthly premium. The Fama-French model explains factor returns."
    )
    mock_pdf.pages = [page1]

    raw_paper = MagicMock()
    raw_paper.title = "Momentum Factor Analysis in US Equities"
    raw_paper.arxiv_id = "2509.21507"
    raw_paper.content_type = "application/pdf"
    raw_paper.bytes = b"%PDF-1.4 mock"

    llm_output = json.dumps(
        [
            {
                "content": "Momentum factor earns 1.2% monthly premium in US equities.",
                "content_type": "STATISTIC",
                "temporal_reference": "1963-2022",
                "market": "US Equities",
                "extraction_confidence": 0.92,
            },
            {
                "content": "The Fama-French model explains factor returns.",
                "content_type": "CLAIM",
                "temporal_reference": None,
                "market": None,
                "extraction_confidence": 0.85,
            },
        ]
    )

    flow = TemporalPaperFlow(TemporalPaperCfg())

    with (
        patch(
            "finsage.flows.temporal_paper.fetch_arxiv",
            new=AsyncMock(return_value=raw_paper),
        ),
        patch(
            "finsage.flows.temporal_paper.parse_pdf",
            new=AsyncMock(return_value=mock_pdf),
        ),
    ):

        async def mock_generate(
            *args: object, **kwargs: object
        ) -> SimpleNamespace:
            return SimpleNamespace(text=llm_output)

        async def mock_embed(text: str) -> list[float]:
            return [0.1, 0.2, 0.3, 0.4, 0.5]

        flow.router.generate = mock_generate  # type: ignore[assignment]
        flow.router.embed = mock_embed  # type: ignore[assignment]

        collection = await flow.build(ArxivIdentifier(id="2509.21507"))
        _print_collection(collection, flow)


def _print_collection(collection: object, flow: TemporalPaperFlow) -> None:
    """Print extracted collection details."""
    from finsage.knowledge.temporal import TKUCollection

    assert isinstance(collection, TKUCollection)
    print(
        f"Extracted {len(collection.tkus)} TKUs from '{collection.source_title}'"
    )
    for tku in collection.tkus[:5]:
        print(f"  [{tku.content_type.value}] {tku.content[:80]}...")
        print(
            f"    validity: {tku.temporal_scope()}  confidence: {tku.composite_confidence():.2f}"
        )

    # Cost report from the router.
    report = flow.router.cost_tracker.get_report()
    print(f"\nCost report: {report}")


async def main() -> None:
    """Run with live services if available, otherwise demonstration mode."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            await _run_live()
            return
        except Exception as exc:
            print(
                f"Live run encountered error: {exc}. Falling back to demonstration."
            )

    await _run_mocked()


if __name__ == "__main__":
    asyncio.run(main())
