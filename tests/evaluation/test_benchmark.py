"""Unit tests for FinBench-50 BenchmarkConstructor."""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from finsage.configs.ollama import OllamaModelCfg, RouterCfg
from finsage.evaluation.benchmark import (
    BenchmarkConstructor,
    _parse_target_date,
)
from finsage.evaluation.models import QuestionType
from finsage.knowledge.temporal import (
    FinanceDomain,
    KnowledgeType,
    TemporalKnowledgeUnit,
    TKUCollection,
)
from finsage.mind.router import LLMRouter


def _make_sample_collection() -> TKUCollection:
    tku1 = TemporalKnowledgeUnit(
        content="Momentum factor earns 1.2% monthly premium in US equities.",
        content_type=KnowledgeType.STATISTIC,
        source_id="paper-001",
        source_title="Test Paper",
        domain=FinanceDomain.FACTOR_INVESTING,
        valid_from=datetime(1963, 1, 1, tzinfo=timezone.utc),
        valid_until=datetime(2022, 12, 31, tzinfo=timezone.utc),
        extraction_confidence=0.95,
        temporal_confidence=0.90,
    )
    tku2 = TemporalKnowledgeUnit(
        content="The Fama-French three-factor model explains return variance.",
        content_type=KnowledgeType.CLAIM,
        source_id="paper-001",
        source_title="Test Paper",
        domain=FinanceDomain.FACTOR_INVESTING,
        extraction_confidence=0.88,
        temporal_confidence=0.85,
    )
    return TKUCollection(
        document_id="paper-001",
        source_title="Test Paper",
        tkus=(tku1, tku2),
    )


class TestBenchmarkConstructor(unittest.IsolatedAsyncioTestCase):
    """Test BenchmarkConstructor question generation and caching."""

    def setUp(self) -> None:
        self.router = LLMRouter(RouterCfg(ollama=OllamaModelCfg(max_retries=1)))
        self.constructor = BenchmarkConstructor(self.router)

    def test_parse_target_date(self) -> None:
        """Target date parsing handles various ISO formats."""
        dt1 = _parse_target_date("1990-01-01")
        self.assertIsNotNone(dt1)
        self.assertEqual(dt1.year, 1990)

        dt2 = _parse_target_date("2020")
        self.assertIsNotNone(dt2)
        self.assertEqual(dt2.year, 2020)

        self.assertIsNone(_parse_target_date(None))
        self.assertIsNone(_parse_target_date("invalid-date"))

    async def test_generate_questions_structure(self) -> None:
        """generate_questions produces 6 questions with required type breakdown."""
        factual_json = json.dumps(
            [
                {
                    "question": "What is the monthly momentum premium?",
                    "ground_truth": "1.2%",
                    "difficulty": "easy",
                },
                {
                    "question": "What model explains return variance?",
                    "ground_truth": "Fama-French",
                    "difficulty": "medium",
                },
                {
                    "question": "What market was examined?",
                    "ground_truth": "US Equities",
                    "difficulty": "easy",
                },
            ]
        )
        analytical_json = json.dumps(
            [
                {
                    "question": "Why does momentum persist across regimes?",
                    "expected_reasoning": "Risk premium vs behavioral delay",
                },
                {
                    "question": "How do transaction costs affect returns?",
                    "expected_reasoning": "High turnover reduces net alpha",
                },
            ]
        )
        temporal_json = json.dumps(
            {
                "question": "As of 1963, what was the established factor finding?",
                "target_date": "1963-01-01",
                "ground_truth": "Momentum premium initiation period",
                "difficulty": "medium",
            }
        )

        responses = [
            SimpleNamespace(text=factual_json),
            SimpleNamespace(text=analytical_json),
            SimpleNamespace(text=temporal_json),
        ]
        self.router.generate = AsyncMock(side_effect=responses)  # type: ignore[assignment]

        coll = _make_sample_collection()
        questions = await self.constructor.generate_questions(
            parsed_text="Full text of test paper.",
            tku_collection=coll,
            paper_id="paper-001",
            paper_title="Test Paper",
            domain=FinanceDomain.FACTOR_INVESTING,
            sample_period_start="1963",
            sample_period_end="2022",
        )

        self.assertEqual(len(questions), 6)
        factual_qs = [
            q for q in questions if q.question_type == QuestionType.FACTUAL
        ]
        analytical_qs = [
            q for q in questions if q.question_type == QuestionType.ANALYTICAL
        ]
        temporal_qs = [
            q for q in questions if q.question_type == QuestionType.TEMPORAL
        ]

        self.assertEqual(len(factual_qs), 3)
        self.assertEqual(len(analytical_qs), 2)
        self.assertEqual(len(temporal_qs), 1)

        # Analytical questions must have ground_truth=None
        for aq in analytical_qs:
            self.assertIsNone(aq.ground_truth)
            self.assertIsNotNone(aq.expected_reasoning)

        # Factual and Temporal must have non-empty ground_truth
        for fq in factual_qs:
            self.assertIsNotNone(fq.ground_truth)
        self.assertIsNotNone(temporal_qs[0].ground_truth)
        self.assertIsNotNone(temporal_qs[0].temporal_constraint)

    async def test_fallback_when_llm_fails(self) -> None:
        """Deterministic fallback produces 6 questions when router fails."""
        self.router.generate = AsyncMock(
            side_effect=RuntimeError("Router offline")
        )  # type: ignore[assignment]

        coll = _make_sample_collection()
        questions = await self.constructor.generate_questions(
            parsed_text="Fallback paper text content.",
            tku_collection=coll,
            paper_id="paper-002",
            paper_title="Fallback Paper",
            domain=FinanceDomain.OPTIONS_PRICING,
            sample_period_start="1990",
            sample_period_end="2020",
        )

        self.assertEqual(len(questions), 6)
        self.assertEqual(
            sum(
                1 for q in questions if q.question_type == QuestionType.FACTUAL
            ),
            3,
        )
        self.assertEqual(
            sum(
                1
                for q in questions
                if q.question_type == QuestionType.ANALYTICAL
            ),
            2,
        )
        self.assertEqual(
            sum(
                1 for q in questions if q.question_type == QuestionType.TEMPORAL
            ),
            1,
        )

    async def test_build_finbench_50_caching(self) -> None:
        """build_finbench_50 persists questions and skips cached paper_ids."""
        self.constructor.generate_questions = AsyncMock()  # type: ignore[assignment]
        mock_q = MagicMock()
        mock_q.paper_id = "paper-cached"
        mock_q.model_dump.return_value = {
            "question_id": "q-1",
            "paper_id": "paper-cached",
            "paper_title": "Cached Paper",
            "domain": "FACTOR_INVESTING",
            "question_text": "Is momentum significant?",
            "question_type": "factual",
            "ground_truth": "Yes",
            "temporal_constraint": None,
            "difficulty": "easy",
            "expected_reasoning": None,
            "created_at": "2026-09-15T00:00:00Z",
        }
        self.constructor.generate_questions.return_value = [mock_q]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "finbench.json"

            paper_list = [
                {
                    "paper_id": "paper-cached",
                    "title": "Cached Paper",
                    "domain": "FACTOR_INVESTING",
                },
            ]

            # First run: writes out
            res1 = await self.constructor.build_finbench_50(
                paper_list, out_file
            )
            self.assertEqual(len(res1), 1)
            self.assertTrue(out_file.exists())

            # Second run: loads from cache, generate_questions should NOT be called again
            self.constructor.generate_questions.reset_mock()
            res2 = await self.constructor.build_finbench_50(
                paper_list, out_file
            )
            self.assertEqual(len(res2), 1)
            self.constructor.generate_questions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
