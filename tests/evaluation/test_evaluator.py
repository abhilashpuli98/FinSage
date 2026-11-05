"""Unit tests for AutoEvaluator and statistical analysis."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pandas as pd

from finsage.configs.ollama import OllamaModelCfg, RouterCfg
from finsage.evaluation.evaluator import AutoEvaluator
from finsage.evaluation.models import (
    BenchmarkQuestion,
    EvalResult,
    QuestionType,
    SystemAnswer,
)
from finsage.knowledge.temporal import FinanceDomain
from finsage.mind.router import LLMRouter


class TestAutoEvaluator(unittest.IsolatedAsyncioTestCase):
    """Test AutoEvaluator scoring, caching, and paired statistics."""

    def setUp(self) -> None:
        self.router = LLMRouter(RouterCfg(ollama=OllamaModelCfg(max_retries=1)))
        self.systems = {
            "naive_rag": AsyncMock(),
            "finsage_full": AsyncMock(),
        }
        self.evaluator = AutoEvaluator(self.router, self.systems)

    async def test_score_factual_answer(self) -> None:
        """Factual questions are scored via judge prompt."""
        self.router.generate = AsyncMock(  # type: ignore[assignment]
            return_value=SimpleNamespace(
                text='{"score": 0.9, "reason": "Correct numbers"}'
            )
        )

        q = BenchmarkQuestion(
            question_id="q-fact-1",
            paper_id="paper-1",
            paper_title="Paper 1",
            domain=FinanceDomain.FACTOR_INVESTING,
            question_text="What is the monthly premium?",
            question_type=QuestionType.FACTUAL,
            ground_truth="1.2% per month",
        )
        ans = SystemAnswer(
            question_id="q-fact-1",
            system_name="finsage_full",
            answer_text="The monthly premium was 1.2%.",
        )

        res = await self.evaluator.score_answer(q, ans)
        self.assertIsInstance(res, EvalResult)
        self.assertEqual(res.score, 0.9)
        self.assertEqual(res.scorer, "llm_factual_judge")
        self.assertEqual(res.notes, "Correct numbers")

    async def test_score_temporal_answer(self) -> None:
        """Temporal questions use composite weighting (0.40 judge + 0.30 year + 0.30 scope)."""
        # Judge gives 1.0
        self.router.generate = AsyncMock(  # type: ignore[assignment]
            return_value=SimpleNamespace(
                text='{"score": 1.0, "reason": "Accurate"}'
            )
        )

        q = BenchmarkQuestion(
            question_id="q-temp-1",
            paper_id="paper-1",
            paper_title="Paper 1",
            domain=FinanceDomain.FACTOR_INVESTING,
            question_text="What was found as of 1990?",
            question_type=QuestionType.TEMPORAL,
            ground_truth="Positive momentum alpha",
            temporal_constraint=datetime(1990, 1, 1, tzinfo=timezone.utc),
        )
        # Answer contains year '1990' and non-empty temporal_scope
        ans = SystemAnswer(
            question_id="q-temp-1",
            system_name="finsage_full",
            answer_text="In 1990, researchers found positive momentum alpha.",
            temporal_scope="1963-1990 (US Equities)",
        )

        res = await self.evaluator.score_answer(q, ans)
        # 0.40 * 1.0 + 0.30 * 1.0 + 0.30 * 1.0 = 1.0
        self.assertAlmostEqual(res.score, 1.0)
        self.assertEqual(res.scorer, "temporal_composite_judge")

        # Now test an answer missing the year and scope
        ans_weak = SystemAnswer(
            question_id="q-temp-1",
            system_name="naive_rag",
            answer_text="Researchers found positive momentum alpha.",
            temporal_scope=None,
        )
        res_weak = await self.evaluator.score_answer(q, ans_weak)
        # 0.40 * 1.0 + 0.30 * 0.0 + 0.30 * 0.0 = 0.40
        self.assertAlmostEqual(res_weak.score, 0.40)

    async def test_score_analytical_answer(self) -> None:
        """Analytical questions score reasoning without requiring ground_truth."""
        self.router.generate = AsyncMock(  # type: ignore[assignment]
            return_value=SimpleNamespace(
                text='{"score": 0.85, "feedback": "Good multi-fact synthesis"}'
            )
        )

        q = BenchmarkQuestion(
            question_id="q-ana-1",
            paper_id="paper-1",
            paper_title="Paper 1",
            domain=FinanceDomain.FACTOR_INVESTING,
            question_text="Why did the factor decay after publication?",
            question_type=QuestionType.ANALYTICAL,
            ground_truth=None,
            expected_reasoning="Arbitrage capital inflows and post-publication crowding.",
        )
        ans = SystemAnswer(
            question_id="q-ana-1",
            system_name="finsage_full",
            answer_text="Arbitrageurs deployed capital following publication, compressing alpha.",
        )

        res = await self.evaluator.score_answer(q, ans)
        self.assertEqual(res.score, 0.85)
        self.assertEqual(res.scorer, "llm_analytical_judge")

    async def test_evaluate_all_with_caching(self) -> None:
        """evaluate_all executes systems and caches per (system, question_id)."""
        q = BenchmarkQuestion(
            question_id="q-cache-1",
            paper_id="p-1",
            paper_title="Paper 1",
            domain=FinanceDomain.FACTOR_INVESTING,
            question_text="Test question?",
            question_type=QuestionType.FACTUAL,
            ground_truth="Answer 1",
        )

        self.systems["naive_rag"].return_value = SystemAnswer(
            question_id="q-cache-1",
            system_name="naive_rag",
            answer_text="Naive Answer",
        )
        self.systems["finsage_full"].return_value = SystemAnswer(
            question_id="q-cache-1",
            system_name="finsage_full",
            answer_text="Full Answer",
        )

        self.router.generate = AsyncMock(  # type: ignore[assignment]
            return_value=SimpleNamespace(text='{"score": 0.8, "reason": "OK"}')
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            # Run 1: evaluates and writes cache
            df1 = await self.evaluator.evaluate_all([q], cache_dir)
            self.assertEqual(len(df1), 2)
            self.assertTrue((cache_dir / "naive_rag_q-cache-1.json").exists())
            self.assertTrue(
                (cache_dir / "finsage_full_q-cache-1.json").exists()
            )

            # Run 2: loads from cache, system callables should NOT be called again
            self.systems["naive_rag"].reset_mock()
            self.systems["finsage_full"].reset_mock()

            df2 = await self.evaluator.evaluate_all([q], cache_dir)
            self.assertEqual(len(df2), 2)
            self.systems["naive_rag"].assert_not_called()
            self.systems["finsage_full"].assert_not_called()

    def test_statistical_analysis(self) -> None:
        """statistical_analysis returns paired t-test p-values and Cohen's d."""
        data = [
            # 5 questions, finsage_full scores higher on all
            {"system": "finsage_full", "question_id": "q1", "score": 0.95},
            {"system": "finsage_full", "question_id": "q2", "score": 0.90},
            {"system": "finsage_full", "question_id": "q3", "score": 0.85},
            {"system": "finsage_full", "question_id": "q4", "score": 0.92},
            {"system": "finsage_full", "question_id": "q5", "score": 0.88},
            {"system": "naive_rag", "question_id": "q1", "score": 0.60},
            {"system": "naive_rag", "question_id": "q2", "score": 0.55},
            {"system": "naive_rag", "question_id": "q3", "score": 0.50},
            {"system": "naive_rag", "question_id": "q4", "score": 0.65},
            {"system": "naive_rag", "question_id": "q5", "score": 0.52},
        ]
        df = pd.DataFrame(data)

        stats = self.evaluator.statistical_analysis(df)
        self.assertIn("finsage_full", stats)
        self.assertIn("naive_rag", stats)

        full_stats = stats["finsage_full"]
        self.assertEqual(full_stats["mean_score"], 0.9)
        self.assertEqual(full_stats["p_value_vs_finsage"], 1.0)
        self.assertEqual(full_stats["cohens_d"], 0.0)

        naive_stats = stats["naive_rag"]
        self.assertAlmostEqual(naive_stats["mean_score"], 0.564, places=2)
        # p-value should be statistically significant (p < 0.01)
        self.assertLess(naive_stats["p_value_vs_finsage"], 0.01)
        # Cohen's d > 0.8 indicates large effect size
        self.assertGreater(naive_stats["cohens_d"], 0.8)


if __name__ == "__main__":
    unittest.main()
