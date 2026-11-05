"""Unit tests for LLMQualityBenchmark (Phase 8)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from finsage.configs.ollama import OllamaModelCfg, RouterCfg
from finsage.evaluation.llm_benchmark import (
    LLMQualityBenchmark,
    _compute_bertscore_or_fallback,
    _error_row,
)
from finsage.mind.router import LLMRouter


def _make_mock_response(text: str, cost: float = 0.001) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        estimated_cost_usd=cost,
        prompt_tokens=50,
        completion_tokens=30,
        model_used="mock-model",
    )


class TestLLMQualityBenchmark(unittest.IsolatedAsyncioTestCase):
    """Test LLMQualityBenchmark extraction, tagging, and summarization."""

    def setUp(self) -> None:
        self.router = LLMRouter(RouterCfg(ollama=OllamaModelCfg(max_retries=1)))
        self.benchmark = LLMQualityBenchmark(
            self.router,
            models=("mock-local", "mock-api"),
        )

    async def test_extraction_benchmark_returns_results(self) -> None:
        """Extraction benchmark returns results for each model × example."""
        extract_response = json.dumps(
            [
                "The annualised momentum premium is 8.3%.",
                "The t-statistic for the momentum premium is 4.1.",
            ]
        )
        # First call: extraction, Second call: scoring judge.
        responses = [
            _make_mock_response(extract_response),
            _make_mock_response('{"score": 0.9, "reason": "Good extraction"}'),
        ]
        call_idx = {"i": 0}

        async def mock_generate(*args, **kwargs):  # type: ignore[no-untyped-def]
            idx = call_idx["i"]
            call_idx["i"] = idx + 1
            return responses[idx % len(responses)]

        self.router.generate = mock_generate  # type: ignore[assignment]

        results = await self.benchmark.run_extraction_benchmark()
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["task"], "extraction")
            self.assertIn("score", r)
            self.assertIn("latency_s", r)
            self.assertIn("model", r)

    async def test_tagging_benchmark_exact_match(self) -> None:
        """Tagging benchmark scores exact domain match as 1.0."""
        self.router.generate = AsyncMock(  # type: ignore[assignment]
            return_value=_make_mock_response("FACTOR_INVESTING")
        )

        results = await self.benchmark.run_tagging_benchmark()
        self.assertGreater(len(results), 0)
        # At least the first gold example expects FACTOR_INVESTING.
        factor_results = [r for r in results if r["score"] == 1.0]
        self.assertGreater(len(factor_results), 0)

    async def test_tagging_benchmark_wrong_domain(self) -> None:
        """Tagging benchmark scores wrong domain as 0.0."""
        self.router.generate = AsyncMock(  # type: ignore[assignment]
            return_value=_make_mock_response("CREDIT_RISK")
        )

        results = await self.benchmark.run_tagging_benchmark()
        # CREDIT_RISK won't match FACTOR_INVESTING for the first example.
        first_result = results[0]
        self.assertEqual(first_result["score"], 0.0)

    async def test_summarization_benchmark_runs(self) -> None:
        """Summarization benchmark executes and returns scores."""
        self.router.generate = AsyncMock(  # type: ignore[assignment]
            return_value=_make_mock_response(
                "Momentum is pervasive across asset classes with 8.3% returns."
            )
        )

        results = await self.benchmark.run_summarization_benchmark()
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["task"], "summarization")
            self.assertGreaterEqual(r["score"], 0.0)
            self.assertLessEqual(r["score"], 1.0)

    async def test_run_all_benchmarks_saves_files(self) -> None:
        """run_all_benchmarks creates CSV and raw JSON files."""
        self.router.generate = AsyncMock(  # type: ignore[assignment]
            return_value=_make_mock_response("FACTOR_INVESTING")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            df = await self.benchmark.run_all_benchmarks(output_dir)

            self.assertFalse(df.empty)
            self.assertTrue((output_dir / "llm_benchmark_raw.json").exists())
            self.assertTrue((output_dir / "llm_benchmark_results.csv").exists())

            # Check DataFrame columns.
            for col in ("task", "model", "mean_score"):
                self.assertIn(col, df.columns)

    async def test_model_failure_produces_error_row(self) -> None:
        """LLM failures produce error rows with score=0.0."""
        self.router.generate = AsyncMock(  # type: ignore[assignment]
            side_effect=RuntimeError("All tiers failed")
        )

        results = await self.benchmark.run_extraction_benchmark()
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r["score"], 0.0)
            self.assertIn("error", r)


class TestBertscoreFallback(unittest.TestCase):
    """Test BERTScore fallback to token overlap."""

    def test_token_overlap_identical(self) -> None:
        """Identical strings produce score ≈ 1.0."""
        score = _compute_bertscore_or_fallback("hello world", "hello world")
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_token_overlap_partial(self) -> None:
        """Partially overlapping strings produce score in (0, 1)."""
        score = _compute_bertscore_or_fallback(
            "momentum premium is significant",
            "momentum premium is negligible in this period",
        )
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_token_overlap_empty(self) -> None:
        """Empty reference produces score = 0.0."""
        score = _compute_bertscore_or_fallback("some text", "")
        self.assertAlmostEqual(score, 0.0)


class TestErrorRow(unittest.TestCase):
    """Test error row helper."""

    def test_error_row_structure(self) -> None:
        row = _error_row("extraction", "gpt-4o", 1.5, "timeout")
        self.assertEqual(row["task"], "extraction")
        self.assertEqual(row["model"], "gpt-4o")
        self.assertEqual(row["score"], 0.0)
        self.assertEqual(row["error"], "timeout")


if __name__ == "__main__":
    unittest.main()
