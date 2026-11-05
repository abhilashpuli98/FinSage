"""Tests for scripts/run_finsage_eval.py end-to-end evaluation runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from finsage.configs.ollama import ModelTier, RouterCfg
from finsage.mind.router import CostTracker, LLMResponse, LLMRouter


def _make_mock_router() -> LLMRouter:
    """Create a mocked LLMRouter that returns deterministic responses."""
    router = LLMRouter(RouterCfg())
    mock_response = LLMResponse(
        text='{"score": 0.8, "reason": "partially correct"}',
        model_used="gpt-4o-mini",
        tier=ModelTier.API_CHEAP,
        prompt_tokens=100,
        completion_tokens=50,
        estimated_cost_usd=0.0001,
        latency_seconds=0.5,
    )
    router.generate = AsyncMock(return_value=mock_response)
    router.embed = AsyncMock(return_value=[0.1] * 128)
    router.cost_tracker = CostTracker()
    return router


class TestRunFinsageEvalImports(unittest.TestCase):
    """Verify that the evaluation runner can be imported."""

    def test_import_runner(self) -> None:
        """The runner module should import without errors."""
        import scripts.run_finsage_eval as runner  # noqa: F811

        self.assertTrue(hasattr(runner, "main"))
        self.assertTrue(hasattr(runner, "run_full_evaluation"))
        self.assertTrue(hasattr(runner, "run_ablation_study"))
        self.assertTrue(hasattr(runner, "run_cost_benchmark"))

    def test_argument_parser(self) -> None:
        """CLI parser should accept expected arguments."""
        import scripts.run_finsage_eval as runner

        parser = runner._build_argument_parser()
        args = parser.parse_args(
            ["--phase", "cost", "--output", "results/test/"]
        )
        self.assertEqual(args.phase, "cost")
        self.assertEqual(args.output, Path("results/test/"))


class TestFallbackQuestions(unittest.TestCase):
    """Test the fallback question generator for smoke tests."""

    def test_fallback_produces_6_questions(self) -> None:
        """Fallback should produce exactly 6 questions."""
        import scripts.run_finsage_eval as runner

        questions = runner._build_fallback_questions()
        self.assertEqual(len(questions), 6)

    def test_fallback_has_correct_types(self) -> None:
        """Fallback questions should cover factual, analytical, temporal."""
        import scripts.run_finsage_eval as runner

        questions = runner._build_fallback_questions()
        types = [q["question_type"] for q in questions]
        self.assertEqual(types.count("FACTUAL"), 3)
        self.assertEqual(types.count("ANALYTICAL"), 2)
        self.assertEqual(types.count("TEMPORAL"), 1)

    def test_fallback_analytical_has_no_ground_truth(self) -> None:
        """Analytical questions should have None ground_truth."""
        import scripts.run_finsage_eval as runner

        questions = runner._build_fallback_questions()
        analytical = [
            q for q in questions if q["question_type"] == "ANALYTICAL"
        ]
        for q in analytical:
            self.assertIsNone(q["ground_truth"])


class TestPrintSummary(unittest.TestCase):
    """Test the summary printer."""

    def test_empty_dataframe(self) -> None:
        """Should handle empty DataFrame without error."""
        import pandas as pd

        import scripts.run_finsage_eval as runner

        df = pd.DataFrame()
        # Should not raise.
        runner._print_summary("Test", df)

    def test_summary_with_data(self) -> None:
        """Should format summary table correctly."""
        import pandas as pd

        import scripts.run_finsage_eval as runner

        df = pd.DataFrame(
            {
                "system": [
                    "finsage_full",
                    "naive_rag",
                    "finsage_full",
                    "naive_rag",
                ],
                "score": [0.9, 0.5, 0.8, 0.6],
                "question_type": ["FACTUAL", "FACTUAL", "TEMPORAL", "TEMPORAL"],
            }
        )
        # Should not raise.
        runner._print_summary("Test Summary", df)


class TestGeneratePaperFiguresImport(unittest.TestCase):
    """Verify figure generation script imports."""

    def test_import_figure_generator(self) -> None:
        """The figure generation module should import without errors."""
        import scripts.generate_paper_figures as figs

        self.assertTrue(hasattr(figs, "main"))
        self.assertTrue(hasattr(figs, "generate_architecture_diagram"))
        self.assertTrue(hasattr(figs, "generate_conflict_graph"))
        self.assertTrue(hasattr(figs, "generate_temporal_improvement"))

    def test_architecture_diagram_generates(self) -> None:
        """Figure 1 should be created as a PNG file."""
        import scripts.generate_paper_figures as figs

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "architecture.png"
            figs.generate_architecture_diagram(out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)

    def test_conflict_graph_generates(self) -> None:
        """Figure 3 should be created as a PNG file."""
        import scripts.generate_paper_figures as figs

        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "conflict.png"
            figs.generate_conflict_graph(out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)

    def test_temporal_improvement_illustrative(self) -> None:
        """Figure 4 fallback (no results) should generate illustrative data."""
        import scripts.generate_paper_figures as figs

        with tempfile.TemporaryDirectory() as tmpdir:
            results = Path(tmpdir) / "results"
            results.mkdir()
            out = Path(tmpdir) / "temporal.png"
            figs.generate_temporal_improvement(results, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
