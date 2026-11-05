"""Offline deterministic tests for FinSage LLM routing."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from finsage.configs.ollama import (
    ModelTier,
    OllamaModelCfg,
    RouterCfg,
    TaskType,
)
from finsage.mind.router import (
    CostTracker,
    LLMResponse,
    LLMRouter,
    safe_json_parse,
)


def _completion(text: str, prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build the LiteLLM-shaped object needed by the router boundary."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


class LLMRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_summarization_routes_to_local_by_default(self) -> None:
        router = LLMRouter(RouterCfg())
        completion = AsyncMock(return_value=_completion("local summary"))

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=True),
            patch("finsage.mind.router.acompletion", new=completion),
        ):
            response = await router.generate(
                TaskType.SUMMARIZATION,
                "Summarize this paper.",
            )

        self.assertEqual(response.text, "local summary")
        self.assertEqual(response.tier, ModelTier.LOCAL)
        self.assertEqual(response.model_used, "ollama/llama3.2")
        self.assertEqual(completion.await_args.kwargs["temperature"], 0.0)
        self.assertEqual(
            completion.await_args.kwargs["api_base"],
            "http://localhost:11434",
        )

    async def test_generation_routes_to_powerful_api_by_default(self) -> None:
        router = LLMRouter(RouterCfg())
        completion = AsyncMock(return_value=_completion("generated analysis"))

        with patch("finsage.mind.router.acompletion", new=completion):
            response = await router.generate(
                TaskType.GENERATION, "Write analysis"
            )

        self.assertEqual(response.tier, ModelTier.API_POWERFUL)
        self.assertEqual(response.model_used, "gpt-4o")
        self.assertEqual(completion.await_args.kwargs["temperature"], 0.7)

    async def test_local_unavailability_falls_back_to_cheap_api(self) -> None:
        router = LLMRouter(RouterCfg(ollama=OllamaModelCfg(max_retries=1)))
        completion = AsyncMock(return_value=_completion("fallback summary"))

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=completion),
        ):
            response = await router.generate(
                TaskType.SUMMARIZATION,
                "Summarize this paper.",
            )

        self.assertEqual(response.tier, ModelTier.API_CHEAP)
        self.assertEqual(response.model_used, "gpt-4o-mini")
        completion.assert_awaited_once()

    async def test_all_failed_generation_tiers_raise_clear_error(self) -> None:
        router = LLMRouter(RouterCfg(ollama=OllamaModelCfg(max_retries=1)))
        completion = AsyncMock(side_effect=ConnectionError("API offline"))

        with (
            patch.object(OllamaModelCfg, "is_available", return_value=False),
            patch("finsage.mind.router.acompletion", new=completion),
        ):
            with self.assertRaisesRegex(RuntimeError, "all LLM tiers failed"):
                await router.generate(TaskType.SUMMARIZATION, "Summarize")

        self.assertEqual(completion.await_count, 2)

    async def test_embed_falls_back_when_local_embedding_fails(self) -> None:
        router = LLMRouter(RouterCfg())
        embedding = AsyncMock(
            side_effect=[
                ConnectionError("Ollama offline"),
                {"data": [{"embedding": [0.25, -0.5]}]},
            ]
        )

        with patch("finsage.mind.router.aembedding", new=embedding):
            result = await router.embed("financial text")

        self.assertEqual(result, [0.25, -0.5])
        self.assertEqual(embedding.await_count, 2)
        self.assertEqual(
            embedding.await_args_list[0].kwargs["model"],
            "ollama/nomic-embed-text",
        )
        self.assertEqual(
            embedding.await_args_list[1].kwargs["model"],
            "text-embedding-3-small",
        )


class CostTrackerTests(unittest.TestCase):
    def test_report_aggregates_calls_tokens_and_cost(self) -> None:
        tracker = CostTracker()
        tracker.track(
            TaskType.EXTRACTION,
            LLMResponse(
                text="first",
                model_used="gpt-4o-mini",
                tier=ModelTier.API_CHEAP,
                prompt_tokens=10,
                completion_tokens=5,
                estimated_cost_usd=0.001,
                latency_seconds=0.1,
            ),
        )
        tracker.track(
            TaskType.EXTRACTION,
            LLMResponse(
                text="second",
                model_used="gpt-4o-mini",
                tier=ModelTier.API_CHEAP,
                prompt_tokens=20,
                completion_tokens=10,
                estimated_cost_usd=0.002,
                latency_seconds=0.2,
            ),
        )

        report = tracker.get_report()[TaskType.EXTRACTION.value]

        self.assertEqual(report["model"], "gpt-4o-mini")
        self.assertEqual(report["total_calls"], 2)
        self.assertEqual(report["total_tokens"], 45)
        self.assertAlmostEqual(float(report["cost_usd"]), 0.003)

    def test_reset_discards_accumulated_records(self) -> None:
        tracker = CostTracker()
        tracker.track(
            TaskType.TAGGING,
            LLMResponse(
                "tag", "ollama/llama3.2", ModelTier.LOCAL, 1, 1, 0.0, 0.1
            ),
        )

        tracker.reset()

        self.assertEqual(tracker.get_report(), {})


class SafeJsonParseTests(unittest.TestCase):
    def test_parses_clean_json(self) -> None:
        self.assertEqual(safe_json_parse('{"score": 0.9}'), {"score": 0.9})

    def test_parses_fenced_json(self) -> None:
        self.assertEqual(
            safe_json_parse("```json\\n[1, 2]\\n```"),
            [1, 2],
        )

    def test_parses_bare_json_prefix(self) -> None:
        self.assertEqual(
            safe_json_parse('json {"valid": true}'), {"valid": True}
        )

    def test_parses_json_embedded_in_explanation(self) -> None:
        self.assertEqual(
            safe_json_parse('Result: {"tags": ["factor"]} Thanks.'),
            {"tags": ["factor"]},
        )

    def test_returns_none_for_non_json_text(self) -> None:
        self.assertIsNone(
            safe_json_parse("The model did not return structured output.")
        )
