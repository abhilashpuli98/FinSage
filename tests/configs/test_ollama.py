"""Tests for FinSage local-model router configuration."""

import unittest
from unittest.mock import MagicMock, patch

from finsage.configs.ollama import (
    ModelTier,
    OllamaModelCfg,
    RouterCfg,
    TaskType,
)


class OllamaModelCfgTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = OllamaModelCfg()

        self.assertEqual(cfg.base_url, "http://localhost:11434")
        self.assertEqual(cfg.generation_model, "llama3.2")
        self.assertEqual(cfg.embedding_model, "nomic-embed-text")
        self.assertEqual(cfg.timeout_seconds, 120)
        self.assertEqual(cfg.max_retries, 3)

    def test_is_available_returns_true_for_successful_tags_response(
        self,
    ) -> None:
        response = MagicMock(status_code=200)
        with patch("httpx.get", return_value=response) as get:
            self.assertTrue(OllamaModelCfg().is_available())

        get.assert_called_once_with(
            "http://localhost:11434/api/tags", timeout=5.0
        )

    def test_is_available_returns_false_when_request_fails(self) -> None:
        with patch("httpx.get", side_effect=OSError("offline")):
            self.assertFalse(OllamaModelCfg().is_available())


class RouterCfgTests(unittest.TestCase):
    def test_defaults_include_the_documented_task_routing(self) -> None:
        cfg = RouterCfg()

        self.assertEqual(
            cfg.default_routing[TaskType.SUMMARIZATION.value],
            ModelTier.LOCAL.value,
        )
        self.assertEqual(
            cfg.default_routing[TaskType.EXTRACTION.value],
            ModelTier.API_CHEAP.value,
        )
        self.assertEqual(
            cfg.default_routing[TaskType.GENERATION.value],
            ModelTier.API_POWERFUL.value,
        )
        self.assertEqual(
            cfg.default_routing[TaskType.EMBEDDING.value],
            ModelTier.LOCAL.value,
        )

    def test_default_routing_is_not_shared_between_instances(self) -> None:
        first = RouterCfg()
        second = RouterCfg()

        first.default_routing[TaskType.TAGGING.value] = (
            ModelTier.API_CHEAP.value
        )

        self.assertEqual(
            second.default_routing[TaskType.TAGGING.value],
            ModelTier.LOCAL.value,
        )

    def test_models_are_frozen(self) -> None:
        cfg = RouterCfg()
        with self.assertRaises(AttributeError):
            cfg.api_cheap_model = "another-model"  # type: ignore[misc]
