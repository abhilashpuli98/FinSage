"""Configuration values for local and API-backed knowledge-pipeline models.

These immutable values describe model selection for the FinSage extraction
router. They do not configure the OpenAI Agents SDK runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    """Kinds of knowledge-pipeline work accepted by the LLM router."""

    SUMMARIZATION = "summarization"
    TAGGING = "tagging"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    VERIFICATION = "verification"
    EMBEDDING = "embedding"


class ModelTier(str, Enum):
    """Cost and capability tiers available to the LLM router."""

    LOCAL = "local"
    API_CHEAP = "api_cheap"
    API_POWERFUL = "api_powerful"


@dataclass(frozen=True)
class OllamaModelCfg:
    """Immutable connection and model settings for a local Ollama server."""

    base_url: str = "http://localhost:11434"
    generation_model: str = "llama3.2"
    embedding_model: str = "nomic-embed-text"
    timeout_seconds: int = 120
    max_retries: int = 3

    def is_available(self) -> bool:
        """Return whether the configured Ollama server responds to its tag API."""
        import httpx

        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
        except Exception:
            return False
        return response.status_code == 200


def _default_routing() -> dict[str, str]:
    """Build the default task-to-tier mapping for ``RouterCfg``."""
    return {
        TaskType.SUMMARIZATION.value: ModelTier.LOCAL.value,
        TaskType.TAGGING.value: ModelTier.LOCAL.value,
        TaskType.EXTRACTION.value: ModelTier.API_CHEAP.value,
        TaskType.GENERATION.value: ModelTier.API_POWERFUL.value,
        TaskType.VERIFICATION.value: ModelTier.API_CHEAP.value,
        TaskType.EMBEDDING.value: ModelTier.LOCAL.value,
    }


@dataclass(frozen=True)
class RouterCfg:
    """Immutable model-tier policy for knowledge-pipeline router calls."""

    ollama: OllamaModelCfg = field(default_factory=OllamaModelCfg)
    api_cheap_model: str = "gpt-4o-mini"
    api_powerful_model: str = "gpt-4o"
    default_routing: dict[str, str] = field(default_factory=_default_routing)
