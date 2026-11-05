"""LLM-agnostic routing for FinSage knowledge-pipeline tasks.

This module routes extraction, tagging, summarization, and embedding calls
through LiteLLM. It deliberately does not replace or wrap the OpenAI Agents
SDK runtime used by the rest of FinSage.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast

from litellm import acompletion, aembedding

from finsage.configs.ollama import ModelTier, RouterCfg, TaskType

_LOGGER = logging.getLogger(__name__)

_CHEAP_INPUT_COST_PER_MILLION = 0.15
_CHEAP_OUTPUT_COST_PER_MILLION = 0.60
_POWERFUL_INPUT_COST_PER_MILLION = 2.50
_POWERFUL_OUTPUT_COST_PER_MILLION = 10.00


@dataclass(frozen=True)
class LLMResponse:
    """A normalized result from one routed generation request."""

    text: str
    model_used: str
    tier: ModelTier
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    latency_seconds: float


class CostTracker:
    """In-memory accounting for LLM responses handled by one router instance."""

    def __init__(self) -> None:
        """Create an empty response ledger."""
        self._records: list[tuple[TaskType, LLMResponse]] = []

    def track(self, task_type: TaskType, response: LLMResponse) -> None:
        """Add one successful routed response to the ledger."""
        self._records.append((task_type, response))

    def get_report(self) -> dict[str, dict[str, str | int | float]]:
        """Return call, token, cost, and model totals grouped by task type."""
        report: dict[str, dict[str, str | int | float]] = {}
        for task_type, response in self._records:
            key = task_type.value
            if key not in report:
                report[key] = {
                    "model": response.model_used,
                    "total_calls": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }
            bucket = report[key]
            if bucket["model"] != response.model_used:
                bucket["model"] = "multiple"
            bucket["total_calls"] = int(bucket["total_calls"]) + 1
            bucket["total_tokens"] = int(bucket["total_tokens"]) + (
                response.prompt_tokens + response.completion_tokens
            )
            bucket["cost_usd"] = float(bucket["cost_usd"]) + (
                response.estimated_cost_usd
            )
        return report

    def reset(self) -> None:
        """Discard every response record accumulated by this tracker."""
        self._records.clear()


class LLMRouter:
    """Route knowledge-pipeline calls through local or API model tiers."""

    def __init__(self, cfg: RouterCfg) -> None:
        """Bind immutable routing policy and initialize per-router accounting."""
        self._cfg = cfg
        self._cost_tracker = CostTracker()
        self._ollama_available: bool | None = None

    @property
    def cost_tracker(self) -> CostTracker:
        """Expose the accounting ledger associated with this router."""
        return self._cost_tracker

    @property
    def cfg(self) -> RouterCfg:
        """Expose the immutable routing policy bound to this router."""
        return self._cfg

    async def generate(
        self,
        task_type: TaskType,
        prompt: str,
        system_prompt: str = "",
        force_tier: ModelTier | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Generate text using the selected tier, falling back after failures.

        Args:
            task_type: Semantic purpose of the knowledge-pipeline request.
            prompt: User content to send to the selected model.
            system_prompt: Optional system instruction.
            force_tier: Optional tier that overrides configured default routing.
            temperature: Optional sampling temperature override.

        Returns:
            A normalized response with usage, cost, model, and latency details.

        Raises:
            RuntimeError: If every configured tier fails.
            ValueError: If the configured tier mapping is invalid.
        """
        target_tier = force_tier or self._configured_tier(task_type)
        resolved_temperature = (
            temperature
            if temperature is not None
            else _default_temperature(task_type)
        )
        messages = _messages(system_prompt, prompt)
        last_error: Exception | None = None

        for tier in _fallback_tiers(target_tier):
            for _ in range(self._cfg.ollama.max_retries):
                try:
                    if (
                        tier is ModelTier.LOCAL
                        and not self._is_ollama_available()
                    ):
                        raise ConnectionError(
                            "the configured Ollama server is unavailable"
                        )
                    model = self._model_for(tier)
                    started = perf_counter()
                    completion = await acompletion(
                        model=model,
                        messages=messages,
                        temperature=resolved_temperature,
                        timeout=self._cfg.ollama.timeout_seconds,
                        **cast(Any, self._completion_kwargs(tier)),
                    )
                    response = _normalize_completion(
                        completion,
                        model=model,
                        tier=tier,
                        latency_seconds=perf_counter() - started,
                    )
                    self._cost_tracker.track(task_type, response)
                    return response
                except Exception as error:
                    last_error = error
                    _LOGGER.warning(
                        "LLM tier %s failed for %s: %s",
                        tier.value,
                        task_type.value,
                        error,
                    )

        message = f"all LLM tiers failed for {task_type.value}"
        raise RuntimeError(message) from last_error

    async def embed(self, text: str) -> list[float]:
        """Embed text locally, falling back to OpenAI's small embedding model.

        Args:
            text: The text to encode.

        Returns:
            The first embedding vector returned by the selected provider.

        Raises:
            RuntimeError: If neither embedding provider returns a vector.
        """
        candidates = (
            (
                f"ollama/{self._cfg.ollama.embedding_model}",
                {"api_base": self._cfg.ollama.base_url},
            ),
            ("text-embedding-3-small", {}),
        )
        last_error: Exception | None = None
        for model, kwargs in candidates:
            try:
                result = await aembedding(
                    model=model,
                    input=[text],
                    timeout=self._cfg.ollama.timeout_seconds,
                    **kwargs,
                )
                return _first_embedding(result)
            except Exception as error:
                last_error = error
                _LOGGER.warning("Embedding model %s failed: %s", model, error)
        raise RuntimeError("all embedding providers failed") from last_error

    def _configured_tier(self, task_type: TaskType) -> ModelTier:
        """Resolve a configured task-to-tier string to a ``ModelTier``."""
        try:
            value = self._cfg.default_routing[task_type.value]
        except KeyError as error:
            raise ValueError(
                f"no routing tier configured for {task_type.value}"
            ) from error
        try:
            return ModelTier(value)
        except ValueError as error:
            raise ValueError(
                f"invalid routing tier {value!r} for {task_type.value}"
            ) from error

    def _is_ollama_available(self) -> bool:
        """Check Ollama once per router instance and cache the result."""
        if self._ollama_available is None:
            self._ollama_available = self._cfg.ollama.is_available()
        return self._ollama_available

    def _model_for(self, tier: ModelTier) -> str:
        """Return the LiteLLM model string for a generation tier."""
        if tier is ModelTier.LOCAL:
            return f"ollama/{self._cfg.ollama.generation_model}"
        if tier is ModelTier.API_CHEAP:
            return self._cfg.api_cheap_model
        return self._cfg.api_powerful_model

    def _completion_kwargs(self, tier: ModelTier) -> dict[str, str]:
        """Return provider-specific completion options for one tier."""
        if tier is ModelTier.LOCAL:
            return {"api_base": self._cfg.ollama.base_url}
        return {}


def safe_json_parse(raw_text: str) -> dict[str, Any] | list[Any] | None:
    """Parse a JSON object or list, including common Markdown-fenced output."""
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\s*", "", text, count=1, flags=re.I)
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    elif text.lower().startswith("json"):
        text = text[4:].lstrip()
        if text.startswith(":"):
            text = text[1:].lstrip()
    parsed = _decode_json_value(text)
    if parsed is not None:
        return parsed

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            candidate, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, (dict, list)):
            return candidate
    return None


def _decode_json_value(text: str) -> dict[str, Any] | list[Any] | None:
    """Return a parsed JSON container or ``None`` for invalid/non-container JSON."""
    try:
        candidate = json.loads(text)
    except json.JSONDecodeError:
        return None
    return candidate if isinstance(candidate, (dict, list)) else None


def _default_temperature(task_type: TaskType) -> float:
    """Return the deterministic default for factual tasks or creative generation."""
    if task_type is TaskType.GENERATION:
        return 0.7
    return 0.0


def _fallback_tiers(target: ModelTier) -> tuple[ModelTier, ...]:
    """Return a deterministic fallback sequence beginning with ``target``."""
    if target is ModelTier.LOCAL:
        return (
            ModelTier.LOCAL,
            ModelTier.API_CHEAP,
            ModelTier.API_POWERFUL,
        )
    if target is ModelTier.API_CHEAP:
        return (
            ModelTier.API_CHEAP,
            ModelTier.API_POWERFUL,
            ModelTier.LOCAL,
        )
    return (
        ModelTier.API_POWERFUL,
        ModelTier.API_CHEAP,
        ModelTier.LOCAL,
    )


def _messages(system_prompt: str, prompt: str) -> list[dict[str, str]]:
    """Build LiteLLM chat messages without emitting an empty system message."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _normalize_completion(
    completion: object,
    *,
    model: str,
    tier: ModelTier,
    latency_seconds: float,
) -> LLMResponse:
    """Convert a LiteLLM completion object or mapping into ``LLMResponse``."""
    choices = _field(completion, "choices", [])
    first_choice = choices[0] if choices else {}
    message = _field(first_choice, "message", {})
    content = _field(message, "content", "")
    usage = _field(completion, "usage", {})
    prompt_tokens = _token_count(_field(usage, "prompt_tokens", 0))
    completion_tokens = _token_count(_field(usage, "completion_tokens", 0))
    return LLMResponse(
        text=str(content or ""),
        model_used=model,
        tier=tier,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=_estimate_cost(
            tier, prompt_tokens, completion_tokens
        ),
        latency_seconds=latency_seconds,
    )


def _first_embedding(result: object) -> list[float]:
    """Extract the first vector from a LiteLLM embedding result."""
    data = _field(result, "data", [])
    if not data:
        raise ValueError("embedding response contained no data")
    embedding = _field(data[0], "embedding", None)
    if not isinstance(embedding, list):
        raise ValueError("embedding response contained no embedding vector")
    return [float(value) for value in embedding]


def _field(value: object, name: str, default: Any) -> Any:
    """Read a field from a mapping or SDK object without binding SDK types."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_count(value: object) -> int:
    """Convert an optional provider usage value to a non-negative integer."""
    if not isinstance(value, (int, float, str)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _estimate_cost(
    tier: ModelTier, prompt_tokens: int, completion_tokens: int
) -> float:
    """Estimate USD cost from static per-million-token model-tier prices."""
    if tier is ModelTier.LOCAL:
        return 0.0
    if tier is ModelTier.API_CHEAP:
        input_rate = _CHEAP_INPUT_COST_PER_MILLION
        output_rate = _CHEAP_OUTPUT_COST_PER_MILLION
    else:
        input_rate = _POWERFUL_INPUT_COST_PER_MILLION
        output_rate = _POWERFUL_OUTPUT_COST_PER_MILLION
    return (
        prompt_tokens * input_rate + completion_tokens * output_rate
    ) / 1_000_000
