"""LLM quality-cost analysis benchmark.

Produces **Table 2** (LLM quality-cost Pareto) of the FinSage paper.
Compares multiple LLM tiers — local (Ollama) and API — across extraction,
tagging, and summarization tasks using curated gold-standard evaluation sets.

The benchmark is *offline-safe*: gold data are loaded from static JSON files
under ``data/benchmark/llm_eval/``.  If the files do not exist on disk,
built-in minimal exemplars are used so unit tests can run without fixtures.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from finsage.configs.ollama import ModelTier, TaskType
from finsage.mind.router import LLMResponse, LLMRouter, safe_json_parse

_LOGGER = logging.getLogger(__name__)

__all__ = ["LLMQualityBenchmark"]


# ── Hardcoded cost constants (approximate USD / 1M tokens) ───────────

_COST_PER_1M: dict[str, tuple[float, float]] = {
    # (input, output)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "ollama/llama3.2": (0.00, 0.00),
    "ollama/mistral": (0.00, 0.00),
}


# ── Built-in minimal exemplars (used when gold files are absent) ─────

_BUILTIN_EXTRACTION_GOLD: list[dict[str, Any]] = [
    {
        "text": (
            "We study the momentum factor premium in US equities from "
            "January 1963 to December 2022. The annualised momentum premium "
            "is 8.3% with a t-statistic of 4.1."
        ),
        "expected_claims": [
            "The annualised momentum premium is 8.3%.",
            "The t-statistic for the momentum premium is 4.1.",
            "The study period spans January 1963 to December 2022.",
        ],
    },
    {
        "text": (
            "The Fama-French three-factor model explains approximately 90% "
            "of cross-sectional return variation in US equities. The value "
            "premium averages 4.8% annually."
        ),
        "expected_claims": [
            "The Fama-French three-factor model explains approximately 90% of "
            "cross-sectional return variation.",
            "The value premium averages 4.8% annually.",
        ],
    },
    {
        "text": (
            "Black-Scholes implied volatility exhibits a persistent skew in "
            "S&P 500 index options, with out-of-the-money puts priced 12% "
            "higher than at-the-money options on average."
        ),
        "expected_claims": [
            "Black-Scholes implied volatility exhibits a persistent skew in "
            "S&P 500 index options.",
            "Out-of-the-money puts are priced 12% higher than at-the-money options.",
        ],
    },
]

_BUILTIN_TAGGING_GOLD: list[dict[str, str]] = [
    {
        "content": "Momentum factor earns 1.2% monthly premium in US equities.",
        "domain": "FACTOR_INVESTING",
    },
    {
        "content": "Black-Scholes implied volatility skew is persistent.",
        "domain": "OPTIONS_PRICING",
    },
    {
        "content": "LSTM networks predict stock returns with 62% accuracy.",
        "domain": "ML_FINANCE",
    },
]

_BUILTIN_SUMMARIZATION_GOLD: list[dict[str, str]] = [
    {
        "text": (
            "We document that the momentum effect is pervasive across asset "
            "classes and geographies. Using data from 1927 to 2023, we show "
            "that a long-short momentum portfolio earns 8.3% annualized with "
            "a Sharpe ratio of 0.55. The effect survives standard risk "
            "adjustments including the Fama-French five-factor model."
        ),
        "reference_summary": (
            "This study demonstrates the pervasiveness of the momentum effect "
            "across asset classes and geographies over nearly a century of data. "
            "A long-short momentum portfolio achieves 8.3% annualized returns "
            "with a 0.55 Sharpe ratio, surviving Fama-French five-factor risk "
            "adjustments."
        ),
    },
]


# ── Extraction scoring prompt ────────────────────────────────────────

_EXTRACTION_EVAL_SYSTEM = (
    "You are evaluating the quality of financial knowledge extraction. "
    "Compare extracted claims against the expected gold-standard claims."
)

_EXTRACTION_EVAL_USER = """\
Expected claims (gold standard):
{expected_claims}

Extracted claims:
{extracted_claims}

Score from 0.0 to 1.0:
- 1.0 = all expected claims are present in the extraction
- 0.5 = roughly half are present
- 0.0 = none are present
Return ONLY valid JSON: {{"score": 0.0, "reason": "concise rationale"}}"""


_EXTRACTION_PROMPT_SYSTEM = (
    "Extract distinct factual claims from financial text. "
    "Return ONLY a valid JSON array of claim strings."
)

_EXTRACTION_PROMPT_USER = """\
Extract all distinct factual claims from this text:
{text}

Return JSON array: ["claim 1", "claim 2", ...]"""


_TAGGING_PROMPT = """\
Classify this financial claim into exactly one domain:
FACTOR_INVESTING, OPTIONS_PRICING, MARKET_MICROSTRUCTURE, CREDIT_RISK, \
ML_FINANCE, BEHAVIORAL_FINANCE, MACRO_ECONOMICS

Claim: {content}
Return ONLY the domain name, nothing else."""


_SUMMARIZATION_PROMPT = """\
Summarize the following financial research text in 2-3 sentences:
{text}"""


# ── LLMQualityBenchmark ─────────────────────────────────────────────


class LLMQualityBenchmark:
    """Benchmark LLM quality and cost across extraction, tagging, and summarization.

    Produces the Pareto frontier data for Table 2 of the FinSage paper.
    All LLM calls go through the ``LLMRouter`` which handles fallback
    and cost tracking.

    Example::

        router = LLMRouter(RouterCfg())
        bench = LLMQualityBenchmark(router)
        df = await bench.run_all_benchmarks(Path("results/llm_bench"))
    """

    def __init__(
        self,
        router: LLMRouter,
        models: tuple[str, ...] | None = None,
    ) -> None:
        """Initialize with an LLM router and optional model list.

        Args:
            router: Bound LLM router for dispatching generation calls.
            models: Override the default model list. Each string must be
                a litellm-compatible model identifier (e.g. ``"gpt-4o-mini"``
                or ``"ollama/llama3.2"``).
        """
        self._router = router
        self._models: tuple[str, ...] = models or (
            "ollama/llama3.2",
            "gpt-4o-mini",
            "gpt-4o",
        )

    @property
    def router(self) -> LLMRouter:
        """The LLM router bound to this benchmark."""
        return self._router

    @property
    def models(self) -> tuple[str, ...]:
        """The list of models under evaluation."""
        return self._models

    # ── Loading gold data ────────────────────────────────────────────

    @staticmethod
    def _load_gold(
        path: Path,
        builtin: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Load gold-standard evaluation data, falling back to builtins."""
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return data  # type: ignore[return-value]
            except Exception as exc:
                _LOGGER.warning("Failed to load gold file %s: %s", path, exc)
        return builtin

    # ── Per-task benchmarks ──────────────────────────────────────────

    async def run_extraction_benchmark(
        self,
        gold_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run extraction quality benchmark across all models.

        For each gold example × model, calls the model to extract claims
        and then scores the extraction quality via an LLM judge.

        Args:
            gold_path: Optional path to ``extraction_gold.json``.

        Returns:
            List of per-example result dicts with model, score, latency,
            and cost fields.
        """
        gold = self._load_gold(
            gold_path or Path("data/benchmark/llm_eval/extraction_gold.json"),
            _BUILTIN_EXTRACTION_GOLD,
        )
        results: list[dict[str, Any]] = []

        for example in gold:
            text = example["text"]
            expected = example["expected_claims"]

            for model in self._models:
                start = time.perf_counter()
                try:
                    resp = await self._call_model(
                        model,
                        _EXTRACTION_PROMPT_USER.format(text=text),
                        system_prompt=_EXTRACTION_PROMPT_SYSTEM,
                    )
                    latency = time.perf_counter() - start
                    extracted = safe_json_parse(resp.text)
                    if not isinstance(extracted, list):
                        extracted = [resp.text.strip()]
                except Exception as exc:
                    _LOGGER.warning("Extraction failed for %s: %s", model, exc)
                    latency = time.perf_counter() - start
                    results.append(
                        _error_row("extraction", model, latency, str(exc))
                    )
                    continue

                # Score with LLM judge.
                score = await self._score_extraction(expected, extracted)

                results.append(
                    {
                        "task": "extraction",
                        "model": model,
                        "score": round(score, 4),
                        "latency_s": round(latency, 4),
                        "cost_usd": round(resp.estimated_cost_usd, 8),
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                    }
                )

        return results

    async def run_tagging_benchmark(
        self,
        gold_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run domain tagging benchmark (exact-match scoring).

        Args:
            gold_path: Optional path to ``tagging_gold.json``.

        Returns:
            List of per-example result dicts.
        """
        gold = self._load_gold(
            gold_path or Path("data/benchmark/llm_eval/tagging_gold.json"),
            _BUILTIN_TAGGING_GOLD,
        )
        results: list[dict[str, Any]] = []

        for example in gold:
            content = example["content"]
            expected_domain = example["domain"].upper().strip()

            for model in self._models:
                start = time.perf_counter()
                try:
                    resp = await self._call_model(
                        model,
                        _TAGGING_PROMPT.format(content=content),
                    )
                    latency = time.perf_counter() - start
                    predicted = resp.text.strip().upper()
                    # Strip any extra text around the domain name.
                    for domain_name in [
                        "FACTOR_INVESTING",
                        "OPTIONS_PRICING",
                        "MARKET_MICROSTRUCTURE",
                        "CREDIT_RISK",
                        "ML_FINANCE",
                        "BEHAVIORAL_FINANCE",
                        "MACRO_ECONOMICS",
                    ]:
                        if domain_name in predicted:
                            predicted = domain_name
                            break
                    score = 1.0 if predicted == expected_domain else 0.0
                except Exception as exc:
                    _LOGGER.warning("Tagging failed for %s: %s", model, exc)
                    latency = time.perf_counter() - start
                    results.append(
                        _error_row("tagging", model, latency, str(exc))
                    )
                    continue

                results.append(
                    {
                        "task": "tagging",
                        "model": model,
                        "score": score,
                        "latency_s": round(latency, 4),
                        "cost_usd": round(resp.estimated_cost_usd, 8),
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                    }
                )

        return results

    async def run_summarization_benchmark(
        self,
        gold_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run summarization benchmark scored via BERTScore.

        Falls back to an LLM judge if ``bert_score`` is not importable.

        Args:
            gold_path: Optional path to ``summarization_gold.json``.

        Returns:
            List of per-example result dicts.
        """
        gold = self._load_gold(
            gold_path
            or Path("data/benchmark/llm_eval/summarization_gold.json"),
            _BUILTIN_SUMMARIZATION_GOLD,
        )
        results: list[dict[str, Any]] = []

        for example in gold:
            text = example["text"]
            reference = example["reference_summary"]

            for model in self._models:
                start = time.perf_counter()
                try:
                    resp = await self._call_model(
                        model,
                        _SUMMARIZATION_PROMPT.format(text=text),
                    )
                    latency = time.perf_counter() - start
                    hypothesis = resp.text.strip()
                except Exception as exc:
                    _LOGGER.warning(
                        "Summarization failed for %s: %s", model, exc
                    )
                    latency = time.perf_counter() - start
                    results.append(
                        _error_row("summarization", model, latency, str(exc))
                    )
                    continue

                score = _compute_bertscore_or_fallback(hypothesis, reference)

                results.append(
                    {
                        "task": "summarization",
                        "model": model,
                        "score": round(score, 4),
                        "latency_s": round(latency, 4),
                        "cost_usd": round(resp.estimated_cost_usd, 8),
                        "prompt_tokens": resp.prompt_tokens,
                        "completion_tokens": resp.completion_tokens,
                    }
                )

        return results

    async def run_all_benchmarks(
        self,
        output_dir: Path,
    ) -> pd.DataFrame:
        """Run extraction, tagging, and summarization benchmarks.

        Aggregates per-model mean scores, latencies, and costs into a
        single DataFrame.  Also saves the raw per-example results and
        an aggregated CSV to ``output_dir``.

        Args:
            output_dir: Directory for result files.

        Returns:
            Aggregated ``DataFrame`` with columns: task, model, mean_score,
            std_score, mean_latency_s, mean_cost_per_call_usd, relative_cost.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        extraction_results = await self.run_extraction_benchmark()
        tagging_results = await self.run_tagging_benchmark()
        summarization_results = await self.run_summarization_benchmark()

        all_results = (
            extraction_results + tagging_results + summarization_results
        )

        # Save raw results.
        raw_path = output_dir / "llm_benchmark_raw.json"
        raw_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        _LOGGER.info("Saved raw benchmark results to %s", raw_path)

        # Aggregate.
        df_raw = pd.DataFrame(all_results)
        if df_raw.empty:
            _LOGGER.warning("No benchmark results collected.")
            return pd.DataFrame()

        grouped = (
            df_raw.groupby(["task", "model"])
            .agg(
                mean_score=("score", "mean"),
                std_score=("score", "std"),
                mean_latency_s=("latency_s", "mean"),
                mean_cost_per_call_usd=("cost_usd", "mean"),
            )
            .reset_index()
        )

        # Fill NaN std (single-example groups) with 0.
        grouped["std_score"] = grouped["std_score"].fillna(0.0)

        # Compute relative cost (normalised to gpt-4o = 1.0).
        gpt4o_cost = grouped.loc[
            grouped["model"] == "gpt-4o", "mean_cost_per_call_usd"
        ].mean()
        if gpt4o_cost and gpt4o_cost > 0:
            grouped["relative_cost"] = (
                grouped["mean_cost_per_call_usd"] / gpt4o_cost
            )
        else:
            grouped["relative_cost"] = 0.0

        # Round for readability.
        for col in (
            "mean_score",
            "std_score",
            "mean_latency_s",
            "relative_cost",
        ):
            grouped[col] = grouped[col].round(4)
        grouped["mean_cost_per_call_usd"] = grouped[
            "mean_cost_per_call_usd"
        ].round(8)

        # Save aggregated CSV.
        csv_path = output_dir / "llm_benchmark_results.csv"
        grouped.to_csv(csv_path, index=False)
        _LOGGER.info("Saved aggregated benchmark to %s", csv_path)

        # Attempt Pareto plot.
        try:
            _save_pareto_plot(grouped, output_dir / "pareto_plot.png")
        except Exception as exc:
            _LOGGER.warning("Pareto plot generation failed: %s", exc)

        return grouped

    # ── Internal helpers ─────────────────────────────────────────────

    async def _call_model(
        self,
        model: str,
        prompt: str,
        system_prompt: str = "",
    ) -> LLMResponse:
        """Route a prompt to a specific model via the router.

        Maps model names to ``ModelTier`` and ``force_tier`` to ensure
        the exact model is used.
        """
        if model.startswith("ollama/"):
            tier = ModelTier.LOCAL
        elif model in ("gpt-4o-mini",):
            tier = ModelTier.API_CHEAP
        else:
            tier = ModelTier.API_POWERFUL

        return await self._router.generate(
            TaskType.EXTRACTION,
            prompt=prompt,
            system_prompt=system_prompt,
            force_tier=tier,
        )

    async def _score_extraction(
        self,
        expected: list[str],
        extracted: list[Any],
    ) -> float:
        """Score extraction quality via LLM judge."""
        prompt = _EXTRACTION_EVAL_USER.format(
            expected_claims="\n".join(f"- {c}" for c in expected),
            extracted_claims="\n".join(f"- {c}" for c in extracted[:20]),
        )
        try:
            resp = await self._router.generate(
                TaskType.VERIFICATION,
                prompt=prompt,
                system_prompt=_EXTRACTION_EVAL_SYSTEM,
            )
            parsed = safe_json_parse(resp.text)
            if isinstance(parsed, dict) and "score" in parsed:
                return float(parsed["score"])
        except Exception as exc:
            _LOGGER.debug("Extraction scoring fallback: %s", exc)

        # Fallback: simple overlap ratio.
        if not expected:
            return 0.5
        hits = sum(
            1
            for exp in expected
            if any(exp.lower()[:40] in str(ext).lower() for ext in extracted)
        )
        return hits / len(expected)


# ── Module-private helpers ───────────────────────────────────────────


def _error_row(
    task: str, model: str, latency: float, reason: str
) -> dict[str, Any]:
    """Create an error result row."""
    return {
        "task": task,
        "model": model,
        "score": 0.0,
        "latency_s": round(latency, 4),
        "cost_usd": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": reason,
    }


def _compute_bertscore_or_fallback(hypothesis: str, reference: str) -> float:
    """Compute BERTScore F1, falling back to simple token overlap."""
    try:
        from bert_score import score as bert_score_fn

        _, _, f1 = bert_score_fn(
            [hypothesis],
            [reference],
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        return float(f1[0])
    except ImportError:
        _LOGGER.debug("bert_score not installed; using token overlap fallback.")
    except Exception as exc:
        _LOGGER.debug("BERTScore computation failed: %s; using fallback.", exc)

    # Token-overlap fallback (Jaccard similarity).
    hyp_tokens = set(hypothesis.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    intersection = hyp_tokens & ref_tokens
    union = hyp_tokens | ref_tokens
    return len(intersection) / len(union) if union else 0.0


def _save_pareto_plot(df: pd.DataFrame, path: Path) -> None:
    """Generate a quality-cost Pareto scatter plot.

    X-axis: mean cost per call (log scale).
    Y-axis: mean quality score.
    Colour by task type, marker shape by model.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _LOGGER.info("matplotlib not installed; skipping Pareto plot.")
        return

    task_colors = {
        "extraction": "#2196F3",
        "tagging": "#4CAF50",
        "summarization": "#FF9800",
    }
    model_markers = {
        "ollama/llama3.2": "o",
        "ollama/mistral": "s",
        "gpt-4o-mini": "^",
        "gpt-4o": "D",
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlabel("Mean Cost per Call (USD, log scale)", fontsize=12)
    ax.set_ylabel("Mean Quality Score", fontsize=12)
    ax.set_title(
        "LLM Quality-Cost Pareto Frontier", fontsize=14, fontweight="bold"
    )

    for _, row in df.iterrows():
        cost = row["mean_cost_per_call_usd"]
        if cost <= 0:
            cost = 1e-8  # Prevent log(0).
        color = task_colors.get(str(row["task"]), "#999999")
        marker = model_markers.get(str(row["model"]), "x")
        ax.scatter(
            cost,
            row["mean_score"],
            c=color,
            marker=marker,
            s=100,
            alpha=0.8,
            edgecolors="black",
            linewidth=0.5,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Legend for tasks.
    from matplotlib.lines import Line2D

    task_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=10,
            label=t,
        )
        for t, c in task_colors.items()
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=m,
            color="w",
            markerfacecolor="gray",
            markersize=10,
            label=n,
        )
        for n, m in model_markers.items()
    ]
    ax.legend(
        handles=task_handles + model_handles,
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOGGER.info("Pareto plot saved to %s", path)
