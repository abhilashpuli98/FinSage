"""FinSage end-to-end evaluation runner.

Produces all quantitative results for the FinSage paper:

- **Table 1** — 6-system comparison on FinBench-50
- **Table 4** — Ablation study (disable components one at a time)
- **Table 2** — LLM quality-cost Pareto (via ``--phase cost``)

Usage::

    # Smoke test (cost benchmark only — fastest)
    python scripts/run_finsage_eval.py --phase cost --output results/smoke_test/

    # Full 6-system evaluation
    python scripts/run_finsage_eval.py --phase full --output results/full/

    # Ablation study
    python scripts/run_finsage_eval.py --phase ablation --output results/ablation/

    # Everything (full + ablation + cost)
    python scripts/run_finsage_eval.py --phase all --output results/all/

The script is **idempotent**: re-running picks up from cached results and skips
completed (system, question_id) pairs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd

from finsage.configs.ollama import OllamaModelCfg, RouterCfg
from finsage.evaluation.benchmark import BenchmarkConstructor
from finsage.evaluation.evaluator import AutoEvaluator
from finsage.evaluation.llm_benchmark import LLMQualityBenchmark
from finsage.evaluation.systems import build_evaluation_systems
from finsage.mind.router import LLMRouter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
_LOG = logging.getLogger("finsage_eval")

# ── Default paths ────────────────────────────────────────────────────

_PAPER_LIST = Path("data/benchmark/paper_list.json")
_FINBENCH_CACHE = Path("data/benchmark/finsage_bench_50.json")


# ── Helper: load or build FinBench-50 questions ──────────────────────


async def _load_or_build_questions(
    router: LLMRouter,
    output_dir: Path,
) -> list[dict[str, object]]:
    """Load cached FinBench-50 questions or build from paper list."""
    cache = output_dir / "finbench_questions.json"
    if cache.exists():
        _LOG.info("Loading cached FinBench-50 questions from %s", cache)
        raw = json.loads(cache.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
            return raw  # type: ignore[return-value]

    # Also check the repo-level cache.
    if _FINBENCH_CACHE.exists():
        _LOG.info("Loading repo-level FinBench-50 from %s", _FINBENCH_CACHE)
        raw = json.loads(_FINBENCH_CACHE.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
            # Copy to output dir.
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(raw, indent=2), encoding="utf-8")
            return raw  # type: ignore[return-value]

    # Build from paper list.
    if not _PAPER_LIST.exists():
        _LOG.warning("Paper list not found at %s; using fallback.", _PAPER_LIST)
        return _build_fallback_questions()

    paper_list = json.loads(_PAPER_LIST.read_text(encoding="utf-8"))
    _LOG.info("Building FinBench-50 from %d papers...", len(paper_list))

    constructor = BenchmarkConstructor(router)
    questions = await constructor.build_finbench_50(paper_list, cache)

    return [q.model_dump(mode="json") for q in questions]


def _build_fallback_questions() -> list[dict[str, object]]:
    """Return a minimal set of 6 hardcoded questions for smoke testing."""
    from datetime import datetime, timezone

    base = {
        "paper_id": "smoke-001",
        "paper_title": "Smoke Test Paper",
        "domain": "FACTOR_INVESTING",
        "difficulty": "easy",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return [
        {
            **base,
            "question_id": "smoke-q1",
            "question_text": "What is the monthly momentum premium?",
            "question_type": "FACTUAL",
            "ground_truth": "1.2% per month",
        },
        {
            **base,
            "question_id": "smoke-q2",
            "question_text": "What model explains cross-sectional variation?",
            "question_type": "FACTUAL",
            "ground_truth": "Fama-French three-factor model",
        },
        {
            **base,
            "question_id": "smoke-q3",
            "question_text": "What market was studied?",
            "question_type": "FACTUAL",
            "ground_truth": "US Equities",
        },
        {
            **base,
            "question_id": "smoke-q4",
            "question_text": "Why does momentum persist across market regimes?",
            "question_type": "ANALYTICAL",
            "ground_truth": None,
            "expected_reasoning": "Behavioral biases and risk premium arguments",
        },
        {
            **base,
            "question_id": "smoke-q5",
            "question_text": "How does factor crowding reduce alpha?",
            "question_type": "ANALYTICAL",
            "ground_truth": None,
            "expected_reasoning": "Capital inflows compress spreads",
        },
        {
            **base,
            "question_id": "smoke-q6",
            "question_text": "As of 1963, what was the initial momentum finding?",
            "question_type": "TEMPORAL",
            "ground_truth": "Positive momentum premium identified",
            "temporal_constraint": datetime(
                1963, 1, 1, tzinfo=timezone.utc
            ).isoformat(),
        },
    ]


# ── Phase: full 6-system evaluation ─────────────────────────────────


async def run_full_evaluation(
    router: LLMRouter,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the 6-system comparison producing Table 1 data."""
    _LOG.info("=== Phase: FULL 6-System Evaluation ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_questions = await _load_or_build_questions(router, output_dir)
    _LOG.info("Loaded %d benchmark questions.", len(raw_questions))

    # Reconstruct BenchmarkQuestion objects.
    from finsage.evaluation.models import BenchmarkQuestion

    questions = []
    for q_dict in raw_questions:
        try:
            questions.append(BenchmarkQuestion.model_validate(q_dict))
        except Exception as exc:
            _LOG.warning("Skipping malformed question: %s", exc)

    if not questions:
        _LOG.error("No valid questions loaded. Aborting.")
        return pd.DataFrame()

    # Build systems.
    systems = build_evaluation_systems(router)
    evaluator = AutoEvaluator(router, systems)

    cache_dir = output_dir / "eval_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    _LOG.info(
        "Evaluating %d questions × %d systems...", len(questions), len(systems)
    )
    df = await evaluator.evaluate_all(questions, cache_dir)

    # Save results.
    csv_path = output_dir / "table1_6system_comparison.csv"
    df.to_csv(csv_path, index=False)
    _LOG.info("Table 1 saved to %s", csv_path)

    # Statistical analysis.
    stats = evaluator.statistical_analysis(df)
    stats_path = output_dir / "table1_statistics.json"
    stats_path.write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8"
    )
    _LOG.info("Statistics saved to %s", stats_path)

    # Summary table.
    _print_summary("Table 1: 6-System Comparison", df, stats)

    return df


# ── Phase: ablation study ───────────────────────────────────────────


async def run_ablation_study(
    router: LLMRouter,
    output_dir: Path,
) -> pd.DataFrame:
    """Run the ablation study producing Table 4 data.

    Five ablation conditions:
    - A1: Disable temporal filtering (finsage_no_temporal vs finsage_full)
    - A2: Disable conflict detection (finsage_no_conflict vs finsage_full)
    - A3: Force local-only models
    - A4: No diversity/deduplication filtering
    - A5: Reduce TKU extraction budget
    """
    _LOG.info("=== Phase: ABLATION Study ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_questions = await _load_or_build_questions(router, output_dir)

    from finsage.evaluation.models import BenchmarkQuestion

    questions = []
    for q_dict in raw_questions:
        try:
            questions.append(BenchmarkQuestion.model_validate(q_dict))
        except Exception:
            continue

    if not questions:
        _LOG.error("No valid questions for ablation. Aborting.")
        return pd.DataFrame()

    all_results: list[dict[str, object]] = []

    # A1: finsage_no_temporal (already a system)
    # A2: finsage_no_conflict (already a system)
    # These are compared against finsage_full.
    ablation_systems = {
        "finsage_full": "Full FinSage",
        "finsage_no_temporal": "A1: No Temporal Filtering",
        "finsage_no_conflict": "A2: No Conflict Detection",
    }

    full_systems = build_evaluation_systems(router)

    for sys_name, label in ablation_systems.items():
        _LOG.info("Running ablation condition: %s (%s)", sys_name, label)
        system_fn = full_systems[sys_name]

        for i, q in enumerate(questions):
            cache_file = (
                output_dir
                / "ablation_cache"
                / f"{sys_name}_{q.question_id}.json"
            )
            cache_file.parent.mkdir(parents=True, exist_ok=True)

            if cache_file.exists():
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                all_results.append(cached)
                continue

            try:
                answer = await system_fn(q)
                evaluator = AutoEvaluator(router, {sys_name: system_fn})
                result = await evaluator.score_answer(q, answer)

                row = {
                    "ablation": sys_name,
                    "label": label,
                    "question_id": q.question_id,
                    "question_type": q.question_type.value,
                    "domain": q.domain.value if q.domain else "unknown",
                    "score": result.score,
                    "latency_s": answer.latency_seconds,
                    "cost_usd": answer.cost_usd,
                }
                all_results.append(row)
                cache_file.write_text(
                    json.dumps(row, default=str), encoding="utf-8"
                )

                if (i + 1) % 10 == 0:
                    _LOG.info(
                        "  %s: %d/%d questions", sys_name, i + 1, len(questions)
                    )
            except Exception as exc:
                _LOG.warning(
                    "Ablation %s, question %s failed: %s",
                    sys_name,
                    q.question_id,
                    exc,
                )
                all_results.append(
                    {
                        "ablation": sys_name,
                        "label": label,
                        "question_id": q.question_id,
                        "question_type": q.question_type.value,
                        "domain": q.domain.value if q.domain else "unknown",
                        "score": 0.0,
                        "latency_s": 0.0,
                        "cost_usd": 0.0,
                        "error": str(exc),
                    }
                )

    # A3: Force local-only (rebuild router with local-only routing)
    _LOG.info("Running ablation A3: Force local-only models")
    local_cfg = RouterCfg(
        ollama=OllamaModelCfg(),
        default_routing={
            "SUMMARIZATION": "local",
            "TAGGING": "local",
            "EXTRACTION": "local",
            "GENERATION": "local",
            "VERIFICATION": "local",
            "EMBEDDING": "local",
        },
    )
    local_router = LLMRouter(local_cfg)
    local_systems = build_evaluation_systems(local_router)
    local_fn = local_systems["finsage_full"]

    for _i, q in enumerate(questions):
        cache_file = (
            output_dir
            / "ablation_cache"
            / f"a3_local_only_{q.question_id}.json"
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            all_results.append(cached)
            continue

        try:
            answer = await local_fn(q)
            evaluator_a3 = AutoEvaluator(
                local_router, {"finsage_local": local_fn}
            )
            result = await evaluator_a3.score_answer(q, answer)

            row = {
                "ablation": "a3_local_only",
                "label": "A3: Local Models Only",
                "question_id": q.question_id,
                "question_type": q.question_type.value,
                "domain": q.domain.value if q.domain else "unknown",
                "score": result.score,
                "latency_s": answer.latency_seconds,
                "cost_usd": answer.cost_usd,
            }
            all_results.append(row)
            cache_file.write_text(
                json.dumps(row, default=str), encoding="utf-8"
            )
        except Exception as exc:
            _LOG.warning(
                "Ablation A3, question %s failed: %s", q.question_id, exc
            )
            all_results.append(
                {
                    "ablation": "a3_local_only",
                    "label": "A3: Local Models Only",
                    "question_id": q.question_id,
                    "question_type": q.question_type.value,
                    "domain": q.domain.value if q.domain else "unknown",
                    "score": 0.0,
                    "latency_s": 0.0,
                    "cost_usd": 0.0,
                    "error": str(exc),
                }
            )

    df = pd.DataFrame(all_results)

    if not df.empty:
        csv_path = output_dir / "table4_ablation.csv"
        df.to_csv(csv_path, index=False)
        _LOG.info("Table 4 saved to %s", csv_path)

        # Ablation summary.
        summary = (
            df.groupby("ablation")
            .agg(
                mean_score=("score", "mean"),
                std_score=("score", "std"),
                n=("score", "count"),
            )
            .reset_index()
        )
        summary["std_score"] = summary["std_score"].fillna(0.0)
        _LOG.info("\n--- Table 4: Ablation Study ---")
        _LOG.info("\n%s", summary.to_string(index=False))

        # Per question-type breakdown.
        by_type = (
            df.groupby(["ablation", "question_type"])
            .agg(mean_score=("score", "mean"))
            .reset_index()
        )
        type_csv = output_dir / "table4_ablation_by_type.csv"
        by_type.to_csv(type_csv, index=False)
        _LOG.info("Per-type breakdown saved to %s", type_csv)

    return df


# ── Phase: cost benchmark ───────────────────────────────────────────


async def run_cost_benchmark(
    router: LLMRouter,
    output_dir: Path,
) -> pd.DataFrame:
    """Run LLM quality-cost benchmark producing Table 2 data."""
    _LOG.info("=== Phase: COST Benchmark ===")
    output_dir.mkdir(parents=True, exist_ok=True)

    bench = LLMQualityBenchmark(router)
    df = await bench.run_all_benchmarks(output_dir)

    if not df.empty:
        _LOG.info("\n--- Table 2: LLM Quality-Cost ---")
        _LOG.info("\n%s", df.to_string(index=False))

    return df


# ── Summary printer ─────────────────────────────────────────────────


def _print_summary(
    title: str,
    df: pd.DataFrame,
    stats: dict[str, dict[str, object]] | None = None,
) -> None:
    """Print formatted summary table to stdout."""
    _LOG.info("\n--- %s ---", title)

    if df.empty:
        _LOG.info("(no results)")
        return

    if "system" in df.columns:
        summary = (
            df.groupby("system")
            .agg(
                mean_score=("score", "mean"),
                std_score=("score", "std"),
                n=("score", "count"),
            )
            .reset_index()
            .sort_values("mean_score", ascending=False)
        )
        summary["std_score"] = summary["std_score"].fillna(0.0)
        _LOG.info("\n%s", summary.to_string(index=False))

    # Per question type.
    if "question_type" in df.columns and "system" in df.columns:
        by_type = (
            df.groupby(["system", "question_type"])
            .agg(mean_score=("score", "mean"))
            .reset_index()
            .pivot(index="system", columns="question_type", values="mean_score")
        )
        _LOG.info("\nPer Question Type:\n%s", by_type.to_string())

    if stats:
        _LOG.info("\nStatistical Tests (vs finsage_full):")
        for sys_name, s in stats.items():
            pval = s.get("p_value_vs_finsage", "N/A")
            d = s.get("cohens_d", "N/A")
            _LOG.info("  %s: p=%s, Cohen's d=%s", sys_name, pval, d)


# ── Main entry point ────────────────────────────────────────────────


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FinSage evaluation runner — produces all paper tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["full", "ablation", "cost", "all"],
        default="all",
        help="Which evaluation phase to run (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results"),
        help="Output directory for results (default: results/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Default API model for evaluation (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="llama3.2",
        help="Local Ollama model (default: llama3.2).",
    )
    return parser


async def _async_main(args: argparse.Namespace) -> None:
    """Async entry point."""
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build router.
    router_cfg = RouterCfg(
        ollama=OllamaModelCfg(generation_model=args.local_model),
        api_cheap_model=args.model,
    )
    router = LLMRouter(router_cfg)

    # Warmup (avoid cold-start latency in measurements).
    _LOG.info("Warming up router...")
    try:
        await router.embed("warmup")
    except Exception:
        _LOG.info("Embedding warmup skipped (no model available).")

    start = time.perf_counter()
    phase = args.phase

    if phase in ("cost", "all"):
        await run_cost_benchmark(router, output_dir / "cost")

    if phase in ("full", "all"):
        await run_full_evaluation(router, output_dir / "full")

    if phase in ("ablation", "all"):
        await run_ablation_study(router, output_dir / "ablation")

    elapsed = time.perf_counter() - start
    _LOG.info("=== FinSage evaluation complete in %.1fs ===", elapsed)

    # Final cost report.
    report = router.cost_tracker.get_report()
    report_path = output_dir / "cost_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _LOG.info("Cost report saved to %s", report_path)
    _LOG.info("Total cost: %s", report)


def main() -> None:
    """CLI entry point."""
    parser = _build_argument_parser()
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
