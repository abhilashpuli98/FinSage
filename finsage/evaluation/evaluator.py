"""Automated evaluation and multi-system benchmarking for FinBench-50.

Implements ``AutoEvaluator`` to score system answers across factual, temporal,
and analytical question types, run parallel evaluations with caching, and compute
paired statistical tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from finsage.configs.ollama import TaskType
from finsage.evaluation.models import (
    BenchmarkQuestion,
    EvalResult,
    QuestionType,
    SystemAnswer,
)
from finsage.mind.router import LLMRouter, safe_json_parse

_LOGGER = logging.getLogger(__name__)

__all__ = ["AutoEvaluator"]

_FACTUAL_JUDGE_PROMPT = """\
Does this answer correctly contain the following information?
Correct answer: {ground_truth}
Given answer: {answer_text}

Score from 0.0 to 1.0: 1.0=fully correct, 0.5=partially correct, 0.0=wrong/missing.
Return ONLY valid JSON: {{"score": 1.0, "reason": "concise rationale"}}"""

_ANALYTICAL_JUDGE_PROMPT = """\
Rate the quality of this answer to the financial research question.
Question: {question_text}
Expected reasoning: {expected_reasoning}
Answer: {answer_text}

Score 0.0 to 1.0: 1.0=comprehensive and accurate, 0.0=irrelevant or wrong.
Return ONLY valid JSON: {{"score": 1.0, "feedback": "concise feedback"}}"""


class AutoEvaluator:
    """Automated benchmark evaluator comparing multi-system answers."""

    def __init__(
        self,
        router: LLMRouter,
        systems: dict[
            str, Callable[[BenchmarkQuestion], Awaitable[SystemAnswer]]
        ],
    ) -> None:
        """Initialize the automated evaluator.

        Args:
            router: Bound LLMRouter used as judge.
            systems: Dictionary mapping system_name to async callable.
        """
        self._router = router
        self._systems = systems

    @property
    def router(self) -> LLMRouter:
        """The judge LLMRouter."""
        return self._router

    @property
    def systems(
        self,
    ) -> dict[str, Callable[[BenchmarkQuestion], Awaitable[SystemAnswer]]]:
        """Configured evaluation systems."""
        return self._systems

    async def score_answer(
        self,
        question: BenchmarkQuestion,
        answer: SystemAnswer,
    ) -> EvalResult:
        """Score one system answer according to its question type.

        - FACTUAL: LLM judge scoring against exact ground truth.
        - TEMPORAL: Weighted composite (0.40 LLM content correctness + 0.30 target
          year mention + 0.30 explicit temporal scope).
        - ANALYTICAL: LLM judge scoring against expected multi-fact reasoning.

        Args:
            question: The benchmark question.
            answer: The candidate answer from an evaluated system.

        Returns:
            ``EvalResult`` containing numeric score in [0.0, 1.0].
        """
        if question.question_type == QuestionType.FACTUAL:
            prompt = _FACTUAL_JUDGE_PROMPT.format(
                ground_truth=question.ground_truth or "",
                answer_text=answer.answer_text[:600],
            )
            raw_score, notes = await self._run_judge_prompt(prompt)
            final_score = max(0.0, min(1.0, raw_score))
            scorer = "llm_factual_judge"

        elif question.question_type == QuestionType.TEMPORAL:
            # Component 1: LLM judge on ground truth (0.40)
            prompt = _FACTUAL_JUDGE_PROMPT.format(
                ground_truth=question.ground_truth or "",
                answer_text=answer.answer_text[:600],
            )
            llm_component, notes = await self._run_judge_prompt(prompt)
            llm_component = max(0.0, min(1.0, llm_component))

            # Component 2: Year mention (0.30)
            year_component = 0.0
            if question.temporal_constraint is not None:
                target_year = str(question.temporal_constraint.year)
                if target_year in answer.answer_text:
                    year_component = 1.0
            elif question.ground_truth:
                for word in question.ground_truth.split():
                    if (
                        word.isdigit()
                        and len(word) == 4
                        and word in answer.answer_text
                    ):
                        year_component = 1.0
                        break

            # Component 3: Non-empty temporal scope (0.30)
            scope_component = (
                1.0
                if answer.temporal_scope and answer.temporal_scope.strip()
                else 0.0
            )

            final_score = (
                (0.40 * llm_component)
                + (0.30 * year_component)
                + (0.30 * scope_component)
            )
            final_score = max(0.0, min(1.0, final_score))
            scorer = "temporal_composite_judge"

        else:  # ANALYTICAL
            prompt = _ANALYTICAL_JUDGE_PROMPT.format(
                question_text=question.question_text,
                expected_reasoning=question.expected_reasoning
                or "Sound quantitative financial reasoning",
                answer_text=answer.answer_text[:800],
            )
            raw_score, notes = await self._run_judge_prompt(prompt)
            final_score = max(0.0, min(1.0, raw_score))
            scorer = "llm_analytical_judge"

        return EvalResult(
            question_id=question.question_id,
            system_name=answer.system_name,
            score=round(final_score, 4),
            scorer=scorer,
            notes=notes,
        )

    async def _run_judge_prompt(self, prompt: str) -> tuple[float, str | None]:
        """Execute a scoring prompt against the router."""
        try:
            resp = await self._router.generate(
                TaskType.VERIFICATION,
                prompt=prompt,
            )
            parsed = safe_json_parse(resp.text)
            if isinstance(parsed, dict) and "score" in parsed:
                score = float(parsed["score"])
                reason = str(
                    parsed.get("reason") or parsed.get("feedback") or ""
                )
                return score, reason
        except Exception as exc:
            _LOGGER.warning("LLM judge evaluation failed: %s", exc)

        return 0.5, "Evaluation fallback default"

    async def evaluate_all(
        self,
        questions: list[BenchmarkQuestion],
        results_path: Path,
    ) -> pd.DataFrame:
        """Run all systems over all questions with caching and concurrency bounding.

        Args:
            questions: List of BenchmarkQuestion instances.
            results_path: Directory to store cached JSON evaluations.

        Returns:
            DataFrame with evaluation rows across systems and questions.
        """
        results_path.mkdir(parents=True, exist_ok=True)
        semaphore = asyncio.Semaphore(5)
        records: list[dict[str, Any]] = []

        async def _evaluate_single(
            system_name: str, q: BenchmarkQuestion
        ) -> dict[str, Any]:
            cache_file = results_path / f"{system_name}_{q.question_id}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    return data
                except Exception:
                    pass

            async with semaphore:
                runner = self._systems[system_name]
                answer = await runner(q)
                eval_res = await self.score_answer(q, answer)

                row = {
                    "system": system_name,
                    "question_id": q.question_id,
                    "question_type": q.question_type.value,
                    "domain": q.domain.value,
                    "difficulty": q.difficulty,
                    "score": eval_res.score,
                    "latency_s": answer.latency_seconds,
                    "cost_usd": answer.cost_usd,
                }
                cache_file.write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )
                return row

        tasks = []
        for sys_name in self._systems:
            for q in questions:
                tasks.append(_evaluate_single(sys_name, q))

        results = await asyncio.gather(*tasks)
        records.extend(results)

        return pd.DataFrame(records)

    def statistical_analysis(
        self, df: pd.DataFrame
    ) -> dict[str, dict[str, Any]]:
        """Run paired t-tests and Cohen's d against finsage_full.

        Args:
            df: Results DataFrame containing 'system', 'question_id', and 'score'.

        Returns:
            Dictionary mapping system_name to statistical metrics:
            {mean_score, std, p_value_vs_finsage, cohens_d, n}.
        """
        stats_report: dict[str, dict[str, Any]] = {}
        target_sys = "finsage_full"

        if target_sys not in df["system"].unique():
            target_sys = df["system"].unique()[0]

        target_df = df[df["system"] == target_sys].set_index("question_id")[
            "score"
        ]

        for sys_name in df["system"].unique():
            sys_df = df[df["system"] == sys_name].set_index("question_id")[
                "score"
            ]
            # Align on common questions
            common_idx = target_df.index.intersection(sys_df.index)
            scores_a = target_df.loc[common_idx].to_numpy(dtype=float)
            scores_b = sys_df.loc[common_idx].to_numpy(dtype=float)

            n = len(common_idx)
            mean_score = float(np.mean(scores_b)) if n > 0 else 0.0
            std_score = float(np.std(scores_b, ddof=1)) if n > 1 else 0.0

            if sys_name == target_sys or n < 2:
                p_value = 1.0
                cohens_d = 0.0
            else:
                # Paired t-test
                diff = scores_a - scores_b
                diff_std = float(np.std(diff, ddof=1))
                if diff_std > 1e-9:
                    t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
                    p_value = float(p_val)
                    cohens_d = float(np.mean(diff) / diff_std)
                else:
                    p_value = 1.0
                    cohens_d = 0.0

            stats_report[sys_name] = {
                "mean_score": round(mean_score, 4),
                "std": round(std_score, 4),
                "p_value_vs_finsage": round(p_value, 6),
                "cohens_d": round(cohens_d, 4),
                "n": n,
            }

        return stats_report
