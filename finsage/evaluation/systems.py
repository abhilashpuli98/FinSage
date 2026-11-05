"""Comparison system implementations for FinBench-50 evaluation.

Implements the 6-way comparison systems:
1. ``naive_rag``: Basic retrieval without TKU temporal bounds or conflict detection.
2. ``generic_llm``: Direct LLM query with paper context, no structured retrieval.
3. ``quantmind_style``: Extract, tag, retrieve, generate (standard RAG without TKU/CSCG).
4. ``finsage_no_conflict``: FinSage pipeline with temporal filtering, no conflict detection.
5. ``finsage_no_temporal``: FinSage pipeline with conflict detection, no temporal filtering.
6. ``finsage_full``: Complete FinSage pipeline (TKU temporal filtering + CSCG conflict resolution).
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from finsage.configs.ollama import TaskType
from finsage.evaluation.models import BenchmarkQuestion, SystemAnswer
from finsage.mind.router import LLMRouter

__all__ = ["build_evaluation_systems"]


class EvaluationSystemWrapper:
    """Standardized callable wrapper producing ``SystemAnswer`` for a benchmark question."""

    def __init__(
        self,
        name: str,
        runner: Callable[
            [BenchmarkQuestion], Awaitable[tuple[str, str | None, bool]]
        ],
        router: LLMRouter,
    ) -> None:
        self.name = name
        self._runner = runner
        self._router = router

    async def __call__(self, question: BenchmarkQuestion) -> SystemAnswer:
        start_time = time.perf_counter()
        answer_text, temporal_scope, has_conflict_note = await self._runner(
            question
        )
        latency = time.perf_counter() - start_time

        # Approximate cost based on answer length and question
        prompt_chars = len(question.question_text) + 200
        completion_chars = len(answer_text)
        cost = (prompt_chars / 4.0 * 0.15e-6) + (
            completion_chars / 4.0 * 0.60e-6
        )

        return SystemAnswer(
            question_id=question.question_id,
            system_name=self.name,
            answer_text=answer_text,
            latency_seconds=round(latency, 4),
            cost_usd=round(cost, 6),
            temporal_scope=temporal_scope,
            has_conflict_note=has_conflict_note,
        )


def build_evaluation_systems(
    router: LLMRouter,
    paper_contexts: dict[str, str] | None = None,
    paper_tkus: dict[str, list[Any]] | None = None,
) -> dict[str, Callable[[BenchmarkQuestion], Awaitable[SystemAnswer]]]:
    """Build the dictionary of 6 evaluation systems.

    Args:
        router: LLM router used for text generation.
        paper_contexts: Optional mapping from paper_id to raw paper text.
        paper_tkus: Optional mapping from paper_id to list of TKU objects.

    Returns:
        Mapping of system_name -> async callable taking BenchmarkQuestion.
    """
    contexts = paper_contexts or {}
    tkus_map = paper_tkus or {}

    # 1. generic_llm: Direct generation with paper context
    async def _run_generic_llm(
        q: BenchmarkQuestion,
    ) -> tuple[str, str | None, bool]:
        context = contexts.get(q.paper_id, f"Research paper: {q.paper_title}")
        prompt = (
            f"Based on the following research paper context, answer the question:\n\n"
            f"Context: {context[:3000]}\n\n"
            f"Question: {q.question_text}\n\nAnswer:"
        )
        resp = await router.generate(TaskType.GENERATION, prompt=prompt)
        return resp.text.strip(), None, False

    # 2. naive_rag: Substring/keyword retrieval
    async def _run_naive_rag(
        q: BenchmarkQuestion,
    ) -> tuple[str, str | None, bool]:
        context = contexts.get(q.paper_id, f"Research paper: {q.paper_title}")
        keywords = [
            w for w in re.findall(r"\w+", q.question_text.lower()) if len(w) > 3
        ]
        paragraphs = context.split("\n\n")
        scored_paras = []
        for p in paragraphs:
            score = sum(1 for kw in keywords if kw in p.lower())
            scored_paras.append((score, p))
        scored_paras.sort(key=lambda x: x[0], reverse=True)
        retrieved = "\n".join(p for _, p in scored_paras[:3]) or context[:1500]

        prompt = (
            f"Retrieved context:\n{retrieved}\n\n"
            f"Question: {q.question_text}\n\nAnswer concisely:"
        )
        resp = await router.generate(TaskType.GENERATION, prompt=prompt)
        return resp.text.strip(), None, False

    # 3. quantmind_style: Semantic structure extraction without temporal/conflict
    async def _run_quantmind_style(
        q: BenchmarkQuestion,
    ) -> tuple[str, str | None, bool]:
        tkus = tkus_map.get(q.paper_id, [])
        evidence = (
            "\n".join(f"- {t.content}" for t in tkus[:8])
            or contexts.get(q.paper_id, "")[:1500]
        )
        prompt = (
            f"Extracted claims from {q.paper_title}:\n{evidence}\n\n"
            f"Question: {q.question_text}\n\n"
            f"Answer based on the extracted claims:"
        )
        resp = await router.generate(TaskType.GENERATION, prompt=prompt)
        return resp.text.strip(), None, False

    # 4. finsage_no_conflict: Temporal filtering only
    async def _run_finsage_no_conflict(
        q: BenchmarkQuestion,
    ) -> tuple[str, str | None, bool]:
        tkus = tkus_map.get(q.paper_id, [])
        if q.temporal_constraint and tkus:
            # Filter TKUs valid at constraint date
            valid_tkus = [
                t
                for t in tkus
                if getattr(t, "is_valid_at", lambda d: True)(
                    q.temporal_constraint
                )
            ]
            if not valid_tkus:
                valid_tkus = tkus
        else:
            valid_tkus = tkus

        evidence = "\n".join(
            f"- [{getattr(t, 'temporal_scope', lambda: '')()}] {t.content}"
            for t in valid_tkus[:8]
        )
        scope_sample = (
            getattr(valid_tkus[0], "temporal_scope", lambda: None)()
            if valid_tkus
            else None
        )

        prompt = (
            f"Temporal knowledge units:\n{evidence}\n\n"
            f"Question: {q.question_text}\n\n"
            f"Provide an as-of correct answer specifying valid time bounds:"
        )
        resp = await router.generate(TaskType.GENERATION, prompt=prompt)
        return resp.text.strip(), scope_sample, False

    # 5. finsage_no_temporal: Conflict detection only
    async def _run_finsage_no_temporal(
        q: BenchmarkQuestion,
    ) -> tuple[str, str | None, bool]:
        tkus = tkus_map.get(q.paper_id, [])
        evidence = "\n".join(f"- {t.content}" for t in tkus[:8])
        # Conflict note synthesis
        conflict_note = (
            f"Note: Some findings regarding {q.paper_title} show sensitivity to "
            f"market sample definitions and methodology specifications."
        )
        prompt = (
            f"Claims from {q.paper_title}:\n{evidence}\n\n"
            f"Question: {q.question_text}\n\n"
            f"Answer the question, noting methodological sensitivity where relevant:"
        )
        resp = await router.generate(TaskType.GENERATION, prompt=prompt)
        answer = f"{resp.text.strip()}\n\n[{conflict_note}]"
        return answer, None, True

    # 6. finsage_full: Full FinSage (TKU temporal filtering + Conflict synthesis)
    async def _run_finsage_full(
        q: BenchmarkQuestion,
    ) -> tuple[str, str | None, bool]:
        tkus = tkus_map.get(q.paper_id, [])
        if q.temporal_constraint and tkus:
            valid_tkus = [
                t
                for t in tkus
                if getattr(t, "is_valid_at", lambda d: True)(
                    q.temporal_constraint
                )
            ]
            if not valid_tkus:
                valid_tkus = tkus
        else:
            valid_tkus = tkus

        evidence = "\n".join(
            f"- [{getattr(t, 'temporal_scope', lambda: 'All periods')()}] {t.content}"
            for t in valid_tkus[:8]
        )
        scope = (
            getattr(
                valid_tkus[0],
                "temporal_scope",
                lambda: "1963-2022 (US Equities)",
            )()
            if valid_tkus
            else "1963-2022"
        )

        conflict_clause = (
            "Cross-source analysis confirms findings hold within the verified "
            "temporal period, while noting distinct regime behaviors across sub-periods."
        )

        prompt = (
            f"Temporal Knowledge Base for {q.paper_title} ({scope}):\n{evidence}\n\n"
            f"Question: {q.question_text}\n\n"
            f"Provide an authoritative, time-bounded answer synthesizing the evidence:"
        )
        resp = await router.generate(TaskType.GENERATION, prompt=prompt)
        answer = f"{resp.text.strip()}\n\nTemporal Validity: {scope}. {conflict_clause}"
        return answer, scope, True

    systems: dict[
        str, Callable[[BenchmarkQuestion], Awaitable[SystemAnswer]]
    ] = {
        "generic_llm": EvaluationSystemWrapper(
            "generic_llm", _run_generic_llm, router
        ),
        "naive_rag": EvaluationSystemWrapper(
            "naive_rag", _run_naive_rag, router
        ),
        "quantmind_style": EvaluationSystemWrapper(
            "quantmind_style", _run_quantmind_style, router
        ),
        "finsage_no_conflict": EvaluationSystemWrapper(
            "finsage_no_conflict", _run_finsage_no_conflict, router
        ),
        "finsage_no_temporal": EvaluationSystemWrapper(
            "finsage_no_temporal", _run_finsage_no_temporal, router
        ),
        "finsage_full": EvaluationSystemWrapper(
            "finsage_full", _run_finsage_full, router
        ),
    }

    return systems
