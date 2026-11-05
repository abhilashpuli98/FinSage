"""FinBench-50 benchmark generation and management.

Provides ``BenchmarkConstructor`` to generate standardized factual, analytical,
and temporal questions from financial research papers and TKUCollections.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from finsage.configs.ollama import TaskType
from finsage.configs.paper import ArxivIdentifier, LocalFilePath
from finsage.configs.temporal import TemporalPaperCfg
from finsage.evaluation.models import BenchmarkQuestion, QuestionType
from finsage.flows.temporal_paper import TemporalPaperFlow
from finsage.knowledge.temporal import FinanceDomain, TKUCollection
from finsage.mind.router import LLMRouter, safe_json_parse

_LOGGER = logging.getLogger(__name__)

__all__ = ["BenchmarkConstructor"]

_FACTUAL_SYSTEM = (
    "You are creating evaluation questions for a financial knowledge benchmark. "
    "Create factual questions with specific, verifiable answers extractable "
    "from the paper."
)

_FACTUAL_USER = """\
Paper: {title}
Key facts: {facts}

Generate exactly 3 factual questions. Each must have a specific, unambiguous \
answer that can be verified against the paper. Cover: (1) a specific \
number/statistic, (2) a formula or model definition, (3) a methodology detail.

Return JSON array:
[{{"question": "...", "ground_truth": "exact answer", "difficulty": "easy|medium|hard"}}]"""

_ANALYTICAL_USER = """\
Paper: {title}
Key claims: {claims}

Generate exactly 2 analytical questions requiring reasoning across multiple \
facts. These require synthesis, not just lookup. No ground truth (open-ended).

Return JSON array:
[{{"question": "...", "expected_reasoning": "what a good answer should cover", \
"difficulty": "medium|hard"}}]"""

_TEMPORAL_USER = """\
Paper: {title}
Data sample period: {sample_period_start} to {sample_period_end}

Generate exactly 1 temporal question that asks what was known or true at a \
specific date. The question must be answerable only if the system knows \
WHEN the evidence applies.
Example: 'As of {sample_period_start}, what did {title} find about factor \
performance in the target market?'

Return JSON: {{"question": "...", "target_date": "YYYY-MM-DD", \
"ground_truth": "...", "difficulty": "medium"}}"""


def _parse_target_date(raw_date: str | None) -> datetime | None:
    """Safely parse an ISO or year-month-day date string."""
    if not raw_date or not isinstance(raw_date, str):
        return None
    raw = raw_date.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _resolve_domain(raw: object) -> FinanceDomain:
    """Convert a raw domain string to a ``FinanceDomain`` enum.

    Falls back to ``FACTOR_INVESTING`` when the value cannot be mapped.
    """
    if isinstance(raw, FinanceDomain):
        return raw
    if isinstance(raw, str):
        key = raw.strip().upper()
        for member in FinanceDomain:
            if member.name == key or member.value == raw.strip().lower():
                return member
    return FinanceDomain.FACTOR_INVESTING


class BenchmarkConstructor:
    """Constructs evaluation questions for FinBench-50.

    Uses LLM routing to extract 6 standardized questions per paper:
    - 3 Factual questions (with exact verifiable ground truth)
    - 2 Analytical questions (requiring multi-fact reasoning)
    - 1 Temporal question (evaluating point-in-time correctness)
    """

    def __init__(self, router: LLMRouter) -> None:
        """Initialize with an LLM router for question generation.

        Args:
            router: Bound router for local or API inference.
        """
        self._router = router

    @property
    def router(self) -> LLMRouter:
        """The router bound to this constructor."""
        return self._router

    async def generate_questions(
        self,
        parsed_text: str,
        tku_collection: TKUCollection,
        paper_id: str,
        paper_title: str,
        domain: FinanceDomain,
        sample_period_start: str = "unknown",
        sample_period_end: str = "present",
    ) -> list[BenchmarkQuestion]:
        """Generate exactly 6 evaluation questions for one paper.

        Args:
            parsed_text: Cleaned full text of the paper.
            tku_collection: Extracted temporal knowledge units for the paper.
            paper_id: Unique paper identifier.
            paper_title: Title of the research paper.
            domain: Target finance domain.
            sample_period_start: Start year/date of empirical data.
            sample_period_end: End year/date of empirical data.

        Returns:
            List of 6 ``BenchmarkQuestion`` objects (3 factual, 2 analytical,
            1 temporal).
        """
        questions: list[BenchmarkQuestion] = []

        # Sort TKUs by composite confidence to pass highest quality evidence.
        sorted_tkus = sorted(
            tku_collection.tkus,
            key=lambda t: t.composite_confidence(),
            reverse=True,
        )
        top_10 = [t.content for t in sorted_tkus[:10]] or [parsed_text[:500]]
        top_5 = [t.content for t in sorted_tkus[:5]] or [parsed_text[:400]]

        # 1. Generate 3 Factual Questions.
        factual_prompt = _FACTUAL_USER.format(title=paper_title, facts=top_10)
        try:
            resp_fact = await self._router.generate(
                TaskType.EXTRACTION,
                prompt=factual_prompt,
                system_prompt=_FACTUAL_SYSTEM,
            )
            parsed_fact = safe_json_parse(resp_fact.text)
        except Exception as exc:
            _LOGGER.warning("Factual question generation failed: %s", exc)
            parsed_fact = None

        if isinstance(parsed_fact, list):
            for item in parsed_fact[:3]:
                if isinstance(item, dict) and "question" in item:
                    diff = item.get("difficulty", "medium")
                    if diff not in ("easy", "medium", "hard"):
                        diff = "medium"
                    questions.append(
                        BenchmarkQuestion(
                            question_id=str(uuid4()),
                            paper_id=paper_id,
                            paper_title=paper_title,
                            domain=domain,
                            question_text=str(item["question"]),
                            question_type=QuestionType.FACTUAL,
                            ground_truth=str(item.get("ground_truth", "")),
                            difficulty=diff,
                        )
                    )

        # Fallback if factual generation was incomplete:
        while len(questions) < 3:
            idx = len(questions) + 1
            fact_claim = top_10[min(idx - 1, len(top_10) - 1)]
            questions.append(
                BenchmarkQuestion(
                    question_id=str(uuid4()),
                    paper_id=paper_id,
                    paper_title=paper_title,
                    domain=domain,
                    question_text=(
                        f"What empirical finding or statistic is reported in "
                        f"'{paper_title}' regarding: {fact_claim[:100]}?"
                    ),
                    question_type=QuestionType.FACTUAL,
                    ground_truth=fact_claim,
                    difficulty="medium",
                )
            )

        # 2. Generate 2 Analytical Questions.
        analytical_prompt = _ANALYTICAL_USER.format(
            title=paper_title, claims=top_5
        )
        try:
            resp_ana = await self._router.generate(
                TaskType.EXTRACTION,
                prompt=analytical_prompt,
                system_prompt=_FACTUAL_SYSTEM,
            )
            parsed_ana = safe_json_parse(resp_ana.text)
        except Exception as exc:
            _LOGGER.warning("Analytical question generation failed: %s", exc)
            parsed_ana = None

        analytical_count = 0
        if isinstance(parsed_ana, list):
            for item in parsed_ana[:2]:
                if isinstance(item, dict) and "question" in item:
                    diff = item.get("difficulty", "hard")
                    if diff not in ("easy", "medium", "hard"):
                        diff = "hard"
                    questions.append(
                        BenchmarkQuestion(
                            question_id=str(uuid4()),
                            paper_id=paper_id,
                            paper_title=paper_title,
                            domain=domain,
                            question_text=str(item["question"]),
                            question_type=QuestionType.ANALYTICAL,
                            ground_truth=None,
                            expected_reasoning=str(
                                item.get("expected_reasoning", "")
                            ),
                            difficulty=diff,
                        )
                    )
                    analytical_count += 1

        while analytical_count < 2:
            analytical_count += 1
            questions.append(
                BenchmarkQuestion(
                    question_id=str(uuid4()),
                    paper_id=paper_id,
                    paper_title=paper_title,
                    domain=domain,
                    question_text=(
                        f"How does the methodology in '{paper_title}' "
                        f"reconcile the factor return variations across regimes?"
                    ),
                    question_type=QuestionType.ANALYTICAL,
                    ground_truth=None,
                    expected_reasoning=(
                        "Synthesis of factor construction, econometric "
                        "controls, and multi-period performance stability."
                    ),
                    difficulty="hard",
                )
            )

        # 3. Generate 1 Temporal Question.
        temporal_prompt = _TEMPORAL_USER.format(
            title=paper_title,
            sample_period_start=sample_period_start,
            sample_period_end=sample_period_end,
        )
        try:
            resp_temp = await self._router.generate(
                TaskType.EXTRACTION,
                prompt=temporal_prompt,
                system_prompt=_FACTUAL_SYSTEM,
            )
            parsed_temp = safe_json_parse(resp_temp.text)
        except Exception as exc:
            _LOGGER.warning("Temporal question generation failed: %s", exc)
            parsed_temp = None

        temporal_q_added = False
        if isinstance(parsed_temp, dict) and "question" in parsed_temp:
            diff = parsed_temp.get("difficulty", "medium")
            if diff not in ("easy", "medium", "hard"):
                diff = "medium"
            target_date = _parse_target_date(parsed_temp.get("target_date"))
            questions.append(
                BenchmarkQuestion(
                    question_id=str(uuid4()),
                    paper_id=paper_id,
                    paper_title=paper_title,
                    domain=domain,
                    question_text=str(parsed_temp["question"]),
                    question_type=QuestionType.TEMPORAL,
                    ground_truth=str(parsed_temp.get("ground_truth", "")),
                    temporal_constraint=target_date,
                    difficulty=diff,
                )
            )
            temporal_q_added = True

        if not temporal_q_added:
            fallback_target = _parse_target_date(sample_period_start)
            questions.append(
                BenchmarkQuestion(
                    question_id=str(uuid4()),
                    paper_id=paper_id,
                    paper_title=paper_title,
                    domain=domain,
                    question_text=(
                        f"As of {sample_period_start}, what was established "
                        f"regarding the core hypotheses analyzed in '{paper_title}'?"
                    ),
                    question_type=QuestionType.TEMPORAL,
                    ground_truth=(
                        f"Empirical evidence across the sample period starting "
                        f"{sample_period_start}."
                    ),
                    temporal_constraint=fallback_target,
                    difficulty="medium",
                )
            )

        return questions[:6]

    async def build_finbench_50(
        self,
        paper_list: list[dict[str, Any]],
        output_path: Path,
    ) -> list[BenchmarkQuestion]:
        """Construct or augment the FinBench-50 benchmark file.

        Args:
            paper_list: List of paper descriptor dictionaries.
            output_path: Target path to store the benchmark questions JSON.

        Returns:
            The complete list of ``BenchmarkQuestion`` objects.
        """
        all_questions: list[BenchmarkQuestion] = []
        cached_ids: set[str] = set()

        if output_path.exists():
            try:
                data = json.loads(output_path.read_text(encoding="utf-8"))
                for item in data:
                    if isinstance(item, dict):
                        q = BenchmarkQuestion.model_validate(item)
                        all_questions.append(q)
                        cached_ids.add(q.paper_id)
                _LOGGER.info(
                    "Loaded %d cached questions across %d papers from %s",
                    len(all_questions),
                    len(cached_ids),
                    output_path,
                )
            except Exception as exc:
                _LOGGER.warning(
                    "Failed to read existing benchmark file %s: %s",
                    output_path,
                    exc,
                )

        flow = TemporalPaperFlow(TemporalPaperCfg())

        for paper in paper_list:
            p_id = str(paper["paper_id"])
            if p_id in cached_ids:
                continue

            title = str(paper.get("title", f"Paper {p_id}"))
            domain = _resolve_domain(paper.get("domain"))
            sample_start = str(paper.get("sample_start", "2000"))
            sample_end = str(paper.get("sample_end", "2020"))

            # Build TKU collection for the paper.
            if "arxiv_id" in paper:
                paper_input = ArxivIdentifier(id=str(paper["arxiv_id"]))
            elif "local_path" in paper:
                paper_input = LocalFilePath(path=Path(paper["local_path"]))
            else:
                paper_input = ArxivIdentifier(id=p_id)

            try:
                collection = await flow.build(paper_input)
                parsed_text = " ".join(t.content for t in collection.tkus)
            except Exception as exc:
                _LOGGER.info(
                    "TemporalPaperFlow extraction fallback for %s: %s",
                    p_id,
                    exc,
                )
                collection = TKUCollection(
                    document_id=p_id,
                    source_title=title,
                    tkus=(),
                )
                parsed_text = title

            paper_questions = await self.generate_questions(
                parsed_text=parsed_text,
                tku_collection=collection,
                paper_id=p_id,
                paper_title=title,
                domain=domain,
                sample_period_start=sample_start,
                sample_period_end=sample_end,
            )

            all_questions.extend(paper_questions)
            cached_ids.add(p_id)

            # Persist checkpoint after each paper.
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    [q.model_dump(mode="json") for q in all_questions],
                    indent=2,
                ),
                encoding="utf-8",
            )

        return all_questions
