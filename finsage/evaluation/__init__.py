"""finsage.evaluation — FinBench-50 benchmark and automated evaluation.

Apex-layer module providing:
- Benchmark models: ``BenchmarkQuestion``, ``QuestionType``, ``SystemAnswer``, ``EvalResult``
- Question construction: ``BenchmarkConstructor``
- Automated scoring and statistical analysis: ``AutoEvaluator``
- Multi-system comparison wrappers: ``build_evaluation_systems``
- LLM quality-cost analysis: ``LLMQualityBenchmark``
"""

from finsage.evaluation.benchmark import BenchmarkConstructor
from finsage.evaluation.evaluator import AutoEvaluator
from finsage.evaluation.llm_benchmark import LLMQualityBenchmark
from finsage.evaluation.models import (
    BenchmarkQuestion,
    EvalResult,
    QuestionType,
    SystemAnswer,
)
from finsage.evaluation.systems import build_evaluation_systems

__all__ = [
    "AutoEvaluator",
    "BenchmarkConstructor",
    "BenchmarkQuestion",
    "EvalResult",
    "LLMQualityBenchmark",
    "QuestionType",
    "SystemAnswer",
    "build_evaluation_systems",
]
