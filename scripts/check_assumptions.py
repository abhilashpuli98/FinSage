"""Assumption checker — discover existing classes, fields, methods."""

import importlib
import inspect
import json
import pathlib

ROOT = pathlib.Path(".").resolve()
print(f"Project root: {ROOT}")
print()

checks = {
    "finsage.knowledge.temporal": [
        "TemporalKnowledgeUnit",
        "TKUCollection",
        "FinanceDomain",
        "KnowledgeType",
        "TemporalDecay",
    ],
    "finsage.knowledge.conflict": [
        "ConflictRecord",
        "ConflictType",
        "ResolutionStatus",
    ],
    "finsage.mind.router": ["LLMRouter", "LLMResponse", "CostTracker"],
    "finsage.mind.conflict": ["ConflictDetector", "ConflictResolver"],
    "finsage.mind.retrieval": ["AgenticRetriever"],
    "finsage.configs.ollama": [
        "OllamaModelCfg",
        "RouterCfg",
        "TaskType",
        "ModelTier",
    ],
    "finsage.flows.temporal_paper": ["TemporalPaperFlow"],
    "finsage.evaluation.models": [
        "BenchmarkQuestion",
        "SystemAnswer",
        "EvalResult",
        "QuestionType",
    ],
    "finsage.evaluation.benchmark": ["BenchmarkConstructor"],
    "finsage.evaluation.evaluator": ["AutoEvaluator"],
    "finsage.evaluation.systems": ["build_evaluation_systems"],
    "finsage.evaluation.llm_benchmark": ["LLMQualityBenchmark"],
}

found: dict[str, dict] = {}
missing: list[str] = []

for mod_name, names in checks.items():
    try:
        mod = importlib.import_module(mod_name)
        for name in names:
            if hasattr(mod, name):
                cls = getattr(mod, name)
                info: dict = {
                    "is_class": inspect.isclass(cls),
                    "is_function": inspect.isfunction(cls),
                }
                if inspect.isclass(cls):
                    info["methods"] = [
                        m for m in dir(cls) if not m.startswith("_")
                    ][:10]
                    info["bases"] = [
                        b.__name__ for b in getattr(cls, "__bases__", [])
                    ]
                    if hasattr(cls, "model_fields"):
                        info["fields"] = list(cls.model_fields.keys())
                    elif hasattr(cls, "__dataclass_fields__"):
                        info["fields"] = list(cls.__dataclass_fields__.keys())
                found[f"{mod_name}.{name}"] = info
            else:
                missing.append(f"{mod_name}.{name}")
    except ImportError as e:
        missing.append(f"{mod_name} IMPORT FAILED: {e}")

print("=== FOUND ===")
for k, v in found.items():
    print(f"  {k}:")
    print(f"    class={v.get('is_class')}, func={v.get('is_function')}")
    if "fields" in v:
        print(f"    fields={v['fields']}")
    if "methods" in v:
        print(f"    methods={v['methods']}")
    if "bases" in v:
        print(f"    bases={v['bases']}")

print()
print("=== MISSING ===")
for m in missing:
    print(f"  {m}")

# Dump full data to JSON for _assumptions.py generation
out_path = ROOT / "scripts" / "_assumption_data.json"
out_path.write_text(json.dumps({"found": found, "missing": missing}, indent=2))
print(f"\nRaw data saved to {out_path}")
