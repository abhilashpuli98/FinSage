"""Tests for the v2 scaffold — validates assumptions file, feature flags, and imports."""

import pathlib
import unittest

import yaml


class TestV2ScaffoldImport(unittest.TestCase):
    """Ensure finsage.v2 package imports cleanly."""

    def test_v2_package_imports(self) -> None:
        import finsage.v2

        self.assertEqual(finsage.v2.__version__, "2.0.0")
        self.assertIsInstance(finsage.v2.__all__, list)
        self.assertEqual(len(finsage.v2.__all__), 0)

    def test_assumptions_module_imports(self) -> None:
        from finsage.v2._assumptions import (
            EXISTING_TKU_CLASS,
            EXISTING_TKU_FIELDS,
            EXISTING_TKU_MODULE,
            V2_PACKAGE_ROOT,
        )

        self.assertEqual(EXISTING_TKU_CLASS, "TemporalKnowledgeUnit")
        self.assertEqual(EXISTING_TKU_MODULE, "finsage.knowledge.temporal")
        self.assertIn("tku_id", EXISTING_TKU_FIELDS)
        self.assertIn("content", EXISTING_TKU_FIELDS)
        self.assertIn("content_hash", EXISTING_TKU_FIELDS)
        self.assertEqual(V2_PACKAGE_ROOT, "finsage.v2")


class TestAssumptionsAccuracy(unittest.TestCase):
    """Cross-validate _assumptions.py against the actual live classes."""

    def test_tku_fields_match(self) -> None:
        from finsage.knowledge.temporal import TemporalKnowledgeUnit
        from finsage.v2._assumptions import EXISTING_TKU_FIELDS

        actual = list(TemporalKnowledgeUnit.model_fields.keys())
        self.assertEqual(sorted(EXISTING_TKU_FIELDS), sorted(actual))

    def test_conflict_record_fields_match(self) -> None:
        from finsage.knowledge.conflict import ConflictRecord
        from finsage.v2._assumptions import EXISTING_CONFLICT_RECORD_FIELDS

        actual = list(ConflictRecord.model_fields.keys())
        self.assertEqual(
            sorted(EXISTING_CONFLICT_RECORD_FIELDS), sorted(actual)
        )

    def test_ollama_cfg_fields_match(self) -> None:
        import dataclasses

        from finsage.configs.ollama import OllamaModelCfg
        from finsage.v2._assumptions import EXISTING_OLLAMA_CFG_FIELDS

        actual = [f.name for f in dataclasses.fields(OllamaModelCfg)]
        self.assertEqual(sorted(EXISTING_OLLAMA_CFG_FIELDS), sorted(actual))

    def test_router_cfg_fields_match(self) -> None:
        import dataclasses

        from finsage.configs.ollama import RouterCfg
        from finsage.v2._assumptions import EXISTING_ROUTER_CFG_FIELDS

        actual = [f.name for f in dataclasses.fields(RouterCfg)]
        self.assertEqual(sorted(EXISTING_ROUTER_CFG_FIELDS), sorted(actual))

    def test_llm_response_fields_match(self) -> None:
        import dataclasses

        from finsage.mind.router import LLMResponse
        from finsage.v2._assumptions import EXISTING_RESPONSE_FIELDS

        actual = [f.name for f in dataclasses.fields(LLMResponse)]
        self.assertEqual(sorted(EXISTING_RESPONSE_FIELDS), sorted(actual))

    def test_benchmark_question_fields_match(self) -> None:
        from finsage.evaluation.models import BenchmarkQuestion
        from finsage.v2._assumptions import EXISTING_BENCHMARK_Q_FIELDS

        actual = list(BenchmarkQuestion.model_fields.keys())
        self.assertEqual(sorted(EXISTING_BENCHMARK_Q_FIELDS), sorted(actual))

    def test_system_answer_fields_match(self) -> None:
        from finsage.evaluation.models import SystemAnswer
        from finsage.v2._assumptions import EXISTING_SYSTEM_ANSWER_FIELDS

        actual = list(SystemAnswer.model_fields.keys())
        self.assertEqual(sorted(EXISTING_SYSTEM_ANSWER_FIELDS), sorted(actual))

    def test_eval_result_fields_match(self) -> None:
        from finsage.evaluation.models import EvalResult
        from finsage.v2._assumptions import EXISTING_EVAL_RESULT_FIELDS

        actual = list(EvalResult.model_fields.keys())
        self.assertEqual(sorted(EXISTING_EVAL_RESULT_FIELDS), sorted(actual))

    def test_finance_domain_values_match(self) -> None:
        from finsage.knowledge.temporal import FinanceDomain
        from finsage.v2._assumptions import EXISTING_FINANCE_DOMAIN_VALUES

        actual = [m.name for m in FinanceDomain]
        self.assertEqual(sorted(EXISTING_FINANCE_DOMAIN_VALUES), sorted(actual))

    def test_conflict_type_values_match(self) -> None:
        from finsage.knowledge.conflict import ConflictType
        from finsage.v2._assumptions import EXISTING_CONFLICT_TYPE_VALUES

        actual = [m.name for m in ConflictType]
        self.assertEqual(sorted(EXISTING_CONFLICT_TYPE_VALUES), sorted(actual))

    def test_task_type_values_match(self) -> None:
        from finsage.configs.ollama import TaskType
        from finsage.v2._assumptions import EXISTING_TASK_TYPE_VALUES

        actual = [m.name for m in TaskType]
        self.assertEqual(sorted(EXISTING_TASK_TYPE_VALUES), sorted(actual))

    def test_model_tier_values_match(self) -> None:
        from finsage.configs.ollama import ModelTier
        from finsage.v2._assumptions import EXISTING_MODEL_TIER_VALUES

        actual = [m.name for m in ModelTier]
        self.assertEqual(sorted(EXISTING_MODEL_TIER_VALUES), sorted(actual))


class TestV2FeaturesConfig(unittest.TestCase):
    """Validate configs/v2_features.yaml structure and defaults."""

    def setUp(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        cfg_path = repo_root / "configs" / "v2_features.yaml"
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

    def test_all_feature_flags_present(self) -> None:
        expected = [
            "probabilistic_tkus",
            "agentic_retrieval",
            "adversarial_robustness",
            "temporal_graph_embed",
            "bayesian_conflict",
            "structured_api",
            "counterfactual",
            "multi_hop_reasoning",
        ]
        actual = list(self.cfg["v2_features"].keys())
        self.assertEqual(sorted(expected), sorted(actual))

    def test_all_flags_false_by_default(self) -> None:
        for flag, value in self.cfg["v2_features"].items():
            with self.subTest(flag=flag):
                self.assertIs(value, False, f"{flag} should default to false")

    def test_v2_settings_keys_present(self) -> None:
        expected_sections = [
            "p_tku",
            "agentic",
            "adversarial",
            "graph_embed",
            "bayesian",
            "api",
            "counterfactual",
            "multi_hop",
        ]
        actual = list(self.cfg["v2_settings"].keys())
        self.assertEqual(sorted(expected_sections), sorted(actual))

    def test_p_tku_settings(self) -> None:
        p = self.cfg["v2_settings"]["p_tku"]
        self.assertEqual(p["uncertainty_model"], "gaussian")
        self.assertAlmostEqual(p["min_confidence"], 0.05)
        self.assertAlmostEqual(p["bayesian_update_prior"], 0.5)

    def test_agentic_settings(self) -> None:
        a = self.cfg["v2_settings"]["agentic"]
        self.assertEqual(a["max_iterations"], 3)
        self.assertAlmostEqual(a["gap_threshold"], 0.6)
        self.assertEqual(a["reflection_model"], "same")

    def test_api_settings(self) -> None:
        api = self.cfg["v2_settings"]["api"]
        self.assertEqual(api["host"], "127.0.0.1")
        self.assertEqual(api["port"], 8001)
        self.assertIs(api["enable_cors"], True)

    def test_multi_hop_settings(self) -> None:
        mh = self.cfg["v2_settings"]["multi_hop"]
        self.assertEqual(mh["max_hops"], 4)
        self.assertEqual(mh["hop_k"], 5)
        self.assertEqual(mh["merge_strategy"], "weighted_union")


if __name__ == "__main__":
    unittest.main()
