"""finsage.configs — public operation configuration + input types.

Public operations use a ``BaseFlowCfg`` subclass and a typed input model or
discriminated union. All cfg / input classes live here so that:

  - YAML / CLI users see a single import surface,
  - JSON schemas can be exported uniformly (for IDE autocomplete),
  - the magic-input resolver (PR5) has one introspection target.
"""

from finsage.configs.base import BaseFlowCfg, BaseInput
from finsage.configs.earnings import EarningsFlowCfg, EarningsInput
from finsage.configs.news import NewsCollectionCfg, NewsWindow
from finsage.configs.ollama import (
    ModelTier,
    OllamaModelCfg,
    RouterCfg,
    TaskType,
)
from finsage.configs.paper import PaperInput, PaperSemanticCfg
from finsage.configs.retrieval import RetrievalCfg
from finsage.configs.structure import PaperStructureCfg
from finsage.configs.temporal import TemporalPaperCfg

__all__ = [
    "BaseFlowCfg",
    "BaseInput",
    "EarningsFlowCfg",
    "EarningsInput",
    "NewsCollectionCfg",
    "NewsWindow",
    "ModelTier",
    "OllamaModelCfg",
    "PaperSemanticCfg",
    "PaperInput",
    "PaperStructureCfg",
    "RetrievalCfg",
    "RouterCfg",
    "TaskType",
    "TemporalPaperCfg",
]
