"""finsage.knowledge — data standard for extracted financial knowledge.

The standard defines three shapes that share `BaseKnowledge`:

- `FlattenKnowledge` — atomic cards (`News`, `Earnings`, `Factor`, `Thesis`).
- `TreeKnowledge` — conventional hierarchical knowledge artifacts.
- `GraphKnowledge` — cross-item edges (placeholder, not implemented).

Every concrete subclass is frozen Pydantic v2 with ``extra="forbid"``,
suitable for ``Agent(output_type=...)`` and round-tripping through JSON.
Paper sources and artifacts are separate immutable models. Search projection
text and vectors remain rebuildable library-owned data.
"""

from finsage.knowledge._base import (
    ArtifactMeta,
    BaseKnowledge,
    Citation,
    ExtractionRef,
    SourceRef,
)
from finsage.knowledge._flatten import FlattenKnowledge
from finsage.knowledge._graph import GraphKnowledge
from finsage.knowledge._tree import (
    StructureTree,
    StructureTreeValidationError,
    TreeKnowledge,
    TreeNode,
)
from finsage.knowledge.conflict import (
    ConflictRecord,
    ConflictType,
    ResolutionStatus,
)
from finsage.knowledge.earnings import Earnings
from finsage.knowledge.factor import Factor
from finsage.knowledge.news import News
from finsage.knowledge.paper import (
    ArtifactLocator,
    LegacyPaper,
    PaperArtifact,
    PaperArtifactKind,
    PaperAssetInput,
    PaperAssetRef,
    PaperBoundingBox,
    PaperChunk,
    PaperChunkingConfig,
    PaperChunkInput,
    PaperChunkSet,
    PaperCitation,
    PaperCitationDraft,
    PaperCitationValidationError,
    PaperGlobalSummary,
    PaperPageInput,
    PaperParsedBlock,
    PaperParsedManifest,
    PaperParsedPage,
    PaperSemanticResult,
    PaperSourceFacts,
    PaperSourceRevision,
    PaperSourceSpan,
    PaperStructureNodeDraft,
    PaperStructureProducer,
    PaperStructureTree,
    PaperStructureTreeDraft,
    PaperSummaryProducer,
    ResolvedPaperArtifact,
)
from finsage.knowledge.temporal import (
    FinanceDomain,
    KnowledgeType,
    TemporalDecay,
    TemporalKnowledgeUnit,
    TKUCollection,
)
from finsage.knowledge.thesis import Thesis

__all__ = [
    # Base
    "ArtifactMeta",
    "BaseKnowledge",
    "Citation",
    "ExtractionRef",
    "SourceRef",
    # Shapes
    "FlattenKnowledge",
    "GraphKnowledge",
    "StructureTree",
    "StructureTreeValidationError",
    "TreeKnowledge",
    "TreeNode",
    # Concrete
    "Earnings",
    "Factor",
    "News",
    "ArtifactLocator",
    "LegacyPaper",
    "PaperArtifact",
    "PaperArtifactKind",
    "PaperAssetInput",
    "PaperAssetRef",
    "PaperBoundingBox",
    "PaperChunk",
    "PaperChunkInput",
    "PaperChunkingConfig",
    "PaperChunkSet",
    "PaperCitation",
    "PaperCitationDraft",
    "PaperCitationValidationError",
    "PaperSemanticResult",
    "PaperGlobalSummary",
    "PaperPageInput",
    "PaperParsedBlock",
    "PaperParsedManifest",
    "PaperParsedPage",
    "PaperSourceFacts",
    "PaperSourceRevision",
    "PaperSourceSpan",
    "PaperStructureNodeDraft",
    "PaperStructureProducer",
    "PaperStructureTree",
    "PaperStructureTreeDraft",
    "PaperSummaryProducer",
    "ResolvedPaperArtifact",
    # Temporal (C1: TKU)
    "FinanceDomain",
    "KnowledgeType",
    "TKUCollection",
    "TemporalDecay",
    "TemporalKnowledgeUnit",
    # Conflict (C2: CSCG)
    "ConflictRecord",
    "ConflictType",
    "ResolutionStatus",
    "Thesis",
]
