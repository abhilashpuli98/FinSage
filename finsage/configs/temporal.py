"""Temporal paper-flow configuration.

``TemporalPaperCfg`` is a ``BaseFlowCfg`` subclass that controls the
temporal knowledge extraction pipeline.  ``TemporalPaperFlow(cfg).build``
dispatches on this cfg type to produce a ``TKUCollection``.
"""

from pydantic import Field

from finsage.configs.base import BaseFlowCfg


class TemporalPaperCfg(BaseFlowCfg):
    """Configuration for ``TemporalPaperFlow.build``.

    Selecting this cfg type at ``TemporalPaperFlow`` construction time
    binds the extraction model, section chunking, and deduplication
    policy for the entire batch.

    ``model`` sets the API model used for extraction.  When
    ``use_local_if_available`` is True and a local Ollama server is
    reachable, extraction falls back to ``local_model`` for cost
    savings.
    """

    model: str = "gpt-4o-mini"
    local_model: str = "llama3.2"
    use_local_if_available: bool = True
    max_tkus_per_section: int = Field(default=20, ge=1)
    semantic_similarity_threshold: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Cosine similarity above which two same-source TKUs are deduplicated.",
    )
    min_section_length: int = Field(
        default=150,
        ge=0,
        description="Sections shorter than this (in characters) are skipped.",
    )
    chunk_chars: int = Field(
        default=1800,
        ge=200,
        description="Target chunk size (in characters) for extraction prompts.",
    )
