"""LLM providers and prompts."""

from __future__ import annotations

from kg_construction.infrastructure.llm.label_proposal import (
    LabelProposalResult,
    LabelProposalSettings,
    OllamaLabelProposalExtractor,
    ProposedEntityLabel,
)
from kg_construction.infrastructure.llm.ollama import (
    OllamaConfig,
    OllamaOutputError,
    OllamaProvider,
)
from kg_construction.infrastructure.llm.prompts import (
    build_graph_extraction_prompt,
    build_label_proposal_prompt,
    build_relation_typing_prompt,
    build_repair_prompt,
)

__all__ = [
    "LabelProposalResult",
    "LabelProposalSettings",
    "OllamaLabelProposalExtractor",
    "OllamaConfig",
    "OllamaOutputError",
    "OllamaProvider",
    "ProposedEntityLabel",
    "build_graph_extraction_prompt",
    "build_label_proposal_prompt",
    "build_relation_typing_prompt",
    "build_repair_prompt",
]
