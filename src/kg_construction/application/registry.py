"""Compatibility re-exports for :mod:`kg_construction.certification.registry`.

The capability registry is certification-owned. Import from the canonical
certification module in new code; this facade preserves existing application
imports for one migration cycle.
"""

from kg_construction.certification.registry import (
    SUPPORT_LEVEL_DEFINITIONS,
    SUPPORT_LEVEL_LABELS,
    PluginRegistry,
    SupportLevel,
    TechniqueDescriptor,
    TechniqueStatus,
    default_registry,
    research_registry,
)

__all__ = [
    "PluginRegistry",
    "SUPPORT_LEVEL_DEFINITIONS",
    "SUPPORT_LEVEL_LABELS",
    "SupportLevel",
    "TechniqueDescriptor",
    "TechniqueStatus",
    "default_registry",
    "research_registry",
]
