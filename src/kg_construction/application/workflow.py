"""Compatibility imports for :mod:`kg_construction.construction.workflow`.

The executable workflow is construction-owned. New source code should import
from ``kg_construction.construction``; this module preserves former application
imports for one migration cycle.
"""

from kg_construction.construction.workflow import (
    ConstructionResult,
    ConstructionWorkflow,
    DraftConstructionWorkflow,
    PublicationGateError,
    PublicationGateReport,
)

__all__ = [
    "ConstructionResult",
    "ConstructionWorkflow",
    "DraftConstructionWorkflow",
    "PublicationGateError",
    "PublicationGateReport",
]
