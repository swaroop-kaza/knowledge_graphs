"""Compatibility imports for the canonical :mod:`kg_construction.construction` API.

New code should import from ``kg_construction`` or
``kg_construction.construction``. This module remains for one compatibility
cycle because older integrations imported the application-layer path directly.
"""

from kg_construction.construction.api import (
    construct_kg,
    convert,
    default_workflow_for_source_type,
    load_sources,
    profile_dataset,
)
from kg_construction.research.api import evaluate_techniques

__all__ = [
    "construct_kg",
    "convert",
    "default_workflow_for_source_type",
    "evaluate_techniques",
    "load_sources",
    "profile_dataset",
]
