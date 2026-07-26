"""Compatibility imports for the canonical locked unstructured runtime.

New source code should import these contracts from
``kg_construction.unstructured.runtime``. This module remains importable for
downstream callers that used the former application-layer location.
"""

from kg_construction.unstructured.runtime import (
    HeldOutClaimCase,
    HeldOutClaimCaseReport,
    HeldOutClaimDataset,
    HeldOutClaimEvaluationReport,
    MandatoryUnstructuredRuntime,
)

__all__ = [
    "HeldOutClaimCase",
    "HeldOutClaimCaseReport",
    "HeldOutClaimDataset",
    "HeldOutClaimEvaluationReport",
    "MandatoryUnstructuredRuntime",
]
