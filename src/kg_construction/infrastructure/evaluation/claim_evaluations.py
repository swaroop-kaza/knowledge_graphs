"""Compatibility imports for canonical unstructured-claim evaluation.

New source code should import these contracts from
``kg_construction.evaluation.claim_evaluations``.
"""

from kg_construction.evaluation.claim_evaluations import (
    ClaimEvaluationSuite,
    EvalScore,
    HardCoverageReport,
    HardCoverageThresholds,
)

__all__ = [
    "ClaimEvaluationSuite",
    "EvalScore",
    "HardCoverageReport",
    "HardCoverageThresholds",
]
