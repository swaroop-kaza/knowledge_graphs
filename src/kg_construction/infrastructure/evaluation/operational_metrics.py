"""Compatibility imports for canonical operational evaluation metrics.

New source code should import these contracts from
``kg_construction.evaluation.operational_metrics``.
"""

from kg_construction.evaluation.operational_metrics import (
    OperationalMetrics,
    OperationalMetricsRecorder,
    RetryObservation,
    RuntimeReport,
)

__all__ = [
    "OperationalMetrics",
    "OperationalMetricsRecorder",
    "RetryObservation",
    "RuntimeReport",
]
