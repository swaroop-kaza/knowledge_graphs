"""Compatibility imports for canonical schema-quality evaluation.

New source code should import these contracts from
``kg_construction.evaluation.schema_quality``.
"""

from kg_construction.evaluation.schema_quality import (
    SchemaQualityEvaluator,
    SchemaQualityGate,
    SchemaQualityGateSettings,
    SchemaQualityReport,
)

__all__ = [
    "SchemaQualityEvaluator",
    "SchemaQualityGate",
    "SchemaQualityGateSettings",
    "SchemaQualityReport",
]
