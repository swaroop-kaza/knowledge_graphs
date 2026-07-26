"""Compatibility exports for the canonical Neo4j evaluation backend.

New code should import these names from
``kg_construction.evaluation.neo4j_workload_backend``.
"""

from __future__ import annotations

from kg_construction.evaluation.neo4j_workload_backend import (
    Neo4jGraphWorkloadBackend,
    Neo4jGraphWorkloadConfig,
)

__all__ = [
    "Neo4jGraphWorkloadBackend",
    "Neo4jGraphWorkloadConfig",
]
