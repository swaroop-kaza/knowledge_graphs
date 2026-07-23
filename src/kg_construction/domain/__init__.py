"""Domain models for Data-to-Knowledge Graph construction."""

from __future__ import annotations

from kg_construction.domain.graph_objects import (
    PropertyEdge,
    PropertyGraph,
    PropertyNode,
    RDFGraph,
    RDFQuad,
    RDFTriple,
)
from kg_construction.domain.modeling import (
    Cardinality,
    ColumnRole,
    ControlledVocabulary,
    OntologyClass,
    OntologyPredicate,
    OntologySchema,
    ReferenceNode,
)
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    ExtractorRun,
    GraphFact,
    RelationCandidate,
    SourceRecord,
    canonical_json,
    content_hash,
    normalize_text,
    stable_id,
)
from kg_construction.domain.schema import (
    GraphSchema,
    NodeTypeDefinition,
    PropertyDefinition,
    RelationshipTypeDefinition,
    SchemaDiffReport,
    SchemaLifecycleStatus,
    SchemaRegistry,
    diff_graph_schemas,
)

__all__ = [
    "Cardinality",
    "ColumnRole",
    "ConstructionBatch",
    "ControlledVocabulary",
    "EntityCandidate",
    "EvidenceRef",
    "ExtractorRun",
    "GraphFact",
    "GraphSchema",
    "NodeTypeDefinition",
    "OntologyClass",
    "OntologyPredicate",
    "OntologySchema",
    "PropertyDefinition",
    "PropertyEdge",
    "PropertyGraph",
    "PropertyNode",
    "RDFGraph",
    "RDFQuad",
    "RDFTriple",
    "ReferenceNode",
    "RelationCandidate",
    "RelationshipTypeDefinition",
    "SchemaDiffReport",
    "SchemaLifecycleStatus",
    "SchemaRegistry",
    "SourceRecord",
    "canonical_json",
    "content_hash",
    "diff_graph_schemas",
    "normalize_text",
    "stable_id",
]
