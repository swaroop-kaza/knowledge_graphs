"""Schema, ontology, and validation infrastructure."""

from __future__ import annotations

from kg_construction.infrastructure.ontology.schema_proposal import (
    SchemaProposalResult,
    SchemaProposalWorkflow,
)
from kg_construction.infrastructure.ontology.schema_registry import InMemorySchemaRegistry
from kg_construction.infrastructure.ontology.shacl import ShaclValidationResult, ShaclValidator
from kg_construction.infrastructure.ontology.structured_ontology import (
    ShaclReleaseReport,
    StructuredOntologyProposal,
    StructuredReleaseGate,
    generate_shacl_shapes,
    materialize_structured_rdf,
    propose_structured_ontology,
)
from kg_construction.infrastructure.ontology.stubs import (
    JSONSchemaValidator,
    OwlOntologyLoader,
    PropertyGraphValidator,
    SchemaDriftValidator,
)
from kg_construction.infrastructure.ontology.validation import (
    CompositeValidator,
    ConfidenceThresholdValidator,
    EvidenceRequiredValidator,
    ReferentialIntegrityValidator,
    SchemaRegistryValidator,
    ValidationFinding,
    ValidationReport,
)
from kg_construction.infrastructure.production.review import HumanReviewQueue

__all__ = [
    "CompositeValidator",
    "ConfidenceThresholdValidator",
    "EvidenceRequiredValidator",
    "HumanReviewQueue",
    "InMemorySchemaRegistry",
    "JSONSchemaValidator",
    "OwlOntologyLoader",
    "PropertyGraphValidator",
    "ReferentialIntegrityValidator",
    "SchemaDriftValidator",
    "SchemaProposalResult",
    "SchemaProposalWorkflow",
    "SchemaRegistryValidator",
    "ShaclReleaseReport",
    "ShaclValidationResult",
    "ShaclValidator",
    "StructuredOntologyProposal",
    "StructuredReleaseGate",
    "ValidationFinding",
    "ValidationReport",
    "generate_shacl_shapes",
    "materialize_structured_rdf",
    "propose_structured_ontology",
]
