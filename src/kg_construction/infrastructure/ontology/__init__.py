"""Schema, ontology, and validation infrastructure."""

from __future__ import annotations

from kg_construction.infrastructure.ontology.schema_proposal import (
    SchemaProposalResult,
    SchemaProposalWorkflow,
)
from kg_construction.infrastructure.ontology.schema_registry import InMemorySchemaRegistry
from kg_construction.infrastructure.ontology.shacl import ShaclValidationResult, ShaclValidator
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
    "SchemaDriftValidator",
    "SchemaProposalResult",
    "SchemaProposalWorkflow",
    "SchemaRegistryValidator",
    "ShaclValidationResult",
    "ShaclValidator",
    "ValidationFinding",
    "ValidationReport",
]
