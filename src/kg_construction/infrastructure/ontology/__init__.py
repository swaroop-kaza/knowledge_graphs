"""Schema, ontology, and validation infrastructure."""

from __future__ import annotations

from kg_construction.infrastructure.ontology.ontology_alignment import (
    AlignmentReport,
    OntologyAligner,
)
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
from kg_construction.infrastructure.ontology.unstructured_governance import (
    ClaimShaclResult,
    FinalResourceShaclResult,
    GovernedClaimShaclGate,
    GovernedPredicate,
    GovernedResourceShaclGate,
    GovernedUnstructuredSchema,
    final_rdf_graph_sha256,
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
    "ClaimShaclResult",
    "FinalResourceShaclResult",
    "ConfidenceThresholdValidator",
    "EvidenceRequiredValidator",
    "GovernedClaimShaclGate",
    "GovernedPredicate",
    "GovernedResourceShaclGate",
    "GovernedUnstructuredSchema",
    "HumanReviewQueue",
    "InMemorySchemaRegistry",
    "JSONSchemaValidator",
    "OwlOntologyLoader",
    "PropertyGraphValidator",
    "ReferentialIntegrityValidator",
    "SchemaDriftValidator",
    "AlignmentReport",
    "OntologyAligner",
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
    "final_rdf_graph_sha256",
    "materialize_structured_rdf",
    "propose_structured_ontology",
]
