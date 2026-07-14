"""Extensible ontology, SHACL, schema proposal, and review adapters."""

from __future__ import annotations

from kg_construction.domain.models import ConstructionBatch
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured
from kg_construction.infrastructure.ontology.validation import ValidationReport
from kg_construction.infrastructure.production.review import HumanReviewQueue


class ExtensibleValidatorStub:
    """Base class for validators that need optional engines or human workflows."""

    name = "validator_stub"
    dependency_hint = "Install and configure the validator-specific dependency."

    def validate(self, batch: ConstructionBatch) -> ValidationReport:
        raise AdapterNotConfigured(
            f"{self.name} is represented as an extensible validator. {self.dependency_hint} "
            f"Batch objects: {len(batch.entities) + len(batch.relations) + len(batch.facts)}"
        )


class JSONSchemaValidator(ExtensibleValidatorStub):
    name = "json_schema_validator"
    dependency_hint = "Provide JSON Schema for LLM graph candidate payloads."


class PropertyGraphValidator(ExtensibleValidatorStub):
    name = "property_graph_validator"
    dependency_hint = "Configure property graph constraints, orphan checks, and duplicate edge checks."


class SchemaDriftValidator(ExtensibleValidatorStub):
    name = "schema_drift_validator"
    dependency_hint = "Provide prior profiles and acceptable drift policy."


class OwlOntologyLoader:
    """Stub for OWL/RDF ontology loading and alignment."""

    name = "owl_ontology_loader"

    def load(self, path: str) -> object:
        raise AdapterNotConfigured(
            "OWL ontology loading requires rdflib/owlready2 configuration. "
            f"Ontology path: {path}"
        )


__all__ = [
    "ExtensibleValidatorStub",
    "HumanReviewQueue",
    "JSONSchemaValidator",
    "OwlOntologyLoader",
    "PropertyGraphValidator",
    "SchemaDriftValidator",
]
