"""Validation layer for candidate graph objects."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch
from kg_construction.domain.schema import SchemaRegistry

ValidationSeverity = Literal["info", "warning", "error"]
ValidationPolicy = Literal["reject", "quarantine", "repair", "send_to_human_review"]


class ValidationFinding(BaseModel):
    """One validation issue."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    object_type: str
    rule: str
    message: str
    severity: ValidationSeverity = "error"
    policy: ValidationPolicy = "reject"


class ValidationReport(BaseModel):
    """Validation result for a construction batch."""

    model_config = ConfigDict(extra="forbid")

    findings: list[ValidationFinding] = Field(default_factory=list)
    checked_count: int = 0

    @property
    def valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def invalid_ids(self) -> set[str]:
        return {finding.object_id for finding in self.findings if finding.severity == "error"}

    def merge(self, other: ValidationReport) -> ValidationReport:
        return ValidationReport(
            findings=[*self.findings, *other.findings],
            checked_count=self.checked_count + other.checked_count,
        )


class EvidenceRequiredValidator:
    """Require every entity, relation, and fact to have evidence."""

    name = "evidence_required_validator"

    def __init__(self, policy: ValidationPolicy = "reject") -> None:
        self.policy = policy

    def validate(self, batch: ConstructionBatch) -> ValidationReport:
        findings: list[ValidationFinding] = []
        for entity in batch.entities:
            if not entity.evidence:
                findings.append(
                    ValidationFinding(
                        object_id=entity.candidate_id,
                        object_type="entity",
                        rule="evidence_required",
                        message="Entity candidate has no evidence.",
                        policy=self.policy,
                    )
                )
        for relation in batch.relations:
            if not relation.evidence:
                findings.append(
                    ValidationFinding(
                        object_id=relation.candidate_id,
                        object_type="relation",
                        rule="evidence_required",
                        message="Relation candidate has no evidence.",
                        policy=self.policy,
                    )
                )
        for fact in batch.facts:
            if not fact.evidence:
                findings.append(
                    ValidationFinding(
                        object_id=fact.fact_id,
                        object_type="fact",
                        rule="evidence_required",
                        message="Graph fact has no evidence.",
                        policy=self.policy,
                    )
                )
        return ValidationReport(
            findings=findings,
            checked_count=len(batch.entities) + len(batch.relations) + len(batch.facts),
        )


class ReferentialIntegrityValidator:
    """Require materialized relationship endpoints to reference current entity IDs."""

    name = "referential_integrity_validator"

    def __init__(self, policy: ValidationPolicy = "reject") -> None:
        self.policy = policy

    def validate(self, batch: ConstructionBatch) -> ValidationReport:
        current_entity_ids = {
            str(entity.properties.get("canonical_entity_id", entity.candidate_id))
            for entity in batch.entities
        }
        known_entity_references = {
            entity.candidate_id
            for entity in batch.entities
        } | current_entity_ids
        aliases = batch.metadata.get("entity_resolution_aliases", {})
        if isinstance(aliases, dict):
            known_entity_references.update(str(alias) for alias in aliases)
            known_entity_references.update(str(target) for target in aliases.values())

        findings: list[ValidationFinding] = []
        checked_count = 0
        for relation in batch.relations:
            for role, endpoint in (("source", relation.source), ("target", relation.target)):
                checked_count += 1
                if endpoint not in current_entity_ids:
                    findings.append(
                        ValidationFinding(
                            object_id=relation.candidate_id,
                            object_type="relation",
                            rule="referential_integrity",
                            message=(
                                f"Relationship {role} endpoint does not reference a current "
                                f"entity node: {endpoint}"
                            ),
                            policy=self.policy,
                        )
                    )

        for fact in batch.facts:
            fact_endpoints = [("subject", fact.subject)]
            if str(fact.properties.get("rdf_object_type", "iri")) != "literal":
                fact_endpoints.append(("object", fact.object))
            for role, endpoint in fact_endpoints:
                checked_count += 1
                if _is_internal_entity_reference(endpoint, known_entity_references) and (
                    endpoint not in current_entity_ids
                ):
                    findings.append(
                        ValidationFinding(
                            object_id=fact.fact_id,
                            object_type="fact",
                            rule="referential_integrity",
                            message=(
                                f"Fact {role} endpoint refers to a non-current entity ID: "
                                f"{endpoint}"
                            ),
                            policy=self.policy,
                        )
                    )

        return ValidationReport(findings=findings, checked_count=checked_count)


def _is_internal_entity_reference(value: str, known_references: set[str]) -> bool:
    return (
        value in known_references
        or value.startswith("canonical_entity_")
        or value.startswith("entity_")
    )


class SchemaRegistryValidator:
    """Validate candidate labels against the configured schema registry."""

    name = "schema_registry_validator"

    def __init__(
        self,
        registry: SchemaRegistry | None = None,
        *,
        policy: ValidationPolicy = "quarantine",
    ) -> None:
        self.registry = registry or SchemaRegistry.permissive()
        self.policy = policy

    def validate(self, batch: ConstructionBatch) -> ValidationReport:
        findings: list[ValidationFinding] = []
        entity_types = _entity_types_by_reference(batch)
        for entity in batch.entities:
            if not self.registry.validate_entity_type(entity.entity_type):
                findings.append(
                    ValidationFinding(
                        object_id=entity.candidate_id,
                        object_type="entity",
                        rule="allowed_label",
                        message=f"Entity type is not allowed: {entity.entity_type}",
                        policy=self.policy,
                    )
                )
            missing = self.registry.required_properties_for(entity.entity_type) - set(entity.properties)
            if missing:
                findings.append(
                    ValidationFinding(
                        object_id=entity.candidate_id,
                        object_type="entity",
                        rule="required_properties",
                        message=f"Missing required properties: {sorted(missing)}",
                        policy=self.policy,
                    )
                )
        for relation in batch.relations:
            if not self.registry.validate_relationship_type(relation.relation_type):
                findings.append(
                    ValidationFinding(
                        object_id=relation.candidate_id,
                        object_type="relation",
                        rule="allowed_relationship",
                        message=f"Relationship type is not allowed: {relation.relation_type}",
                        policy=self.policy,
                    )
                )
                continue
            source_types = entity_types.get(relation.source, set())
            target_types = entity_types.get(relation.target, set())
            if not _validate_relationship_for_type_sets(
                self.registry,
                relation.relation_type,
                source_types=source_types,
                target_types=target_types,
            ):
                findings.append(
                    ValidationFinding(
                        object_id=relation.candidate_id,
                        object_type="relation",
                        rule="relationship_domain_range",
                        message=(
                            f"Relationship {relation.relation_type} does not match "
                            "the generated source/target type constraints: "
                            f"source_types={sorted(source_types)}, "
                            f"target_types={sorted(target_types)}."
                        ),
                        policy=self.policy,
                    )
                )
        return ValidationReport(
            findings=findings,
            checked_count=len(batch.entities) + len(batch.relations),
        )


def _entity_types_by_reference(batch: ConstructionBatch) -> dict[str, set[str]]:
    """Index entity types by candidate and materialized canonical references."""

    entity_types: dict[str, set[str]] = {}
    for entity in batch.entities:
        references = {entity.candidate_id}
        for property_name in ("canonical_entity_id", "canonical_id"):
            canonical_id = entity.properties.get(property_name)
            if canonical_id:
                references.add(str(canonical_id))
        for reference in references:
            entity_types.setdefault(reference, set()).add(entity.entity_type)
    return entity_types


def _validate_relationship_for_type_sets(
    registry: SchemaRegistry,
    relationship_type: str,
    *,
    source_types: set[str],
    target_types: set[str],
) -> bool:
    """Require every possible canonical-cluster type pairing to satisfy the schema."""

    source_options: list[str | None] = list(sorted(source_types))
    target_options: list[str | None] = list(sorted(target_types))
    if not source_options:
        source_options.append(None)
    if not target_options:
        target_options.append(None)
    return all(
        registry.validate_relationship(
            relationship_type,
            source_type=source_type,
            target_type=target_type,
        )
        for source_type in source_options
        for target_type in target_options
    )


class ConfidenceThresholdValidator:
    """Reject or quarantine candidates below a configured confidence threshold."""

    name = "confidence_threshold_validator"

    def __init__(self, threshold: float = 0.5, *, policy: ValidationPolicy = "quarantine") -> None:
        self.threshold = threshold
        self.policy = policy

    def validate(self, batch: ConstructionBatch) -> ValidationReport:
        findings: list[ValidationFinding] = []
        for object_type, candidates in (
            ("entity", batch.entities),
            ("relation", batch.relations),
            ("fact", batch.facts),
        ):
            for candidate in candidates:
                object_id = getattr(candidate, "candidate_id", getattr(candidate, "fact_id", "unknown"))
                if candidate.confidence < self.threshold:
                    findings.append(
                        ValidationFinding(
                            object_id=object_id,
                            object_type=object_type,
                            rule="confidence_threshold",
                            message=f"Confidence {candidate.confidence:.3f} is below {self.threshold:.3f}.",
                            policy=self.policy,
                        )
                    )
        return ValidationReport(
            findings=findings,
            checked_count=len(batch.entities) + len(batch.relations) + len(batch.facts),
        )


class CompositeValidator:
    """Run multiple validators and merge reports."""

    name = "composite_validator"

    def __init__(self, validators: list[Any]) -> None:
        self.validators = validators

    def validate(self, batch: ConstructionBatch) -> ValidationReport:
        report = ValidationReport()
        for validator in self.validators:
            validate = validator.validate
            report = report.merge(validate(batch))
        return report


__all__ = [
    "CompositeValidator",
    "ConfidenceThresholdValidator",
    "EvidenceRequiredValidator",
    "ReferentialIntegrityValidator",
    "SchemaRegistryValidator",
    "ValidationFinding",
    "ValidationPolicy",
    "ValidationReport",
]
