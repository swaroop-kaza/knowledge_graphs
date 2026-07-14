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
        entity_types = {
            entity.candidate_id: entity.entity_type
            for entity in batch.entities
        }
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
            if not self.registry.validate_relationship(
                relation.relation_type,
                source_type=entity_types.get(relation.source),
                target_type=entity_types.get(relation.target),
            ):
                findings.append(
                    ValidationFinding(
                        object_id=relation.candidate_id,
                        object_type="relation",
                        rule="relationship_domain_range",
                        message=(
                            f"Relationship {relation.relation_type} does not match "
                            "the generated source/target type constraints."
                        ),
                        policy=self.policy,
                    )
                )
        return ValidationReport(
            findings=findings,
            checked_count=len(batch.entities) + len(batch.relations),
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
    "SchemaRegistryValidator",
    "ValidationFinding",
    "ValidationPolicy",
    "ValidationReport",
]
