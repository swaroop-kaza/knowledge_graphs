"""Validation contracts remain domain-owned and infrastructure-compatible."""

from __future__ import annotations

from kg_construction.domain.models import ConstructionBatch
from kg_construction.domain.validation import (
    ValidationFinding,
    ValidationReport,
)
from kg_construction.infrastructure.ontology.validation import (
    ValidationFinding as InfrastructureValidationFinding,
)
from kg_construction.infrastructure.ontology.validation import (
    ValidationReport as InfrastructureValidationReport,
)
from kg_construction.ports.validators import GraphValidator


class _DomainContractValidator:
    name = "domain_contract_validator"

    def validate(self, batch: ConstructionBatch) -> ValidationReport:
        return ValidationReport(checked_count=len(batch.entities))


def test_infrastructure_validation_exports_preserve_type_identity() -> None:
    assert InfrastructureValidationFinding is ValidationFinding
    assert InfrastructureValidationReport is ValidationReport


def test_graph_validator_protocol_accepts_domain_contract() -> None:
    validator = _DomainContractValidator()

    assert isinstance(validator, GraphValidator)
    assert validator.validate(ConstructionBatch()).checked_count == 0


def test_validation_report_merge_preserves_findings_and_coverage() -> None:
    warning = ValidationFinding(
        object_id="entity-1",
        object_type="entity",
        rule="review",
        message="Needs review.",
        severity="warning",
        policy="send_to_human_review",
    )
    error = ValidationFinding(
        object_id="entity-2",
        object_type="entity",
        rule="invalid",
        message="Invalid candidate.",
    )

    merged = ValidationReport(
        findings=[warning],
        checked_count=1,
    ).merge(
        ValidationReport(findings=[error], checked_count=2)
    )

    assert merged.checked_count == 3
    assert [finding.object_id for finding in merged.findings] == [
        "entity-1",
        "entity-2",
    ]
    assert merged.invalid_ids == {"entity-2"}
    assert not merged.valid
