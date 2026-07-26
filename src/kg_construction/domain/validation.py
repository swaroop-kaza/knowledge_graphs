"""Domain contracts for validating candidate knowledge-graph objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ValidationSeverity = Literal["info", "warning", "error"]
ValidationPolicy = Literal["reject", "quarantine", "repair", "send_to_human_review"]


class ValidationFinding(BaseModel):
    """One validation issue associated with a candidate graph object."""

    model_config = ConfigDict(extra="forbid")

    object_id: str
    object_type: str
    rule: str
    message: str
    severity: ValidationSeverity = "error"
    policy: ValidationPolicy = "reject"


class ValidationReport(BaseModel):
    """Validation findings and coverage for one construction batch."""

    model_config = ConfigDict(extra="forbid")

    findings: list[ValidationFinding] = Field(default_factory=list)
    checked_count: int = 0

    @property
    def valid(self) -> bool:
        """Return whether the report contains no error-severity findings."""

        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def invalid_ids(self) -> set[str]:
        """Return object identifiers associated with error findings."""

        return {
            finding.object_id
            for finding in self.findings
            if finding.severity == "error"
        }

    def merge(self, other: ValidationReport) -> ValidationReport:
        """Combine findings and checked-object counts without mutating either report."""

        return ValidationReport(
            findings=[*self.findings, *other.findings],
            checked_count=self.checked_count + other.checked_count,
        )


__all__ = [
    "ValidationFinding",
    "ValidationPolicy",
    "ValidationReport",
    "ValidationSeverity",
]
