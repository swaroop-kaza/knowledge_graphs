"""Error and invalid-candidate quarantine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch, SourceRecord, stable_id, utc_now
from kg_construction.domain.validation import ValidationFinding, ValidationReport

JsonDict = dict[str, Any]
QuarantineReason = Literal["exception", "validation_finding"]


class QuarantineRecord(BaseModel):
    """Durable quarantine record for a failed source or invalid candidate."""

    model_config = ConfigDict(extra="forbid")

    quarantine_id: str
    reason: QuarantineReason
    stage: str
    object_id: str | None = None
    object_type: str | None = None
    source_id: str | None = None
    source_name: str | None = None
    rule: str | None = None
    message: str
    severity: str = "error"
    payload: JsonDict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())


class ErrorQuarantine:
    """Append-only JSONL quarantine store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def quarantine_exception(
        self,
        *,
        stage: str,
        error: BaseException,
        source: SourceRecord | None = None,
        payload: JsonDict | None = None,
    ) -> QuarantineRecord:
        record = QuarantineRecord(
            quarantine_id=stable_id(
                "quarantine",
                "exception",
                stage,
                type(error).__name__,
                str(error),
                source.source_id if source else "",
            ),
            reason="exception",
            stage=stage,
            source_id=source.source_id if source else None,
            source_name=source.source_name if source else None,
            message=str(error),
            payload={"error_type": type(error).__name__, **(payload or {})},
        )
        self.append(record)
        return record

    def quarantine_findings(
        self,
        *,
        batch: ConstructionBatch,
        validation_report: ValidationReport,
    ) -> list[QuarantineRecord]:
        source_lookup = self._candidate_source_lookup(batch)
        records: list[QuarantineRecord] = []
        for finding in validation_report.findings:
            if finding.policy != "quarantine" and finding.severity != "error":
                continue
            source = source_lookup.get(finding.object_id)
            record = self.from_finding(finding, source=source)
            self.append(record)
            records.append(record)
        return records

    def from_finding(
        self,
        finding: ValidationFinding,
        *,
        source: SourceRecord | None = None,
    ) -> QuarantineRecord:
        return QuarantineRecord(
            quarantine_id=stable_id(
                "quarantine",
                "validation",
                finding.object_id,
                finding.rule,
                finding.message,
            ),
            reason="validation_finding",
            stage="validation",
            object_id=finding.object_id,
            object_type=finding.object_type,
            source_id=source.source_id if source else None,
            source_name=source.source_name if source else None,
            rule=finding.rule,
            message=finding.message,
            severity=finding.severity,
            payload={"finding": finding.model_dump(mode="json")},
        )

    def append(self, record: QuarantineRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
            handle.write("\n")

    def load_records(self) -> list[QuarantineRecord]:
        if not self.path.exists():
            return []
        return [
            QuarantineRecord.model_validate(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    @staticmethod
    def _candidate_source_lookup(batch: ConstructionBatch) -> dict[str, SourceRecord]:
        lookup: dict[str, SourceRecord] = {}
        for candidate in [*batch.entities, *batch.relations, *batch.facts]:
            object_id = getattr(candidate, "candidate_id", getattr(candidate, "fact_id", "unknown"))
            for evidence in getattr(candidate, "evidence", []):
                for source in batch.source_records:
                    if evidence.source_id == source.source_id:
                        lookup[object_id] = source
                        break
                if object_id in lookup:
                    break
        return lookup


__all__ = ["ErrorQuarantine", "QuarantineReason", "QuarantineRecord"]
