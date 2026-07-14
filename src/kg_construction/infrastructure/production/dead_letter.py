"""Dead-letter queue and replay tooling for failed ingestion batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph
from kg_construction.domain.models import stable_id, utc_now

JsonDict = dict[str, Any]
DeadLetterStatus = Literal["pending", "replayed", "failed", "skipped"]
GraphModel = Literal["property_graph", "rdf_graph", "unknown"]


class DeadLetterWriter(Protocol):
    """Minimal writer protocol used by the replay tool."""

    name: str

    def write_property_graph(self, graph: PropertyGraph) -> object:
        """Write a property graph."""
        ...

    def write_rdf_graph(self, graph: RDFGraph) -> object:
        """Write an RDF graph."""
        ...


class DeadLetterRecord(BaseModel):
    """Durable record for a write batch that failed after retry."""

    model_config = ConfigDict(extra="forbid")

    dead_letter_id: str
    stage: str
    store_name: str
    operation_name: str
    graph_model: GraphModel = "unknown"
    status: DeadLetterStatus = "pending"
    attempt_count: int = 1
    error_type: str
    message: str
    payload: JsonDict = Field(default_factory=dict)
    metadata: JsonDict = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())
    updated_at: str | None = None
    replay_error: str | None = None


class DeadLetterReplayReport(BaseModel):
    """Summary emitted by the replay tool."""

    model_config = ConfigDict(extra="forbid")

    attempted: int = 0
    replayed: int = 0
    failed: int = 0
    skipped: int = 0
    dry_run: bool = False
    replayed_ids: list[str] = Field(default_factory=list)
    failed_ids: list[str] = Field(default_factory=list)
    skipped_ids: list[str] = Field(default_factory=list)


class DeadLetterQueue:
    """JSONL-backed dead-letter queue with status updates for replay."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record_failure(
        self,
        *,
        stage: str,
        store_name: str,
        operation_name: str,
        graph_model: GraphModel,
        error: BaseException,
        attempt_count: int,
        payload: JsonDict,
        metadata: JsonDict | None = None,
    ) -> DeadLetterRecord:
        record = DeadLetterRecord(
            dead_letter_id=stable_id(
                "deadletter",
                stage,
                store_name,
                operation_name,
                graph_model,
                type(error).__name__,
                str(error),
                json.dumps(payload, sort_keys=True, default=str),
            ),
            stage=stage,
            store_name=store_name,
            operation_name=operation_name,
            graph_model=graph_model,
            attempt_count=attempt_count,
            error_type=type(error).__name__,
            message=str(error),
            payload=payload,
            metadata=metadata or {},
        )
        self.append(record)
        return record

    def append(self, record: DeadLetterRecord) -> None:
        records = self.load_records()
        existing_index = next(
            (index for index, item in enumerate(records) if item.dead_letter_id == record.dead_letter_id),
            None,
        )
        if existing_index is None:
            records.append(record)
        else:
            records[existing_index] = record
        self._write_records(records)

    def load_records(
        self,
        *,
        status: DeadLetterStatus | None = None,
        store_name: str | None = None,
        limit: int | None = None,
    ) -> list[DeadLetterRecord]:
        if not self.path.exists():
            return []
        records = [
            DeadLetterRecord.model_validate(json.loads(line))
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if status is not None:
            records = [record for record in records if record.status == status]
        if store_name is not None:
            records = [record for record in records if record.store_name == store_name]
        if limit is not None:
            records = records[:limit]
        return records

    def update_status(
        self,
        dead_letter_id: str,
        *,
        status: DeadLetterStatus,
        replay_error: str | None = None,
    ) -> DeadLetterRecord:
        records = self.load_records()
        for index, record in enumerate(records):
            if record.dead_letter_id != dead_letter_id:
                continue
            updated = record.model_copy(
                update={
                    "status": status,
                    "updated_at": utc_now().isoformat(),
                    "replay_error": replay_error,
                }
            )
            records[index] = updated
            self._write_records(records)
            return updated
        raise KeyError(dead_letter_id)

    def _write_records(self, records: list[DeadLetterRecord]) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")


class DeadLetterReplayTool:
    """Replay pending dead-letter graph writes through configured store writers."""

    def __init__(self, queue: DeadLetterQueue) -> None:
        self.queue = queue

    def replay(
        self,
        writers: dict[str, DeadLetterWriter],
        *,
        store_name: str | None = None,
        limit: int | None = None,
        dry_run: bool = False,
    ) -> DeadLetterReplayReport:
        report = DeadLetterReplayReport(dry_run=dry_run)
        records = self.queue.load_records(status="pending", store_name=store_name, limit=limit)
        for record in records:
            report.attempted += 1
            if dry_run:
                report.skipped += 1
                report.skipped_ids.append(record.dead_letter_id)
                continue
            writer = writers.get(record.store_name)
            if writer is None:
                report.skipped += 1
                report.skipped_ids.append(record.dead_letter_id)
                self.queue.update_status(
                    record.dead_letter_id,
                    status="skipped",
                    replay_error=f"No writer configured for store '{record.store_name}'.",
                )
                continue
            try:
                self._replay_record(record, writer)
                self.queue.update_status(record.dead_letter_id, status="replayed")
                report.replayed += 1
                report.replayed_ids.append(record.dead_letter_id)
            except Exception as exc:
                self.queue.update_status(
                    record.dead_letter_id,
                    status="failed",
                    replay_error=f"{type(exc).__name__}: {exc}",
                )
                report.failed += 1
                report.failed_ids.append(record.dead_letter_id)
        return report

    @staticmethod
    def _replay_record(record: DeadLetterRecord, writer: DeadLetterWriter) -> object:
        graph_payload = record.payload.get("graph", record.payload)
        if record.graph_model == "property_graph":
            return writer.write_property_graph(PropertyGraph.model_validate(graph_payload))
        if record.graph_model == "rdf_graph":
            return writer.write_rdf_graph(RDFGraph.model_validate(graph_payload))
        raise ValueError(f"Cannot replay unknown graph model: {record.graph_model}")


__all__ = [
    "DeadLetterQueue",
    "DeadLetterRecord",
    "DeadLetterReplayReport",
    "DeadLetterReplayTool",
    "DeadLetterStatus",
]
