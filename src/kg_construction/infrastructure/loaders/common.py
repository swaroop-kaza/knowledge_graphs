"""Shared helpers for source loaders."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from kg_construction.domain.models import EvidenceRef, SourceRecord

JsonDict = dict[str, Any]


class AdapterNotConfigured(RuntimeError):
    """Raised when a configured adapter is missing required runtime resources."""


def attach_source_evidence(
    source: SourceRecord,
    *,
    location: JsonDict,
    text: str | None = None,
    metadata: JsonDict | None = None,
) -> SourceRecord:
    """Attach source-level evidence metadata without changing the stable source ID."""

    evidence = EvidenceRef.build(source, location=location, text=text, metadata=metadata or {})
    source.metadata.setdefault("source_evidence", []).append(evidence.model_dump(mode="json"))
    return source


def file_location(path: str | Path, *, kind: str, **extra: Any) -> JsonDict:
    """Build a stable source evidence location for file-backed loaders."""

    payload: JsonDict = {"kind": kind, "path": str(Path(path))}
    payload.update(extra)
    return payload


def jsonable_value(value: Any) -> Any:
    """Normalize loader values into JSON-friendly scalars."""

    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, dict):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [jsonable_value(item) for item in value]
    return value


def normalize_rows(rows: list[JsonDict]) -> list[JsonDict]:
    """Return rows with JSON-friendly values and string keys."""

    return [
        {str(column): jsonable_value(value) for column, value in row.items()}
        for row in rows
    ]


__all__ = [
    "AdapterNotConfigured",
    "attach_source_evidence",
    "file_location",
    "jsonable_value",
    "normalize_rows",
]
