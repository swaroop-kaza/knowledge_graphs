"""Evidence helpers for rows, cells, spans, and source fragments."""

from __future__ import annotations

from typing import Any

from kg_construction.domain.models import EvidenceRef, SourceRecord


def row_evidence(source: SourceRecord, row_index: int, row: dict[str, Any]) -> EvidenceRef:
    """Create evidence for a structured row."""

    return EvidenceRef.build(
        source,
        location={"kind": "row", "row_index": row_index},
        text=repr(row),
        metadata={"columns": sorted(row)},
    )


def cell_evidence(
    source: SourceRecord,
    *,
    row_index: int,
    column: str,
    value: object,
) -> EvidenceRef:
    """Create evidence for a single table cell."""

    return EvidenceRef.build(
        source,
        location={"kind": "cell", "row_index": row_index, "column": column},
        text=str(value),
    )


def span_evidence(source: SourceRecord, *, start: int, end: int, text: str) -> EvidenceRef:
    """Create evidence for a text span."""

    return EvidenceRef.build(
        source,
        location={"kind": "span", "start": start, "end": end},
        text=text,
    )


__all__ = ["cell_evidence", "row_evidence", "span_evidence"]
