"""Pydantic domain models with deterministic IDs and evidence-first provenance."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

JsonDict = dict[str, Any]
ValidationStatus = Literal["candidate", "valid", "invalid", "quarantined", "human_review"]

_WHITESPACE_RE = re.compile(r"\s+")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(UTC)


def normalize_text(value: str) -> str:
    """Return a stable text representation for matching and hashing."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.casefold()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def content_hash(value: str, *, length: int = 32, normalize: bool = True) -> str:
    """Hash text content with optional normalization."""

    payload = normalize_text(value) if normalize else value
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if length <= 0 or length > len(digest):
        return digest
    return digest[:length]


def _canonical_json_default(value: object) -> object:
    """Return deterministic JSON-compatible values for common source types."""

    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (Path, UUID)):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=canonical_json)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return str(value)


def canonical_json(value: object) -> str:
    """Serialize JSON-like data canonically for hashing and comparisons.

    Object keys are sorted recursively and insignificant whitespace is removed.
    This deliberately preserves array order; callers that model an unordered row
    collection must sort the individually serialized rows before hashing.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_canonical_json_default,
    )


def stable_id(prefix: str, *parts: object, length: int = 20) -> str:
    """Build a deterministic ID from a prefix and stable parts."""

    payload = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{content_hash(payload, length=length, normalize=False)}"


class StrictModel(BaseModel):
    """Base model with strict extra-field handling."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class SourceRecord(StrictModel):
    """Normalized source unit loaded from files, APIs, tables, or documents."""

    source_id: str
    source_type: str
    source_name: str
    content: str | None = None
    records: list[JsonDict] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source_type: str = "text",
        source_name: str = "memory://text",
        metadata: JsonDict | None = None,
    ) -> SourceRecord:
        # Source snapshots are byte/text-content identities, not matching keys.
        # Case and whitespace changes can alter offsets and meaning, so they
        # must produce a new snapshot even though normalized matching may later
        # treat some extracted values as equivalent.
        source_id = stable_id(
            "source",
            source_type,
            source_name,
            content_hash(text, normalize=False),
        )
        return cls(
            source_id=source_id,
            source_type=source_type,
            source_name=source_name,
            content=text,
            metadata=metadata or {},
        )

    @classmethod
    def from_rows(
        cls,
        rows: list[JsonDict],
        *,
        source_type: str,
        source_name: str,
        metadata: JsonDict | None = None,
    ) -> SourceRecord:
        # A table snapshot is treated as a collection of rows.  Canonicalize
        # every row before sorting so dictionary insertion order cannot alter
        # the content-addressed source ID.
        canonical_rows = sorted(canonical_json(row) for row in rows)
        row_hash = content_hash(canonical_json(canonical_rows), normalize=False)
        source_id = stable_id("source", source_type, source_name, row_hash)
        return cls(
            source_id=source_id,
            source_type=source_type,
            source_name=source_name,
            records=rows,
            metadata=metadata or {},
        )

    @classmethod
    def from_json(
        cls,
        payload: Any,
        *,
        source_name: str,
        metadata: JsonDict | None = None,
    ) -> SourceRecord:
        records = payload if isinstance(payload, list) else []
        canonical_payload = canonical_json(payload)
        content = None if isinstance(payload, list) else canonical_payload
        source_metadata = {**(metadata or {}), "json_payload": payload}
        source_id = stable_id(
            "source",
            "json",
            source_name,
            content_hash(canonical_payload, normalize=False),
        )
        return cls(
            source_id=source_id,
            source_type="json",
            source_name=source_name,
            content=content,
            records=records,
            metadata=source_metadata,
        )


class EvidenceRef(StrictModel):
    """Traceable evidence for a generated candidate or graph fact."""

    evidence_id: str
    source_id: str
    source_type: str
    source_name: str
    location: JsonDict = Field(default_factory=dict)
    text: str | None = None
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def build(
        cls,
        source: SourceRecord,
        *,
        location: JsonDict | None = None,
        text: str | None = None,
        metadata: JsonDict | None = None,
    ) -> EvidenceRef:
        evidence_location = location or {}
        evidence_id = stable_id(
            "evidence",
            source.source_id,
            source.source_type,
            source.source_name,
            canonical_json(evidence_location),
            text or "",
        )
        return cls(
            evidence_id=evidence_id,
            source_id=source.source_id,
            source_type=source.source_type,
            source_name=source.source_name,
            location=evidence_location,
            text=text,
            metadata=metadata or {},
        )


class ExtractorRun(StrictModel):
    """Metadata describing an extraction or mapping execution."""

    run_id: str
    extractor_name: str
    technique: str
    schema_version: str = "v1"
    metadata: JsonDict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @classmethod
    def build(
        cls,
        *,
        extractor_name: str,
        technique: str,
        schema_version: str = "v1",
        metadata: JsonDict | None = None,
    ) -> ExtractorRun:
        run_id = stable_id("run", extractor_name, technique, schema_version, utc_now().isoformat())
        return cls(
            run_id=run_id,
            extractor_name=extractor_name,
            technique=technique,
            schema_version=schema_version,
            metadata=metadata or {},
        )


class EntityCandidate(StrictModel):
    """Candidate entity emitted by a loader, mapper, extractor, or LLM."""

    candidate_id: str
    entity_type: str
    name: str
    properties: JsonDict = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extractor_name: str
    technique: str
    schema_version: str = "v1"
    created_at: datetime = Field(default_factory=utc_now)
    provenance_metadata: JsonDict = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        entity_type: str,
        name: str,
        evidence: list[EvidenceRef],
        extractor_name: str,
        technique: str,
        properties: JsonDict | None = None,
        confidence: float = 1.0,
        schema_version: str = "v1",
        provenance_metadata: JsonDict | None = None,
    ) -> EntityCandidate:
        evidence_ids = tuple(sorted(item.evidence_id for item in evidence))
        candidate_id = stable_id(
            "entity",
            entity_type,
            normalize_text(name),
            evidence_ids,
            extractor_name,
            canonical_json(properties or {}),
            canonical_json(provenance_metadata or {}),
        )
        return cls(
            candidate_id=candidate_id,
            entity_type=entity_type,
            name=name,
            properties=properties or {},
            evidence=evidence,
            confidence=confidence,
            extractor_name=extractor_name,
            technique=technique,
            schema_version=schema_version,
            provenance_metadata=provenance_metadata or {},
        )


class RelationCandidate(StrictModel):
    """Candidate relationship emitted by a mapper, extractor, or LLM."""

    candidate_id: str
    source: str
    relation_type: str
    target: str
    properties: JsonDict = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extractor_name: str
    technique: str
    schema_version: str = "v1"
    created_at: datetime = Field(default_factory=utc_now)
    provenance_metadata: JsonDict = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        source: str,
        relation_type: str,
        target: str,
        evidence: list[EvidenceRef],
        extractor_name: str,
        technique: str,
        properties: JsonDict | None = None,
        confidence: float = 1.0,
        schema_version: str = "v1",
        provenance_metadata: JsonDict | None = None,
    ) -> RelationCandidate:
        evidence_ids = tuple(sorted(item.evidence_id for item in evidence))
        candidate_id = stable_id(
            "relation",
            source,
            relation_type,
            target,
            evidence_ids,
            extractor_name,
            canonical_json(properties or {}),
            canonical_json(provenance_metadata or {}),
        )
        return cls(
            candidate_id=candidate_id,
            source=source,
            relation_type=relation_type,
            target=target,
            properties=properties or {},
            evidence=evidence,
            confidence=confidence,
            extractor_name=extractor_name,
            technique=technique,
            schema_version=schema_version,
            provenance_metadata=provenance_metadata or {},
        )


class GraphFact(StrictModel):
    """Validated assertion ready for graph materialization."""

    fact_id: str
    subject: str
    predicate: str
    object: str
    properties: JsonDict = Field(default_factory=dict)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    extractor_name: str
    technique: str
    schema_version: str = "v1"
    validation_status: ValidationStatus = "candidate"
    created_at: datetime = Field(default_factory=utc_now)
    provenance_metadata: JsonDict = Field(default_factory=dict)

    @classmethod
    def build(
        cls,
        *,
        subject: str,
        predicate: str,
        object: str,
        evidence: list[EvidenceRef],
        extractor_name: str,
        technique: str,
        properties: JsonDict | None = None,
        confidence: float = 1.0,
        schema_version: str = "v1",
        validation_status: ValidationStatus = "candidate",
        provenance_metadata: JsonDict | None = None,
    ) -> GraphFact:
        evidence_ids = tuple(sorted(item.evidence_id for item in evidence))
        fact_id = stable_id(
            "fact",
            subject,
            predicate,
            object,
            evidence_ids,
            extractor_name,
            canonical_json(properties or {}),
            canonical_json(provenance_metadata or {}),
        )
        return cls(
            fact_id=fact_id,
            subject=subject,
            predicate=predicate,
            object=object,
            properties=properties or {},
            evidence=evidence,
            confidence=confidence,
            extractor_name=extractor_name,
            technique=technique,
            schema_version=schema_version,
            validation_status=validation_status,
            provenance_metadata=provenance_metadata or {},
        )


class ConstructionBatch(StrictModel):
    """Intermediate graph construction payload passed between pipeline stages."""

    source_records: list[SourceRecord] = Field(default_factory=list)
    entities: list[EntityCandidate] = Field(default_factory=list)
    relations: list[RelationCandidate] = Field(default_factory=list)
    facts: list[GraphFact] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)

    def merged(self, other: ConstructionBatch) -> ConstructionBatch:
        """Return a new batch containing objects from both batches."""

        # Source IDs are content-addressed snapshot identities.  Pipeline
        # stages commonly return the source they consumed, so concatenating
        # batches must be idempotent for source records even when the same
        # snapshot appears in both operands.
        source_records_by_id: dict[str, SourceRecord] = {}
        for source_record in [*self.source_records, *other.source_records]:
            source_records_by_id.setdefault(source_record.source_id, source_record)

        return ConstructionBatch(
            source_records=list(source_records_by_id.values()),
            entities=[*self.entities, *other.entities],
            relations=[*self.relations, *other.relations],
            facts=[*self.facts, *other.facts],
            metadata={**self.metadata, **other.metadata},
        )
