"""Pydantic domain models with deterministic IDs and evidence-first provenance."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any, Literal

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
        source_id = stable_id("source", source_type, source_name, content_hash(text))
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
        row_hash = content_hash(repr(sorted(rows, key=repr)), normalize=False)
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
        content = None if isinstance(payload, list) else json.dumps(payload, sort_keys=True)
        source_metadata = {**(metadata or {}), "json_payload": payload}
        source_id = stable_id("source", "json", source_name, content_hash(repr(payload), normalize=False))
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
            sorted(evidence_location.items()),
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
        candidate_id = stable_id("entity", entity_type, normalize_text(name), evidence_ids, extractor_name)
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
        candidate_id = stable_id("relation", source, relation_type, target, evidence_ids, extractor_name)
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
        fact_id = stable_id("fact", subject, predicate, object, evidence_ids, extractor_name)
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

        return ConstructionBatch(
            source_records=[*self.source_records, *other.source_records],
            entities=[*self.entities, *other.entities],
            relations=[*self.relations, *other.relations],
            facts=[*self.facts, *other.facts],
            metadata={**self.metadata, **other.metadata},
        )
