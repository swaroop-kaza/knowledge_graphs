"""Immutable, evidence-first contracts for unstructured KG construction.

The contracts deliberately keep four identities apart:

``DocumentRef``
    The immutable source snapshot.
``Evidence``
    An exact quote in that snapshot.
``Mention``
    A typed occurrence of text. Coreference clustering is not entity linking.
``CanonicalEntity``
    An identity created only by an explicit resolver.

Claims reference both their surface arguments and, when available, mention and
canonical-entity identifiers. This makes it possible to retain what a document
actually said while projecting a separately curated canonical graph.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_construction.domain.models import (
    SourceRecord,
    canonical_json,
    content_hash,
    stable_id,
)


class TextSpan(BaseModel):
    """An end-exclusive Unicode-code-point span and its exact quote."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def _valid_span(self) -> TextSpan:
        if self.end <= self.start:
            raise ValueError(f"span end ({self.end}) must be after start ({self.start})")
        if self.end - self.start != len(self.text):
            raise ValueError(
                "span width must equal the number of Unicode code points in its text"
            )
        return self

    def is_exact_in(self, document_text: str) -> bool:
        """Return whether the offsets quote exactly ``text`` in ``document_text``."""

        return self.end <= len(document_text) and document_text[self.start : self.end] == self.text


class DocumentRef(BaseModel):
    """Identity and integrity metadata for one immutable document snapshot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    content_digest: str = Field(min_length=64, max_length=64)
    text_length: int = Field(ge=0)

    @classmethod
    def from_source(cls, source: SourceRecord) -> DocumentRef:
        text = source.content or ""
        return cls(
            document_id=stable_id("document", source.source_type, source.source_name),
            version=source.source_id,
            source_name=source.source_name,
            source_type=source.source_type,
            content_digest=content_hash(text, length=64, normalize=False),
            text_length=len(text),
        )

    def matches_text(self, document_text: str) -> bool:
        """Return whether text is the exact snapshot represented by this reference."""

        return (
            len(document_text) == self.text_length
            and content_hash(document_text, length=64, normalize=False) == self.content_digest
        )


class EvidenceKind(StrEnum):
    SENTENCE = "sentence"
    PASSAGE = "passage"
    MENTION = "mention"


class Evidence(BaseModel):
    """A separately identified exact quote in a document version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    document_version: str = Field(min_length=1)
    kind: EvidenceKind
    span: TextSpan
    prefix: str = ""
    suffix: str = ""

    @classmethod
    def build(
        cls,
        document: DocumentRef,
        *,
        span: TextSpan,
        kind: EvidenceKind,
        prefix: str = "",
        suffix: str = "",
    ) -> Evidence:
        return cls(
            evidence_id=stable_id(
                "evidence",
                document.version,
                kind,
                span.start,
                span.end,
                span.text,
            ),
            document_id=document.document_id,
            document_version=document.version,
            kind=kind,
            span=span,
            prefix=prefix,
            suffix=suffix,
        )

    def is_exact_for(self, document: DocumentRef, document_text: str) -> bool:
        """Return whether identity, snapshot digest, offsets, and quote all agree."""

        return (
            self.document_id == document.document_id
            and self.document_version == document.version
            and document.matches_text(document_text)
            and self.span.is_exact_in(document_text)
        )


class Provenance(BaseModel):
    """Deterministic metadata for the provider that produced an item."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    extractor: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @property
    def complete(self) -> bool:
        return all(
            value.strip()
            for value in (
                self.extractor,
                self.provider,
                self.model_version,
                self.configuration_id,
            )
        )


class Modality(StrEnum):
    """How strongly the source commits to a claim."""

    ASSERTED = "asserted"
    PROBABLE = "probable"
    POSSIBLE = "possible"
    HYPOTHETICAL = "hypothetical"


class ClaimKind(StrEnum):
    RELATION = "relation"
    EVENT = "event"
    FACT = "fact"
    INTENT = "intent"


class ReviewState(StrEnum):
    EXTRACTED = "extracted"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


class ResolutionStatus(StrEnum):
    UNRESOLVED = "unresolved"
    COREFERENCE_ONLY = "coreference_only"
    LINKED = "linked"
    REVIEW = "review"


class Mention(BaseModel):
    """One immutable typed occurrence; it is never itself a canonical entity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_version: str = Field(min_length=1)
    evidence: Evidence
    surface: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    provenance: Provenance
    coreference_cluster_id: str | None = None
    coreference_anchor: str | None = None
    canonical_entity_id: str | None = None
    resolution_status: ResolutionStatus = ResolutionStatus.UNRESOLVED
    resolution_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent_identity(self) -> Mention:
        if self.document_version != self.evidence.document_version:
            raise ValueError("mention and evidence must reference the same document version")
        if self.surface != self.evidence.span.text:
            raise ValueError("mention surface must exactly equal its evidence quote")
        if self.resolution_status == ResolutionStatus.LINKED and not self.canonical_entity_id:
            raise ValueError("a linked mention requires a canonical_entity_id")
        if self.canonical_entity_id and self.resolution_status != ResolutionStatus.LINKED:
            raise ValueError("canonical_entity_id is only valid for linked mentions")
        return self

    @property
    def mention_id(self) -> str:
        return stable_id(
            "mention",
            self.document_version,
            self.evidence.span.start,
            self.evidence.span.end,
            self.entity_type,
            self.surface,
            self.provenance.extractor,
        )

    @property
    def unresolved(self) -> bool:
        return self.resolution_status != ResolutionStatus.LINKED

    # Compatibility conveniences for callers that previously consumed the
    # preliminary contract. They intentionally do not conflate cluster and entity.
    @property
    def span(self) -> TextSpan:
        return self.evidence.span

    @property
    def text(self) -> str:
        return self.surface

    @property
    def type(self) -> str:
        return self.entity_type

    @property
    def entity_id(self) -> str | None:
        return self.canonical_entity_id


class CanonicalEntity(BaseModel):
    """A resolver-created identity backed by one or more mentions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    mention_ids: tuple[str, ...] = Field(min_length=1)
    aliases: tuple[str, ...] = ()
    external_system: str | None = None
    external_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _external_identity_pair(self) -> CanonicalEntity:
        if bool(self.external_system) != bool(self.external_id):
            raise ValueError("external_system and external_id must be provided together")
        if len(set(self.mention_ids)) != len(self.mention_ids):
            raise ValueError("canonical entity mention_ids must be unique")
        return self


class EventRole(BaseModel):
    """One typed participant in an event claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1)
    value: str = Field(min_length=1)
    mention_id: str | None = None
    canonical_entity_id: str | None = None


class Claim(BaseModel):
    """An immutable relation, event, fact, or intent with exact evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: DocumentRef
    kind: ClaimKind
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str | None = None
    subject_mention_id: str | None = None
    object_mention_id: str | None = None
    subject_entity_id: str | None = None
    object_entity_id: str | None = None
    event_roles: tuple[EventRole, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    negated: bool = False
    modality: Modality = Modality.ASSERTED
    attribution: str | None = None
    temporal: str | None = None
    quantity: str | None = None
    provenance: Provenance
    review_state: ReviewState = ReviewState.EXTRACTED

    @model_validator(mode="after")
    def _well_formed_claim(self) -> Claim:
        if self.kind in {ClaimKind.RELATION, ClaimKind.FACT} and not self.object:
            raise ValueError(f"{self.kind.value} claims require an object")
        if self.kind == ClaimKind.EVENT and not self.event_roles:
            raise ValueError("event claims require at least one event role")
        for item in self.evidence:
            if (
                item.document_id != self.document.document_id
                or item.document_version != self.document.version
            ):
                raise ValueError("claim evidence must belong to the claim's document version")
        return self

    @property
    def claim_id(self) -> str:
        roles = tuple(
            (role.role, role.value, role.mention_id, role.canonical_entity_id)
            for role in self.event_roles
        )
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        return stable_id(
            "claim",
            self.document.version,
            self.kind,
            self.subject,
            self.predicate,
            self.object or "",
            canonical_json(roles),
            canonical_json(evidence_ids),
            self.negated,
            self.modality,
            self.attribution or "",
            self.temporal or "",
            self.quantity or "",
            self.provenance.extractor,
        )

    def has_exact_evidence(self, document_text: str) -> bool:
        """Return whether all non-empty evidence exactly quotes this snapshot."""

        return bool(self.evidence) and all(
            item.is_exact_for(self.document, document_text) for item in self.evidence
        )

    def has_complete_provenance(self, document_text: str) -> bool:
        """Return whether evidence and provider metadata are release-grade."""

        return (
            self.has_exact_evidence(document_text)
            and self.document.matches_text(document_text)
            and self.provenance.complete
        )

    def promoted(self) -> Claim:
        return self.model_copy(update={"review_state": ReviewState.PROMOTED})

    def quarantined(self) -> Claim:
        return self.model_copy(update={"review_state": ReviewState.QUARANTINED})


__all__ = [
    "CanonicalEntity",
    "Claim",
    "ClaimKind",
    "DocumentRef",
    "EventRole",
    "Evidence",
    "EvidenceKind",
    "Mention",
    "Modality",
    "Provenance",
    "ResolutionStatus",
    "ReviewState",
    "TextSpan",
]
