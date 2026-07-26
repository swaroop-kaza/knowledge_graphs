"""Evidence-first construction for unstructured documents.

The pipeline keeps source observations immutable and only promotes claims after
their exact offsets, document version, provider provenance, confidence, and
ontology predicate have passed. Model-backed components are injected through
small provider protocols; deterministic providers make the same contracts
testable offline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.claim import (
    CanonicalEntity,
    Claim,
    ClaimKind,
    DocumentRef,
    EventRole,
    Evidence,
    EvidenceKind,
    Mention,
    Modality,
    Provenance,
    ResolutionStatus,
    TextSpan,
)
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    GraphFact,
    RelationCandidate,
    SourceRecord,
    normalize_text,
    stable_id,
)
from kg_construction.infrastructure.ontology.unstructured_governance import (
    ClaimShaclResult,
    GovernedClaimShaclGate,
    GovernedUnstructuredSchema,
)

_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
_NEGATION_RE = re.compile(r"\b(?:no|not|never|without|failed\s+to|did\s+not)\b|n['’]t\b", re.I)
_POSSIBLE_RE = re.compile(r"\b(?:may|might|could|possibly|perhaps|allegedly|reportedly)\b", re.I)
_PROBABLE_RE = re.compile(r"\b(?:likely|probably|expected\s+to)\b", re.I)
_HYPOTHETICAL_RE = re.compile(r"\b(?:if|assuming|suppose|hypothetically)\b", re.I)
_ATTRIBUTION_RE = re.compile(
    r"(?i:according\s+to|reported\s+by|stated\s+by)\s+"
    r"(?P<source>[A-Z][\w.@+-]*(?:\s+[A-Z][\w.@+-]*)*)"
)
_TEMPORAL_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}|"
    r"(?:in|during|before|after|since|until)\s+(?:the\s+)?(?:[A-Za-z-]+\s+){0,2}\d{4}|"
    r"(?:in|during|before|after|since|until)\s+(?:the\s+)?[A-Za-z-]+|"
    r"(?<![\w-])\d{4}(?![\w-])|yesterday|today|tomorrow|last\s+[A-Za-z-]+"
    r")\b",
    re.I,
)
_QUANTITY_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|kg|g|mg|km|m|cm|mm|hours?|days?|items?|units?)\b",
    re.I,
)

_SPECIFIC_MENTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Date", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("EmailAddress", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("Version", re.compile(r"\bv\d+(?:\.\d+)+\b")),
    ("Identifier", re.compile(r"\b[A-Z]{2,}-\d+\b")),
)
_PROPER_NAME_RE = re.compile(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*\b")
_NON_ENTITY_CAPITALS = frozenset(
    {
        "according",
        "after",
        "although",
        "before",
        "during",
        "however",
        "if",
        "in",
        "later",
        "meanwhile",
        "on",
        "since",
        "the",
        "then",
        "today",
        "when",
        "while",
        "yesterday",
    }
)
_ORGANIZATION_SUFFIXES = frozenset(
    {"agency", "association", "bank", "company", "corp", "corporation", "institute", "labs", "university"}
)
_PLACE_SUFFIXES = frozenset({"city", "country", "park", "province", "state"})
_ARTIFACT_SUFFIXES = frozenset(
    {
        "api",
        "application",
        "bombe",
        "database",
        "engine",
        "machine",
        "model",
        "platform",
        "product",
        "service",
        "store",
        "system",
    }
)

_RELATION_PATTERNS: tuple[tuple[str, ClaimKind, re.Pattern[str]], ...] = (
    ("affected", ClaimKind.RELATION, re.compile(r"\b(?:affected|affects)\b", re.I)),
    ("calls", ClaimKind.RELATION, re.compile(r"\b(?:called|calls)\b", re.I)),
    ("uses", ClaimKind.RELATION, re.compile(r"\b(?:used|uses|using)\b", re.I)),
    ("depends_on", ClaimKind.RELATION, re.compile(r"\bdepends?\s+on\b", re.I)),
    ("located_in", ClaimKind.RELATION, re.compile(r"\b(?:located|based)\s+in\b", re.I)),
    ("part_of", ClaimKind.RELATION, re.compile(r"\b(?:part|member)\s+of\b", re.I)),
    ("reports_to", ClaimKind.RELATION, re.compile(r"\breports?\s+to\b", re.I)),
    ("contains", ClaimKind.RELATION, re.compile(r"\b(?:contained|contains)\b", re.I)),
    ("produces", ClaimKind.RELATION, re.compile(r"\b(?:produced|produces)\b", re.I)),
    ("works_at", ClaimKind.RELATION, re.compile(r"\b(?:worked|works|working)\s+at\b", re.I)),
    ("works_with", ClaimKind.RELATION, re.compile(r"\b(?:worked|works|working)\s+with\b", re.I)),
    ("designed", ClaimKind.RELATION, re.compile(r"\b(?:designed|designs)\b", re.I)),
    ("created", ClaimKind.RELATION, re.compile(r"\b(?:created|creates)\b", re.I)),
    ("acquired", ClaimKind.RELATION, re.compile(r"\b(?:acquired|acquires)\b", re.I)),
    ("founded", ClaimKind.RELATION, re.compile(r"\b(?:founded|founds)\b", re.I)),
    ("built", ClaimKind.RELATION, re.compile(r"\b(?:built|builds)\b", re.I)),
    ("joined", ClaimKind.RELATION, re.compile(r"\b(?:joined|joins)\b", re.I)),
    ("owns", ClaimKind.RELATION, re.compile(r"\b(?:owned|owns)\b", re.I)),
    ("manages", ClaimKind.RELATION, re.compile(r"\b(?:managed|manages)\b", re.I)),
    ("reported", ClaimKind.RELATION, re.compile(r"\b(?:reported|reports)\b", re.I)),
    ("is_a", ClaimKind.FACT, re.compile(r"\b(?:is|was|are|were)\b", re.I)),
)
_EVENT_PREDICATES = frozenset({"acquired", "built", "created", "founded", "joined"})
_INTENT_RE = re.compile(
    r"\b(?:intends?|intended|plans?|planned|aims?|aimed|wants?|wanted)\s+to\s+"
    r"(?P<verb>[a-z][a-z_-]*)\b",
    re.I,
)


class RuntimeProfile(StrEnum):
    DETERMINISTIC = "deterministic"
    MODEL_BACKED = "model_backed"


class ProviderContractError(ValueError):
    """Raised when a provider emits offsets or identities outside the contract."""


class UnstructuredPipelineConfig(BaseModel):
    """Promotion and execution policy for the pipeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile: RuntimeProfile = RuntimeProfile.DETERMINISTIC
    minimum_claim_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_context_chars: int = Field(default=24, ge=0, le=512)
    require_governed_schema: bool = True
    require_pre_promotion_shacl: bool = True
    require_canonical_identity: bool = True


@runtime_checkable
class Segmenter(Protocol):
    def segment(self, document: DocumentRef, text: str) -> tuple[Evidence, ...]: ...


@runtime_checkable
class MentionDetector(Protocol):
    def detect(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
    ) -> tuple[Mention, ...]: ...


@runtime_checkable
class CoreferenceResolver(Protocol):
    def cluster(self, mentions: tuple[Mention, ...]) -> tuple[Mention, ...]: ...


@runtime_checkable
class ClaimExtractor(Protocol):
    def extract(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
        mentions: tuple[Mention, ...],
    ) -> tuple[Claim, ...]: ...


@runtime_checkable
class AttributeExtractor(Protocol):
    def annotate(self, text: str, claims: tuple[Claim, ...]) -> tuple[Claim, ...]: ...


class IdentityResolution(BaseModel):
    """Immutable output from an identity provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mentions: tuple[Mention, ...]
    canonical_entities: tuple[CanonicalEntity, ...] = ()
    provider_participation_ids: tuple[str, ...] = ()
    nil_policy: str = "retain_unresolved"
    nil_cluster_ids: tuple[str, ...] = ()
    decisions: tuple[dict[str, object], ...] = ()


@runtime_checkable
class IdentityResolver(Protocol):
    def resolve(self, mentions: tuple[Mention, ...]) -> IdentityResolution: ...


class EntityLink(BaseModel):
    """A real provider's proposed external identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    external_system: str = Field(min_length=1)
    external_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    entity_type: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


@runtime_checkable
class EntityLinkProvider(Protocol):
    """Boundary for Wikidata, DBpedia, cross-encoder, or project linkers."""

    def link(
        self,
        canonical_name: str,
        entity_type: str,
        mentions: tuple[Mention, ...],
    ) -> EntityLink | None: ...


class PipelineResult(BaseModel):
    """Immutable evidence, observations, claims, and curated output for one run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document: DocumentRef
    document_text: str = Field(repr=False)
    segments: tuple[Evidence, ...]
    mentions: tuple[Mention, ...]
    quarantined_mentions: tuple[Mention, ...] = ()
    entity_type_alignment_results: tuple[dict[str, object], ...] = ()
    off_ontology_entity_types: tuple[str, ...] = ()
    canonical_entities: tuple[CanonicalEntity, ...]
    claims: tuple[Claim, ...]
    promoted: tuple[Claim, ...]
    quarantined: tuple[Claim, ...]
    claim_ontology_alignment_results: tuple[dict[str, object], ...] = ()
    off_ontology_predicates: tuple[str, ...] = ()
    governed_schema_id: str | None = None
    governed_schema_version: str | None = None
    governed_schema_sha256: str | None = None
    shacl_executed: bool = False
    shacl_conforms: bool = False
    claim_shacl_results: tuple[ClaimShaclResult, ...] = ()
    provider_participation_ids: tuple[str, ...] = ()
    identity_nil_policy: str = "retain_unresolved"
    identity_nil_cluster_ids: tuple[str, ...] = ()
    identity_decisions: tuple[dict[str, object], ...] = ()
    claim_argument_resolution_results: tuple[dict[str, object], ...] = ()
    semantic_claim_merge_results: tuple[dict[str, object], ...] = ()

    @property
    def unresolved_mentions(self) -> tuple[Mention, ...]:
        return tuple(mention for mention in self.mentions if mention.unresolved)

    @property
    def exact_evidence_coverage(self) -> float:
        if not self.promoted:
            return 1.0
        exact = sum(
            claim.has_exact_evidence(self.document_text) for claim in self.promoted
        )
        return exact / len(self.promoted)

    @property
    def provenance_coverage(self) -> float:
        if not self.promoted:
            return 1.0
        complete = sum(
            claim.has_complete_provenance(self.document_text) for claim in self.promoted
        )
        return complete / len(self.promoted)


class UnstructuredPipeline:
    """Run evidence-first extraction with injected model or deterministic providers."""

    name = "evidence_first_unstructured_pipeline"
    technique = "evidence_first_semantic_extraction"
    schema_version = "v2"

    def __init__(
        self,
        *,
        config: UnstructuredPipelineConfig | None = None,
        segmenter: Segmenter | None = None,
        mention_detector: MentionDetector | None = None,
        coreference: CoreferenceResolver | None = None,
        claim_extractor: ClaimExtractor | None = None,
        attribute_extractor: AttributeExtractor | None = None,
        identity_resolver: IdentityResolver | None = None,
        ontology_predicates: set[str] | None = None,
        governed_schema: GovernedUnstructuredSchema | None = None,
        claim_shacl_gate: GovernedClaimShaclGate | None = None,
    ) -> None:
        self.config = config or UnstructuredPipelineConfig()
        if self.config.profile == RuntimeProfile.MODEL_BACKED and (
            mention_detector is None or claim_extractor is None
        ):
            raise ValueError(
                "model_backed profile requires explicit mention_detector and claim_extractor providers"
            )
        self.segmenter = segmenter or RuleSegmenter(
            context_chars=self.config.evidence_context_chars
        )
        self.mention_detector = mention_detector or RuleMentionDetector()
        self.coreference = coreference or ConservativeCoreferenceResolver()
        self.claim_extractor = claim_extractor or PatternClaimExtractor()
        self.attribute_extractor = attribute_extractor or RuleAttributeExtractor()
        if (
            governed_schema is not None
            and ontology_predicates is not None
            and governed_schema.predicate_names != frozenset(ontology_predicates)
        ):
            raise ValueError(
                "governed_schema and ontology_predicates declare different publication terms"
            )
        if governed_schema is None and ontology_predicates is not None:
            governed_schema = GovernedUnstructuredSchema.from_predicates(
                ontology_predicates
            )
        if (
            governed_schema is None
            and self.config.profile == RuntimeProfile.MODEL_BACKED
            and self.config.require_governed_schema
        ):
            governed_schema = GovernedUnstructuredSchema.from_package_resource()
        self.governed_schema = governed_schema
        self.claim_shacl_gate = claim_shacl_gate or (
            GovernedClaimShaclGate(governed_schema)
            if governed_schema is not None
            else None
        )
        if (
            self.config.profile == RuntimeProfile.MODEL_BACKED
            and self.config.require_pre_promotion_shacl
            and self.claim_shacl_gate is None
        ):
            raise ValueError(
                "model_backed profile requires the governed pre-promotion SHACL gate"
            )
        self.identity_resolver = identity_resolver or (
            ExplicitNilObservedResolver()
            if (
                self.config.profile == RuntimeProfile.MODEL_BACKED
                and self.config.require_canonical_identity
            )
            else KeepUnresolvedResolver()
        )
        self.ontology_predicates = (
            self.governed_schema.predicate_names
            if self.governed_schema is not None
            else None
        )
        self._pipeline_results: list[PipelineResult] = []

    def run(self, source: SourceRecord) -> PipelineResult:
        document = DocumentRef.from_source(source)
        text = source.content or ""
        segments = tuple(self.segmenter.segment(document, text))
        self._validate_evidence(segments, document, text, "segmenter")

        mentions = tuple(self.mention_detector.detect(document, text, segments))
        self._validate_mentions(mentions, document, text)
        (
            mentions,
            quarantined_mentions,
            type_alignment_results,
            off_ontology_entity_types,
        ) = self._align_mention_types(mentions)
        mentions = tuple(self.coreference.cluster(mentions))
        self._validate_mentions(mentions, document, text)

        claims = tuple(self.claim_extractor.extract(document, text, segments, mentions))
        claims = tuple(self.attribute_extractor.annotate(text, claims))

        resolution = self.identity_resolver.resolve(mentions)
        self._validate_mentions(resolution.mentions, document, text)
        (
            claims,
            argument_resolution_validity,
            argument_resolution_results,
        ) = self._resolve_claim_arguments(claims, resolution.mentions)
        claims = self._bind_canonical_entities(claims, resolution.mentions)

        (
            claims,
            ontology_alignment_validity,
            claim_ontology_alignment_results,
            off_ontology,
        ) = self._align(claims)
        (
            claims,
            ontology_alignment_validity,
            argument_resolution_validity,
            claim_ontology_alignment_results,
            semantic_claim_merge_results,
        ) = self._consolidate_semantic_claims(
            claims,
            ontology_alignment_validity,
            argument_resolution_validity,
            claim_ontology_alignment_results,
            text,
            self.config.minimum_claim_confidence,
        )
        promoted, quarantined, shacl_results = self._promote_or_quarantine(
            claims,
            text,
            ontology_alignment_validity,
            argument_resolution_validity,
        )
        promoted_claim_ids = {claim.claim_id for claim in promoted}
        promoted_shacl_results = tuple(
            item
            for item in shacl_results
            if item.claim_id in promoted_claim_ids
        )
        participation = self._provider_participation_ids(resolution)
        schema = self.governed_schema
        return PipelineResult(
            document=document,
            document_text=text,
            segments=segments,
            mentions=resolution.mentions,
            quarantined_mentions=quarantined_mentions,
            entity_type_alignment_results=type_alignment_results,
            off_ontology_entity_types=tuple(sorted(off_ontology_entity_types)),
            canonical_entities=resolution.canonical_entities,
            claims=claims,
            promoted=promoted,
            quarantined=quarantined,
            claim_ontology_alignment_results=claim_ontology_alignment_results,
            off_ontology_predicates=tuple(sorted(off_ontology)),
            governed_schema_id=schema.schema_id if schema is not None else None,
            governed_schema_version=schema.version if schema is not None else None,
            governed_schema_sha256=(
                schema.source_sha256 if schema is not None else None
            ),
            shacl_executed=self.claim_shacl_gate is not None,
            shacl_conforms=(
                self.claim_shacl_gate is not None
                and {item.claim_id for item in promoted_shacl_results}
                == promoted_claim_ids
                and all(
                    item.executed and item.conforms
                    for item in promoted_shacl_results
                )
            ),
            claim_shacl_results=shacl_results,
            provider_participation_ids=participation,
            identity_nil_policy=resolution.nil_policy,
            identity_nil_cluster_ids=resolution.nil_cluster_ids,
            identity_decisions=resolution.decisions,
            claim_argument_resolution_results=argument_resolution_results,
            semantic_claim_merge_results=semantic_claim_merge_results,
        )

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        """Execute the pipeline and materialize its layers for ``ConstructionWorkflow``.

        The batch keeps document, mention, observed/coreference identity,
        canonical identity, claim, and event resources distinct.  Direct
        semantic relations and facts are emitted in addition to the reified
        claim resources, and every object carries an exact source selector.
        """

        result = self.run(source)
        self._pipeline_results.append(result)
        return self._to_construction_batch(source, result)

    def reset_results(self) -> None:
        """Clear retained results before one workflow invocation."""

        self._pipeline_results.clear()

    @property
    def pipeline_results(self) -> tuple[PipelineResult, ...]:
        """Return immutable domain results retained for publication/audit."""

        return tuple(self._pipeline_results)

    def _to_construction_batch(
        self,
        source: SourceRecord,
        result: PipelineResult,
    ) -> ConstructionBatch:
        evidence_cache: dict[str, EvidenceRef] = {}

        def evidence_ref(item: Evidence) -> EvidenceRef:
            cached = evidence_cache.get(item.evidence_id)
            if cached is not None:
                return cached
            converted = EvidenceRef.build(
                source,
                location={
                    "kind": "span",
                    "start": item.span.start,
                    "end": item.span.end,
                    "evidence_kind": item.kind.value,
                },
                text=item.span.text,
                metadata={
                    "domain_evidence_id": item.evidence_id,
                    "document_id": item.document_id,
                    "document_version": item.document_version,
                    "prefix": item.prefix,
                    "suffix": item.suffix,
                },
            )
            evidence_cache[item.evidence_id] = converted
            return converted

        document_evidence = EvidenceRef.build(
            source,
            location={
                "kind": "document",
                "start": 0,
                "end": len(result.document_text),
            },
            text=result.document_text,
            metadata={
                "document_id": result.document.document_id,
                "document_version": result.document.version,
                "content_digest": result.document.content_digest,
            },
        )
        pipeline_provenance = {
            "provider": "unstructured_pipeline",
            "model_version": self.schema_version,
            "configuration_id": self.config.profile.value,
        }

        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        facts: list[GraphFact] = []

        document_node = EntityCandidate.build(
            entity_type="Document",
            name=result.document.source_name,
            properties={
                "domain_document_id": result.document.document_id,
                "document_version": result.document.version,
                "source_type": result.document.source_type,
                "content_digest": result.document.content_digest,
                "text_length": result.document.text_length,
            },
            evidence=[document_evidence],
            confidence=1.0,
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.schema_version,
            provenance_metadata=pipeline_provenance,
        )
        entities.append(document_node)

        mention_nodes: dict[str, EntityCandidate] = {}
        mention_by_id = {mention.mention_id: mention for mention in result.mentions}
        for mention in result.mentions:
            provenance = self._candidate_provenance(mention.provenance)
            node = EntityCandidate.build(
                entity_type="Mention",
                name=mention.surface,
                properties={
                    "domain_mention_id": mention.mention_id,
                    "document_id": result.document.document_id,
                    "document_version": mention.document_version,
                    "surface": mention.surface,
                    "detected_entity_type": mention.entity_type,
                    "start": mention.span.start,
                    "end": mention.span.end,
                    "coreference_cluster_id": mention.coreference_cluster_id,
                    "coreference_anchor": mention.coreference_anchor,
                    "linked_canonical_entity_id": mention.canonical_entity_id,
                    "resolution_status": mention.resolution_status.value,
                    "resolution_confidence": mention.resolution_confidence,
                },
                evidence=[evidence_ref(mention.evidence)],
                confidence=mention.provenance.confidence,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
                provenance_metadata=provenance,
            )
            mention_nodes[mention.mention_id] = node
            entities.append(node)
            relations.append(
                RelationCandidate.build(
                    source=document_node.candidate_id,
                    relation_type="HAS_MENTION",
                    target=node.candidate_id,
                    properties={
                        "document_version": mention.document_version,
                        "domain_mention_id": mention.mention_id,
                    },
                    evidence=[evidence_ref(mention.evidence)],
                    confidence=mention.provenance.confidence,
                    extractor_name=self.name,
                    technique=self.technique,
                    schema_version=self.schema_version,
                    provenance_metadata=provenance,
                )
            )

        # Actual canonical identities only exist when the explicit resolver
        # created them.  Unlinked/coreference-only mentions get observed entity
        # resources below; those must never be mislabeled as canonical.
        identity_reference_by_mention: dict[str, str] = {}
        canonical_nodes_by_domain_id: dict[str, EntityCandidate] = {}
        for canonical in result.canonical_entities:
            support = [
                evidence_ref(mention_by_id[mention_id].evidence)
                for mention_id in canonical.mention_ids
                if mention_id in mention_by_id
            ]
            node = EntityCandidate.build(
                entity_type="CanonicalEntity",
                name=canonical.canonical_name,
                properties={
                    # This reserved property makes the workflow materialize the
                    # resolver's stable identity as the graph node identifier.
                    "canonical_entity_id": canonical.entity_id,
                    "domain_canonical_entity_id": canonical.entity_id,
                    "semantic_entity_type": canonical.entity_type,
                    "mention_ids": list(canonical.mention_ids),
                    "aliases": list(canonical.aliases),
                    "external_system": canonical.external_system,
                    "external_id": canonical.external_id,
                    "identity_layer": "canonical",
                },
                evidence=support,
                confidence=canonical.confidence,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
                provenance_metadata={
                    **pipeline_provenance,
                    "resolution_provider": canonical.external_system,
                },
            )
            canonical_nodes_by_domain_id[canonical.entity_id] = node
            entities.append(node)
            for mention_id in canonical.mention_ids:
                identity_reference_by_mention[mention_id] = canonical.entity_id

        observed_groups: dict[str, list[Mention]] = {}
        for mention in result.mentions:
            if mention.mention_id in identity_reference_by_mention:
                continue
            group_key = mention.coreference_cluster_id or mention.mention_id
            observed_groups.setdefault(group_key, []).append(mention)

        for group_key, members in observed_groups.items():
            anchor = members[0].coreference_anchor or members[0].surface
            entity_type = next(
                (
                    member.entity_type
                    for member in members
                    if member.entity_type != "Entity"
                ),
                members[0].entity_type,
            )
            support = [evidence_ref(member.evidence) for member in members]
            node = EntityCandidate.build(
                entity_type=entity_type,
                name=anchor,
                properties={
                    "identity_layer": "observed",
                    "semantic_entity_type": entity_type,
                    "identity_status": (
                        "coreference_only" if len(members) > 1 else "unresolved"
                    ),
                    "coreference_cluster_id": group_key,
                    "coreference_anchor": anchor,
                    "mention_ids": [member.mention_id for member in members],
                    "aliases": sorted({member.surface for member in members} - {anchor}),
                },
                evidence=support,
                confidence=min(member.provenance.confidence for member in members),
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
                provenance_metadata={
                    **pipeline_provenance,
                    "provider": "conservative_coreference",
                    "configuration_id": "whole-token-unambiguous-v1",
                },
            )
            entities.append(node)
            for member in members:
                identity_reference_by_mention[member.mention_id] = node.candidate_id

        for mention in result.mentions:
            target = identity_reference_by_mention[mention.mention_id]
            relation_type = (
                "REFERS_TO_CANONICAL_ENTITY"
                if mention.canonical_entity_id
                else "DENOTES_OBSERVED_ENTITY"
            )
            relations.append(
                RelationCandidate.build(
                    source=mention_nodes[mention.mention_id].candidate_id,
                    relation_type=relation_type,
                    target=target,
                    properties={
                        "resolution_status": mention.resolution_status.value,
                        "coreference_cluster_id": mention.coreference_cluster_id,
                    },
                    evidence=[evidence_ref(mention.evidence)],
                    confidence=mention.resolution_confidence
                    or mention.provenance.confidence,
                    extractor_name=self.name,
                    technique=self.technique,
                    schema_version=self.schema_version,
                    provenance_metadata=self._candidate_provenance(
                        mention.provenance
                    ),
                )
            )

        for claim in result.promoted:
            subject_ref = self._claim_argument_reference(
                claim.subject_mention_id,
                identity_reference_by_mention,
                argument_name="subject",
                claim_id=claim.claim_id,
            )
            object_ref = (
                self._claim_argument_reference(
                    claim.object_mention_id,
                    identity_reference_by_mention,
                    argument_name="object",
                    claim_id=claim.claim_id,
                )
                if claim.object is not None
                else None
            )
            role_references = tuple(
                (
                    role,
                    self._claim_argument_reference(
                        role.mention_id,
                        identity_reference_by_mention,
                        argument_name=f"event_role:{role.role}",
                        claim_id=claim.claim_id,
                    ),
                )
                for role in claim.event_roles
            )
            claim_evidence = [evidence_ref(item) for item in claim.evidence]
            provenance = self._candidate_provenance(claim.provenance)
            properties = {
                **self._claim_properties(claim),
                "promotion_status": "promoted",
            }
            claim_node = EntityCandidate.build(
                entity_type="Event" if claim.kind == ClaimKind.EVENT else "Claim",
                name=claim.claim_id,
                properties=properties,
                evidence=claim_evidence,
                confidence=claim.provenance.confidence,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
                provenance_metadata=provenance,
            )
            entities.append(claim_node)
            relations.append(
                RelationCandidate.build(
                    source=document_node.candidate_id,
                    relation_type="ASSERTS",
                    target=claim_node.candidate_id,
                    properties={
                        "claim_kind": claim.kind.value,
                        "domain_claim_id": claim.claim_id,
                    },
                    evidence=claim_evidence,
                    confidence=claim.provenance.confidence,
                    extractor_name=self.name,
                    technique=self.technique,
                    schema_version=self.schema_version,
                    provenance_metadata=provenance,
                )
            )

            if claim.kind == ClaimKind.EVENT:
                for role, role_ref in role_references:
                    relations.append(
                        RelationCandidate.build(
                            source=claim_node.candidate_id,
                            relation_type=f"HAS_{self._relationship_type(role.role)}",
                            target=role_ref,
                            properties={
                                "domain_claim_id": claim.claim_id,
                                "event_predicate": claim.predicate,
                                "role": role.role,
                            },
                            evidence=claim_evidence,
                            confidence=claim.provenance.confidence,
                            extractor_name=self.name,
                            technique=self.technique,
                            schema_version=self.schema_version,
                            provenance_metadata=provenance,
                        )
                    )
                continue

            relations.append(
                RelationCandidate.build(
                    source=claim_node.candidate_id,
                    relation_type="HAS_SUBJECT",
                    target=subject_ref,
                    properties={"domain_claim_id": claim.claim_id},
                    evidence=claim_evidence,
                    confidence=claim.provenance.confidence,
                    extractor_name=self.name,
                    technique=self.technique,
                    schema_version=self.schema_version,
                    provenance_metadata=provenance,
                )
            )
            if object_ref is not None:
                relations.append(
                    RelationCandidate.build(
                        source=claim_node.candidate_id,
                        relation_type="HAS_OBJECT",
                        target=object_ref,
                        properties={"domain_claim_id": claim.claim_id},
                        evidence=claim_evidence,
                        confidence=claim.provenance.confidence,
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.schema_version,
                        provenance_metadata=provenance,
                    )
                )

            if object_ref is None:
                continue
            if claim.kind in {ClaimKind.RELATION, ClaimKind.INTENT}:
                relations.append(
                    RelationCandidate.build(
                        source=subject_ref,
                        relation_type=self._relationship_type(claim.predicate),
                        target=object_ref,
                        properties=properties,
                        evidence=claim_evidence,
                        confidence=claim.provenance.confidence,
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.schema_version,
                        provenance_metadata=provenance,
                    )
                )
            elif claim.kind == ClaimKind.FACT:
                facts.append(
                    GraphFact.build(
                        subject=subject_ref,
                        predicate=claim.predicate,
                        object=object_ref,
                        properties={**properties, "rdf_object_type": "iri"},
                        evidence=claim_evidence,
                        confidence=claim.provenance.confidence,
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.schema_version,
                        provenance_metadata=provenance,
                    )
                )

        # Low-confidence or off-ontology claims are materialized and flagged
        # rather than dropped, so the returned graph always contains every
        # extracted claim together with its quality/provenance status.
        for claim in result.quarantined:
            claim_evidence = [evidence_ref(item) for item in claim.evidence]
            provenance = self._candidate_provenance(claim.provenance)
            properties = {
                **self._claim_properties(claim),
                "promotion_status": "quarantined",
            }
            claim_node = EntityCandidate.build(
                entity_type="Event" if claim.kind == ClaimKind.EVENT else "Claim",
                name=claim.claim_id,
                properties=properties,
                evidence=claim_evidence,
                confidence=claim.provenance.confidence,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
                provenance_metadata=provenance,
            )
            entities.append(claim_node)
            relations.append(
                RelationCandidate.build(
                    source=document_node.candidate_id,
                    relation_type="ASSERTS",
                    target=claim_node.candidate_id,
                    properties={
                        "claim_kind": claim.kind.value,
                        "domain_claim_id": claim.claim_id,
                        "promotion_status": "quarantined",
                    },
                    evidence=claim_evidence,
                    confidence=claim.provenance.confidence,
                    extractor_name=self.name,
                    technique=self.technique,
                    schema_version=self.schema_version,
                    provenance_metadata=provenance,
                )
            )

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            facts=facts,
            metadata={
                "extractor": self.name,
                "technique": self.technique,
                "schema_version": self.schema_version,
                "document_id": result.document.document_id,
                "document_version": result.document.version,
                "runtime_profile": self.config.profile.value,
                "promoted_claim_count": len(result.promoted),
                "quarantined_claim_count": len(result.quarantined),
                "quarantined_claim_ids": [
                    claim.claim_id for claim in result.quarantined
                ],
                "quarantined_mention_count": len(result.quarantined_mentions),
                "quarantined_mention_ids": [
                    mention.mention_id for mention in result.quarantined_mentions
                ],
                "off_ontology_entity_types": list(
                    result.off_ontology_entity_types
                ),
                "entity_type_alignment_results": list(
                    result.entity_type_alignment_results
                ),
                "off_ontology_predicates": list(result.off_ontology_predicates),
                "claim_ontology_alignment_results": list(
                    result.claim_ontology_alignment_results
                ),
                "exact_evidence_coverage": result.exact_evidence_coverage,
                "provenance_coverage": result.provenance_coverage,
                "mandatory_mention_providers": list(
                    getattr(self.mention_detector, "provider_ids", ())
                ),
                "mandatory_claim_providers": list(
                    getattr(self.claim_extractor, "provider_ids", ())
                ),
                "mention_provider_agreement": [
                    agreement.model_dump(mode="json")
                    for agreement in getattr(
                        self.mention_detector, "last_agreements", ()
                    )
                ],
                "nested_mention_suppressions": [
                    suppression.model_dump(mode="json")
                    for suppression in getattr(
                        self.mention_detector, "last_suppressions", ()
                    )
                ],
                "claim_provider_agreement": [
                    agreement.model_dump(mode="json")
                    for agreement in getattr(
                        self.claim_extractor, "last_agreements", ()
                    )
                ],
                "claim_argument_resolution_results": list(
                    result.claim_argument_resolution_results
                ),
                "semantic_claim_merge_results": list(
                    result.semantic_claim_merge_results
                ),
                "resource_layers": [
                    "Document",
                    "Mention",
                    "ObservedEntity",
                    "CanonicalEntity",
                    "Claim",
                    "Event",
                ],
            },
        )

    @staticmethod
    def _candidate_provenance(provenance: Provenance) -> dict[str, object]:
        return {
            "extractor": provenance.extractor,
            "provider": provenance.provider,
            "model_version": provenance.model_version,
            "configuration_id": provenance.configuration_id,
            "confidence": provenance.confidence,
        }

    @staticmethod
    def _claim_properties(claim: Claim) -> dict[str, object]:
        return {
            "domain_claim_id": claim.claim_id,
            "claim_kind": claim.kind.value,
            "subject_surface": claim.subject,
            "predicate": claim.predicate,
            "object_surface": claim.object,
            "subject_mention_id": claim.subject_mention_id,
            "object_mention_id": claim.object_mention_id,
            "subject_entity_id": claim.subject_entity_id,
            "object_entity_id": claim.object_entity_id,
            "negated": claim.negated,
            "modality": claim.modality.value,
            "attribution": claim.attribution,
            "temporal": claim.temporal,
            "quantity": claim.quantity,
            "event_roles": [
                {
                    "role": role.role,
                    "value": role.value,
                    "mention_id": role.mention_id,
                    "canonical_entity_id": role.canonical_entity_id,
                }
                for role in claim.event_roles
            ],
            "review_state": claim.review_state.value,
            "domain_evidence_ids": [item.evidence_id for item in claim.evidence],
        }

    @staticmethod
    def _claim_argument_reference(
        mention_id: str | None,
        by_mention: dict[str, str],
        *,
        argument_name: str,
        claim_id: str,
    ) -> str:
        if not mention_id:
            raise ProviderContractError(
                f"promoted claim {claim_id} has no {argument_name} mention anchor"
            )
        reference = by_mention.get(mention_id)
        if reference is None:
            raise ProviderContractError(
                f"promoted claim {claim_id} has an unknown {argument_name} "
                f"mention anchor {mention_id}"
            )
        return reference

    @staticmethod
    def _relationship_type(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
        return normalized or "RELATED_TO"

    @staticmethod
    def _validate_evidence(
        evidence: Iterable[Evidence],
        document: DocumentRef,
        text: str,
        provider: str,
    ) -> None:
        for item in evidence:
            if not item.is_exact_for(document, text):
                raise ProviderContractError(
                    f"{provider} emitted inexact evidence {item.evidence_id}"
                )

    @classmethod
    def _validate_mentions(
        cls,
        mentions: tuple[Mention, ...],
        document: DocumentRef,
        text: str,
    ) -> None:
        seen: set[str] = set()
        for mention in mentions:
            if mention.mention_id in seen:
                raise ProviderContractError(
                    f"duplicate mention identity {mention.mention_id}"
                )
            seen.add(mention.mention_id)
            cls._validate_evidence((mention.evidence,), document, text, "mention provider")

    def _align_mention_types(
        self,
        mentions: tuple[Mention, ...],
    ) -> tuple[
        tuple[Mention, ...],
        tuple[Mention, ...],
        tuple[dict[str, object], ...],
        set[str],
    ]:
        """Align every provider type before coreference or quarantine its mention."""

        if self.governed_schema is None:
            return mentions, (), (), set()
        aligned_mentions: list[Mention] = []
        quarantined_mentions: list[Mention] = []
        results: list[dict[str, object]] = []
        off_ontology: set[str] = set()
        for mention in mentions:
            provider_type = mention.entity_type
            provider_mention_id = mention.mention_id
            governed_type = self.governed_schema.aligned_entity_type(provider_type)
            if governed_type is None:
                off_ontology.add(provider_type)
                quarantined = mention.model_copy(
                    update={"resolution_status": ResolutionStatus.REVIEW}
                )
                quarantined_mentions.append(quarantined)
                status = "quarantined"
                governed_mention_id: str | None = None
            else:
                provenance = mention.provenance.model_copy(
                    update={
                        "configuration_id": (
                            f"{mention.provenance.configuration_id};"
                            "governed_entity_type_alignment="
                            f"{provider_type}->{governed_type};"
                            f"schema_sha256={self.governed_schema.source_sha256}"
                        )
                    }
                )
                aligned = mention.model_copy(
                    update={
                        "entity_type": governed_type,
                        "provenance": provenance,
                    }
                )
                aligned_mentions.append(aligned)
                status = "aligned"
                governed_mention_id = aligned.mention_id
            results.append(
                {
                    "provider_mention_id": provider_mention_id,
                    "governed_mention_id": governed_mention_id,
                    "document_version": mention.document_version,
                    "start": mention.span.start,
                    "end": mention.span.end,
                    "surface": mention.surface,
                    "provider_entity_type": provider_type,
                    "governed_entity_type": governed_type,
                    "status": status,
                    "schema_id": self.governed_schema.schema_id,
                    "schema_version": self.governed_schema.version,
                    "schema_sha256": self.governed_schema.source_sha256,
                }
            )
        return (
            tuple(aligned_mentions),
            tuple(quarantined_mentions),
            tuple(results),
            off_ontology,
        )

    def _align(
        self,
        claims: tuple[Claim, ...],
    ) -> tuple[
        tuple[Claim, ...],
        tuple[bool, ...],
        tuple[dict[str, object], ...],
        set[str],
    ]:
        if self.governed_schema is None:
            return claims, (True,) * len(claims), (), set()
        aligned_claims: list[Claim] = []
        alignment_validity: list[bool] = []
        alignment_results: list[dict[str, object]] = []
        off_ontology: set[str] = set()
        for claim in claims:
            provider_claim_id = claim.claim_id
            provider_predicate = claim.predicate
            aligned = self.governed_schema.aligned_predicate(
                claim.predicate,
                kind=claim.kind,
            )
            if aligned is None:
                off_ontology.add(claim.predicate)
                aligned_claims.append(claim)
                alignment_validity.append(False)
                governed_claim_id: str | None = None
                status = "quarantined"
            else:
                governed_claim = claim.model_copy(update={"predicate": aligned})
                aligned_claims.append(governed_claim)
                alignment_validity.append(True)
                governed_claim_id = governed_claim.claim_id
                status = "aligned"
            alignment_results.append(
                {
                    "provider_claim_id": provider_claim_id,
                    "governed_claim_id": governed_claim_id,
                    "claim_kind": claim.kind.value,
                    "provider_predicate": provider_predicate,
                    "governed_predicate": aligned,
                    "status": status,
                    "schema_id": self.governed_schema.schema_id,
                    "schema_version": self.governed_schema.version,
                    "schema_sha256": self.governed_schema.source_sha256,
                }
            )
        return (
            tuple(aligned_claims),
            tuple(alignment_validity),
            tuple(alignment_results),
            off_ontology,
        )

    @classmethod
    def _resolve_claim_arguments(
        cls,
        claims: tuple[Claim, ...],
        mentions: tuple[Mention, ...],
    ) -> tuple[
        tuple[Claim, ...],
        tuple[bool, ...],
        tuple[dict[str, object], ...],
    ]:
        """Bind claim arguments only through exact, occurrence-level anchors.

        A provider-supplied mention ID is authoritative and must resolve to a
        surface-compatible mention contained by the claim evidence. When an ID
        is absent, the pipeline may recover it only when exactly one compatible
        mention occurrence is contained by the evidence. Unknown IDs,
        mismatches, missing anchors, and repeated compatible occurrences make
        the complete claim ineligible for promotion.
        """

        by_id = {mention.mention_id: mention for mention in mentions}
        resolved_claims: list[Claim] = []
        validity: list[bool] = []
        results: list[dict[str, object]] = []

        for claim in claims:
            original_claim_id = claim.claim_id
            argument_results: list[dict[str, object]] = []

            subject, subject_result = cls._resolve_claim_argument(
                argument_name="subject",
                value=claim.subject,
                mention_id=claim.subject_mention_id,
                evidence=claim.evidence,
                mentions=mentions,
                by_id=by_id,
            )
            argument_results.append(subject_result)

            obj: Mention | None = None
            if claim.object is not None:
                obj, object_result = cls._resolve_claim_argument(
                    argument_name="object",
                    value=claim.object,
                    mention_id=claim.object_mention_id,
                    evidence=claim.evidence,
                    mentions=mentions,
                    by_id=by_id,
                )
                argument_results.append(object_result)
            elif claim.object_mention_id is not None:
                argument_results.append(
                    {
                        "argument": "object",
                        "value": None,
                        "provided_mention_id": claim.object_mention_id,
                        "resolved_mention_id": None,
                        "candidate_mention_ids": [],
                        "status": "unexpected_mention_id",
                    }
                )

            resolved_roles: list[EventRole] = []
            for index, role in enumerate(claim.event_roles):
                matched, role_result = cls._resolve_claim_argument(
                    argument_name=f"event_role:{index}:{role.role}",
                    value=role.value,
                    mention_id=role.mention_id,
                    evidence=claim.evidence,
                    mentions=mentions,
                    by_id=by_id,
                )
                argument_results.append(role_result)
                resolved_roles.append(
                    role.model_copy(
                        update={
                            "mention_id": (
                                matched.mention_id if matched is not None else role.mention_id
                            )
                        }
                    )
                )

            accepted = all(
                item["status"] in {"provided_id", "unique_evidence_anchor"}
                for item in argument_results
            )
            updated = claim.model_copy(
                update={
                    "subject_mention_id": (
                        subject.mention_id
                        if subject is not None
                        else claim.subject_mention_id
                    ),
                    "object_mention_id": (
                        obj.mention_id if obj is not None else claim.object_mention_id
                    ),
                    "event_roles": tuple(resolved_roles),
                }
            )
            resolved_claims.append(updated)
            validity.append(accepted)
            results.append(
                {
                    "input_claim_id": original_claim_id,
                    "resolved_claim_id": updated.claim_id,
                    "status": "resolved" if accepted else "quarantined",
                    "arguments": argument_results,
                    "evidence_ids": [item.evidence_id for item in claim.evidence],
                    "provider": claim.provenance.provider,
                    "extractor": claim.provenance.extractor,
                }
            )
        return tuple(resolved_claims), tuple(validity), tuple(results)

    @staticmethod
    def _resolve_claim_argument(
        *,
        argument_name: str,
        value: str,
        mention_id: str | None,
        evidence: tuple[Evidence, ...],
        mentions: tuple[Mention, ...],
        by_id: dict[str, Mention],
    ) -> tuple[Mention | None, dict[str, object]]:
        value_key = normalize_text(value)

        def surface_matches(mention: Mention) -> bool:
            keys = {normalize_text(mention.surface)}
            if mention.coreference_anchor:
                keys.add(normalize_text(mention.coreference_anchor))
            return value_key in keys

        def evidence_contains(mention: Mention) -> bool:
            return any(
                item.span.start <= mention.span.start
                and mention.span.end <= item.span.end
                for item in evidence
            )

        compatible = tuple(
            mention
            for mention in mentions
            if surface_matches(mention) and evidence_contains(mention)
        )
        base: dict[str, object] = {
            "argument": argument_name,
            "value": value,
            "provided_mention_id": mention_id,
            "resolved_mention_id": None,
            "candidate_mention_ids": [
                mention.mention_id for mention in compatible
            ],
        }
        if mention_id is not None:
            selected = by_id.get(mention_id)
            if selected is None:
                return None, {**base, "status": "unknown_mention_id"}
            if not surface_matches(selected):
                return None, {**base, "status": "surface_mismatch"}
            if not evidence_contains(selected):
                return None, {**base, "status": "outside_claim_evidence"}
            return selected, {
                **base,
                "resolved_mention_id": selected.mention_id,
                "status": "provided_id",
            }
        if len(compatible) == 1:
            selected = compatible[0]
            return selected, {
                **base,
                "resolved_mention_id": selected.mention_id,
                "status": "unique_evidence_anchor",
            }
        if compatible:
            return None, {**base, "status": "ambiguous_evidence_anchor"}
        return None, {**base, "status": "unresolved_evidence_anchor"}

    @classmethod
    def _consolidate_semantic_claims(
        cls,
        claims: tuple[Claim, ...],
        ontology_alignment_validity: tuple[bool, ...],
        argument_resolution_validity: tuple[bool, ...],
        alignment_results: tuple[dict[str, object], ...],
        document_text: str,
        minimum_claim_confidence: float,
    ) -> tuple[
        tuple[Claim, ...],
        tuple[bool, ...],
        tuple[bool, ...],
        tuple[dict[str, object], ...],
        tuple[dict[str, object], ...],
    ]:
        """Merge provider duplicates after occurrence and ontology convergence.

        Providers can ground the same assertion with different evidence kinds
        (for example, a sentence versus an identical passage). Those are
        independent observations of one semantic claim, not two graph
        relationships. Consolidation happens only after exact argument
        resolution and governed-predicate alignment so distinct occurrences,
        claim kinds, directions, and qualifiers remain distinct.
        """

        if len(ontology_alignment_validity) != len(claims):
            raise ProviderContractError(
                "claim ontology alignment did not account for every claim"
            )
        if len(argument_resolution_validity) != len(claims):
            raise ProviderContractError(
                "claim argument resolution did not account for every claim"
            )

        grouped: dict[tuple[object, ...], list[int]] = {}
        group_order: list[tuple[object, ...]] = []
        for index, claim in enumerate(claims):
            if (
                ontology_alignment_validity[index]
                and argument_resolution_validity[index]
                and claim.has_complete_provenance(document_text)
                and claim.provenance.confidence
                >= minimum_claim_confidence
            ):
                key = ("semantic", *cls._semantic_claim_occurrence_key(claim))
            else:
                key = ("ineligible", index)
            if key not in grouped:
                grouped[key] = []
                group_order.append(key)
            grouped[key].append(index)

        consolidated: list[Claim] = []
        consolidated_ontology: list[bool] = []
        consolidated_arguments: list[bool] = []
        merge_results: list[dict[str, object]] = []
        publication_ids: dict[str, str] = {}
        for key in group_order:
            indices = grouped[key]
            members = tuple(claims[index] for index in indices)
            merged = (
                cls._merge_semantic_claim_group(members)
                if len(members) > 1
                else members[0]
            )
            consolidated.append(merged)
            consolidated_ontology.append(
                all(ontology_alignment_validity[index] for index in indices)
            )
            consolidated_arguments.append(
                all(argument_resolution_validity[index] for index in indices)
            )
            for member in members:
                publication_ids[member.claim_id] = merged.claim_id
            if len(members) > 1:
                merge_results.append(
                    {
                        "semantic_claim_key": stable_id(
                            "semantic_claim",
                            cls._semantic_claim_occurrence_key(merged),
                        ),
                        "input_claim_ids": [
                            member.claim_id for member in members
                        ],
                        "output_claim_id": merged.claim_id,
                        "merged_claim_count": len(members),
                        "providers": sorted(
                            {
                                provider
                                for member in members
                                for provider in cls._provenance_tokens(
                                    member.provenance.provider
                                )
                            }
                        ),
                        "evidence_ids": [
                            evidence.evidence_id
                            for evidence in merged.evidence
                        ],
                        "supports": [
                            {
                                "claim_id": member.claim_id,
                                "evidence_ids": [
                                    evidence.evidence_id
                                    for evidence in member.evidence
                                ],
                                "provenance": member.provenance.model_dump(
                                    mode="json"
                                ),
                            }
                            for member in members
                        ],
                    }
                )

        updated_alignment = tuple(
            (
                {
                    **item,
                    "publication_claim_id": publication_ids[
                        str(item["governed_claim_id"])
                    ],
                }
                if item.get("governed_claim_id") is not None
                and str(item["governed_claim_id"]) in publication_ids
                else item
            )
            for item in alignment_results
        )
        return (
            tuple(consolidated),
            tuple(consolidated_ontology),
            tuple(consolidated_arguments),
            updated_alignment,
            tuple(merge_results),
        )

    @staticmethod
    def _semantic_claim_occurrence_key(claim: Claim) -> tuple[object, ...]:
        def argument_key(
            mention_id: str | None,
            entity_id: str | None,
            surface: str | None,
        ) -> tuple[str, str]:
            if mention_id:
                return "mention", mention_id
            if entity_id:
                return "entity", entity_id
            return "surface", normalize_text(surface or "")

        roles = tuple(
            sorted(
                (
                    role.role.casefold(),
                    argument_key(
                        role.mention_id,
                        role.canonical_entity_id,
                        role.value,
                    ),
                )
                for role in claim.event_roles
            )
        )
        evidence_anchors = tuple(
            sorted(
                {
                    (
                        evidence.document_version,
                        evidence.span.start,
                        evidence.span.end,
                        evidence.span.text,
                    )
                    for evidence in claim.evidence
                }
            )
        )
        return (
            claim.document.version,
            claim.kind.value,
            argument_key(
                claim.subject_mention_id,
                claim.subject_entity_id,
                claim.subject,
            ),
            claim.predicate.casefold(),
            argument_key(
                claim.object_mention_id,
                claim.object_entity_id,
                claim.object,
            ),
            roles,
            evidence_anchors,
            claim.negated,
            claim.modality.value,
            normalize_text(claim.attribution or ""),
            normalize_text(claim.temporal or ""),
            normalize_text(claim.quantity or ""),
        )

    @classmethod
    def _merge_semantic_claim_group(
        cls,
        claims: tuple[Claim, ...],
    ) -> Claim:
        if len(claims) < 2:
            raise ValueError("semantic claim consolidation requires duplicates")

        def provider_count(claim: Claim) -> int:
            return max(
                1,
                len(cls._provenance_tokens(claim.provenance.provider)),
            )

        exemplar = min(
            claims,
            key=lambda claim: (
                -provider_count(claim),
                -claim.provenance.confidence,
                claim.claim_id,
            ),
        )
        evidence_by_id = {
            evidence.evidence_id: evidence
            for claim in claims
            for evidence in claim.evidence
        }
        evidence = tuple(
            sorted(
                evidence_by_id.values(),
                key=lambda item: (
                    item.span.start,
                    item.span.end,
                    item.kind.value,
                    item.evidence_id,
                ),
            )
        )
        providers = sorted(
            {
                provider
                for claim in claims
                for provider in cls._provenance_tokens(
                    claim.provenance.provider
                )
            }
        )
        versions = sorted(
            {
                version
                for claim in claims
                for version in cls._provenance_tokens(
                    claim.provenance.model_version
                )
            }
        )
        member_ids = sorted(claim.claim_id for claim in claims)
        return exemplar.model_copy(
            update={
                "evidence": evidence,
                "provenance": Provenance(
                    extractor="semantic_claim_consolidator",
                    provider="+".join(providers),
                    model_version="+".join(versions),
                    configuration_id=stable_id(
                        "semantic_claim_union",
                        *member_ids,
                    ),
                    confidence=min(
                        claim.provenance.confidence for claim in claims
                    ),
                ),
            }
        )

    @staticmethod
    def _provenance_tokens(value: str) -> tuple[str, ...]:
        return tuple(
            token
            for token in (item.strip() for item in value.split("+"))
            if token
        )

    def _promote_or_quarantine(
        self,
        claims: tuple[Claim, ...],
        text: str,
        ontology_alignment_validity: tuple[bool, ...],
        argument_resolution_validity: tuple[bool, ...],
    ) -> tuple[
        tuple[Claim, ...],
        tuple[Claim, ...],
        tuple[ClaimShaclResult, ...],
    ]:
        promoted: list[Claim] = []
        quarantined: list[Claim] = []
        shacl_results: list[ClaimShaclResult] = []
        seen_claim_ids: set[str] = set()
        if len(argument_resolution_validity) != len(claims):
            raise ProviderContractError(
                "claim argument resolution did not account for every claim"
            )
        if len(ontology_alignment_validity) != len(claims):
            raise ProviderContractError(
                "claim ontology alignment did not account for every claim"
            )
        for index, claim in enumerate(claims):
            is_duplicate = claim.claim_id in seen_claim_ids
            seen_claim_ids.add(claim.claim_id)
            ontology_aligned = ontology_alignment_validity[index]
            shacl_conforms = True
            if (
                self.claim_shacl_gate is not None
                and ontology_aligned
                and not is_duplicate
            ):
                shacl_result = self.claim_shacl_gate.validate(claim, text)
                shacl_results.append(shacl_result)
                shacl_conforms = shacl_result.executed and shacl_result.conforms
            accepted = (
                not is_duplicate
                and argument_resolution_validity[index]
                and claim.has_complete_provenance(text)
                and claim.provenance.confidence >= self.config.minimum_claim_confidence
                and ontology_aligned
                and shacl_conforms
            )
            (promoted if accepted else quarantined).append(
                claim.promoted() if accepted else claim.quarantined()
            )
        return tuple(promoted), tuple(quarantined), tuple(shacl_results)

    def _provider_participation_ids(
        self,
        resolution: IdentityResolution,
    ) -> tuple[str, ...]:
        """Return every provider invoked by extraction and identity resolution."""

        provider_ids: list[str] = []
        for component in (self.mention_detector, self.claim_extractor):
            values = getattr(component, "provider_ids", ())
            if isinstance(values, tuple | list):
                provider_ids.extend(str(value) for value in values)
        provider_ids.extend(resolution.provider_participation_ids)
        return tuple(dict.fromkeys(provider_ids))

    @staticmethod
    def _bind_canonical_entities(
        claims: tuple[Claim, ...],
        mentions: tuple[Mention, ...],
    ) -> tuple[Claim, ...]:
        by_id = {mention.mention_id: mention for mention in mentions}
        bound: list[Claim] = []
        for claim in claims:
            subject = by_id.get(claim.subject_mention_id or "")
            obj = by_id.get(claim.object_mention_id or "")
            roles = tuple(
                role.model_copy(
                    update={
                        "value": (
                            by_id[role.mention_id].coreference_anchor
                            or by_id[role.mention_id].surface
                            if role.mention_id in by_id
                            else role.value
                        ),
                        "canonical_entity_id": (
                            by_id[role.mention_id].canonical_entity_id
                            if role.mention_id in by_id
                            else role.canonical_entity_id
                        )
                    }
                )
                for role in claim.event_roles
            )
            bound.append(
                claim.model_copy(
                    update={
                        "subject": (
                            subject.coreference_anchor or subject.surface
                            if subject is not None
                            else claim.subject
                        ),
                        "object": (
                            obj.coreference_anchor or obj.surface
                            if obj is not None
                            else claim.object
                        ),
                        "subject_entity_id": (
                            subject.canonical_entity_id
                            if subject is not None
                            else claim.subject_entity_id
                        ),
                        "object_entity_id": (
                            obj.canonical_entity_id
                            if obj is not None
                            else claim.object_entity_id
                        ),
                        "event_roles": roles,
                    }
                )
            )
        return tuple(bound)


class RuleSegmenter:
    """Deterministic sentence segmentation with exact contextual selectors."""

    def __init__(self, *, context_chars: int = 24) -> None:
        self.context_chars = context_chars

    def segment(self, document: DocumentRef, text: str) -> tuple[Evidence, ...]:
        segments: list[Evidence] = []
        for match in _SENTENCE_RE.finditer(text):
            raw = match.group()
            leading = len(raw) - len(raw.lstrip())
            trailing = len(raw) - len(raw.rstrip())
            start = match.start() + leading
            end = match.end() - trailing
            if start >= end:
                continue
            span = TextSpan(start=start, end=end, text=text[start:end])
            segments.append(
                Evidence.build(
                    document,
                    span=span,
                    kind=EvidenceKind.SENTENCE,
                    prefix=text[max(0, start - self.context_chars) : start],
                    suffix=text[end : min(len(text), end + self.context_chars)],
                )
            )
        return tuple(segments)


class RuleMentionDetector:
    """Conservative offline mention detector; real NER providers use the same port."""

    extractor = "rule_mention_detector"
    provider_id = "deterministic_rule_mentions"

    @staticmethod
    def _provenance(confidence: float) -> Provenance:
        return Provenance(
            extractor=RuleMentionDetector.extractor,
            provider="local_rules",
            model_version="mention-rules-v2",
            configuration_id="default-en-v2",
            confidence=confidence,
        )

    def detect(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
    ) -> tuple[Mention, ...]:
        del segments  # provider receives structure even though these rules scan globally
        claimed: list[tuple[int, int]] = []
        mentions: list[Mention] = []

        for mention_type, pattern in _SPECIFIC_MENTION_PATTERNS:
            for match in pattern.finditer(text):
                mentions.append(
                    self._mention(
                        document,
                        text,
                        match.start(),
                        match.end(),
                        mention_type,
                        confidence=0.99,
                    )
                )
                claimed.append((match.start(), match.end()))

        for match in _PROPER_NAME_RE.finditer(text):
            start, end = match.span()
            if any(start < claimed_end and end > claimed_start for claimed_start, claimed_end in claimed):
                continue
            surface = match.group()
            if normalize_text(surface) in _NON_ENTITY_CAPITALS:
                continue
            mentions.append(
                self._mention(
                    document,
                    text,
                    start,
                    end,
                    self._infer_type(text, start, surface),
                    confidence=0.78,
                )
            )
        return tuple(sorted(mentions, key=lambda item: (item.span.start, item.span.end)))

    def _mention(
        self,
        document: DocumentRef,
        text: str,
        start: int,
        end: int,
        entity_type: str,
        *,
        confidence: float,
    ) -> Mention:
        span = TextSpan(start=start, end=end, text=text[start:end])
        evidence = Evidence.build(document, span=span, kind=EvidenceKind.MENTION)
        return Mention(
            document_version=document.version,
            evidence=evidence,
            surface=span.text,
            entity_type=entity_type,
            provenance=self._provenance(confidence),
        )

    @staticmethod
    def _infer_type(text: str, start: int, surface: str) -> str:
        tokens = normalize_text(surface).split()
        final = tokens[-1]
        if final in _ORGANIZATION_SUFFIXES:
            return "Organization"
        if final in _PLACE_SUFFIXES:
            return "Place"
        if final in _ARTIFACT_SUFFIXES:
            return "Artifact"
        before = text[max(0, start - 32) : start]
        if re.search(r"(?i:according\s+to|reported\s+by|stated\s+by)\s*$", before):
            return "Person"
        if len(tokens) >= 2:
            return "Person"
        return "Entity"


class ConservativeCoreferenceResolver:
    """Cluster exact names and unique full-person surname mentions only.

    Substring matching is intentionally forbidden: ``Ann`` cannot merge with
    ``Anna`` and ``York`` cannot merge with ``Yorkshire``. An ambiguous surname
    such as ``Smith`` stays in its own cluster when more than one full-name anchor
    exists.
    """

    def cluster(self, mentions: tuple[Mention, ...]) -> tuple[Mention, ...]:
        exact_groups: dict[str, list[Mention]] = {}
        for mention in mentions:
            exact_groups.setdefault(normalize_text(mention.surface), []).append(mention)

        full_people: dict[str, set[str]] = {}
        anchor_surface: dict[str, str] = {}
        for mention in mentions:
            tokens = normalize_text(mention.surface).split()
            if mention.entity_type == "Person" and len(tokens) >= 2:
                key = " ".join(tokens)
                full_people.setdefault(tokens[-1], set()).add(key)
                current = anchor_surface.get(key)
                if current is None or len(mention.surface) > len(current):
                    anchor_surface[key] = mention.surface

        group_key_by_mention: dict[str, str] = {}
        for mention in mentions:
            key = normalize_text(mention.surface)
            tokens = key.split()
            if len(tokens) == 1 and len(full_people.get(tokens[0], set())) == 1:
                key = next(iter(full_people[tokens[0]]))
            group_key_by_mention[mention.mention_id] = key

        grouped: dict[str, list[Mention]] = {}
        for mention in mentions:
            grouped.setdefault(group_key_by_mention[mention.mention_id], []).append(mention)

        clustered: list[Mention] = []
        for key, members in grouped.items():
            anchor = anchor_surface.get(
                key,
                sorted(
                    (member.surface for member in members),
                    key=lambda value: (-len(value), value.casefold()),
                )[0],
            )
            anchor_type = sorted(
                member.entity_type
                for member in members
                if member.entity_type != "Entity"
            )
            entity_type = anchor_type[0] if anchor_type else members[0].entity_type
            cluster_id = stable_id(
                "coref",
                members[0].document_version,
                entity_type,
                key,
            )
            status = (
                ResolutionStatus.COREFERENCE_ONLY
                if len(members) > 1 or normalize_text(anchor) != normalize_text(members[0].surface)
                else ResolutionStatus.UNRESOLVED
            )
            for member in members:
                clustered.append(
                    member.model_copy(
                        update={
                            "entity_type": (
                                entity_type if member.entity_type == "Entity" else member.entity_type
                            ),
                            "coreference_cluster_id": cluster_id,
                            "coreference_anchor": anchor,
                            "resolution_status": status,
                        }
                    )
                )
        return tuple(
            sorted(clustered, key=lambda item: (item.span.start, item.span.end))
        )


# Backwards-compatible name for callers of the preliminary implementation.
StringCoreference = ConservativeCoreferenceResolver


class PatternClaimExtractor:
    """Deterministic semantic-pattern extractor with event and intent output."""

    extractor = "pattern_claim_extractor"
    provider_id = "deterministic_pattern_claims"

    @staticmethod
    def _provenance(confidence: float = 0.72) -> Provenance:
        return Provenance(
            extractor=PatternClaimExtractor.extractor,
            provider="local_rules",
            model_version="claim-patterns-v2",
            configuration_id="default-en-v2",
            confidence=confidence,
        )

    def extract(
        self,
        document: DocumentRef,
        text: str,
        segments: tuple[Evidence, ...],
        mentions: tuple[Mention, ...],
    ) -> tuple[Claim, ...]:
        claims: list[Claim] = []
        for segment in segments:
            segment_mentions = tuple(
                mention
                for mention in mentions
                if segment.span.start <= mention.span.start
                and mention.span.end <= segment.span.end
                and self._is_argument(mention)
            )
            for intent in _INTENT_RE.finditer(segment.span.text):
                absolute_start = segment.span.start + intent.start()
                absolute_end = segment.span.start + intent.end()
                arguments = self._arguments(
                    segment_mentions, absolute_start, absolute_end
                )
                if arguments is None:
                    continue
                subject, obj = arguments
                claims.append(
                    self._binary_claim(
                        document,
                        ClaimKind.INTENT,
                        normalize_text(intent.group("verb")).replace(" ", "_"),
                        subject,
                        obj,
                        segment,
                    )
                )

            for predicate, kind, pattern in _RELATION_PATTERNS:
                for relation in pattern.finditer(segment.span.text):
                    absolute_start = segment.span.start + relation.start()
                    absolute_end = segment.span.start + relation.end()
                    arguments = self._arguments(
                        segment_mentions, absolute_start, absolute_end
                    )
                    if arguments is None:
                        continue
                    subject, obj = arguments
                    binary = self._binary_claim(
                        document, kind, predicate, subject, obj, segment
                    )
                    claims.append(binary)
                    if predicate in _EVENT_PREDICATES:
                        claims.append(
                            Claim(
                                document=document,
                                kind=ClaimKind.EVENT,
                                subject=binary.subject,
                                predicate=predicate,
                                object=binary.object,
                                subject_mention_id=subject.mention_id,
                                object_mention_id=obj.mention_id,
                                event_roles=(
                                    EventRole(
                                        role="agent",
                                        value=binary.subject,
                                        mention_id=subject.mention_id,
                                    ),
                                    EventRole(
                                        role="theme",
                                        value=binary.object or obj.surface,
                                        mention_id=obj.mention_id,
                                    ),
                                ),
                                evidence=(segment,),
                                provenance=self._provenance(0.70),
                            )
                        )
        return tuple(claims)

    @classmethod
    def _binary_claim(
        cls,
        document: DocumentRef,
        kind: ClaimKind,
        predicate: str,
        subject: Mention,
        obj: Mention,
        evidence: Evidence,
    ) -> Claim:
        return Claim(
            document=document,
            kind=kind,
            subject=subject.coreference_anchor or subject.surface,
            predicate=predicate,
            object=obj.coreference_anchor or obj.surface,
            subject_mention_id=subject.mention_id,
            object_mention_id=obj.mention_id,
            evidence=(evidence,),
            provenance=cls._provenance(),
        )

    @staticmethod
    def _arguments(
        mentions: tuple[Mention, ...],
        predicate_start: int,
        predicate_end: int,
    ) -> tuple[Mention, Mention] | None:
        before = [mention for mention in mentions if mention.span.end <= predicate_start]
        after = [mention for mention in mentions if mention.span.start >= predicate_end]
        if not before or not after:
            return None
        return max(before, key=lambda item: item.span.end), min(
            after, key=lambda item: item.span.start
        )

    @staticmethod
    def _is_argument(mention: Mention) -> bool:
        return mention.entity_type not in {
            "Date",
            "EmailAddress",
            "Version",
        }


class RuleAttributeExtractor:
    """Preserve source negation, modality, time, attribution, and quantity."""

    def annotate(self, text: str, claims: tuple[Claim, ...]) -> tuple[Claim, ...]:
        del text
        annotated: list[Claim] = []
        for claim in claims:
            evidence_text = " ".join(item.span.text for item in claim.evidence)
            modality = claim.modality
            if modality == Modality.ASSERTED:
                if _POSSIBLE_RE.search(evidence_text):
                    modality = Modality.POSSIBLE
                elif _PROBABLE_RE.search(evidence_text):
                    modality = Modality.PROBABLE
                elif _HYPOTHETICAL_RE.search(evidence_text):
                    modality = Modality.HYPOTHETICAL
            attribution = _ATTRIBUTION_RE.search(evidence_text)
            temporal = _TEMPORAL_RE.search(evidence_text)
            quantity = _QUANTITY_RE.search(evidence_text)
            annotated.append(
                claim.model_copy(
                    update={
                        "negated": claim.negated or bool(_NEGATION_RE.search(evidence_text)),
                        "modality": modality,
                        "attribution": claim.attribution
                        or (attribution.group("source") if attribution else None),
                        "temporal": claim.temporal
                        or (temporal.group(0) if temporal else None),
                        "quantity": claim.quantity
                        or (quantity.group(0) if quantity else None),
                    }
                )
            )
        return tuple(annotated)


class KeepUnresolvedResolver:
    """Create no canonical identity when no explicit resolver is configured."""

    def resolve(self, mentions: tuple[Mention, ...]) -> IdentityResolution:
        return IdentityResolution(mentions=mentions)


class ExplicitNilObservedResolver:
    """Retain NIL mentions as observed/coreference identities, never linked."""

    provider_id = "explicit_nil_retain_observed"
    nil_policy = "retain_observed"

    def resolve(self, mentions: tuple[Mention, ...]) -> IdentityResolution:
        grouped: dict[str, list[Mention]] = {}
        for mention in mentions:
            key = mention.coreference_cluster_id or mention.mention_id
            grouped.setdefault(key, []).append(mention)

        resolved: dict[str, Mention] = {}
        decisions: list[dict[str, object]] = []
        nil_clusters: list[str] = []
        for cluster_id, members_list in sorted(grouped.items()):
            members = tuple(members_list)
            anchor = members[0].coreference_anchor or members[0].surface
            confidence = min(member.provenance.confidence for member in members)
            nil_clusters.append(cluster_id)
            decisions.append(
                {
                    "cluster_id": cluster_id,
                    "canonical_entity_id": None,
                    "canonical_name": anchor,
                    "external_match": False,
                    "nil_policy": self.nil_policy,
                    "provider_ids": [self.provider_id],
                }
            )
            for member in members:
                resolved[member.mention_id] = member.model_copy(
                    update={
                        "resolution_status": (
                            ResolutionStatus.COREFERENCE_ONLY
                            if member.coreference_cluster_id is not None
                            else ResolutionStatus.UNRESOLVED
                        ),
                        "resolution_confidence": confidence,
                    }
                )
        return IdentityResolution(
            mentions=tuple(resolved[item.mention_id] for item in mentions),
            provider_participation_ids=(self.provider_id,),
            nil_policy=self.nil_policy,
            nil_cluster_ids=tuple(nil_clusters),
            decisions=tuple(decisions),
        )


class ProviderIdentityResolver:
    """Resolve coreference clusters through an injected real entity-link provider."""

    def __init__(
        self,
        provider: EntityLinkProvider,
        *,
        minimum_confidence: float = 0.9,
    ) -> None:
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be between zero and one")
        self.provider = provider
        self.minimum_confidence = minimum_confidence
        self.provider_id = getattr(
            provider,
            "provider_id",
            provider.__class__.__name__,
        )

    def resolve(self, mentions: tuple[Mention, ...]) -> IdentityResolution:
        grouped: dict[str, list[Mention]] = {}
        for mention in mentions:
            key = mention.coreference_cluster_id or mention.mention_id
            grouped.setdefault(key, []).append(mention)

        resolved_by_id: dict[str, Mention] = {item.mention_id: item for item in mentions}
        entities: list[CanonicalEntity] = []
        for members_list in grouped.values():
            members = tuple(members_list)
            anchor = members[0].coreference_anchor or members[0].surface
            entity_type = members[0].entity_type
            link = self.provider.link(anchor, entity_type, members)
            if link is None:
                continue
            if link.confidence < self.minimum_confidence:
                for member in members:
                    resolved_by_id[member.mention_id] = member.model_copy(
                        update={
                            "resolution_status": ResolutionStatus.REVIEW,
                            "resolution_confidence": link.confidence,
                        }
                    )
                continue

            entity_id = stable_id(
                "canonical", link.external_system, link.external_id
            )
            entity = CanonicalEntity(
                entity_id=entity_id,
                canonical_name=link.canonical_name,
                entity_type=link.entity_type,
                mention_ids=tuple(member.mention_id for member in members),
                aliases=tuple(
                    sorted({member.surface for member in members} - {link.canonical_name})
                ),
                external_system=link.external_system,
                external_id=link.external_id,
                confidence=link.confidence,
            )
            entities.append(entity)
            for member in members:
                resolved_by_id[member.mention_id] = member.model_copy(
                    update={
                        "canonical_entity_id": entity_id,
                        "resolution_status": ResolutionStatus.LINKED,
                        "resolution_confidence": link.confidence,
                    }
                )
        return IdentityResolution(
            mentions=tuple(resolved_by_id[item.mention_id] for item in mentions),
            canonical_entities=tuple(entities),
            provider_participation_ids=(str(self.provider_id),),
            nil_policy="retain_unresolved",
            nil_cluster_ids=tuple(
                key
                for key, members in grouped.items()
                if all(
                    resolved_by_id[member.mention_id].resolution_status
                    != ResolutionStatus.LINKED
                    for member in members
                )
            ),
        )


__all__ = [
    "AttributeExtractor",
    "ClaimExtractor",
    "ConservativeCoreferenceResolver",
    "CoreferenceResolver",
    "EntityLink",
    "EntityLinkProvider",
    "ExplicitNilObservedResolver",
    "IdentityResolution",
    "IdentityResolver",
    "KeepUnresolvedResolver",
    "MentionDetector",
    "PatternClaimExtractor",
    "PipelineResult",
    "ProviderContractError",
    "ProviderIdentityResolver",
    "RuleAttributeExtractor",
    "RuleMentionDetector",
    "RuleSegmenter",
    "RuntimeProfile",
    "Segmenter",
    "StringCoreference",
    "UnstructuredPipeline",
    "UnstructuredPipelineConfig",
]
