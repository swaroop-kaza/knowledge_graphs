"""Scoped, conjunctive resolution pipeline for unstructured publication."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    GraphFact,
    RelationCandidate,
    stable_id,
)
from kg_construction.infrastructure.resolution.common import (
    reconcile_entity_references,
)

_TECHNICAL_TYPES = frozenset({"Document", "Mention", "Claim", "Event"})
_WIKIDATA_ID_RE = re.compile(r"(?:^|/)(Q[1-9][0-9]*)$", re.IGNORECASE)
_IDENTITY_LINK_RELATIONS = frozenset(
    {"DENOTES_OBSERVED_ENTITY", "RESOLVED_TO_CANONICAL_ENTITY"}
)


class Resolver(Protocol):
    """Minimal resolver contract for the conjunctive ensemble."""

    name: str

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch: ...


class ExternalIdentityPromotionPolicy(BaseModel):
    """Conservative release policy for turning linker candidates into identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_id: str = Field(min_length=1)
    minimum_provider_consensus: int = Field(default=2, ge=2)
    minimum_confidence: float = Field(default=0.90, ge=0.0, le=1.0)


class ExternalIdentityDecision(BaseModel):
    """One auditable accepted, conflicting, pre-existing, or NIL decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster_id: str
    observed_candidate_ids: tuple[str, ...]
    status: Literal["accepted", "conflict", "nil", "preexisting"]
    canonical_entity_id: str | None = None
    external_system: str | None = None
    external_id: str | None = None
    canonical_name: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    provider_ids: tuple[str, ...] = ()
    conflict_identity_keys: tuple[str, ...] = ()
    review_required: bool
    reason: str


class GovernedExternalIdentityPromoter:
    """Promote only provider-consensus external links to canonical resources.

    Search results remain candidates until this gate runs.  One high-scoring
    nearest neighbour is never enough.  Two providers must independently agree
    on the same normalized external-system/identifier pair.  Disagreement
    inside a namespace is sent to review and unmatched resources explicitly
    remain NIL/observed.
    """

    name = "governed_external_identity_promoter"
    technique = "provider_consensus_external_identity_promotion"

    def __init__(
        self,
        *,
        governed_entity_types: Sequence[str],
        policy: ExternalIdentityPromotionPolicy,
    ) -> None:
        self.governed_entity_types = frozenset(governed_entity_types)
        self.policy = policy
        self.last_decisions: tuple[ExternalIdentityDecision, ...] = ()

    def promote(self, batch: ConstructionBatch) -> ConstructionBatch:
        candidates = [
            entity
            for entity in batch.entities
            if self._eligible(entity)
        ]
        clusters: dict[str, list[EntityCandidate]] = {}
        for entity in candidates:
            cluster_id = str(
                entity.properties.get("canonical_entity_id")
                or entity.candidate_id
            )
            clusters.setdefault(cluster_id, []).append(entity)

        decisions: list[ExternalIdentityDecision] = []
        accepted: list[
            tuple[
                ExternalIdentityDecision,
                list[EntityCandidate],
                tuple[dict[str, object], ...],
            ]
        ] = []
        for cluster_id, members in sorted(clusters.items()):
            existing = self._preexisting_decision(cluster_id, members)
            if existing is not None:
                decisions.append(existing)
                continue
            votes = self._trusted_votes(members)
            decision = self._decide(cluster_id, members, votes)
            decisions.append(decision)
            if decision.status == "accepted":
                accepted.append((decision, members, votes))

        # The same external key must never collapse incompatible governed types.
        accepted_by_key: dict[str, list[int]] = {}
        for index, (decision, _members, _votes) in enumerate(accepted):
            key = f"{decision.external_system}:{decision.external_id}"
            accepted_by_key.setdefault(key, []).append(index)
        rejected_indices: set[int] = set()
        for key, indices in accepted_by_key.items():
            semantic_types = {
                self._semantic_type(member)
                for index in indices
                for member in accepted[index][1]
            }
            if len(semantic_types) <= 1:
                continue
            rejected_indices.update(indices)
            for index in indices:
                prior, _members, _votes = accepted[index]
                replacement = prior.model_copy(
                    update={
                        "status": "conflict",
                        "canonical_entity_id": None,
                        "review_required": True,
                        "conflict_identity_keys": (key,),
                        "reason": "external identity has conflicting governed entity types",
                    }
                )
                decisions[decisions.index(prior)] = replacement

        accepted = [
            row for index, row in enumerate(accepted) if index not in rejected_indices
        ]
        decision_by_candidate = {
            candidate_id: decision
            for decision in decisions
            for candidate_id in decision.observed_candidate_ids
        }
        canonical_nodes = self._canonical_nodes(accepted)
        updated_entities = [
            self._updated_entity(entity, decision_by_candidate.get(entity.candidate_id))
            for entity in batch.entities
        ]
        accepted_by_mention: dict[str, ExternalIdentityDecision] = {}
        reference_to_canonical: dict[str, str] = {}
        for decision, members, _votes in accepted:
            canonical_id = str(decision.canonical_entity_id)
            reference_to_canonical[decision.cluster_id] = canonical_id
            for member in members:
                reference_to_canonical[member.candidate_id] = canonical_id
                for mention_id in member.properties.get("mention_ids", []):
                    accepted_by_mention[str(mention_id)] = decision

        updated_entities = [
            self._updated_mention(entity, accepted_by_mention)
            for entity in updated_entities
        ]
        rewritten_relations = [
            self._rewrite_relation(relation, reference_to_canonical)
            for relation in batch.relations
        ]
        rewritten_facts = [
            self._rewrite_fact(fact, reference_to_canonical)
            for fact in batch.facts
        ]
        relation_keys = {
            (relation.source, relation.relation_type, relation.target)
            for relation in rewritten_relations
        }
        for decision, members, _votes in accepted:
            canonical_id = str(decision.canonical_entity_id)
            for member in members:
                relation_key = (
                    member.candidate_id,
                    "RESOLVED_TO_CANONICAL_ENTITY",
                    canonical_id,
                )
                if relation_key not in relation_keys:
                    rewritten_relations.append(
                        self._identity_relation(
                            source=member.candidate_id,
                            target=canonical_id,
                            relation_type="RESOLVED_TO_CANONICAL_ENTITY",
                            evidence=member.evidence,
                            confidence=float(decision.confidence or 0.0),
                            properties={
                                "external_system": decision.external_system,
                                "external_id": decision.external_id,
                                "provider_ids": list(decision.provider_ids),
                                "governed_schema_id": self.policy.schema_id,
                            },
                        )
                    )
                    relation_keys.add(relation_key)
            mention_ids = {
                str(mention_id)
                for member in members
                for mention_id in member.properties.get("mention_ids", [])
            }
            for mention in updated_entities:
                if (
                    mention.entity_type != "Mention"
                    or str(mention.properties.get("domain_mention_id")) not in mention_ids
                ):
                    continue
                relation_key = (
                    mention.candidate_id,
                    "REFERS_TO_CANONICAL_ENTITY",
                    canonical_id,
                )
                if relation_key in relation_keys:
                    continue
                rewritten_relations.append(
                    self._identity_relation(
                        source=mention.candidate_id,
                        target=canonical_id,
                        relation_type="REFERS_TO_CANONICAL_ENTITY",
                        evidence=mention.evidence,
                        confidence=float(decision.confidence or 0.0),
                        properties={
                            "resolution_status": "externally_linked",
                            "external_system": decision.external_system,
                            "external_id": decision.external_id,
                        },
                    )
                )
                relation_keys.add(relation_key)

        serialized = [
            decision.model_dump(mode="json")
            for decision in decisions
        ]
        self.last_decisions = tuple(decisions)
        review = [
            row for row in serialized if row["review_required"] is True
        ]
        return batch.model_copy(
            update={
                "entities": [*updated_entities, *canonical_nodes],
                "relations": rewritten_relations,
                "facts": rewritten_facts,
                "metadata": {
                    **batch.metadata,
                    "external_identity_promotion": {
                        "executed": True,
                        "schema_id": self.policy.schema_id,
                        "minimum_provider_consensus": (
                            self.policy.minimum_provider_consensus
                        ),
                        "minimum_confidence": self.policy.minimum_confidence,
                        "decisions": serialized,
                        "accepted_count": sum(
                            decision.status == "accepted" for decision in decisions
                        ),
                        "conflict_count": sum(
                            decision.status == "conflict" for decision in decisions
                        ),
                        "nil_count": sum(
                            decision.status == "nil" for decision in decisions
                        ),
                    },
                    "external_identity_review_queue": review,
                },
            }
        )

    def _eligible(self, entity: EntityCandidate) -> bool:
        if entity.entity_type in _TECHNICAL_TYPES:
            return False
        if entity.properties.get("identity_layer") not in {"observed", "canonical"}:
            return False
        return self._semantic_type(entity) in self.governed_entity_types

    @staticmethod
    def _semantic_type(entity: EntityCandidate) -> str:
        if entity.entity_type == "CanonicalEntity":
            return str(entity.properties.get("semantic_entity_type", ""))
        return entity.entity_type

    @staticmethod
    def _preexisting_decision(
        cluster_id: str,
        members: list[EntityCandidate],
    ) -> ExternalIdentityDecision | None:
        existing = [
            entity
            for entity in members
            if entity.properties.get("identity_layer") == "canonical"
            and entity.properties.get("external_system")
            and entity.properties.get("external_id")
        ]
        if not existing:
            return None
        keys = {
            (
                str(entity.properties["external_system"]),
                str(entity.properties["external_id"]),
            )
            for entity in existing
        }
        if len(keys) != 1:
            return ExternalIdentityDecision(
                cluster_id=cluster_id,
                observed_candidate_ids=tuple(
                    sorted(entity.candidate_id for entity in members)
                ),
                status="conflict",
                conflict_identity_keys=tuple(
                    sorted(f"{system}:{identifier}" for system, identifier in keys)
                ),
                review_required=True,
                reason="pre-existing canonical resources disagree on external identity",
            )
        system, identifier = next(iter(keys))
        entity = existing[0]
        return ExternalIdentityDecision(
            cluster_id=cluster_id,
            observed_candidate_ids=tuple(
                sorted(member.candidate_id for member in members)
            ),
            status="preexisting",
            canonical_entity_id=str(
                entity.properties.get("canonical_entity_id") or cluster_id
            ),
            external_system=system,
            external_id=identifier,
            canonical_name=entity.name,
            confidence=entity.confidence,
            review_required=False,
            reason="canonical external identity already existed before post-link promotion",
        )

    def _trusted_votes(
        self,
        members: list[EntityCandidate],
    ) -> tuple[dict[str, object], ...]:
        votes: dict[tuple[str, str, str], dict[str, object]] = {}
        for member in members:
            links = member.properties.get("external_links", [])
            if not isinstance(links, list):
                continue
            for raw in links:
                if not isinstance(raw, dict):
                    continue
                provider = str(raw.get("provider", "")).strip()
                identity = _normalized_external_identity(
                    provider,
                    str(raw.get("id", "")),
                )
                if not provider or identity is None:
                    continue
                system, identifier = identity
                confidence = float(raw.get("confidence", 0.0))
                review_required = bool(raw.get("review_required", True))
                if (
                    confidence < self.policy.minimum_confidence
                    or review_required
                ):
                    continue
                vote_key = (provider, system, identifier)
                row = {
                    "provider_id": provider,
                    "external_system": system,
                    "external_id": identifier,
                    "raw_external_id": str(raw.get("id", "")),
                    "label": str(raw.get("label") or member.name),
                    "confidence": confidence,
                    "source_candidate_id": member.candidate_id,
                }
                current = votes.get(vote_key)
                if current is None or _float_value(current["confidence"]) < confidence:
                    votes[vote_key] = row
        return tuple(
            votes[vote_key]
            for vote_key in sorted(votes)
        )

    def _decide(
        self,
        cluster_id: str,
        members: list[EntityCandidate],
        votes: tuple[dict[str, object], ...],
    ) -> ExternalIdentityDecision:
        member_ids = tuple(sorted(member.candidate_id for member in members))
        by_identity: dict[tuple[str, str], list[dict[str, object]]] = {}
        by_system: dict[str, set[str]] = {}
        for vote in votes:
            system = str(vote["external_system"])
            identifier = str(vote["external_id"])
            by_identity.setdefault((system, identifier), []).append(vote)
            by_system.setdefault(system, set()).add(identifier)
        namespace_conflicts = {
            f"{system}:{identifier}"
            for system, identifiers in by_system.items()
            if len(identifiers) > 1
            for identifier in identifiers
        }
        if namespace_conflicts:
            return ExternalIdentityDecision(
                cluster_id=cluster_id,
                observed_candidate_ids=member_ids,
                status="conflict",
                provider_ids=tuple(
                    sorted({str(vote["provider_id"]) for vote in votes})
                ),
                conflict_identity_keys=tuple(sorted(namespace_conflicts)),
                review_required=True,
                reason="trusted providers disagree inside one external namespace",
            )
        qualified = [
            (identity, identity_votes)
            for identity, identity_votes in by_identity.items()
            if len(
                {str(vote["provider_id"]) for vote in identity_votes}
            )
            >= self.policy.minimum_provider_consensus
        ]
        if len(qualified) > 1:
            return ExternalIdentityDecision(
                cluster_id=cluster_id,
                observed_candidate_ids=member_ids,
                status="conflict",
                provider_ids=tuple(
                    sorted({str(vote["provider_id"]) for vote in votes})
                ),
                conflict_identity_keys=tuple(
                    sorted(
                        f"{system}:{identifier}"
                        for (system, identifier), _identity_votes in qualified
                    )
                ),
                review_required=True,
                reason="multiple external identities independently reached consensus",
            )
        if not qualified:
            return ExternalIdentityDecision(
                cluster_id=cluster_id,
                observed_candidate_ids=member_ids,
                status="nil",
                provider_ids=tuple(
                    sorted({str(vote["provider_id"]) for vote in votes})
                ),
                review_required=False,
                reason="no external identity reached the governed provider consensus",
            )
        (system, identifier), identity_votes = qualified[0]
        ranked = sorted(
            identity_votes,
            key=lambda vote: (
                -_float_value(vote["confidence"]),
                str(vote["provider_id"]),
            ),
        )
        providers = tuple(
            sorted({str(vote["provider_id"]) for vote in identity_votes})
        )
        confidence = min(
            _float_value(vote["confidence"]) for vote in identity_votes
        )
        return ExternalIdentityDecision(
            cluster_id=cluster_id,
            observed_candidate_ids=member_ids,
            status="accepted",
            canonical_entity_id=stable_id(
                "canonical_entity",
                system,
                identifier,
            ),
            external_system=system,
            external_id=identifier,
            canonical_name=str(ranked[0]["label"]),
            confidence=confidence,
            provider_ids=providers,
            review_required=False,
            reason="independent providers agreed above the governed threshold",
        )

    def _canonical_nodes(
        self,
        accepted: list[
            tuple[
                ExternalIdentityDecision,
                list[EntityCandidate],
                tuple[dict[str, object], ...],
            ]
        ],
    ) -> list[EntityCandidate]:
        grouped: dict[
            str,
            list[
                tuple[
                    ExternalIdentityDecision,
                    list[EntityCandidate],
                    tuple[dict[str, object], ...],
                ]
            ],
        ] = {}
        for row in accepted:
            grouped.setdefault(str(row[0].canonical_entity_id), []).append(row)
        nodes: list[EntityCandidate] = []
        for canonical_id, rows in sorted(grouped.items()):
            decisions = [row[0] for row in rows]
            members = [member for row in rows for member in row[1]]
            votes = [vote for row in rows for vote in row[2]]
            evidence_by_id = {
                evidence.evidence_id: evidence
                for member in members
                for evidence in member.evidence
            }
            mention_ids = sorted(
                {
                    str(mention_id)
                    for member in members
                    for mention_id in member.properties.get("mention_ids", [])
                }
            )
            aliases = sorted(
                {
                    member.name
                    for member in members
                }
                | {
                    str(alias)
                    for member in members
                    for alias in member.properties.get("aliases", [])
                }
            )
            first = decisions[0]
            semantic_type = self._semantic_type(members[0])
            nodes.append(
                EntityCandidate.build(
                    entity_type="CanonicalEntity",
                    name=str(first.canonical_name),
                    properties={
                        "canonical_entity_id": canonical_id,
                        "domain_canonical_entity_id": canonical_id,
                        "semantic_entity_type": semantic_type,
                        "identity_layer": "canonical",
                        "identity_status": "externally_linked",
                        "external_system": first.external_system,
                        "external_id": first.external_id,
                        "mention_ids": mention_ids,
                        "aliases": aliases,
                        "linked_observed_candidate_ids": sorted(
                            member.candidate_id for member in members
                        ),
                        "provider_ids": sorted(
                            {
                                str(vote["provider_id"])
                                for vote in votes
                            }
                        ),
                        "governed_schema_id": self.policy.schema_id,
                        "promotion_policy": self.technique,
                    },
                    evidence=[
                        evidence_by_id[key] for key in sorted(evidence_by_id)
                    ],
                    confidence=min(
                        float(decision.confidence or 0.0)
                        for decision in decisions
                    ),
                    extractor_name=self.name,
                    technique=self.technique,
                    schema_version=members[0].schema_version,
                    provenance_metadata={
                        "provider": self.name,
                        "configuration_id": self.policy.schema_id,
                        "model_version": "v1",
                    },
                )
            )
        return nodes

    @staticmethod
    def _updated_entity(
        entity: EntityCandidate,
        decision: ExternalIdentityDecision | None,
    ) -> EntityCandidate:
        if decision is None or entity.entity_type == "CanonicalEntity":
            return entity
        properties = {
            **entity.properties,
            "external_identity_status": decision.status,
            "external_identity_nil": decision.status == "nil",
            "external_identity_review_required": decision.review_required,
            "external_identity_decision_reason": decision.reason,
        }
        if decision.status == "accepted":
            properties.update(
                {
                    "identity_status": "externally_linked",
                    "linked_canonical_entity_id": decision.canonical_entity_id,
                    "external_system": decision.external_system,
                    "external_id": decision.external_id,
                    "external_identity_confidence": decision.confidence,
                    "external_identity_provider_ids": list(decision.provider_ids),
                }
            )
        elif decision.status == "conflict":
            properties["external_identity_conflicts"] = list(
                decision.conflict_identity_keys
            )
            properties["review_required"] = True
        return entity.model_copy(update={"properties": properties})

    @staticmethod
    def _updated_mention(
        entity: EntityCandidate,
        accepted_by_mention: dict[str, ExternalIdentityDecision],
    ) -> EntityCandidate:
        if entity.entity_type != "Mention":
            return entity
        mention_id = str(entity.properties.get("domain_mention_id", ""))
        decision = accepted_by_mention.get(mention_id)
        if decision is None:
            return entity
        return entity.model_copy(
            update={
                "properties": {
                    **entity.properties,
                    "linked_canonical_entity_id": decision.canonical_entity_id,
                    "resolution_status": "externally_linked",
                    "resolution_confidence": decision.confidence,
                    "external_system": decision.external_system,
                    "external_id": decision.external_id,
                }
            }
        )

    @staticmethod
    def _rewrite_relation(
        relation: RelationCandidate,
        reference_to_canonical: dict[str, str],
    ) -> RelationCandidate:
        if relation.relation_type in _IDENTITY_LINK_RELATIONS:
            return relation
        source = reference_to_canonical.get(relation.source, relation.source)
        target = reference_to_canonical.get(relation.target, relation.target)
        if source == relation.source and target == relation.target:
            return relation
        properties = dict(relation.properties)
        if source != relation.source:
            properties.setdefault("original_source", relation.source)
        if target != relation.target:
            properties.setdefault("original_target", relation.target)
        return relation.model_copy(
            update={"source": source, "target": target, "properties": properties}
        )

    @staticmethod
    def _rewrite_fact(
        fact: GraphFact,
        reference_to_canonical: dict[str, str],
    ) -> GraphFact:
        subject = reference_to_canonical.get(fact.subject, fact.subject)
        object_is_literal = str(
            fact.properties.get("rdf_object_type", "iri")
        ) == "literal"
        object_value = (
            fact.object
            if object_is_literal
            else reference_to_canonical.get(fact.object, fact.object)
        )
        if subject == fact.subject and object_value == fact.object:
            return fact
        properties = dict(fact.properties)
        if subject != fact.subject:
            properties.setdefault("original_subject", fact.subject)
        if object_value != fact.object:
            properties.setdefault("original_object", fact.object)
        return fact.model_copy(
            update={
                "subject": subject,
                "object": object_value,
                "properties": properties,
            }
        )

    def _identity_relation(
        self,
        *,
        source: str,
        target: str,
        relation_type: str,
        evidence: list,
        confidence: float,
        properties: dict[str, object],
    ) -> RelationCandidate:
        return RelationCandidate.build(
            source=source,
            relation_type=relation_type,
            target=target,
            properties=properties,
            evidence=evidence,
            confidence=confidence,
            extractor_name=self.name,
            technique=self.technique,
            schema_version="v1",
            provenance_metadata={
                "provider": self.name,
                "configuration_id": self.policy.schema_id,
                "model_version": "v1",
            },
        )


class MandatoryScopedResolutionEnsemble:
    """Run every resolver only on governed semantic identity resources.

    Extraction produces technical resources (documents, mentions, reified
    claims/events) alongside observed/canonical business identities. Sending
    technical nodes to an entity linker is both semantically wrong and an
    external-service amplification bug. This adapter preserves them byte-for-
    byte while every mandatory resolver processes the eligible identity slice.
    """

    name = "mandatory_scoped_resolution_ensemble"
    technique = "governed_identity_resolution_and_external_linking"

    def __init__(
        self,
        resolvers: Sequence[Resolver],
        *,
        governed_entity_types: Sequence[str],
        schema_id: str = "governed-unstructured-core",
        promotion_policy: ExternalIdentityPromotionPolicy | None = None,
    ) -> None:
        if not resolvers:
            raise ValueError("mandatory scoped resolution requires resolvers")
        self.resolvers = tuple(resolvers)
        self.resolver_provider_ids = tuple(
            str(getattr(resolver, "name", resolver.__class__.__name__))
            for resolver in self.resolvers
        )
        if len(self.resolver_provider_ids) != len(set(self.resolver_provider_ids)):
            raise ValueError("mandatory scoped resolver IDs must be unique")
        self.governed_entity_types = frozenset(governed_entity_types)
        if not self.governed_entity_types:
            raise ValueError("governed entity types must not be empty")
        self.external_identity_promoter = GovernedExternalIdentityPromoter(
            governed_entity_types=governed_entity_types,
            policy=promotion_policy
            or ExternalIdentityPromotionPolicy(schema_id=schema_id),
        )
        self.provider_ids = (
            *self.resolver_provider_ids,
            self.external_identity_promoter.name,
        )
        self.last_participation: tuple[dict[str, object], ...] = ()

    def eligible(self, entity: EntityCandidate) -> bool:
        """Return whether an entity is a governed linkable identity resource."""

        if entity.entity_type in _TECHNICAL_TYPES:
            return False
        semantic_type = (
            str(entity.properties.get("semantic_entity_type", ""))
            if entity.entity_type == "CanonicalEntity"
            else entity.entity_type
        )
        if semantic_type not in self.governed_entity_types:
            return False
        return entity.properties.get("identity_layer") in {"observed", "canonical"}

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        self.last_participation = ()
        eligible = self._eligible_entities(batch)
        eligible_ids = {entity.candidate_id for entity in eligible}
        technical_before = {
            entity.candidate_id: entity
            for entity in batch.entities
            if entity.candidate_id not in eligible_ids
        }
        working = ConstructionBatch(
            source_records=list(batch.source_records),
            entities=eligible,
            metadata=dict(batch.metadata),
        )
        participation: list[dict[str, object]] = []
        for resolver, provider_id in zip(
            self.resolvers,
            self.resolver_provider_ids,
            strict=True,
        ):
            working = resolver.resolve(working)
            participation.append(
                {
                    "provider_id": provider_id,
                    "executed": True,
                    "attempted_entity_count": len(batch.entities),
                    "eligible_entity_count": len(eligible),
                    "excluded_technical_entity_count": len(batch.entities) - len(eligible),
                    "output_entity_count": len(working.entities),
                }
            )

        resolved_by_candidate = {
            entity.candidate_id: entity for entity in working.entities
        }
        entities = [
            resolved_by_candidate.get(entity.candidate_id, entity)
            for entity in batch.entities
        ]
        for candidate_id, original in technical_before.items():
            current = next(
                entity for entity in entities if entity.candidate_id == candidate_id
            )
            if current != original:
                raise RuntimeError(
                    f"technical entity {candidate_id} changed during scoped resolution"
                )
        metadata = {
            **batch.metadata,
            **working.metadata,
            "mandatory_resolution_participation": participation,
            "mandatory_resolution_provider_ids": list(self.provider_ids),
            "resolution_eligibility": {
                "attempted_entity_count": len(batch.entities),
                "eligible_entity_count": len(eligible),
                "excluded_entity_count": len(batch.entities) - len(eligible),
                "eligible_candidate_ids": sorted(eligible_ids),
            },
        }
        merged = reconcile_entity_references(
            batch.model_copy(update={"entities": entities, "metadata": metadata})
        )
        promoted = self.external_identity_promoter.promote(merged)
        decisions = self.external_identity_promoter.last_decisions
        participation.append(
            {
                "provider_id": self.external_identity_promoter.name,
                "executed": True,
                "attempted_entity_count": len(batch.entities),
                "eligible_entity_count": len(eligible),
                "excluded_technical_entity_count": len(batch.entities) - len(eligible),
                "output_entity_count": len(promoted.entities),
                "accepted_identity_count": sum(
                    decision.status == "accepted" for decision in decisions
                ),
                "conflict_identity_count": sum(
                    decision.status == "conflict" for decision in decisions
                ),
                "nil_identity_count": sum(
                    decision.status == "nil" for decision in decisions
                ),
            }
        )
        self.last_participation = tuple(participation)
        return promoted.model_copy(
            update={
                "metadata": {
                    **promoted.metadata,
                    "mandatory_resolution_participation": list(
                        self.last_participation
                    ),
                    "mandatory_resolution_provider_ids": list(self.provider_ids),
                }
            }
        )

    def _eligible_entities(
        self,
        batch: ConstructionBatch,
    ) -> list[EntityCandidate]:
        candidates = [entity for entity in batch.entities if self.eligible(entity)]
        canonical_mentions = {
            str(mention_id)
            for entity in candidates
            if entity.properties.get("identity_layer") == "canonical"
            for mention_id in entity.properties.get("mention_ids", [])
        }
        selected: list[EntityCandidate] = []
        for entity in candidates:
            if entity.properties.get("identity_layer") != "observed":
                selected.append(entity)
                continue
            observed_mentions = {
                str(mention_id)
                for mention_id in entity.properties.get("mention_ids", [])
            }
            if observed_mentions and observed_mentions & canonical_mentions:
                continue
            selected.append(entity)
        return selected


def _normalized_external_identity(
    provider: str,
    identifier: str,
) -> tuple[str, str] | None:
    raw = identifier.strip()
    if not raw:
        return None
    wikidata = _WIKIDATA_ID_RE.search(raw)
    if provider == "wikidata" or (provider == "openrefine" and wikidata):
        qid = wikidata.group(1).upper() if wikidata else raw.upper()
        return ("wikidata", qid)
    if provider == "dbpedia" or "dbpedia.org/resource/" in raw:
        normalized = re.sub(
            r"^http://dbpedia\.org/resource/",
            "https://dbpedia.org/resource/",
            raw,
        )
        return ("dbpedia", normalized)
    return (provider, raw)


def _float_value(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"external identity confidence is not numeric: {value!r}")


__all__ = [
    "ExternalIdentityDecision",
    "ExternalIdentityPromotionPolicy",
    "GovernedExternalIdentityPromoter",
    "MandatoryScopedResolutionEnsemble",
]
