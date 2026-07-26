"""Shared entity resolution helpers and merge audit objects."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    GraphFact,
    RelationCandidate,
    stable_id,
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_SPACES_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class MergeDecision:
    """One resolver decision linking a candidate to a canonical cluster."""

    candidate_id: str
    canonical_key: str
    canonical_name: str
    confidence: float
    review_required: bool
    reason: str


def resolution_key(value: str) -> str:
    """Return a stable normalized key for ER comparisons."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _NON_ALNUM_RE.sub(" ", normalized)
    normalized = _SPACES_RE.sub(" ", normalized)
    return normalized.strip()


def compact_resolution_key(value: str) -> str:
    """Return a punctuation/space-insensitive key for known aliases and acronyms."""

    return resolution_key(value).replace(" ", "")


def canonical_entity_id(entity_type: str, canonical_key: str) -> str:
    """Build a deterministic canonical entity id."""

    return stable_id("canonical_entity", entity_type, canonical_key)


def current_canonical_name(entity: EntityCandidate) -> str:
    """Return the best available canonical display name for a candidate."""

    value = entity.properties.get("canonical_name")
    return str(value) if value else entity.name


def apply_merge_decisions(
    batch: ConstructionBatch,
    decisions: dict[str, MergeDecision],
    *,
    resolver_name: str,
    resolver_technique: str,
) -> ConstructionBatch:
    """Apply canonical IDs, relation rewrites, and merge audit metadata."""

    candidate_to_canonical = {
        candidate_id: canonical_entity_id(entity.entity_type, decisions[candidate_id].canonical_key)
        for candidate_id, entity in ((entity.candidate_id, entity) for entity in batch.entities)
        if candidate_id in decisions
    }
    cluster_members: dict[str, list[str]] = {}
    for candidate_id, canonical_id in candidate_to_canonical.items():
        cluster_members.setdefault(canonical_id, []).append(candidate_id)

    resolved_entities: list[EntityCandidate] = []
    for entity in batch.entities:
        decision = decisions.get(entity.candidate_id)
        if decision is None:
            resolved_entities.append(entity)
            continue
        canonical_id = candidate_to_canonical[entity.candidate_id]
        merged_from = sorted(cluster_members.get(canonical_id, [entity.candidate_id]))
        confidence = round(decision.confidence, 6)
        review_required = decision.review_required
        properties = {
            **entity.properties,
            "canonical_entity_id": canonical_id,
            "canonical_id": canonical_id,
            "canonical_name": decision.canonical_name,
            "merged_from": merged_from,
            "resolver": resolver_name,
            "resolution_method": resolver_technique,
            "resolution_confidence": confidence,
            "review_required": review_required,
            "merge_group_size": len(merged_from),
            "merge_reason": decision.reason,
        }
        resolved_entities.append(entity.model_copy(update={"properties": properties}))

    audit = merge_audit(batch.entities, decisions, cluster_members, resolver_name, resolver_technique)
    previous_audit = list(batch.metadata.get("entity_resolution_audit", []))
    metadata = {
        **batch.metadata,
        "entity_resolution_audit": [*previous_audit, *audit],
        "entity_resolution_aliases": _updated_resolution_aliases(batch, resolved_entities),
        "resolver": resolver_name,
    }
    return reconcile_entity_references(
        batch.model_copy(update={"entities": resolved_entities, "metadata": metadata})
    )


def reconcile_entity_references(batch: ConstructionBatch) -> ConstructionBatch:
    """Rewrite entity references to the current canonical IDs.

    Resolution is deliberately non-destructive: entity candidates retain their original IDs while
    ``canonical_entity_id`` identifies the node that will be materialized.  This helper uses the
    original endpoint saved on relations/facts plus a flattened alias history, so applying a second
    resolver cannot leave an endpoint pointing at a canonical ID created by an earlier resolver.
    """

    aliases = _resolution_aliases(batch)
    if not aliases:
        return batch

    resolved_relations = [_reconcile_relation(relation, aliases) for relation in batch.relations]
    resolved_facts = [_reconcile_fact(fact, aliases) for fact in batch.facts]
    metadata = {
        **batch.metadata,
        "entity_resolution_aliases": dict(sorted(aliases.items())),
    }
    return batch.model_copy(
        update={
            "relations": resolved_relations,
            "facts": resolved_facts,
            "metadata": metadata,
        }
    )


def _updated_resolution_aliases(
    previous_batch: ConstructionBatch,
    resolved_entities: list[EntityCandidate],
) -> dict[str, str]:
    """Return a flattened candidate/historical-canonical to current-ID map."""

    final_by_candidate = {
        entity.candidate_id: str(entity.properties.get("canonical_entity_id", entity.candidate_id))
        for entity in resolved_entities
    }
    old_targets: dict[str, set[str]] = {}
    for entity in previous_batch.entities:
        old_id = entity.properties.get("canonical_entity_id")
        final_id = final_by_candidate.get(entity.candidate_id, entity.candidate_id)
        if old_id:
            old_targets.setdefault(str(old_id), set()).add(final_id)
    unambiguous_old_to_final = {
        old_id: next(iter(final_ids))
        for old_id, final_ids in old_targets.items()
        if len(final_ids) == 1
    }

    aliases: dict[str, str] = {}
    previous_aliases = previous_batch.metadata.get("entity_resolution_aliases", {})
    if isinstance(previous_aliases, dict):
        for alias, target in previous_aliases.items():
            alias_value = str(alias)
            target_value = str(target)
            aliases[alias_value] = unambiguous_old_to_final.get(target_value, target_value)
    aliases.update(unambiguous_old_to_final)
    aliases.update(final_by_candidate)

    current_ids = set(final_by_candidate.values())
    for current_id in current_ids:
        aliases[current_id] = current_id
    return _flatten_aliases(aliases, current_ids=current_ids)


def _resolution_aliases(batch: ConstructionBatch) -> dict[str, str]:
    canonicalized_entities = [
        entity
        for entity in batch.entities
        if entity.properties.get("canonical_entity_id")
        or (
            entity.properties.get("identity_layer")
            in {"observed", "canonical"}
            and entity.properties.get("linked_canonical_entity_id")
        )
    ]
    stored_aliases = batch.metadata.get("entity_resolution_aliases", {})
    if not canonicalized_entities and not isinstance(stored_aliases, dict):
        return {}
    if not canonicalized_entities and not stored_aliases:
        return {}

    aliases = (
        {str(alias): str(target) for alias, target in stored_aliases.items()}
        if isinstance(stored_aliases, dict)
        else {}
    )
    current_ids: set[str] = set()
    for entity in batch.entities:
        local_canonical_id = str(
            entity.properties.get(
                "canonical_entity_id",
                entity.candidate_id,
            )
        )
        linked_canonical_id = (
            str(entity.properties["linked_canonical_entity_id"])
            if (
                entity.properties.get("identity_layer")
                in {"observed", "canonical"}
                and entity.properties.get("linked_canonical_entity_id")
            )
            else None
        )
        endpoint_id = linked_canonical_id or local_canonical_id
        if (
            endpoint_id != entity.candidate_id
            or entity.properties.get("canonical_entity_id")
        ):
            aliases[entity.candidate_id] = endpoint_id
            aliases[local_canonical_id] = endpoint_id
            aliases[endpoint_id] = endpoint_id
            current_ids.add(endpoint_id)
    return _flatten_aliases(aliases, current_ids=current_ids)


def _flatten_aliases(aliases: dict[str, str], *, current_ids: set[str]) -> dict[str, str]:
    flattened: dict[str, str] = {}
    for alias in aliases:
        current = alias
        seen: set[str] = set()
        while current in aliases and current not in seen:
            seen.add(current)
            target = aliases[current]
            if target == current:
                break
            current = target
        flattened[alias] = current
    for current_id in current_ids:
        flattened[current_id] = current_id
    return flattened


def _resolve_endpoint(value: str, aliases: dict[str, str]) -> str:
    return aliases.get(value, value)


def _reconcile_relation(
    relation: RelationCandidate,
    aliases: dict[str, str],
) -> RelationCandidate:
    prior_source = str(relation.properties.get("original_source", relation.source))
    prior_target = str(relation.properties.get("original_target", relation.target))
    source = _resolve_endpoint(prior_source, aliases)
    target = _resolve_endpoint(prior_target, aliases)
    if source == relation.source and target == relation.target:
        return relation
    properties = {
        **relation.properties,
        "original_source": prior_source,
        "original_target": prior_target,
    }
    return relation.model_copy(update={"source": source, "target": target, "properties": properties})


def _reconcile_fact(fact: GraphFact, aliases: dict[str, str]) -> GraphFact:
    prior_subject = str(fact.properties.get("original_subject", fact.subject))
    prior_object = str(fact.properties.get("original_object", fact.object))
    subject = _resolve_endpoint(prior_subject, aliases)
    object_is_literal = str(fact.properties.get("rdf_object_type", "iri")) == "literal"
    object_value = prior_object if object_is_literal else _resolve_endpoint(prior_object, aliases)

    properties = dict(fact.properties)
    if "rdf_subject_iri" in properties:
        properties["rdf_subject_iri"] = _resolve_endpoint(str(properties["rdf_subject_iri"]), aliases)
    if "rdf_object" in properties and not object_is_literal:
        properties["rdf_object"] = _resolve_endpoint(str(properties["rdf_object"]), aliases)
    if subject == fact.subject and object_value == fact.object and properties == fact.properties:
        return fact
    properties["original_subject"] = prior_subject
    properties["original_object"] = prior_object
    return fact.model_copy(
        update={"subject": subject, "object": object_value, "properties": properties}
    )


def merge_audit(
    entities: list[EntityCandidate],
    decisions: dict[str, MergeDecision],
    cluster_members: dict[str, list[str]],
    resolver_name: str,
    resolver_technique: str,
) -> list[dict[str, object]]:
    """Build merge audit rows for canonical clusters."""

    entity_by_id = {entity.candidate_id: entity for entity in entities}
    audit: list[dict[str, object]] = []
    for canonical_id, members in sorted(cluster_members.items()):
        member_decisions = [decisions[member] for member in members if member in decisions]
        if not member_decisions:
            continue
        confidences = [decision.confidence for decision in member_decisions]
        canonical_names = [decision.canonical_name for decision in member_decisions]
        audit.append(
            {
                "canonical_id": canonical_id,
                "canonical_entity_id": canonical_id,
                "canonical_name": canonical_names[0],
                "merged_from": sorted(members),
                "merged_names": sorted(entity_by_id[member].name for member in members if member in entity_by_id),
                "resolver": resolver_name,
                "method": resolver_technique,
                "confidence": round(min(confidences), 6),
                "review_required": any(decision.review_required for decision in member_decisions),
                "reasons": sorted({decision.reason for decision in member_decisions}),
                "reversible": True,
            }
        )
    return audit


def decisions_from_groups(
    entities: list[EntityCandidate],
    *,
    key_fn: Callable[[EntityCandidate], str],
    resolver_name: str,
    confidence: float,
    review_required: bool = False,
    reason: str,
) -> dict[str, MergeDecision]:
    """Build merge decisions by grouping entities with a key function."""

    _ = resolver_name
    groups: dict[tuple[str, str], list[EntityCandidate]] = {}
    for entity in entities:
        key = key_fn(entity)
        if not key:
            continue
        groups.setdefault((entity.entity_type, key), []).append(entity)

    decisions: dict[str, MergeDecision] = {}
    for (_entity_type, canonical_key), members in groups.items():
        canonical_name = sorted(members, key=lambda entity: (-len(entity.name), entity.name.casefold()))[0].name
        for entity in members:
            decisions[entity.candidate_id] = MergeDecision(
                candidate_id=entity.candidate_id,
                canonical_key=canonical_key,
                canonical_name=canonical_name,
                confidence=confidence,
                review_required=review_required,
                reason=reason,
            )
    return decisions


__all__ = [
    "MergeDecision",
    "apply_merge_decisions",
    "canonical_entity_id",
    "compact_resolution_key",
    "current_canonical_name",
    "decisions_from_groups",
    "reconcile_entity_references",
    "resolution_key",
]
