"""Shared entity resolution helpers and merge audit objects."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from kg_construction.domain.models import ConstructionBatch, EntityCandidate, stable_id

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

    resolved_relations = []
    for relation in batch.relations:
        source = candidate_to_canonical.get(relation.source, relation.source)
        target = candidate_to_canonical.get(relation.target, relation.target)
        properties = {
            **relation.properties,
            "original_source": relation.source,
            "original_target": relation.target,
        }
        resolved_relations.append(
            relation.model_copy(update={"source": source, "target": target, "properties": properties})
        )

    audit = merge_audit(batch.entities, decisions, cluster_members, resolver_name, resolver_technique)
    previous_audit = list(batch.metadata.get("entity_resolution_audit", []))
    metadata = {
        **batch.metadata,
        "entity_resolution_audit": [*previous_audit, *audit],
        "resolver": resolver_name,
    }
    return batch.model_copy(
        update={"entities": resolved_entities, "relations": resolved_relations, "metadata": metadata}
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
    "resolution_key",
]
