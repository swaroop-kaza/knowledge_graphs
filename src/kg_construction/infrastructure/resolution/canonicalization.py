"""Canonicalise and link entity mentions extracted from unstructured text.

Runs as a resolver so it reuses the shared ``apply_merge_decisions`` machinery
(canonical-id assignment, relation/fact endpoint rewrite, merge audit) -- the same
mechanism that collapses duplicates in structured graphs.

- **Coreference / NIL clustering:** mentions of the same type are grouped by a
  normalised key, and (optionally) a partial name is merged into its fuller form
  (``"Turing"`` into ``"Alan Turing"``).
- **Linking with NIL handling:** an optional injectable ``EntityLinker`` maps a
  cluster to a knowledge-base id. Without one, the cluster is a NIL cluster with a
  stable local id. Every entity records ``link_status`` (``linked`` or ``nil``) so
  the unlinked rate can be measured.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from kg_construction.domain.models import ConstructionBatch, EntityCandidate
from kg_construction.infrastructure.resolution.common import (
    MergeDecision,
    apply_merge_decisions,
    resolution_key,
)

logger = logging.getLogger(__name__)


class LinkResult(BaseModel):
    """A knowledge-base link for an entity mention cluster."""

    model_config = ConfigDict(extra="forbid")

    canonical_id: str
    canonical_name: str
    knowledge_base: str
    confidence: float = 1.0


class EntityLinker(Protocol):
    """Injectable linker; returns a link or ``None`` (NIL)."""

    def link(self, name: str, entity_type: str) -> LinkResult | None:
        """Return a knowledge-base link for a mention, or ``None`` for NIL."""


class UnstructuredCanonicalizer:
    """Default canonicalisation/linking resolver for extracted entities."""

    name = "unstructured_canonicalizer"
    technique = "entity_linking_canonicalization"

    def __init__(
        self,
        linker: EntityLinker | None = None,
        *,
        coreference: bool = True,
        nil_confidence: float = 0.85,
    ) -> None:
        self.linker = linker
        self.coreference = coreference
        self.nil_confidence = nil_confidence

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        if not batch.entities:
            return batch

        groups = self._groups(batch.entities)
        decisions: dict[str, MergeDecision] = {}
        link_status: dict[str, str] = {}
        knowledge_base: dict[str, str] = {}

        for (entity_type, canonical_key), members in groups.items():
            canonical_name = self._canonical_name(members)
            link = self.linker.link(canonical_name, entity_type) if self.linker else None
            if link is not None:
                decision_key = f"kb:{link.knowledge_base}:{link.canonical_id}"
                status, confidence, name = "linked", link.confidence, link.canonical_name or canonical_name
            else:
                decision_key, status, confidence, name = canonical_key, "nil", self.nil_confidence, canonical_name
            for entity in members:
                decisions[entity.candidate_id] = MergeDecision(
                    candidate_id=entity.candidate_id,
                    canonical_key=decision_key,
                    canonical_name=name,
                    confidence=confidence,
                    review_required=False,
                    reason=f"unstructured_{status}",
                )
                link_status[entity.candidate_id] = status
                if link is not None:
                    knowledge_base[entity.candidate_id] = link.knowledge_base

        resolved = apply_merge_decisions(
            batch, decisions, resolver_name=self.name, resolver_technique=self.technique
        )
        entities = [
            self._with_link_metadata(entity, link_status, knowledge_base)
            for entity in resolved.entities
        ]
        linked = sum(1 for status in link_status.values() if status == "linked")
        logger.info(
            "Canonicalised %d mentions into %d clusters (%d linked, %d NIL)",
            len(link_status),
            len(groups),
            linked,
            len(link_status) - linked,
        )
        return resolved.model_copy(update={"entities": entities})

    @staticmethod
    def _with_link_metadata(
        entity: EntityCandidate,
        link_status: dict[str, str],
        knowledge_base: dict[str, str],
    ) -> EntityCandidate:
        status = link_status.get(entity.candidate_id)
        if status is None:
            return entity
        properties = {**entity.properties, "link_status": status}
        if entity.candidate_id in knowledge_base:
            properties["knowledge_base"] = knowledge_base[entity.candidate_id]
        return entity.model_copy(update={"properties": properties})

    def _groups(
        self, entities: list[EntityCandidate]
    ) -> dict[tuple[str, str], list[EntityCandidate]]:
        initial: dict[tuple[str, str], list[EntityCandidate]] = {}
        for entity in entities:
            key = resolution_key(entity.name) or entity.candidate_id
            initial.setdefault((entity.entity_type, key), []).append(entity)
        if not self.coreference:
            return initial

        keys_by_type: dict[str, list[str]] = {}
        for entity_type, key in initial:
            keys_by_type.setdefault(entity_type, []).append(key)

        merged: dict[tuple[str, str], list[EntityCandidate]] = {}
        for (entity_type, key), members in initial.items():
            canonical = self._coreference_key(key, keys_by_type[entity_type])
            merged.setdefault((entity_type, canonical), []).extend(members)
        return merged

    @staticmethod
    def _coreference_key(key: str, keys: list[str]) -> str:
        # Fold a partial mention into its fuller form (e.g. "turing" into "alan
        # turing"), preferring the longest superset; never the reverse.
        tokens = set(key.split())
        ordered = sorted(set(keys), key=lambda item: (-len(item.split()), -len(item), item))
        for candidate in ordered:
            if candidate != key and tokens < set(candidate.split()):
                return candidate
        return key

    @staticmethod
    def _canonical_name(members: list[EntityCandidate]) -> str:
        return sorted(members, key=lambda entity: (-len(entity.name), entity.name.casefold()))[0].name


__all__ = ["EntityLinker", "LinkResult", "UnstructuredCanonicalizer"]
