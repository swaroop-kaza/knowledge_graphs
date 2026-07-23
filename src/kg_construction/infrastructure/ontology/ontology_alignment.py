"""Align extracted entity and relation types to an ontology.

Maps the types produced by extraction onto the classes and predicates of an
``OntologySchema``, reports how much conforms, and surfaces off-ontology terms so
unexpected (potentially hallucinated) types are visible.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.modeling import OntologySchema
from kg_construction.domain.models import ConstructionBatch
from kg_construction.infrastructure.resolution.common import resolution_key

logger = logging.getLogger(__name__)


class AlignmentReport(BaseModel):
    """How well an extracted batch conforms to an ontology."""

    model_config = ConfigDict(extra="forbid")

    entity_type_alignment: dict[str, str | None] = Field(default_factory=dict)
    relation_type_alignment: dict[str, str | None] = Field(default_factory=dict)
    in_ontology_entity_rate: float = 1.0
    in_ontology_relation_rate: float = 1.0
    off_ontology_entity_types: list[str] = Field(default_factory=list)
    off_ontology_relation_types: list[str] = Field(default_factory=list)


class OntologyAligner:
    """Align extracted types to ontology classes and predicates."""

    name = "ontology_aligner"

    def __init__(self, ontology: OntologySchema) -> None:
        self.ontology = ontology
        self._class_by_key = {resolution_key(item.name): item.name for item in ontology.classes}
        self._predicate_by_key = {resolution_key(item.name): item.name for item in ontology.predicates}

    def aligned_class(self, entity_type: str) -> str | None:
        return self._class_by_key.get(resolution_key(entity_type))

    def aligned_predicate(self, relation_type: str) -> str | None:
        return self._predicate_by_key.get(resolution_key(relation_type))

    def align(self, batch: ConstructionBatch) -> AlignmentReport:
        entity_types = sorted({entity.entity_type for entity in batch.entities})
        relation_types = sorted({relation.relation_type for relation in batch.relations})
        entity_alignment = {name: self.aligned_class(name) for name in entity_types}
        relation_alignment = {name: self.aligned_predicate(name) for name in relation_types}

        aligned_entities = sum(1 for entity in batch.entities if self.aligned_class(entity.entity_type))
        aligned_relations = sum(
            1 for relation in batch.relations if self.aligned_predicate(relation.relation_type)
        )
        report = AlignmentReport(
            entity_type_alignment=entity_alignment,
            relation_type_alignment=relation_alignment,
            in_ontology_entity_rate=(aligned_entities / len(batch.entities)) if batch.entities else 1.0,
            in_ontology_relation_rate=(aligned_relations / len(batch.relations)) if batch.relations else 1.0,
            off_ontology_entity_types=[name for name, target in entity_alignment.items() if target is None],
            off_ontology_relation_types=[name for name, target in relation_alignment.items() if target is None],
        )
        logger.info(
            "Ontology alignment: entities %.2f, relations %.2f in ontology (off-ontology: %s)",
            report.in_ontology_entity_rate,
            report.in_ontology_relation_rate,
            report.off_ontology_entity_types + report.off_ontology_relation_types,
        )
        return report


__all__ = ["AlignmentReport", "OntologyAligner"]
