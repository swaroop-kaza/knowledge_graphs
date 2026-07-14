"""Coverage and completeness metrics."""

from __future__ import annotations

from typing import Any

from kg_construction.domain.models import ConstructionBatch


class CoverageMetrics:
    """Compute graph coverage estimates."""

    name = "coverage_metrics"

    def evaluate(self, batch: ConstructionBatch, *, estimated_entities: int = 0, estimated_relations: int = 0) -> dict[str, float]:
        unique_entities = {
            (entity.entity_type.casefold(), entity.name.casefold().strip())
            for entity in batch.entities
        }
        unique_relations = {
            (relation.source, relation.relation_type, relation.target)
            for relation in batch.relations
        }
        source_ids = {source.source_id for source in batch.source_records}
        objects: list[Any] = [*batch.entities, *batch.relations, *batch.facts]
        contributing_sources = {
            evidence.source_id
            for item in objects
            for evidence in item.evidence
        }
        expected_entity_count = max(estimated_entities, len(unique_entities), 1)
        expected_relation_count = max(estimated_relations, len(unique_relations), 1)
        return {
            "entity_coverage": min(1.0, len(unique_entities) / expected_entity_count),
            "relation_coverage": min(1.0, len(unique_relations) / expected_relation_count),
            "source_coverage": 1.0 if not source_ids else len(contributing_sources) / len(source_ids),
            "attribute_coverage": self._attribute_coverage(batch),
        }

    @staticmethod
    def _attribute_coverage(batch: ConstructionBatch) -> float:
        if not batch.entities:
            return 1.0
        filled = sum(
            1
            for entity in batch.entities
            for value in entity.properties.values()
            if value not in (None, "")
        )
        possible = sum(max(1, len(entity.properties)) for entity in batch.entities)
        return filled / possible if possible else 1.0


__all__ = ["CoverageMetrics"]
