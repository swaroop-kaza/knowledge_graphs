"""Turn structured rows into a well-formed, evidence-backed knowledge graph.

Driven by a resolved modelling contract, each row becomes a typed node; a
categorical dimension materialised as a node collapses into a single shared,
deduplicated concept node linked by a typed edge (so a value is stored once,
never repeated as a literal on every row); a foreign key becomes a typed edge to
the referenced entity — and a foreign key whose target is not available is
skipped, never fabricated as a placeholder; measures, temporals, and free text
stay as literal properties.

Business-entity identity (the merge key) is kept separate from source-row
identity (provenance of the physical row). Every occurrence of the same business
key shares one ``canonical_entity_id`` under its declared identity scope, so the
property-graph builder merges them into a single node automatically — the graph
equivalent of merging on a business key.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from kg_construction.application.resolved_plan import (
    IdentityScope,
    Materialization,
    OnMissingTarget,
    ResolvedAttribute,
    ResolvedModelingPlan,
)
from kg_construction.domain.evidence import cell_evidence, row_evidence
from kg_construction.domain.modeling import ColumnRole, ReferenceNode
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    RelationCandidate,
    SourceRecord,
    stable_id,
)
from kg_construction.domain.normalization import normalize_value

JsonDict = dict[str, Any]

logger = logging.getLogger(__name__)

_NAME_COLUMN_RE = re.compile(r"(name|title|label)", re.IGNORECASE)


class SemanticTableMapper:
    """Map rows to typed nodes, shared concept nodes, and typed relationships."""

    name = "semantic_table_mapper"
    technique = "policy_driven_semantic_mapping"

    def __init__(self, plan: ResolvedModelingPlan) -> None:
        self.plan = plan
        self._attributes = {attribute.column: attribute for attribute in plan.attributes}
        self._relationship_by_column = {
            relationship.source_columns[0]: relationship
            for relationship in plan.relationships
            if relationship.source_columns
        }
        self._fk_by_column = {
            foreign_key.columns[0]: foreign_key
            for foreign_key in plan.foreign_keys
            if foreign_key.columns
        }
        record_class = plan.clazz(plan.record_class)
        self._identity_keys = record_class.identity_keys
        self._identity_scope = record_class.identity_scope

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        records = source.records
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        skipped_missing_targets = 0

        for row_index, row in enumerate(records):
            business_value = self._business_value(row, row_index)
            business_id = self._business_id(source, business_value)
            source_row_id = stable_id("source_row", source.source_id, str(row_index))
            row_ev = row_evidence(source, row_index, row)

            properties: JsonDict = {
                "canonical_entity_id": business_id,
                "business_key": business_value,
                "identity_key": f"{self._identity_scope.value}:{self.plan.record_class}:{business_value}",
                "identity_scope": self._identity_scope.value,
                "source_row_id": source_row_id,
            }
            pending_concepts: list[tuple[ResolvedAttribute, object]] = []
            pending_relations: list[tuple[ResolvedAttribute, str, object]] = []

            for attribute in self.plan.attributes:
                value = row.get(attribute.column)
                if value in (None, ""):
                    continue
                if attribute.materialization in (Materialization.PROPERTY, Materialization.IDENTITY):
                    properties[attribute.column] = normalize_value(value, attribute.datatype)
                elif attribute.materialization is Materialization.NODE:
                    pending_concepts.append((attribute, value))
                    pending_relations.append((attribute, self._concept_id(attribute, value), value))
                elif attribute.materialization is Materialization.EDGE:
                    target_id = self._edge_target(attribute, value, source)
                    if target_id is None:
                        # Unresolvable foreign key: retain the reference value so no
                        # data is lost, but never fabricate a placeholder entity.
                        properties[attribute.column] = normalize_value(value, attribute.datatype)
                        skipped_missing_targets += 1
                    else:
                        pending_relations.append((attribute, target_id, value))

            entities.append(
                EntityCandidate.build(
                    entity_type=self.plan.record_class,
                    name=self._record_name(row, business_value),
                    evidence=[row_ev],
                    extractor_name=self.name,
                    technique=self.technique,
                    properties=properties,
                )
            )
            for attribute, value in pending_concepts:
                entities.append(self._concept_node(attribute, value, source, row_index))
            for attribute, target_id, value in pending_relations:
                relations.append(self._relation(attribute, business_id, target_id, value, source, row_index))

        logger.info(
            "Mapped %d rows of %s into %d entities and %d relations (%d edges skipped for missing targets)",
            len(records),
            self.plan.record_class,
            len(entities),
            len(relations),
            skipped_missing_targets,
        )
        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            metadata={
                "mapper": self.name,
                "record_type": self.plan.record_class,
                "table_name": self.plan.table_name,
                "identity_scope": self._identity_scope.value,
                "skipped_missing_targets": skipped_missing_targets,
            },
        )

    # -- Identity ---------------------------------------------------------------

    def _business_value(self, row: JsonDict, row_index: int) -> str:
        values = [str(row[key]) for key in self._identity_keys if row.get(key) not in (None, "")]
        return "|".join(values) if values else f"row-{row_index}"

    def _business_id(self, source: SourceRecord, business_value: str) -> str:
        if self._identity_scope is IdentityScope.GLOBAL:
            return stable_id("entity", self.plan.record_class, business_value)
        return stable_id("entity", source.source_name, self.plan.record_class, business_value)

    def _record_name(self, row: JsonDict, business_value: str) -> str:
        for attribute in self.plan.attributes:
            if attribute.semantic_role is ColumnRole.FREE_TEXT and _NAME_COLUMN_RE.search(attribute.column):
                value = row.get(attribute.column)
                if value not in (None, ""):
                    return str(value)
        return business_value

    # -- Concept nodes and edges ------------------------------------------------

    def _canonical(self, attribute: ResolvedAttribute, value: object) -> str:
        rule = attribute.canonicalization
        return rule.apply(str(value)) if rule is not None else str(value)

    def _concept_id(self, attribute: ResolvedAttribute, value: object) -> str:
        assert attribute.target is not None
        return ReferenceNode(reference_type=attribute.target, value=self._canonical(attribute, value)).node_id

    def _concept_node(
        self, attribute: ResolvedAttribute, value: object, source: SourceRecord, row_index: int
    ) -> EntityCandidate:
        assert attribute.target is not None
        canonical = self._canonical(attribute, value)
        concept_id = self._concept_id(attribute, value)
        cell_ev = cell_evidence(source, row_index=row_index, column=attribute.column, value=value)
        vocabulary = next(
            (item for item in self.plan.controlled_vocabularies if item.name == attribute.controlled_vocabulary),
            None,
        )
        code = vocabulary.canonical_value(str(value)) if vocabulary is not None else canonical
        return EntityCandidate.build(
            entity_type=attribute.target,
            name=str(value),
            evidence=[cell_ev],
            extractor_name=self.name,
            technique=self.technique,
            properties={
                "canonical_entity_id": concept_id,
                "canonical_value": canonical,
                "code": code or canonical,
                "raw_value": str(value),
                "identity_scope": "global_reference",
            },
        )

    def _edge_target(self, attribute: ResolvedAttribute, value: object, source: SourceRecord) -> str | None:
        foreign_key = self._fk_by_column.get(attribute.column)
        if foreign_key is None:
            return None
        if foreign_key.self_referential:
            return self._business_id(source, str(value))
        if foreign_key.target_available:
            return self._business_id(source, str(value))
        if foreign_key.on_missing is OnMissingTarget.ERROR:
            raise ValueError(
                f"foreign key {attribute.column!r} has no available target for value {value!r}"
            )
        # SKIP (or unmodelled QUARANTINE): never fabricate a placeholder entity.
        return None

    def _relation(
        self,
        attribute: ResolvedAttribute,
        source_id: str,
        target_id: str,
        value: object,
        source: SourceRecord,
        row_index: int,
    ) -> RelationCandidate:
        relationship = self._relationship_by_column[attribute.column]
        cell_ev = cell_evidence(source, row_index=row_index, column=attribute.column, value=value)
        return RelationCandidate.build(
            source=source_id,
            relation_type=relationship.name,
            target=target_id,
            evidence=[cell_ev],
            extractor_name=self.name,
            technique=self.technique,
            properties={"source_column": attribute.column, "source_value": str(value)},
        )


__all__ = ["SemanticTableMapper"]
