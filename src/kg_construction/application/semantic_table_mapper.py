"""Turn structured rows into a well-formed, evidence-backed knowledge graph.

Following a ``ModelingPolicy``, each row becomes a typed node; a categorical
dimension collapses into a single shared, deduplicated reference node linked by a
typed edge (so a value is stored once, never repeated as a literal on every row);
a foreign key becomes a typed edge to the referenced entity rather than a copied
value; and measure, temporal, and free-text columns stay as node properties. A
supernode guard keeps very low-cardinality categoricals as properties instead of
nodes.

Every occurrence of the same canonical value shares one ``canonical_entity_id``,
so the property-graph builder merges them into a single node automatically -- the
graph equivalent of merging on a business key. The mapper plugs into the workflow
via ``map_source`` and needs no dedicated graph builder.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from kg_construction.application.modeling_policy import ColumnPolicy, ModelingPolicy
from kg_construction.domain.evidence import cell_evidence, row_evidence
from kg_construction.domain.modeling import ColumnRole, ReferenceNode
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    RelationCandidate,
    SourceRecord,
    stable_id,
)

JsonDict = dict[str, Any]

logger = logging.getLogger(__name__)

_NAME_COLUMN_RE = re.compile(r"(name|title|label)", re.IGNORECASE)
_PROPERTY_ROLES = {ColumnRole.MEASURE, ColumnRole.TEMPORAL, ColumnRole.FREE_TEXT, ColumnRole.IDENTIFIER}


class SemanticTableMapper:
    """Map rows to typed nodes, shared reference nodes, and typed relationships."""

    name = "semantic_table_mapper"
    technique = "policy_driven_semantic_mapping"

    def __init__(self, policy: ModelingPolicy) -> None:
        self.policy = policy

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        records = source.records
        distinct_counts = self._distinct_counts(records)
        node_categoricals, property_categoricals = self._materialization(distinct_counts)

        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []

        for row_index, row in enumerate(records):
            identity_value = self._identity_value(row, row_index)
            record_id = stable_id("record", source.source_name, self.policy.record_type, identity_value)
            row_ev = row_evidence(source, row_index, row)

            properties: JsonDict = {
                "canonical_entity_id": record_id,
                "identity_key": f"source_record:{record_id}",
                "identity_scope": source.source_name,
            }
            for column_policy in self.policy.columns:
                value = row.get(column_policy.column)
                if value in (None, ""):
                    continue
                keep_as_property = column_policy.role in _PROPERTY_ROLES or (
                    column_policy.role == ColumnRole.CATEGORICAL_DIMENSION
                    and column_policy.column in property_categoricals
                )
                if keep_as_property:
                    properties[column_policy.column] = value

            entities.append(
                EntityCandidate.build(
                    entity_type=self.policy.record_type,
                    name=self._record_name(row, identity_value),
                    evidence=[row_ev],
                    extractor_name=self.name,
                    technique=self.technique,
                    properties=properties,
                )
            )

            for column_policy in self.policy.columns:
                value = row.get(column_policy.column)
                if value in (None, "") or column_policy.reference_type is None:
                    continue
                reference_type = column_policy.reference_type
                is_node_categorical = (
                    column_policy.role == ColumnRole.CATEGORICAL_DIMENSION
                    and column_policy.column in node_categoricals
                )
                if not is_node_categorical and column_policy.role != ColumnRole.ENTITY_REFERENCE:
                    continue

                cell_ev = cell_evidence(
                    source, row_index=row_index, column=column_policy.column, value=value
                )
                if column_policy.role == ColumnRole.ENTITY_REFERENCE and reference_type == self.policy.record_type:
                    # Foreign key to another row of the same table: a typed edge to
                    # that record, never a copied literal.
                    target_id = stable_id("record", source.source_name, self.policy.record_type, str(value))
                else:
                    reference_id, reference_entity = self._reference_entity(
                        reference_type, column_policy, value, cell_ev
                    )
                    entities.append(reference_entity)
                    target_id = reference_id

                relations.append(
                    RelationCandidate.build(
                        source=record_id,
                        relation_type=column_policy.target_relationship(),
                        target=target_id,
                        evidence=[cell_ev],
                        extractor_name=self.name,
                        technique=self.technique,
                        properties={"source_column": column_policy.column, "source_value": str(value)},
                    )
                )

        logger.info(
            "Mapped %d rows of %s into %d entities and %d relations "
            "(dimension nodes: %s, kept as properties: %s)",
            len(records),
            self.policy.record_type,
            len(entities),
            len(relations),
            sorted(node_categoricals),
            sorted(property_categoricals),
        )
        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            metadata={
                "mapper": self.name,
                "record_type": self.policy.record_type,
                "table_name": self.policy.table_name,
                "node_categoricals": sorted(node_categoricals),
                "property_categoricals": sorted(property_categoricals),
                "categorical_distinct_counts": distinct_counts,
            },
        )

    def _reference_entity(
        self,
        reference_type: str,
        column_policy: ColumnPolicy,
        value: object,
        evidence: EvidenceRef,
    ) -> tuple[str, EntityCandidate]:
        canonical = self.policy.canonicalization_for(column_policy.column).apply(str(value))
        reference_id = ReferenceNode(reference_type=reference_type, value=canonical).node_id
        entity = EntityCandidate.build(
            entity_type=reference_type,
            name=str(value),
            evidence=[evidence],
            extractor_name=self.name,
            technique=self.technique,
            properties={
                "canonical_entity_id": reference_id,
                "canonical_value": canonical,
                "identity_scope": "global_reference",
            },
        )
        return reference_id, entity

    def _identity_value(self, row: JsonDict, row_index: int) -> str:
        values = [str(row.get(column)) for column in self.policy.id_columns if row.get(column) not in (None, "")]
        return "|".join(values) if values else f"row-{row_index}"

    def _record_name(self, row: JsonDict, identity_value: str) -> str:
        for column_policy in self.policy.columns:
            if column_policy.role == ColumnRole.FREE_TEXT and _NAME_COLUMN_RE.search(column_policy.column):
                value = row.get(column_policy.column)
                if value not in (None, ""):
                    return str(value)
        return identity_value

    def _distinct_counts(self, records: list[JsonDict]) -> dict[str, int]:
        columns = [
            column_policy.column
            for column_policy in self.policy.columns
            if column_policy.role == ColumnRole.CATEGORICAL_DIMENSION
        ]
        return {
            column: len({str(row.get(column)) for row in records if row.get(column) not in (None, "")})
            for column in columns
        }

    def _materialization(self, distinct_counts: dict[str, int]) -> tuple[set[str], set[str]]:
        thresholds = self.policy.thresholds
        node_categoricals: set[str] = set()
        property_categoricals: set[str] = set()
        for column_policy in self.policy.columns:
            if column_policy.role != ColumnRole.CATEGORICAL_DIMENSION:
                continue
            distinct = distinct_counts.get(column_policy.column, 0)
            if thresholds.promote_categoricals_to_nodes and distinct > thresholds.keep_as_property_max_distinct:
                node_categoricals.add(column_policy.column)
            else:
                property_categoricals.add(column_policy.column)
        return node_categoricals, property_categoricals


__all__ = ["SemanticTableMapper"]
