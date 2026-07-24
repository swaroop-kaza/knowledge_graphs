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
from collections import Counter
from typing import Any, cast

from kg_construction.application.resolved_plan import (
    ForeignKeyResolution,
    IdentityScope,
    Materialization,
    OnMissingTarget,
    ResolvedAttribute,
    ResolvedModelingPlan,
)
from kg_construction.application.structured_identity import (
    TargetKeyIndex,
    canonical_key,
    validate_unique_key,
)
from kg_construction.domain.evidence import cell_evidence, row_evidence
from kg_construction.domain.modeling import ColumnRole
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    RelationCandidate,
    SourceRecord,
    canonical_json,
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

    def __init__(
        self,
        plan: ResolvedModelingPlan,
        *,
        target_key_index: TargetKeyIndex
        | dict[tuple[str, str], set[str]]
        | None = None,
    ) -> None:
        self.plan = plan
        self._target_key_index = target_key_index
        self._attributes = {attribute.column: attribute for attribute in plan.attributes}
        self._relationship_by_column = {
            relationship.source_columns[0]: relationship
            for relationship in plan.relationships
            if relationship.source_columns
        }
        self._relationship_by_name = {
            relationship.name: relationship for relationship in plan.relationships
        }
        self._fk_by_column = {
            column: foreign_key
            for foreign_key in plan.foreign_keys
            for column in foreign_key.columns
        }
        self._fk_columns = set(self._fk_by_column)
        record_class = plan.clazz(plan.record_class)
        self._identity_keys = record_class.identity_keys
        self._identity_scope = record_class.identity_scope

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        records = source.records
        self._validate_identity(records)
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        skipped_missing_targets = 0
        keyless_occurrences: Counter[str] = Counter()

        for row_index, row in enumerate(records):
            if self._identity_keys:
                business_value = self._business_value(row, row_index)
            else:
                canonical_row = canonical_json(row)
                occurrence = keyless_occurrences[canonical_row]
                keyless_occurrences[canonical_row] += 1
                business_value = self._business_value(
                    row,
                    row_index,
                    duplicate_occurrence=occurrence,
                )
            business_id = self._business_id(source, business_value)
            source_row_id = stable_id(
                "source_row",
                source.source_id,
                business_value,
            )
            row_ev = row_evidence(source, row_index, row)

            properties: JsonDict = {
                "canonical_entity_id": business_id,
                "business_key": business_value,
                "identity_key": self._identity_key(source, business_value),
                "identity_scope": self._identity_scope.value,
                "source_row_id": source_row_id,
            }
            pending_concepts: list[tuple[ResolvedAttribute, object]] = []
            pending_relations: list[tuple[ResolvedAttribute, str, object]] = []

            for attribute in self.plan.attributes:
                value = row.get(attribute.column)
                if value in (None, ""):
                    if attribute.cardinality.required:
                        raise ValueError(
                            f"required column {attribute.column!r} is missing on row {row_index}"
                        )
                    continue
                if attribute.materialization in (Materialization.PROPERTY, Materialization.IDENTITY):
                    properties[attribute.target or attribute.column] = normalize_value(
                        value, attribute.datatype
                    )
                elif attribute.materialization is Materialization.NODE:
                    pending_concepts.append((attribute, value))
                    try:
                        concept_id = self._concept_id(attribute, value)
                    except ValueError as exc:
                        raise ValueError(
                            f"{exc}; source_id={source.source_id!r}, "
                            f"source_name={source.source_name!r}, "
                            f"row={row_index}, column={attribute.column!r}, "
                            f"raw_value={value!r}"
                        ) from exc
                    pending_relations.append((attribute, concept_id, value))
                elif (
                    attribute.materialization is Materialization.EDGE
                    and attribute.column not in self._fk_columns
                ):
                    # Declared FKs, including composite keys, are processed once
                    # below rather than once per participating column.
                    properties[attribute.column] = normalize_value(value, attribute.datatype)

            for foreign_key in self.plan.foreign_keys:
                raw_values = [row.get(column) for column in foreign_key.columns]
                if any(value in (None, "") for value in raw_values):
                    for column, value in zip(foreign_key.columns, raw_values, strict=True):
                        if value not in (None, ""):
                            properties[column] = value
                    continue
                target_id = self._foreign_key_target(foreign_key, raw_values, source)
                if target_id is None:
                    for column, value in zip(foreign_key.columns, raw_values, strict=True):
                        properties[column] = value
                    skipped_missing_targets += 1
                    continue
                relations.append(
                    self._foreign_key_relation(
                        foreign_key,
                        source_id=business_id,
                        target_id=target_id,
                        values=raw_values,
                        source=source,
                        row_index=row_index,
                    )
                )

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
                "resolved_modeling_plans": {
                    self.plan.plan_id: self.plan.model_dump(mode="json")
                },
            },
        )

    # -- Identity ---------------------------------------------------------------

    def _business_value(
        self,
        row: JsonDict,
        row_index: int,
        *,
        duplicate_occurrence: int = 0,
    ) -> str:
        if self._identity_keys:
            values = [row[key] for key in self._identity_keys]
            datatypes = [
                self._attributes[key].datatype for key in self._identity_keys
            ]
            return canonical_key(values, datatypes)
        # The row index is deliberately absent: a canonical row-content identity
        # plus the occurrence number among identical rows survives reordering.
        return canonical_json(
            {
                "row": row,
                "duplicate_occurrence": duplicate_occurrence,
            }
        )

    def _validate_identity(self, records: list[JsonDict]) -> None:
        if not self._identity_keys:
            return
        validate_unique_key(
            records,
            columns=self._identity_keys,
            datatypes=[
                self._attributes[key].datatype for key in self._identity_keys
            ],
            key_name=(
                f"primary key {self.plan.record_class}"
                f"({', '.join(self._identity_keys)})"
            ),
        )

    def _business_id(self, source: SourceRecord, business_value: str) -> str:
        return self._entity_id(
            entity_class=self.plan.record_class,
            identity_scope=self._identity_scope,
            source_name=source.source_name,
            business_value=business_value,
        )

    @staticmethod
    def _entity_id(
        *,
        entity_class: str,
        identity_scope: IdentityScope,
        source_name: str,
        business_value: str,
    ) -> str:
        if identity_scope is IdentityScope.GLOBAL:
            return stable_id("entity", entity_class, business_value)
        return stable_id("entity", source_name, entity_class, business_value)

    def _identity_key(self, source: SourceRecord, business_value: str) -> str:
        if self._identity_scope is IdentityScope.GLOBAL:
            return f"global:{self.plan.record_class}:{business_value}"
        return (
            f"{self._identity_scope.value}:{source.source_name}:"
            f"{self.plan.record_class}:{business_value}"
        )

    def _record_name(self, row: JsonDict, business_value: str) -> str:
        for attribute in self.plan.attributes:
            if attribute.semantic_role is ColumnRole.FREE_TEXT and _NAME_COLUMN_RE.search(attribute.column):
                value = row.get(attribute.column)
                if value not in (None, ""):
                    return str(value)
        return business_value

    # -- Concept nodes and edges ------------------------------------------------

    def _canonical(self, attribute: ResolvedAttribute, value: object) -> str:
        vocabulary = next(
            (
                item
                for item in self.plan.controlled_vocabularies
                if item.name == attribute.controlled_vocabulary
                or item.name == attribute.target
            ),
            None,
        )
        if vocabulary is not None:
            canonical = vocabulary.canonical_value(str(value))
            if canonical is None:
                raise ValueError(
                    f"value {value!r} is not permitted by closed vocabulary {vocabulary.name!r}"
                )
            return canonical
        rule = attribute.canonicalization
        return rule.apply(str(value)) if rule is not None else str(value)

    def _concept_id(self, attribute: ResolvedAttribute, value: object) -> str:
        assert attribute.target is not None
        canonical = self._canonical(attribute, value)
        concept_class = self.plan.clazz(attribute.target)
        scheme = concept_class.concept_scheme or attribute.target
        return stable_id("concept", scheme, canonical)

    def _concept_node(
        self, attribute: ResolvedAttribute, value: object, source: SourceRecord, row_index: int
    ) -> EntityCandidate:
        assert attribute.target is not None
        canonical = self._canonical(attribute, value)
        concept_id = self._concept_id(attribute, value)
        cell_ev = cell_evidence(source, row_index=row_index, column=attribute.column, value=value)
        vocabulary = next(
            (
                item
                for item in self.plan.controlled_vocabularies
                if item.name == attribute.controlled_vocabulary
                or item.name == attribute.target
            ),
            None,
        )
        code = vocabulary.canonical_value(str(value)) if vocabulary is not None else canonical
        scheme = self.plan.clazz(attribute.target).concept_scheme or attribute.target
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
                "concept_scheme": scheme,
                "raw_value": str(value),
                "raw_values": [str(value)],
                "source_occurrences": [
                    {
                        "source_id": source.source_id,
                        "source_name": source.source_name,
                        "row": row_index,
                        "column": attribute.column,
                        "raw_value": str(value),
                    }
                ],
                "identity_scope": "global_reference",
            },
        )

    def _foreign_key_target(
        self,
        foreign_key: ForeignKeyResolution,
        values: list[object],
        source: SourceRecord,
    ) -> str | None:
        attributes = [self._attributes[column] for column in foreign_key.columns]
        target_datatypes: list[str | None] = list(
            foreign_key.target_key_datatypes
        )
        if (
            not target_datatypes
            and foreign_key.target_key
            and all(column in self._attributes for column in foreign_key.target_key)
        ):
            target_datatypes = [
                self._attributes[column].datatype
                for column in foreign_key.target_key
            ]
        if not target_datatypes:
            target_datatypes = [
                attribute.datatype for attribute in attributes
            ]
        business_value = canonical_key(
            values,
            target_datatypes,
        )
        if foreign_key.self_referential or foreign_key.target_available:
            target_source = foreign_key.target_source_name or source.source_name
            if self._target_key_index is not None:
                target_index = cast(
                    dict[object, object],
                    self._target_key_index,
                )
                lookup = (
                    target_source,
                    foreign_key.target_class,
                    tuple(foreign_key.target_key),
                    business_value,
                )
                target_id = target_index.get(lookup)
                if isinstance(target_id, str):
                    return target_id

                # Backward compatibility for callers that supplied the original
                # ``(source, class) -> set[canonical primary key]`` index.
                legacy_available = target_index.get(
                    (target_source, foreign_key.target_class),
                    set(),
                )
                if (
                    isinstance(legacy_available, set)
                    and business_value in legacy_available
                ):
                    target_id = self._entity_id(
                        entity_class=foreign_key.target_class,
                        identity_scope=foreign_key.target_identity_scope,
                        source_name=target_source,
                        business_value=business_value,
                    )
                    return target_id
                if foreign_key.on_missing is OnMissingTarget.ERROR:
                    raise ValueError(
                        f"foreign key {foreign_key.columns!r} has no target row "
                        f"for values {values!r} via target key "
                        f"{foreign_key.target_key!r}"
                    )
                return None
            target_class = self.plan.clazz(foreign_key.target_class)
            if (
                foreign_key.target_key
                and foreign_key.target_key != target_class.identity_keys
            ):
                raise ValueError(
                    f"foreign key {foreign_key.columns!r} targets alternate key "
                    f"{foreign_key.target_key!r}; a prepared target-key index is "
                    "required to resolve it to the primary entity"
                )
            return self._entity_id(
                entity_class=foreign_key.target_class,
                identity_scope=foreign_key.target_identity_scope,
                source_name=target_source,
                business_value=business_value,
            )
        if foreign_key.on_missing is OnMissingTarget.ERROR:
            raise ValueError(
                f"foreign key {foreign_key.columns!r} has no available target for values {values!r}"
            )
        # SKIP (or unmodelled QUARANTINE): never fabricate a placeholder entity.
        return None

    def _foreign_key_relation(
        self,
        foreign_key: ForeignKeyResolution,
        *,
        source_id: str,
        target_id: str,
        values: list[object],
        source: SourceRecord,
        row_index: int,
    ) -> RelationCandidate:
        relationship = self._relationship_by_name[foreign_key.relationship_name]
        evidence = [
            cell_evidence(source, row_index=row_index, column=column, value=value)
            for column, value in zip(foreign_key.columns, values, strict=True)
        ]
        return RelationCandidate.build(
            source=source_id,
            relation_type=relationship.name,
            target=target_id,
            evidence=evidence,
            extractor_name=self.name,
            technique=self.technique,
            properties={
                "source_columns": list(foreign_key.columns),
                "target_columns": list(foreign_key.target_key),
                "source_values": list(values),
            },
        )

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
