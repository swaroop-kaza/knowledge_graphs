"""Direct row-to-node and foreign-key-to-edge mapping."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.evidence import row_evidence
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    RelationCandidate,
    SourceRecord,
    stable_id,
)

JsonDict = dict[str, Any]


class ForeignKeyMappingConfig(BaseModel):
    """Configuration for mapping a column value to an edge."""

    model_config = ConfigDict(extra="forbid")

    source_column: str
    target_column: str
    relationship_type: str
    target_node_type: str | None = None


class DirectTableMappingConfig(BaseModel):
    """Configuration for deterministic table mapping."""

    model_config = ConfigDict(extra="forbid")

    table_name: str = "records"
    node_type: str = "Record"
    id_columns: list[str] = Field(default_factory=list)
    name_column: str | None = None
    property_columns: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyMappingConfig] = Field(default_factory=list)
    schema_version: str = "v1"


def infer_column_type(values: list[object]) -> str:
    """Infer a simple column type from observed values."""

    non_empty = [value for value in values if value not in (None, "")]
    if not non_empty:
        return "unknown"
    if all(str(value).isdigit() for value in non_empty):
        return "integer"
    for caster, type_name in ((float, "number"),):
        try:
            for value in non_empty:
                caster(str(value))
        except ValueError:
            continue
        return type_name
    lowered = {str(value).casefold() for value in non_empty}
    if lowered <= {"true", "false", "0", "1", "yes", "no"}:
        return "boolean"
    return "string"


def profile_records(records: list[JsonDict]) -> JsonDict:
    """Profile table shape for schema inference and drift checks."""

    columns = sorted({column for row in records for column in row})
    type_profile = {
        column: infer_column_type([row.get(column) for row in records])
        for column in columns
    }
    null_counts = {
        column: sum(1 for row in records if row.get(column) in (None, ""))
        for column in columns
    }
    value_counts = {
        column: len({str(row.get(column)) for row in records if row.get(column) not in (None, "")})
        for column in columns
    }
    return {
        "columns": columns,
        "row_count": len(records),
        "types": type_profile,
        "null_counts": null_counts,
        "distinct_counts": value_counts,
    }


class DirectTableMapper:
    """Map rows to entity candidates and configured foreign keys to relations."""

    name = "direct_table_mapper"
    technique = "direct_structured_mapping"

    def __init__(self, config: DirectTableMappingConfig | None = None) -> None:
        self.config = config or DirectTableMappingConfig()

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        records = source.records
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        row_entity_ids: dict[int, str] = {}
        target_indexes = self._build_target_indexes(records)

        for row_index, row in enumerate(records):
            evidence = [row_evidence(source, row_index, row)]
            name = self._row_name(row, row_index)
            properties = self._row_properties(row)
            properties.update(
                {
                    "_source_table": self.config.table_name,
                    "_row_index": row_index,
                    "_stable_row_id": stable_id(
                        "row",
                        source.source_id,
                        self.config.table_name,
                        self._row_key(row, row_index),
                    ),
                }
            )
            entity = EntityCandidate.build(
                entity_type=self.config.node_type,
                name=name,
                properties=properties,
                evidence=evidence,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.config.schema_version,
            )
            entities.append(entity)
            row_entity_ids[row_index] = entity.candidate_id

        for row_index, row in enumerate(records):
            for foreign_key in self.config.foreign_keys:
                target_value = row.get(foreign_key.source_column)
                if target_value in (None, ""):
                    continue
                target_index = target_indexes.get(foreign_key.target_column, {}).get(str(target_value))
                if target_index is None or target_index not in row_entity_ids:
                    continue
                source_entity_id = row_entity_ids[row_index]
                target_entity_id = row_entity_ids[target_index]
                relations.append(
                    RelationCandidate.build(
                        source=source_entity_id,
                        relation_type=foreign_key.relationship_type,
                        target=target_entity_id,
                        properties={
                            "source_column": foreign_key.source_column,
                            "target_column": foreign_key.target_column,
                            "source_value": target_value,
                        },
                        evidence=[row_evidence(source, row_index, row)],
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.config.schema_version,
                    )
                )

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            metadata={
                "mapper": self.name,
                "profile": profile_records(records),
                "potential_key_columns": self._potential_key_columns(records),
            },
        )

    def _row_key(self, row: JsonDict, row_index: int) -> tuple[object, ...]:
        if self.config.id_columns:
            return tuple(row.get(column) for column in self.config.id_columns)
        return (row_index, repr(sorted(row.items())))

    def _row_name(self, row: JsonDict, row_index: int) -> str:
        if self.config.name_column and row.get(self.config.name_column):
            return str(row[self.config.name_column])
        if self.config.id_columns:
            return "|".join(str(row.get(column, "")) for column in self.config.id_columns)
        for column in ("name", "title", "id", "email"):
            if row.get(column):
                return str(row[column])
        return f"{self.config.node_type}:{row_index}"

    def _row_properties(self, row: JsonDict) -> JsonDict:
        if self.config.property_columns:
            columns = self.config.property_columns
        else:
            foreign_key_columns = {item.source_column for item in self.config.foreign_keys}
            columns = [column for column in row if column not in foreign_key_columns]
        return {column: row.get(column) for column in columns}

    @staticmethod
    def _build_target_indexes(records: list[JsonDict]) -> dict[str, dict[str, int]]:
        indexes: dict[str, dict[str, int]] = {}
        for row_index, row in enumerate(records):
            for column, value in row.items():
                if value in (None, ""):
                    continue
                indexes.setdefault(column, {}).setdefault(str(value), row_index)
        return indexes

    @staticmethod
    def _potential_key_columns(records: list[JsonDict]) -> list[str]:
        if not records:
            return []
        row_count = len(records)
        columns = sorted({column for row in records for column in row})
        keys: list[str] = []
        for column in columns:
            values = [row.get(column) for row in records if row.get(column) not in (None, "")]
            counts = Counter(str(value) for value in values)
            if len(values) == row_count and all(count == 1 for count in counts.values()):
                keys.append(column)
        return keys


__all__ = [
    "DirectTableMapper",
    "DirectTableMappingConfig",
    "ForeignKeyMappingConfig",
    "infer_column_type",
    "profile_records",
]
