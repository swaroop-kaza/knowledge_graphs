"""Structured data to graph schema proposal workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import ConstructionBatch, SourceRecord, stable_id
from kg_construction.domain.schema import (
    GraphSchema,
    JsonType,
    NodeTypeDefinition,
    PropertyDefinition,
    RelationshipTypeDefinition,
    SchemaDiffReport,
    SchemaLifecycleStatus,
    diff_graph_schemas,
)
from kg_construction.infrastructure.mapping.direct_table_mapper import profile_records


class SchemaProposalResult(BaseModel):
    """Schema proposal plus generated ontology artifacts."""

    model_config = ConfigDict(extra="forbid")

    graph_schema: GraphSchema
    shacl_turtle: str
    owl_turtle: str | None = None
    diff_report: SchemaDiffReport | None = None
    review_state: SchemaLifecycleStatus = "proposed"
    evidence: dict[str, object] = Field(default_factory=dict)


class SchemaProposalWorkflow:
    """Infer graph schema candidates from structured table profiles."""

    name = "schema_proposal_workflow"

    def __init__(
        self,
        *,
        namespace: str = "urn:kg:schema:",
        version: str = "v1",
        emit_owl: bool = True,
    ) -> None:
        self.namespace = namespace
        self.version = version
        self.emit_owl = emit_owl

    def propose(
        self,
        data: ConstructionBatch | list[SourceRecord],
        *,
        previous_schema: GraphSchema | None = None,
        schema_id: str | None = None,
    ) -> SchemaProposalResult:
        """Infer a graph schema from structured source records."""

        sources = data.source_records if isinstance(data, ConstructionBatch) else data
        node_types: list[NodeTypeDefinition] = []
        relationship_types: list[RelationshipTypeDefinition] = []
        table_evidence: list[dict[str, object]] = []

        for source in sources:
            if not source.records:
                continue
            table_name = table_name_from_source(source)
            profile = profile_records(source.records)
            node_type = infer_node_type(table_name, profile)
            null_rates = null_rates_from_profile(profile)
            primary_keys = infer_primary_keys(profile)
            foreign_keys = infer_foreign_keys(source.records, primary_keys)
            properties = [
                infer_property(column, profile, null_rates, primary_keys)
                for column in profile.get("columns", [])
            ]
            node_types.append(
                NodeTypeDefinition(
                    name=node_type,
                    properties=properties,
                    primary_keys=primary_keys,
                    source_table=table_name,
                    status="proposed",
                    confidence=node_confidence(profile, primary_keys, foreign_keys),
                    description=f"Inferred from structured table '{table_name}'.",
                )
            )
            relationship_types.extend(
                infer_relationships(
                    node_type=node_type,
                    foreign_keys=foreign_keys,
                    profile=profile,
                )
            )
            table_evidence.append(
                {
                    "source_id": source.source_id,
                    "source_name": source.source_name,
                    "table_name": table_name,
                    "node_type": node_type,
                    "primary_keys": primary_keys,
                    "foreign_keys": foreign_keys,
                    "column_types": column_types_from_profile(profile),
                    "null_rates": null_rates,
                    "profile": profile,
                }
            )

        schema = GraphSchema(
            schema_id=schema_id or stable_id("schema", *(source.source_id for source in sources)),
            version=self.version,
            namespace=self.namespace,
            status="proposed",
            node_types=dedupe_node_types(node_types),
            relationship_types=dedupe_relationship_types(relationship_types),
            metadata={
                "workflow": self.name,
                "source_count": len(sources),
                "structured_source_count": sum(1 for source in sources if source.records),
            },
        )
        diff_report = diff_graph_schemas(previous_schema, schema) if previous_schema else None
        return SchemaProposalResult(
            graph_schema=schema,
            shacl_turtle=schema.to_shacl_turtle(),
            owl_turtle=schema.to_owl_turtle() if self.emit_owl else None,
            diff_report=diff_report,
            review_state=schema.status,
            evidence={"tables": table_evidence},
        )

    @staticmethod
    def approve(schema: GraphSchema) -> GraphSchema:
        """Approve a proposed schema."""

        return schema.approve()

    @staticmethod
    def reject(schema: GraphSchema) -> GraphSchema:
        """Reject a proposed schema."""

        return schema.reject()

    @staticmethod
    def promote(schema: GraphSchema) -> GraphSchema:
        """Promote an approved schema into active registry use."""

        return schema.promote()

    @staticmethod
    def diff(previous_schema: GraphSchema, current_schema: GraphSchema) -> SchemaDiffReport:
        """Generate a deterministic schema diff report."""

        return diff_graph_schemas(previous_schema, current_schema)


def table_name_from_source(source: SourceRecord) -> str:
    """Infer a stable table name from source metadata or path."""

    if source.metadata.get("table_name"):
        return str(source.metadata["table_name"])
    stem = Path(source.source_name).stem
    return stem or source.source_type


def infer_node_type(table_name: str, profile: dict[str, Any]) -> str:
    """Infer a graph node label from table and column signals."""

    columns = {str(column).casefold() for column in profile.get("columns", [])}
    if {"name", "email"} <= columns or "manager_id" in columns:
        return "Person"
    table = singularize(table_name.replace("_", " ").replace("-", " ")).strip()
    return "".join(part.capitalize() for part in table.split()) or "Record"


def infer_property(
    column: str,
    profile: dict[str, Any],
    null_rates: dict[str, float],
    primary_keys: list[str],
) -> PropertyDefinition:
    """Infer a graph property definition from a structured column profile."""

    row_count = int(profile.get("row_count", 0))
    distinct_counts = profile.get("distinct_counts", {})
    null_rate = float(null_rates.get(column, 0.0))
    value_type = to_json_type(str(profile.get("types", {}).get(column, "string")), column)
    unique = row_count > 0 and distinct_counts.get(column) == row_count
    required = null_rate == 0.0 and (column in primary_keys or not column.endswith("_id"))
    return PropertyDefinition(
        name=column,
        value_type=value_type,
        required=required,
        nullable=null_rate > 0.0,
        unique=unique,
        source_column=column,
        null_rate=null_rate,
        confidence=property_confidence(row_count, null_rate, unique),
        status="proposed",
        description=f"Inferred from source column '{column}'.",
    )


def null_rates_from_profile(profile: dict[str, Any]) -> dict[str, float]:
    """Compute null rates from a table profile."""

    row_count = max(1, int(profile.get("row_count", 0)))
    return {
        str(column): count / row_count
        for column, count in profile.get("null_counts", {}).items()
    }


def column_types_from_profile(profile: dict[str, Any]) -> dict[str, str]:
    """Return string column type mappings from a table profile."""

    return {
        str(column): str(column_type)
        for column, column_type in profile.get("types", {}).items()
    }


def infer_primary_keys(profile: dict[str, Any]) -> list[str]:
    """Infer primary keys with explicit identifier columns before uniqueness fallback."""

    columns = [str(column) for column in profile.get("columns", [])]
    row_count = int(profile.get("row_count", 0))
    distinct_counts = profile.get("distinct_counts", {})
    preferred = [
        column
        for column in columns
        if column.casefold() in {"id", "uuid", "key"}
        and row_count > 0
        and distinct_counts.get(column) == row_count
    ]
    if preferred:
        return preferred[:1]
    fallback = [
        column
        for column in columns
        if row_count > 0 and distinct_counts.get(column) == row_count
    ]
    return fallback[:1]


def infer_foreign_keys(
    records: list[dict[str, Any]],
    primary_keys: list[str],
) -> list[tuple[str, str]]:
    """Infer foreign keys using detected primary-key value domains."""

    if not records or not primary_keys:
        return []
    values_by_key = {
        key: {str(row.get(key)) for row in records if row.get(key) not in (None, "")}
        for key in primary_keys
    }
    foreign_keys: list[tuple[str, str]] = []
    for column in sorted({key for row in records for key in row}):
        if not column.endswith("_id") or column in primary_keys:
            continue
        values = {str(row.get(column)) for row in records if row.get(column) not in (None, "")}
        for primary_key, primary_values in values_by_key.items():
            if values and values <= primary_values:
                foreign_keys.append((column, primary_key))
    return foreign_keys


def infer_relationships(
    *,
    node_type: str,
    foreign_keys: list[tuple[str, str]],
    profile: dict[str, Any],
) -> list[RelationshipTypeDefinition]:
    """Infer relationship schemas from foreign-key signals."""

    relationships: list[RelationshipTypeDefinition] = []
    for source_column, target_column in foreign_keys:
        relationship_type = relationship_type_from_fk(source_column)
        null_count = profile.get("null_counts", {}).get(source_column, 0)
        row_count = int(profile.get("row_count", 0))
        confidence = 0.95 if row_count and null_count < row_count else 0.75
        relationships.append(
            RelationshipTypeDefinition(
                name=relationship_type,
                source_types=[node_type],
                target_types=[node_type],
                source_columns=[source_column],
                target_columns=[target_column],
                status="proposed",
                confidence=confidence,
                description=(
                    f"Inferred from foreign key {source_column} -> {target_column}."
                ),
            )
        )
    return relationships


def relationship_type_from_fk(source_column: str) -> str:
    """Turn a foreign-key column into a graph relationship label."""

    lowered = source_column.casefold()
    if lowered in {"manager_id", "reports_to_id", "supervisor_id"}:
        return "REPORTS_TO"
    if lowered in {"parent_id", "parent_key"}:
        return "CHILD_OF"
    base = lowered.removesuffix("_id").removesuffix("_key")
    return f"HAS_{base.upper()}" if base else "REFERENCES"


def to_json_type(column_type: str, column: str) -> JsonType:
    """Map column profiling output into schema value types."""

    lowered = column.casefold()
    if lowered.endswith("_date") or lowered == "date":
        return "date"
    if "time" in lowered or lowered.endswith("_at"):
        return "datetime"
    if column_type == "integer":
        return "integer"
    if column_type == "number":
        return "number"
    if column_type == "boolean":
        return "boolean"
    return "string"


def node_confidence(
    profile: dict[str, Any],
    primary_keys: list[str],
    foreign_keys: list[tuple[str, str]],
) -> float:
    """Score schema proposal confidence from structural signals."""

    score = 0.45
    if primary_keys:
        score += 0.25
    if foreign_keys:
        score += 0.15
    if profile.get("columns"):
        score += 0.10
    return min(1.0, score)


def property_confidence(row_count: int, null_rate: float, unique: bool) -> float:
    """Score confidence for one inferred property."""

    score = 0.70 if row_count else 0.45
    if null_rate == 0.0:
        score += 0.15
    if unique:
        score += 0.10
    return min(1.0, score)


def dedupe_node_types(node_types: list[NodeTypeDefinition]) -> list[NodeTypeDefinition]:
    """Merge node definitions with the same name."""

    merged: dict[str, NodeTypeDefinition] = {}
    for node in node_types:
        if node.name not in merged:
            merged[node.name] = node
            continue
        existing = merged[node.name]
        properties = {
            property_definition.name: property_definition
            for property_definition in [*existing.properties, *node.properties]
        }
        merged[node.name] = existing.model_copy(
            update={
                "properties": list(properties.values()),
                "primary_keys": sorted({*existing.primary_keys, *node.primary_keys}),
                "confidence": max(existing.confidence, node.confidence),
            }
        )
    return [merged[name] for name in sorted(merged)]


def dedupe_relationship_types(
    relationship_types: list[RelationshipTypeDefinition],
) -> list[RelationshipTypeDefinition]:
    """Merge relationship definitions with the same name."""

    merged: dict[str, RelationshipTypeDefinition] = {}
    for relationship in relationship_types:
        if relationship.name not in merged:
            merged[relationship.name] = relationship
            continue
        existing = merged[relationship.name]
        merged[relationship.name] = existing.model_copy(
            update={
                "source_types": sorted({*existing.source_types, *relationship.source_types}),
                "target_types": sorted({*existing.target_types, *relationship.target_types}),
                "source_columns": sorted({*existing.source_columns, *relationship.source_columns}),
                "target_columns": sorted({*existing.target_columns, *relationship.target_columns}),
                "confidence": max(existing.confidence, relationship.confidence),
            }
        )
    return [merged[name] for name in sorted(merged)]


def singularize(value: str) -> str:
    """Small deterministic singularizer for schema labels."""

    value = value.strip()
    lowered = value.casefold()
    if lowered.endswith("ies") and len(value) > 3:
        return f"{value[:-3]}y"
    if lowered.endswith("people"):
        return value[:-6] + "person"
    if lowered.endswith("s") and len(value) > 1:
        return value[:-1]
    return value


__all__ = ["SchemaProposalResult", "SchemaProposalWorkflow"]
