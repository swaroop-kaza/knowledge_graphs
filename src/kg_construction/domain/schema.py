"""Schema and ontology-facing domain objects."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JsonType = Literal["string", "number", "integer", "boolean", "object", "array", "date", "datetime"]
SchemaLifecycleStatus = Literal["proposed", "approved", "rejected", "promoted"]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _term(name: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in name)
    return safe.strip("_") or "unnamed"


def _xsd_datatype(value_type: JsonType) -> str:
    mapping = {
        "string": "xsd:string",
        "number": "xsd:decimal",
        "integer": "xsd:integer",
        "boolean": "xsd:boolean",
        "date": "xsd:date",
        "datetime": "xsd:dateTime",
        "object": "xsd:string",
        "array": "xsd:string",
    }
    return mapping[value_type]


class PropertyDefinition(BaseModel):
    """Definition for a node or relationship property."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value_type: JsonType = "string"
    required: bool = False
    nullable: bool = True
    unique: bool = False
    source_column: str | None = None
    null_rate: float = 0.0
    confidence: float = 1.0
    status: SchemaLifecycleStatus = "proposed"
    description: str | None = None


class NodeTypeDefinition(BaseModel):
    """Allowed node/entity type definition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    properties: list[PropertyDefinition] = Field(default_factory=list)
    primary_keys: list[str] = Field(default_factory=list)
    source_table: str | None = None
    status: SchemaLifecycleStatus = "proposed"
    confidence: float = 1.0
    description: str | None = None

    @property
    def required_properties(self) -> set[str]:
        return {item.name for item in self.properties if item.required}


class RelationshipTypeDefinition(BaseModel):
    """Allowed relationship type with optional domain/range constraints."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    properties: list[PropertyDefinition] = Field(default_factory=list)
    source_columns: list[str] = Field(default_factory=list)
    target_columns: list[str] = Field(default_factory=list)
    status: SchemaLifecycleStatus = "proposed"
    confidence: float = 1.0
    description: str | None = None


class GraphSchema(BaseModel):
    """Versioned schema registry payload."""

    model_config = ConfigDict(extra="forbid")

    schema_id: str = "default"
    version: str = "v1"
    namespace: str = "urn:kg:schema:"
    status: SchemaLifecycleStatus = "proposed"
    node_types: list[NodeTypeDefinition] = Field(default_factory=list)
    relationship_types: list[RelationshipTypeDefinition] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now_iso)

    def with_status(self, status: SchemaLifecycleStatus) -> GraphSchema:
        """Return a copy of the schema with lifecycle status propagated."""

        return self.model_copy(
            update={
                "status": status,
                "node_types": [
                    node.model_copy(
                        update={
                            "status": status,
                            "properties": [
                                prop.model_copy(update={"status": status})
                                for prop in node.properties
                            ],
                        }
                    )
                    for node in self.node_types
                ],
                "relationship_types": [
                    relation.model_copy(
                        update={
                            "status": status,
                            "properties": [
                                prop.model_copy(update={"status": status})
                                for prop in relation.properties
                            ],
                        }
                    )
                    for relation in self.relationship_types
                ],
            }
        )

    def approve(self) -> GraphSchema:
        """Mark a proposed schema as approved by human or policy review."""

        return self.with_status("approved")

    def reject(self) -> GraphSchema:
        """Mark a proposed schema as rejected."""

        return self.with_status("rejected")

    def promote(self) -> GraphSchema:
        """Mark an approved schema as promoted into the active registry."""

        return self.with_status("promoted")

    def to_shacl_turtle(self) -> str:
        """Serialize node and relationship constraints as lightweight SHACL Turtle."""

        lines = [
            "@prefix sh: <http://www.w3.org/ns/shacl#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            f"@prefix kg: <{self.namespace}> .",
            "",
        ]
        for node in self.node_types:
            shape_name = f"{_term(node.name)}Shape"
            lines.extend(
                [
                    f"kg:{shape_name} a sh:NodeShape ;",
                    f"  sh:targetClass kg:{_term(node.name)} ;",
                ]
            )
            property_blocks = [
                self._shacl_property_block(property_definition)
                for property_definition in node.properties
            ]
            property_blocks.extend(
                self._shacl_relationship_block(relationship)
                for relationship in self.relationship_types
                if node.name in relationship.source_types
            )
            if property_blocks:
                for index, block in enumerate(property_blocks):
                    suffix = " ." if index == len(property_blocks) - 1 else " ;"
                    lines.append(f"  sh:property {block}{suffix}")
            else:
                lines[-1] = lines[-1].rstrip(" ;") + " ."
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def to_owl_turtle(self) -> str:
        """Serialize an OWL-lite/RDFS vocabulary for classes and properties."""

        lines = [
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            f"@prefix kg: <{self.namespace}> .",
            "",
            "kg:Ontology a owl:Ontology .",
            "",
        ]
        for node in self.node_types:
            lines.append(f"kg:{_term(node.name)} a owl:Class .")
        if self.node_types:
            lines.append("")
        for node in self.node_types:
            for property_definition in node.properties:
                lines.extend(
                    [
                        f"kg:{_term(property_definition.name)} a owl:DatatypeProperty ;",
                        f"  rdfs:domain kg:{_term(node.name)} ;",
                        f"  rdfs:range {_xsd_datatype(property_definition.value_type)} .",
                        "",
                    ]
                )
        for relationship in self.relationship_types:
            domain = relationship.source_types[0] if relationship.source_types else "Thing"
            range_type = relationship.target_types[0] if relationship.target_types else "Thing"
            lines.extend(
                [
                    f"kg:{_term(relationship.name)} a owl:ObjectProperty ;",
                    f"  rdfs:domain kg:{_term(domain)} ;",
                    f"  rdfs:range kg:{_term(range_type)} .",
                    "",
                ]
            )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _shacl_property_block(property_definition: PropertyDefinition) -> str:
        constraints = [
            f"sh:path kg:{_term(property_definition.name)}",
            f"sh:datatype {_xsd_datatype(property_definition.value_type)}",
            "sh:maxCount 1",
        ]
        if property_definition.required:
            constraints.append("sh:minCount 1")
        return "[ " + " ; ".join(constraints) + " ]"

    @staticmethod
    def _shacl_relationship_block(relationship: RelationshipTypeDefinition) -> str:
        constraints = [f"sh:path kg:{_term(relationship.name)}"]
        if relationship.target_types:
            constraints.append(f"sh:class kg:{_term(relationship.target_types[0])}")
        return "[ " + " ; ".join(constraints) + " ]"


class SchemaDiffReport(BaseModel):
    """Human-readable schema diff between two proposal runs."""

    model_config = ConfigDict(extra="forbid")

    added_node_types: list[str] = Field(default_factory=list)
    removed_node_types: list[str] = Field(default_factory=list)
    changed_node_types: list[str] = Field(default_factory=list)
    added_relationship_types: list[str] = Field(default_factory=list)
    removed_relationship_types: list[str] = Field(default_factory=list)
    changed_relationship_types: list[str] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return any(
            [
                self.added_node_types,
                self.removed_node_types,
                self.changed_node_types,
                self.added_relationship_types,
                self.removed_relationship_types,
                self.changed_relationship_types,
            ]
        )


def diff_graph_schemas(previous: GraphSchema, current: GraphSchema) -> SchemaDiffReport:
    """Return a deterministic diff report for two graph schemas."""

    previous_nodes = {node.name: node for node in previous.node_types}
    current_nodes = {node.name: node for node in current.node_types}
    previous_relationships = {
        relationship.name: relationship
        for relationship in previous.relationship_types
    }
    current_relationships = {
        relationship.name: relationship
        for relationship in current.relationship_types
    }
    return SchemaDiffReport(
        added_node_types=sorted(set(current_nodes) - set(previous_nodes)),
        removed_node_types=sorted(set(previous_nodes) - set(current_nodes)),
        changed_node_types=sorted(
            name
            for name in set(previous_nodes) & set(current_nodes)
            if previous_nodes[name].model_dump(mode="json")
            != current_nodes[name].model_dump(mode="json")
        ),
        added_relationship_types=sorted(
            set(current_relationships) - set(previous_relationships)
        ),
        removed_relationship_types=sorted(
            set(previous_relationships) - set(current_relationships)
        ),
        changed_relationship_types=sorted(
            name
            for name in set(previous_relationships) & set(current_relationships)
            if previous_relationships[name].model_dump(mode="json")
            != current_relationships[name].model_dump(mode="json")
        ),
    )


class SchemaRegistry:
    """In-memory schema registry used by validators and extractors."""

    def __init__(self, schema: GraphSchema | None = None) -> None:
        self.schema = schema or GraphSchema()
        self._node_types = {item.name: item for item in self.schema.node_types}
        self._relationship_types = {item.name: item for item in self.schema.relationship_types}

    @classmethod
    def permissive(cls) -> SchemaRegistry:
        """Build a registry that allows any type."""

        return cls(GraphSchema(schema_id="permissive", version="v1"))

    @classmethod
    def from_allowed_types(
        cls,
        *,
        node_types: list[str],
        relationship_types: list[str],
        schema_id: str = "configured",
        version: str = "v1",
    ) -> SchemaRegistry:
        """Build a lightweight schema from allowed label lists."""

        return cls(
            GraphSchema(
                schema_id=schema_id,
                version=version,
                node_types=[NodeTypeDefinition(name=name) for name in node_types],
                relationship_types=[RelationshipTypeDefinition(name=name) for name in relationship_types],
            )
        )

    def validate_entity_type(self, entity_type: str) -> bool:
        """Return true when the entity type is allowed."""

        return not self._node_types or entity_type in self._node_types

    def validate_relationship_type(self, relationship_type: str) -> bool:
        """Return true when the relationship type is allowed."""

        return not self._relationship_types or relationship_type in self._relationship_types

    def validate_relationship(
        self,
        relationship_type: str,
        *,
        source_type: str | None = None,
        target_type: str | None = None,
    ) -> bool:
        """Validate relationship label and optional domain/range constraints."""

        definition = self._relationship_types.get(relationship_type)
        if definition is None:
            return not self._relationship_types
        if source_type and definition.source_types and source_type not in definition.source_types:
            return False
        return not (target_type and definition.target_types and target_type not in definition.target_types)

    def required_properties_for(self, entity_type: str) -> set[str]:
        """Return required properties for a node type."""

        definition = self._node_types.get(entity_type)
        return set() if definition is None else definition.required_properties


__all__ = [
    "GraphSchema",
    "JsonType",
    "NodeTypeDefinition",
    "PropertyDefinition",
    "RelationshipTypeDefinition",
    "SchemaDiffReport",
    "SchemaLifecycleStatus",
    "SchemaRegistry",
    "diff_graph_schemas",
]
