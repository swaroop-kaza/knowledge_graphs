"""Semi-structured data mapping adapters."""

from __future__ import annotations

import ast
import json
from typing import Any

from jsonpath_ng import parse
from jsonpath_ng.exceptions import JSONPathError
from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    RelationCandidate,
    SourceRecord,
    normalize_text,
)
from kg_construction.infrastructure.mapping.direct_table_mapper import (
    DirectTableMapper,
    DirectTableMappingConfig,
)
from kg_construction.infrastructure.mapping.rdf_mapping_stubs import ExtensibleMappingStub

JsonDict = dict[str, Any]


class JsonPropertyMapping(BaseModel):
    """Map one JSONPath selector to an entity property."""

    model_config = ConfigDict(extra="forbid")

    property_name: str
    path: str
    required: bool = False


class JsonArrayEdgeMapping(BaseModel):
    """Map object-array JSONPath matches to child entities and edges."""

    model_config = ConfigDict(extra="forbid")

    path: str
    relationship_type: str | None = None
    target_node_type: str | None = None
    target_name_path: str | None = None
    property_paths: list[JsonPropertyMapping] = Field(default_factory=list)


class JsonPathMappingConfig(BaseModel):
    """Configuration for JSONPath-backed JSON-to-graph mapping."""

    model_config = ConfigDict(extra="forbid")

    root_path: str | None = None
    node_type: str = "JSONRecord"
    name_path: str | None = None
    include_scalar_properties: bool = True
    include_array_edges: bool = True
    property_paths: list[JsonPropertyMapping] = Field(default_factory=list)
    array_edges: list[JsonArrayEdgeMapping] = Field(default_factory=list)
    schema_version: str = "v1"


def flatten_json(value: Any, *, prefix: str = "") -> JsonDict:
    """Flatten nested JSON into dotted-key properties."""

    if isinstance(value, dict):
        flattened: JsonDict = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_json(child, prefix=child_prefix))
        return flattened
    if isinstance(value, list):
        return {prefix: [flatten_json(item) if isinstance(item, dict) else item for item in value]}
    return {prefix: value}


class DottedPathJsonMapper:
    """Backwards-compatible lightweight mapper that flattens JSON into dotted keys."""

    name = "dotted_path_json_mapper"
    technique = "dotted_path_json_mapping"

    def __init__(self, node_type: str = "JSONRecord") -> None:
        self.node_type = node_type

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        payload = payload_from_source(source)
        records = payload if isinstance(payload, list) else [payload] if isinstance(payload, dict) else []
        flattened_records = [flatten_json(row) for row in records]
        mapper = DirectTableMapper(
            DirectTableMappingConfig(
                table_name="json",
                node_type=self.node_type,
                id_columns=[],
                name_column=None,
            )
        )
        return mapper.map_source(source.model_copy(update={"records": flattened_records}))


class JsonPathMapper:
    """Map JSON objects to graph candidates using JSONPath selectors and path evidence."""

    name = "jsonpath_mapper"
    technique = "jsonpath_json_query_mapping"

    def __init__(
        self,
        config: JsonPathMappingConfig | None = None,
        *,
        node_type: str = "JSONRecord",
    ) -> None:
        self.config = config or JsonPathMappingConfig(node_type=node_type)

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        payload = payload_from_source(source)
        root_path = self.config.root_path or ("$[*]" if isinstance(payload, list) else "$")
        root_matches = _jsonpath_find(payload, root_path)
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        missing_paths: list[str] = []

        for root_index, root_match in enumerate(root_matches):
            root_value = root_match.value
            root_source_path = normalize_match_path(root_match.full_path)
            root_evidence = json_evidence(source, path=root_source_path, value=root_value)
            properties, property_evidence, property_paths = self._root_properties(
                source,
                root_value,
                root_source_path,
                missing_paths,
            )
            properties.update(
                {
                    "_json_path": root_source_path,
                    "_json_property_paths": property_paths,
                    "_root_index": root_index,
                }
            )
            root_entity = EntityCandidate.build(
                entity_type=self.config.node_type,
                name=self._root_name(root_value, root_index, root_source_path),
                properties=properties,
                evidence=[root_evidence, *property_evidence],
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.config.schema_version,
                provenance_metadata={"json_path": root_source_path},
            )
            entities.append(root_entity)

            child_entities, child_relations = self._array_edges(
                source,
                root_value,
                root_source_path,
                root_entity.candidate_id,
            )
            entities.extend(child_entities)
            relations.extend(child_relations)

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            metadata={
                "mapper": self.name,
                "technique": self.technique,
                "root_path": root_path,
                "property_paths": [item.path for item in self.config.property_paths],
                "array_edge_paths": [
                    item.path for item in self.config.array_edges
                ] if self.config.array_edges else "auto",
                "missing_paths": sorted(set(missing_paths)),
            },
        )

    def _root_properties(
        self,
        source: SourceRecord,
        root_value: Any,
        root_source_path: str,
        missing_paths: list[str],
    ) -> tuple[JsonDict, list[EvidenceRef], JsonDict]:
        properties: JsonDict = {}
        evidence: list[EvidenceRef] = []
        property_paths: JsonDict = {}

        if self.config.property_paths:
            for mapping in self.config.property_paths:
                matches = _jsonpath_find(root_value, mapping.path)
                if not matches:
                    if mapping.required:
                        missing_paths.append(mapping.path)
                    continue
                values: list[Any] = []
                paths: list[str] = []
                for match in matches:
                    source_path = join_json_paths(root_source_path, normalize_match_path(match.full_path))
                    values.append(match.value)
                    paths.append(source_path)
                    evidence.append(json_evidence(source, path=source_path, value=match.value))
                properties[mapping.property_name] = values[0] if len(values) == 1 else values
                property_paths[mapping.property_name] = paths[0] if len(paths) == 1 else paths
            return properties, evidence, property_paths

        if self.config.include_scalar_properties:
            for property_name, source_path, value in iter_scalar_leaves(root_value, root_source_path):
                properties[property_name] = value
                property_paths[property_name] = source_path
                evidence.append(json_evidence(source, path=source_path, value=value))
        return properties, evidence, property_paths

    def _array_edges(
        self,
        source: SourceRecord,
        root_value: Any,
        root_source_path: str,
        root_entity_id: str,
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        if not self.config.include_array_edges:
            return [], []
        if self.config.array_edges:
            return self._configured_array_edges(source, root_value, root_source_path, root_entity_id)
        return self._automatic_array_edges(source, root_value, root_source_path, root_entity_id)

    def _configured_array_edges(
        self,
        source: SourceRecord,
        root_value: Any,
        root_source_path: str,
        root_entity_id: str,
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        for mapping in self.config.array_edges:
            for index, match in enumerate(_jsonpath_find(root_value, mapping.path)):
                if not isinstance(match.value, dict):
                    continue
                item_path = join_json_paths(root_source_path, normalize_match_path(match.full_path))
                entity, relation = self._child_entity_and_relation(
                    source=source,
                    item=match.value,
                    item_path=item_path,
                    index=index,
                    root_entity_id=root_entity_id,
                    relationship_type=mapping.relationship_type or relationship_type_from_path(item_path),
                    target_node_type=mapping.target_node_type or entity_type_from_path(item_path),
                    target_name_path=mapping.target_name_path,
                    property_paths=mapping.property_paths,
                )
                entities.append(entity)
                relations.append(relation)
        return entities, relations

    def _automatic_array_edges(
        self,
        source: SourceRecord,
        root_value: Any,
        root_source_path: str,
        root_entity_id: str,
    ) -> tuple[list[EntityCandidate], list[RelationCandidate]]:
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        for index, item_path, item in iter_object_array_items(root_value, root_source_path):
            entity, relation = self._child_entity_and_relation(
                source=source,
                item=item,
                item_path=item_path,
                index=index,
                root_entity_id=root_entity_id,
                relationship_type=relationship_type_from_path(item_path),
                target_node_type=entity_type_from_path(item_path),
            )
            entities.append(entity)
            relations.append(relation)
        return entities, relations

    def _child_entity_and_relation(
        self,
        *,
        source: SourceRecord,
        item: JsonDict,
        item_path: str,
        index: int,
        root_entity_id: str,
        relationship_type: str,
        target_node_type: str,
        target_name_path: str | None = None,
        property_paths: list[JsonPropertyMapping] | None = None,
    ) -> tuple[EntityCandidate, RelationCandidate]:
        item_evidence = json_evidence(source, path=item_path, value=item)
        properties, scalar_evidence, property_source_paths = self._child_properties(
            source,
            item,
            item_path,
            property_paths or [],
        )
        properties.update(
            {
                "_json_path": item_path,
                "_json_property_paths": property_source_paths,
                "_array_index": index,
            }
        )
        entity = EntityCandidate.build(
            entity_type=target_node_type,
            name=self._child_name(item, index, item_path, target_name_path),
            properties=properties,
            evidence=[item_evidence, *scalar_evidence],
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.config.schema_version,
            provenance_metadata={"json_path": item_path},
        )
        relation = RelationCandidate.build(
            source=root_entity_id,
            relation_type=relationship_type,
            target=entity.candidate_id,
            properties={
                "source_path": item_path,
                "array_index": index,
                "target_node_type": target_node_type,
            },
            evidence=[item_evidence],
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.config.schema_version,
            provenance_metadata={"json_path": item_path},
        )
        return entity, relation

    def _child_properties(
        self,
        source: SourceRecord,
        item: JsonDict,
        item_path: str,
        property_paths: list[JsonPropertyMapping],
    ) -> tuple[JsonDict, list[EvidenceRef], JsonDict]:
        properties: JsonDict = {}
        evidence: list[EvidenceRef] = []
        source_paths: JsonDict = {}
        if property_paths:
            for mapping in property_paths:
                matches = _jsonpath_find(item, mapping.path)
                if not matches:
                    continue
                values: list[Any] = []
                paths: list[str] = []
                for match in matches:
                    source_path = join_json_paths(item_path, normalize_match_path(match.full_path))
                    values.append(match.value)
                    paths.append(source_path)
                    evidence.append(json_evidence(source, path=source_path, value=match.value))
                properties[mapping.property_name] = values[0] if len(values) == 1 else values
                source_paths[mapping.property_name] = paths[0] if len(paths) == 1 else paths
            return properties, evidence, source_paths

        for property_name, source_path, value in iter_scalar_leaves(item, item_path):
            properties[property_name] = value
            source_paths[property_name] = source_path
            evidence.append(json_evidence(source, path=source_path, value=value))
        return properties, evidence, source_paths

    def _root_name(self, root_value: Any, root_index: int, root_source_path: str) -> str:
        if self.config.name_path:
            matches = _jsonpath_find(root_value, self.config.name_path)
            if matches:
                return str(matches[0].value)
        if isinstance(root_value, dict):
            for key in ("name", "title", "id", "email"):
                if root_value.get(key):
                    return str(root_value[key])
        return f"{self.config.node_type}:{root_source_path}:{root_index}"

    @staticmethod
    def _child_name(
        item: JsonDict,
        index: int,
        item_path: str,
        target_name_path: str | None = None,
    ) -> str:
        if target_name_path:
            matches = _jsonpath_find(item, target_name_path)
            if matches:
                return str(matches[0].value)
        for key in ("name", "title", "id", "code", "email"):
            if item.get(key):
                return str(item[key])
        return f"{entity_type_from_path(item_path)}:{index}"


def payload_from_source(source: SourceRecord) -> Any:
    """Return the original JSON payload from a source record."""

    if "json_payload" in source.metadata:
        return source.metadata["json_payload"]
    if source.records:
        return source.records
    if source.content:
        try:
            return json.loads(source.content)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(source.content)
            except (SyntaxError, ValueError):
                return []
    return []


def _jsonpath_find(value: Any, expression: str):
    try:
        return parse(expression).find(value)
    except JSONPathError as exc:
        raise ValueError(f"Invalid JSONPath expression {expression!r}: {exc}") from exc


def json_evidence(source: SourceRecord, *, path: str, value: Any) -> EvidenceRef:
    """Build evidence for an exact JSON source path."""

    return EvidenceRef.build(
        source,
        location={"kind": "jsonpath", "path": path},
        text=json_text(value),
        metadata={"json_path": path},
    )


def json_text(value: Any) -> str:
    """Serialize JSON values for compact evidence text."""

    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def normalize_match_path(path: Any) -> str:
    """Normalize jsonpath-ng path strings into predictable JSONPath notation."""

    raw_path = str(path).replace("(", "").replace(")", "").replace(".[", "[")
    if raw_path == "$":
        return "$"
    if raw_path.startswith("$"):
        return raw_path
    if raw_path.startswith("["):
        return f"${raw_path}"
    return f"$.{raw_path}"


def join_json_paths(root_path: str, child_path: str) -> str:
    """Join a root JSONPath and a child path matched inside that root value."""

    if child_path == "$":
        return root_path
    if root_path == "$":
        return child_path
    if child_path.startswith("$."):
        return f"{root_path}.{child_path[2:]}"
    if child_path.startswith("$["):
        return f"{root_path}{child_path[1:]}"
    return f"{root_path}.{child_path.lstrip('$').lstrip('.')}"


def iter_scalar_leaves(value: Any, root_path: str, property_prefix: str = ""):
    """Yield dotted property names, exact JSONPaths, and scalar values."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{root_path}.{key}" if root_path != "$" else f"$.{key}"
            child_property = f"{property_prefix}.{key}" if property_prefix else str(key)
            yield from iter_scalar_leaves(child, child_path, child_property)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            child_path = f"{root_path}[{index}]"
            child_property = f"{property_prefix}[{index}]" if property_prefix else f"[{index}]"
            yield from iter_scalar_leaves(child, child_path, child_property)
        return
    yield property_prefix, root_path, value


def iter_object_array_items(value: Any, root_path: str):
    """Yield nested object-array items with their exact source paths."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{root_path}.{key}" if root_path != "$" else f"$.{key}"
            yield from iter_object_array_items(child, child_path)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            item_path = f"{root_path}[{index}]"
            if isinstance(child, dict):
                yield index, item_path, child
            yield from iter_object_array_items(child, item_path)


def relationship_type_from_path(path: str) -> str:
    """Create a stable relationship type from the array segment in a JSONPath."""

    key = path.rstrip("]").split(".")[-1].split("[")[0]
    normalized = singularize(key).upper()
    return f"HAS_{normalized}" if normalized else "HAS_JSON_ITEM"


def entity_type_from_path(path: str) -> str:
    """Create a readable entity type from the array segment in a JSONPath."""

    key = path.rstrip("]").split(".")[-1].split("[")[0]
    return singularize(key).title().replace("_", "").replace("-", "") or "JSONItem"


def _first_value(row: JsonDict, *keys: str) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
        dotted = _dotted_value(row, key)
        if dotted not in (None, ""):
            return dotted
    return None


def _dotted_value(row: JsonDict, key: str) -> Any:
    current: Any = row
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current


def singularize(value: str) -> str:
    """Small deterministic singularizer for generated labels."""

    if value.endswith("ies") and len(value) > 3:
        return f"{value[:-3]}y"
    if value.endswith("s") and len(value) > 1:
        return value[:-1]
    return value


class XMLPathMapper(ExtensibleMappingStub):
    name = "xmlpath_mapper"
    technique = "xpath_xml_mapping"
    dependency_hint = "Configure XPath selectors for entity, property, and relationship extraction."


class APIResponseMapper(ExtensibleMappingStub):
    name = "api_response_mapper"
    technique = "api_response_mapping"
    dependency_hint = "Configure response paths, pagination cursors, and entity keys."


class EventGraphMapper:
    """Map event payload rows into observability/event graph candidates."""

    name = "event_graph_mapper"
    technique = "event_payload_to_event_graph"

    def __init__(self, *, schema_version: str = "v1") -> None:
        self.schema_version = schema_version

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        entity_index: dict[tuple[str, str], EntityCandidate] = {}
        event_by_id: dict[str, EntityCandidate] = {}
        span_by_id: dict[str, EntityCandidate] = {}

        for index, row in enumerate(source.records):
            evidence = EvidenceRef.build(
                source,
                location={"kind": "event_payload_row", "row_index": index},
                text=repr(row),
                metadata={"mapper": self.name},
            )
            event_id = _first_value(row, "event_id", "event.id", "id", "message_id") or f"event:{index}"
            event_type = str(_first_value(row, "event_type", "event.type", "type", "name") or "Event")
            event = EntityCandidate.build(
                entity_type="Event",
                name=str(event_id),
                properties={
                    **row,
                    "event_id": event_id,
                    "event_type": event_type,
                    "timestamp": _first_value(row, "timestamp", "time", "event.time", "observed_at"),
                },
                evidence=[evidence],
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
            )
            entities.append(event)
            event_by_id[str(event_id)] = event

            for entity_type, keys, relation_type in [
                ("Actor", ("actor", "actor.id", "user", "principal"), "PERFORMED_BY"),
                ("Object", ("object", "object.id", "resource", "target"), "ACTED_ON"),
                ("System", ("service", "service.name", "app", "application"), "EMITTED_BY"),
                ("Trace", ("trace_id", "trace.id", "traceId"), "IN_TRACE"),
                ("Span", ("span_id", "span.id", "spanId"), "IN_SPAN"),
            ]:
                value = _first_value(row, *keys)
                if value in (None, ""):
                    continue
                target = self._get_or_create(entity_index, entity_type, str(value), source, evidence)
                if target not in entities:
                    entities.append(target)
                if entity_type == "Span":
                    span_by_id[str(value)] = target
                relations.append(
                    RelationCandidate.build(
                        source=event.candidate_id,
                        relation_type=relation_type,
                        target=target.candidate_id,
                        properties={"event_id": event_id, "source_key": keys[0]},
                        evidence=[evidence],
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.schema_version,
                    )
                )

        for index, row in enumerate(source.records):
            event_id = str(_first_value(row, "event_id", "event.id", "id", "message_id") or f"event:{index}")
            event_candidate = event_by_id.get(event_id)
            if event_candidate is None:
                continue
            evidence = EvidenceRef.build(
                source,
                location={"kind": "event_payload_row", "row_index": index, "relation": "correlation"},
                text=repr(row),
            )
            parent_span_id = _first_value(row, "parent_span_id", "parent.span.id", "parentSpanId")
            span_id = _first_value(row, "span_id", "span.id", "spanId")
            if parent_span_id and span_id and str(parent_span_id) in span_by_id and str(span_id) in span_by_id:
                relations.append(
                    RelationCandidate.build(
                        source=span_by_id[str(parent_span_id)].candidate_id,
                        relation_type="PARENT_OF_SPAN",
                        target=span_by_id[str(span_id)].candidate_id,
                        evidence=[evidence],
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.schema_version,
                    )
                )
            caused_by = _first_value(row, "caused_by", "causedBy", "parent_event_id")
            if caused_by and str(caused_by) in event_by_id:
                relations.append(
                    RelationCandidate.build(
                        source=event_by_id[str(caused_by)].candidate_id,
                        relation_type="CAUSED",
                        target=event_candidate.candidate_id,
                        evidence=[evidence],
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.schema_version,
                    )
                )

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            metadata={
                "mapper": self.name,
                "technique": self.technique,
                "event_count": len(source.records),
                "trace_span_correlation": True,
                "causality_edges": len([relation for relation in relations if relation.relation_type == "CAUSED"]),
            },
        )

    def _get_or_create(
        self,
        entity_index: dict[tuple[str, str], EntityCandidate],
        entity_type: str,
        name: str,
        source: SourceRecord,
        evidence: EvidenceRef,
    ) -> EntityCandidate:
        key = (entity_type, normalize_text(name))
        if key in entity_index:
            return entity_index[key]
        entity = EntityCandidate.build(
            entity_type=entity_type,
            name=name,
            properties={"source_name": source.source_name},
            evidence=[evidence],
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.schema_version,
        )
        entity_index[key] = entity
        return entity


class LogEventMapper:
    """Map parsed log rows into event graph candidates."""

    name = "log_event_mapper"
    technique = "log_template_to_event_graph"

    def __init__(self, *, schema_version: str = "v1") -> None:
        self.schema_version = schema_version

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        entities: list[EntityCandidate] = []
        relations: list[RelationCandidate] = []
        entity_index: dict[tuple[str, str], EntityCandidate] = {}

        for index, row in enumerate(source.records):
            evidence = log_row_evidence(source, row, index)
            event = EntityCandidate.build(
                entity_type="LogEvent",
                name=self._event_name(row, index),
                properties={**row, "event_type": self._event_type(row)},
                evidence=[evidence],
                confidence=1.0,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
            )
            entities.append(event)

            for entity_type, value, relation_type in self._related_values(row):
                target = self._get_or_create_related_entity(
                    source,
                    entity_index,
                    entity_type=entity_type,
                    name=value,
                    evidence=evidence,
                )
                if target not in entities:
                    entities.append(target)
                relations.append(
                    RelationCandidate.build(
                        source=event.candidate_id,
                        relation_type=relation_type,
                        target=target.candidate_id,
                        properties={
                            "line_number": row.get("line_number"),
                            "parser": row.get("parser"),
                        },
                        evidence=[evidence],
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.schema_version,
                    )
                )

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            metadata={
                "mapper": self.name,
                "technique": self.technique,
                "event_count": len(source.records),
            },
        )

    @staticmethod
    def _event_name(row: JsonDict, index: int) -> str:
        for key in ("incident_id", "event.id", "request_id"):
            if row.get(key):
                return str(row[key])
        timestamp = row.get("timestamp")
        message = str(row.get("message", "")).strip()
        if timestamp and message:
            return f"{timestamp}: {message[:80]}"
        return f"log_event:{row.get('line_number', index + 1)}"

    @staticmethod
    def _event_type(row: JsonDict) -> str:
        parser = str(row.get("parser", "generic"))
        if parser.startswith("apache"):
            status = str(row.get("status", ""))
            return "HttpErrorEvent" if status.startswith(("4", "5")) else "HttpAccessEvent"
        level = str(row.get("level", "INFO")).upper()
        return f"{level.title()}LogEvent"

    @staticmethod
    def _related_values(row: JsonDict) -> list[tuple[str, str, str]]:
        related: list[tuple[str, str, str]] = []
        for key in ("service", "service.name", "app", "application"):
            if row.get(key):
                related.append(("System", str(row[key]), "EMITTED_BY"))
                break
        if row.get("logger"):
            related.append(("Logger", str(row["logger"]), "EMITTED_BY"))
        if row.get("client_ip"):
            related.append(("IPAddress", str(row["client_ip"]), "OBSERVED_CLIENT"))
        if row.get("url"):
            related.append(("Endpoint", str(row["url"]), "REQUESTED_ENDPOINT"))
        if row.get("level"):
            related.append(("LogLevel", str(row["level"]).upper(), "HAS_LEVEL"))
        if row.get("template_id"):
            related.append(("LogTemplate", str(row["template_id"]), "MATCHES_TEMPLATE"))
        return [(entity_type, value, relation_type) for entity_type, value, relation_type in related if value]

    def _get_or_create_related_entity(
        self,
        source: SourceRecord,
        entity_index: dict[tuple[str, str], EntityCandidate],
        *,
        entity_type: str,
        name: str,
        evidence: EvidenceRef,
    ) -> EntityCandidate:
        key = (entity_type, normalize_text(name))
        if key in entity_index:
            return entity_index[key]
        entity = EntityCandidate.build(
            entity_type=entity_type,
            name=name,
            properties={"source_name": source.source_name},
            evidence=[evidence],
            confidence=1.0,
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.schema_version,
        )
        entity_index[key] = entity
        return entity


class HTMLMetadataMapper:
    """Map HTML/page metadata into document, metadata, link, and JSON-LD graph candidates."""

    name = "html_metadata_mapper"
    technique = "html_schema_org_jsonld_mapping"

    def __init__(self, *, schema_version: str = "v1") -> None:
        self.schema_version = schema_version

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        html_metadata = source.metadata.get("html_metadata", {})
        if not isinstance(html_metadata, dict):
            html_metadata = {}
        title = str(html_metadata.get("title") or source.metadata.get("title") or source.source_name)
        document_evidence = html_metadata_evidence(source, key="document", value=title)
        document = EntityCandidate.build(
            entity_type="WebPage" if source.source_type == "webpage" else "HTMLDocument",
            name=title,
            properties={
                "source_name": source.source_name,
                "title": title,
                "description": html_metadata.get("description", ""),
                "canonical_url": html_metadata.get("canonical_url", ""),
                "source_type": source.source_type,
            },
            evidence=[document_evidence],
            confidence=1.0,
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.schema_version,
        )
        entities = [document]
        relations: list[RelationCandidate] = []

        for index, tag in enumerate(_list_of_dicts(html_metadata.get("meta_tags"))):
            evidence = html_metadata_evidence(
                source,
                key=str(tag.get("key", f"meta[{index}]")),
                value=str(tag.get("content", "")),
                kind="html_meta_tag",
            )
            meta_entity = EntityCandidate.build(
                entity_type="HTMLMetaTag",
                name=f"{tag.get('key', 'meta')}: {str(tag.get('content', ''))[:80]}",
                properties={**tag, "index": index},
                evidence=[evidence],
                confidence=1.0,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
            )
            entities.append(meta_entity)
            relations.append(
                self._relation(document, meta_entity, "HAS_METADATA", evidence, {"metadata_key": tag.get("key")})
            )

        for index, link in enumerate(_list_of_dicts(html_metadata.get("links"))):
            href = str(link.get("href", ""))
            if not href:
                continue
            evidence = html_metadata_evidence(source, key=f"link[{index}]", value=href, kind="html_link")
            link_entity = EntityCandidate.build(
                entity_type="HTMLLink",
                name=href,
                properties={**link, "index": index},
                evidence=[evidence],
                confidence=1.0,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
            )
            entities.append(link_entity)
            relations.append(
                self._relation(
                    document,
                    link_entity,
                    "HAS_CANONICAL_LINK" if "canonical" in str(link.get("rel", "")).casefold() else "LINKS_TO",
                    evidence,
                    {"rel": link.get("rel", "")},
                )
            )

        for index, item in enumerate(_list_of_dicts(html_metadata.get("json_ld"))):
            jsonld_type = item.get("@type", item.get("type", "Thing"))
            name = str(item.get("name") or item.get("headline") or f"jsonld:{index}")
            evidence = html_metadata_evidence(source, key=f"json_ld[{index}]", value=json_text(item), kind="json_ld")
            jsonld_entity = EntityCandidate.build(
                entity_type="SchemaOrgEntity",
                name=name,
                properties={
                    "jsonld_type": jsonld_type,
                    "payload": item,
                    "index": index,
                },
                evidence=[evidence],
                confidence=1.0,
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.schema_version,
            )
            entities.append(jsonld_entity)
            relations.append(
                self._relation(document, jsonld_entity, "HAS_STRUCTURED_DATA", evidence, {"jsonld_type": jsonld_type})
            )

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            metadata={
                "mapper": self.name,
                "technique": self.technique,
                "meta_tag_count": len(_list_of_dicts(html_metadata.get("meta_tags"))),
                "link_count": len(_list_of_dicts(html_metadata.get("links"))),
                "json_ld_count": len(_list_of_dicts(html_metadata.get("json_ld"))),
            },
        )

    def _relation(
        self,
        source_entity: EntityCandidate,
        target_entity: EntityCandidate,
        relation_type: str,
        evidence: EvidenceRef,
        properties: JsonDict | None = None,
    ) -> RelationCandidate:
        return RelationCandidate.build(
            source=source_entity.candidate_id,
            relation_type=relation_type,
            target=target_entity.candidate_id,
            properties=properties or {},
            evidence=[evidence],
            extractor_name=self.name,
            technique=self.technique,
            schema_version=self.schema_version,
        )


def log_row_evidence(source: SourceRecord, row: JsonDict, index: int) -> EvidenceRef:
    """Create evidence for one parsed log row."""

    line_number = int(row.get("line_number", index + 1) or index + 1)
    return EvidenceRef.build(
        source,
        location={"kind": "log_line", "line_number": line_number, "parser": row.get("parser", "generic")},
        text=str(row.get("raw") or row.get("message") or row),
        metadata={"line_number": line_number, "parser": row.get("parser", "generic")},
    )


def html_metadata_evidence(
    source: SourceRecord,
    *,
    key: str,
    value: str,
    kind: str = "html_metadata",
) -> EvidenceRef:
    """Create evidence for one parsed HTML metadata value."""

    return EvidenceRef.build(
        source,
        location={"kind": kind, "key": key},
        text=value,
        metadata={"metadata_key": key},
    )


def _list_of_dicts(value: object) -> list[JsonDict]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


__all__ = [
    "APIResponseMapper",
    "DottedPathJsonMapper",
    "EventGraphMapper",
    "HTMLMetadataMapper",
    "JsonArrayEdgeMapping",
    "JsonPathMapper",
    "JsonPathMappingConfig",
    "JsonPropertyMapping",
    "LogEventMapper",
    "XMLPathMapper",
    "flatten_json",
]
