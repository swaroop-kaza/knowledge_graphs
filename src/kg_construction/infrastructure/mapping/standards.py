"""Executable standards-based mapping support for R2RML, RML, and CSVW."""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.evidence import cell_evidence, row_evidence
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    GraphFact,
    RelationCandidate,
    SourceRecord,
    stable_id,
)
from kg_construction.infrastructure.loaders.common import AdapterNotConfigured
from kg_construction.infrastructure.mapping.semi_structured import (
    json_evidence,
    payload_from_source,
)

JsonDict = dict[str, Any]
TermRole = Literal["subject", "predicate", "object", "graph"]
TermType = Literal["iri", "literal", "blank_node"]
ReferenceFormulation = Literal["csv", "jsonpath", "xpath", "sql2008table", "sql2008query", "row"]

RR_NS = "http://www.w3.org/ns/r2rml#"
RML_LEGACY_NS = "http://semweb.mmlab.be/ns/rml#"
RML_NS = "http://w3id.org/rml/"
QL_LEGACY_NS = "http://semweb.mmlab.be/ns/ql#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"
CSVW_NS = "http://www.w3.org/ns/csvw#"

RDF_TYPE = f"{RDF_NS}type"
XSD_STRING = f"{XSD_NS}string"

_TEMPLATE_TOKEN_RE = re.compile(r"{([^{}]+)}")
_SAFE_LABEL_RE = re.compile(r"[^A-Za-z0-9_]+")


class TermMapConfig(BaseModel):
    """Common R2RML/RML term map subset used by executable mappers."""

    model_config = ConfigDict(extra="forbid")

    constant: str | None = None
    template: str | None = None
    column: str | None = None
    reference: str | None = None
    term_type: TermType | None = None
    datatype: str | None = None
    language: str | None = None


class PredicateObjectMapConfig(BaseModel):
    """Executable predicate-object map."""

    model_config = ConfigDict(extra="forbid")

    predicate_map: TermMapConfig
    object_map: TermMapConfig = Field(default_factory=TermMapConfig)
    graph_map: TermMapConfig | None = None
    parent_triples_map: str | None = None
    join_conditions: list[JoinConditionConfig] = Field(default_factory=list)


class JoinConditionConfig(BaseModel):
    """R2RML/RML referencing object map join condition."""

    model_config = ConfigDict(extra="forbid")

    child: str
    parent: str


class TriplesMapConfig(BaseModel):
    """Executable triples map for R2RML or RML."""

    model_config = ConfigDict(extra="forbid")

    mapping_id: str
    subject_map: TermMapConfig
    predicate_object_maps: list[PredicateObjectMapConfig] = Field(default_factory=list)
    class_iris: list[str] = Field(default_factory=list)
    logical_table: str | None = None
    sql_query: str | None = None
    logical_source: str | None = None
    iterator: str | None = None
    reference_formulation: ReferenceFormulation = "row"


class R2RMLMappingConfig(BaseModel):
    """R2RML executable mapping document."""

    model_config = ConfigDict(extra="forbid")

    base_iri: str = "urn:r2rml:"
    schema_version: str = "v1"
    triples_maps: list[TriplesMapConfig] = Field(default_factory=list)


class RMLMappingConfig(BaseModel):
    """RML executable mapping document."""

    model_config = ConfigDict(extra="forbid")

    base_iri: str = "urn:rml:"
    schema_version: str = "v1"
    triples_maps: list[TriplesMapConfig] = Field(default_factory=list)


class CSVWColumnConfig(BaseModel):
    """CSVW column metadata subset."""

    model_config = ConfigDict(extra="allow")

    name: str
    titles: str | list[str] | dict[str, Any] | None = None
    datatype: str | dict[str, Any] | None = None
    propertyUrl: str | None = None
    valueUrl: str | None = None
    aboutUrl: str | None = None
    suppressOutput: bool = False
    required: bool = False
    null: str | list[str] | None = None
    separator: str | None = None
    ordered: bool | None = None
    virtual: bool = False


class CSVWForeignKeyConfig(BaseModel):
    """CSVW foreign key subset."""

    model_config = ConfigDict(extra="forbid")

    columnReference: str | list[str]
    reference: dict[str, Any]


class CSVWTableSchemaConfig(BaseModel):
    """CSVW tableSchema subset."""

    model_config = ConfigDict(extra="allow")

    columns: list[CSVWColumnConfig] = Field(default_factory=list)
    primaryKey: str | list[str] | None = None
    foreignKeys: list[CSVWForeignKeyConfig] = Field(default_factory=list)


class CSVWTableConfig(BaseModel):
    """CSVW table metadata subset."""

    model_config = ConfigDict(extra="allow")

    url: str | None = None
    aboutUrl: str | None = None
    dialect: CSVWDialectConfig | None = None
    tableSchema: CSVWTableSchemaConfig = Field(default_factory=CSVWTableSchemaConfig)


class CSVWMappingConfig(BaseModel):
    """CSVW metadata document subset."""

    model_config = ConfigDict(extra="allow")

    base_iri: str = "urn:csvw:"
    schema_version: str = "v1"
    dialect: CSVWDialectConfig | None = None
    tables: list[CSVWTableConfig] = Field(default_factory=list)

    @classmethod
    def from_metadata(cls, metadata: JsonDict, *, base_iri: str = "urn:csvw:") -> CSVWMappingConfig:
        """Normalize a CSVW JSON metadata document into mapping config."""

        payload = dict(metadata)
        if "tables" not in payload and ("url" in payload or "tableSchema" in payload):
            payload = {"tables": [payload]}
        payload.setdefault("base_iri", base_iri)
        return cls.model_validate(payload)


class CSVWDialectConfig(BaseModel):
    """CSVW dialect subset for parsing CSV files before mapping."""

    model_config = ConfigDict(extra="allow")

    delimiter: str = ","
    quoteChar: str | None = '"'
    doubleQuote: bool = True
    skipRows: int = 0
    skipColumns: int = 0
    header: bool = True
    headerRowCount: int = 1
    trim: bool | str = False
    commentPrefix: str | None = None
    encoding: str = "utf-8"
    lineTerminators: list[str] | str | None = None


class CSVWValidationFinding(BaseModel):
    """CSVW validation finding."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    message: str
    row_index: int | None = None
    column: str | None = None
    severity: Literal["error", "warning"] = "error"


class CSVWValidationReport(BaseModel):
    """CSVW validation report."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    checked_rows: int
    findings: list[CSVWValidationFinding] = Field(default_factory=list)


class CSVWDialectParser:
    """Parse CSV text/files using CSVW dialect metadata."""

    name = "csvw_dialect_parser"

    def __init__(self, dialect: CSVWDialectConfig | None = None) -> None:
        self.dialect = dialect or CSVWDialectConfig()

    def parse_text(self, text: str) -> list[JsonDict]:
        lines = text.splitlines()
        if self.dialect.commentPrefix:
            lines = [
                line
                for line in lines
                if not line.lstrip().startswith(str(self.dialect.commentPrefix))
            ]
        if self.dialect.skipRows:
            lines = lines[self.dialect.skipRows :]
        reader = csv.reader(
            io.StringIO("\n".join(lines)),
            delimiter=self.dialect.delimiter,
            quotechar=self.dialect.quoteChar,
            doublequote=self.dialect.doubleQuote,
        )
        rows = [list(row) for row in reader]
        if self.dialect.skipColumns:
            rows = [row[self.dialect.skipColumns :] for row in rows]
        if not rows:
            return []
        if self.dialect.header and self.dialect.headerRowCount:
            header_rows = rows[: self.dialect.headerRowCount]
            headers = _csvw_headers(header_rows)
            data_rows = rows[self.dialect.headerRowCount :]
        else:
            width = max((len(row) for row in rows), default=0)
            headers = [f"column_{index + 1}" for index in range(width)]
            data_rows = rows
        parsed: list[JsonDict] = []
        for row in data_rows:
            values = [_csvw_trim(value, self.dialect.trim) for value in row]
            parsed.append({header: values[index] if index < len(values) else "" for index, header in enumerate(headers)})
        return parsed

    def parse_file(self, path: str | Path) -> list[JsonDict]:
        return self.parse_text(Path(path).read_text(encoding=self.dialect.encoding))


class CSVWValidator:
    """Validate tabular rows against CSVW table schema and dialect expectations."""

    name = "csvw_validator"

    def validate_rows(self, rows: list[JsonDict], table: CSVWTableConfig) -> CSVWValidationReport:
        findings: list[CSVWValidationFinding] = []
        schema = table.tableSchema
        columns = schema.columns
        column_names = [column.name for column in columns]
        for row_index, row in enumerate(rows):
            for column in columns:
                value = row.get(column.name)
                null_values = set(_as_list(column.null))
                is_null = value in (None, "") or str(value) in null_values
                if column.required and is_null:
                    findings.append(
                        CSVWValidationFinding(
                            rule="required",
                            message=f"Required column {column.name!r} is missing.",
                            row_index=row_index,
                            column=column.name,
                        )
                    )
                if column.name not in row and not column.virtual:
                    findings.append(
                        CSVWValidationFinding(
                            rule="column_present",
                            message=f"Column {column.name!r} is not present in row.",
                            row_index=row_index,
                            column=column.name,
                        )
                    )
            extra_columns = set(row) - set(column_names)
            for column_name in sorted(extra_columns):
                findings.append(
                    CSVWValidationFinding(
                        rule="unexpected_column",
                        message=f"Column {column_name!r} is not described by CSVW metadata.",
                        row_index=row_index,
                        column=column_name,
                        severity="warning",
                    )
                )
        primary_keys = _as_list(schema.primaryKey)
        if primary_keys:
            seen: dict[tuple[str, ...], int] = {}
            for row_index, row in enumerate(rows):
                key = tuple(str(row.get(column, "")) for column in primary_keys)
                if any(value == "" for value in key):
                    findings.append(
                        CSVWValidationFinding(
                            rule="primary_key_not_null",
                            message=f"Primary key {primary_keys!r} is incomplete.",
                            row_index=row_index,
                            severity="error",
                        )
                    )
                    continue
                if key in seen:
                    findings.append(
                        CSVWValidationFinding(
                            rule="primary_key_unique",
                            message=f"Duplicate primary key {key!r}.",
                            row_index=row_index,
                            severity="error",
                        )
                    )
                seen[key] = row_index
        return CSVWValidationReport(
            valid=not any(finding.severity == "error" for finding in findings),
            checked_rows=len(rows),
            findings=findings,
        )


@dataclass(frozen=True, slots=True)
class MappedTerm:
    """One generated RDF term."""

    value: str
    object_type: Literal["iri", "literal"]
    datatype: str | None = None
    language: str | None = None
    reference: str | None = None
    raw_value: Any = None


class R2RMLMapper:
    """Map relational rows to RDF facts using an executable R2RML subset."""

    name = "r2rml_mapper"
    technique = "r2rml_relational_to_rdf"

    def __init__(self, config: R2RMLMappingConfig | None = None) -> None:
        self.config = config

    @classmethod
    def from_turtle(cls, turtle: str, *, base_iri: str = "urn:r2rml:") -> R2RMLMapper:
        """Build an R2RML mapper from a Turtle mapping document."""

        return cls(_parse_r2rml_turtle(turtle, base_iri=base_iri))

    @classmethod
    def from_turtle_file(cls, path: str | Path, *, base_iri: str = "urn:r2rml:") -> R2RMLMapper:
        """Build an R2RML mapper from a Turtle mapping file."""

        return cls.from_turtle(Path(path).read_text(encoding="utf-8"), base_iri=base_iri)

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        if self.config is None or not self.config.triples_maps:
            raise AdapterNotConfigured("R2RML mapping requires a configured triples map document.")
        return _merge_batches(
            [
                _map_rows_with_triples_maps(
                    source=source,
                    rows=source.records,
                    triples_maps=[
                        triples_map
                        for triples_map in self.config.triples_maps
                        if _logical_table_matches(source, triples_map)
                    ],
                    mapper_name=self.name,
                    technique=self.technique,
                    schema_version=self.config.schema_version,
                    standard="R2RML",
                    reference_getter=lambda row, reference: _row_reference(row, reference),
                    context_evidence=lambda row_index, row, _path: row_evidence(source, row_index, row),
                    row_label=lambda triples_map: _entity_label_from_triples_map(triples_map, "RelationalRow"),
                ),
                self._referencing_object_map_batch([source]),
            ]
        )

    def map_sources(self, sources: list[SourceRecord]) -> ConstructionBatch:
        """Map multiple relational sources and resolve referencing object-map joins."""

        if self.config is None or not self.config.triples_maps:
            raise AdapterNotConfigured("R2RML mapping requires a configured triples map document.")
        batches = [self.map_source(source) for source in sources]
        return _merge_batches([*batches, self._referencing_object_map_batch(sources)])

    def _referencing_object_map_batch(self, sources: list[SourceRecord]) -> ConstructionBatch:
        if self.config is None:
            return ConstructionBatch(source_records=sources)

        def contexts_for(source: SourceRecord, triples_map: TriplesMapConfig) -> list[_RMLContext]:
            return [_RMLContext(row=row) for row in source.records if _logical_table_matches(source, triples_map)]

        def get_reference(row: JsonDict, reference: str, _formulation: ReferenceFormulation | None = None) -> Any:
            return _row_reference(row, reference)

        def get_evidence(
            source: SourceRecord,
            row_index: int,
            row: JsonDict,
            _path: str | None,
        ) -> EvidenceRef:
            return row_evidence(source, row_index, row)

        return _referencing_object_map_batch(
            sources=sources,
            triples_maps=self.config.triples_maps,
            mapper_name=self.name,
            technique=self.technique,
            schema_version=self.config.schema_version,
            standard="R2RML",
            contexts_for=contexts_for,
            reference_getter=get_reference,
            context_evidence=get_evidence,
        )


class RMLMapper:
    """Map CSV, JSON, XML, or row-shaped sources to RDF facts using RML triples maps."""

    name = "rml_mapper"
    technique = "rml_heterogeneous_to_rdf"

    def __init__(self, config: RMLMappingConfig | None = None) -> None:
        self.config = config

    @classmethod
    def from_turtle(cls, turtle: str, *, base_iri: str = "urn:rml:") -> RMLMapper:
        """Build an RML mapper from a Turtle mapping document."""

        return cls(_parse_rml_turtle(turtle, base_iri=base_iri))

    @classmethod
    def from_turtle_file(cls, path: str | Path, *, base_iri: str = "urn:rml:") -> RMLMapper:
        """Build an RML mapper from a Turtle mapping file."""

        return cls.from_turtle(Path(path).read_text(encoding="utf-8"), base_iri=base_iri)

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        if self.config is None or not self.config.triples_maps:
            raise AdapterNotConfigured("RML mapping requires a configured triples map document.")
        batches: list[ConstructionBatch] = []
        for triples_map in self.config.triples_maps:
            contexts = _rml_contexts(source, triples_map)
            contexts_by_index = dict(enumerate(contexts))
            formulation = triples_map.reference_formulation

            def get_reference(
                row: JsonDict,
                reference: str,
                *,
                bound_formulation: ReferenceFormulation = formulation,
            ) -> Any:
                return _rml_reference(row, reference, bound_formulation)

            def get_evidence(
                row_index: int,
                row: JsonDict,
                path: str | None,
                *,
                bound_contexts: dict[int, _RMLContext] = contexts_by_index,
            ) -> EvidenceRef:
                return _rml_evidence(source, row_index, row, path, bound_contexts)

            batches.append(
                _map_rows_with_triples_maps(
                    source=source,
                    rows=[context.row for context in contexts],
                    triples_maps=[triples_map],
                    mapper_name=self.name,
                    technique=self.technique,
                    schema_version=self.config.schema_version,
                    standard="RML",
                    reference_getter=get_reference,
                    context_evidence=get_evidence,
                    row_label=lambda item: _entity_label_from_triples_map(item, "RMLRecord"),
                )
            )
        return _merge_batches(batches)

    def map_sources(self, sources: list[SourceRecord]) -> ConstructionBatch:
        """Map multiple heterogeneous sources and resolve RML referencing joins."""

        if self.config is None or not self.config.triples_maps:
            raise AdapterNotConfigured("RML mapping requires a configured triples map document.")
        batches = [self.map_source(source) for source in sources]

        def contexts_for(source: SourceRecord, triples_map: TriplesMapConfig) -> list[_RMLContext]:
            return _rml_contexts(source, triples_map)

        def get_reference(
            row: JsonDict,
            reference: str,
            formulation: ReferenceFormulation | None = None,
        ) -> Any:
            return _rml_reference(row, reference, formulation or "row")

        def get_evidence(
            source: SourceRecord,
            row_index: int,
            row: JsonDict,
            path: str | None,
        ) -> EvidenceRef:
            return EvidenceRef.build(
                source,
                location={"kind": "rml_join", "row_index": row_index, "path": path},
                text=repr(row),
                metadata={"standard": "RML"},
            )

        join_batch = _referencing_object_map_batch(
            sources=sources,
            triples_maps=self.config.triples_maps,
            mapper_name=self.name,
            technique=self.technique,
            schema_version=self.config.schema_version,
            standard="RML",
            contexts_for=contexts_for,
            reference_getter=get_reference,
            context_evidence=get_evidence,
        )
        return _merge_batches([*batches, join_batch])


class CSVWMapper:
    """Map tabular data to RDF facts using CSVW metadata."""

    name = "csvw_mapper"
    technique = "csvw_metadata_to_rdf"

    def __init__(
        self,
        metadata: JsonDict | CSVWMappingConfig | str | Path | None = None,
        *,
        base_iri: str = "urn:csvw:",
    ) -> None:
        self.config: CSVWMappingConfig | None
        if isinstance(metadata, CSVWMappingConfig):
            self.config = metadata
        elif isinstance(metadata, str | Path):
            self.config = CSVWMappingConfig.from_metadata(
                json.loads(Path(metadata).read_text(encoding="utf-8")),
                base_iri=base_iri,
            )
        elif isinstance(metadata, dict):
            self.config = CSVWMappingConfig.from_metadata(metadata, base_iri=base_iri)
        else:
            self.config = None

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        if self.config is None:
            raise AdapterNotConfigured("CSVW mapping requires a CSVW metadata JSON document.")
        table = _csvw_table_for_source(self.config, source)
        if table is None:
            raise AdapterNotConfigured(f"CSVW metadata has no table matching source {source.source_name!r}.")
        rows = source.records
        schema = table.tableSchema
        validation_report = CSVWValidator().validate_rows(rows, table)
        primary_keys = _as_list(schema.primaryKey)
        columns = schema.columns or [
            CSVWColumnConfig(name=column)
            for column in sorted({key for row in rows for key in row})
        ]
        entity_type = _safe_label(Path(table.url or source.source_name).stem or "CSVWRow")
        subject_by_key: dict[tuple[str, ...], str] = {}
        entity_by_key: dict[tuple[str, ...], str] = {}
        entities: list[EntityCandidate] = []
        facts: list[GraphFact] = []
        relations = []
        missing_required: list[dict[str, object]] = []

        for row_index, row in enumerate(rows):
            subject_iri = _csvw_subject_iri(self.config, table, row, row_index, primary_keys)
            row_key = _csvw_row_key(row, primary_keys, row_index)
            subject_by_key[row_key] = subject_iri
            evidence = row_evidence(source, row_index, row)
            entity = EntityCandidate.build(
                entity_type=entity_type,
                name=subject_iri,
                properties={
                    **row,
                    "rdf_subject_iri": subject_iri,
                    "csvw_table_url": table.url,
                    "csvw_primary_key": primary_keys,
                    "_row_index": row_index,
                },
                evidence=[evidence],
                extractor_name=self.name,
                technique=self.technique,
                schema_version=self.config.schema_version,
                provenance_metadata={"standard": "CSVW", "table_url": table.url},
            )
            entities.append(entity)
            entity_by_key[row_key] = entity.candidate_id

            class_iri = _iri_join(self.config.base_iri, entity_type)
            facts.append(
                _rdf_fact(
                    source=source,
                    subject=subject_iri,
                    predicate=RDF_TYPE,
                    object_value=class_iri,
                    object_type="iri",
                    evidence=evidence,
                    mapper_name=self.name,
                    technique=self.technique,
                    schema_version=self.config.schema_version,
                    standard="CSVW",
                    mapping_id=f"csvw:{table.url or source.source_name}",
                    properties={"csvw_role": "row_type"},
                )
            )
            for column in columns:
                value = row.get(column.name)
                if value in (None, ""):
                    if column.required:
                        missing_required.append(
                            {"row_index": row_index, "column": column.name, "source": source.source_name}
                        )
                    continue
                if column.suppressOutput:
                    continue
                predicate_iri = _csvw_predicate_iri(self.config, column)
                object_value, object_type = _csvw_object_value(column, row, value)
                column_evidence = cell_evidence(source, row_index=row_index, column=column.name, value=value)
                facts.append(
                    _rdf_fact(
                        source=source,
                        subject=subject_iri,
                        predicate=predicate_iri,
                        object_value=object_value,
                        object_type=object_type,
                        datatype=_csvw_datatype_iri(column.datatype) if object_type == "literal" else None,
                        evidence=column_evidence,
                        mapper_name=self.name,
                        technique=self.technique,
                        schema_version=self.config.schema_version,
                        standard="CSVW",
                        mapping_id=f"csvw:{table.url or source.source_name}:{column.name}",
                        properties={
                            "csvw_column": column.name,
                            "csvw_datatype": column.datatype,
                            "csvw_propertyUrl": column.propertyUrl,
                            "csvw_valueUrl": column.valueUrl,
                        },
                    )
                )

        for foreign_key in schema.foreignKeys:
            fk_relations, fk_facts = _csvw_foreign_key_outputs(
                source=source,
                rows=rows,
                table=table,
                config=self.config,
                foreign_key=foreign_key,
                subject_by_key=subject_by_key,
                entity_by_key=entity_by_key,
                entity_type=entity_type,
                schema_version=self.config.schema_version,
            )
            relations.extend(fk_relations)
            facts.extend(fk_facts)

        return ConstructionBatch(
            source_records=[source],
            entities=entities,
            relations=relations,
            facts=facts,
            metadata={
                "mapper": self.name,
                "standard": "CSVW",
                "table_url": table.url,
                "primary_keys": primary_keys,
                "column_count": len(columns),
                "missing_required_columns": missing_required,
                "csvw_validation": validation_report.model_dump(mode="json"),
                "generated_facts": len(facts),
            },
        )


@dataclass(frozen=True, slots=True)
class _RMLContext:
    row: JsonDict
    path: str | None = None


def _map_rows_with_triples_maps(
    *,
    source: SourceRecord,
    rows: list[JsonDict],
    triples_maps: list[TriplesMapConfig],
    mapper_name: str,
    technique: str,
    schema_version: str,
    standard: str,
    reference_getter: Callable[[JsonDict, str], Any],
    context_evidence: Callable[[int, JsonDict, str | None], EvidenceRef],
    row_label: Callable[[TriplesMapConfig], str],
) -> ConstructionBatch:
    entities: list[EntityCandidate] = []
    facts: list[GraphFact] = []
    if not triples_maps:
        return ConstructionBatch(
            source_records=[source],
            metadata={"mapper": mapper_name, "standard": standard, "generated_facts": 0},
        )

    for triples_map in triples_maps:
        for row_index, row in enumerate(rows):
            subject = _evaluate_term(
                triples_map.subject_map,
                row,
                role="subject",
                reference_getter=reference_getter,
            )
            if subject is None:
                continue
            evidence = context_evidence(row_index, row, None)
            entity_type = row_label(triples_map)
            entities.append(
                EntityCandidate.build(
                    entity_type=entity_type,
                    name=subject.value,
                    properties={
                        **row,
                        "rdf_subject_iri": subject.value,
                        "mapping_id": triples_map.mapping_id,
                        "standard": standard,
                        "_row_index": row_index,
                    },
                    evidence=[evidence],
                    extractor_name=mapper_name,
                    technique=technique,
                    schema_version=schema_version,
                    provenance_metadata={"standard": standard, "mapping_id": triples_map.mapping_id},
                )
            )
            for class_iri in triples_map.class_iris:
                facts.append(
                    _rdf_fact(
                        source=source,
                        subject=subject.value,
                        predicate=RDF_TYPE,
                        object_value=class_iri,
                        object_type="iri",
                        evidence=evidence,
                        mapper_name=mapper_name,
                        technique=technique,
                        schema_version=schema_version,
                        standard=standard,
                        mapping_id=triples_map.mapping_id,
                        properties={"term_map_role": "rr:class"},
                    )
                )
            for predicate_object_map in triples_map.predicate_object_maps:
                predicate = _evaluate_term(
                    predicate_object_map.predicate_map,
                    row,
                    role="predicate",
                    reference_getter=reference_getter,
                )
                obj = _evaluate_term(
                    predicate_object_map.object_map,
                    row,
                    role="object",
                    reference_getter=reference_getter,
                )
                if predicate is None or obj is None:
                    continue
                graph = None
                if predicate_object_map.graph_map is not None:
                    graph_term = _evaluate_term(
                        predicate_object_map.graph_map,
                        row,
                        role="graph",
                        reference_getter=reference_getter,
                    )
                    graph = graph_term.value if graph_term is not None else None
                object_evidence = _term_evidence(
                    source=source,
                    row=row,
                    row_index=row_index,
                    reference=obj.reference,
                    raw_value=obj.raw_value,
                    fallback=evidence,
                    standard=standard,
                    mapping_id=triples_map.mapping_id,
                )
                facts.append(
                    _rdf_fact(
                        source=source,
                        subject=subject.value,
                        predicate=predicate.value,
                        object_value=obj.value,
                        object_type=obj.object_type,
                        datatype=obj.datatype,
                        evidence=object_evidence,
                        mapper_name=mapper_name,
                        technique=technique,
                        schema_version=schema_version,
                        standard=standard,
                        mapping_id=triples_map.mapping_id,
                        graph=graph,
                        properties={
                            "subject_map": triples_map.subject_map.model_dump(exclude_none=True),
                            "predicate_map": predicate_object_map.predicate_map.model_dump(exclude_none=True),
                            "object_map": predicate_object_map.object_map.model_dump(exclude_none=True),
                            "source_reference": obj.reference,
                            "raw_value": obj.raw_value,
                        },
                    )
                )

    return ConstructionBatch(
        source_records=[source],
        entities=entities,
        facts=facts,
        metadata={
            "mapper": mapper_name,
            "standard": standard,
            "triples_map_count": len(triples_maps),
            "generated_entities": len(entities),
            "generated_facts": len(facts),
        },
    )


def _referencing_object_map_batch(
    *,
    sources: list[SourceRecord],
    triples_maps: list[TriplesMapConfig],
    mapper_name: str,
    technique: str,
    schema_version: str,
    standard: str,
    contexts_for: Callable[[SourceRecord, TriplesMapConfig], list[_RMLContext]],
    reference_getter: Callable[[JsonDict, str, ReferenceFormulation | None], Any],
    context_evidence: Callable[[SourceRecord, int, JsonDict, str | None], EvidenceRef],
) -> ConstructionBatch:
    indexed: dict[str, list[tuple[SourceRecord, int, _RMLContext, MappedTerm]]] = {}
    for triples_map in triples_maps:
        rows_for_map: list[tuple[SourceRecord, int, _RMLContext, MappedTerm]] = []
        for source in sources:
            if standard == "R2RML" and not _logical_table_matches(source, triples_map):
                continue
            if standard == "RML" and triples_map.logical_source:
                source_tokens = {source.source_name, Path(source.source_name).name, Path(source.source_name).stem}
                if triples_map.logical_source not in source_tokens:
                    continue
            subject_reference_getter = _reference_getter_for_formulation(
                reference_getter,
                triples_map.reference_formulation,
            )
            for row_index, context in enumerate(contexts_for(source, triples_map)):
                subject = _evaluate_term(
                    triples_map.subject_map,
                    context.row,
                    role="subject",
                    reference_getter=subject_reference_getter,
                )
                if subject is not None:
                    rows_for_map.append((source, row_index, context, subject))
        indexed[triples_map.mapping_id] = rows_for_map

    facts: list[GraphFact] = []
    relations: list[RelationCandidate] = []
    for child_map in triples_maps:
        child_rows = indexed.get(child_map.mapping_id, [])
        if not child_rows:
            continue
        for predicate_object_map in child_map.predicate_object_maps:
            if not predicate_object_map.parent_triples_map:
                continue
            parent_rows = indexed.get(predicate_object_map.parent_triples_map, [])
            if not parent_rows:
                continue
            predicate_reference_getter = _reference_getter_for_formulation(
                reference_getter,
                child_map.reference_formulation,
            )
            for child_source, child_index, child_context, child_subject in child_rows:
                predicate = _evaluate_term(
                    predicate_object_map.predicate_map,
                    child_context.row,
                    role="predicate",
                    reference_getter=predicate_reference_getter,
                )
                if predicate is None:
                    continue
                child_evidence = context_evidence(child_source, child_index, child_context.row, child_context.path)
                for parent_source, parent_index, parent_context, parent_subject in parent_rows:
                    if not _join_conditions_match(
                        predicate_object_map.join_conditions,
                        child_context.row,
                        parent_context.row,
                        child_formulation=child_map.reference_formulation,
                        parent_formulation=_triples_map_by_id(triples_maps, predicate_object_map.parent_triples_map).reference_formulation,
                        reference_getter=reference_getter,
                    ):
                        continue
                    parent_evidence = context_evidence(
                        parent_source,
                        parent_index,
                        parent_context.row,
                        parent_context.path,
                    )
                    facts.append(
                        _rdf_fact(
                            source=child_source,
                            subject=child_subject.value,
                            predicate=predicate.value,
                            object_value=parent_subject.value,
                            object_type="iri",
                            evidence=child_evidence,
                            mapper_name=mapper_name,
                            technique=technique,
                            schema_version=schema_version,
                            standard=standard,
                            mapping_id=child_map.mapping_id,
                            properties={
                                "referencing_object_map": True,
                                "parent_triples_map": predicate_object_map.parent_triples_map,
                                "join_conditions": [
                                    item.model_dump(mode="json")
                                    for item in predicate_object_map.join_conditions
                                ],
                                "parent_evidence_id": parent_evidence.evidence_id,
                            },
                        )
                    )
                    relations.append(
                        RelationCandidate.build(
                            source=stable_id("rdf_subject", child_subject.value),
                            relation_type=_safe_label(_iri_local_name(predicate.value)).upper(),
                            target=stable_id("rdf_subject", parent_subject.value),
                            properties={
                                "rdf_subject": child_subject.value,
                                "rdf_predicate": predicate.value,
                                "rdf_object": parent_subject.value,
                                "referencing_object_map": True,
                                "parent_triples_map": predicate_object_map.parent_triples_map,
                            },
                            evidence=[child_evidence, parent_evidence],
                            extractor_name=mapper_name,
                            technique=technique,
                            schema_version=schema_version,
                            provenance_metadata={"standard": standard, "mapping_id": child_map.mapping_id},
                        )
                    )
    return ConstructionBatch(
        source_records=sources,
        relations=relations,
        facts=facts,
        metadata={
            "mapper": mapper_name,
            "standard": standard,
            "referencing_object_map_facts": len(facts),
            "referencing_object_map_relations": len(relations),
        },
    )


def _join_conditions_match(
    conditions: list[JoinConditionConfig],
    child_row: JsonDict,
    parent_row: JsonDict,
    *,
    child_formulation: ReferenceFormulation,
    parent_formulation: ReferenceFormulation,
    reference_getter: Callable[[JsonDict, str, ReferenceFormulation | None], Any],
) -> bool:
    if not conditions:
        return True
    for condition in conditions:
        child_value = reference_getter(child_row, condition.child, child_formulation)
        parent_value = reference_getter(parent_row, condition.parent, parent_formulation)
        if child_value in (None, "") or parent_value in (None, ""):
            return False
        if str(child_value) != str(parent_value):
            return False
    return True


def _reference_getter_for_formulation(
    reference_getter: Callable[[JsonDict, str, ReferenceFormulation | None], Any],
    formulation: ReferenceFormulation,
) -> Callable[[JsonDict, str], Any]:
    def get_reference(row: JsonDict, reference: str) -> Any:
        return reference_getter(row, reference, formulation)

    return get_reference


def _triples_map_by_id(triples_maps: list[TriplesMapConfig], mapping_id: str) -> TriplesMapConfig:
    for triples_map in triples_maps:
        if triples_map.mapping_id == mapping_id:
            return triples_map
    raise KeyError(mapping_id)


def _rdf_fact(
    *,
    source: SourceRecord,
    subject: str,
    predicate: str,
    object_value: str,
    object_type: Literal["iri", "literal"],
    evidence: EvidenceRef,
    mapper_name: str,
    technique: str,
    schema_version: str,
    standard: str,
    mapping_id: str,
    datatype: str | None = None,
    graph: str | None = None,
    properties: JsonDict | None = None,
) -> GraphFact:
    _ = source
    return GraphFact.build(
        subject=subject,
        predicate=predicate,
        object=object_value,
        evidence=[evidence],
        extractor_name=mapper_name,
        technique=technique,
        schema_version=schema_version,
        properties={
            "rdf_subject_iri": subject,
            "rdf_predicate_iri": predicate,
            "rdf_object": object_value,
            "rdf_object_type": object_type,
            "rdf_datatype": datatype,
            "rdf_graph": graph,
            "standard": standard,
            "mapping_id": mapping_id,
            **(properties or {}),
        },
        provenance_metadata={"standard": standard, "mapping_id": mapping_id},
    )


def _evaluate_term(
    term_map: TermMapConfig,
    row: JsonDict,
    *,
    role: TermRole,
    reference_getter: Callable[[JsonDict, str], Any],
) -> MappedTerm | None:
    reference = term_map.reference or term_map.column
    raw_value: Any = None
    if term_map.constant is not None:
        value = str(term_map.constant)
        term_type = term_map.term_type or ("literal" if term_map.datatype or term_map.language else "iri")
        return _mapped_term(value, role=role, term_type=term_type, term_map=term_map, raw_value=value)
    if term_map.template is not None:
        rendered = _render_template(term_map.template, lambda name: reference_getter(row, name))
        if not rendered:
            return None
        term_type = term_map.term_type or ("literal" if role == "object" and term_map.datatype else "iri")
        return _mapped_term(rendered, role=role, term_type=term_type, term_map=term_map, raw_value=rendered)
    if reference:
        raw_value = reference_getter(row, reference)
        if raw_value in (None, ""):
            return None
        term_type = term_map.term_type or ("literal" if role == "object" else "iri")
        return _mapped_term(
            str(raw_value),
            role=role,
            term_type=term_type,
            term_map=term_map,
            reference=reference,
            raw_value=raw_value,
        )
    return None


def _mapped_term(
    value: str,
    *,
    role: TermRole,
    term_type: TermType,
    term_map: TermMapConfig,
    reference: str | None = None,
    raw_value: Any = None,
) -> MappedTerm:
    if role in {"subject", "predicate", "graph"}:
        object_type: Literal["iri", "literal"] = "iri"
    elif term_type == "literal":
        object_type = "literal"
    else:
        object_type = "iri"
    if term_type == "blank_node":
        value = f"urn:kg:blank:{quote(value, safe='')}"
        object_type = "iri"
    return MappedTerm(
        value=value,
        object_type=object_type,
        datatype=term_map.datatype if object_type == "literal" else None,
        language=term_map.language if object_type == "literal" else None,
        reference=reference,
        raw_value=raw_value,
    )


def _render_template(template: str, value_getter: Callable[[str], Any]) -> str:
    missing_value = False

    def replace(match: re.Match[str]) -> str:
        nonlocal missing_value
        value = value_getter(match.group(1))
        if value in (None, ""):
            missing_value = True
            return ""
        return quote(str(value), safe="")

    rendered = _TEMPLATE_TOKEN_RE.sub(replace, template)
    return "" if missing_value else rendered


def _row_reference(row: JsonDict, reference: str) -> Any:
    if reference in row:
        return row[reference]
    return _dotted_reference(row, reference)


def _rml_reference(row: JsonDict, reference: str, formulation: ReferenceFormulation) -> Any:
    if formulation == "jsonpath" and reference.startswith("$"):
        try:
            from jsonpath_ng import parse

            matches = parse(reference).find(row)
        except Exception:
            matches = []
        if not matches:
            return None
        values = [match.value for match in matches]
        return values[0] if len(values) == 1 else values
    return _row_reference(row, reference)


def _dotted_reference(row: JsonDict, reference: str) -> Any:
    current: Any = row
    for part in reference.replace("/", ".").strip(".").split("."):
        if not part:
            continue
        if isinstance(current, dict):
            if part in current:
                current = current[part]
                continue
            return None
        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if 0 <= index < len(current):
                current = current[index]
                continue
            return None
        return None
    return current


def _term_evidence(
    *,
    source: SourceRecord,
    row: JsonDict,
    row_index: int,
    reference: str | None,
    raw_value: Any,
    fallback: EvidenceRef,
    standard: str,
    mapping_id: str,
) -> EvidenceRef:
    if reference and reference in row:
        return cell_evidence(source, row_index=row_index, column=reference, value=raw_value)
    if reference:
        return EvidenceRef.build(
            source,
            location={"kind": f"{standard.lower()}_reference", "row_index": row_index, "reference": reference},
            text=str(raw_value),
            metadata={"standard": standard, "mapping_id": mapping_id},
        )
    return fallback


def _rml_contexts(source: SourceRecord, triples_map: TriplesMapConfig) -> list[_RMLContext]:
    formulation = triples_map.reference_formulation
    if formulation in {"csv", "row", "sql2008table", "sql2008query"}:
        return [_RMLContext(row=row) for row in source.records]
    if formulation == "jsonpath":
        payload = payload_from_source(source)
        iterator = triples_map.iterator or ("$[*]" if isinstance(payload, list) else "$")
        from jsonpath_ng import parse

        matches = parse(iterator).find(payload)
        return [
            _RMLContext(row=match.value if isinstance(match.value, dict) else {"value": match.value}, path=str(match.full_path))
            for match in matches
        ]
    if formulation == "xpath":
        if source.records:
            row_paths = source.metadata.get("row_paths", [])
            return [
                _RMLContext(row=row, path=str(row_paths[index]) if index < len(row_paths) else None)
                for index, row in enumerate(source.records)
            ]
        if source.content:
            return _xml_contexts_from_text(source.content, triples_map.iterator)
    return []


def _xml_contexts_from_text(text: str, iterator: str | None) -> list[_RMLContext]:
    root = ET.fromstring(text)
    path = iterator or f".//{_xml_local(root.tag)}"
    search_path = path
    if search_path.startswith("/"):
        search_path = "." + search_path
    elements = root.findall(search_path)
    if not elements and _xml_local(root.tag) in path:
        elements = [root]
    return [
        _RMLContext(row=_flatten_xml_element(element), path=path)
        for element in elements
    ]


def _flatten_xml_element(element: ET.Element, prefix: str = "") -> JsonDict:
    row: JsonDict = {}
    name = _xml_local(element.tag)
    current = f"{prefix}.{name}" if prefix else name
    if element.text and element.text.strip():
        row[current] = element.text.strip()
        row[name] = element.text.strip()
    for child in list(element):
        row.update(_flatten_xml_element(child, current))
    return row


def _xml_local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":", 1)[-1]


def _rml_evidence(
    source: SourceRecord,
    row_index: int,
    row: JsonDict,
    path: str | None,
    context_lookup: dict[int, _RMLContext],
) -> EvidenceRef:
    context = context_lookup.get(row_index)
    if context and context.path:
        if source.source_type == "json":
            return json_evidence(source, path=context.path, value=row)
        return EvidenceRef.build(
            source,
            location={"kind": "rml_iterator", "row_index": row_index, "path": context.path},
            text=repr(row),
            metadata={"standard": "RML"},
        )
    if path:
        return EvidenceRef.build(
            source,
            location={"kind": "rml_iterator", "row_index": row_index, "path": path},
            text=repr(row),
            metadata={"standard": "RML"},
        )
    return row_evidence(source, row_index, row)


def _logical_table_matches(source: SourceRecord, triples_map: TriplesMapConfig) -> bool:
    if not triples_map.logical_table and not triples_map.sql_query:
        return True
    if triples_map.sql_query:
        return True
    table = str(triples_map.logical_table or "").casefold()
    source_table_names = {
        str(source.metadata.get("table", "")).casefold(),
        str(source.metadata.get("table_name", "")).casefold(),
        Path(source.source_name.split("#")[-1]).stem.casefold(),
        source.source_name.split("#")[-1].casefold(),
    }
    return table in source_table_names


def _entity_label_from_triples_map(triples_map: TriplesMapConfig, fallback: str) -> str:
    if triples_map.class_iris:
        return _safe_label(_iri_local_name(triples_map.class_iris[0]))
    if triples_map.logical_table:
        return _safe_label(triples_map.logical_table)
    if triples_map.logical_source:
        return _safe_label(Path(triples_map.logical_source).stem)
    return fallback


def _parse_r2rml_turtle(turtle: str, *, base_iri: str) -> R2RMLMappingConfig:
    graph, rr, _rml_terms = _parse_mapping_graph(turtle)
    triples_maps: list[TriplesMapConfig] = []
    for triples_map_node in _triples_map_nodes(graph, rr):
        logical_table = _first_resource(graph, triples_map_node, rr.logicalTable)
        subject_map_node = _first_resource(graph, triples_map_node, rr.subjectMap)
        if subject_map_node is None:
            subject_shortcut = _first_any(graph, triples_map_node, [rr.subject])
            if subject_shortcut is None:
                continue
            subject_map = _constant_term_map(subject_shortcut, role="subject")
            class_iris: list[str] = []
        else:
            subject_map = _term_map_from_node(graph, subject_map_node, role="subject")
            class_iris = [str(value) for value in graph.objects(subject_map_node, rr["class"])]
        triples_maps.append(
            TriplesMapConfig(
                mapping_id=_node_id(triples_map_node),
                logical_table=_literal_from_node(graph, logical_table, rr.tableName) if logical_table is not None else None,
                sql_query=_literal_from_node(graph, logical_table, rr.sqlQuery) if logical_table is not None else None,
                subject_map=subject_map,
                class_iris=class_iris,
                predicate_object_maps=_predicate_object_maps(graph, triples_map_node, rr),
                reference_formulation="row",
            )
        )
    return R2RMLMappingConfig(base_iri=base_iri, triples_maps=triples_maps)


def _parse_rml_turtle(turtle: str, *, base_iri: str) -> RMLMappingConfig:
    graph, rr, rml_terms = _parse_mapping_graph(turtle)
    triples_maps: list[TriplesMapConfig] = []
    for triples_map_node in _triples_map_nodes(graph, rr, logical_source_terms=rml_terms["logicalSource"]):
        logical_source_node = _first_any(graph, triples_map_node, rml_terms["logicalSource"])
        subject_map_node = _first_resource(graph, triples_map_node, rr.subjectMap)
        if logical_source_node is None or subject_map_node is None:
            continue
        source_value = _literal_from_any(graph, logical_source_node, rml_terms["source"])
        iterator = _literal_from_any(graph, logical_source_node, rml_terms["iterator"])
        formulation = _reference_formulation(
            _resource_from_any(graph, logical_source_node, rml_terms["referenceFormulation"])
        )
        subject_map = _term_map_from_node(graph, subject_map_node, role="subject", rml_terms=rml_terms)
        class_iris = [str(value) for value in graph.objects(subject_map_node, rr["class"])]
        triples_maps.append(
            TriplesMapConfig(
                mapping_id=_node_id(triples_map_node),
                logical_source=source_value,
                iterator=iterator,
                reference_formulation=formulation,
                subject_map=subject_map,
                class_iris=class_iris,
                predicate_object_maps=_predicate_object_maps(graph, triples_map_node, rr, rml_terms=rml_terms),
            )
        )
    return RMLMappingConfig(base_iri=base_iri, triples_maps=triples_maps)


def _parse_mapping_graph(turtle: str):
    from rdflib import Graph, Namespace

    graph = Graph()
    graph.parse(data=turtle, format="turtle")
    rr = Namespace(RR_NS)
    rml_legacy = Namespace(RML_LEGACY_NS)
    rml = Namespace(RML_NS)
    rml_terms = {
        "logicalSource": [rml_legacy.logicalSource, rml.logicalSource],
        "source": [rml_legacy.source, rml.source],
        "iterator": [rml_legacy.iterator, rml.iterator],
        "referenceFormulation": [rml_legacy.referenceFormulation, rml.referenceFormulation],
        "reference": [rml_legacy.reference, rml.reference],
    }
    return graph, rr, rml_terms


def _triples_map_nodes(graph: Any, rr: Any, logical_source_terms: list[Any] | None = None) -> list[Any]:
    nodes = set(graph.subjects(rr.logicalTable, None)) | set(graph.subjects(rr.subjectMap, None))
    for term in logical_source_terms or []:
        nodes.update(graph.subjects(term, None))
    return sorted(nodes, key=str)


def _predicate_object_maps(
    graph: Any,
    triples_map_node: Any,
    rr: Any,
    *,
    rml_terms: dict[str, list[Any]] | None = None,
) -> list[PredicateObjectMapConfig]:
    results: list[PredicateObjectMapConfig] = []
    for pom_node in graph.objects(triples_map_node, rr.predicateObjectMap):
        predicate_maps = [_constant_term_map(value, role="predicate") for value in graph.objects(pom_node, rr.predicate)]
        predicate_maps.extend(
            _term_map_from_node(graph, node, role="predicate", rml_terms=rml_terms)
            for node in graph.objects(pom_node, rr.predicateMap)
        )
        object_maps = [_constant_term_map(value, role="object") for value in graph.objects(pom_node, rr.object)]
        object_maps.extend(
            _term_map_from_node(graph, node, role="object", rml_terms=rml_terms)
            for node in graph.objects(pom_node, rr.objectMap)
        )
        graph_maps = [_constant_term_map(value, role="graph") for value in graph.objects(pom_node, rr.graph)]
        graph_maps.extend(
            _term_map_from_node(graph, node, role="graph", rml_terms=rml_terms)
            for node in graph.objects(pom_node, rr.graphMap)
        )
        graph_map = graph_maps[0] if graph_maps else None
        for predicate_map in predicate_maps:
            for object_map in object_maps:
                results.append(
                    PredicateObjectMapConfig(
                        predicate_map=predicate_map,
                        object_map=object_map,
                        graph_map=graph_map,
                    )
                )
    return results


def _term_map_from_node(
    graph: Any,
    node: Any,
    *,
    role: TermRole,
    rml_terms: dict[str, list[Any]] | None = None,
) -> TermMapConfig:
    from rdflib import Namespace

    rr = Namespace(RR_NS)
    constant = _first_any(graph, node, [rr.constant])
    if constant is not None:
        return _constant_term_map(constant, role=role)
    term_type = _term_type(_resource_from_any(graph, node, [rr.termType]))
    datatype = _resource_from_any(graph, node, [rr.datatype])
    language = _literal_from_any(graph, node, [rr.language])
    reference_terms = rml_terms["reference"] if rml_terms else []
    return TermMapConfig(
        template=_literal_from_any(graph, node, [rr.template]),
        column=_literal_from_any(graph, node, [rr.column]),
        reference=_literal_from_any(graph, node, reference_terms),
        term_type=term_type,
        datatype=datatype,
        language=language,
    )


def _constant_term_map(value: Any, *, role: TermRole) -> TermMapConfig:
    from rdflib import Literal, URIRef

    if isinstance(value, URIRef):
        return TermMapConfig(constant=str(value), term_type="iri")
    if isinstance(value, Literal):
        return TermMapConfig(
            constant=str(value),
            term_type="literal",
            datatype=str(value.datatype) if value.datatype else None,
            language=str(value.language) if value.language else None,
        )
    return TermMapConfig(constant=str(value), term_type="literal" if role == "object" else "iri")


def _first_any(graph: Any, subject: Any, predicates: Iterable[Any]) -> Any | None:
    for predicate in predicates:
        for value in graph.objects(subject, predicate):
            return value
    return None


def _first_resource(graph: Any, subject: Any, predicate: Any) -> Any | None:
    for value in graph.objects(subject, predicate):
        return value
    return None


def _literal_from_node(graph: Any, subject: Any | None, predicate: Any) -> str | None:
    if subject is None:
        return None
    return _literal_from_any(graph, subject, [predicate])


def _literal_from_any(graph: Any, subject: Any | None, predicates: Iterable[Any]) -> str | None:
    if subject is None:
        return None
    value = _first_any(graph, subject, predicates)
    return str(value) if value is not None else None


def _resource_from_any(graph: Any, subject: Any | None, predicates: Iterable[Any]) -> str | None:
    if subject is None:
        return None
    value = _first_any(graph, subject, predicates)
    return str(value) if value is not None else None


def _term_type(value: str | None) -> TermType | None:
    if value is None:
        return None
    lowered = _iri_local_name(value).casefold()
    if lowered == "iri":
        return "iri"
    if lowered == "literal":
        return "literal"
    if lowered == "blanknode":
        return "blank_node"
    return None


def _reference_formulation(value: str | None) -> ReferenceFormulation:
    if value is None:
        return "row"
    lowered = _iri_local_name(value).casefold()
    if lowered in {"csv", "tsv"}:
        return "csv"
    if lowered == "jsonpath":
        return "jsonpath"
    if lowered == "xpath":
        return "xpath"
    if lowered == "sql2008query":
        return "sql2008query"
    if lowered == "sql2008table":
        return "sql2008table"
    return "row"


def _node_id(node: Any) -> str:
    text = str(node)
    if "#" in text:
        return text.rsplit("#", 1)[-1]
    if "/" in text:
        return text.rstrip("/").rsplit("/", 1)[-1]
    return stable_id("triples_map", text)


def _iri_local_name(value: str) -> str:
    return value.rstrip("/#").rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _safe_label(value: str) -> str:
    cleaned = _SAFE_LABEL_RE.sub(" ", value).strip()
    if not cleaned:
        return "Record"
    return "".join(part[:1].upper() + part[1:] for part in cleaned.split())


def _csvw_table_for_source(config: CSVWMappingConfig, source: SourceRecord) -> CSVWTableConfig | None:
    if len(config.tables) == 1:
        return config.tables[0]
    source_names = {
        Path(source.source_name).name,
        source.source_name,
        Path(source.source_name.split("#")[-1]).name,
    }
    for table in config.tables:
        if table.url in source_names or (table.url and Path(table.url).name in source_names):
            return table
    return None


def _csvw_subject_iri(
    config: CSVWMappingConfig,
    table: CSVWTableConfig,
    row: JsonDict,
    row_index: int,
    primary_keys: list[str],
) -> str:
    if table.aboutUrl:
        return _render_template(table.aboutUrl, lambda name: row.get(name, ""))
    if primary_keys:
        key = "-".join(quote(str(row.get(column, "")), safe="") for column in primary_keys)
    else:
        key = str(row_index)
    table_name = Path(table.url or "table").stem or "table"
    return _iri_join(config.base_iri, table_name, key)


def _csvw_row_key(row: JsonDict, primary_keys: list[str], row_index: int) -> tuple[str, ...]:
    if not primary_keys:
        return (str(row_index),)
    return tuple(str(row.get(column, "")) for column in primary_keys)


def _csvw_predicate_iri(config: CSVWMappingConfig, column: CSVWColumnConfig) -> str:
    if column.propertyUrl:
        return column.propertyUrl
    return _iri_join(config.base_iri, "property", column.name)


def _csvw_object_value(
    column: CSVWColumnConfig,
    row: JsonDict,
    value: Any,
) -> tuple[str, Literal["iri", "literal"]]:
    if column.valueUrl:
        return _render_template(column.valueUrl, lambda name: row.get(name, "")), "iri"
    return str(value), "literal"


def _csvw_datatype_iri(datatype: str | dict[str, Any] | None) -> str:
    if datatype is None:
        return XSD_STRING
    base = str(datatype.get("base", "string")) if isinstance(datatype, dict) else str(datatype)
    if base.startswith(("http://", "https://", "urn:")):
        return base
    mapping = {
        "string": XSD_STRING,
        "integer": f"{XSD_NS}integer",
        "int": f"{XSD_NS}integer",
        "number": f"{XSD_NS}decimal",
        "decimal": f"{XSD_NS}decimal",
        "double": f"{XSD_NS}double",
        "boolean": f"{XSD_NS}boolean",
        "date": f"{XSD_NS}date",
        "datetime": f"{XSD_NS}dateTime",
        "dateTime": f"{XSD_NS}dateTime",
        "anyURI": f"{XSD_NS}anyURI",
    }
    return mapping.get(base, f"{XSD_NS}{base}")


def _csvw_foreign_key_outputs(
    *,
    source: SourceRecord,
    rows: list[JsonDict],
    table: CSVWTableConfig,
    config: CSVWMappingConfig,
    foreign_key: CSVWForeignKeyConfig,
    subject_by_key: dict[tuple[str, ...], str],
    entity_by_key: dict[tuple[str, ...], str],
    entity_type: str,
    schema_version: str,
) -> tuple[list[RelationCandidate], list[GraphFact]]:
    column_refs = _as_list(foreign_key.columnReference)
    ref_column_refs = _as_list(foreign_key.reference.get("columnReference"))
    predicate = _iri_join(config.base_iri, "foreignKey", "_".join(column_refs))
    relation_type = _safe_label(f"HAS_{'_'.join(column_refs)}").upper()
    relations: list[RelationCandidate] = []
    facts: list[GraphFact] = []
    target_index = {
        tuple(str(row.get(column, "")) for column in ref_column_refs): index
        for index, row in enumerate(rows)
    }
    primary_keys = _as_list(table.tableSchema.primaryKey)
    for row_index, row in enumerate(rows):
        child_values = tuple(str(row.get(column, "")) for column in column_refs)
        if not any(child_values):
            continue
        child_key = _csvw_row_key(row, primary_keys, row_index)
        source_iri = subject_by_key[child_key]
        target_row_index = target_index.get(child_values)
        if target_row_index is not None:
            target_row = rows[target_row_index]
            target_key = _csvw_row_key(target_row, primary_keys, target_row_index)
            target_iri = subject_by_key.get(target_key, _csvw_subject_iri(config, table, target_row, target_row_index, primary_keys))
            target_entity_id = entity_by_key.get(target_key, target_iri)
        else:
            resource = str(foreign_key.reference.get("resource") or table.url or "table")
            target_iri = _iri_join(config.base_iri, Path(resource).stem, "-".join(child_values))
            target_entity_id = target_iri
        evidence = cell_evidence(source, row_index=row_index, column=column_refs[0], value=child_values[0])
        relations.append(
            RelationCandidate.build(
                source=entity_by_key[child_key],
                relation_type=relation_type,
                target=target_entity_id,
                properties={
                    "rdf_subject_iri": source_iri,
                    "rdf_predicate_iri": predicate,
                    "rdf_object": target_iri,
                    "csvw_columnReference": column_refs,
                    "csvw_reference": foreign_key.reference,
                    "target_entity_type": entity_type,
                },
                evidence=[evidence],
                extractor_name="csvw_mapper",
                technique="csvw_metadata_to_rdf",
                schema_version=schema_version,
                provenance_metadata={"standard": "CSVW", "mapping_id": "csvw:foreignKey"},
            )
        )
        facts.append(
            _rdf_fact(
                source=source,
                subject=source_iri,
                predicate=predicate,
                object_value=target_iri,
                object_type="iri",
                evidence=evidence,
                mapper_name="csvw_mapper",
                technique="csvw_metadata_to_rdf",
                schema_version=schema_version,
                standard="CSVW",
                mapping_id="csvw:foreignKey",
                properties={
                    "csvw_columnReference": column_refs,
                    "csvw_reference": foreign_key.reference,
                },
            )
        )
    return relations, facts


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _iri_join(*parts: str) -> str:
    cleaned = [part.strip("/") for part in parts if part not in (None, "")]
    if not cleaned:
        return "urn:kg:unknown"
    first = cleaned[0]
    rest = [quote(part, safe="") for part in cleaned[1:]]
    if first.startswith("urn:"):
        return ":".join([first.rstrip(":"), *rest])
    return "/".join([first, *rest])


def _merge_batches(batches: list[ConstructionBatch]) -> ConstructionBatch:
    if not batches:
        return ConstructionBatch()
    result = batches[0]
    for batch in batches[1:]:
        result = result.merged(batch)
    return result.model_copy(
        update={
            "metadata": {
                **result.metadata,
                "generated_facts": sum(len(batch.facts) for batch in batches),
                "generated_entities": sum(len(batch.entities) for batch in batches),
            }
        }
    )


def _csvw_headers(header_rows: list[list[str]]) -> list[str]:
    if not header_rows:
        return []
    if len(header_rows) == 1:
        return [str(value).strip() or f"column_{index + 1}" for index, value in enumerate(header_rows[0])]
    width = max(len(row) for row in header_rows)
    headers: list[str] = []
    for index in range(width):
        parts = [
            str(row[index]).strip()
            for row in header_rows
            if index < len(row) and str(row[index]).strip()
        ]
        headers.append("_".join(parts) or f"column_{index + 1}")
    return headers


def _csvw_trim(value: str, trim: bool | str) -> str:
    if trim is True or str(trim).casefold() in {"true", "start", "end"}:
        if str(trim).casefold() == "start":
            return value.lstrip()
        if str(trim).casefold() == "end":
            return value.rstrip()
        return value.strip()
    return value


__all__ = [
    "CSVWColumnConfig",
    "CSVWForeignKeyConfig",
    "CSVWMapper",
    "CSVWMappingConfig",
    "CSVWTableConfig",
    "CSVWTableSchemaConfig",
    "PredicateObjectMapConfig",
    "R2RMLMapper",
    "R2RMLMappingConfig",
    "RMLMapper",
    "RMLMappingConfig",
    "TermMapConfig",
    "TriplesMapConfig",
]
