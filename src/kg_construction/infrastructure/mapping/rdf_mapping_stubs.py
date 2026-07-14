"""Standards-based RDF, JSON-LD, and virtual KG mapping adapters."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from kg_construction.domain.models import ConstructionBatch, EvidenceRef, GraphFact, SourceRecord
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured

JsonDict = dict[str, Any]
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XSD_NS = "http://www.w3.org/2001/XMLSchema#"


class ExtensibleMappingStub:
    """Base class for optional mapping techniques."""

    name = "mapping_stub"
    technique = "mapping_stub"
    dependency_hint = "Provide a mapping specification and engine integration."

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        raise AdapterNotConfigured(
            f"{self.name} is represented as an extensible mapping adapter. "
            f"{self.dependency_hint} Source: {source.source_name}"
        )


@dataclass(frozen=True, slots=True)
class JSONLDMappingConfig:
    """JSON-LD mapper configuration."""

    base_iri: str | None = None
    schema_version: str = "v1"
    named_graph: str | None = None
    use_pyld_expansion: bool = True


class JSONLDMapper:
    """Expand JSON-LD and generate evidence-backed RDF graph facts."""

    name = "jsonld_mapper"
    technique = "jsonld_expansion_to_rdf_dataset"

    def __init__(self, config: JSONLDMappingConfig | None = None) -> None:
        self.config = config or JSONLDMappingConfig()

    def expand(self, payload: JsonDict | list[JsonDict]) -> Any:
        """Return JSON-LD expansion using pyld when installed, else a lightweight fallback."""

        if self.config.use_pyld_expansion:
            try:
                from pyld import jsonld

                return jsonld.expand(payload)
            except ImportError:
                pass
        return _lightweight_jsonld_expand(payload)

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        payload = _payload_from_source(source)
        evidence = EvidenceRef.build(
            source,
            location={"kind": "jsonld_document"},
            text=source.content or json.dumps(payload, sort_keys=True),
            metadata={"standard": "JSON-LD"},
        )
        facts = self._facts_from_rdflib(payload, source, evidence)
        expanded = self.expand(payload)
        return ConstructionBatch(
            source_records=[source],
            facts=facts,
            metadata={
                "mapper": self.name,
                "standard": "JSON-LD",
                "expanded": expanded,
                "rdf_dataset_graph": self.config.named_graph,
                "generated_facts": len(facts),
            },
        )

    def _facts_from_rdflib(
        self,
        payload: JsonDict | list[JsonDict],
        source: SourceRecord,
        evidence: EvidenceRef,
    ) -> list[GraphFact]:
        try:
            from rdflib import ConjunctiveGraph
            from rdflib import Literal as RDFLiteral
        except ImportError as exc:
            raise AdapterNotConfigured("JSON-LD RDF dataset generation requires rdflib.") from exc

        graph = ConjunctiveGraph()
        graph.parse(data=json.dumps(payload), format="json-ld", publicID=self.config.base_iri)
        facts: list[GraphFact] = []
        for context in graph.contexts():
            graph_name = str(context.identifier)
            if graph_name.startswith("N"):
                graph_name = self.config.named_graph or "urn:kg:graph:jsonld"
            for subject, predicate, obj in context:
                object_type: Literal["iri", "literal"] = "literal" if isinstance(obj, RDFLiteral) else "iri"
                facts.append(
                    GraphFact.build(
                        subject=str(subject),
                        predicate=str(predicate),
                        object=str(obj),
                        evidence=[evidence],
                        extractor_name=self.name,
                        technique=self.technique,
                        schema_version=self.config.schema_version,
                        properties={
                            "rdf_subject_iri": str(subject),
                            "rdf_predicate_iri": str(predicate),
                            "rdf_object": str(obj),
                            "rdf_object_type": object_type,
                            "rdf_datatype": str(obj.datatype) if isinstance(obj, RDFLiteral) and obj.datatype else None,
                            "rdf_language": obj.language if isinstance(obj, RDFLiteral) else None,
                            "rdf_graph": graph_name,
                            "standard": "JSON-LD",
                        },
                        provenance_metadata={"standard": "JSON-LD", "source_name": source.source_name},
                    )
                )
        return facts


@dataclass(frozen=True, slots=True)
class OBDAPredicateMapping:
    """Virtual predicate mapping used by the lightweight OBDA rewriter."""

    predicate_iri: str
    table: str
    subject_template: str
    object_column: str | None = None
    object_template: str | None = None
    object_is_iri: bool = False
    subject_alias: str = "s"
    object_alias: str = "o"


@dataclass(frozen=True, slots=True)
class OBDARewritePlan:
    """Rewritten SQL plan for one SPARQL-like triple pattern."""

    predicate_iri: str
    sql: str
    variables: list[str] = field(default_factory=list)
    mapping: JsonDict = field(default_factory=dict)


class OBDAVirtualGraphMapper(ExtensibleMappingStub):
    """Lightweight OBDA/virtual KG query rewriter for mapped triple patterns."""

    name = "obda_virtual_graph_mapper"
    technique = "obda_virtual_knowledge_graph_query_rewriting"

    def __init__(self, mappings: list[OBDAPredicateMapping] | None = None) -> None:
        self.mappings = mappings or []

    def rewrite_select(self, sparql: str) -> list[OBDARewritePlan]:
        """Rewrite basic SPARQL triple patterns into SQL SELECT plans."""

        patterns = _sparql_triple_patterns(sparql)
        plans: list[OBDARewritePlan] = []
        for subject, predicate, obj in patterns:
            mapping = self._mapping_for(predicate)
            if mapping is None:
                continue
            variables = [value[1:] for value in (subject, obj) if value.startswith("?")]
            sql = _obda_sql_for_pattern(mapping, subject, obj)
            plans.append(
                OBDARewritePlan(
                    predicate_iri=mapping.predicate_iri,
                    sql=sql,
                    variables=variables,
                    mapping={
                        "table": mapping.table,
                        "subject_template": mapping.subject_template,
                        "object_column": mapping.object_column,
                        "object_template": mapping.object_template,
                        "object_is_iri": mapping.object_is_iri,
                    },
                )
            )
        return plans

    def map_source(self, source: SourceRecord) -> ConstructionBatch:
        """Materialize configured predicate mappings from already-loaded rows."""

        if not self.mappings:
            raise AdapterNotConfigured("OBDA virtual mapper requires predicate mappings.")
        facts: list[GraphFact] = []
        for row_index, row in enumerate(source.records):
            evidence = EvidenceRef.build(
                source,
                location={"kind": "obda_row", "row_index": row_index},
                text=repr(row),
                metadata={"standard": "OBDA"},
            )
            for mapping in self.mappings:
                subject = _render_template(mapping.subject_template, row)
                obj = _render_template(mapping.object_template, row) if mapping.object_template else row.get(mapping.object_column or "")
                if not subject or obj in (None, ""):
                    continue
                facts.append(
                    GraphFact.build(
                        subject=subject,
                        predicate=mapping.predicate_iri,
                        object=str(obj),
                        evidence=[evidence],
                        extractor_name=self.name,
                        technique=self.technique,
                        properties={
                            "rdf_object_type": "iri" if mapping.object_is_iri else "literal",
                            "standard": "OBDA",
                            "virtual_mapping_table": mapping.table,
                        },
                    )
                )
        return ConstructionBatch(
            source_records=[source],
            facts=facts,
            metadata={"mapper": self.name, "standard": "OBDA", "generated_facts": len(facts)},
        )

    def _mapping_for(self, predicate: str) -> OBDAPredicateMapping | None:
        normalized = predicate.strip("<>")
        for mapping in self.mappings:
            if mapping.predicate_iri == normalized:
                return mapping
        return None


def _payload_from_source(source: SourceRecord) -> JsonDict | list[JsonDict]:
    if "json_payload" in source.metadata:
        payload = source.metadata["json_payload"]
    elif source.content:
        payload = json.loads(source.content)
    elif source.records:
        payload = source.records
    else:
        payload = {}
    if isinstance(payload, dict | list):
        return payload
    return {"value": payload}


def _lightweight_jsonld_expand(payload: JsonDict | list[JsonDict]) -> Any:
    def expand_item(item: JsonDict) -> JsonDict:
        context = item.get("@context", {})
        if not isinstance(context, dict):
            context = {}
        expanded: JsonDict = {}
        for key, value in item.items():
            if key == "@context":
                continue
            expanded_key = context.get(key, key)
            if isinstance(expanded_key, dict):
                expanded_key = expanded_key.get("@id", key)
            expanded[str(expanded_key)] = value
        return expanded

    if isinstance(payload, list):
        return [expand_item(item) if isinstance(item, dict) else item for item in payload]
    return [expand_item(payload)]


_SPARQL_TRIPLE_RE = re.compile(
    r"(?P<s>\?\w+|<[^>]+>)\s+(?P<p><[^>]+>)\s+(?P<o>\?\w+|<[^>]+>|\"[^\"]+\")\s*\.",
    flags=re.MULTILINE,
)


def _sparql_triple_patterns(sparql: str) -> list[tuple[str, str, str]]:
    return [
        (match.group("s"), match.group("p"), match.group("o"))
        for match in _SPARQL_TRIPLE_RE.finditer(sparql)
    ]


def _obda_sql_for_pattern(mapping: OBDAPredicateMapping, subject: str, obj: str) -> str:
    selects: list[str] = []
    where: list[str] = []
    if subject.startswith("?"):
        selects.append(f"{_template_sql(mapping.subject_template)} AS {subject[1:]}")
    else:
        where.append(f"{_template_sql(mapping.subject_template)} = '{subject.strip('<>')}'")
    object_sql = _template_sql(mapping.object_template) if mapping.object_template else mapping.object_column or "NULL"
    if obj.startswith("?"):
        selects.append(f"{object_sql} AS {obj[1:]}")
    else:
        object_value = obj.strip("<>\"")
        where.append(f"{object_sql} = '{object_value}'")
    select_clause = ", ".join(selects) if selects else "1"
    where_clause = f" WHERE {' AND '.join(where)}" if where else ""
    return f"SELECT {select_clause} FROM {mapping.table}{where_clause}"


def _template_sql(template: str | None) -> str:
    if not template:
        return "NULL"
    pieces: list[str] = []
    last = 0
    for match in re.finditer(r"{([^{}]+)}", template):
        literal = template[last : match.start()]
        if literal:
            pieces.append(f"'{literal}'")
        pieces.append(str(match.group(1)))
        last = match.end()
    tail = template[last:]
    if tail:
        pieces.append(f"'{tail}'")
    return " || ".join(pieces) if pieces else f"'{template}'"


def _render_template(template: str | None, row: JsonDict) -> str:
    if not template:
        return ""

    def replace(match: re.Match[str]) -> str:
        value = row.get(match.group(1), "")
        return str(value)

    rendered = re.sub(r"{([^{}]+)}", replace, template)
    return rendered if "{" not in rendered else ""


__all__ = [
    "ExtensibleMappingStub",
    "JSONLDMapper",
    "JSONLDMappingConfig",
    "OBDAPredicateMapping",
    "OBDARewritePlan",
    "OBDAVirtualGraphMapper",
]
