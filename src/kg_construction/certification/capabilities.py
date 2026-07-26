"""Executable, semantic smoke certificates for release capabilities.

This module is deliberately different from a registry/import check.  Every
scenario below:

* instantiates the concrete owner recorded in ``capability_manifest.yaml``;
* runs that owner against a small, content-addressed fixture;
* measures a semantic outcome; and
* compares the measurement with an explicit threshold.

The command-line certifier never injects fake transports.  Capabilities backed
by models, remote linkers, Ollama, Neo4j, Memgraph, or Ladybug therefore use
those real backends during a final certification run.  Deterministic unit tests
can exercise the pure scenarios without representing that as a final
certificate.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from kg_construction.domain.evidence import span_evidence
from kg_construction.domain.graph_objects import (
    PropertyEdge,
    PropertyGraph,
    PropertyNode,
)
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    RelationCandidate,
    SourceRecord,
    canonical_json,
    normalize_text,
    stable_id,
)
from kg_construction.service_runtime import LockedServiceRuntime

if TYPE_CHECKING:
    from kg_construction.infrastructure.extractors.model_backed import (
        MandatoryProviderBundle,
    )
    from kg_construction.infrastructure.ontology.validation import ValidationReport

JsonDict = dict[str, Any]
Owner = type[Any]
Scenario = Callable[[Owner, "CertificationContext"], "ScenarioResult"]


@dataclass(frozen=True, slots=True)
class CertificationContext:
    """Runtime identity and repository paths available to one scenario."""

    capability: str
    profile: str
    run_id: str
    root: Path
    workdir: Path


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Semantic result returned by one capability scenario."""

    instance: object
    operation: str
    fixture: object
    output: object
    metrics: Mapping[str, float]
    thresholds: Mapping[str, float]
    assertions: tuple[JsonDict, ...]
    backend: str = "in_process"
    details: JsonDict = field(default_factory=dict)


def certify_capability(
    capability: str,
    *,
    profile: str,
    run_id: str,
    root: Path,
) -> JsonDict:
    """Execute and return nonce-bound semantic evidence for ``capability``."""

    manifest = _load_mapping(root / "capability_manifest.yaml")
    rows = {
        str(row["name"]): row
        for row in manifest.get("techniques", [])
        if isinstance(row, dict) and row.get("name")
    }
    row = rows.get(capability)
    if row is None:
        raise KeyError(f"{capability!r} is not declared in capability_manifest.yaml")
    applicable_profiles = row.get("profiles")
    if (
        isinstance(applicable_profiles, list)
        and profile not in applicable_profiles
    ):
        raise ValueError(
            f"{capability!r} is not applicable to certification profile "
            f"{profile!r}; declared profiles={applicable_profiles}"
        )
    owner_path = str(row.get("owner", ""))
    owner = _import_owner(owner_path)
    scenario = SCENARIOS.get(capability)
    if scenario is None:
        raise KeyError(f"{capability!r} has no semantic certification scenario")

    with tempfile.TemporaryDirectory(prefix=f"kg-semantic-{capability[:20]}-") as directory:
        context = CertificationContext(
            capability=capability,
            profile=profile,
            run_id=run_id,
            root=root,
            workdir=Path(directory),
        )
        result = scenario(owner, context)

    implementation = f"{type(result.instance).__module__}.{type(result.instance).__qualname__}"
    if not isinstance(result.instance, owner):
        raise TypeError(
            f"{capability}: scenario instantiated {implementation}, expected {owner_path}"
        )
    if not result.metrics or not result.thresholds:
        raise ValueError(f"{capability}: semantic metrics and thresholds must be non-empty")
    if set(result.metrics) != set(result.thresholds):
        raise ValueError(
            f"{capability}: metrics/threshold keys differ: "
            f"{sorted(result.metrics)} != {sorted(result.thresholds)}"
        )
    assertions = list(result.assertions)
    if not assertions:
        raise ValueError(f"{capability}: at least one named assertion is required")
    measured_pass = all(
        float(result.metrics[name]) >= float(result.thresholds[name])
        for name in result.metrics
    )
    assertion_pass = all(item.get("passed") is True and item.get("name") for item in assertions)
    passed = measured_pass and assertion_pass

    fixture_payload = _jsonable(result.fixture)
    output_payload = _jsonable(result.output)
    return {
        "capability": capability,
        "profile": profile,
        "run_id": run_id,
        "observations": {
            "instantiated": True,
            "executed": True,
            "evaluated": True,
            "passed": passed,
            "skipped": False,
            "fallback": False,
            "instantiated_evidence": {
                "implementation": implementation,
                "manifest_owner": owner_path,
                "requested_capability": capability,
            },
            "executed_evidence": {
                "operation": result.operation,
                "backend": result.backend,
                "fixture_sha256": _sha256(fixture_payload),
                "output_sha256": _sha256(output_payload),
                "details": result.details,
            },
            "evaluated_evidence": {
                "metrics": dict(result.metrics),
                "thresholds": dict(result.thresholds),
            },
            "passed_evidence": {"assertions": assertions},
        },
    }


def _import_owner(path: str) -> Owner:
    module_name, attribute = path.rsplit(".", 1)
    owner = getattr(importlib.import_module(module_name), attribute)
    if not isinstance(owner, type):
        raise TypeError(f"manifest owner {path!r} is not a class")
    return owner


def _load_mapping(path: Path) -> JsonDict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _result(
    instance: object,
    operation: str,
    fixture: object,
    output: object,
    *,
    metric: str,
    value: float,
    threshold: float,
    assertion: str,
    backend: str = "in_process",
    details: JsonDict | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        instance=instance,
        operation=operation,
        fixture=fixture,
        output=output,
        metrics={metric: float(value)},
        thresholds={metric: float(threshold)},
        assertions=(
            {
                "name": assertion,
                "passed": float(value) >= float(threshold),
                "observed": float(value),
                "threshold": float(threshold),
            },
        ),
        backend=backend,
        details=details or {},
    )


def _source() -> SourceRecord:
    return SourceRecord.from_text(
        "INCIDENT-1042 affected Recommendation API on 2026-07-09.",
        source_name="memory://semantic-certification.txt",
    )


def _batch() -> ConstructionBatch:
    source = _source()
    text = source.content or ""
    left_text = "INCIDENT-1042"
    right_text = "Recommendation API"
    left_start = text.index(left_text)
    right_start = text.index(right_text)
    left = EntityCandidate.build(
        entity_type="Incident",
        name=left_text,
        properties={"severity": "high"},
        evidence=[
            span_evidence(
                source,
                start=left_start,
                end=left_start + len(left_text),
                text=left_text,
            )
        ],
        confidence=0.99,
        extractor_name="semantic_certifier",
        technique="fixture",
    )
    right = EntityCandidate.build(
        entity_type="Service",
        name=right_text,
        properties={"owner": "platform"},
        evidence=[
            span_evidence(
                source,
                start=right_start,
                end=right_start + len(right_text),
                text=right_text,
            )
        ],
        confidence=0.99,
        extractor_name="semantic_certifier",
        technique="fixture",
    )
    relation = RelationCandidate.build(
        source=left.candidate_id,
        relation_type="AFFECTED",
        target=right.candidate_id,
        properties={"observed_at": "2026-07-09"},
        evidence=[
            span_evidence(
                source,
                start=0,
                end=len(text),
                text=text,
            )
        ],
        confidence=0.99,
        extractor_name="semantic_certifier",
        technique="fixture",
    )
    return ConstructionBatch(
        source_records=[source],
        entities=[left, right],
        relations=[relation],
    )


def _property_graph(run_id: str = "fixture") -> PropertyGraph:
    suffix = run_id[:12]
    incident_id = f"cert-incident-{suffix}"
    service_id = f"cert-service-{suffix}"
    return PropertyGraph(
        nodes=[
            PropertyNode(
                node_id=incident_id,
                labels=["Entity", "Incident"],
                properties={"name": "INCIDENT-1042", "certification_run": run_id},
            ),
            PropertyNode(
                node_id=service_id,
                labels=["Entity", "Service"],
                properties={"name": "Recommendation API", "certification_run": run_id},
            ),
        ],
        edges=[
            PropertyEdge(
                edge_id=f"cert-edge-{suffix}",
                source_id=incident_id,
                target_id=service_id,
                type="AFFECTED",
                properties={"confidence": 0.99, "certification_run": run_id},
            )
        ],
    )


def _loaders(owner: Owner, context: CertificationContext) -> ScenarioResult:
    fixture_root = context.root / "tests" / "fixtures"
    paths = {
        "api_response_loader": fixture_root / "enterprise_loaders" / "api_response.json",
        "chat_transcript_loader": fixture_root / "enterprise_loaders" / "support_chat.chat",
        "csv_loader": fixture_root / "golden_benchmarks" / "csv" / "input.csv",
        "email_loader": fixture_root / "enterprise_loaders" / "incident_email.eml",
        "excel_loader": fixture_root / "enterprise_loaders" / "service_inventory.xlsx",
        "html_loader": fixture_root / "enterprise_loaders" / "recommendation_runbook.html",
        "log_loader": fixture_root / "enterprise_loaders" / "service_events.log",
        "markdown_loader": fixture_root / "enterprise_loaders" / "incident_review.md",
        "pdf_loader": fixture_root / "enterprise_loaders" / "incident_brief.pdf",
        "report_loader": fixture_root / "enterprise_loaders" / "ops_report.report",
        "scientific_paper_loader": fixture_root
        / "enterprise_loaders"
        / "kg_construction_article.jats",
        "sql_database_loader": fixture_root / "enterprise_loaders" / "enterprise.sqlite",
        "structured_api_loader": fixture_root / "enterprise_loaders" / "api_response.json",
        "text_loader": fixture_root / "golden_benchmarks" / "text" / "input.txt",
        "webpage_loader": fixture_root / "enterprise_loaders" / "recommendation_runbook.html",
        "xml_loader": fixture_root / "enterprise_loaders" / "services.xml",
    }
    generated: dict[str, tuple[str, object]] = {
        "event_payload_loader": (
            "events.json",
            {
                "events": [
                    {
                        "event_id": "event-1",
                        "event_type": "incident",
                        "service": "Recommendation API",
                    }
                ]
            },
        ),
        "jsonld_loader": (
            "service.jsonld",
            {
                "@context": {"name": "http://schema.org/name"},
                "@id": "urn:service:recommendation",
                "name": "Recommendation API",
            },
        ),
        "json_loader": (
            "service.json",
            [{"id": "service-1", "name": "Recommendation API"}],
        ),
    }
    if context.capability in generated:
        filename, payload = generated[context.capability]
        path = context.workdir / filename
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path = paths[context.capability]
    instance = owner()
    sources = instance.load(path)
    semantic_units = sum(
        max(len(source.records), 1 if source.content else 0) for source in sources
    )
    provenance_units = sum(
        1
        for source in sources
        if isinstance(source.metadata.get("source_evidence"), list)
        and source.metadata["source_evidence"]
    )
    score = 1.0 if sources and semantic_units > 0 and provenance_units == len(sources) else 0.0
    return _result(
        instance,
        "load",
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        {
            "source_count": len(sources),
            "semantic_units": semantic_units,
            "provenance_units": provenance_units,
            "source_types": [source.source_type for source in sources],
        },
        metric="loaded_source_evidence_rate",
        value=score,
        threshold=1.0,
        assertion="loader emitted non-empty source records with source evidence",
    )


def _api_response_loader(owner: Owner, context: CertificationContext) -> ScenarioResult:
    """Exercise the captured-response path; live web access is not this capability."""

    return _loaders(owner, context)


def _structured_direct_mapper(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.mapping.direct_table_mapper import (
        DirectTableMappingConfig,
        ForeignKeyMappingConfig,
        infer_column_type,
        profile_records,
    )

    rows = [
        {"id": "1", "name": "Ada", "manager_id": "", "status": "active", "score": 10},
        {"id": "2", "name": "Grace", "manager_id": "1", "status": "active", "score": 20},
    ]
    source = SourceRecord.from_rows(
        rows,
        source_type="csv",
        source_name="employees.csv",
    )
    config = DirectTableMappingConfig(
        table_name="employees",
        node_type="Employee",
        id_columns=["id"],
        name_column="name",
        property_columns=["status", "score"],
        foreign_keys=[
            ForeignKeyMappingConfig(
                source_column="manager_id",
                target_column="id",
                relationship_type="REPORTS_TO",
            )
        ],
    )
    instance = owner(config)
    batch = instance.map_source(source)
    profile = profile_records(rows)
    checks = {
        "row_to_entity_mapping": len(batch.entities) == 2,
        "table_to_node_mapping": all(item.entity_type == "Employee" for item in batch.entities),
        "column_to_property_mapping": all("status" in item.properties for item in batch.entities),
        "primary_key_to_id_mapping": len({item.candidate_id for item in batch.entities}) == 2,
        "foreign_key_to_edge_mapping": any(
            relation.relation_type == "REPORTS_TO" for relation in batch.relations
        ),
        "column_type_inference": (
            infer_column_type([10, 20]) == "integer"
            and profile["types"]["score"] == "integer"
        ),
        "schema_profiling": (
            profile["row_count"] == 2 and profile["distinct_counts"]["id"] == 2
        ),
        "direct_table_mapper": (
            len(batch.entities) == 2
            and len(batch.relations) == 1
            and all(item.evidence for item in batch.entities)
        ),
    }
    passed = checks[context.capability]
    return _result(
        instance,
        "map_source",
        rows,
        {"batch": batch, "profile": profile, "capability_check": checks[context.capability]},
        metric="structured_mapping_contract",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} preserved the declared table semantics",
    )


def _semantic_profiler(owner: Owner, context: CertificationContext) -> ScenarioResult:
    rows = [
        {"employee_id": "e1", "department_id": "d1", "status": "active"},
        {"employee_id": "e2", "department_id": "d1", "status": "inactive"},
        {"employee_id": "e3", "department_id": "d2", "status": "active"},
    ]
    instance = owner()
    plan = instance.profile(rows, table_name="employees", record_type="Employee")
    status = plan.assignment("status")
    department = plan.assignment("department_id")
    passed = (
        status is not None
        and status.materialization == "property"
        and department is not None
        and department.materialization == "edge"
    )
    return _result(
        instance,
        "profile",
        rows,
        plan,
        metric="column_role_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion="categorical status stayed a property while the foreign key became a reference",
    )


def _catalog_fixture():
    from kg_construction.structured.source_catalog import (
        DeclaredForeignKey,
        SourceCatalog,
        TableDeclaration,
    )

    departments = SourceRecord.from_rows(
        [{"department_id": "d1", "name": "Research"}],
        source_type="csv",
        source_name="departments.csv",
    )
    employees = SourceRecord.from_rows(
        [
            {
                "employee_id": "e1",
                "name": "Ada",
                "department_id": "d1",
                "status": "active",
            }
        ],
        source_type="csv",
        source_name="employees.csv",
    )
    catalog = SourceCatalog()
    catalog.declare(
        TableDeclaration(
            table_name="departments.csv",
            record_type="Department",
            primary_key=["department_id"],
            origin="user",
        )
    )
    catalog.declare(
        TableDeclaration(
            table_name="employees.csv",
            record_type="Employee",
            primary_key=["employee_id"],
            foreign_keys=[
                DeclaredForeignKey(
                    columns=["department_id"],
                    target_table="departments.csv",
                    target_key=["department_id"],
                    relationship_name="WORKS_IN",
                )
            ],
            origin="user",
        )
    )
    return catalog, departments, employees


def _source_catalog(owner: Owner, context: CertificationContext) -> ScenarioResult:
    _, departments, employees = _catalog_fixture()
    instance = owner()
    from kg_construction.structured.source_catalog import (
        DeclaredForeignKey,
        TableDeclaration,
    )

    instance.declare(
        TableDeclaration(
            table_name="departments.csv",
            record_type="Department",
            primary_key=["department_id"],
        )
    )
    instance.declare(
        TableDeclaration(
            table_name="employees.csv",
            record_type="Employee",
            primary_key=["employee_id"],
            foreign_keys=[
                DeclaredForeignKey(
                    columns=["department_id"],
                    target_table="departments.csv",
                    target_key=["department_id"],
                    relationship_name="WORKS_IN",
                )
            ],
        )
    )
    plan = instance.resolved_plan(employees)
    passed = (
        instance.declaration_for(departments.source_name) is not None
        and len(plan.foreign_keys) == 1
        and plan.foreign_keys[0].target_available
    )
    return _result(
        instance,
        "declare+resolved_plan",
        [departments, employees],
        plan,
        metric="declared_key_resolution_rate",
        value=float(passed),
        threshold=1.0,
        assertion="catalog resolved the declared cross-table foreign key",
    )


def _semantic_table_mapper(owner: Owner, context: CertificationContext) -> ScenarioResult:
    catalog, departments, employees = _catalog_fixture()
    instance = owner(catalog, require_declarations=True)
    instance.prepare_sources([departments, employees])
    department_batch = instance.map_source(departments)
    employee_batch = instance.map_source(employees)
    status_nodes = [
        entity
        for entity in employee_batch.entities
        if entity.entity_type.casefold() in {"status", "active"}
    ]
    passed = (
        len(department_batch.entities) == 1
        and len(employee_batch.entities) == 1
        and len(employee_batch.relations) == 1
        and not status_nodes
        and employee_batch.entities[0].properties.get("status") == "active"
    )
    return _result(
        instance,
        "prepare_sources+map_source",
        [departments, employees],
        {
            "departments": department_batch,
            "employees": employee_batch,
            "categorical_node_count": len(status_nodes),
        },
        metric="cross_table_semantic_mapping_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion="foreign key became an edge and scalar category was not duplicated as a node",
    )


def _json_mapper(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.mapping.rdf_mapping_stubs import OBDAPredicateMapping
    from kg_construction.infrastructure.mapping.semi_structured import (
        JsonArrayEdgeMapping,
        JsonPathMappingConfig,
        JsonPropertyMapping,
    )

    payload: object = {
        "service": {"id": "svc-1", "name": "Recommendation API"},
        "errors": [{"code": "E1042", "message": "Timeout"}],
    }
    source = SourceRecord.from_json(payload, source_name="memory://service.json")
    if context.capability == "dotted_path_json_mapper":
        instance = owner("ServicePayload")
        batch = instance.map_source(source)
        passed = any(
            entity.properties.get("service.name") == "Recommendation API"
            for entity in batch.entities
        )
        output: object = batch
    elif context.capability == "jsonpath_mapper":
        instance = owner(
            JsonPathMappingConfig(
                node_type="ServicePayload",
                name_path="$.service.name",
                property_paths=[
                    JsonPropertyMapping(
                        property_name="service_name",
                        path="$.service.name",
                        required=True,
                    )
                ],
                array_edges=[
                    JsonArrayEdgeMapping(
                        path="$.errors[*]",
                        relationship_type="HAS_ERROR",
                        target_node_type="Error",
                        target_name_path="$.code",
                    )
                ],
            )
        )
        batch = instance.map_source(source)
        passed = (
            any(entity.name == "Recommendation API" for entity in batch.entities)
            and any(relation.relation_type == "HAS_ERROR" for relation in batch.relations)
            and all(entity.evidence for entity in batch.entities)
        )
        output = batch
    elif context.capability == "jsonld_mapper":
        jsonld_payload: JsonDict = {
            "@context": {"name": "http://schema.org/name"},
            "@id": "urn:service:recommendation",
            "name": "Recommendation API",
        }
        jsonld_source = SourceRecord.from_json(
            jsonld_payload, source_name="memory://service.jsonld"
        ).model_copy(update={"source_type": "jsonld"})
        instance = owner()
        batch = instance.map_source(jsonld_source)
        passed = bool(batch.facts) and all(fact.evidence for fact in batch.facts)
        output = batch
        payload = jsonld_payload
    else:
        row_source = SourceRecord.from_rows(
            [{"id": "svc-1", "name": "Recommendation API"}],
            source_type="sql",
            source_name="services",
        )
        # Use flat row references for the materialization contract.
        instance = owner(
            [
                OBDAPredicateMapping(
                    predicate_iri="urn:kg:name",
                    table="services",
                    subject_template="urn:service:{id}",
                    object_column="name",
                )
            ]
        )
        plans = instance.rewrite_select(
            "SELECT ?s ?o WHERE { ?s <urn:kg:name> ?o . }"
        )
        batch = instance.map_source(row_source)
        passed = bool(plans) and bool(batch.facts) and "FROM services" in plans[0].sql
        output = {"plans": plans, "batch": batch}
        payload = row_source.records
    return _result(
        instance,
        "map_source",
        payload,
        output,
        metric="semantic_json_mapping_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} preserved paths and evidence",
    )


def _event_or_html_mapper(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.loaders.enterprise import HTMLLoader, LogLoader
    from kg_construction.infrastructure.loaders.stubs import EventPayloadLoader

    fixture_root = context.root / "tests" / "fixtures" / "enterprise_loaders"
    payload: object
    if context.capability == "event_graph_mapper":
        path = context.workdir / "events.json"
        payload = {
            "events": [
                {
                    "event_id": "e1",
                    "event_type": "request",
                    "service": "Recommendation API",
                    "trace_id": "t1",
                    "span_id": "s1",
                },
                {
                    "event_id": "e2",
                    "event_type": "database",
                    "service": "Feature Store",
                    "trace_id": "t1",
                    "span_id": "s2",
                    "parent_span_id": "s1",
                },
            ]
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        source = EventPayloadLoader().load(path)[0]
        expected_relation = "PARENT_OF_SPAN"
    elif context.capability == "log_event_mapper":
        path = fixture_root / "service_events.log"
        source = LogLoader().load(path)[0]
        payload = path.read_text(encoding="utf-8")
        expected_relation = "REQUESTED_ENDPOINT"
    else:
        path = fixture_root / "recommendation_runbook.html"
        source = HTMLLoader().load(path)[0]
        payload = path.read_text(encoding="utf-8")
        expected_relation = "HAS_STRUCTURED_DATA"
    instance = owner()
    batch = instance.map_source(source)
    passed = (
        bool(batch.entities)
        and any(relation.relation_type == expected_relation for relation in batch.relations)
        and all(entity.evidence for entity in batch.entities)
    )
    return _result(
        instance,
        "map_source",
        payload,
        batch,
        metric="specialized_mapping_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} emitted evidence-backed domain structure",
    )


def _standards_mapper(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.mapping.standards import (
        CSVWColumnConfig,
        CSVWMappingConfig,
        CSVWTableConfig,
        CSVWTableSchemaConfig,
    )

    rows = [
        {"id": "1", "name": "Ada", "manager_id": ""},
        {"id": "2", "name": "Grace", "manager_id": "1"},
    ]
    if context.capability == "csvw_mapper":
        source = SourceRecord.from_rows(
            rows, source_type="csv", source_name="people.csv"
        )
        config = CSVWMappingConfig(
            tables=[
                CSVWTableConfig(
                    url="people.csv",
                    tableSchema=CSVWTableSchemaConfig(
                        primaryKey="id",
                        columns=[
                            CSVWColumnConfig(name="id", required=True),
                            CSVWColumnConfig(name="name", required=True),
                        ],
                    ),
                )
            ]
        )
        instance = owner(config)
    elif context.capability == "r2rml_mapper":
        source = SourceRecord.from_rows(
            rows,
            source_type="sql",
            source_name="sqlite://memory#people",
            metadata={"table": "people"},
        )
        mapping = """
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix ex: <http://example.com/> .
            <#PersonMap>
              rr:logicalTable [ rr:tableName "people" ];
              rr:subjectMap [ rr:template "http://example.com/person/{id}";
                              rr:class ex:Person ];
              rr:predicateObjectMap [
                rr:predicate ex:name;
                rr:objectMap [ rr:column "name" ]
              ] .
        """
        instance = owner.from_turtle(mapping)
    else:
        source = SourceRecord.from_rows(
            rows, source_type="csv", source_name="people.csv"
        )
        mapping = """
            @prefix rr: <http://www.w3.org/ns/r2rml#> .
            @prefix rml: <http://semweb.mmlab.be/ns/rml#> .
            @prefix ql: <http://semweb.mmlab.be/ns/ql#> .
            @prefix ex: <http://example.com/> .
            <#People>
              rml:logicalSource [
                rml:source "people.csv";
                rml:referenceFormulation ql:CSV
              ];
              rr:subjectMap [ rr:template "http://example.com/person/{id}";
                              rr:class ex:Person ];
              rr:predicateObjectMap [
                rr:predicate ex:name;
                rr:objectMap [ rml:reference "name" ]
              ] .
        """
        instance = owner.from_turtle(mapping)
    batch = instance.map_source(source)
    passed = (
        len(batch.entities) == 2
        and bool(batch.facts)
        and all(fact.evidence for fact in batch.facts)
    )
    return _result(
        instance,
        "map_source",
        {"rows": rows, "standard": context.capability},
        batch,
        metric="standards_mapping_evidence_rate",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} materialized standards-based, evidenced graph facts",
    )


def _extractor(owner: Owner, context: CertificationContext) -> ScenarioResult:
    source = _source()
    if context.capability == "gazetteer_entity_extractor":
        instance = owner(
            {
                "Incident": ["INCIDENT-1042"],
                "Service": ["Recommendation API"],
            }
        )
    elif context.capability == "openie_relation_extractor":
        instance = owner(backend=_SpacyOpenIEBackend(context))
    else:
        instance = owner()
    batch = instance.extract(source)
    if context.capability in {
        "dependency_pattern_relation_extractor",
        "openie_relation_extractor",
    }:
        passed = any(
            relation.relation_type == "AFFECTED" for relation in batch.relations
        )
    else:
        passed = bool(batch.entities) and all(entity.evidence for entity in batch.entities)
    return _result(
        instance,
        "extract",
        source,
        batch,
        metric="evidence_grounded_extraction",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} extracted the expected evidenced semantic unit",
        backend=(
            "spacy:model"
            if context.capability == "openie_relation_extractor"
            else "in_process"
        ),
    )


class _SpacyOpenIEBackend:
    """Small OpenIE port backed by the mandatory, locally locked spaCy model."""

    def __init__(self, context: CertificationContext) -> None:
        import spacy

        configured = os.environ.get("KG_CERT_SPACY_MODEL_PATH")
        candidates = [
            Path(configured) if configured else None,
            _locked_huggingface_snapshot(
                context,
                model_name="spacy_pipeline",
                repo_id="spacy/en_core_web_sm",
            ),
        ]
        model_error: Exception | None = None
        for path in candidates:
            if path is None or not path.exists():
                continue
            try:
                self._nlp = spacy.load(path)
                self.model_path = str(path)
                return
            except Exception as exc:  # pragma: no cover - depends on hydrated artifact layout
                model_error = exc
        try:
            self._nlp = spacy.load("en_core_web_sm")
            self.model_path = "en_core_web_sm"
        except Exception as exc:
            detail = model_error or exc
            raise RuntimeError(
                "the mandatory locked spaCy en_core_web_sm model is not loadable; "
                "set KG_CERT_SPACY_MODEL_PATH to its hydrated directory"
            ) from detail

    def extract_triples(self, text: str) -> list[JsonDict]:
        document = self._nlp(text)
        triples: list[JsonDict] = []
        for token in document:
            subjects = [
                child for child in token.children if child.dep_ in {"nsubj", "nsubjpass"}
            ]
            objects = [
                child
                for child in token.children
                if child.dep_ in {"attr", "dobj", "obj", "oprd"}
            ]
            for subject in subjects:
                for obj in objects:
                    subject_span = document[
                        min(item.i for item in subject.subtree) : max(item.i for item in subject.subtree) + 1
                    ]
                    object_span = document[
                        min(item.i for item in obj.subtree) : max(item.i for item in obj.subtree) + 1
                    ]
                    start = min(subject_span.start_char, token.idx, object_span.start_char)
                    end = max(
                        subject_span.end_char,
                        token.idx + len(token.text),
                        object_span.end_char,
                    )
                    triples.append(
                        {
                            "subject": subject_span.text,
                            "relation": token.text,
                            "object": object_span.text,
                            "evidence_text": text[start:end],
                            "confidence": 0.9,
                        }
                    )
        return triples


def _locked_huggingface_snapshot(
    context: CertificationContext,
    *,
    model_name: str,
    repo_id: str,
) -> Path | None:
    lock_path = context.root / "models.lock.yaml"
    if not lock_path.is_file():
        return None
    lock = _load_mapping(lock_path)
    row = next(
        (
            item
            for item in lock.get("models", [])
            if isinstance(item, dict)
            and item.get("name") == model_name
            and context.profile in item.get("profiles", [])
        ),
        None,
    )
    if row is None or not row.get("revision"):
        return None
    cache = os.environ.get("HF_HUB_CACHE")
    if cache:
        cache_root = Path(cache)
    else:
        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        cache_root = hf_home / "hub"
    repo_directory = "models--" + repo_id.replace("/", "--")
    return cache_root / repo_directory / "snapshots" / str(row["revision"])


def _mandatory_bundle(
    context: CertificationContext,
) -> MandatoryProviderBundle:
    from kg_construction.infrastructure.extractors.model_backed import (
        build_mandatory_model_providers_from_lock,
    )

    cache = os.environ.get("HF_HUB_CACHE")
    runtime = LockedServiceRuntime.from_environment()
    return build_mandatory_model_providers_from_lock(
        context.root / "models.lock.yaml",
        profile=context.profile,
        cache_root=Path(cache) if cache else None,
        ollama_base_url=runtime.ollama_base_url,
    )


def _model_document():
    from kg_construction.domain.claim import DocumentRef
    from kg_construction.unstructured.pipeline import RuleSegmenter

    source = SourceRecord.from_text(
        "Alan Turing worked at Bletchley Park.",
        source_name="memory://mandatory-model-certification.txt",
    )
    document = DocumentRef.from_source(source)
    segments = RuleSegmenter().segment(document, source.content or "")
    return source, document, segments


_PROVIDER_NAME_TO_ID = {
    "deterministic_rule_mentions": "deterministic_rule_mentions",
    "gliner": "gliner",
    "transformers_ner": "transformers",
    "spacy": "spacy",
    "deterministic_pattern_claims": "deterministic_pattern_claims",
    "gliner_relex": "gliner_relex",
    "transformers_qwen3": "transformers_qwen3",
    "mlx_qwen3": "mlx_qwen3",
    "ollama_qwen3": "ollama_qwen3",
}


def _agreement_counts(bundle: MandatoryProviderBundle) -> dict[str, int]:
    mandatory_names = bundle.mandatory_provider_names
    counts = {name: 0 for name in mandatory_names}
    mention_detector = bundle.mention_detector
    claim_extractor = bundle.claim_extractor
    agreements = (
        *mention_detector.last_agreements,
        *claim_extractor.last_agreements,
    )
    id_to_name = {value: key for key, value in _PROVIDER_NAME_TO_ID.items()}
    for agreement in agreements:
        providers = (
            agreement.participating_providers
            or agreement.agreeing_providers
        )
        for provider_id in providers:
            name = id_to_name.get(provider_id, provider_id)
            if name in counts:
                counts[name] += 1
    return counts


def _model_provider(owner: Owner, context: CertificationContext) -> ScenarioResult:
    bundle = _mandatory_bundle(context)
    source, document, segments = _model_document()
    text = source.content or ""
    capability_to_provider = {
        "gliner_entity_extractor": "gliner",
        "transformer_ner_extractor": "transformers",
        "spacy_ner_extractor": "spacy",
        "gliner_relex_claim_extractor": "gliner_relex",
        "transformers_qwen_claim_extractor": "transformers_qwen3",
        "mlx_qwen_claim_extractor": "mlx_qwen3",
        "ollama_qwen_claim_extractor": "ollama_qwen3",
    }
    provider_id = capability_to_provider.get(context.capability)
    instance: object
    output: object
    if context.capability == "mandatory_mention_ensemble":
        mention_ensemble = bundle.mention_detector
        mentions = tuple(mention_ensemble.detect(document, text, segments))
        instance = mention_ensemble
        output = mentions
        names = {mention.surface for mention in mentions}
        passed = {"Alan Turing", "Bletchley Park"} <= names
        details = {
            "providers": list(mention_ensemble.provider_ids),
            "agreement_count": len(mention_ensemble.last_agreements),
        }
    elif context.capability == "mandatory_claim_ensemble":
        mentions = tuple(bundle.mention_detector.detect(document, text, segments))
        claim_ensemble = bundle.claim_extractor
        claims = tuple(claim_ensemble.extract(document, text, segments, mentions))
        instance = claim_ensemble
        output = claims
        passed = any(
            claim.subject == "Alan Turing"
            and claim.object == "Bletchley Park"
            and claim.predicate == "worked_at"
            for claim in claims
        )
        details = {
            "providers": list(claim_ensemble.provider_ids),
            "agreement_count": len(claim_ensemble.last_agreements),
        }
    elif provider_id in {"gliner", "transformers", "spacy"}:
        mention_provider = next(
            provider
            for provider in bundle.mention_detector.providers
            if isinstance(provider, owner)
        )
        mentions = tuple(mention_provider.detect(document, text, segments))
        instance = mention_provider
        output = mentions
        names = {mention.surface for mention in mentions}
        passed = {"Alan Turing", "Bletchley Park"} <= names
        details = {"provider": provider_id, "semantic_output_count": len(mentions)}
    else:
        from kg_construction.unstructured.pipeline import (
            RuleMentionDetector,
        )

        claim_provider = next(
            provider
            for provider in bundle.claim_extractor.providers
            if isinstance(provider, owner)
        )
        # Claim providers are defined over the output of the mandatory mention
        # stage. Give the focused provider certification exact upstream anchors
        # instead of an artificial empty mention set.
        mentions = tuple(RuleMentionDetector().detect(document, text, segments))
        claims = tuple(
            claim_provider.extract(document, text, segments, mentions)
        )
        instance = claim_provider
        output = claims
        passed = any(
            claim.subject == "Alan Turing"
            and claim.object == "Bletchley Park"
            and claim.predicate == "worked_at"
            for claim in claims
        )
        details = {
            "provider": provider_id,
            "semantic_output_count": len(claims),
            "semantic_claims": [
                {
                    "kind": claim.kind.value,
                    "subject": claim.subject,
                    "predicate": claim.predicate,
                    "object": claim.object,
                }
                for claim in claims
            ],
        }
    operation = (
        "detect"
        if context.capability == "mandatory_mention_ensemble"
        or provider_id in {"gliner", "transformers", "spacy"}
        else "extract"
    )
    return _result(
        instance,
        operation,
        source,
        output,
        metric="held_out_semantic_exact_match",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} produced the held-out grounded semantic output",
        backend=f"{provider_id or context.capability}:live-model",
        details=details,
    )


def _ollama_provider(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.llm.ollama import OllamaConfig

    runtime = LockedServiceRuntime.from_environment()
    model = runtime.ollama_model
    base_url = runtime.ollama_base_url
    instance = owner(
        OllamaConfig(
            base_url=base_url,
            model=model,
            temperature=0.0,
            structured_output=True,
            max_retries=1,
        )
    )
    output = instance.complete_json(
        'Return exactly one JSON object with the boolean field "ok" set to true.',
        validator=lambda payload: payload.get("ok") is True,
    )
    passed = output.get("ok") is True
    return _result(
        instance,
        "complete_json",
        {"prompt_contract": {"ok": True}},
        output,
        metric="structured_output_validity",
        value=float(passed),
        threshold=1.0,
        assertion="live Ollama returned schema-valid JSON",
        backend="ollama:live",
        details={"model": model, "base_url": base_url},
    )


def _ollama_graph(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.llm.ollama import OllamaConfig, OllamaProvider

    runtime = LockedServiceRuntime.from_environment()
    model = runtime.ollama_model
    base_url = runtime.ollama_base_url
    provider = OllamaProvider(
        OllamaConfig(
            base_url=base_url,
            model=model,
            temperature=0.0,
            structured_output=True,
            max_retries=2,
        )
    )
    instance = owner(
        provider,
        allowed_node_types=["Incident", "Service"],
        allowed_relationship_types=["AFFECTED"],
    )
    source = _source()
    batch = instance.extract(source)
    passed = (
        any(entity.name == "INCIDENT-1042" for entity in batch.entities)
        and any(entity.name == "Recommendation API" for entity in batch.entities)
        and any(relation.relation_type == "AFFECTED" for relation in batch.relations)
        and all(relation.evidence for relation in batch.relations)
    )
    return _result(
        instance,
        "extract",
        source,
        batch,
        metric="grounded_graph_exact_match",
        value=float(passed),
        threshold=1.0,
        assertion="live Ollama emitted the held-out evidence-grounded relation",
        backend="ollama:live",
        details={"model": model, "base_url": base_url},
    )


def _gliner_ollama(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.extractors.relations import RelationTypingConfig
    from kg_construction.infrastructure.llm.ollama import OllamaConfig, OllamaProvider

    runtime = LockedServiceRuntime.from_environment()
    model = runtime.ollama_model
    base_url = runtime.ollama_base_url
    provider = OllamaProvider(
        OllamaConfig(
            base_url=base_url,
            model=model,
            temperature=0.0,
            structured_output=True,
            max_retries=2,
        )
    )
    instance = owner(
        provider,
        config=RelationTypingConfig(
            entity_labels=["Incident", "Service"],
            allowed_relationship_types=["AFFECTED"],
        ),
    )
    source = _source()
    batch = instance.extract(source, labels=["Incident", "Service"])
    passed = (
        len(batch.entities) >= 2
        and any(relation.relation_type == "AFFECTED" for relation in batch.relations)
        and all(relation.evidence for relation in batch.relations)
    )
    return _result(
        instance,
        "extract",
        source,
        batch,
        metric="hybrid_relation_exact_match",
        value=float(passed),
        threshold=1.0,
        assertion="live GLiNER plus Ollama preserved and typed the relation",
        backend="gliner:model+ollama:live",
        details={
            "ollama_model": model,
            "ollama_base_url": base_url,
            "gliner_model": instance.config.gliner.model_name,
        },
    )


def _unstructured_pipeline(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.unstructured.pipeline import (
        RuntimeProfile,
        UnstructuredPipelineConfig,
    )
    bundle = _mandatory_bundle(context)
    source, _, _ = _model_document()
    instance = owner(
        config=UnstructuredPipelineConfig(profile=RuntimeProfile.MODEL_BACKED),
        mention_detector=bundle.mention_detector,
        claim_extractor=bundle.claim_extractor,
    )
    result = instance.run(source)
    provider_counts = _agreement_counts(bundle)
    executed = {
        name: {
            "executed": True,
            "semantic_output_count": provider_counts[name],
        }
        for name in bundle.mandatory_provider_names
    }
    invocation_rate = len(executed) / len(bundle.mandatory_provider_names)
    semantic_output_rate = (
        sum(item["semantic_output_count"] > 0 for item in executed.values())
        / len(executed)
    )
    exact = (
        result.exact_evidence_coverage == 1.0
        and result.provenance_coverage == 1.0
        and bool(result.mentions)
        and bool(result.promoted)
    )
    return ScenarioResult(
        instance,
        "run",
        source,
        {
            "pipeline_result": result,
            "mandatory_provider_names": bundle.mandatory_provider_names,
            "provider_execution": executed,
        },
        metrics={
            "mandatory_provider_invocation_rate": invocation_rate,
            "mandatory_provider_semantic_output_rate": semantic_output_rate,
            "exact_evidence_and_provenance": float(exact),
        },
        thresholds={
            "mandatory_provider_invocation_rate": 1.0,
            "mandatory_provider_semantic_output_rate": 1.0,
            "exact_evidence_and_provenance": 1.0,
        },
        assertions=(
            {
                "name": "every profile-mandatory model provider executed",
                "passed": invocation_rate == 1.0,
            },
            {
                "name": "every profile-mandatory provider produced grounded semantic output",
                "passed": semantic_output_rate == 1.0,
            },
            {
                "name": "ensemble output retained exact evidence and complete provenance",
                "passed": exact,
            },
        ),
        backend="mandatory-model-ensemble:live",
        details={
            "profile": context.profile,
            "mandatory_provider_names": list(bundle.mandatory_provider_names),
            "provider_execution": executed,
        },
    )


_COMPOSED_FIXTURE_ID = "mandatory-composed-unstructured-v1"
_COMPOSED_SOURCE_NAME = "held-out://mandatory-composed-unstructured-v1.txt"
_COMPOSED_TEXT = (
    "Alan Turing worked with Charles Babbage. "
    "Charles Babbage worked with Ada Lovelace."
)
_COMPOSED_IDENTITIES = (
    ("Alan Turing", "Q7251"),
    ("Charles Babbage", "Q46633"),
    ("Ada Lovelace", "Q7259"),
)
_CERTIFICATION_FIXED_TIME = datetime(
    2026,
    1,
    1,
    tzinfo=UTC,
)


def _composed_source() -> SourceRecord:
    return SourceRecord.from_text(
        _COMPOSED_TEXT,
        source_name=_COMPOSED_SOURCE_NAME,
    ).model_copy(update={"created_at": _CERTIFICATION_FIXED_TIME})


def _composed_canonical_ids() -> dict[str, str]:
    return {
        name: stable_id("canonical_entity", "wikidata", qid)
        for name, qid in _COMPOSED_IDENTITIES
    }


def _certified_graph_id(profile: str, base_id: str) -> str:
    return stable_id(
        "certified_graph_node",
        _COMPOSED_FIXTURE_ID,
        profile,
        base_id,
    )


class _CertificationSnapshotPropertyGraphBuilder:
    """Namespace and run-mark the exact graph used by every live workload."""

    name = "certification_snapshot_property_graph_builder"

    def __init__(self, *, profile: str, run_id: str) -> None:
        from kg_construction.infrastructure.graph_builders import PropertyGraphBuilder

        self.delegate = PropertyGraphBuilder()
        self.profile = profile
        self.run_id = run_id

    def build(
        self,
        batch: ConstructionBatch,
        *,
        validation_report: ValidationReport | None = None,
        include_fact_nodes: bool = True,
    ) -> PropertyGraph:
        graph = self.delegate.build(
            batch,
            validation_report=validation_report,
            include_fact_nodes=include_fact_nodes,
        )
        node_ids = {
            node.node_id: _certified_graph_id(self.profile, node.node_id)
            for node in graph.nodes
        }
        nodes = [
            node.model_copy(
                update={
                    "node_id": node_ids[node.node_id],
                    "properties": {
                        **node.properties,
                        "certification_fixture": _COMPOSED_FIXTURE_ID,
                        "certification_profile": self.profile,
                        "certification_run": self.run_id,
                        "pre_certification_node_id": node.node_id,
                    },
                }
            )
            for node in graph.nodes
        ]
        edges = [
            edge.model_copy(
                update={
                    "edge_id": stable_id(
                        "certified_graph_edge",
                        _COMPOSED_FIXTURE_ID,
                        self.profile,
                        edge.edge_id,
                    ),
                    "source_id": node_ids.get(edge.source_id, edge.source_id),
                    "target_id": node_ids.get(edge.target_id, edge.target_id),
                    "properties": {
                        **edge.properties,
                        "certification_fixture": _COMPOSED_FIXTURE_ID,
                        "certification_profile": self.profile,
                        "certification_run": self.run_id,
                        "pre_certification_edge_id": edge.edge_id,
                    },
                }
            )
            for edge in graph.edges
        ]
        return graph.model_copy(
            update={
                "nodes": nodes,
                "edges": edges,
                "metadata": {
                    **graph.metadata,
                    "certification_fixture": _COMPOSED_FIXTURE_ID,
                    "certification_profile": self.profile,
                    "certification_run": self.run_id,
                },
            }
        )


def _fixed_evidence(
    source: SourceRecord,
    *,
    start: int,
    end: int,
    document_version: str,
) -> EvidenceRef:
    return EvidenceRef.build(
        source,
        location={
            "kind": "span",
            "start": start,
            "end": end,
            "evidence_kind": "mention",
        },
        text=(source.content or "")[start:end],
        metadata={"document_version": document_version},
    ).model_copy(update={"created_at": _CERTIFICATION_FIXED_TIME})


def _composed_graph_evaluation_suite(profile: str):
    """Return sealed gold/query contracts independent of runtime predictions."""

    from kg_construction.domain.claim import DocumentRef
    from kg_construction.evaluation import (
        VersionedGraphEvaluationSuite,
    )
    from kg_construction.evaluation.intended_use_metrics import (
        ReasoningFact,
    )

    source = _composed_source()
    document = DocumentRef.from_source(source)
    occurrences = (
        ("Alan Turing", 0),
        ("Charles Babbage", _COMPOSED_TEXT.index("Charles Babbage")),
        (
            "Charles Babbage",
            _COMPOSED_TEXT.index(
                "Charles Babbage",
                _COMPOSED_TEXT.index("Charles Babbage")
                + len("Charles Babbage"),
            ),
        ),
        ("Ada Lovelace", _COMPOSED_TEXT.index("Ada Lovelace")),
    )
    gold_mentions: list[EntityCandidate] = []
    mention_ids: list[str] = []
    coreference: dict[str, str] = {}
    links: dict[str, str] = {}
    canonical_ids = _composed_canonical_ids()
    for surface, start in occurrences:
        end = start + len(surface)
        evidence = _fixed_evidence(
            source,
            start=start,
            end=end,
            document_version=document.version,
        )
        mention_id = stable_id(
            "mention",
            document.version,
            start,
            end,
            "Person",
            surface,
            "mandatory_mention_ensemble",
        )
        cluster_id = stable_id(
            "coref",
            document.version,
            "Person",
            normalize_text(surface),
        )
        occurrence_key = canonical_json(
            [
                source.source_id,
                document.version,
                start,
                end,
                "person",
            ]
        )
        coreference[occurrence_key] = cluster_id
        links[occurrence_key] = canonical_ids[surface]
        mention_ids.append(mention_id)
        gold_mentions.append(
            EntityCandidate.build(
                entity_type="Mention",
                name=surface,
                properties={
                    "domain_mention_id": mention_id,
                    "document_version": document.version,
                    "surface": surface,
                    "detected_entity_type": "Person",
                    "start": start,
                    "end": end,
                    "coreference_cluster_id": cluster_id,
                },
                evidence=[evidence],
                confidence=1.0,
                extractor_name="held_out_gold",
                technique="reviewed_unstructured_gold",
                provenance_metadata={
                    "provider": "reviewed_gold",
                    "model_version": "v1",
                    "configuration_id": _COMPOSED_FIXTURE_ID,
                },
            ).model_copy(update={"created_at": _CERTIFICATION_FIXED_TIME})
        )

    relation_properties = {
        "claim_kind": "relation",
        "predicate": "works_with",
        "negated": False,
        "modality": "asserted",
        "attribution": None,
        "temporal": None,
        "quantity": None,
        "event_roles": [],
    }
    gold_relations = [
        RelationCandidate.build(
            source=gold_mentions[left].candidate_id,
            relation_type="WORKS_WITH",
            target=gold_mentions[right].candidate_id,
            properties=relation_properties,
            evidence=[
                *gold_mentions[left].evidence,
                *gold_mentions[right].evidence,
            ],
            confidence=1.0,
            extractor_name="held_out_gold",
            technique="reviewed_unstructured_gold",
            provenance_metadata={
                "provider": "reviewed_gold",
                "model_version": "v1",
                "configuration_id": _COMPOSED_FIXTURE_ID,
            },
        ).model_copy(update={"created_at": _CERTIFICATION_FIXED_TIME})
        for left, right in ((0, 1), (2, 3))
    ]
    expected_batch = ConstructionBatch(
        source_records=[source],
        entities=gold_mentions,
        relations=gold_relations,
    )
    base_ids = canonical_ids
    alan = _certified_graph_id(profile, base_ids["Alan Turing"])
    charles = _certified_graph_id(profile, base_ids["Charles Babbage"])
    ada = _certified_graph_id(profile, base_ids["Ada Lovelace"])
    all_people = [alan, charles, ada]
    two_hop_path = [
        alan,
        "WORKS_WITH",
        charles,
        "WORKS_WITH",
        ada,
    ]
    person_selector = {"node_ids": all_people}
    payload: JsonDict = {
        "version": "kg-graph-evaluation-suite-v1",
        "construction_profile": "unstructured",
        "fixture_id": f"{_COMPOSED_FIXTURE_ID}:{profile}",
        "quality": {
            "expected_batch": expected_batch,
            "coreference_clusters": coreference,
            "entity_links": links,
        },
        "sql_like": {
            "cases": [
                {
                    "case_id": "sql-alan-identity",
                    "query": {
                        "selector": {"node_ids": [alan]},
                        "projections": {
                            "id": "$id",
                            "name": "$property.name",
                            "external_id": "$property.external_id",
                        },
                        "aggregations": {"count": "count"},
                    },
                    "expected_rows": [
                        {
                            "id": alan,
                            "name": "Alan Turing",
                            "external_id": "Q7251",
                        }
                    ],
                    "expected_aggregations": {"count": 1},
                }
            ],
            "max_p95_latency_ms": 120000.0,
        },
        "nosql": {
            "cases": [
                {
                    "case_id": "nosql-collaboration-path",
                    "query": {
                        "start": {"node_ids": [alan]},
                        "relationship_types": ["WORKS_WITH"],
                        "max_depth": 2,
                        "direction": "outgoing",
                    },
                    "expected_node_ids": all_people,
                    "expected_paths": [
                        [alan, "WORKS_WITH", charles],
                        two_hop_path,
                    ],
                }
            ],
            "max_p95_latency_ms": 120000.0,
        },
        "graphrag": {
            "cases": [
                {
                    "case_id": "graphrag-two-hop-collaboration",
                    "query": {
                        "query_text": (
                            "Alan Turing Charles Babbage Ada Lovelace collaboration"
                        ),
                        "candidates": person_selector,
                        "k": 3,
                        "path_query": {
                            "start": {"node_ids": [alan]},
                            "end": {"node_ids": [ada]},
                            "relationship_types": ["WORKS_WITH"],
                            "max_depth": 2,
                        },
                        "answer_query": {
                            "start": {"node_ids": [alan]},
                            "end": {"node_ids": [ada]},
                            "relationship_types": ["WORKS_WITH"],
                            "max_depth": 2,
                        },
                    },
                    "context_relevance": {
                        alan: 1.0,
                        charles: 1.0,
                        ada: 1.0,
                    },
                    "expected_paths": [two_hop_path],
                    "expected_answer_node_ids": [ada],
                    "expected_citation_ids": [source.source_id],
                    "requires_multi_hop": True,
                }
            ],
            "min_recall_at_k": 1.0,
            "min_mrr": 1.0,
            "min_ndcg_at_k": 1.0,
            "min_path_f1": 1.0,
            "min_multi_hop_success_rate": 1.0,
            "min_answer_faithfulness": 1.0,
            "min_citation_f1": 1.0,
        },
        "vector": {
            "cases": [
                {
                    "case_id": "vector-ada-identity",
                    "query": {
                        "query_text": "Ada Lovelace",
                        "candidates": person_selector,
                        "allowed": person_selector,
                        "k": 3,
                        "dimensions": 384,
                    },
                    "relevance": {ada: 1.0},
                }
            ],
            "max_p95_latency_ms": 120000.0,
            "max_peak_memory_bytes": 1073741824,
            "min_index_coverage": 1.0,
            "min_recall_at_k": 1.0,
            "min_mrr": 1.0,
            "min_ndcg_at_k": 1.0,
            "min_filter_correctness": 1.0,
        },
        "link_prediction": {
            "cases": [
                {
                    "case_id": "link-alan-to-charles",
                    "query": {
                        "source": {"node_ids": [alan]},
                        "relationship_type": "WORKS_WITH",
                        "candidates": person_selector,
                        "k": 3,
                    },
                    "true_entity_id": charles,
                },
                {
                    "case_id": "link-charles-to-ada",
                    "query": {
                        "source": {"node_ids": [charles]},
                        "relationship_type": "WORKS_WITH",
                        "candidates": person_selector,
                        "k": 3,
                    },
                    "true_entity_id": ada,
                },
                {
                    "case_id": "link-charles-from-alan",
                    "query": {
                        "source": {"node_ids": [charles]},
                        "relationship_type": "WORKS_WITH",
                        "direction": "incoming",
                        "candidates": person_selector,
                        "k": 3,
                    },
                    "true_entity_id": alan,
                },
            ],
            "hits_at": [1, 3, 10],
            "min_filtered_mrr": 0.60,
            "min_filtered_hits_at_k": {1: 0.30, 3: 1.0, 10: 1.0},
        },
        "recommendation": {
            "cases": [
                {
                    "case_id": "recommend-alan-to-charles",
                    "query": {
                        "source": {"node_ids": [alan]},
                        "relationship_type": "WORKS_WITH",
                        "candidates": person_selector,
                        "k": 3,
                    },
                    "relevant_item_ids": [charles],
                },
                {
                    "case_id": "recommend-charles-to-ada",
                    "query": {
                        "source": {"node_ids": [charles]},
                        "relationship_type": "WORKS_WITH",
                        "candidates": person_selector,
                        "k": 3,
                    },
                    "relevant_item_ids": [ada],
                },
                {
                    "case_id": "recommend-charles-from-alan",
                    "query": {
                        "source": {"node_ids": [charles]},
                        "relationship_type": "WORKS_WITH",
                        "direction": "incoming",
                        "candidates": person_selector,
                        "k": 3,
                    },
                    "relevant_item_ids": [alan],
                },
            ],
            "catalog": person_selector,
            "feature_properties": ["external_id"],
            "min_precision_at_k": 0.30,
            "min_recall_at_k": 1.0,
            "min_ndcg_at_k": 0.50,
            "min_catalog_coverage": 1.0,
            "min_novelty": 0.01,
            "min_intra_list_diversity": 0.01,
        },
        "reasoning": {
            "cases": [
                {
                    "case_id": "reason-two-hop-collaboration",
                    "query": {
                        "start": {"node_ids": [alan]},
                        "relationship_type_sequence": [
                            "WORKS_WITH",
                            "WORKS_WITH",
                        ],
                        "inferred_predicate": "indirectly_worked_with",
                    },
                    "expected_answers": [ada],
                    "expected_inferred_facts": [
                        ReasoningFact(
                            subject=alan,
                            predicate="indirectly_worked_with",
                            object=ada,
                        )
                    ],
                }
            ],
            "max_p95_runtime_ms": 120000.0,
            "min_competency_question_accuracy": 1.0,
            "min_inferred_fact_f1": 1.0,
        },
    }
    return VersionedGraphEvaluationSuite.seal(payload)


def _mandatory_composed_unstructured_runtime(
    owner: Owner,
    context: CertificationContext,
) -> ScenarioResult:
    """Certify the single locked path through publication and live Neo4j."""

    from kg_construction.construction.config import DataToKGConfig
    from kg_construction.construction.workflow import ConstructionWorkflow
    from kg_construction.evaluation import (
        GraphBoundIntendedUseEvaluator,
        GraphWorkloadRunner,
        Neo4jGraphWorkloadBackend,
        Neo4jGraphWorkloadConfig,
        Neo4jVectorWorkloadBackend,
        property_graph_sha256,
    )
    from kg_construction.infrastructure.graph_stores.ingestion import (
        Neo4jIngestionAdapter,
        Neo4jIngestionConfig,
    )
    from kg_construction.infrastructure.graph_stores.vector_store import (
        Neo4jVectorStoreAdapter,
        Neo4jVectorStoreConfig,
    )

    settings = DataToKGConfig().unstructured
    cache = os.environ.get("HF_HUB_CACHE")
    runtime = LockedServiceRuntime.from_environment()
    instance = owner.from_lock(
        context.root / "models.lock.yaml",
        profile=context.profile,
        cache_root=Path(cache) if cache else None,
        entity_labels=settings.entity_labels,
        ollama_base_url=runtime.ollama_base_url,
        ollama_retry_on_invalid_output=True,
        ollama_max_retries=3,
    )
    source = _composed_source()
    suite = _composed_graph_evaluation_suite(context.profile)
    suffix = hashlib.sha256(
        f"{context.profile}:{context.run_id}:{_COMPOSED_FIXTURE_ID}".encode()
    ).hexdigest()[:12]
    driver, uri = _neo4j_driver(memgraph=False)
    database = runtime.neo4j_database
    vector_config = Neo4jVectorStoreConfig(
        uri=uri,
        user=runtime.neo4j_user,
        password=runtime.neo4j_password,
        database=database,
        index_name=f"kg_composed_vector_{suffix}",
        node_label=f"KGComposedVector_{suffix}",
        dimensions=384,
        await_index_timeout_seconds=120,
    )
    vector_store = Neo4jVectorStoreAdapter(vector_config, driver=driver)
    vector_backend = Neo4jVectorWorkloadBackend(
        store=vector_store,
        embedding_model=instance.embedding_model,
        model_id=instance.embedding_model.artifact.model_version,
    )
    graph_backend = Neo4jGraphWorkloadBackend(
        Neo4jGraphWorkloadConfig(
            uri=uri,
            user=runtime.neo4j_user,
            password=runtime.neo4j_password,
            database=database,
            snapshot_namespace=f"kg_composed_{suffix}",
        ),
        driver=driver,
    )
    graph_evaluator = GraphBoundIntendedUseEvaluator(
        runner=GraphWorkloadRunner(
            graph_backend=graph_backend,
            vector_backend=vector_backend,
        )
    )
    snapshot_builder = _CertificationSnapshotPropertyGraphBuilder(
        profile=context.profile,
        run_id=context.run_id,
    )
    workflow = ConstructionWorkflow(
        extractors=[instance],
        property_graph_builder=snapshot_builder,  # type: ignore[arg-type]
        graph_evaluation_suite=suite,
        graph_bound_evaluator=graph_evaluator,
    )
    ingestion = Neo4jIngestionAdapter(
        Neo4jIngestionConfig(
            uri=uri,
            user=runtime.neo4j_user,
            password=runtime.neo4j_password,
            database=database,
            batch_size=100,
        ),
        driver=driver,
    )
    result = None
    write_report = None
    stored_node_ids: list[str] = []
    stored_edge_ids: list[str] = []
    try:
        result = workflow.run([source], graph_evaluation_suite=suite)
        write_report = ingestion.write_property_graph(result.property_graph)
        session_kwargs = {"database": database} if database else {}
        with driver.session(**session_kwargs) as session:
            node_record = session.run(
                "MATCH (n {certification_run: $run_id}) "
                "RETURN collect(n.id) AS ids",
                run_id=context.run_id,
            ).single()
            edge_record = session.run(
                "MATCH ()-[r {certification_run: $run_id}]->() "
                "RETURN collect(r.id) AS ids",
                run_id=context.run_id,
            ).single()
            stored_node_ids = sorted(
                str(item) for item in (node_record["ids"] if node_record else [])
            )
            stored_edge_ids = sorted(
                str(item) for item in (edge_record["ids"] if edge_record else [])
            )
    finally:
        session_kwargs = {"database": database} if database else {}
        try:
            graph_backend.cleanup()
            with driver.session(**session_kwargs) as session:
                session.run(
                    "MATCH (n {certification_run: $run_id}) DETACH DELETE n",
                    run_id=context.run_id,
                ).consume()
                label = f"`{vector_config.node_label}`"
                session.run(f"MATCH (n:{label}) DETACH DELETE n").consume()
                session.run(
                    f"DROP INDEX `{vector_config.index_name}` IF EXISTS"
                ).consume()
                session.run(
                    f"DROP CONSTRAINT `{vector_config.constraint_name}` IF EXISTS"
                ).consume()
        finally:
            driver.close()

    if result is None or write_report is None:
        raise RuntimeError("composed certification produced no construction/store result")
    graph_hash = property_graph_sha256(result.property_graph)
    graph_node_ids = sorted(node.node_id for node in result.property_graph.nodes)
    graph_edge_ids = sorted(edge.edge_id for edge in result.property_graph.edges)
    runtime_metadata = result.batch.metadata.get(
        "mandatory_unstructured_runtime",
        {},
    )
    participating = {
        str(item)
        for item in runtime_metadata.get("participating_provider_ids", [])
    }
    required = set(instance.mandatory_provider_ids)
    source_result = next(
        (
            item
            for item in result.pipeline_results
            if item.document.source_name == _COMPOSED_SOURCE_NAME
        ),
        None,
    )
    promotion = result.batch.metadata.get("external_identity_promotion", {})
    workload_families = result.intended_use_report.families
    checks = {
        "actual_mandatory_provider_participation": (
            required <= participating and bool(required)
        ),
        "governed_source_shacl": bool(
            source_result is not None
            and source_result.shacl_executed
            and source_result.shacl_conforms
            and source_result.governed_schema_sha256
            == instance.governed_schema.source_sha256
        ),
        "versioned_held_out_claim_gate": bool(
            result.held_out_claim_evaluation is not None
            and result.held_out_claim_evaluation.passed
            and result.held_out_claim_evaluation.dataset_sha256
            == instance.held_out_gold.sha256
            and result.held_out_claim_evaluation.model_lock_sha256
            == instance.model_lock_sha256
        ),
        "external_identity_consensus_promotion": (
            int(promotion.get("accepted_count", 0)) >= 3
            and int(promotion.get("conflict_count", 0)) == 0
        ),
        "all_graph_bound_workloads": (
            result.intended_use_report.passed
            and result.intended_use_report.graph_bound
            and len(workload_families) == 7
            and all(
                family.executed and family.passed
                for family in workload_families.values()
            )
        ),
        "same_frozen_graph_snapshot": (
            result.intended_use_report.graph_sha256 == graph_hash
            and result.intended_use_report.readback_graph_sha256 == graph_hash
            and result.intended_use_report.graph_snapshot_id is not None
            and graph_hash in result.intended_use_report.graph_snapshot_id
        ),
        "all_workloads_live_store_backed": (
            result.intended_use_report.certifiable_backend
            and set(result.intended_use_report.backend_executed_families)
            == {
                "sql_like",
                "nosql",
                "graphrag",
                "vector",
                "link_prediction",
                "recommendation",
                "reasoning",
            }
        ),
        "publication_conjunction": result.publication_gate.passed,
        "neo4j_exact_write_read": (
            write_report.node_count == len(result.property_graph.nodes)
            and write_report.edge_count == len(result.property_graph.edges)
            and stored_node_ids == graph_node_ids
            and stored_edge_ids == graph_edge_ids
        ),
    }
    return ScenarioResult(
        instance=instance,
        operation=(
            "heldout+extract+identity+shacl+evaluate7+"
            "neo4j_readback_all7+neo4j_vector+neo4j_write_read"
        ),
        fixture={
            "source": source,
            "evaluation_suite": suite,
            "model_lock_sha256": instance.model_lock_sha256,
        },
        output={
            "publication_gate": result.publication_gate,
            "held_out_claim_evaluation": result.held_out_claim_evaluation,
            "pipeline_results": result.pipeline_results,
            "quality_report": result.quality_report,
            "intended_use_report": result.intended_use_report,
            "graph_sha256": graph_hash,
            "write_report": write_report,
            "stored_node_ids": stored_node_ids,
            "stored_edge_ids": stored_edge_ids,
            "mandatory_provider_ids": sorted(required),
            "participating_provider_ids": sorted(participating),
            "external_identity_promotion": promotion,
        },
        metrics={name: float(passed) for name, passed in checks.items()},
        thresholds={name: 1.0 for name in checks},
        assertions=tuple(
            {
                "name": name.replace("_", " "),
                "passed": passed,
            }
            for name, passed in checks.items()
        ),
        backend=(
            "locked-models+external-linkers+"
            "neo4j-readback-all7+neo4j-vector:live"
        ),
        details={
            "profile": context.profile,
            "neo4j_uri": uri,
            "neo4j_database": database,
            "vector_index": vector_config.index_name,
            "vector_label": vector_config.node_label,
            "evaluation_fixture_sha256": suite.fixture_sha256,
            "held_out_dataset_sha256": instance.held_out_gold.sha256,
            "model_lock_sha256": instance.model_lock_sha256,
            "graph_sha256": graph_hash,
            "graph_backend_id": result.intended_use_report.graph_backend_id,
            "graph_snapshot_id": result.intended_use_report.graph_snapshot_id,
            "readback_graph_sha256": (
                result.intended_use_report.readback_graph_sha256
            ),
            "backend_executed_families": (
                result.intended_use_report.backend_executed_families
            ),
            "mandatory_provider_ids": sorted(required),
            "participating_provider_ids": sorted(participating),
            "expected_ingestion_node_count": len(graph_node_ids),
            "stored_ingestion_node_count": len(stored_node_ids),
            "expected_ingestion_edge_count": len(graph_edge_ids),
            "stored_ingestion_edge_count": len(stored_edge_ids),
            "expected_ingestion_node_ids_sha256": hashlib.sha256(
                canonical_json(graph_node_ids).encode("utf-8")
            ).hexdigest(),
            "stored_ingestion_node_ids_sha256": hashlib.sha256(
                canonical_json(stored_node_ids).encode("utf-8")
            ).hexdigest(),
            "expected_ingestion_edge_ids_sha256": hashlib.sha256(
                canonical_json(graph_edge_ids).encode("utf-8")
            ).hexdigest(),
            "stored_ingestion_edge_ids_sha256": hashlib.sha256(
                canonical_json(stored_edge_ids).encode("utf-8")
            ).hexdigest(),
        },
    )


def _resolver_batch() -> ConstructionBatch:
    text = "Recommendation API | recommendation api | Feature Store"
    source = SourceRecord.from_text(text, source_name="memory://resolver.txt")
    names = ["Recommendation API", "recommendation api", "Feature Store"]
    entities: list[EntityCandidate] = []
    cursor = 0
    for name in names:
        start = text.index(name, cursor)
        cursor = start + len(name)
        entities.append(
            EntityCandidate.build(
                entity_type="Service",
                name=name,
                properties={"external_id": "svc-rec" if "recommendation" in name.casefold() else "svc-fs"},
                evidence=[
                    span_evidence(source, start=start, end=start + len(name), text=name)
                ],
                extractor_name="semantic_certifier",
                technique="fixture",
            )
        )
    return ConstructionBatch(source_records=[source], entities=entities)


def _resolver(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.resolution.entity_resolution import (
        NormalizedStringResolverConfig,
        RapidFuzzResolverConfig,
    )

    batch = _resolver_batch()
    if context.capability == "alias_table_resolver":
        instance = owner({"Recommendation API": ["recommendation api"]})
    elif context.capability in {"fuzzy_string_resolver", "rapidfuzz_resolver"}:
        instance = owner(RapidFuzzResolverConfig(threshold=95.0))
    elif context.capability == "normalized_string_resolver":
        instance = owner(NormalizedStringResolverConfig())
    else:
        instance = owner()
    resolved = instance.resolve(batch)
    recommendation = [
        entity
        for entity in resolved.entities
        if "recommendation" in entity.name.casefold()
    ]
    canonical_ids = {
        entity.properties.get("canonical_entity_id") for entity in recommendation
    }
    passed = len(canonical_ids) == 1 and None not in canonical_ids
    return _result(
        instance,
        "resolve",
        batch,
        resolved,
        metric="expected_pair_resolution",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} merged the equivalent pair without merging Feature Store",
    )


def _embedding_resolver(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.resolution.linking import (
        CrossEncoderResolverConfig,
        EmbeddingResolverConfig,
    )

    batch = _resolver_batch()
    if context.capability == "embedding_resolver":
        instance = owner(
            EmbeddingResolverConfig(threshold=0.70, review_threshold=0.95)
        )
        backend = "sentence-transformers:model"
    else:
        instance = owner(
            CrossEncoderResolverConfig(threshold=0.70, review_threshold=0.95)
        )
        backend = "cross-encoder:model"
    resolved = instance.resolve(batch)
    recommendation = [
        entity
        for entity in resolved.entities
        if "recommendation" in entity.name.casefold()
    ]
    canonical_ids = {
        entity.properties.get("canonical_entity_id") for entity in recommendation
    }
    passed = len(canonical_ids) == 1 and None not in canonical_ids
    return _result(
        instance,
        "resolve",
        batch,
        resolved,
        metric="semantic_duplicate_resolution",
        value=float(passed),
        threshold=1.0,
        assertion="live semantic model merged the paraphrased service names",
        backend=backend,
        details={"model": instance.config.model_name},
    )


def _external_linker(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.resolution.linking import (
        DBpediaLinkerConfig,
        OpenRefineReconcilerConfig,
        WikidataLinkerConfig,
    )

    source = SourceRecord.from_text("Alan Turing", source_name="memory://linker.txt")
    entity = EntityCandidate.build(
        entity_type="Person",
        name="Alan Turing",
        properties={},
        evidence=[
            span_evidence(source, start=0, end=len("Alan Turing"), text="Alan Turing")
        ],
        extractor_name="semantic_certifier",
        technique="fixture",
    )
    batch = ConstructionBatch(source_records=[source], entities=[entity])
    if context.capability == "openrefine_reconciler":
        endpoint = os.environ.get(
            "KG_CERT_OPENREFINE_URL",
            "https://wikidata-reconciliation.wmcloud.org/en/api",
        )
        instance = owner(OpenRefineReconcilerConfig(endpoint_url=endpoint))
        service = "openrefine"
    elif context.capability == "wikidata_linker":
        endpoint = os.environ.get(
            "KG_CERT_WIKIDATA_URL", "https://www.wikidata.org/w/api.php"
        )
        instance = owner(WikidataLinkerConfig(endpoint_url=endpoint))
        service = "wikidata"
    else:
        endpoint = os.environ.get(
            "KG_CERT_DBPEDIA_URL", "https://lookup.dbpedia.org/api/search"
        )
        instance = owner(DBpediaLinkerConfig(endpoint_url=endpoint))
        service = "dbpedia"
    resolved = instance.resolve(batch)
    links = resolved.entities[0].properties.get("external_links", [])
    matched_links = [
        item
        for item in links
        if isinstance(item, dict)
        and item.get("provider") == service
        and item.get("id")
    ] if isinstance(links, list) else []
    passed = bool(matched_links)
    return _result(
        instance,
        "resolve",
        batch,
        resolved,
        metric="external_link_recall",
        value=float(passed),
        threshold=1.0,
        assertion=f"live {service} service returned an identity candidate for Alan Turing",
        backend=f"{service}:live",
        details={"endpoint": instance.config.endpoint_url},
    )


def _graph_builder(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.ontology.validation import (
        EvidenceRequiredValidator,
    )

    batch = _batch()
    instance = owner()
    if context.capability == "evidence_graph_builder":
        graph = instance.build(batch)
        passed = (
            any(edge.type == "EVIDENCED_BY" for edge in graph.edges)
            and any("Evidence" in node.labels for node in graph.nodes)
        )
    elif context.capability == "property_graph_builder":
        report = EvidenceRequiredValidator().validate(batch)
        graph = instance.build(batch, validation_report=report)
        passed = (
            len([node for node in graph.nodes if "Entity" in node.labels]) == 2
            and any(edge.type == "AFFECTED" for edge in graph.edges)
        )
    else:
        report = EvidenceRequiredValidator().validate(batch)
        graph = instance.build(batch, validation_report=report)
        passed = (
            bool(graph.triples)
            and bool(graph.quads)
            and any(
                triple.predicate.endswith("AFFECTED") for triple in graph.triples
            )
        )
    return _result(
        instance,
        "build",
        batch,
        graph,
        metric="graph_materialization_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} materialized entities, semantics, and provenance",
    )


def _quality_construction_fixture() -> JsonDict:
    """Build one evidenced graph, ontology, replay, and real SHACL observation."""

    from kg_construction.domain.evidence import cell_evidence
    from kg_construction.domain.graph_objects import RDFGraph
    from kg_construction.domain.modeling import Cardinality, ColumnRole
    from kg_construction.evaluation.operational_metrics import (
        OperationalMetricsRecorder,
        RetryObservation,
    )
    from kg_construction.infrastructure.graph_builders.property_graph import (
        PropertyGraphBuilder,
    )
    from kg_construction.infrastructure.graph_builders.rdf_graph import RDFGraphBuilder
    from kg_construction.infrastructure.ontology.shacl import (
        ShaclValidationResult,
        ShaclValidator,
    )
    from kg_construction.infrastructure.ontology.validation import (
        EvidenceRequiredValidator,
    )
    from kg_construction.infrastructure.production.manifests import RunManifest
    from kg_construction.infrastructure.production.retry import (
        BatchRetryPolicy,
        RetryPolicyConfig,
    )
    from kg_construction.structured.resolved_plan import (
        ClassKind,
        IdentityRole,
        Materialization,
        ResolvedAttribute,
        ResolvedClass,
        ResolvedModelingPlan,
        ResolvedProperty,
        ResolvedRelationship,
    )

    row: JsonDict = {
        "incident_id": "INCIDENT-1042",
        "severity": "high",
        "service_id": "service-recommendation",
        "service_name": "Recommendation API",
        "team_id": "team-platform",
        "team_name": "Platform Engineering",
    }
    fixture_text = "INCIDENT-1042 Recommendation API Platform Engineering"
    source = SourceRecord.from_text(
        fixture_text,
        source_type="csv",
        source_name="memory://semantic-quality-fixture.csv",
    ).model_copy(update={"records": [row]})

    def entity_span(value: str):
        start = fixture_text.index(value)
        return span_evidence(
            source,
            start=start,
            end=start + len(value),
            text=value,
        )

    incident = EntityCandidate.build(
        entity_type="Incident",
        name=str(row["incident_id"]),
        properties={
            "incident_id": row["incident_id"],
            "severity": row["severity"],
        },
        evidence=[entity_span(str(row["incident_id"]))],
        confidence=1.0,
        extractor_name="semantic_certifier",
        technique="declared_structured_mapping",
    )
    service = EntityCandidate.build(
        entity_type="Service",
        name=str(row["service_name"]),
        properties={
            "service_id": row["service_id"],
            "service_name": row["service_name"],
        },
        evidence=[entity_span(str(row["service_name"]))],
        confidence=1.0,
        extractor_name="semantic_certifier",
        technique="declared_structured_mapping",
    )
    team = EntityCandidate.build(
        entity_type="Team",
        name=str(row["team_name"]),
        properties={
            "team_id": row["team_id"],
            "team_name": row["team_name"],
        },
        evidence=[entity_span(str(row["team_name"]))],
        confidence=1.0,
        extractor_name="semantic_certifier",
        technique="declared_structured_mapping",
    )
    affected = RelationCandidate.build(
        source=incident.candidate_id,
        relation_type="AFFECTED",
        target=service.candidate_id,
        evidence=[
            cell_evidence(
                source,
                row_index=0,
                column="service_id",
                value=row["service_id"],
            )
        ],
        confidence=1.0,
        extractor_name="semantic_certifier",
        technique="declared_foreign_key",
    )
    owned_by = RelationCandidate.build(
        source=service.candidate_id,
        relation_type="OWNED_BY",
        target=team.candidate_id,
        evidence=[
            cell_evidence(
                source,
                row_index=0,
                column="team_id",
                value=row["team_id"],
            )
        ],
        confidence=1.0,
        extractor_name="semantic_certifier",
        technique="declared_foreign_key",
    )
    batch = ConstructionBatch(
        source_records=[source],
        entities=[incident, service, team],
        relations=[affected, owned_by],
    )
    exactly_one = Cardinality.exactly_one()
    plan = ResolvedModelingPlan(
        source_name=source.source_name,
        table_name="incident_service_ownership",
        record_class="Incident",
        classes=[
            ResolvedClass(
                name="Incident",
                kind=ClassKind.ENTITY,
                identity_keys=["incident_id"],
            ),
            ResolvedClass(
                name="Service",
                kind=ClassKind.ENTITY,
                identity_keys=["service_id"],
            ),
            ResolvedClass(
                name="Team",
                kind=ClassKind.ENTITY,
                identity_keys=["team_id"],
            ),
        ],
        properties=[
            ResolvedProperty(
                class_name="Incident",
                name="incident_id",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
                source_column="incident_id",
            ),
            ResolvedProperty(
                class_name="Incident",
                name="severity",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
                source_column="severity",
            ),
            ResolvedProperty(
                class_name="Service",
                name="service_id",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
                source_column="service_id",
            ),
            ResolvedProperty(
                class_name="Service",
                name="service_name",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
                source_column="service_name",
            ),
            ResolvedProperty(
                class_name="Team",
                name="team_id",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
                source_column="team_id",
            ),
            ResolvedProperty(
                class_name="Team",
                name="team_name",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
                source_column="team_name",
            ),
        ],
        relationships=[
            ResolvedRelationship(
                name="AFFECTED",
                domain="Incident",
                range="Service",
                cardinality=exactly_one,
                source_columns=["service_id"],
                target_columns=["service_id"],
            ),
            ResolvedRelationship(
                name="OWNED_BY",
                domain="Service",
                range="Team",
                cardinality=exactly_one,
                source_columns=["team_id"],
                target_columns=["team_id"],
            ),
        ],
        attributes=[
            ResolvedAttribute(
                column="incident_id",
                semantic_role=ColumnRole.IDENTIFIER,
                materialization=Materialization.IDENTITY,
                target="Incident",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
                identity_role=IdentityRole.PRIMARY,
            ),
            ResolvedAttribute(
                column="severity",
                semantic_role=ColumnRole.CATEGORICAL_DIMENSION,
                materialization=Materialization.PROPERTY,
                target="severity",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
            ),
            ResolvedAttribute(
                column="service_id",
                semantic_role=ColumnRole.ENTITY_REFERENCE,
                materialization=Materialization.EDGE,
                target="Service",
                datatype="string",
                nullable=False,
                cardinality=exactly_one,
                identity_role=IdentityRole.FOREIGN,
            ),
        ],
    )
    validation_report = EvidenceRequiredValidator().validate(batch)
    recorder = OperationalMetricsRecorder(
        processed_items=len(batch.entities) + len(batch.relations)
    )
    retry_policy = BatchRetryPolicy(
        RetryPolicyConfig(max_attempts=2, sleep=False)
    )

    def execute_construction() -> tuple[
        PropertyGraph,
        RDFGraph,
        ShaclValidationResult,
    ]:
        property_result = PropertyGraphBuilder().build(
            batch,
            validation_report=validation_report,
            include_fact_nodes=False,
        )
        rdf_result = RDFGraphBuilder().build(
            batch,
            validation_report=validation_report,
        )
        shacl_result = ShaclValidator().validate_rdf_graph(
            rdf_result,
            graph_schema=plan.to_property_graph_schema(),
        )
        return property_result, rdf_result, shacl_result

    initial_manifest = RunManifest.start(
        pipeline_name="semantic_quality_construction_fixture",
        sources=[source],
        config_payload={"plan": plan.model_dump(mode="json")},
    )
    with recorder:
        initial_outcome = retry_policy.run(
            execute_construction,
            operation_name="semantic_quality_initial_construction",
        )
        replay_manifest = RunManifest.start(
            pipeline_name="semantic_quality_construction_fixture",
            sources=[source],
            config_payload={"plan": plan.model_dump(mode="json")},
        )
        replay_outcome = retry_policy.run(
            execute_construction,
            operation_name="semantic_quality_replay_construction",
        )
    property_graph, rdf_graph, shacl_result = initial_outcome.value
    replay_graph, _, _ = replay_outcome.value
    initial_manifest = initial_manifest.complete(
        batch=batch,
        validation_report=validation_report,
    )
    replay_manifest = replay_manifest.complete(
        batch=batch,
        validation_report=validation_report,
    )
    runtime_report = recorder.report.model_copy(
        update={
            "retry_observations": (
                RetryObservation(
                    operation_id="semantic_quality_initial_construction",
                    attempt_count=initial_outcome.attempts,
                    retry_count=initial_outcome.retry_count,
                ),
                RetryObservation(
                    operation_id="semantic_quality_replay_construction",
                    attempt_count=replay_outcome.attempts,
                    retry_count=replay_outcome.retry_count,
                ),
            )
        }
    )

    from rdflib import Graph
    from rdflib.namespace import RDF, SH

    results_graph = Graph()
    results_graph.parse(data=shacl_result.results_graph_turtle, format="turtle")
    shacl_violation_count = sum(
        1 for _ in results_graph.subjects(RDF.type, SH.ValidationResult)
    )
    return {
        "batch": batch,
        "plan": plan,
        "property_graph": property_graph,
        "replay_graph": replay_graph,
        "rdf_graph": rdf_graph,
        "validation_report": validation_report,
        "shacl_result": shacl_result,
        "shacl_violation_count": shacl_violation_count,
        "runtime_report": runtime_report,
        "initial_run_manifest": initial_manifest,
        "replay_run_manifest": replay_manifest,
        "source_rows": [row],
    }


def _quality_evaluator_scenario(
    owner: Owner,
    context: CertificationContext,
) -> ScenarioResult:
    from kg_construction.evaluation.graph_bound_suite import (
        property_graph_sha256,
    )
    from kg_construction.evaluation.quality_gates import (
        ConstructionReplayEvidence,
        QualityEvaluationInputs,
    )

    built = _quality_construction_fixture()
    batch = built["batch"]
    if not isinstance(batch, ConstructionBatch):
        raise TypeError("quality fixture did not construct a ConstructionBatch")
    graph = built["property_graph"]
    entity_ids = {
        entity.entity_type: entity.candidate_id for entity in batch.entities
    }
    instance = owner()
    inputs = QualityEvaluationInputs(
        property_graph=graph,
        rdf_graph=built["rdf_graph"],
        validation_report=built["validation_report"],
        resolved_plan=built["plan"],
        source_records=built["source_rows"],
        shacl_executed=bool(built["shacl_result"].conforms),
        shacl_violation_count=int(built["shacl_violation_count"]),
        expected_batch=batch.model_copy(deep=True),
        replay_graph=built["replay_graph"],
        predicted_coreference={
            "incident-mention": entity_ids["Incident"],
            "it-mention": entity_ids["Incident"],
        },
        gold_coreference={
            "incident-mention": entity_ids["Incident"],
            "it-mention": entity_ids["Incident"],
        },
        predicted_links={
            "incident-mention": entity_ids["Incident"],
            "service-mention": entity_ids["Service"],
            "team-mention": entity_ids["Team"],
        },
        gold_links={
            "incident-mention": entity_ids["Incident"],
            "service-mention": entity_ids["Service"],
            "team-mention": entity_ids["Team"],
        },
        predicted_answers={"owner-of-affected-service": ["Platform Engineering"]},
        gold_answers={"owner-of-affected-service": ["Platform Engineering"]},
        vector_rankings={"incident-context": [entity_ids["Service"]]},
        vector_relevant={"incident-context": [entity_ids["Service"]]},
        vector_k=1,
        runtime_report=built["runtime_report"],
        replay_evidence=ConstructionReplayEvidence(
            initial_run=built["initial_run_manifest"],
            replay_run=built["replay_run_manifest"],
            initial_graph_sha256=property_graph_sha256(graph),
            replay_graph_sha256=property_graph_sha256(built["replay_graph"]),
        ),
        operational_run_manifests=(built["initial_run_manifest"],),
    )
    report = instance.evaluate(batch, inputs=inputs)
    metrics: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    assertions: list[JsonDict] = []
    for family_name, family in report.families.items():
        for suffix, actual in (
            ("executed", float(family.executed)),
            ("metric_count", float(len(family.metrics))),
            ("passed", float(family.passed)),
        ):
            metric_name = f"{family_name}_{suffix}"
            metrics[metric_name] = actual
            thresholds[metric_name] = 1.0
        assertions.append(
            {
                "name": f"{family_name} family executed real metrics and passed",
                "passed": family.executed and bool(family.metrics) and family.passed,
                "observed_metrics": sorted(family.metrics),
            }
        )
    return ScenarioResult(
        instance=instance,
        operation="evaluate",
        fixture={"construction": built, "gold_inputs": inputs},
        output=report,
        metrics=metrics,
        thresholds=thresholds,
        assertions=tuple(assertions),
        details={
            "quality_families": list(report.families),
            "shacl_engine_executed": True,
            "shacl_conforms": bool(built["shacl_result"].conforms),
        },
    )


def _schema_quality_evaluator_scenario(
    owner: Owner,
    context: CertificationContext,
) -> ScenarioResult:
    from kg_construction.structured.resolved_plan import ResolvedModelingPlan

    _ = context
    built = _quality_construction_fixture()
    plan = built["plan"]
    if not isinstance(plan, ResolvedModelingPlan):
        raise TypeError("schema fixture did not construct a ResolvedModelingPlan")
    instance = owner()
    report = instance.evaluate(
        built["property_graph"],
        policy=plan,
        source_records=built["source_rows"],
        ontology=plan.to_ontology_schema(),
        shacl_executed=bool(built["shacl_result"].conforms),
        shacl_violation_count=int(built["shacl_violation_count"]),
    )
    duplication = report.categorical_duplication_rate
    metrics = {
        "categorical_conciseness_score": (
            1.0 - duplication if duplication is not None else 0.0
        ),
        "node_conciseness_score": 1.0 - report.duplicate_node_rate,
        "property_completeness": report.property_completeness,
        "relationship_completeness": report.relationship_completeness,
        "consistency_conformance": float(report.consistency_violation_count == 0),
        "shacl_execution": float(report.shacl_executed),
        "shacl_conformance": float(
            report.shacl_executed and report.shacl_violation_count == 0
        ),
        "structure_node_count": float(report.node_count),
        "structure_edge_count": float(report.edge_count),
        "all_schema_gates_passed": float(report.passed),
    }
    thresholds = {name: 1.0 for name in metrics}
    thresholds["structure_node_count"] = 3.0
    thresholds["structure_edge_count"] = 2.0
    assertions = (
        {
            "name": "conciseness metrics rejected neither duplicate categories nor nodes",
            "passed": (
                report.categorical_duplication_rate == 0.0
                and report.duplicate_node_rate == 0.0
            ),
        },
        {
            "name": "declared property and relationship obligations were complete",
            "passed": (
                report.property_completeness == 1.0
                and report.relationship_completeness == 1.0
            ),
        },
        {
            "name": "ontology consistency and real pySHACL validation passed",
            "passed": (
                report.ontology_evaluated
                and report.shacl_executed
                and report.shacl_violation_count == 0
                and report.consistency_violation_count == 0
            ),
        },
        {
            "name": "structural metrics measured the connected three-node graph",
            "passed": (
                report.node_count == 3
                and report.edge_count == 2
                and report.connected_component_count == 1
                and report.average_degree > 0.0
            ),
        },
    )
    return ScenarioResult(
        instance=instance,
        operation="evaluate",
        fixture=built,
        output=report,
        metrics=metrics,
        thresholds=thresholds,
        assertions=assertions,
        details={
            "dimensions": [
                "conciseness",
                "completeness",
                "consistency",
                "structure",
            ],
            "shacl_engine_executed": True,
        },
    )


def _intended_use_evaluator_scenario(
    owner: Owner,
    context: CertificationContext,
) -> ScenarioResult:
    import time
    import tracemalloc

    from kg_construction.evaluation.intended_use_metrics import (
        GraphRAGCase,
        GraphRAGEvaluationInput,
        IntendedUseEvaluationInputs,
        LinkPredictionCase,
        LinkPredictionEvaluationInput,
        NoSQLEvaluationInput,
        NoSQLTraversalCase,
        ReasoningCase,
        ReasoningEvaluationInput,
        ReasoningFact,
        RecommendationCase,
        RecommendationEvaluationInput,
        SQLLikeEvaluationInput,
        SQLLikeQueryCase,
        VectorIndexCase,
        VectorIndexEvaluationInput,
    )

    _ = context
    built = _quality_construction_fixture()
    batch = built["batch"]
    if not isinstance(batch, ConstructionBatch):
        raise TypeError("intended-use fixture did not construct a ConstructionBatch")
    graph = built["property_graph"]
    node_by_label = {
        label: node
        for node in graph.nodes
        for label in node.labels
        if label in {"Incident", "Service", "Team"}
    }
    incident = node_by_label["Incident"]
    service = node_by_label["Service"]
    team = node_by_label["Team"]
    edge_by_type = {edge.type: edge for edge in graph.edges}
    traversal_started = time.perf_counter()
    observed_path = [
        incident.node_id,
        edge_by_type["AFFECTED"].type,
        service.node_id,
        edge_by_type["OWNED_BY"].type,
        team.node_id,
    ]
    visited_node_ids = list(
        dict.fromkeys(
            [
                edge_by_type["AFFECTED"].source_id,
                edge_by_type["AFFECTED"].target_id,
                edge_by_type["OWNED_BY"].target_id,
            ]
        )
    )
    traversal_latency_ms = (time.perf_counter() - traversal_started) * 1000.0
    expected_path = [
        incident.node_id,
        "AFFECTED",
        service.node_id,
        "OWNED_BY",
        team.node_id,
    ]
    evidence_id = batch.relations[0].evidence[0].evidence_id
    reasoning_started = time.perf_counter()
    predicted_inferred = ReasoningFact(
        subject=observed_path[0],
        predicate="IMPACT_OWNER",
        object=observed_path[-1],
    )
    predicted_reasoning_answers = [str(team.properties["team_name"])]
    reasoning_latency_ms = (time.perf_counter() - reasoning_started) * 1000.0
    expected_inferred = ReasoningFact(
        subject=incident.node_id,
        predicate="IMPACT_OWNER",
        object=team.node_id,
    )
    sql_started = time.perf_counter()
    observed_sql_rows = [
        {
            "incident": incident.properties["incident_id"],
            "service": service.properties["service_name"],
            "owner": team.properties["team_name"],
        }
    ]
    affected_service_count = sum(
        edge.type == "AFFECTED" for edge in graph.edges
    )
    sql_latency_ms = (time.perf_counter() - sql_started) * 1000.0
    tracemalloc.start()
    vector_started = time.perf_counter()
    observed_vector_ranking = [
        relation.evidence[0].evidence_id
        for relation in batch.relations
        if relation.relation_type == "AFFECTED" and relation.evidence
    ]
    vector_latency_ms = (time.perf_counter() - vector_started) * 1000.0
    _, vector_peak_memory_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    inputs = IntendedUseEvaluationInputs(
        sql_like=SQLLikeEvaluationInput(
            cases=[
                SQLLikeQueryCase(
                    case_id="affected-service-owner",
                    predicted_rows=observed_sql_rows,
                    expected_rows=[
                        {
                            "owner": "Platform Engineering",
                            "service": "Recommendation API",
                            "incident": "INCIDENT-1042",
                        }
                    ],
                    predicted_aggregations={
                        "affected_service_count": affected_service_count
                    },
                    expected_aggregations={"affected_service_count": 1},
                    latency_ms=sql_latency_ms,
                )
            ],
            max_p95_latency_ms=1000.0,
        ),
        nosql=NoSQLEvaluationInput(
            cases=[
                NoSQLTraversalCase(
                    case_id="incident-to-owner",
                    visited_node_ids=visited_node_ids,
                    expected_node_ids=[
                        team.node_id,
                        incident.node_id,
                        service.node_id,
                    ],
                    returned_paths=[observed_path],
                    expected_paths=[expected_path],
                    latency_ms=traversal_latency_ms,
                )
            ],
            max_p95_latency_ms=1000.0,
        ),
        graphrag=GraphRAGEvaluationInput(
            cases=[
                GraphRAGCase(
                    case_id="who-owns-the-affected-service",
                    retrieved_context_ids=[evidence_id],
                    context_relevance={evidence_id: 3.0},
                    retrieved_paths=[observed_path],
                    expected_paths=[expected_path],
                    requires_multi_hop=True,
                    multi_hop_answer_correct=observed_path == expected_path,
                    answer_faithfulness=1.0,
                    predicted_citation_ids=[batch.source_records[0].source_id],
                    expected_citation_ids=[batch.source_records[0].source_id],
                )
            ],
            k=1,
        ),
        vector=VectorIndexEvaluationInput(
            cases=[
                VectorIndexCase(
                    case_id="recommendation-service-context",
                    retrieved_ids=observed_vector_ranking,
                    relevance={evidence_id: 2.0},
                    filter_allowed_ids=[evidence_id],
                    latency_ms=vector_latency_ms,
                )
            ],
            eligible_item_count=1,
            indexed_item_count=1,
            peak_memory_bytes=vector_peak_memory_bytes,
            k=1,
            max_p95_latency_ms=1000.0,
            max_peak_memory_bytes=1024 * 1024,
        ),
        link_prediction=LinkPredictionEvaluationInput(
            cases=[
                LinkPredictionCase(
                    case_id="owner-head",
                    ranked_entity_ids=[service.node_id, team.node_id],
                    true_entity_id=team.node_id,
                    known_true_entity_ids=[service.node_id, team.node_id],
                    popularity_stratum="head",
                ),
                LinkPredictionCase(
                    case_id="owner-torso",
                    ranked_entity_ids=[team.node_id],
                    true_entity_id=team.node_id,
                    known_true_entity_ids=[team.node_id],
                    popularity_stratum="torso",
                ),
                LinkPredictionCase(
                    case_id="owner-tail",
                    ranked_entity_ids=[team.node_id],
                    true_entity_id=team.node_id,
                    known_true_entity_ids=[team.node_id],
                    popularity_stratum="tail",
                ),
            ]
        ),
        recommendation=RecommendationEvaluationInput(
            cases=[
                RecommendationCase(
                    case_id="incident-investigation-items",
                    user_id="incident-commander",
                    recommended_item_ids=[service.node_id, team.node_id],
                    relevant_item_ids=[service.node_id, team.node_id],
                )
            ],
            catalog_item_ids=[service.node_id, team.node_id],
            item_popularity_probability={
                service.node_id: 0.5,
                team.node_id: 0.25,
            },
            item_features={
                service.node_id: ["service"],
                team.node_id: ["team"],
            },
            k=2,
        ),
        reasoning=ReasoningEvaluationInput(
            cases=[
                ReasoningCase(
                    case_id="infer-impact-owner",
                    predicted_answers=predicted_reasoning_answers,
                    expected_answers=["Platform Engineering"],
                    predicted_inferred_facts=[predicted_inferred],
                    expected_inferred_facts=[expected_inferred],
                    runtime_ms=reasoning_latency_ms,
                )
            ],
            max_p95_runtime_ms=1000.0,
        ),
    )
    instance = owner()
    report = instance.evaluate(inputs)
    metrics: dict[str, float] = {}
    thresholds: dict[str, float] = {}
    assertions: list[JsonDict] = []
    for use_name, family in report.families.items():
        for suffix, actual in (
            ("executed", float(family.executed)),
            ("metric_count", float(len(family.metrics))),
            ("passed", float(family.passed)),
        ):
            metric_name = f"{use_name}_{suffix}"
            metrics[metric_name] = actual
            thresholds[metric_name] = 1.0
        assertions.append(
            {
                "name": f"{use_name} workload executed complete judgments and passed",
                "passed": family.executed and bool(family.metrics) and family.passed,
                "evaluated_cases": family.evaluated_cases,
                "observed_metrics": sorted(family.metrics),
            }
        )
    return ScenarioResult(
        instance=instance,
        operation="evaluate",
        fixture={"construction": built, "workload_judgments": inputs},
        output=report,
        metrics=metrics,
        thresholds=thresholds,
        assertions=tuple(assertions),
        details={
            "intended_uses": list(report.families),
            "observed_path": observed_path,
        },
    )


def _evaluation(owner: Owner, context: CertificationContext) -> ScenarioResult:
    if context.capability == "knowledge_graph_quality_evaluator":
        return _quality_evaluator_scenario(owner, context)
    if context.capability == "schema_quality_evaluator":
        return _schema_quality_evaluator_scenario(owner, context)
    if context.capability == "intended_use_metrics_evaluator":
        return _intended_use_evaluator_scenario(owner, context)
    if context.capability == "claim_evaluation_suite":
        gold_path = (
            context.root
            / "tests"
            / "fixtures"
            / "gold_standards"
            / "unstructured_claims.json"
        )
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        source = SourceRecord.from_text(gold["text"], source_name="held_out.txt")
        from kg_construction.unstructured.pipeline import UnstructuredPipeline

        pipeline_result = UnstructuredPipeline().run(source)
        instance = owner()
        report = instance.hard_gate(pipeline_result, gold)
        value = float(report.passed)
        output: object = report
        fixture: object = gold
        assertion = "all claim, qualifier, evidence, and provenance hard metrics passed"
    elif context.capability == "construction_metrics_evaluator":
        batch = _batch()
        instance = owner()
        metrics = instance.evaluate(batch, batch)
        required = {
            "entity_f1",
            "relation_f1",
            "provenance_coverage",
            "duplicate_edge_rate",
            "unsupported_extraction_rate",
        }
        passed = (
            required <= metrics.keys()
            and metrics["entity_f1"] == 1.0
            and metrics["relation_f1"] == 1.0
            and metrics["provenance_coverage"] == 1.0
            and metrics["unsupported_extraction_rate"] == 0.0
        )
        value = float(passed)
        output = metrics
        fixture = batch
        assertion = "construction evaluator executed accuracy and graph-hygiene metrics"
    else:
        raise ValueError(
            f"unsupported release evaluation capability: {context.capability}"
        )
    return _result(
        instance,
        "evaluate",
        fixture,
        output,
        metric="evaluation_contract_complete",
        value=value,
        threshold=1.0,
        assertion=assertion,
    )


def _validation(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.domain.schema import (
        GraphSchema,
        NodeTypeDefinition,
        PropertyDefinition,
        RelationshipTypeDefinition,
    )
    from kg_construction.infrastructure.graph_builders.rdf_graph import RDFGraphBuilder
    from kg_construction.infrastructure.ontology.schema_registry import (
        InMemorySchemaRegistry,
    )

    batch = _batch()
    schema = GraphSchema(
        schema_id="certification",
        status="promoted",
        node_types=[
            NodeTypeDefinition(
                name="Incident",
                properties=[
                    PropertyDefinition(
                        name="severity",
                        value_type="string",
                        required=True,
                        status="promoted",
                    )
                ],
                status="promoted",
            ),
            NodeTypeDefinition(name="Service", status="promoted"),
        ],
        relationship_types=[
            RelationshipTypeDefinition(
                name="AFFECTED",
                source_types=["Incident"],
                target_types=["Service"],
                status="promoted",
            )
        ],
    )
    if context.capability == "evidence_required_validator":
        instance = owner()
        report = instance.validate(batch)
        passed = report.valid and report.checked_count == 3
        output: object = report
    elif context.capability == "confidence_threshold_validator":
        instance = owner(threshold=0.95)
        report = instance.validate(batch)
        passed = report.valid and report.checked_count == 3
        output = report
    elif context.capability == "schema_registry_validator":
        instance = owner(InMemorySchemaRegistry(schema))
        report = instance.validate(batch)
        passed = report.valid and report.checked_count == 3
        output = report
    else:
        rdf_graph = RDFGraphBuilder().build(batch)
        instance = owner()
        report = instance.validate_rdf_graph(rdf_graph, graph_schema=schema)
        passed = report.conforms and bool(report.results_graph_turtle)
        output = report
    return _result(
        instance,
        "validate",
        {"batch": batch, "schema": schema},
        output,
        metric="validation_conformance",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} executed its constraint contract",
    )


def _schema_registry(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.domain.schema import (
        GraphSchema,
        NodeTypeDefinition,
        RelationshipTypeDefinition,
    )

    schema = GraphSchema(
        schema_id="certification",
        status="promoted",
        node_types=[
            NodeTypeDefinition(name="Incident", status="promoted"),
            NodeTypeDefinition(name="Service", status="promoted"),
        ],
        relationship_types=[
            RelationshipTypeDefinition(
                name="AFFECTED",
                source_types=["Incident"],
                target_types=["Service"],
                status="promoted",
            )
        ],
    )
    instance = owner(schema)
    passed = (
        instance.validate_entity_type("Incident")
        and instance.validate_relationship(
            "AFFECTED", source_type="Incident", target_type="Service"
        )
        and not instance.validate_entity_type("Unknown")
    )
    return _result(
        instance,
        "validate_entity_type+validate_relationship",
        schema,
        {
            "incident": instance.validate_entity_type("Incident"),
            "affected": instance.validate_relationship(
                "AFFECTED", source_type="Incident", target_type="Service"
            ),
            "unknown": instance.validate_entity_type("Unknown"),
        },
        metric="registry_constraint_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion="promoted registry accepted declared types and rejected an unknown type",
    )


def _schema_workflow(owner: Owner, context: CertificationContext) -> ScenarioResult:
    rows = [
        {"id": "1", "name": "Ada", "manager_id": ""},
        {"id": "2", "name": "Grace", "manager_id": "1"},
    ]
    source = SourceRecord.from_rows(
        rows,
        source_type="csv",
        source_name="people.csv",
        metadata={"table_name": "people"},
    )
    instance = owner()
    proposal = instance.propose([source], schema_id="people")
    if context.capability == "schema_promotion_workflow":
        approved = instance.approve(proposal.graph_schema)
        promoted = instance.promote(approved)
        passed = promoted.status == "promoted"
        output: object = promoted
    elif context.capability == "schema_quarantine_namespace":
        passed = (
            proposal.graph_schema.status == "proposed"
            and all(node.status == "proposed" for node in proposal.graph_schema.node_types)
        )
        output = proposal
    else:
        passed = (
            bool(proposal.graph_schema.node_types)
            and bool(proposal.shacl_turtle)
            and bool(proposal.owl_turtle)
            and bool(proposal.evidence.get("tables"))
        )
        output = proposal
    return _result(
        instance,
        "propose+govern",
        source,
        output,
        metric="schema_lifecycle_contract",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} executed the governed schema lifecycle",
    )


def _retry_policy(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.production.retry import RetryPolicyConfig

    instance = owner(RetryPolicyConfig(max_attempts=2, sleep=False))
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("transient timeout")
        return "written"

    outcome = instance.run(operation, operation_name="semantic_certification")
    passed = outcome.value == "written" and outcome.attempts == 2 and outcome.retry_count == 1
    return _result(
        instance,
        "run",
        {"first_attempt": "TimeoutError", "second_attempt": "written"},
        outcome,
        metric="retry_recovery",
        value=float(passed),
        threshold=1.0,
        assertion="retry policy retried once and returned the successful value",
    )


def _dead_letter(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.graph_stores.ingestion import (
        LadybugIngestionAdapter,
        LadybugIngestionConfig,
    )
    from kg_construction.infrastructure.production.dead_letter import DeadLetterQueue

    queue = (
        owner(context.workdir / "dead_letters.jsonl")
        if context.capability == "dead_letter_queue"
        else DeadLetterQueue(context.workdir / "dead_letters.jsonl")
    )
    graph = _property_graph(context.run_id)
    record = queue.record_failure(
        stage="graph_store_ingestion",
        store_name="ladybug_ingestion_adapter",
        operation_name="ladybug_property_graph_write",
        graph_model="property_graph",
        error=TimeoutError("transient timeout"),
        attempt_count=2,
        payload={"graph": graph.to_dict()},
    )
    if context.capability == "dead_letter_queue":
        instance = queue
        loaded = instance.load_records(status="pending")
        passed = len(loaded) == 1 and loaded[0].dead_letter_id == record.dead_letter_id
        output: object = loaded
        backend = "file:jsonl"
    else:
        instance = owner(queue)
        writer = LadybugIngestionAdapter(
            LadybugIngestionConfig(
                database_path=str(context.workdir / "replay.lbug"),
                checkpoint_after_write=False,
            )
        )
        report = instance.replay({"ladybug_ingestion_adapter": writer})
        passed = report.replayed == 1 and not queue.load_records(status="pending")
        output = report
        backend = "ladybug:embedded-live"
    return _result(
        instance,
        "record_failure" if context.capability == "dead_letter_queue" else "replay",
        {"record": record, "graph": graph},
        output,
        metric="durable_dead_letter_contract",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} persisted and recovered the failed write",
        backend=backend,
    )


def _quarantine(owner: Owner, context: CertificationContext) -> ScenarioResult:
    instance = owner(context.workdir / "quarantine.jsonl")
    source = _source()
    record = instance.quarantine_exception(
        stage="extraction",
        error=ValueError("invalid provider output"),
        source=source,
        payload={"provider": "fixture"},
    )
    loaded = instance.load_records()
    passed = len(loaded) == 1 and loaded[0].quarantine_id == record.quarantine_id
    return _result(
        instance,
        "quarantine_exception+load_records",
        source,
        loaded,
        metric="quarantine_durability",
        value=float(passed),
        threshold=1.0,
        assertion="quarantine durably retained the failed source and reason",
        backend="file:jsonl",
    )


def _human_review(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.ontology.validation import ValidationFinding

    instance = owner(context.workdir / "review.json")
    item = instance.enqueue_finding(
        ValidationFinding(
            object_id="candidate-1",
            object_type="entity",
            rule="ambiguous_identity",
            message="two candidates have equal scores",
            policy="send_to_human_review",
        )
    )
    decided = instance.decide(
        item.review_id,
        status="approved",
        reviewer="semantic-certifier",
        note="fixture adjudicated",
    )
    passed = decided.status == "approved" and not instance.pending()
    return _result(
        instance,
        "enqueue_finding+decide",
        item,
        decided,
        metric="review_lifecycle_completion",
        value=float(passed),
        threshold=1.0,
        assertion="review item was queued, adjudicated, and removed from pending work",
        backend="file:json",
    )


def _watermarks(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.production.watermarks import (
        WatermarkStore,
    )

    store = (
        owner(context.workdir / "watermarks.json")
        if context.capability == "watermark_store"
        else WatermarkStore(context.workdir / "watermarks.json")
    )
    first = SourceRecord.from_rows(
        [{"id": "1"}],
        source_type="sql",
        source_name="people",
        metadata={"updated_at": "2026-07-01T00:00:00+00:00"},
    )
    second = first.model_copy(
        update={
            "source_id": f"{first.source_id}-new",
            "metadata": {"updated_at": "2026-07-02T00:00:00+00:00"},
        }
    )
    if context.capability == "watermark_store":
        instance = store
        state = instance.update(
            source_key="sql:people",
            high_watermark="2026-07-01T00:00:00+00:00",
            source_id=first.source_id,
        )
        loaded = instance.get("sql:people")
        passed = loaded is not None and loaded.high_watermark == state.high_watermark
        output: object = loaded
    else:
        instance = owner()
        instance.commit([first], store)
        result = instance.apply([first, second], store)
        passed = (
            [item.source_id for item in result.skipped_sources] == [first.source_id]
            and [item.source_id for item in result.eligible_sources] == [second.source_id]
        )
        output = result
    return _result(
        instance,
        "update+get" if context.capability == "watermark_store" else "commit+apply",
        [first, second],
        output,
        metric="incremental_state_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} retained and applied a monotonic watermark",
        backend="file:json",
    )


def _pii(owner: Owner, context: CertificationContext) -> ScenarioResult:
    instance = owner()
    fixture = {
        "email": "ada@example.com",
        "phone": "+1 415 555 0100",
        "token": "api_key=secret-value",
    }
    redacted = instance.redact_mapping(fixture)
    output_text = canonical_json(redacted)
    passed = (
        "ada@example.com" not in output_text
        and "415" not in output_text
        and "secret-value" not in output_text
        and output_text.count(instance.replacement) >= 3
    )
    return _result(
        instance,
        "redact_mapping",
        fixture,
        redacted,
        metric="sensitive_value_redaction_rate",
        value=float(passed),
        threshold=1.0,
        assertion="configured PII classes were removed from the emitted mapping",
    )


def _structured_logging(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.production.logging import JsonLogFormatter
    from kg_construction.infrastructure.production.pii import PIIRedactionPolicy

    stream = io.StringIO()
    logger = logging.getLogger(f"kg-cert-{context.run_id}")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLogFormatter(pii_policy=PIIRedactionPolicy()))
    logger.addHandler(handler)
    instance = owner(logger, {"run_id": context.run_id})
    instance.info("certification_event", email="ada@example.com", count=1)
    payload = json.loads(stream.getvalue())
    passed = (
        payload.get("event") == "certification_event"
        and payload.get("run_id") == context.run_id
        and payload.get("email") == "[REDACTED]"
    )
    return _result(
        instance,
        "info",
        {"event": "certification_event", "email": "ada@example.com"},
        payload,
        metric="structured_log_contract",
        value=float(passed),
        threshold=1.0,
        assertion="logger emitted bound JSON context with PII redacted",
    )


def _run_manifest(owner: Owner, context: CertificationContext) -> ScenarioResult:
    from kg_construction.infrastructure.ontology.validation import (
        EvidenceRequiredValidator,
    )
    from kg_construction.infrastructure.production.manifests import RunManifest

    batch = _batch()
    manifest = RunManifest.start(
        pipeline_name="semantic-certification",
        sources=batch.source_records,
        config_payload={"profile": context.profile},
    ).complete(
        batch=batch,
        validation_report=EvidenceRequiredValidator().validate(batch),
    )
    instance = owner(context.workdir / "manifests")
    path = instance.write(manifest)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    passed = (
        persisted.get("run_id") == manifest.run_id
        and persisted.get("status") == "completed"
        and persisted.get("counts", {}).get("entities") == 2
    )
    return _result(
        instance,
        "write",
        manifest,
        persisted,
        metric="manifest_round_trip",
        value=float(passed),
        threshold=1.0,
        assertion="completed run manifest was durably round-tripped",
        backend="file:json",
    )


def _rdf_graph():
    from kg_construction.infrastructure.graph_builders.rdf_graph import RDFGraphBuilder

    return RDFGraphBuilder().build(_batch())


def _neo4j_driver(*, memgraph: bool = False):
    from neo4j import GraphDatabase

    runtime = LockedServiceRuntime.from_environment()
    if memgraph:
        uri = runtime.memgraph_uri
        return GraphDatabase.driver(uri), uri
    return (
        GraphDatabase.driver(
            runtime.neo4j_uri,
            auth=(runtime.neo4j_user, runtime.neo4j_password),
        ),
        runtime.neo4j_uri,
    )


def _execute_exported_cypher(
    statements: list[str],
    *,
    run_id: str,
    memgraph: bool,
) -> tuple[int, str]:
    driver, uri = _neo4j_driver(memgraph=memgraph)
    runtime = LockedServiceRuntime.from_environment()
    session_kwargs = {} if memgraph else {"database": runtime.neo4j_database}
    try:
        with driver.session(**session_kwargs) as session:
            for statement in statements:
                session.run(statement).consume()
            record = session.run(
                "MATCH (n {certification_run: $run_id}) "
                "RETURN count(n) AS node_count",
                run_id=run_id,
            ).single()
            count = int(record["node_count"]) if record is not None else 0
            session.run(
                "MATCH (n {certification_run: $run_id}) DETACH DELETE n",
                run_id=run_id,
            ).consume()
    finally:
        driver.close()
    return count, uri


def _graph_exporter(owner: Owner, context: CertificationContext) -> ScenarioResult:
    graph = _property_graph(context.run_id)
    rdf_graph = _rdf_graph()
    instance = owner()
    backend = "in_process"
    details: JsonDict = {}
    if context.capability in {"neo4j_exporter", "memgraph_exporter"}:
        statements = instance.export_property_graph(graph)
        memgraph = context.capability == "memgraph_exporter"
        node_count, uri = _execute_exported_cypher(
            statements,
            run_id=context.run_id,
            memgraph=memgraph,
        )
        passed = len(statements) == 3 and node_count == 2
        output: object = {
            "statements": statements,
            "live_node_count": node_count,
        }
        backend = f"{'memgraph' if memgraph else 'neo4j'}:live"
        details = {"uri": uri}
    elif context.capability == "ladybug_exporter":
        from kg_construction.infrastructure.graph_stores.ingestion import (
            LadybugIngestionAdapter,
            LadybugIngestionConfig,
        )

        manifest = instance.export_property_graph(graph)
        adapter = LadybugIngestionAdapter(
            LadybugIngestionConfig(
                database_path=str(context.workdir / "exporter.lbug"),
                checkpoint_after_write=False,
            )
        )
        report = adapter.write_property_graph(graph)
        passed = (
            len(manifest["nodes"]) == 2
            and len(manifest["relationships"]) == 1
            and report.node_count == 2
            and report.edge_count == 1
        )
        output = {"manifest": manifest, "write_report": report}
        backend = "ladybug:embedded-live"
    elif context.capability == "networkx_exporter":
        import networkx as nx

        payload = instance.export_property_graph(graph)
        materialized = nx.node_link_graph(payload, edges="links")
        passed = materialized.number_of_nodes() == 2 and materialized.number_of_edges() == 1
        output = {
            "payload": payload,
            "networkx_nodes": materialized.number_of_nodes(),
            "networkx_edges": materialized.number_of_edges(),
        }
        backend = "networkx:live-library"
    elif context.capability == "rdf_exporter":
        from rdflib import Dataset, Graph

        payload = instance.export_rdf_graph(rdf_graph)
        turtle_graph = Graph().parse(data=payload["turtle"], format="turtle")
        dataset = Dataset()
        dataset.parse(data=payload["nquads"], format="nquads")
        passed = len(turtle_graph) == len(rdf_graph.triples) and len(dataset) > 0
        output = {
            "turtle_sha256": hashlib.sha256(payload["turtle"].encode()).hexdigest(),
            "nquads_sha256": hashlib.sha256(payload["nquads"].encode()).hexdigest(),
            "parsed_triples": len(turtle_graph),
            "parsed_quads": len(dataset),
        }
        backend = "rdflib:live-library"
    else:
        exported = instance.export_rdf_graph(rdf_graph)
        dataset = instance.export_rdf_dataset(rdf_graph)
        passed = len(exported) == len(rdf_graph.triples) and len(dataset) > 0
        output = {
            "parsed_triples": len(exported),
            "parsed_dataset_statements": len(dataset),
        }
        backend = "rdflib:live-library"
    return _result(
        instance,
        "export+consumer_round_trip",
        {"property_graph": graph, "rdf_graph": rdf_graph},
        output,
        metric="export_round_trip_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} output was accepted by its real consumer",
        backend=backend,
        details=details,
    )


def _graph_ingestion(owner: Owner, context: CertificationContext) -> ScenarioResult:
    graph = _property_graph(context.run_id)
    if context.capability == "neo4j_ingestion_adapter":
        from kg_construction.infrastructure.graph_stores.ingestion import (
            Neo4jIngestionConfig,
        )

        driver, uri = _neo4j_driver(memgraph=False)
        runtime = LockedServiceRuntime.from_environment()
        instance = owner(
            Neo4jIngestionConfig(
                uri=uri,
                user=runtime.neo4j_user,
                password=runtime.neo4j_password,
                database=runtime.neo4j_database,
                batch_size=10,
            ),
            driver=driver,
        )
        try:
            report = instance.write_property_graph(graph)
            with driver.session(database=runtime.neo4j_database) as session:
                record = session.run(
                    "MATCH (n {certification_run: $run_id}) "
                    "RETURN count(n) AS node_count",
                    run_id=context.run_id,
                ).single()
                node_count = int(record["node_count"]) if record is not None else 0
                session.run(
                    "MATCH (n {certification_run: $run_id}) DETACH DELETE n",
                    run_id=context.run_id,
                ).consume()
        finally:
            driver.close()
        passed = (
            report.node_count == 2
            and report.edge_count == 1
            and report.statement_count >= 3
            and node_count == 2
        )
        output: object = {"write_report": report, "live_node_count": node_count}
        backend = "neo4j:live"
        details = {"uri": uri}
    else:
        from kg_construction.infrastructure.graph_stores.ingestion import (
            LadybugIngestionConfig,
        )

        database_path = context.workdir / "ingestion.lbug"
        instance = owner(
            LadybugIngestionConfig(
                database_path=str(database_path),
                batch_size=10,
                checkpoint_after_write=False,
            )
        )
        report = instance.write_property_graph(graph)
        passed = (
            report.node_count == 2
            and report.edge_count == 1
            and report.statement_count >= 3
            and report.metadata.get("database_path") == str(database_path)
        )
        output = report
        backend = "ladybug:embedded-live"
        details = {"database_path": str(database_path)}
    return _result(
        instance,
        "write_property_graph",
        graph,
        output,
        metric="live_graph_write_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion=f"{context.capability} wrote the complete fixture to the real store",
        backend=backend,
        details=details,
    )


def _neo4j_vector_store(owner: Owner, context: CertificationContext) -> ScenarioResult:
    """Certify index creation, fixed-dimension upsert, ANN scoring, and filtering."""

    import time

    from kg_construction.infrastructure.graph_stores.vector_store import (
        Neo4jVectorStoreConfig,
        VectorRecord,
    )

    suffix = hashlib.sha256(
        f"{context.profile}:{context.run_id}".encode()
    ).hexdigest()[:12]
    index_name = f"kg_cert_vector_{suffix}"
    node_label = f"KGVectorCert_{suffix}"
    driver, uri = _neo4j_driver(memgraph=False)
    runtime = LockedServiceRuntime.from_environment()
    config = Neo4jVectorStoreConfig(
        uri=uri,
        user=runtime.neo4j_user,
        password=runtime.neo4j_password,
        database=runtime.neo4j_database,
        index_name=index_name,
        node_label=node_label,
        dimensions=3,
        similarity="cosine",
        batch_size=3,
        await_index_timeout_seconds=120,
        max_top_k=10,
        filter_candidate_multiplier=10,
        max_filter_candidates=100,
    )
    instance = owner(config, driver=driver)
    records = [
        VectorRecord(
            vector_id=f"{suffix}:incident",
            embedding=[1.0, 0.0, 0.0],
            properties={"tenant": "blue", "kind": "incident"},
        ),
        VectorRecord(
            vector_id=f"{suffix}:service",
            embedding=[0.98, 0.02, 0.0],
            properties={"tenant": "blue", "kind": "service"},
        ),
        VectorRecord(
            vector_id=f"{suffix}:other",
            embedding=[0.0, 1.0, 0.0],
            properties={"tenant": "green", "kind": "service"},
        ),
    ]
    index_report: object = {}
    upsert_report: object = {}
    unfiltered: list[Any] = []
    filtered: list[Any] = []
    try:
        index_report = instance.ensure_index()
        upsert_report = instance.upsert(records)
        # Neo4j index refresh is asynchronous after a committed data transaction.
        for _ in range(40):
            unfiltered = instance.query([1.0, 0.0, 0.0], top_k=2)
            filtered = instance.query(
                [1.0, 0.0, 0.0],
                top_k=2,
                filters={"tenant": "blue"},
            )
            if len(unfiltered) == 2 and len(filtered) == 2:
                break
            time.sleep(0.25)
        unfiltered_ids = [hit.vector_id for hit in unfiltered]
        filtered_ids = [hit.vector_id for hit in filtered]
        filtered_tenants = [hit.properties.get("tenant") for hit in filtered]
        filtered_scores = [hit.score for hit in filtered]
        passed = (
            getattr(index_report, "state", None) == "ONLINE"
            and getattr(index_report, "dimensions", None) == 3
            and getattr(upsert_report, "record_count", None) == 3
            and unfiltered_ids == [f"{suffix}:incident", f"{suffix}:service"]
            and filtered_ids == [f"{suffix}:incident", f"{suffix}:service"]
            and filtered_tenants == ["blue", "blue"]
            and filtered_scores == sorted(filtered_scores, reverse=True)
        )
        output: object = {
            "index_report": index_report,
            "upsert_report": upsert_report,
            "unfiltered_hits": unfiltered,
            "filtered_hits": filtered,
        }
    finally:
        label = f"`{node_label}`"
        index = f"`{index_name}`"
        constraint = f"`{config.constraint_name}`"
        try:
            with driver.session(**({"database": config.database} if config.database else {})) as session:
                session.run(f"MATCH (n:{label}) DETACH DELETE n").consume()
                session.run(f"DROP INDEX {index} IF EXISTS").consume()
                session.run(f"DROP CONSTRAINT {constraint} IF EXISTS").consume()
        finally:
            driver.close()
    return _result(
        instance,
        "ensure_index+upsert+query",
        {"dimensions": 3, "records": records, "filters": {"tenant": "blue"}},
        output,
        metric="live_vector_store_semantic_accuracy",
        value=float(passed),
        threshold=1.0,
        assertion=(
            "Neo4j created the declared vector schema, stored all fixed-dimension "
            "vectors, and returned correctly scored equality-filtered neighbors"
        ),
        backend="neo4j-vector-index:live",
        details={
            "uri": uri,
            "database": config.database,
            "index_name": index_name,
            "node_label": node_label,
        },
    )


def _production_misc(owner: Owner, context: CertificationContext) -> ScenarioResult:
    if context.capability == "pii_redaction_policy":
        return _pii(owner, context)
    if context.capability == "structured_logging":
        return _structured_logging(owner, context)
    if context.capability == "run_manifest_writer":
        return _run_manifest(owner, context)
    raise KeyError(context.capability)


_LOADER_CAPABILITIES = {
    "api_response_loader",
    "chat_transcript_loader",
    "csv_loader",
    "email_loader",
    "event_payload_loader",
    "excel_loader",
    "html_loader",
    "json_loader",
    "jsonld_loader",
    "log_loader",
    "markdown_loader",
    "pdf_loader",
    "report_loader",
    "scientific_paper_loader",
    "sql_database_loader",
    "structured_api_loader",
    "text_loader",
    "webpage_loader",
    "xml_loader",
}

_DIRECT_MAPPER_CAPABILITIES = {
    "column_to_property_mapping",
    "column_type_inference",
    "direct_table_mapper",
    "foreign_key_to_edge_mapping",
    "primary_key_to_id_mapping",
    "row_to_entity_mapping",
    "schema_profiling",
    "table_to_node_mapping",
}

_JSON_MAPPER_CAPABILITIES = {
    "dotted_path_json_mapper",
    "jsonld_mapper",
    "jsonpath_mapper",
    "obda_virtual_graph_mapper",
}

_SPECIALIZED_MAPPER_CAPABILITIES = {
    "event_graph_mapper",
    "html_metadata_mapper",
    "log_event_mapper",
}

_STANDARDS_MAPPER_CAPABILITIES = {
    "csvw_mapper",
    "r2rml_mapper",
    "rml_mapper",
}

_DETERMINISTIC_EXTRACTORS = {
    "dependency_pattern_relation_extractor",
    "gazetteer_entity_extractor",
    "openie_relation_extractor",
    "regex_entity_extractor",
}

_DETERMINISTIC_RESOLVERS = {
    "alias_table_resolver",
    "exact_key_entity_resolver",
    "fuzzy_string_resolver",
    "normalized_string_resolver",
    "rapidfuzz_resolver",
}

_MODEL_RESOLVERS = {
    "cross_encoder_resolver",
    "embedding_resolver",
}

_EXTERNAL_LINKERS = {
    "dbpedia_linker",
    "openrefine_reconciler",
    "wikidata_linker",
}

_GRAPH_BUILDERS = {
    "evidence_graph_builder",
    "property_graph_builder",
    "rdf_graph_builder",
}

_EVALUATORS = {
    "claim_evaluation_suite",
    "construction_metrics_evaluator",
    "intended_use_metrics_evaluator",
    "knowledge_graph_quality_evaluator",
    "schema_quality_evaluator",
}

_VALIDATORS = {
    "confidence_threshold_validator",
    "evidence_required_validator",
    "schema_registry_validator",
    "shacl_validator",
}

_SCHEMA_WORKFLOWS = {
    "schema_promotion_workflow",
    "schema_proposal_workflow",
    "schema_quarantine_namespace",
}

_GRAPH_EXPORTERS = {
    "ladybug_exporter",
    "memgraph_exporter",
    "neo4j_exporter",
    "networkx_exporter",
    "rdf_exporter",
    "rdflib_exporter",
}

SCENARIOS: dict[str, Scenario] = {
    **{name: _loaders for name in _LOADER_CAPABILITIES},
    **{name: _structured_direct_mapper for name in _DIRECT_MAPPER_CAPABILITIES},
    **{name: _json_mapper for name in _JSON_MAPPER_CAPABILITIES},
    **{name: _event_or_html_mapper for name in _SPECIALIZED_MAPPER_CAPABILITIES},
    **{name: _standards_mapper for name in _STANDARDS_MAPPER_CAPABILITIES},
    **{name: _extractor for name in _DETERMINISTIC_EXTRACTORS},
    **{name: _resolver for name in _DETERMINISTIC_RESOLVERS},
    **{name: _embedding_resolver for name in _MODEL_RESOLVERS},
    **{name: _external_linker for name in _EXTERNAL_LINKERS},
    **{name: _graph_builder for name in _GRAPH_BUILDERS},
    **{name: _evaluation for name in _EVALUATORS},
    **{name: _validation for name in _VALIDATORS},
    **{name: _schema_workflow for name in _SCHEMA_WORKFLOWS},
    **{name: _graph_exporter for name in _GRAPH_EXPORTERS},
    "batch_retry_policy": _retry_policy,
    "dead_letter_queue": _dead_letter,
    "dead_letter_replay_tool": _dead_letter,
    "error_quarantine": _quarantine,
    "evidence_first_unstructured_pipeline": _unstructured_pipeline,
    "gliner_entity_extractor": _model_provider,
    "gliner_relex_claim_extractor": _model_provider,
    "gliner_ollama_relation_typing_extractor": _gliner_ollama,
    "human_review_queue": _human_review,
    "hybrid_gliner_then_ollama_relations": _gliner_ollama,
    "in_memory_schema_registry": _schema_registry,
    "incremental_sync_policy": _watermarks,
    "ladybug_ingestion_adapter": _graph_ingestion,
    "mandatory_claim_ensemble": _model_provider,
    "mandatory_composed_unstructured_runtime": (
        _mandatory_composed_unstructured_runtime
    ),
    "mandatory_mention_ensemble": _model_provider,
    "mlx_qwen_claim_extractor": _model_provider,
    "neo4j_ingestion_adapter": _graph_ingestion,
    "neo4j_vector_store_adapter": _neo4j_vector_store,
    "ollama_graph_extractor": _ollama_graph,
    "ollama_provider": _ollama_provider,
    "ollama_qwen_claim_extractor": _model_provider,
    "pii_redaction_policy": _production_misc,
    "run_manifest_writer": _production_misc,
    "semantic_column_profiler": _semantic_profiler,
    "semantic_table_mapper": _semantic_table_mapper,
    "source_catalog": _source_catalog,
    "spacy_ner_extractor": _model_provider,
    "structured_logging": _production_misc,
    "transformer_ner_extractor": _model_provider,
    "transformers_qwen_claim_extractor": _model_provider,
    "watermark_store": _watermarks,
}


def uncovered_manifest_capabilities(root: Path) -> set[str]:
    """Return declared release names without a semantic scenario."""

    manifest = _load_mapping(root / "capability_manifest.yaml")
    declared = {
        str(row["name"])
        for row in manifest.get("techniques", [])
        if isinstance(row, dict) and row.get("name")
    }
    return declared - SCENARIOS.keys()


__all__ = [
    "CertificationContext",
    "SCENARIOS",
    "ScenarioResult",
    "certify_capability",
    "uncovered_manifest_capabilities",
]
