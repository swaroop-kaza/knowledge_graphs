from __future__ import annotations

from types import SimpleNamespace

from kg_construction.domain.evidence import cell_evidence
from kg_construction.domain.graph_objects import (
    PropertyEdge,
    PropertyGraph,
    PropertyNode,
    RDFGraph,
)
from kg_construction.domain.modeling import (
    Cardinality,
    OntologyClass,
    OntologyPredicate,
    OntologySchema,
)
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    EvidenceRef,
    RelationCandidate,
    SourceRecord,
)
from kg_construction.infrastructure.evaluation import (
    QUALITY_THRESHOLDS_VERSION,
    KnowledgeGraphQualityEvaluator,
    QualityEvaluationInputs,
    RuntimeReport,
    SchemaQualityEvaluator,
)
from kg_construction.infrastructure.ontology.validation import ValidationReport


def _ontology() -> OntologySchema:
    return OntologySchema(
        classes=[
            OntologyClass(name="Employee", key_properties=["employee_id"]),
            OntologyClass(
                name="Department", role="reference", key_properties=["value"]
            ),
        ],
        predicates=[
            OntologyPredicate(
                name="employee_id",
                domain="Employee",
                range="string",
                kind="datatype",
                cardinality=Cardinality.exactly_one(),
            ),
            OntologyPredicate(
                name="has_department",
                domain="Employee",
                range="Department",
                kind="object",
                cardinality=Cardinality.exactly_one(),
            ),
        ],
    )


def _graph() -> PropertyGraph:
    return PropertyGraph(
        nodes=[
            PropertyNode(
                node_id="employee-1",
                labels=["Entity", "Employee"],
                properties={"employee_id": "1", "name": "Alex"},
            ),
            PropertyNode(
                node_id="department-engineering",
                labels=["Reference", "Department"],
                properties={"value": "engineering", "name": "Engineering"},
            ),
        ],
        edges=[
            PropertyEdge(
                edge_id="edge-1",
                source_id="employee-1",
                target_id="department-engineering",
                type="has_department",
            )
        ],
    )


def _batch() -> ConstructionBatch:
    rows = [{"employee_id": "1", "department": "Engineering"}]
    source = SourceRecord.from_text(
        "Alex Engineering",
        source_type="csv",
        source_name="employees.csv",
    ).model_copy(update={"records": rows})
    employee_evidence = EvidenceRef.build(
        source,
        location={"kind": "span", "start": 0, "end": 4},
        text="Alex",
        metadata={"document_version": source.source_id},
    )
    department_evidence = EvidenceRef.build(
        source,
        location={"kind": "span", "start": 5, "end": 16},
        text="Engineering",
        metadata={"document_version": source.source_id},
    )
    department_cell_evidence = cell_evidence(
        source,
        row_index=0,
        column="department",
        value="Engineering",
    )
    entity = EntityCandidate(
        candidate_id="employee-1",
        entity_type="Employee",
        name="Alex",
        evidence=[employee_evidence],
        extractor_name="test",
        technique="gold",
    )
    department = EntityCandidate(
        candidate_id="department-engineering",
        entity_type="Department",
        name="Engineering",
        evidence=[department_evidence],
        extractor_name="test",
        technique="gold",
    )
    relation = RelationCandidate(
        candidate_id="edge-1",
        source=entity.candidate_id,
        target=department.candidate_id,
        relation_type="has_department",
        evidence=[department_cell_evidence],
        extractor_name="test",
        technique="gold",
    )
    return ConstructionBatch(
        source_records=[source],
        entities=[entity, department],
        relations=[relation],
    )


def _all_inputs() -> QualityEvaluationInputs:
    graph = _graph()
    policy = SimpleNamespace(
        record_class="Employee",
        attributes=[
            SimpleNamespace(column="department", materialization="node"),
        ],
    )
    return QualityEvaluationInputs(
        property_graph=graph,
        rdf_graph=RDFGraph(),
        validation_report=ValidationReport(),
        policy=policy,
        ontology=_ontology(),
        source_records=[{"employee_id": "1", "department": "Engineering"}],
        shacl_executed=True,
        shacl_violation_count=0,
        expected_batch=_batch(),
        replay_graph=graph.model_copy(deep=True),
        predicted_coreference={"m1": "alex", "m2": "alex"},
        gold_coreference={"m1": "alex", "m2": "alex"},
        predicted_links={"m1": "employee-1", "m2": "employee-1"},
        gold_links={"m1": "employee-1", "m2": "employee-1"},
        predicted_answers={"q1": ["Engineering"]},
        gold_answers={"q1": ["Engineering"]},
        vector_rankings={"q1": ["department-engineering"]},
        vector_relevant={"q1": ["department-engineering"]},
        runtime_report=RuntimeReport(
            processed_items=1,
            latency_seconds=0.01,
            throughput=100.0,
            retry_rate=0.0,
        ),
        successful_runs=1,
        total_runs=1,
    )


def test_complete_quality_contract_executes_every_family() -> None:
    report = KnowledgeGraphQualityEvaluator().evaluate(_batch(), inputs=_all_inputs())

    assert report.thresholds_version == QUALITY_THRESHOLDS_VERSION
    assert report.missing_families == []
    assert report.passed
    assert all(family.executed for family in report.families.values())
    assert all(family.passed for family in report.families.values())


def test_missing_inputs_are_explicit_failed_families() -> None:
    report = KnowledgeGraphQualityEvaluator().evaluate(_batch())

    assert not report.passed
    assert set(report.missing_families) == set(report.families)
    assert not report.families["vector"].executed
    assert not report.families["vector"].gates[0].passed
    assert "requires" in report.families["vector"].reason


def test_governed_unstructured_shacl_evidence_is_bound_to_the_report() -> None:
    report = KnowledgeGraphQualityEvaluator().evaluate(
        _batch(),
        inputs=QualityEvaluationInputs(
            evaluation_profile="unstructured",
            property_graph=_graph(),
            governed_schema_id="governed-unstructured-core",
            governed_schema_version="v1",
            governed_schema_sha256="a" * 64,
            shacl_executed=True,
            shacl_checked_count=3,
            shacl_violation_count=0,
        ),
    )

    assert report.governed_schema_id == "governed-unstructured-core"
    assert report.governed_schema_version == "v1"
    assert report.governed_schema_sha256 == "a" * 64
    assert report.shacl_checked_count == 3
    assert report.shacl_violation_count == 0
    assert report.families["structured"].executed


def test_declared_missing_properties_and_relationships_lower_completeness() -> None:
    graph = PropertyGraph(
        nodes=[
            PropertyNode(
                node_id="employee-1",
                labels=["Entity", "Employee"],
                properties={"name": "Alex"},
            )
        ]
    )

    report = SchemaQualityEvaluator().evaluate(graph, ontology=_ontology())

    assert report.property_completeness == 0.0
    assert report.relationship_completeness == 0.0
    assert not report.shacl_executed
    assert not report.passed
    assert not next(g for g in report.gates if g.gate == "shacl_execution").passed


def test_duplicate_nodes_use_declared_keys_not_display_names() -> None:
    graph = PropertyGraph(
        nodes=[
            PropertyNode(
                node_id="one",
                labels=["Entity", "Employee"],
                properties={"employee_id": "1", "name": "Alex"},
            ),
            PropertyNode(
                node_id="two",
                labels=["Entity", "Employee"],
                properties={"employee_id": "2", "name": "Alex"},
            ),
        ]
    )
    report = SchemaQualityEvaluator().evaluate(
        graph,
        ontology=_ontology(),
        shacl_executed=True,
        shacl_violation_count=0,
    )
    assert report.duplicate_node_rate == 0.0

    duplicated = graph.model_copy(deep=True)
    duplicated.nodes[1].properties["employee_id"] = "1"
    duplicate_report = SchemaQualityEvaluator().evaluate(
        duplicated,
        ontology=_ontology(),
        shacl_executed=True,
        shacl_violation_count=0,
    )
    assert duplicate_report.duplicate_node_rate == 0.5


def test_datatype_domain_and_cardinality_are_evaluated() -> None:
    ontology = OntologySchema(
        classes=[OntologyClass(name="Employee", key_properties=["employee_id"])],
        predicates=[
            OntologyPredicate(
                name="employee_id",
                domain="Employee",
                range="integer",
                kind="datatype",
                cardinality=Cardinality.exactly_one(),
            )
        ],
    )
    graph = PropertyGraph(
        nodes=[
            PropertyNode(
                node_id="employee-1",
                labels=["Entity", "Employee"],
                properties={"employee_id": "not-an-integer"},
            )
        ]
    )

    report = SchemaQualityEvaluator().evaluate(
        graph,
        ontology=ontology,
        shacl_executed=True,
        shacl_violation_count=0,
    )

    assert report.domain_range_violation_rate == 1.0
    assert report.consistency_violation_count == 1
    assert not report.passed
