from __future__ import annotations

from types import SimpleNamespace

import pytest

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
from kg_construction.evaluation import (
    QUALITY_THRESHOLDS_VERSION,
    ConstructionReplayEvidence,
    KnowledgeGraphQualityEvaluator,
    QualityEvaluationInputs,
    RetryObservation,
    RuntimeReport,
    SchemaQualityEvaluator,
    property_graph_sha256,
)
from kg_construction.infrastructure.ontology.validation import ValidationReport
from kg_construction.infrastructure.production import RunManifest


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
    initial_run = _terminal_manifest("quality-run-initial")
    replay_run = _terminal_manifest("quality-run-replay")
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
        replay_evidence=ConstructionReplayEvidence(
            initial_run=initial_run,
            replay_run=replay_run,
            initial_graph_sha256=property_graph_sha256(graph),
            replay_graph_sha256=property_graph_sha256(graph),
        ),
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
            retry_observations=(
                RetryObservation(
                    operation_id="quality-construction",
                    attempt_count=1,
                    retry_count=0,
                ),
            ),
        ),
        operational_run_manifests=(initial_run,),
    )


def _terminal_manifest(
    run_id: str,
    *,
    status: str = "completed",
    config_hash: str = "config-sha256",
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        pipeline_name="quality-test-workflow",
        status=status,
        completed_at="2026-07-25T00:00:00+00:00",
        config_hash=config_hash,
        input_hash="input-sha256",
        git_commit="0123456789abcdef0123456789abcdef01234567",
        source_ids=["source-quality-test"],
        source_names=["memory://quality-test"],
        errors=(
            []
            if status == "completed"
            else [{"stage": "workflow", "error_type": "RuntimeError"}]
        ),
    )


def test_complete_quality_contract_executes_every_family() -> None:
    report = KnowledgeGraphQualityEvaluator().evaluate(_batch(), inputs=_all_inputs())

    assert report.thresholds_version == QUALITY_THRESHOLDS_VERSION
    assert report.missing_families == []
    assert report.passed
    assert all(family.executed for family in report.families.values())
    assert all(family.passed for family in report.families.values())
    structured_gate_metrics = {
        gate.metric for gate in report.families["structured"].gates
    }
    assert {
        "schema_gate_node_conciseness",
        "schema_gate_connectivity",
        "schema_gate_consistency",
        "schema_gate_categorical_conciseness",
        "schema_gate_categorical_occurrence_reconciliation",
        "schema_gate_property_completeness",
        "schema_gate_relationship_completeness",
        "schema_gate_shacl_execution",
        "schema_gate_shacl_conformance",
    } <= structured_gate_metrics


def test_idempotence_requires_manifest_bound_workflow_replay() -> None:
    inputs = _all_inputs().model_copy(
        update={
            "replay_evidence": None,
        }
    )

    report = KnowledgeGraphQualityEvaluator().evaluate(_batch(), inputs=inputs)

    assert not report.families["idempotence"].executed
    assert not report.families["idempotence"].passed
    assert "manifests" in report.families["idempotence"].reason


def test_idempotence_rejects_graphs_that_do_not_match_replay_evidence() -> None:
    inputs = _all_inputs()
    inputs = inputs.model_copy(
        update={
            "replay_evidence": inputs.replay_evidence.model_copy(
                update={"replay_graph_sha256": "0" * 64}
            )
            if inputs.replay_evidence is not None
            else None,
        }
    )

    report = KnowledgeGraphQualityEvaluator().evaluate(_batch(), inputs=inputs)

    assert not report.families["idempotence"].executed
    assert "hashes do not match" in report.families["idempotence"].reason


def test_idempotence_ignores_only_recursive_construction_audit_timestamps() -> None:
    initial = _graph()
    replay = _graph()
    initial.nodes[0].properties.update(
        {
            "created_at": "2026-07-25T00:00:00+00:00",
            "candidate_provenance": [
                {
                    "created_at": "2026-07-25T00:00:00+00:00",
                    "extractor": "test",
                }
            ],
        }
    )
    replay.nodes[0].properties.update(
        {
            "created_at": "2026-07-25T00:00:01+00:00",
            "candidate_provenance": [
                {
                    "created_at": "2026-07-25T00:00:01+00:00",
                    "extractor": "test",
                }
            ],
        }
    )
    initial_run = _terminal_manifest("timestamp-initial")
    replay_run = _terminal_manifest("timestamp-replay")
    inputs = _all_inputs().model_copy(
        update={
            "property_graph": initial,
            "replay_graph": replay,
            "replay_evidence": ConstructionReplayEvidence(
                initial_run=initial_run,
                replay_run=replay_run,
                initial_graph_sha256=property_graph_sha256(initial),
                replay_graph_sha256=property_graph_sha256(replay),
            ),
        }
    )

    report = KnowledgeGraphQualityEvaluator().evaluate(_batch(), inputs=inputs)

    idempotence = report.families["idempotence"]
    assert idempotence.passed
    assert idempotence.metrics["graph_idempotence_score"] == 1.0
    assert idempotence.metrics["idempotence_added_object_count"] == 0.0
    assert idempotence.metrics["idempotence_missing_object_count"] == 0.0


def test_idempotence_keeps_semantic_temporal_properties_in_comparison() -> None:
    initial = _graph()
    replay = _graph()
    initial.nodes[0].properties["occurred_at"] = "2026-07-25T00:00:00+00:00"
    replay.nodes[0].properties["occurred_at"] = "2026-07-26T00:00:00+00:00"
    initial_run = _terminal_manifest("semantic-time-initial")
    replay_run = _terminal_manifest("semantic-time-replay")
    inputs = _all_inputs().model_copy(
        update={
            "property_graph": initial,
            "replay_graph": replay,
            "replay_evidence": ConstructionReplayEvidence(
                initial_run=initial_run,
                replay_run=replay_run,
                initial_graph_sha256=property_graph_sha256(initial),
                replay_graph_sha256=property_graph_sha256(replay),
            ),
        }
    )

    report = KnowledgeGraphQualityEvaluator().evaluate(_batch(), inputs=inputs)

    idempotence = report.families["idempotence"]
    assert not idempotence.passed
    assert idempotence.metrics["graph_idempotence_score"] < 1.0
    assert idempotence.metrics["idempotence_added_object_count"] == 1.0
    assert idempotence.metrics["idempotence_missing_object_count"] == 1.0


def test_replay_evidence_requires_same_reproducible_configuration() -> None:
    graph = _graph()

    with pytest.raises(ValueError, match="reproducibility fields"):
        ConstructionReplayEvidence(
            initial_run=_terminal_manifest("initial"),
            replay_run=_terminal_manifest("replay", config_hash="different-config"),
            initial_graph_sha256=property_graph_sha256(graph),
            replay_graph_sha256=property_graph_sha256(graph),
        )


def test_operational_family_requires_retry_and_terminal_run_observations() -> None:
    inputs = _all_inputs().model_copy(
        update={
            "runtime_report": RuntimeReport(
                processed_items=1,
                latency_seconds=0.01,
                throughput=100.0,
            ),
            "operational_run_manifests": (),
        }
    )

    report = KnowledgeGraphQualityEvaluator().evaluate(_batch(), inputs=inputs)

    assert not report.families["operational"].executed
    assert not report.families["operational"].metrics
    assert "retry observations" in report.families["operational"].reason


def test_operational_metrics_are_derived_from_observed_attempts_and_manifests() -> None:
    inputs = _all_inputs().model_copy(
        update={
            "runtime_report": RuntimeReport(
                processed_items=2,
                latency_seconds=0.02,
                throughput=100.0,
                retry_observations=(
                    RetryObservation(
                        operation_id="first-write",
                        attempt_count=2,
                        retry_count=1,
                    ),
                    RetryObservation(
                        operation_id="second-write",
                        attempt_count=1,
                        retry_count=0,
                    ),
                ),
            ),
            "operational_run_manifests": (
                _terminal_manifest("successful-run"),
                _terminal_manifest("failed-run", status="failed"),
            ),
        }
    )

    report = KnowledgeGraphQualityEvaluator().evaluate(_batch(), inputs=inputs)
    operational = report.families["operational"]

    assert operational.executed
    assert operational.metrics["retry_rate"] == 0.5
    assert operational.metrics["retry_count"] == 1.0
    assert operational.metrics["retry_attempt_count"] == 3.0
    assert operational.metrics["pipeline_success_rate"] == 0.5
    assert operational.metrics["pipeline_run_support"] == 2.0
    assert not operational.passed


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
