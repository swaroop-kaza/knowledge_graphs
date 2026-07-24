"""One mandatory, versioned quality contract for constructed knowledge graphs.

The evaluator deliberately reports every metric family.  A family that cannot be
executed is not treated as a successful ``N/A``: it is returned with
``executed=False`` and its mandatory gates fail.  This makes the report suitable
for the public construction facade and for CI/release decisions.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_construction.domain.graph_objects import PropertyGraph, RDFGraph
from kg_construction.domain.modeling import OntologySchema
from kg_construction.domain.models import (
    ConstructionBatch,
    EntityCandidate,
    GraphFact,
    RelationCandidate,
)
from kg_construction.infrastructure.evaluation.accuracy_metrics import AccuracyMetrics
from kg_construction.infrastructure.evaluation.coverage_metrics import CoverageMetrics
from kg_construction.infrastructure.evaluation.operational_metrics import (
    OperationalMetrics,
    RuntimeReport,
)
from kg_construction.infrastructure.evaluation.schema_quality import (
    SchemaQualityEvaluator,
    SchemaQualityGateSettings,
)
from kg_construction.infrastructure.evaluation.validation_metrics import ValidationMetrics
from kg_construction.infrastructure.ontology.validation import ValidationReport

QualityFamily = Literal[
    "formation",
    "structured",
    "category",
    "idempotence",
    "ner",
    "relation",
    "coreference",
    "linking",
    "query",
    "vector",
    "operational",
]
QualityEvaluationProfile = Literal["structured", "unstructured", "mixed"]
ThresholdOperator = Literal[">=", "<="]

QUALITY_THRESHOLDS_VERSION = "kg-quality-v3"


class QualityThreshold(BaseModel):
    """A mandatory threshold for one metric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    family: QualityFamily
    metric: str
    operator: ThresholdOperator
    threshold: float


def _default_thresholds() -> list[QualityThreshold]:
    return [
        QualityThreshold(
            family="formation",
            metric="evidence_coverage",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="formation",
            metric="provenance_coverage",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="formation",
            metric="exact_evidence_coverage",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="formation",
            metric="dangling_edge_rate",
            operator="<=",
            threshold=0.0,
        ),
        QualityThreshold(
            family="formation",
            metric="duplicate_edge_rate",
            operator="<=",
            threshold=0.0,
        ),
        QualityThreshold(
            family="formation",
            metric="unsupported_extraction_rate",
            operator="<=",
            threshold=0.0,
        ),
        QualityThreshold(
            family="formation",
            metric="source_coverage",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="formation",
            metric="validation_passed",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="formation",
            metric="constructed_object_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="structured",
            metric="property_completeness",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="structured",
            metric="relationship_completeness",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="structured",
            metric="consistency_violation_count",
            operator="<=",
            threshold=0.0,
        ),
        QualityThreshold(
            family="structured",
            metric="shacl_executed",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="structured",
            metric="shacl_violation_count",
            operator="<=",
            threshold=0.0,
        ),
        QualityThreshold(
            family="structured",
            metric="graph_object_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="category",
            metric="categorical_duplication_rate",
            operator="<=",
            threshold=0.0,
        ),
        QualityThreshold(
            family="idempotence",
            metric="graph_idempotence_score",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(family="ner", metric="entity_f1", operator=">=", threshold=0.80),
        QualityThreshold(family="ner", metric="span_f1", operator=">=", threshold=0.80),
        QualityThreshold(
            family="ner", metric="entity_gold_support", operator=">=", threshold=1.0
        ),
        QualityThreshold(
            family="ner",
            metric="entity_prediction_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="ner",
            metric="entity_gold_coverage",
            operator=">=",
            threshold=0.80,
        ),
        QualityThreshold(
            family="ner", metric="span_gold_support", operator=">=", threshold=1.0
        ),
        QualityThreshold(
            family="ner",
            metric="span_prediction_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="ner",
            metric="span_gold_coverage",
            operator=">=",
            threshold=0.80,
        ),
        QualityThreshold(
            family="relation", metric="relation_f1", operator=">=", threshold=0.80
        ),
        QualityThreshold(
            family="relation",
            metric="relation_gold_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="relation",
            metric="relation_prediction_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="relation",
            metric="relation_gold_coverage",
            operator=">=",
            threshold=0.80,
        ),
        QualityThreshold(
            family="coreference",
            metric="coreference_b3_f1",
            operator=">=",
            threshold=0.80,
        ),
        QualityThreshold(
            family="coreference",
            metric="coreference_gold_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="coreference",
            metric="coreference_prediction_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="linking",
            metric="entity_linking_accuracy",
            operator=">=",
            threshold=0.80,
        ),
        QualityThreshold(
            family="linking",
            metric="entity_linking_gold_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="linking",
            metric="entity_linking_prediction_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="query",
            metric="competency_question_accuracy",
            operator=">=",
            threshold=0.80,
        ),
        QualityThreshold(
            family="query",
            metric="query_gold_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="query",
            metric="query_prediction_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="vector", metric="vector_recall_at_k", operator=">=", threshold=0.70
        ),
        QualityThreshold(
            family="vector",
            metric="vector_gold_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="vector",
            metric="vector_prediction_support",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="operational",
            metric="pipeline_success_rate",
            operator=">=",
            threshold=1.0,
        ),
        QualityThreshold(
            family="operational", metric="retry_rate", operator="<=", threshold=0.05
        ),
    ]


class VersionedQualityThresholds(BaseModel):
    """Immutable threshold set; changing a threshold requires a new version."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = QUALITY_THRESHOLDS_VERSION
    thresholds: list[QualityThreshold] = Field(default_factory=_default_thresholds)

    @model_validator(mode="after")
    def _unique_metrics(self) -> VersionedQualityThresholds:
        keys = [(item.family, item.metric) for item in self.thresholds]
        if len(keys) != len(set(keys)):
            raise ValueError("quality thresholds must be unique by family and metric")
        declared = {item.family for item in self.thresholds}
        missing = set(_FAMILY_ORDER) - declared
        if missing:
            raise ValueError(f"quality thresholds are missing mandatory families: {sorted(missing)}")
        return self


class QualityGateResult(BaseModel):
    """Result of applying one versioned threshold."""

    model_config = ConfigDict(extra="forbid")

    family: QualityFamily
    metric: str
    operator: ThresholdOperator
    threshold: float
    actual: float | None = None
    executed: bool
    passed: bool
    reason: str


class MetricFamilyReport(BaseModel):
    """Metrics, execution state, and gates for one evaluation family."""

    model_config = ConfigDict(extra="forbid")

    family: QualityFamily
    applicable: bool = True
    executed: bool
    metrics: dict[str, float] = Field(default_factory=dict)
    gates: list[QualityGateResult] = Field(default_factory=list)
    passed: bool = False
    reason: str


class KnowledgeGraphQualityReport(BaseModel):
    """Complete release report across all mandatory metric families."""

    model_config = ConfigDict(extra="forbid")

    thresholds_version: str
    families: dict[QualityFamily, MetricFamilyReport]
    executed_families: list[QualityFamily]
    missing_families: list[QualityFamily]
    applicable_families: list[QualityFamily]
    not_applicable_families: list[QualityFamily]
    evaluation_profile: QualityEvaluationProfile
    governed_schema_id: str | None = None
    governed_schema_version: str | None = None
    governed_schema_sha256: str | None = None
    shacl_checked_count: int | None = Field(default=None, ge=0)
    shacl_violation_count: int | None = Field(default=None, ge=0)
    passed: bool

    def metric_payload(self) -> dict[str, float]:
        """Flatten executed metrics for report writers and benchmark adapters."""

        return {
            metric: value
            for report in self.families.values()
            for metric, value in report.metrics.items()
        }


class QualityEvaluationInputs(BaseModel):
    """Gold/use-case inputs needed to execute every quality family.

    ``subject`` passed to :meth:`KnowledgeGraphQualityEvaluator.evaluate` supplies
    a batch and, for a full construction result, its property graph.  These inputs
    supply the modelling contract, replay output, gold annotations, retrieval
    judgments, and runtime observations.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    evaluation_profile: QualityEvaluationProfile = "mixed"
    property_graph: PropertyGraph | None = None
    rdf_graph: RDFGraph | None = None
    validation_report: ValidationReport | None = None
    policy: object | None = None
    ontology: OntologySchema | None = None
    resolved_plan: object | None = None
    source_records: list[dict[str, object]] | None = None
    shacl_executed: bool = False
    shacl_violation_count: int | None = None
    shacl_checked_count: int | None = Field(default=None, ge=0)
    governed_schema_id: str | None = None
    governed_schema_version: str | None = None
    governed_schema_sha256: str | None = None
    expected_batch: ConstructionBatch | None = None
    replay_graph: PropertyGraph | None = None
    predicted_coreference: dict[str, str] | None = None
    gold_coreference: dict[str, str] | None = None
    predicted_links: dict[str, str | None] | None = None
    gold_links: dict[str, str | None] | None = None
    predicted_answers: dict[str, list[str]] | None = None
    gold_answers: dict[str, list[str]] | None = None
    vector_rankings: dict[str, list[str]] | None = None
    vector_relevant: dict[str, list[str]] | None = None
    vector_k: int = Field(default=10, ge=1)
    runtime_report: RuntimeReport | None = None
    successful_runs: int | None = Field(default=None, ge=0)
    total_runs: int | None = Field(default=None, ge=0)


_FAMILY_ORDER: tuple[QualityFamily, ...] = (
    "formation",
    "structured",
    "category",
    "idempotence",
    "ner",
    "relation",
    "coreference",
    "linking",
    "query",
    "vector",
    "operational",
)

_UNIVERSAL_FAMILIES: frozenset[QualityFamily] = frozenset(
    {
        "formation",
        "category",
        "idempotence",
        "query",
        "vector",
        "operational",
    }
)
_APPLICABLE_FAMILIES: dict[
    QualityEvaluationProfile,
    frozenset[QualityFamily],
] = {
    "structured": _UNIVERSAL_FAMILIES | {"structured"},
    "unstructured": _UNIVERSAL_FAMILIES
    | {"structured", "ner", "relation", "coreference", "linking"},
    "mixed": frozenset(_FAMILY_ORDER),
}


class KnowledgeGraphQualityEvaluator:
    """Evaluate a construction result/batch against one mandatory quality contract."""

    name = "knowledge_graph_quality_evaluator"

    def __init__(
        self,
        thresholds: VersionedQualityThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or VersionedQualityThresholds()

    def evaluate(
        self,
        subject: object,
        *,
        inputs: QualityEvaluationInputs | None = None,
    ) -> KnowledgeGraphQualityReport:
        """Evaluate ``ConstructionResult`` or ``ConstructionBatch`` plus gold inputs."""

        supplied = inputs or QualityEvaluationInputs()
        batch = self._batch(subject)
        graph = supplied.property_graph or getattr(subject, "property_graph", None)
        rdf_graph = supplied.rdf_graph or getattr(subject, "rdf_graph", None)
        validation_report = supplied.validation_report or getattr(
            subject,
            "validation_report",
            None,
        )
        source_records = supplied.source_records or self._source_rows(batch)
        ontology = supplied.ontology or self._ontology(supplied.resolved_plan)
        policy = supplied.resolved_plan or supplied.policy

        metric_sets: dict[QualityFamily, tuple[bool, dict[str, float], str]] = {}
        schema_gate_results: list[QualityGateResult] = []
        if (
            isinstance(graph, PropertyGraph)
            and isinstance(rdf_graph, RDFGraph)
            and isinstance(validation_report, ValidationReport)
        ):
            formation = ValidationMetrics().evaluate(
                batch,
                validation_report,
                property_graph=graph,
                rdf_graph=rdf_graph,
            )
            formation.update(CoverageMetrics().evaluate(batch))
            formation.update(_exact_traceability_metrics(batch))
            formation.update(
                {
                    "validation_passed": float(validation_report.valid),
                    "constructed_object_support": float(
                        len(batch.entities) + len(batch.relations) + len(batch.facts)
                    ),
                }
            )
            metric_sets["formation"] = (
                True,
                formation,
                "integrity, evidence, provenance, and source coverage evaluated",
            )
        else:
            metric_sets["formation"] = (
                False,
                {},
                "requires property/RDF graphs and the construction validation report",
            )

        if graph is not None and ontology is not None:
            schema = SchemaQualityEvaluator().evaluate(
                graph,
                policy=policy,
                source_records=source_records,
                ontology=ontology,
                shacl_executed=supplied.shacl_executed,
                shacl_violation_count=supplied.shacl_violation_count,
                gate_settings=SchemaQualityGateSettings(),
            )
            payload = schema.metric_payload()
            payload.update(
                {
                    "property_completeness": schema.property_completeness,
                    "relationship_completeness": schema.relationship_completeness,
                    "shacl_executed": float(schema.shacl_executed),
                    "shacl_violation_count": float(schema.shacl_violation_count or 0),
                    "graph_object_support": float(
                        len(graph.nodes) + len(graph.edges)
                    ),
                }
            )
            for gate in schema.gates:
                metric_name = f"schema_gate_{gate.gate}"
                payload[metric_name] = float(gate.passed)
                schema_gate_results.append(
                    QualityGateResult(
                        family="structured",
                        metric=metric_name,
                        operator=">=",
                        threshold=1.0,
                        actual=float(gate.passed),
                        executed=True,
                        passed=gate.passed,
                        reason=(
                            f"schema gate {gate.gate!r} satisfied"
                            if gate.passed
                            else (
                                f"schema gate {gate.gate!r} failed: "
                                f"{gate.actual} {gate.operator} {gate.threshold}"
                            )
                        ),
                    )
                )
            metric_sets["structured"] = (True, payload, "schema obligations evaluated")
        elif (
            graph is not None
            and supplied.governed_schema_id
            and supplied.governed_schema_version
            and supplied.governed_schema_sha256
            and supplied.shacl_executed
            and supplied.shacl_checked_count is not None
            and supplied.shacl_checked_count > 0
            and supplied.shacl_violation_count is not None
        ):
            checked = supplied.shacl_checked_count
            violations = supplied.shacl_violation_count
            conformance = max(0.0, 1.0 - violations / checked)
            metric_sets["structured"] = (
                True,
                {
                    "property_completeness": conformance,
                    "relationship_completeness": conformance,
                    "consistency_violation_count": float(violations),
                    "shacl_executed": 1.0,
                    "shacl_violation_count": float(violations),
                    "graph_object_support": float(
                        len(graph.nodes) + len(graph.edges)
                    ),
                    "categorical_duplication_rate": (
                        _canonical_identity_duplication_rate(graph)
                    ),
                    "governed_schema_claim_support": float(checked),
                    "governed_schema_conformance": conformance,
                },
                (
                    "versioned governed unstructured schema "
                    f"{supplied.governed_schema_id}@"
                    f"{supplied.governed_schema_version} "
                    f"({supplied.governed_schema_sha256}) evaluated by SHACL"
                ),
            )
        else:
            metric_sets["structured"] = (
                False,
                {},
                "requires a property graph and resolved ontology",
            )

        category_value = (
            metric_sets["structured"][1].get("categorical_duplication_rate")
            if metric_sets["structured"][0]
            else None
        )
        metric_sets["category"] = (
            category_value is not None,
            (
                {"categorical_duplication_rate": category_value}
                if category_value is not None
                else {}
            ),
            (
                "categorical materialization evaluated"
                if category_value is not None
                else "requires source rows and a resolved modelling plan/policy"
            ),
        )

        if graph is not None and supplied.replay_graph is not None:
            metric_sets["idempotence"] = (
                True,
                _idempotence_metrics(graph, supplied.replay_graph),
                "initial and replay graphs compared",
            )
        else:
            metric_sets["idempotence"] = (
                False,
                {},
                "requires both the initial and replay property graphs",
            )

        if supplied.expected_batch is not None:
            accuracy = AccuracyMetrics().evaluate(batch, supplied.expected_batch)
            metric_sets["ner"] = (
                True,
                {
                    "entity_precision": accuracy["entity_precision"],
                    "entity_recall": accuracy["entity_recall"],
                    "entity_f1": accuracy["entity_f1"],
                    "entity_gold_support": accuracy["entity_gold_support"],
                    "entity_prediction_support": accuracy[
                        "entity_prediction_support"
                    ],
                    "entity_gold_coverage": accuracy["entity_gold_coverage"],
                    "span_precision": accuracy["span_precision"],
                    "span_recall": accuracy["span_recall"],
                    "span_f1": accuracy["span_f1"],
                    "span_gold_support": accuracy["span_gold_support"],
                    "span_prediction_support": accuracy[
                        "span_prediction_support"
                    ],
                    "span_gold_coverage": accuracy["span_gold_coverage"],
                },
                "entity and span gold evaluated",
            )
            metric_sets["relation"] = (
                True,
                {
                    "relation_precision": accuracy["relation_precision"],
                    "relation_recall": accuracy["relation_recall"],
                    "relation_f1": accuracy["relation_f1"],
                    "relation_gold_support": accuracy["relation_gold_support"],
                    "relation_prediction_support": accuracy[
                        "relation_prediction_support"
                    ],
                    "relation_gold_coverage": accuracy[
                        "relation_gold_coverage"
                    ],
                },
                "relation gold evaluated",
            )
        else:
            metric_sets["ner"] = (False, {}, "requires a gold ConstructionBatch")
            metric_sets["relation"] = (False, {}, "requires a gold ConstructionBatch")

        metric_sets["coreference"] = self._mapping_family(
            supplied.predicted_coreference,
            supplied.gold_coreference,
            _coreference_metrics,
            "requires predicted and gold mention-to-cluster mappings",
        )
        metric_sets["linking"] = self._mapping_family(
            supplied.predicted_links,
            supplied.gold_links,
            _linking_metrics,
            "requires predicted and gold mention-to-entity mappings",
        )
        metric_sets["query"] = self._mapping_family(
            supplied.predicted_answers,
            supplied.gold_answers,
            _query_metrics,
            "requires predicted and gold competency-query answers",
        )
        if (
            supplied.vector_rankings is not None
            and supplied.vector_relevant is not None
            and supplied.vector_relevant
            and all(supplied.vector_relevant.values())
        ):
            metric_sets["vector"] = (
                True,
                _vector_metrics(
                    supplied.vector_rankings,
                    supplied.vector_relevant,
                    k=supplied.vector_k,
                ),
                f"ranked retrieval evaluated at k={supplied.vector_k}",
            )
        else:
            metric_sets["vector"] = (
                False,
                {},
                "requires non-empty vector rankings and positive relevance judgments",
            )

        if (
            supplied.runtime_report is not None
            and supplied.total_runs is not None
            and supplied.total_runs > 0
            and supplied.successful_runs is not None
            and 0 <= supplied.successful_runs <= supplied.total_runs
        ):
            operational = OperationalMetrics().evaluate(supplied.runtime_report)
            total = supplied.total_runs
            successful = supplied.successful_runs
            success_rate = successful / total
            operational["pipeline_success_rate"] = success_rate
            metric_sets["operational"] = (True, operational, "runtime execution measured")
        else:
            metric_sets["operational"] = (
                False,
                {},
                "requires a measured RuntimeReport and non-empty run support",
            )

        applicable = _APPLICABLE_FAMILIES[supplied.evaluation_profile]
        families = {
            family: self._family_report(
                family,
                *metric_sets[family],
                applicable=family in applicable,
            )
            for family in _FAMILY_ORDER
        }
        if schema_gate_results:
            structured = families["structured"]
            families["structured"] = structured.model_copy(
                update={
                    "gates": [*structured.gates, *schema_gate_results],
                    "passed": (
                        structured.passed
                        and all(gate.passed for gate in schema_gate_results)
                    ),
                    "reason": (
                        f"{structured.reason}; every schema-quality gate propagated"
                    ),
                }
            )
        missing = [
            family
            for family in _FAMILY_ORDER
            if family in applicable and not families[family].executed
        ]
        return KnowledgeGraphQualityReport(
            thresholds_version=self.thresholds.version,
            families=families,
            executed_families=[
                family for family in _FAMILY_ORDER if families[family].executed
            ],
            missing_families=missing,
            applicable_families=[
                family for family in _FAMILY_ORDER if family in applicable
            ],
            not_applicable_families=[
                family for family in _FAMILY_ORDER if family not in applicable
            ],
            evaluation_profile=supplied.evaluation_profile,
            governed_schema_id=supplied.governed_schema_id,
            governed_schema_version=supplied.governed_schema_version,
            governed_schema_sha256=supplied.governed_schema_sha256,
            shacl_checked_count=supplied.shacl_checked_count,
            shacl_violation_count=supplied.shacl_violation_count,
            passed=not missing
            and all(families[family].passed for family in applicable),
        )

    @staticmethod
    def _batch(subject: object) -> ConstructionBatch:
        if isinstance(subject, ConstructionBatch):
            return subject
        batch = getattr(subject, "batch", None)
        if not isinstance(batch, ConstructionBatch):
            raise TypeError("subject must be a ConstructionBatch or expose .batch")
        return batch

    @staticmethod
    def _source_rows(batch: ConstructionBatch) -> list[dict[str, object]] | None:
        rows = [row for source in batch.source_records for row in source.records]
        return rows or None

    @staticmethod
    def _ontology(resolved_plan: object | None) -> OntologySchema | None:
        if resolved_plan is None:
            return None
        project = getattr(resolved_plan, "to_ontology_schema", None)
        ontology = project() if callable(project) else None
        return ontology if isinstance(ontology, OntologySchema) else None

    @staticmethod
    def _mapping_family(
        predicted: Mapping[str, object] | None,
        gold: Mapping[str, object] | None,
        evaluator: object,
        missing_reason: str,
    ) -> tuple[bool, dict[str, float], str]:
        if predicted is None or gold is None:
            return False, {}, missing_reason
        if not gold:
            return False, {}, f"{missing_reason}; gold support must be non-empty"
        metrics = evaluator(predicted, gold)  # type: ignore[operator]
        return True, metrics, "gold judgments evaluated"

    def _family_report(
        self,
        family: QualityFamily,
        executed: bool,
        metrics: dict[str, float],
        reason: str,
        *,
        applicable: bool,
    ) -> MetricFamilyReport:
        if not applicable:
            return MetricFamilyReport(
                family=family,
                applicable=False,
                executed=False,
                metrics={},
                gates=[],
                passed=False,
                reason="not applicable to the declared construction pillar",
            )
        gates: list[QualityGateResult] = []
        for rule in self.thresholds.thresholds:
            if rule.family != family:
                continue
            actual = metrics.get(rule.metric) if executed else None
            valid = actual is not None and math.isfinite(actual)
            if actual is None or not math.isfinite(actual):
                passed = False
            elif rule.operator == ">=":
                passed = actual >= rule.threshold
            else:
                passed = actual <= rule.threshold
            gates.append(
                QualityGateResult(
                    family=family,
                    metric=rule.metric,
                    operator=rule.operator,
                    threshold=rule.threshold,
                    actual=actual,
                    executed=executed and valid,
                    passed=passed,
                    reason=(
                        "threshold satisfied"
                        if passed
                        else (
                            f"metric {rule.metric!r} was not executed"
                            if not valid
                            else "threshold not satisfied"
                        )
                    ),
                )
            )
        return MetricFamilyReport(
            family=family,
            applicable=True,
            executed=executed,
            metrics=metrics,
            gates=gates,
            passed=executed and all(gate.passed for gate in gates),
            reason=reason,
        )


def _graph_fingerprint(graph: PropertyGraph) -> set[str]:
    nodes = {
        json.dumps(
            {
                "kind": "node",
                "id": node.node_id,
                "labels": sorted(node.labels),
                "properties": node.properties,
            },
            sort_keys=True,
            default=str,
        )
        for node in graph.nodes
    }
    edges = {
        json.dumps(
            {
                "kind": "edge",
                "id": edge.edge_id,
                "source": edge.source_id,
                "target": edge.target_id,
                "type": edge.type,
                "properties": edge.properties,
            },
            sort_keys=True,
            default=str,
        )
        for edge in graph.edges
    }
    return nodes | edges


def _canonical_identity_duplication_rate(graph: PropertyGraph) -> float:
    """Measure duplicate governed identities without using runtime node IDs."""

    canonical = [
        node
        for node in graph.nodes
        if "CanonicalEntity" in node.labels
        or node.properties.get("identity_layer") == "canonical"
    ]
    if not canonical:
        return 0.0
    keys = [
        (
            str(node.properties.get("external_system", "")).casefold(),
            str(node.properties.get("external_id", "")).casefold(),
            str(node.properties.get("semantic_entity_type", "")).casefold(),
            str(node.properties.get("name", "")).casefold(),
        )
        for node in canonical
    ]
    return 1.0 - len(set(keys)) / len(keys)


def _idempotence_metrics(first: PropertyGraph, replay: PropertyGraph) -> dict[str, float]:
    left = _graph_fingerprint(first)
    right = _graph_fingerprint(replay)
    if not left and not right:
        score = 1.0
    else:
        union = left | right
        score = len(left & right) / len(union) if union else 1.0
    return {
        "graph_idempotence_score": score,
        "idempotence_added_object_count": float(len(right - left)),
        "idempotence_missing_object_count": float(len(left - right)),
    }


def _coreference_metrics(
    predicted: Mapping[str, object],
    gold: Mapping[str, object],
) -> dict[str, float]:
    mentions = set(predicted) | set(gold)
    if not mentions:
        return {
            "coreference_b3_precision": 0.0,
            "coreference_b3_recall": 0.0,
            "coreference_b3_f1": 0.0,
            "coreference_mention_coverage": 0.0,
            "coreference_gold_support": 0.0,
            "coreference_prediction_support": 0.0,
        }
    predicted_clusters = {
        mention: {
            candidate
            for candidate in mentions
            if predicted.get(candidate, f"__predicted_missing__:{candidate}")
            == predicted.get(mention, f"__predicted_missing__:{mention}")
        }
        for mention in mentions
    }
    gold_clusters = {
        mention: {
            candidate
            for candidate in mentions
            if gold.get(candidate, f"__gold_missing__:{candidate}")
            == gold.get(mention, f"__gold_missing__:{mention}")
        }
        for mention in mentions
    }
    precision = sum(
        len(predicted_clusters[m] & gold_clusters[m]) / len(predicted_clusters[m])
        for m in mentions
    ) / len(mentions)
    recall = sum(
        len(predicted_clusters[m] & gold_clusters[m]) / len(gold_clusters[m])
        for m in mentions
    ) / len(mentions)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "coreference_b3_precision": precision,
        "coreference_b3_recall": recall,
        "coreference_b3_f1": f1,
        "coreference_mention_coverage": len(set(predicted) & set(gold)) / len(set(gold))
        if gold
        else 0.0,
        "coreference_gold_support": float(len(gold)),
        "coreference_prediction_support": float(len(predicted)),
    }


def _linking_metrics(
    predicted: Mapping[str, object],
    gold: Mapping[str, object],
) -> dict[str, float]:
    mentions = set(predicted) | set(gold)
    if not mentions:
        return {
            "entity_linking_accuracy": 0.0,
            "entity_linking_non_nil_accuracy": 0.0,
            "entity_linking_mention_coverage": 0.0,
            "entity_linking_gold_support": 0.0,
            "entity_linking_prediction_support": 0.0,
        }
    correct = sum(predicted.get(item) == gold.get(item) for item in mentions)
    non_nil = {item for item, value in gold.items() if value not in (None, "", "NIL")}
    non_nil_correct = sum(predicted.get(item) == gold.get(item) for item in non_nil)
    return {
        "entity_linking_accuracy": correct / len(mentions),
        "entity_linking_non_nil_accuracy": (
            non_nil_correct / len(non_nil) if non_nil else 0.0
        ),
        "entity_linking_mention_coverage": len(set(predicted) & set(gold)) / len(set(gold))
        if gold
        else 0.0,
        "entity_linking_gold_support": float(len(gold)),
        "entity_linking_prediction_support": float(len(predicted)),
    }


def _query_metrics(
    predicted: Mapping[str, object],
    gold: Mapping[str, object],
) -> dict[str, float]:
    query_ids = set(predicted) | set(gold)
    if not query_ids:
        return {
            "competency_question_accuracy": 0.0,
            "query_answer_f1": 0.0,
            "query_prediction_completeness": 0.0,
            "query_gold_support": 0.0,
            "query_prediction_support": 0.0,
        }
    exact = 0
    f1_total = 0.0
    for query_id in query_ids:
        predicted_answers = _answer_set(predicted.get(query_id))
        gold_answers = _answer_set(gold.get(query_id))
        exact += predicted_answers == gold_answers
        overlap = len(predicted_answers & gold_answers)
        precision = overlap / len(predicted_answers) if predicted_answers else 0.0
        recall = overlap / len(gold_answers) if gold_answers else 1.0
        f1_total += (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "competency_question_accuracy": exact / len(query_ids),
        "query_answer_f1": f1_total / len(query_ids),
        "query_prediction_completeness": len(set(predicted) & set(gold)) / len(set(gold))
        if gold
        else 0.0,
        "query_gold_support": float(len(gold)),
        "query_prediction_support": float(len(predicted)),
    }


def _answer_set(value: object) -> set[str]:
    """Normalize one answer collection without treating a scalar as a collection."""

    if not isinstance(value, list | tuple | set | frozenset):
        return set()
    return {str(item).casefold() for item in value}


def _vector_metrics(
    rankings: Mapping[str, list[str]],
    relevant: Mapping[str, list[str]],
    *,
    k: int,
) -> dict[str, float]:
    query_ids = set(rankings) | set(relevant)
    if not query_ids:
        return {
            "vector_recall_at_k": 0.0,
            "vector_mrr_at_k": 0.0,
            "vector_query_coverage": 0.0,
            "vector_gold_support": 0.0,
            "vector_prediction_support": 0.0,
        }
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    for query_id in query_ids:
        ranked = list(rankings.get(query_id, []))[:k]
        expected = set(relevant.get(query_id, []))
        recalls.append(len(set(ranked) & expected) / len(expected) if expected else 0.0)
        first = next(
            (index for index, item in enumerate(ranked, start=1) if item in expected),
            None,
        )
        reciprocal_ranks.append(1.0 / first if first is not None else 0.0)
    return {
        "vector_recall_at_k": sum(recalls) / len(recalls),
        "vector_mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "vector_query_coverage": len(set(rankings) & set(relevant)) / len(set(relevant))
        if relevant
        else 0.0,
        "vector_gold_support": float(len(relevant)),
        "vector_prediction_support": float(len(rankings)),
    }


def _exact_traceability_metrics(batch: ConstructionBatch) -> dict[str, float]:
    objects: list[EntityCandidate | RelationCandidate | GraphFact] = [
        *batch.entities,
        *batch.relations,
        *batch.facts,
    ]
    if not objects:
        return {
            "exact_evidence_coverage": 1.0,
            "basic_provenance_coverage": 1.0,
            "model_provenance_coverage": 1.0,
        }
    sources = {source.source_id: source for source in batch.source_records}
    exact = sum(
        bool(item.evidence)
        and all(_evidence_is_exact(evidence, sources) for evidence in item.evidence)
        for item in objects
    )
    basic_provenance = sum(
        bool(
            item.extractor_name.strip()
            and item.technique.strip()
            and item.schema_version.strip()
        )
        for item in objects
    )
    model_objects = [
        item
        for item in objects
        if item.provenance_metadata.get("provider")
        or item.provenance_metadata.get("model_version")
    ]
    model_complete = sum(
        all(
            str(item.provenance_metadata.get(key, "")).strip()
            for key in ("provider", "model_version", "configuration_id")
        )
        for item in model_objects
    )
    return {
        "exact_evidence_coverage": exact / len(objects),
        "basic_provenance_coverage": basic_provenance / len(objects),
        "model_provenance_coverage": (
            model_complete / len(model_objects) if model_objects else 1.0
        ),
    }


def _evidence_is_exact(
    evidence: object,
    sources: Mapping[str, object],
) -> bool:
    source_id = str(getattr(evidence, "source_id", ""))
    source = sources.get(source_id)
    location = getattr(evidence, "location", None)
    quote = getattr(evidence, "text", None)
    if source is None or not isinstance(location, dict) or not location:
        return False
    content = getattr(source, "content", None)
    records = getattr(source, "records", [])
    kind = str(location.get("kind", ""))
    start = location.get("start")
    end = location.get("end")
    if isinstance(content, str) and isinstance(start, int) and isinstance(end, int):
        return (
            0 <= start <= end <= len(content)
            and isinstance(quote, str)
            and content[start:end] == quote
        )
    row_index = location.get("row_index")
    if kind == "row" and isinstance(row_index, int):
        return (
            0 <= row_index < len(records)
            and quote == repr(records[row_index])
        )
    if kind == "cell" and isinstance(row_index, int):
        column = location.get("column")
        return (
            0 <= row_index < len(records)
            and isinstance(column, str)
            and column in records[row_index]
            and quote == str(records[row_index][column])
        )
    # Structured-path/loader evidence is exact by stable source snapshot plus a
    # non-empty selector. Its parser-specific selector is validated by the
    # producing loader/mapper contract rather than guessed here.
    return bool(source_id and kind)


__all__ = [
    "QUALITY_THRESHOLDS_VERSION",
    "KnowledgeGraphQualityEvaluator",
    "KnowledgeGraphQualityReport",
    "MetricFamilyReport",
    "QualityEvaluationInputs",
    "QualityGateResult",
    "QualityThreshold",
    "VersionedQualityThresholds",
]
