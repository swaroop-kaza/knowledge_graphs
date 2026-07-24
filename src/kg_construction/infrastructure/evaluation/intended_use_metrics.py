"""Intended-use benchmarks for a constructed knowledge graph.

Formation metrics are necessary, but they are not sufficient to claim that a
graph is useful.  This module evaluates the graph through the workloads it is
expected to serve:

* SQL-like result and aggregation queries;
* document/NoSQL-style traversals and paths;
* GraphRAG retrieval and answer grounding;
* vector search;
* filtered knowledge-graph link prediction;
* recommendation; and
* deductive reasoning and competency questions.

Every evaluator requires explicit gold judgments and operational observations.
An absent or incomplete benchmark is represented by ``executed=False`` and
``passed=False``.  It is never converted into a successful zero-valued metric.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ScalarValue = str | int | float | bool | None
IntendedUse = Literal[
    "sql_like",
    "nosql",
    "graphrag",
    "vector",
    "link_prediction",
    "recommendation",
    "reasoning",
]
ThresholdOperator = Literal[">=", "<="]
PopularityStratum = Literal["head", "torso", "tail"]


class _Contract(BaseModel):
    """Strict base class for benchmark inputs and reports."""

    model_config = ConfigDict(extra="forbid")


class MetricThresholdResult(_Contract):
    """The observed result of applying one intended-use threshold."""

    metric: str
    operator: ThresholdOperator
    threshold: float
    actual: float
    passed: bool


class PopularityStratumReport(_Contract):
    """Filtered-ranking metrics for one entity-popularity stratum."""

    case_count: int = Field(ge=1)
    mrr: float = Field(ge=0.0, le=1.0)
    hits_at_k: dict[int, float]

    @model_validator(mode="after")
    def _valid_hits(self) -> PopularityStratumReport:
        if not self.hits_at_k:
            raise ValueError("hits_at_k must contain at least one cutoff")
        if any(cutoff < 1 for cutoff in self.hits_at_k):
            raise ValueError("Hits@k cutoffs must be positive")
        if any(value < 0.0 or value > 1.0 for value in self.hits_at_k.values()):
            raise ValueError("Hits@k values must be in [0, 1]")
        return self


class IntendedUseMetricReport(_Contract):
    """Metrics and pass state for one intended-use benchmark."""

    intended_use: IntendedUse
    executed: bool
    passed: bool
    reason: str
    evaluated_cases: int = Field(default=0, ge=0)
    metrics: dict[str, float] = Field(default_factory=dict)
    thresholds: list[MetricThresholdResult] = Field(default_factory=list)
    popularity_strata: dict[PopularityStratum, PopularityStratumReport] = Field(
        default_factory=dict
    )
    workload_support: int = Field(default=0, ge=0)
    graph_sha256: str | None = None
    fixture_sha256: str | None = None
    graph_backend_id: str | None = None
    graph_snapshot_id: str | None = None
    query_sha256: list[str] = Field(default_factory=list)
    observed_latency_ms: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _execution_state_is_honest(self) -> IntendedUseMetricReport:
        if not self.executed:
            if self.passed:
                raise ValueError("an unexecuted benchmark cannot pass")
            if self.evaluated_cases != 0 or self.metrics or self.thresholds:
                raise ValueError(
                    "an unexecuted benchmark cannot publish cases, metrics, or thresholds"
                )
        elif self.evaluated_cases < 1:
            raise ValueError("an executed benchmark must evaluate at least one case")
        if self.workload_support not in {0, self.evaluated_cases}:
            raise ValueError(
                "workload_support must be zero for legacy reports or equal evaluated_cases"
            )
        evidence_fields = (
            self.graph_sha256,
            self.fixture_sha256,
            self.graph_backend_id,
            self.graph_snapshot_id,
            self.query_sha256,
            self.observed_latency_ms,
        )
        if any(evidence_fields) and not all(
            (
                self.graph_sha256,
                self.fixture_sha256,
                self.graph_backend_id,
                self.graph_snapshot_id,
                self.query_sha256,
                self.observed_latency_ms,
            )
        ):
            raise ValueError(
                "graph-bound evidence requires graph/fixture hashes, query hashes, and latencies"
            )
        if self.graph_sha256 is not None:
            if self.workload_support != self.evaluated_cases:
                raise ValueError(
                    "graph-bound evidence requires per-family workload support"
                )
            if len(self.query_sha256) != self.evaluated_cases:
                raise ValueError("one query hash is required for every evaluated case")
            if len(self.observed_latency_ms) != self.evaluated_cases:
                raise ValueError("one observed latency is required for every evaluated case")
        return self


class IntendedUseEvaluationReport(_Contract):
    """Complete report across every supported graph workload."""

    families: dict[IntendedUse, IntendedUseMetricReport]
    executed_uses: list[IntendedUse]
    missing_uses: list[IntendedUse]
    passed: bool
    suite_version: str | None = None
    graph_sha256: str | None = None
    fixture_sha256: str | None = None
    graph_backend_id: str | None = None
    graph_snapshot_id: str | None = None
    graph_bound: bool = False
    construction_profile: Literal["structured", "unstructured", "mixed"] | None = None
    training_graph_sha256: str | None = None
    held_out_edge_ids: list[str] = Field(default_factory=list)
    derived_popularity_strata: dict[str, PopularityStratum] = Field(
        default_factory=dict
    )
    formation_gold_executed: bool = False
    formation_gold_passed: bool = False
    formation_gold_metrics: dict[str, float] = Field(default_factory=dict)
    formation_gold_assertions: dict[str, bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _summary_matches_families(self) -> IntendedUseEvaluationReport:
        expected = set(_USE_ORDER)
        if set(self.families) != expected:
            raise ValueError("the report must contain every intended-use family")
        executed = {
            intended_use
            for intended_use, report in self.families.items()
            if report.executed
        }
        if set(self.executed_uses) != executed:
            raise ValueError("executed_uses must match family execution states")
        if set(self.missing_uses) != expected - executed:
            raise ValueError("missing_uses must match unexecuted families")
        family_pass = not self.missing_uses and all(
            report.passed for report in self.families.values()
        )
        formation_required = self.construction_profile in {"structured", "mixed"}
        expected_pass = family_pass and (
            self.formation_gold_passed if formation_required else True
        )
        if self.passed != expected_pass:
            raise ValueError(
                "overall pass state must match mandatory family and formation states"
            )
        if self.graph_bound:
            if not all(
                (
                    self.suite_version,
                    self.graph_sha256,
                    self.fixture_sha256,
                    self.graph_backend_id,
                    self.graph_snapshot_id,
                    self.construction_profile,
                    self.training_graph_sha256,
                    self.held_out_edge_ids,
                    self.derived_popularity_strata,
                )
            ):
                raise ValueError(
                    "a graph-bound report requires suite, graph, fixture, backend, "
                    "snapshot, held-out-training, and construction-profile evidence"
                )
            if set(self.derived_popularity_strata.values()) != {
                "head",
                "torso",
                "tail",
            }:
                raise ValueError(
                    "graph-bound link prediction requires training-derived "
                    "head, torso, and tail support"
                )
            if any(
                family.graph_sha256 != self.graph_sha256
                or family.fixture_sha256 != self.fixture_sha256
                or family.graph_backend_id != self.graph_backend_id
                or family.graph_snapshot_id != self.graph_snapshot_id
                for family in self.families.values()
            ):
                raise ValueError(
                    "every graph-bound family must reference the evaluated graph and fixture"
                )
            if formation_required:
                if (
                    not self.formation_gold_executed
                    or not self.formation_gold_metrics
                    or not self.formation_gold_assertions
                    or self.formation_gold_passed
                    != all(self.formation_gold_assertions.values())
                ):
                    raise ValueError(
                        "structured/mixed graph-bound reports require a non-empty, "
                        "conjunctive structured-formation evaluation"
                    )
            elif any(
                (
                    self.formation_gold_executed,
                    self.formation_gold_passed,
                    self.formation_gold_metrics,
                    self.formation_gold_assertions,
                )
            ):
                raise ValueError(
                    "unstructured graph-bound reports cannot claim "
                    "structured-formation execution"
                )
        elif any(
            (
                self.suite_version,
                self.graph_sha256,
                self.fixture_sha256,
                self.graph_backend_id,
                self.graph_snapshot_id,
                self.construction_profile,
                self.training_graph_sha256,
                self.held_out_edge_ids,
                self.derived_popularity_strata,
                self.formation_gold_executed,
                self.formation_gold_passed,
                self.formation_gold_metrics,
                self.formation_gold_assertions,
            )
        ):
            raise ValueError("unbound reports cannot claim graph-bound suite evidence")
        return self


class SQLLikeQueryCase(_Contract):
    """Observed and gold outputs for one SQL-like/property query."""

    case_id: str = Field(min_length=1)
    predicted_rows: list[dict[str, ScalarValue]]
    expected_rows: list[dict[str, ScalarValue]]
    predicted_aggregations: dict[str, ScalarValue] = Field(default_factory=dict)
    expected_aggregations: dict[str, ScalarValue] = Field(default_factory=dict)
    ordered_results: bool = False
    latency_ms: float = Field(ge=0.0)


class SQLLikeEvaluationInput(_Contract):
    """SQL-like benchmark cases, thresholds, and latency service objective."""

    cases: list[SQLLikeQueryCase] = Field(default_factory=list)
    max_p95_latency_ms: float = Field(gt=0.0)
    numeric_relative_tolerance: float = Field(default=1e-6, ge=0.0)
    numeric_absolute_tolerance: float = Field(default=1e-9, ge=0.0)
    min_query_result_exact_match: float = Field(default=1.0, ge=0.0, le=1.0)
    min_aggregation_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)


class NoSQLTraversalCase(_Contract):
    """Observed and gold nodes/paths for one document-style traversal."""

    case_id: str = Field(min_length=1)
    visited_node_ids: list[str]
    expected_node_ids: list[str]
    returned_paths: list[list[str]]
    expected_paths: list[list[str]]
    latency_ms: float = Field(ge=0.0)


class NoSQLEvaluationInput(_Contract):
    """Traversal benchmark cases and mandatory quality/latency targets."""

    cases: list[NoSQLTraversalCase] = Field(default_factory=list)
    max_p95_latency_ms: float = Field(gt=0.0)
    min_traversal_result_exact_match: float = Field(default=1.0, ge=0.0, le=1.0)
    min_traversal_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_path_f1: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphRAGCase(_Contract):
    """Gold contexts, paths, answer judgments, and citations for one question."""

    case_id: str = Field(min_length=1)
    retrieved_context_ids: list[str]
    context_relevance: dict[str, float]
    retrieved_paths: list[list[str]]
    expected_paths: list[list[str]]
    requires_multi_hop: bool
    multi_hop_answer_correct: bool | None = None
    answer_faithfulness: float = Field(ge=0.0, le=1.0)
    predicted_citation_ids: list[str]
    expected_citation_ids: list[str]

    @model_validator(mode="after")
    def _judgments_are_valid(self) -> GraphRAGCase:
        if self.requires_multi_hop and self.multi_hop_answer_correct is None:
            raise ValueError(
                "multi_hop_answer_correct is required for a multi-hop case"
            )
        _validate_relevance(self.context_relevance, label="context relevance")
        return self


class GraphRAGEvaluationInput(_Contract):
    """GraphRAG benchmark cases and acceptance thresholds."""

    cases: list[GraphRAGCase] = Field(default_factory=list)
    k: int = Field(default=10, ge=1)
    min_recall_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    min_mrr: float = Field(default=0.60, ge=0.0, le=1.0)
    min_ndcg_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    min_path_f1: float = Field(default=0.70, ge=0.0, le=1.0)
    min_multi_hop_success_rate: float = Field(default=0.70, ge=0.0, le=1.0)
    min_answer_faithfulness: float = Field(default=0.80, ge=0.0, le=1.0)
    min_citation_f1: float = Field(default=0.80, ge=0.0, le=1.0)


class VectorIndexCase(_Contract):
    """One filtered vector query with graded relevance judgments."""

    case_id: str = Field(min_length=1)
    retrieved_ids: list[str]
    relevance: dict[str, float]
    filter_allowed_ids: list[str]
    latency_ms: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _judgments_are_valid(self) -> VectorIndexCase:
        _validate_relevance(self.relevance, label="vector relevance")
        return self


class VectorIndexEvaluationInput(_Contract):
    """Vector-index population, query judgments, and resource objectives."""

    cases: list[VectorIndexCase] = Field(default_factory=list)
    eligible_item_count: int = Field(gt=0)
    indexed_item_count: int = Field(ge=0)
    peak_memory_bytes: int = Field(ge=0)
    k: int = Field(default=10, ge=1)
    max_p95_latency_ms: float = Field(gt=0.0)
    max_peak_memory_bytes: int = Field(gt=0)
    min_index_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_recall_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    min_mrr: float = Field(default=0.60, ge=0.0, le=1.0)
    min_ndcg_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    min_filter_correctness: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _index_counts_are_consistent(self) -> VectorIndexEvaluationInput:
        if self.indexed_item_count > self.eligible_item_count:
            raise ValueError("indexed_item_count cannot exceed eligible_item_count")
        return self


class LinkPredictionCase(_Contract):
    """One raw ranking plus truths required by the filtered protocol."""

    case_id: str = Field(min_length=1)
    ranked_entity_ids: list[str]
    true_entity_id: str = Field(min_length=1)
    known_true_entity_ids: list[str]
    popularity_stratum: PopularityStratum


class LinkPredictionEvaluationInput(_Contract):
    """Filtered link-prediction cases and ranking cutoffs."""

    cases: list[LinkPredictionCase] = Field(default_factory=list)
    hits_at: list[int] = Field(default_factory=lambda: [1, 3, 10])
    min_filtered_mrr: float = Field(default=0.20, ge=0.0, le=1.0)
    min_filtered_hits_at_k: dict[int, float] = Field(
        default_factory=lambda: {1: 0.10, 3: 0.20, 10: 0.30}
    )

    @model_validator(mode="after")
    def _cutoffs_are_consistent(self) -> LinkPredictionEvaluationInput:
        if not self.hits_at or any(cutoff < 1 for cutoff in self.hits_at):
            raise ValueError("hits_at must contain positive cutoffs")
        if len(self.hits_at) != len(set(self.hits_at)):
            raise ValueError("hits_at cutoffs must be unique")
        if set(self.min_filtered_hits_at_k) != set(self.hits_at):
            raise ValueError(
                "min_filtered_hits_at_k must define exactly the requested Hits@k cutoffs"
            )
        if any(
            threshold < 0.0 or threshold > 1.0
            for threshold in self.min_filtered_hits_at_k.values()
        ):
            raise ValueError("Hits@k thresholds must be in [0, 1]")
        return self


class RecommendationCase(_Contract):
    """Top-k recommendations and human/behavioral relevance judgments."""

    case_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    recommended_item_ids: list[str]
    relevant_item_ids: list[str]


class RecommendationEvaluationInput(_Contract):
    """Recommendation judgments, catalog metadata, and quality thresholds."""

    cases: list[RecommendationCase] = Field(default_factory=list)
    catalog_item_ids: list[str]
    item_popularity_probability: dict[str, float]
    item_features: dict[str, list[str]]
    k: int = Field(default=10, ge=1)
    novelty_probability_floor: float = Field(default=1e-12, gt=0.0, le=1.0)
    min_precision_at_k: float = Field(default=0.01, ge=0.0, le=1.0)
    min_recall_at_k: float = Field(default=0.01, ge=0.0, le=1.0)
    min_ndcg_at_k: float = Field(default=0.01, ge=0.0, le=1.0)
    min_catalog_coverage: float = Field(default=0.01, ge=0.0, le=1.0)
    min_novelty: float = Field(default=0.01, ge=0.0)
    min_intra_list_diversity: float = Field(default=0.01, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _probabilities_are_valid(self) -> RecommendationEvaluationInput:
        invalid = {
            item_id: probability
            for item_id, probability in self.item_popularity_probability.items()
            if probability < 0.0 or probability > 1.0
        }
        if invalid:
            raise ValueError(
                "item popularity probabilities must be in [0, 1]: "
                f"{sorted(invalid)}"
            )
        return self


class ReasoningFact(_Contract):
    """A canonical inferred fact used for exact benchmark comparison."""

    subject: str
    predicate: str
    object: str


class ReasoningCase(_Contract):
    """One competency question and the inferred facts used to answer it."""

    case_id: str = Field(min_length=1)
    predicted_answers: list[str]
    expected_answers: list[str]
    predicted_inferred_facts: list[ReasoningFact]
    expected_inferred_facts: list[ReasoningFact]
    runtime_ms: float = Field(ge=0.0)


class ReasoningEvaluationInput(_Contract):
    """Reasoning benchmark and correctness/runtime objectives."""

    cases: list[ReasoningCase] = Field(default_factory=list)
    max_p95_runtime_ms: float = Field(gt=0.0)
    min_competency_question_accuracy: float = Field(
        default=0.80, ge=0.0, le=1.0
    )
    min_inferred_fact_f1: float = Field(default=0.80, ge=0.0, le=1.0)


class IntendedUseEvaluationInputs(_Contract):
    """Gold suites for every use; each omitted suite is a reported failed execution."""

    sql_like: SQLLikeEvaluationInput | None = None
    nosql: NoSQLEvaluationInput | None = None
    graphrag: GraphRAGEvaluationInput | None = None
    vector: VectorIndexEvaluationInput | None = None
    link_prediction: LinkPredictionEvaluationInput | None = None
    recommendation: RecommendationEvaluationInput | None = None
    reasoning: ReasoningEvaluationInput | None = None


_USE_ORDER: tuple[IntendedUse, ...] = (
    "sql_like",
    "nosql",
    "graphrag",
    "vector",
    "link_prediction",
    "recommendation",
    "reasoning",
)


class IntendedUseMetricsEvaluator:
    """Evaluate a graph's mandatory downstream workload benchmarks."""

    name = "intended_use_metrics_evaluator"

    def evaluate(
        self,
        inputs: IntendedUseEvaluationInputs | None = None,
    ) -> IntendedUseEvaluationReport:
        supplied = inputs or IntendedUseEvaluationInputs()
        families: dict[IntendedUse, IntendedUseMetricReport] = {
            "sql_like": self.evaluate_sql_like(supplied.sql_like),
            "nosql": self.evaluate_nosql(supplied.nosql),
            "graphrag": self.evaluate_graphrag(supplied.graphrag),
            "vector": self.evaluate_vector(supplied.vector),
            "link_prediction": self.evaluate_link_prediction(
                supplied.link_prediction
            ),
            "recommendation": self.evaluate_recommendation(
                supplied.recommendation
            ),
            "reasoning": self.evaluate_reasoning(supplied.reasoning),
        }
        executed = [use for use in _USE_ORDER if families[use].executed]
        missing = [use for use in _USE_ORDER if not families[use].executed]
        return IntendedUseEvaluationReport(
            families=families,
            executed_uses=executed,
            missing_uses=missing,
            passed=not missing and all(report.passed for report in families.values()),
        )

    def evaluate_sql_like(
        self,
        suite: SQLLikeEvaluationInput | None,
    ) -> IntendedUseMetricReport:
        if suite is None or not suite.cases:
            return _not_executed(
                "sql_like",
                "requires SQL-like cases with expected rows, aggregations, and latency",
            )
        if not any(case.expected_aggregations for case in suite.cases):
            return _not_executed(
                "sql_like",
                "requires at least one expected aggregation judgment",
            )
        row_true_positive = 0
        row_predicted = 0
        row_expected = 0
        exact_matches = 0
        correct_aggregations = 0
        aggregation_judgments = 0
        latencies: list[float] = []
        for case in suite.cases:
            predicted = [_canonical_row(row) for row in case.predicted_rows]
            expected = [_canonical_row(row) for row in case.expected_rows]
            if case.ordered_results:
                exact_matches += int(predicted == expected)
            else:
                exact_matches += int(Counter(predicted) == Counter(expected))
            predicted_counter = Counter(predicted)
            expected_counter = Counter(expected)
            row_true_positive += sum((predicted_counter & expected_counter).values())
            row_predicted += sum(predicted_counter.values())
            row_expected += sum(expected_counter.values())
            aggregate_names = set(case.predicted_aggregations) | set(
                case.expected_aggregations
            )
            for name in aggregate_names:
                aggregation_judgments += 1
                if name not in case.predicted_aggregations:
                    continue
                if name not in case.expected_aggregations:
                    continue
                correct_aggregations += int(
                    _scalar_equal(
                        case.predicted_aggregations[name],
                        case.expected_aggregations[name],
                        relative_tolerance=suite.numeric_relative_tolerance,
                        absolute_tolerance=suite.numeric_absolute_tolerance,
                    )
                )
            latencies.append(case.latency_ms)
        row_precision, row_recall, row_f1 = _precision_recall_f1(
            row_true_positive,
            row_predicted,
            row_expected,
        )
        metrics = {
            "query_result_exact_match": exact_matches / len(suite.cases),
            "row_precision": row_precision,
            "row_recall": row_recall,
            "row_f1": row_f1,
            "aggregation_accuracy": (
                correct_aggregations / aggregation_judgments
                if aggregation_judgments
                else 0.0
            ),
            **_latency_metrics(latencies, prefix="latency"),
        }
        return _executed_report(
            intended_use="sql_like",
            metrics=metrics,
            cases=len(suite.cases),
            minimums={
                "query_result_exact_match": suite.min_query_result_exact_match,
                "aggregation_accuracy": suite.min_aggregation_accuracy,
            },
            maximums={"latency_p95_ms": suite.max_p95_latency_ms},
        )

    def evaluate_nosql(
        self,
        suite: NoSQLEvaluationInput | None,
    ) -> IntendedUseMetricReport:
        if suite is None or not suite.cases:
            return _not_executed(
                "nosql",
                "requires traversal cases with expected nodes, paths, and latency",
            )
        if not any(case.expected_paths for case in suite.cases):
            return _not_executed(
                "nosql",
                "requires at least one expected traversal path",
            )
        node_true_positive = 0
        node_predicted = 0
        node_expected = 0
        exact_matches = 0
        path_true_positive = 0
        path_predicted = 0
        path_expected = 0
        latencies: list[float] = []
        for case in suite.cases:
            predicted_nodes = set(case.visited_node_ids)
            expected_nodes = set(case.expected_node_ids)
            exact_matches += int(predicted_nodes == expected_nodes)
            node_true_positive += len(predicted_nodes & expected_nodes)
            node_predicted += len(predicted_nodes)
            node_expected += len(expected_nodes)
            predicted_paths = {_canonical_path(path) for path in case.returned_paths}
            expected_paths = {_canonical_path(path) for path in case.expected_paths}
            path_true_positive += len(predicted_paths & expected_paths)
            path_predicted += len(predicted_paths)
            path_expected += len(expected_paths)
            latencies.append(case.latency_ms)
        node_precision, node_coverage, node_f1 = _precision_recall_f1(
            node_true_positive,
            node_predicted,
            node_expected,
        )
        path_precision, path_recall, path_f1 = _precision_recall_f1(
            path_true_positive,
            path_predicted,
            path_expected,
        )
        metrics = {
            "traversal_result_exact_match": exact_matches / len(suite.cases),
            "traversal_precision": node_precision,
            "traversal_coverage": node_coverage,
            "traversal_f1": node_f1,
            "path_precision": path_precision,
            "path_recall": path_recall,
            "path_f1": path_f1,
            **_latency_metrics(latencies, prefix="latency"),
        }
        return _executed_report(
            intended_use="nosql",
            metrics=metrics,
            cases=len(suite.cases),
            minimums={
                "traversal_result_exact_match": (
                    suite.min_traversal_result_exact_match
                ),
                "traversal_coverage": suite.min_traversal_coverage,
                "path_f1": suite.min_path_f1,
            },
            maximums={"latency_p95_ms": suite.max_p95_latency_ms},
        )

    def evaluate_graphrag(
        self,
        suite: GraphRAGEvaluationInput | None,
    ) -> IntendedUseMetricReport:
        if suite is None or not suite.cases:
            return _not_executed(
                "graphrag",
                "requires GraphRAG relevance, path, answer, and citation judgments",
            )
        if any(not _positive_relevance(case.context_relevance) for case in suite.cases):
            return _not_executed(
                "graphrag",
                "every GraphRAG case requires at least one positively relevant context",
            )
        if not any(case.expected_paths for case in suite.cases):
            return _not_executed(
                "graphrag",
                "requires at least one gold reasoning path",
            )
        multi_hop = [case for case in suite.cases if case.requires_multi_hop]
        if not multi_hop:
            return _not_executed(
                "graphrag",
                "requires at least one multi-hop answer judgment",
            )
        if not any(case.expected_citation_ids for case in suite.cases):
            return _not_executed(
                "graphrag",
                "requires at least one gold citation",
            )
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        path_true_positive = 0
        path_predicted = 0
        path_expected = 0
        citation_true_positive = 0
        citation_predicted = 0
        citation_expected = 0
        for case in suite.cases:
            ranking = _unique(case.retrieved_context_ids)[: suite.k]
            relevant = _positive_relevance(case.context_relevance)
            recalls.append(len(set(ranking) & set(relevant)) / len(relevant))
            reciprocal_ranks.append(_reciprocal_rank(ranking, relevant))
            ndcgs.append(_ndcg_at_k(ranking, case.context_relevance, suite.k))
            predicted_paths = {_canonical_path(path) for path in case.retrieved_paths}
            expected_paths = {_canonical_path(path) for path in case.expected_paths}
            path_true_positive += len(predicted_paths & expected_paths)
            path_predicted += len(predicted_paths)
            path_expected += len(expected_paths)
            predicted_citations = set(case.predicted_citation_ids)
            expected_citations = set(case.expected_citation_ids)
            citation_true_positive += len(predicted_citations & expected_citations)
            citation_predicted += len(predicted_citations)
            citation_expected += len(expected_citations)
        path_precision, path_recall, path_f1 = _precision_recall_f1(
            path_true_positive,
            path_predicted,
            path_expected,
        )
        citation_precision, citation_recall, citation_f1 = _precision_recall_f1(
            citation_true_positive,
            citation_predicted,
            citation_expected,
        )
        metrics = {
            "context_recall_at_k": _mean(recalls),
            "mrr": _mean(reciprocal_ranks),
            "ndcg_at_k": _mean(ndcgs),
            "path_precision": path_precision,
            "path_recall": path_recall,
            "path_f1": path_f1,
            "multi_hop_success_rate": _mean(
                [
                    float(case.multi_hop_answer_correct)
                    for case in multi_hop
                    if case.multi_hop_answer_correct is not None
                ]
            ),
            "answer_faithfulness": _mean(
                [case.answer_faithfulness for case in suite.cases]
            ),
            "citation_precision": citation_precision,
            "citation_recall": citation_recall,
            "citation_f1": citation_f1,
        }
        return _executed_report(
            intended_use="graphrag",
            metrics=metrics,
            cases=len(suite.cases),
            minimums={
                "context_recall_at_k": suite.min_recall_at_k,
                "mrr": suite.min_mrr,
                "ndcg_at_k": suite.min_ndcg_at_k,
                "path_f1": suite.min_path_f1,
                "multi_hop_success_rate": suite.min_multi_hop_success_rate,
                "answer_faithfulness": suite.min_answer_faithfulness,
                "citation_f1": suite.min_citation_f1,
            },
        )

    def evaluate_vector(
        self,
        suite: VectorIndexEvaluationInput | None,
    ) -> IntendedUseMetricReport:
        if suite is None or not suite.cases:
            return _not_executed(
                "vector",
                "requires vector relevance/filter judgments and resource observations",
            )
        if any(not _positive_relevance(case.relevance) for case in suite.cases):
            return _not_executed(
                "vector",
                "every vector query requires at least one positively relevant item",
            )
        inconsistent = [
            case.case_id
            for case in suite.cases
            if not set(_positive_relevance(case.relevance)).issubset(
                case.filter_allowed_ids
            )
        ]
        if inconsistent:
            return _not_executed(
                "vector",
                "relevant items must satisfy each query filter; inconsistent cases: "
                + ", ".join(sorted(inconsistent)),
            )
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        filter_correct = 0
        filter_returned = 0
        filter_exact_queries = 0
        latencies: list[float] = []
        for case in suite.cases:
            ranking = _unique(case.retrieved_ids)[: suite.k]
            relevant = _positive_relevance(case.relevance)
            allowed = set(case.filter_allowed_ids)
            recalls.append(len(set(ranking) & set(relevant)) / len(relevant))
            reciprocal_ranks.append(_reciprocal_rank(ranking, relevant))
            ndcgs.append(_ndcg_at_k(ranking, case.relevance, suite.k))
            filter_correct += sum(item_id in allowed for item_id in ranking)
            filter_returned += len(ranking)
            filter_exact_queries += int(all(item_id in allowed for item_id in ranking))
            latencies.append(case.latency_ms)
        filter_correctness = (
            filter_correct / filter_returned if filter_returned else 1.0
        )
        metrics = {
            "index_coverage": suite.indexed_item_count / suite.eligible_item_count,
            "recall_at_k": _mean(recalls),
            "mrr": _mean(reciprocal_ranks),
            "ndcg_at_k": _mean(ndcgs),
            "filter_correctness": filter_correctness,
            "filter_exact_query_rate": filter_exact_queries / len(suite.cases),
            "peak_memory_bytes": float(suite.peak_memory_bytes),
            **_latency_metrics(latencies, prefix="latency"),
        }
        return _executed_report(
            intended_use="vector",
            metrics=metrics,
            cases=len(suite.cases),
            minimums={
                "index_coverage": suite.min_index_coverage,
                "recall_at_k": suite.min_recall_at_k,
                "mrr": suite.min_mrr,
                "ndcg_at_k": suite.min_ndcg_at_k,
                "filter_correctness": suite.min_filter_correctness,
            },
            maximums={
                "latency_p95_ms": suite.max_p95_latency_ms,
                "peak_memory_bytes": float(suite.max_peak_memory_bytes),
            },
        )

    def evaluate_link_prediction(
        self,
        suite: LinkPredictionEvaluationInput | None,
    ) -> IntendedUseMetricReport:
        if suite is None or not suite.cases:
            return _not_executed(
                "link_prediction",
                "requires filtered rankings, known truths, and popularity strata",
            )
        present_strata = {case.popularity_stratum for case in suite.cases}
        missing_strata = {"head", "torso", "tail"} - present_strata
        if missing_strata:
            return _not_executed(
                "link_prediction",
                "requires head, torso, and tail popularity judgments; missing "
                + ", ".join(sorted(missing_strata)),
            )
        reciprocal_ranks: list[float] = []
        hits: dict[int, list[float]] = {cutoff: [] for cutoff in suite.hits_at}
        by_stratum: dict[PopularityStratum, list[int | None]] = {
            "head": [],
            "torso": [],
            "tail": [],
        }
        for case in suite.cases:
            filtered = _filtered_ranking(
                case.ranked_entity_ids,
                true_entity_id=case.true_entity_id,
                known_true_entity_ids=case.known_true_entity_ids,
            )
            rank = _rank(filtered, case.true_entity_id)
            by_stratum[case.popularity_stratum].append(rank)
            reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
            for cutoff in suite.hits_at:
                hits[cutoff].append(float(rank is not None and rank <= cutoff))
        metrics = {"filtered_mrr": _mean(reciprocal_ranks)}
        metrics.update(
            {
                f"filtered_hits_at_{cutoff}": _mean(values)
                for cutoff, values in hits.items()
            }
        )
        strata = {
            stratum: PopularityStratumReport(
                case_count=len(ranks),
                mrr=_mean(
                    [1.0 / rank if rank is not None else 0.0 for rank in ranks]
                ),
                hits_at_k={
                    cutoff: _mean(
                        [
                            float(rank is not None and rank <= cutoff)
                            for rank in ranks
                        ]
                    )
                    for cutoff in suite.hits_at
                },
            )
            for stratum, ranks in by_stratum.items()
        }
        report = _executed_report(
            intended_use="link_prediction",
            metrics=metrics,
            cases=len(suite.cases),
            minimums={
                "filtered_mrr": suite.min_filtered_mrr,
                **{
                    f"filtered_hits_at_{cutoff}": threshold
                    for cutoff, threshold in suite.min_filtered_hits_at_k.items()
                },
            },
        )
        return report.model_copy(update={"popularity_strata": strata})

    def evaluate_recommendation(
        self,
        suite: RecommendationEvaluationInput | None,
    ) -> IntendedUseMetricReport:
        if suite is None or not suite.cases:
            return _not_executed(
                "recommendation",
                "requires recommendation relevance judgments and catalog metadata",
            )
        catalog = set(suite.catalog_item_ids)
        if not catalog or len(catalog) != len(suite.catalog_item_ids):
            return _not_executed(
                "recommendation",
                "requires a non-empty catalog with unique item identifiers",
            )
        if any(not case.relevant_item_ids for case in suite.cases):
            return _not_executed(
                "recommendation",
                "every recommendation case requires at least one relevant item",
            )
        ranked_items = {
            item_id
            for case in suite.cases
            for item_id in _unique(case.recommended_item_ids)[: suite.k]
        }
        unknown = ranked_items - catalog
        missing_popularity = ranked_items - set(suite.item_popularity_probability)
        missing_features = ranked_items - set(suite.item_features)
        if unknown or missing_popularity or missing_features:
            problems: list[str] = []
            if unknown:
                problems.append(f"items outside catalog={sorted(unknown)}")
            if missing_popularity:
                problems.append(
                    f"missing popularity={sorted(missing_popularity)}"
                )
            if missing_features:
                problems.append(f"missing features={sorted(missing_features)}")
            return _not_executed(
                "recommendation",
                "incomplete recommendation metadata: " + "; ".join(problems),
            )
        precisions: list[float] = []
        recalls: list[float] = []
        ndcgs: list[float] = []
        novelties: list[float] = []
        diversities: list[float] = []
        recommended_catalog_items: set[str] = set()
        for case in suite.cases:
            ranking = _unique(case.recommended_item_ids)[: suite.k]
            relevant = set(case.relevant_item_ids)
            true_positive = len(set(ranking) & relevant)
            precisions.append(true_positive / suite.k)
            recalls.append(true_positive / len(relevant))
            binary_relevance = {item_id: 1.0 for item_id in relevant}
            ndcgs.append(_ndcg_at_k(ranking, binary_relevance, suite.k))
            recommended_catalog_items.update(ranking)
            novelties.extend(
                -math.log2(
                    max(
                        suite.item_popularity_probability[item_id],
                        suite.novelty_probability_floor,
                    )
                )
                for item_id in ranking
            )
            diversities.append(
                _intra_list_diversity(ranking, suite.item_features)
            )
        metrics = {
            "precision_at_k": _mean(precisions),
            "recall_at_k": _mean(recalls),
            "ndcg_at_k": _mean(ndcgs),
            "catalog_coverage": len(recommended_catalog_items) / len(catalog),
            "novelty": _mean(novelties),
            "intra_list_diversity": _mean(diversities),
        }
        return _executed_report(
            intended_use="recommendation",
            metrics=metrics,
            cases=len(suite.cases),
            minimums={
                "precision_at_k": suite.min_precision_at_k,
                "recall_at_k": suite.min_recall_at_k,
                "ndcg_at_k": suite.min_ndcg_at_k,
                "catalog_coverage": suite.min_catalog_coverage,
                "novelty": suite.min_novelty,
                "intra_list_diversity": suite.min_intra_list_diversity,
            },
        )

    def evaluate_reasoning(
        self,
        suite: ReasoningEvaluationInput | None,
    ) -> IntendedUseMetricReport:
        if suite is None or not suite.cases:
            return _not_executed(
                "reasoning",
                "requires competency-question and inferred-fact judgments plus runtime",
            )
        if any(not case.expected_answers for case in suite.cases):
            return _not_executed(
                "reasoning",
                "every competency question requires at least one expected answer",
            )
        if not any(case.expected_inferred_facts for case in suite.cases):
            return _not_executed(
                "reasoning",
                "requires at least one gold inferred fact",
            )
        exact_answers = 0
        answer_true_positive = 0
        answer_predicted = 0
        answer_expected = 0
        fact_true_positive = 0
        fact_predicted = 0
        fact_expected = 0
        runtimes: list[float] = []
        for case in suite.cases:
            predicted_answers = set(case.predicted_answers)
            expected_answers = set(case.expected_answers)
            exact_answers += int(predicted_answers == expected_answers)
            answer_true_positive += len(predicted_answers & expected_answers)
            answer_predicted += len(predicted_answers)
            answer_expected += len(expected_answers)
            predicted_facts = {_fact_key(fact) for fact in case.predicted_inferred_facts}
            expected_facts = {_fact_key(fact) for fact in case.expected_inferred_facts}
            fact_true_positive += len(predicted_facts & expected_facts)
            fact_predicted += len(predicted_facts)
            fact_expected += len(expected_facts)
            runtimes.append(case.runtime_ms)
        answer_precision, answer_recall, answer_f1 = _precision_recall_f1(
            answer_true_positive,
            answer_predicted,
            answer_expected,
        )
        fact_precision, fact_recall, fact_f1 = _precision_recall_f1(
            fact_true_positive,
            fact_predicted,
            fact_expected,
        )
        metrics = {
            "competency_question_accuracy": exact_answers / len(suite.cases),
            "answer_precision": answer_precision,
            "answer_recall": answer_recall,
            "answer_f1": answer_f1,
            "inferred_fact_precision": fact_precision,
            "inferred_fact_recall": fact_recall,
            "inferred_fact_f1": fact_f1,
            **_latency_metrics(runtimes, prefix="runtime"),
        }
        return _executed_report(
            intended_use="reasoning",
            metrics=metrics,
            cases=len(suite.cases),
            minimums={
                "competency_question_accuracy": (
                    suite.min_competency_question_accuracy
                ),
                "inferred_fact_f1": suite.min_inferred_fact_f1,
            },
            maximums={"runtime_p95_ms": suite.max_p95_runtime_ms},
        )


def _not_executed(
    intended_use: IntendedUse,
    reason: str,
) -> IntendedUseMetricReport:
    return IntendedUseMetricReport(
        intended_use=intended_use,
        executed=False,
        passed=False,
        reason=reason,
    )


def _executed_report(
    *,
    intended_use: IntendedUse,
    metrics: dict[str, float],
    cases: int,
    minimums: Mapping[str, float] | None = None,
    maximums: Mapping[str, float] | None = None,
) -> IntendedUseMetricReport:
    thresholds: list[MetricThresholdResult] = []
    for metric, threshold in (minimums or {}).items():
        actual = metrics[metric]
        thresholds.append(
            MetricThresholdResult(
                metric=metric,
                operator=">=",
                threshold=threshold,
                actual=actual,
                passed=actual >= threshold,
            )
        )
    for metric, threshold in (maximums or {}).items():
        actual = metrics[metric]
        thresholds.append(
            MetricThresholdResult(
                metric=metric,
                operator="<=",
                threshold=threshold,
                actual=actual,
                passed=actual <= threshold,
            )
        )
    failed = [check.metric for check in thresholds if not check.passed]
    reason = (
        f"{cases} judged cases evaluated"
        if not failed
        else "failed thresholds: " + ", ".join(failed)
    )
    return IntendedUseMetricReport(
        intended_use=intended_use,
        executed=True,
        passed=not failed,
        reason=reason,
        evaluated_cases=cases,
        metrics=metrics,
        thresholds=thresholds,
    )


def _canonical_row(row: Mapping[str, ScalarValue]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_path(path: Sequence[str]) -> tuple[str, ...]:
    return tuple(path)


def _scalar_equal(
    predicted: ScalarValue,
    expected: ScalarValue,
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> bool:
    if (
        isinstance(predicted, (int, float))
        and not isinstance(predicted, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        return math.isclose(
            float(predicted),
            float(expected),
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
    return type(predicted) is type(expected) and predicted == expected


def _precision_recall_f1(
    true_positive: int,
    predicted: int,
    expected: int,
) -> tuple[float, float, float]:
    if not predicted and not expected:
        return 0.0, 0.0, 0.0
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected if expected else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def _latency_metrics(values: Sequence[float], *, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean_ms": _mean(values),
        f"{prefix}_p50_ms": _percentile(values, 0.50),
        f"{prefix}_p95_ms": _percentile(values, 0.95),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    """Return the observed nearest-rank percentile (no interpolation)."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _validate_relevance(
    relevance: Mapping[str, float],
    *,
    label: str,
) -> None:
    invalid = {
        identifier: grade
        for identifier, grade in relevance.items()
        if not math.isfinite(grade) or grade < 0.0
    }
    if invalid:
        raise ValueError(f"{label} grades must be finite and non-negative: {invalid}")


def _positive_relevance(relevance: Mapping[str, float]) -> dict[str, float]:
    return {
        identifier: grade
        for identifier, grade in relevance.items()
        if grade > 0.0
    }


def _reciprocal_rank(
    ranking: Sequence[str],
    relevant: Mapping[str, float],
) -> float:
    for index, identifier in enumerate(ranking, start=1):
        if relevant.get(identifier, 0.0) > 0.0:
            return 1.0 / index
    return 0.0


def _ndcg_at_k(
    ranking: Sequence[str],
    relevance: Mapping[str, float],
    k: int,
) -> float:
    dcg = sum(
        (2.0 ** relevance.get(identifier, 0.0) - 1.0) / math.log2(index + 1.0)
        for index, identifier in enumerate(ranking[:k], start=1)
    )
    ideal_grades = sorted(
        (grade for grade in relevance.values() if grade > 0.0),
        reverse=True,
    )[:k]
    ideal_dcg = sum(
        (2.0**grade - 1.0) / math.log2(index + 1.0)
        for index, grade in enumerate(ideal_grades, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _filtered_ranking(
    ranking: Sequence[str],
    *,
    true_entity_id: str,
    known_true_entity_ids: Sequence[str],
) -> list[str]:
    other_truths = set(known_true_entity_ids) - {true_entity_id}
    return [
        entity_id
        for entity_id in _unique(ranking)
        if entity_id not in other_truths
    ]


def _rank(ranking: Sequence[str], expected: str) -> int | None:
    try:
        return ranking.index(expected) + 1
    except ValueError:
        return None


def _intra_list_diversity(
    ranking: Sequence[str],
    item_features: Mapping[str, Sequence[str]],
) -> float:
    if len(ranking) < 2:
        return 0.0
    distances: list[float] = []
    for left_index, left in enumerate(ranking):
        left_features = set(item_features[left])
        for right in ranking[left_index + 1 :]:
            right_features = set(item_features[right])
            union = left_features | right_features
            similarity = (
                len(left_features & right_features) / len(union)
                if union
                else 1.0
            )
            distances.append(1.0 - similarity)
    return _mean(distances)


def _fact_key(fact: ReasoningFact) -> tuple[str, str, str]:
    return fact.subject, fact.predicate, fact.object


__all__ = [
    "GraphRAGCase",
    "GraphRAGEvaluationInput",
    "IntendedUseEvaluationInputs",
    "IntendedUseEvaluationReport",
    "IntendedUseMetricReport",
    "IntendedUseMetricsEvaluator",
    "LinkPredictionCase",
    "LinkPredictionEvaluationInput",
    "MetricThresholdResult",
    "NoSQLEvaluationInput",
    "NoSQLTraversalCase",
    "PopularityStratumReport",
    "ReasoningCase",
    "ReasoningEvaluationInput",
    "ReasoningFact",
    "RecommendationCase",
    "RecommendationEvaluationInput",
    "SQLLikeEvaluationInput",
    "SQLLikeQueryCase",
    "VectorIndexCase",
    "VectorIndexEvaluationInput",
]
