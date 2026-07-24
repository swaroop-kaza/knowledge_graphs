"""Versioned, graph-bound intended-use evaluation suites.

The public construction path accepts gold judgments and declarative workload
queries, never caller-supplied predictions.  This module executes those queries
against one frozen :class:`PropertyGraph`, records observed latency, and then
feeds the observations to the metric implementations in
``intended_use_metrics``.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from kg_construction.domain.graph_objects import (
    PropertyGraph,
    PropertyNode,
)
from kg_construction.domain.models import ConstructionBatch, canonical_json
from kg_construction.infrastructure.evaluation.intended_use_metrics import (
    GraphRAGCase,
    GraphRAGEvaluationInput,
    IntendedUseEvaluationInputs,
    IntendedUseEvaluationReport,
    IntendedUseMetricsEvaluator,
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

GRAPH_EVALUATION_SUITE_VERSION = "kg-graph-evaluation-suite-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"[\w.-]+", re.UNICODE)


class _SuiteContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphNodeSelector(_SuiteContract):
    """Deterministic node predicate used by workload queries."""

    node_ids: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    properties: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_constraint(self) -> GraphNodeSelector:
        if not (self.node_ids or self.labels or self.properties):
            raise ValueError("a graph node selector requires at least one constraint")
        return self


class SQLNodeQuery(_SuiteContract):
    selector: GraphNodeSelector
    projections: dict[str, str]

    @model_validator(mode="after")
    def _has_projection(self) -> SQLNodeQuery:
        if not self.projections:
            raise ValueError("SQL-like node queries require at least one projection")
        return self


class GraphTraversalQuery(_SuiteContract):
    start: GraphNodeSelector
    relationship_types: tuple[str, ...]
    max_depth: int = Field(default=1, ge=1, le=16)
    direction: Literal["outgoing", "incoming", "both"] = "outgoing"

    @model_validator(mode="after")
    def _has_relationship_type(self) -> GraphTraversalQuery:
        if not self.relationship_types:
            raise ValueError("a traversal query requires relationship types")
        return self


class PathQuery(_SuiteContract):
    start: GraphNodeSelector
    end: GraphNodeSelector
    relationship_types: tuple[str, ...]
    max_depth: int = Field(default=4, ge=1, le=16)
    direction: Literal["outgoing", "incoming", "both"] = "outgoing"


class LexicalRetrievalQuery(_SuiteContract):
    query_text: str = Field(min_length=1)
    candidates: GraphNodeSelector
    k: int = Field(default=10, ge=1)
    path_query: PathQuery | None = None
    answer_query: PathQuery


class VectorSearchQuery(_SuiteContract):
    query_text: str = Field(min_length=1)
    candidates: GraphNodeSelector
    allowed: GraphNodeSelector
    k: int = Field(default=10, ge=1)
    dimensions: int = Field(default=64, ge=8, le=4096)


class NeighborRankingQuery(_SuiteContract):
    source: GraphNodeSelector
    relationship_type: str = Field(min_length=1)
    candidates: GraphNodeSelector
    direction: Literal["outgoing", "incoming"] = "outgoing"
    k: int = Field(default=10, ge=1)


class ReasoningPathQuery(_SuiteContract):
    start: GraphNodeSelector
    relationship_type_sequence: tuple[str, ...]
    inferred_predicate: str = Field(min_length=1)

    @model_validator(mode="after")
    def _has_steps(self) -> ReasoningPathQuery:
        if not self.relationship_type_sequence:
            raise ValueError("reasoning queries require at least one relationship step")
        return self


class BoundSQLLikeCase(_SuiteContract):
    case_id: str = Field(min_length=1)
    query: SQLNodeQuery
    expected_rows: list[dict[str, str | int | float | bool | None]]
    expected_aggregations: dict[str, str | int | float | bool | None]
    ordered_results: bool = False

    @model_validator(mode="after")
    def _has_gold(self) -> BoundSQLLikeCase:
        if not self.expected_rows or not self.expected_aggregations:
            raise ValueError("SQL-like cases require non-empty row and aggregation gold")
        return self


class BoundSQLLikeSuite(_SuiteContract):
    cases: tuple[BoundSQLLikeCase, ...]
    max_p95_latency_ms: float = Field(gt=0.0)
    min_query_result_exact_match: float = Field(default=1.0, ge=0.0, le=1.0)
    min_aggregation_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)


class BoundNoSQLCase(_SuiteContract):
    case_id: str = Field(min_length=1)
    query: GraphTraversalQuery
    expected_node_ids: list[str]
    expected_paths: list[list[str]]

    @model_validator(mode="after")
    def _has_gold(self) -> BoundNoSQLCase:
        if not self.expected_node_ids or not self.expected_paths:
            raise ValueError("NoSQL cases require non-empty node and path gold")
        return self


class BoundNoSQLSuite(_SuiteContract):
    cases: tuple[BoundNoSQLCase, ...]
    max_p95_latency_ms: float = Field(gt=0.0)
    min_traversal_result_exact_match: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    min_traversal_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_path_f1: float = Field(default=1.0, ge=0.0, le=1.0)


class BoundGraphRAGCase(_SuiteContract):
    case_id: str = Field(min_length=1)
    query: LexicalRetrievalQuery
    context_relevance: dict[str, float]
    expected_paths: list[list[str]]
    expected_answer_node_ids: list[str]
    expected_citation_ids: list[str]
    requires_multi_hop: bool

    @model_validator(mode="after")
    def _has_gold(self) -> BoundGraphRAGCase:
        if (
            not any(value > 0.0 for value in self.context_relevance.values())
            or not self.expected_paths
            or not self.expected_answer_node_ids
            or not self.expected_citation_ids
        ):
            raise ValueError(
                "GraphRAG cases require relevance, path, answer, and citation gold"
            )
        if self.requires_multi_hop and self.query.path_query is None:
            raise ValueError("multi-hop GraphRAG cases require a path query")
        return self


class BoundGraphRAGSuite(_SuiteContract):
    cases: tuple[BoundGraphRAGCase, ...]
    min_recall_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    min_mrr: float = Field(default=0.60, ge=0.0, le=1.0)
    min_ndcg_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    min_path_f1: float = Field(default=0.70, ge=0.0, le=1.0)
    min_multi_hop_success_rate: float = Field(default=0.70, ge=0.0, le=1.0)
    min_answer_faithfulness: float = Field(default=0.80, ge=0.0, le=1.0)
    min_citation_f1: float = Field(default=0.80, ge=0.0, le=1.0)


class BoundVectorCase(_SuiteContract):
    case_id: str = Field(min_length=1)
    query: VectorSearchQuery
    relevance: dict[str, float]

    @model_validator(mode="after")
    def _has_gold(self) -> BoundVectorCase:
        if not any(value > 0.0 for value in self.relevance.values()):
            raise ValueError("vector cases require positive relevance gold")
        return self


class BoundVectorSuite(_SuiteContract):
    cases: tuple[BoundVectorCase, ...]
    max_p95_latency_ms: float = Field(gt=0.0)
    max_peak_memory_bytes: int = Field(gt=0)
    min_index_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_recall_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    min_mrr: float = Field(default=0.60, ge=0.0, le=1.0)
    min_ndcg_at_k: float = Field(default=0.70, ge=0.0, le=1.0)
    min_filter_correctness: float = Field(default=1.0, ge=0.0, le=1.0)


class BoundLinkPredictionCase(_SuiteContract):
    case_id: str = Field(min_length=1)
    query: NeighborRankingQuery
    true_entity_id: str = Field(min_length=1)


class BoundLinkPredictionSuite(_SuiteContract):
    cases: tuple[BoundLinkPredictionCase, ...]
    hits_at: list[int] = Field(default_factory=lambda: [1, 3, 10])
    min_filtered_mrr: float = Field(default=0.20, ge=0.0, le=1.0)
    min_filtered_hits_at_k: dict[int, float] = Field(
        default_factory=lambda: {1: 0.10, 3: 0.20, 10: 0.30}
    )

    @model_validator(mode="after")
    def _has_statistical_support(self) -> BoundLinkPredictionSuite:
        if len(self.cases) < 3:
            raise ValueError(
                "link prediction requires at least three held-out cases"
            )
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("link-prediction case identifiers must be unique")
        return self


class BoundRecommendationCase(_SuiteContract):
    case_id: str = Field(min_length=1)
    query: NeighborRankingQuery
    relevant_item_ids: list[str]

    @model_validator(mode="after")
    def _has_gold(self) -> BoundRecommendationCase:
        if not self.relevant_item_ids:
            raise ValueError("recommendation cases require relevance gold")
        return self


class BoundRecommendationSuite(_SuiteContract):
    cases: tuple[BoundRecommendationCase, ...]
    catalog: GraphNodeSelector
    feature_properties: tuple[str, ...] = ()
    novelty_probability_floor: float = Field(default=1e-12, gt=0.0, le=1.0)
    min_precision_at_k: float = Field(default=0.01, ge=0.0, le=1.0)
    min_recall_at_k: float = Field(default=0.01, ge=0.0, le=1.0)
    min_ndcg_at_k: float = Field(default=0.01, ge=0.0, le=1.0)
    min_catalog_coverage: float = Field(default=0.01, ge=0.0, le=1.0)
    min_novelty: float = Field(default=0.01, ge=0.0)
    min_intra_list_diversity: float = Field(default=0.01, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _has_statistical_support(self) -> BoundRecommendationSuite:
        if len(self.cases) < 3:
            raise ValueError(
                "recommendation requires at least three held-out cases"
            )
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("recommendation case identifiers must be unique")
        if not self.feature_properties:
            raise ValueError(
                "recommendation requires item feature properties for diversity"
            )
        return self


class BoundReasoningCase(_SuiteContract):
    case_id: str = Field(min_length=1)
    query: ReasoningPathQuery
    expected_answers: list[str]
    expected_inferred_facts: list[ReasoningFact]

    @model_validator(mode="after")
    def _has_gold(self) -> BoundReasoningCase:
        if not self.expected_answers or not self.expected_inferred_facts:
            raise ValueError("reasoning cases require answer and inferred-fact gold")
        return self


class BoundReasoningSuite(_SuiteContract):
    cases: tuple[BoundReasoningCase, ...]
    max_p95_runtime_ms: float = Field(gt=0.0)
    min_competency_question_accuracy: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
    )
    min_inferred_fact_f1: float = Field(default=0.80, ge=0.0, le=1.0)


class StructuredFormationGold(_SuiteContract):
    """Curated structured-model obligations, independent of graph predictions."""

    record_label: str = Field(min_length=1)
    expected_record_count: int = Field(ge=1)
    expected_node_counts: dict[str, int]
    expected_relationship_counts: dict[str, int]
    required_properties: dict[str, tuple[str, ...]]
    forbidden_properties: dict[str, tuple[str, ...]]
    identity_keys: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def _is_meaningful(self) -> StructuredFormationGold:
        if not self.expected_node_counts or any(
            not label or count < 1
            for label, count in self.expected_node_counts.items()
        ):
            raise ValueError(
                "structured formation requires positive expected node counts"
            )
        if (
            self.expected_node_counts.get(self.record_label)
            != self.expected_record_count
        ):
            raise ValueError(
                "record-label node count must equal expected_record_count"
            )
        if not self.expected_relationship_counts or not any(
            count > 0 for count in self.expected_relationship_counts.values()
        ) or any(
            not relationship or count < 0
            for relationship, count in self.expected_relationship_counts.items()
        ):
            raise ValueError(
                "structured formation requires a substantive relationship gold set"
            )
        for name, obligations in (
            ("required_properties", self.required_properties),
            ("forbidden_properties", self.forbidden_properties),
            ("identity_keys", self.identity_keys),
        ):
            if not obligations or any(
                label not in self.expected_node_counts
                or not properties
                or len(properties) != len(set(properties))
                or any(not property_name for property_name in properties)
                for label, properties in obligations.items()
            ):
                raise ValueError(
                    f"{name} must define non-empty, unique obligations for "
                    "expected node labels"
                )
        return self


class GraphBoundQualityGold(_SuiteContract):
    """Externally curated quality gold; no predicted fields are accepted."""

    expected_batch: ConstructionBatch = Field(default_factory=ConstructionBatch)
    coreference_clusters: dict[str, str] = Field(default_factory=dict)
    entity_links: dict[str, str | None] = Field(default_factory=dict)
    structured_formation: StructuredFormationGold | None = None


class VersionedGraphEvaluationSuite(_SuiteContract):
    """All mandatory quality/use fixtures, sealed by a stable SHA-256."""

    version: Literal["kg-graph-evaluation-suite-v1"] = (
        "kg-graph-evaluation-suite-v1"
    )
    construction_profile: Literal["structured", "unstructured", "mixed"]
    fixture_id: str = Field(min_length=1)
    fixture_sha256: str
    quality: GraphBoundQualityGold
    sql_like: BoundSQLLikeSuite
    nosql: BoundNoSQLSuite
    graphrag: BoundGraphRAGSuite
    vector: BoundVectorSuite
    link_prediction: BoundLinkPredictionSuite
    recommendation: BoundRecommendationSuite
    reasoning: BoundReasoningSuite

    @model_validator(mode="after")
    def _hash_is_exact(
        self,
        info: ValidationInfo,
    ) -> VersionedGraphEvaluationSuite:
        if not _SHA256_RE.fullmatch(self.fixture_sha256):
            raise ValueError("fixture_sha256 must be a lowercase SHA-256 digest")
        if not (info.context or {}).get("skip_fixture_hash"):
            actual = evaluation_fixture_sha256(self)
            if actual != self.fixture_sha256:
                raise ValueError(
                    "fixture_sha256 mismatch: "
                    f"declared={self.fixture_sha256}, actual={actual}"
                )
        if any(
            not getattr(self, name).cases
            for name in (
                "sql_like",
                "nosql",
                "graphrag",
                "vector",
                "link_prediction",
                "recommendation",
                "reasoning",
            )
        ):
            raise ValueError("every intended-use family requires at least one case")
        if self.construction_profile in {"unstructured", "mixed"} and (
            not self.quality.expected_batch.entities
            or not self.quality.expected_batch.relations
            or not self.quality.coreference_clusters
            or not self.quality.entity_links
        ):
            raise ValueError(
                "unstructured/mixed suites require occurrence gold plus "
                "coreference and linking judgments"
            )
        if (
            self.construction_profile in {"structured", "mixed"}
            and self.quality.structured_formation is None
        ):
            raise ValueError(
                "structured/mixed suites require substantive structured-formation gold"
            )
        if (
            self.construction_profile == "unstructured"
            and self.quality.structured_formation is not None
        ):
            raise ValueError(
                "unstructured suites cannot claim structured-formation execution"
            )
        return self

    @classmethod
    def seal(cls, payload: Mapping[str, object]) -> VersionedGraphEvaluationSuite:
        """Validate an unhashed payload after inserting its canonical digest."""

        raw = dict(payload)
        raw.pop("fixture_sha256", None)
        normalized = cls.model_validate(
            {**raw, "fixture_sha256": "0" * 64},
            context={"skip_fixture_hash": True},
        )
        digest = evaluation_fixture_sha256(normalized)
        return cls.model_validate(
            {
                **normalized.model_dump(
                    mode="json",
                    exclude={"fixture_sha256"},
                ),
                "fixture_sha256": digest,
            }
        )

    @classmethod
    def from_path(cls, path: str | Path) -> VersionedGraphEvaluationSuite:
        suite_path = Path(path)
        payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("evaluation suite must contain a YAML/JSON object")
        return cls.model_validate(payload)


@dataclass(frozen=True)
class GraphBoundExecution:
    inputs: IntendedUseEvaluationInputs
    graph_sha256: str
    fixture_sha256: str
    query_hashes: dict[str, list[str]]
    observed_latencies_ms: dict[str, list[float]]
    predicted_answers: dict[str, list[str]]
    gold_answers: dict[str, list[str]]
    vector_rankings: dict[str, list[str]]
    vector_relevant: dict[str, list[str]]
    graph_backend_id: str
    graph_snapshot_id: str
    certifiable_backend: bool
    training_graph_sha256: str
    held_out_edge_ids: tuple[str, ...]
    derived_popularity_strata: dict[str, Literal["head", "torso", "tail"]]
    formation_gold_executed: bool
    formation_gold_passed: bool
    formation_gold_metrics: dict[str, float]
    formation_gold_assertions: dict[str, bool]


@dataclass(frozen=True)
class _NeighborProtocolCase:
    source_id: str
    candidate_ids: tuple[str, ...]
    known_true_entity_ids: tuple[str, ...]


@dataclass(frozen=True)
class VectorWorkloadObservation:
    ranking: list[str]
    allowed_ids: list[str]
    eligible_ids: list[str]
    peak_memory_bytes: int


@dataclass(frozen=True)
class GraphAnswerObservation:
    answer_node_ids: list[str]
    paths: list[list[str]]
    citation_ids: list[str]
    faithfulness: float


class VectorWorkloadBackend(Protocol):
    """Backend boundary for an actual vector index workload."""

    backend_id: str
    certifiable: bool

    def snapshot_id(self, graph_sha256: str) -> str: ...

    def query(
        self,
        graph: PropertyGraph,
        query: VectorSearchQuery,
    ) -> VectorWorkloadObservation: ...


class GraphAnswerBackend(Protocol):
    """Backend boundary for answer generation and provenance entailment."""

    backend_id: str

    def answer(
        self,
        graph: PropertyGraph,
        *,
        query: LexicalRetrievalQuery,
        retrieved_context_ids: Sequence[str],
    ) -> GraphAnswerObservation: ...


class DeterministicVectorWorkloadBackend:
    """In-memory vector backend for unit tests; never release-certifiable."""

    backend_id = "deterministic-hash-vector-unit-test"
    certifiable = False

    def snapshot_id(self, graph_sha256: str) -> str:
        return f"{self.backend_id}:{graph_sha256}"

    def query(
        self,
        graph: PropertyGraph,
        query: VectorSearchQuery,
    ) -> VectorWorkloadObservation:
        eligible = _select_nodes(graph, query.candidates)
        allowed_ids = {
            node.node_id for node in _select_nodes(graph, query.allowed)
        }
        query_vector = _text_vector(query.query_text, query.dimensions)
        scored = [
            (
                _cosine(
                    query_vector,
                    _text_vector(_node_search_text(node), query.dimensions),
                ),
                node.node_id,
            )
            for node in eligible
            if node.node_id in allowed_ids
        ]
        ranking = [
            node_id
            for _, node_id in sorted(scored, key=lambda item: (-item[0], item[1]))[
                : query.k
            ]
        ]
        return VectorWorkloadObservation(
            ranking=ranking,
            allowed_ids=sorted(allowed_ids),
            eligible_ids=[node.node_id for node in eligible],
            peak_memory_bytes=len(eligible) * query.dimensions * 8,
        )


class Neo4jVectorWorkloadBackend:
    """Live Neo4j ANN backend using injected locked-model embeddings."""

    certifiable = True

    def __init__(
        self,
        *,
        store: object,
        embedding_model: object,
        model_id: str,
        refresh_attempts: int = 40,
        refresh_interval_seconds: float = 0.25,
    ) -> None:
        if refresh_attempts < 1:
            raise ValueError("refresh_attempts must be at least 1")
        if refresh_interval_seconds < 0.0:
            raise ValueError("refresh_interval_seconds must be non-negative")
        self.store = store
        self.embedding_model = embedding_model
        self.model_id = model_id
        self.refresh_attempts = refresh_attempts
        self.refresh_interval_seconds = refresh_interval_seconds
        config = getattr(store, "config", None)
        index_name = getattr(config, "index_name", "unknown-index")
        self.backend_id = f"neo4j-vector:{index_name}:{model_id}"

    def snapshot_id(self, graph_sha256: str) -> str:
        return f"{self.backend_id}:{graph_sha256}"

    def query(
        self,
        graph: PropertyGraph,
        query: VectorSearchQuery,
    ) -> VectorWorkloadObservation:
        from kg_construction.infrastructure.graph_stores.vector_store import (
            VectorRecord,
        )

        eligible = _select_nodes(graph, query.candidates)
        allowed_ids = {
            node.node_id for node in _select_nodes(graph, query.allowed)
        }
        texts = [query.query_text, *(_node_search_text(node) for node in eligible)]
        encode = getattr(self.embedding_model, "encode", None)
        if not callable(encode):
            raise TypeError("the locked embedding backend must expose encode(texts)")
        vectors = encode(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("the embedding backend returned the wrong vector count")
        graph_hash = property_graph_sha256(graph)
        records = [
            VectorRecord(
                vector_id=node.node_id,
                embedding=vectors[index + 1],
                properties={
                    "evaluation_graph_sha256": graph_hash,
                    "evaluation_filter_allowed": node.node_id in allowed_ids,
                },
            )
            for index, node in enumerate(eligible)
        ]
        upsert = getattr(self.store, "upsert", None)
        search = getattr(self.store, "query", None)
        if not callable(upsert) or not callable(search):
            raise TypeError("the vector store must expose upsert() and query()")
        upsert(records)
        expected_hit_count = min(
            query.k,
            sum(node.node_id in allowed_ids for node in eligible),
        )
        hits = []
        for attempt in range(self.refresh_attempts):
            hits = search(
                vectors[0],
                top_k=query.k,
                filters={
                    "evaluation_graph_sha256": graph_hash,
                    "evaluation_filter_allowed": True,
                },
            )
            if len(hits) >= expected_hit_count:
                break
            if (
                attempt + 1 < self.refresh_attempts
                and self.refresh_interval_seconds
            ):
                time.sleep(self.refresh_interval_seconds)
        if len(hits) < expected_hit_count:
            raise RuntimeError(
                "Neo4j vector index did not expose the complete frozen graph "
                f"snapshot after {self.refresh_attempts} attempts: "
                f"expected={expected_hit_count}, observed={len(hits)}, "
                f"graph_sha256={graph_hash}"
            )
        ranking = [str(hit.vector_id) for hit in hits]
        dimensions = len(vectors[0])
        return VectorWorkloadObservation(
            ranking=ranking,
            allowed_ids=sorted(allowed_ids),
            eligible_ids=[node.node_id for node in eligible],
            peak_memory_bytes=len(eligible) * dimensions * 8,
        )


class EvidenceGroundedPathAnswerBackend:
    """Answer a graph query by path entailment and exact source provenance."""

    backend_id = "property-graph-path-entailment-v1"

    def answer(
        self,
        graph: PropertyGraph,
        *,
        query: LexicalRetrievalQuery,
        retrieved_context_ids: Sequence[str],
    ) -> GraphAnswerObservation:
        path_query = query.answer_query
        starts = [node.node_id for node in _select_nodes(graph, path_query.start)]
        ends = {
            node.node_id for node in _select_nodes(graph, path_query.end)
        }
        _, candidates = _traverse(
            graph,
            starts,
            relationship_types=set(path_query.relationship_types),
            max_depth=path_query.max_depth,
            direction=path_query.direction,
        )
        paths = [path for path in candidates if path[-1] in ends]
        answers = sorted({path[-1] for path in paths})
        node_by_id = {node.node_id: node for node in graph.nodes}
        answer_sources = {
            answer: {
                str(source_id)
                for source_id in _as_list(
                    node_by_id[answer].properties.get("source_ids", [])
                )
            }
            for answer in answers
        }
        citations = sorted(
            {
                source_id
                for sources in answer_sources.values()
                for source_id in sources
            }
        )
        retrieved = set(retrieved_context_ids)
        supported = sum(
            answer in retrieved and bool(answer_sources[answer])
            for answer in answers
        )
        faithfulness = supported / len(answers) if answers else 0.0
        return GraphAnswerObservation(
            answer_node_ids=answers,
            paths=paths,
            citation_ids=citations,
            faithfulness=faithfulness,
        )


class GraphWorkloadRunner:
    """Execute all seven workload families against one immutable graph snapshot."""

    def __init__(
        self,
        *,
        vector_backend: VectorWorkloadBackend | None = None,
        answer_backend: GraphAnswerBackend | None = None,
    ) -> None:
        self.vector_backend = (
            vector_backend or DeterministicVectorWorkloadBackend()
        )
        self.answer_backend = answer_backend or EvidenceGroundedPathAnswerBackend()

    def run(
        self,
        graph: PropertyGraph,
        suite: VersionedGraphEvaluationSuite,
    ) -> GraphBoundExecution:
        frozen = graph.model_copy(deep=True)
        before = property_graph_sha256(frozen)
        (
            training_graph,
            held_out_edge_ids,
            link_protocol,
            recommendation_protocol,
            derived_popularity_strata,
        ) = self._prepare_held_out_protocol(frozen, suite)
        training_graph_sha256 = property_graph_sha256(training_graph)
        if training_graph_sha256 == before:
            raise ValueError(
                "held-out evaluation must remove at least one gold edge "
                "from the training graph"
            )
        (
            formation_gold_executed,
            formation_gold_passed,
            formation_gold_metrics,
            formation_gold_assertions,
        ) = _evaluate_structured_formation(
            frozen,
            suite.quality.structured_formation,
        )
        query_hashes: dict[str, list[str]] = defaultdict(list)
        latencies: dict[str, list[float]] = defaultdict(list)

        sql_cases: list[SQLLikeQueryCase] = []
        for sql_case in suite.sql_like.cases:
            (rows, aggregations), elapsed = self._timed(
                lambda sql_case=sql_case: self._run_sql(frozen, sql_case)
            )
            sql_cases.append(
                SQLLikeQueryCase(
                    case_id=sql_case.case_id,
                    predicted_rows=rows,
                    expected_rows=sql_case.expected_rows,
                    predicted_aggregations=aggregations,
                    expected_aggregations=sql_case.expected_aggregations,
                    ordered_results=sql_case.ordered_results,
                    latency_ms=elapsed,
                )
            )
            self._record(
                "sql_like",
                sql_case.query,
                elapsed,
                query_hashes,
                latencies,
            )

        nosql_cases: list[NoSQLTraversalCase] = []
        for nosql_case in suite.nosql.cases:
            (nodes, paths), elapsed = self._timed(
                lambda nosql_case=nosql_case: self._run_traversal(
                    frozen,
                    nosql_case.query,
                )
            )
            nosql_cases.append(
                NoSQLTraversalCase(
                    case_id=nosql_case.case_id,
                    visited_node_ids=nodes,
                    expected_node_ids=nosql_case.expected_node_ids,
                    returned_paths=paths,
                    expected_paths=nosql_case.expected_paths,
                    latency_ms=elapsed,
                )
            )
            self._record(
                "nosql",
                nosql_case.query,
                elapsed,
                query_hashes,
                latencies,
            )

        graphrag_cases: list[GraphRAGCase] = []
        for graphrag_case in suite.graphrag.cases:
            (contexts, answer), elapsed = self._timed(
                lambda graphrag_case=graphrag_case: self._run_graphrag(
                    frozen,
                    graphrag_case.query,
                )
            )
            expected_answers = set(graphrag_case.expected_answer_node_ids)
            graphrag_cases.append(
                GraphRAGCase(
                    case_id=graphrag_case.case_id,
                    retrieved_context_ids=contexts,
                    context_relevance=graphrag_case.context_relevance,
                    retrieved_paths=answer.paths,
                    expected_paths=graphrag_case.expected_paths,
                    requires_multi_hop=graphrag_case.requires_multi_hop,
                    multi_hop_answer_correct=(
                        set(answer.answer_node_ids) == expected_answers
                        if graphrag_case.requires_multi_hop
                        else None
                    ),
                    answer_faithfulness=answer.faithfulness,
                    predicted_citation_ids=answer.citation_ids,
                    expected_citation_ids=graphrag_case.expected_citation_ids,
                )
            )
            self._record(
                "graphrag",
                graphrag_case.query,
                elapsed,
                query_hashes,
                latencies,
            )

        vector_cases: list[VectorIndexCase] = []
        vector_rankings: dict[str, list[str]] = {}
        vector_relevant: dict[str, list[str]] = {}
        eligible_ids: set[str] = set()
        peak_memory = 0
        for vector_case in suite.vector.cases:
            observation, elapsed = self._timed(
                lambda vector_case=vector_case: self._run_vector(
                    frozen,
                    vector_case.query,
                )
            )
            eligible_ids.update(observation.eligible_ids)
            peak_memory = max(peak_memory, observation.peak_memory_bytes)
            vector_cases.append(
                VectorIndexCase(
                    case_id=vector_case.case_id,
                    retrieved_ids=observation.ranking,
                    relevance=vector_case.relevance,
                    filter_allowed_ids=observation.allowed_ids,
                    latency_ms=elapsed,
                )
            )
            vector_rankings[vector_case.case_id] = observation.ranking
            vector_relevant[vector_case.case_id] = [
                item_id
                for item_id, grade in vector_case.relevance.items()
                if grade > 0.0
            ]
            self._record(
                "vector",
                vector_case.query,
                elapsed,
                query_hashes,
                latencies,
            )

        link_cases: list[LinkPredictionCase] = []
        for link_case in suite.link_prediction.cases:
            protocol = link_protocol[link_case.case_id]
            ranking, elapsed = self._timed(
                lambda link_case=link_case: self._run_neighbor_ranking(
                    training_graph,
                    link_case.query,
                )
            )
            link_cases.append(
                LinkPredictionCase(
                    case_id=link_case.case_id,
                    ranked_entity_ids=ranking,
                    true_entity_id=link_case.true_entity_id,
                    known_true_entity_ids=list(
                        protocol.known_true_entity_ids
                    ),
                    popularity_stratum=derived_popularity_strata[
                        link_case.case_id
                    ],
                )
            )
            self._record(
                "link_prediction",
                link_case.query,
                elapsed,
                query_hashes,
                latencies,
            )

        recommendation_cases: list[RecommendationCase] = []
        for recommendation_case in suite.recommendation.cases:
            protocol = recommendation_protocol[recommendation_case.case_id]
            ranking, elapsed = self._timed(
                lambda recommendation_case=recommendation_case: (
                    self._run_neighbor_ranking(
                        training_graph,
                        recommendation_case.query,
                    )
                )
            )
            recommendation_cases.append(
                RecommendationCase(
                    case_id=recommendation_case.case_id,
                    user_id=protocol.source_id,
                    recommended_item_ids=ranking[
                        : recommendation_case.query.k
                    ],
                    relevant_item_ids=recommendation_case.relevant_item_ids,
                )
            )
            self._record(
                "recommendation",
                recommendation_case.query,
                elapsed,
                query_hashes,
                latencies,
            )
        catalog_nodes = _select_nodes(
            training_graph,
            suite.recommendation.catalog,
        )
        catalog_ids = [node.node_id for node in catalog_nodes]
        popularity = _catalog_popularity(training_graph, catalog_ids)
        features = {
            node.node_id: _node_features(
                node,
                suite.recommendation.feature_properties,
            )
            for node in catalog_nodes
        }

        reasoning_cases: list[ReasoningCase] = []
        predicted_answers: dict[str, list[str]] = {}
        gold_answers: dict[str, list[str]] = {}
        for reasoning_case in suite.reasoning.cases:
            (answers, facts), elapsed = self._timed(
                lambda reasoning_case=reasoning_case: self._run_reasoning(
                    frozen,
                    reasoning_case.query,
                )
            )
            reasoning_cases.append(
                ReasoningCase(
                    case_id=reasoning_case.case_id,
                    predicted_answers=answers,
                    expected_answers=reasoning_case.expected_answers,
                    predicted_inferred_facts=facts,
                    expected_inferred_facts=reasoning_case.expected_inferred_facts,
                    runtime_ms=elapsed,
                )
            )
            predicted_answers[reasoning_case.case_id] = answers
            gold_answers[reasoning_case.case_id] = (
                reasoning_case.expected_answers
            )
            self._record(
                "reasoning",
                reasoning_case.query,
                elapsed,
                query_hashes,
                latencies,
            )

        after = property_graph_sha256(frozen)
        if before != after:
            raise RuntimeError("workload evaluation mutated the frozen graph snapshot")

        inputs = IntendedUseEvaluationInputs(
            sql_like=SQLLikeEvaluationInput(
                cases=sql_cases,
                max_p95_latency_ms=suite.sql_like.max_p95_latency_ms,
                min_query_result_exact_match=(
                    suite.sql_like.min_query_result_exact_match
                ),
                min_aggregation_accuracy=suite.sql_like.min_aggregation_accuracy,
            ),
            nosql=NoSQLEvaluationInput(
                cases=nosql_cases,
                max_p95_latency_ms=suite.nosql.max_p95_latency_ms,
                min_traversal_result_exact_match=(
                    suite.nosql.min_traversal_result_exact_match
                ),
                min_traversal_coverage=suite.nosql.min_traversal_coverage,
                min_path_f1=suite.nosql.min_path_f1,
            ),
            graphrag=GraphRAGEvaluationInput(
                cases=graphrag_cases,
                k=max(case.query.k for case in suite.graphrag.cases),
                min_recall_at_k=suite.graphrag.min_recall_at_k,
                min_mrr=suite.graphrag.min_mrr,
                min_ndcg_at_k=suite.graphrag.min_ndcg_at_k,
                min_path_f1=suite.graphrag.min_path_f1,
                min_multi_hop_success_rate=(
                    suite.graphrag.min_multi_hop_success_rate
                ),
                min_answer_faithfulness=suite.graphrag.min_answer_faithfulness,
                min_citation_f1=suite.graphrag.min_citation_f1,
            ),
            vector=VectorIndexEvaluationInput(
                cases=vector_cases,
                eligible_item_count=len(eligible_ids),
                indexed_item_count=len(eligible_ids),
                peak_memory_bytes=peak_memory,
                k=max(case.query.k for case in suite.vector.cases),
                max_p95_latency_ms=suite.vector.max_p95_latency_ms,
                max_peak_memory_bytes=suite.vector.max_peak_memory_bytes,
                min_index_coverage=suite.vector.min_index_coverage,
                min_recall_at_k=suite.vector.min_recall_at_k,
                min_mrr=suite.vector.min_mrr,
                min_ndcg_at_k=suite.vector.min_ndcg_at_k,
                min_filter_correctness=suite.vector.min_filter_correctness,
            ),
            link_prediction=LinkPredictionEvaluationInput(
                cases=link_cases,
                hits_at=suite.link_prediction.hits_at,
                min_filtered_mrr=suite.link_prediction.min_filtered_mrr,
                min_filtered_hits_at_k=(
                    suite.link_prediction.min_filtered_hits_at_k
                ),
            ),
            recommendation=RecommendationEvaluationInput(
                cases=recommendation_cases,
                catalog_item_ids=catalog_ids,
                item_popularity_probability=popularity,
                item_features=features,
                k=max(case.query.k for case in suite.recommendation.cases),
                novelty_probability_floor=(
                    suite.recommendation.novelty_probability_floor
                ),
                min_precision_at_k=suite.recommendation.min_precision_at_k,
                min_recall_at_k=suite.recommendation.min_recall_at_k,
                min_ndcg_at_k=suite.recommendation.min_ndcg_at_k,
                min_catalog_coverage=suite.recommendation.min_catalog_coverage,
                min_novelty=suite.recommendation.min_novelty,
                min_intra_list_diversity=(
                    suite.recommendation.min_intra_list_diversity
                ),
            ),
            reasoning=ReasoningEvaluationInput(
                cases=reasoning_cases,
                max_p95_runtime_ms=suite.reasoning.max_p95_runtime_ms,
                min_competency_question_accuracy=(
                    suite.reasoning.min_competency_question_accuracy
                ),
                min_inferred_fact_f1=suite.reasoning.min_inferred_fact_f1,
            ),
        )
        return GraphBoundExecution(
            inputs=inputs,
            graph_sha256=before,
            fixture_sha256=suite.fixture_sha256,
            query_hashes=dict(query_hashes),
            observed_latencies_ms=dict(latencies),
            predicted_answers=predicted_answers,
            gold_answers=gold_answers,
            vector_rankings=vector_rankings,
            vector_relevant=vector_relevant,
            graph_backend_id=(
                f"{self.answer_backend.backend_id}+{self.vector_backend.backend_id}"
            ),
            graph_snapshot_id=self.vector_backend.snapshot_id(before),
            certifiable_backend=self.vector_backend.certifiable,
            training_graph_sha256=training_graph_sha256,
            held_out_edge_ids=held_out_edge_ids,
            derived_popularity_strata=derived_popularity_strata,
            formation_gold_executed=formation_gold_executed,
            formation_gold_passed=formation_gold_passed,
            formation_gold_metrics=formation_gold_metrics,
            formation_gold_assertions=formation_gold_assertions,
        )

    @staticmethod
    def _prepare_held_out_protocol(
        graph: PropertyGraph,
        suite: VersionedGraphEvaluationSuite,
    ) -> tuple[
        PropertyGraph,
        tuple[str, ...],
        dict[str, _NeighborProtocolCase],
        dict[str, _NeighborProtocolCase],
        dict[str, Literal["head", "torso", "tail"]],
    ]:
        held_out: set[str] = set()
        link_protocol: dict[str, _NeighborProtocolCase] = {}
        recommendation_protocol: dict[str, _NeighborProtocolCase] = {}
        link_signatures: set[tuple[str, str, str, str]] = set()
        recommendation_signatures: set[
            tuple[str, str, str, tuple[str, ...]]
        ] = set()

        for case in suite.link_prediction.cases:
            source_id, candidates = _resolve_neighbor_scope(
                graph,
                case.query,
                family="link prediction",
                case_id=case.case_id,
            )
            if case.true_entity_id not in candidates:
                raise ValueError(
                    f"link-prediction case {case.case_id!r} true entity is "
                    "outside its candidate pool"
                )
            known_truths = _known_neighbor_truths(
                graph,
                source_id=source_id,
                candidate_ids=candidates,
                relationship_type=case.query.relationship_type,
                direction=case.query.direction,
            )
            if case.true_entity_id not in known_truths:
                raise ValueError(
                    f"link-prediction case {case.case_id!r} gold edge does "
                    "not exist in the frozen graph"
                )
            signature = (
                source_id,
                case.query.relationship_type,
                case.query.direction,
                case.true_entity_id,
            )
            if signature in link_signatures:
                raise ValueError(
                    "link-prediction cases must be semantically unique; "
                    f"duplicate case {case.case_id!r}"
                )
            link_signatures.add(signature)
            held_out.update(
                _matching_edge_ids(
                    graph,
                    source_id=source_id,
                    target_ids={case.true_entity_id},
                    relationship_type=case.query.relationship_type,
                    direction=case.query.direction,
                )
            )
            link_protocol[case.case_id] = _NeighborProtocolCase(
                source_id=source_id,
                candidate_ids=tuple(sorted(candidates)),
                known_true_entity_ids=tuple(sorted(known_truths)),
            )

        catalog_ids = {
            node.node_id
            for node in _select_nodes(graph, suite.recommendation.catalog)
        }
        if len(catalog_ids) < 3:
            raise ValueError(
                "recommendation requires a catalog of at least three items"
            )
        catalog_by_id = {node.node_id: node for node in graph.nodes}
        for item_id in catalog_ids:
            node = catalog_by_id[item_id]
            if not any(
                node.properties.get(property_name) not in (None, "")
                for property_name in suite.recommendation.feature_properties
            ):
                raise ValueError(
                    "every recommendation catalog item requires at least one "
                    f"declared feature property; missing for {item_id!r}"
                )

        for case in suite.recommendation.cases:
            source_id, candidates = _resolve_neighbor_scope(
                graph,
                case.query,
                family="recommendation",
                case_id=case.case_id,
            )
            if not candidates.issubset(catalog_ids):
                raise ValueError(
                    f"recommendation case {case.case_id!r} candidates must "
                    "all belong to the declared catalog"
                )
            relevant = set(case.relevant_item_ids)
            if len(relevant) != len(case.relevant_item_ids):
                raise ValueError(
                    f"recommendation case {case.case_id!r} contains duplicate "
                    "relevance judgments"
                )
            if not relevant.issubset(candidates):
                raise ValueError(
                    f"recommendation case {case.case_id!r} has relevant items "
                    "outside its candidate pool"
                )
            known_truths = _known_neighbor_truths(
                graph,
                source_id=source_id,
                candidate_ids=candidates,
                relationship_type=case.query.relationship_type,
                direction=case.query.direction,
            )
            if relevant != known_truths:
                raise ValueError(
                    f"recommendation case {case.case_id!r} relevance gold must "
                    "exactly cover all known positives in the frozen graph"
                )
            signature = (
                source_id,
                case.query.relationship_type,
                case.query.direction,
                tuple(sorted(relevant)),
            )
            if signature in recommendation_signatures:
                raise ValueError(
                    "recommendation cases must be semantically unique; "
                    f"duplicate case {case.case_id!r}"
                )
            recommendation_signatures.add(signature)
            held_out.update(
                _matching_edge_ids(
                    graph,
                    source_id=source_id,
                    target_ids=relevant,
                    relationship_type=case.query.relationship_type,
                    direction=case.query.direction,
                )
            )
            recommendation_protocol[case.case_id] = _NeighborProtocolCase(
                source_id=source_id,
                candidate_ids=tuple(sorted(candidates)),
                known_true_entity_ids=tuple(sorted(known_truths)),
            )

        if not held_out:
            raise ValueError(
                "held-out link/recommendation gold did not resolve to graph edges"
            )
        training_graph = PropertyGraph(
            nodes=[node.model_copy(deep=True) for node in graph.nodes],
            edges=[
                edge.model_copy(deep=True)
                for edge in graph.edges
                if edge.edge_id not in held_out
            ],
            metadata=dict(graph.metadata),
        )
        candidate_ids = {
            candidate_id
            for protocol in link_protocol.values()
            for candidate_id in protocol.candidate_ids
        }
        entity_strata = _training_popularity_strata(
            training_graph,
            candidate_ids,
        )
        derived_popularity_strata = {
            case.case_id: entity_strata[case.true_entity_id]
            for case in suite.link_prediction.cases
        }
        missing_strata = {"head", "torso", "tail"} - set(
            derived_popularity_strata.values()
        )
        if missing_strata:
            raise ValueError(
                "training-derived link-prediction support must include head, "
                f"torso, and tail cases; missing {sorted(missing_strata)}"
            )
        return (
            training_graph,
            tuple(sorted(held_out)),
            link_protocol,
            recommendation_protocol,
            derived_popularity_strata,
        )

    @staticmethod
    def _timed(operation: Any) -> tuple[Any, float]:
        started = time.perf_counter_ns()
        value = operation()
        elapsed_ms = max((time.perf_counter_ns() - started) / 1_000_000.0, 0.0)
        return value, elapsed_ms

    @staticmethod
    def _record(
        family: str,
        query: BaseModel,
        elapsed_ms: float,
        hashes: dict[str, list[str]],
        latencies: dict[str, list[float]],
    ) -> None:
        hashes[family].append(_payload_sha256(query.model_dump(mode="json")))
        latencies[family].append(elapsed_ms)

    @staticmethod
    def _run_sql(
        graph: PropertyGraph,
        case: BoundSQLLikeCase,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        nodes = _select_nodes(graph, case.query.selector)
        rows = [
            {
                alias: _project_node(node, expression)
                for alias, expression in case.query.projections.items()
            }
            for node in nodes
        ]
        aggregations: dict[str, object] = {
            name: len(nodes)
            for name in case.expected_aggregations
        }
        return rows, aggregations

    @staticmethod
    def _run_traversal(
        graph: PropertyGraph,
        query: GraphTraversalQuery,
    ) -> tuple[list[str], list[list[str]]]:
        starts = [node.node_id for node in _select_nodes(graph, query.start)]
        return _traverse(
            graph,
            starts,
            relationship_types=set(query.relationship_types),
            max_depth=query.max_depth,
            direction=query.direction,
        )

    def _run_graphrag(
        self,
        graph: PropertyGraph,
        query: LexicalRetrievalQuery,
    ) -> tuple[list[str], GraphAnswerObservation]:
        query_terms = set(_tokens(query.query_text))
        scored = []
        for node in _select_nodes(graph, query.candidates):
            node_terms = set(_tokens(_node_search_text(node)))
            score = len(query_terms & node_terms) / max(1, len(query_terms | node_terms))
            if score > 0.0:
                scored.append((score, node.node_id))
        contexts = [
            node_id
            for _, node_id in sorted(scored, key=lambda item: (-item[0], item[1]))[
                : query.k
            ]
        ]
        answer = self.answer_backend.answer(
            graph,
            query=query,
            retrieved_context_ids=contexts,
        )
        return contexts, answer

    def _run_vector(
        self,
        graph: PropertyGraph,
        query: VectorSearchQuery,
    ) -> VectorWorkloadObservation:
        return self.vector_backend.query(graph, query)

    @staticmethod
    def _run_neighbor_ranking(
        graph: PropertyGraph,
        query: NeighborRankingQuery,
    ) -> list[str]:
        sources = [node.node_id for node in _select_nodes(graph, query.source)]
        candidates = {
            node.node_id for node in _select_nodes(graph, query.candidates)
        }
        adjacency = _typed_adjacency(
            graph,
            relationship_type=query.relationship_type,
            direction=query.direction,
        )
        direct = {
            target
            for source in sources
            for target in adjacency.get(source, set())
            if target in candidates
        }
        scored: list[tuple[int, int, str]] = []
        for candidate in candidates:
            common = sum(
                len(adjacency.get(source, set()) & adjacency.get(candidate, set()))
                for source in sources
            )
            scored.append((int(candidate in direct), common, candidate))
        return [
            candidate
            for _, _, candidate in sorted(
                scored,
                key=lambda item: (-item[0], -item[1], item[2]),
            )[: query.k]
        ]

    @staticmethod
    def _run_reasoning(
        graph: PropertyGraph,
        query: ReasoningPathQuery,
    ) -> tuple[list[str], list[ReasoningFact]]:
        starts = [node.node_id for node in _select_nodes(graph, query.start)]
        current = set(starts)
        for relationship_type in query.relationship_type_sequence:
            adjacency = _typed_adjacency(
                graph,
                relationship_type=relationship_type,
                direction="outgoing",
            )
            current = {
                target
                for source in current
                for target in adjacency.get(source, set())
            }
        answers = sorted(current)
        facts = [
            ReasoningFact(
                subject=source,
                predicate=query.inferred_predicate,
                object=target,
            )
            for source in sorted(starts)
            for target in answers
        ]
        return answers, facts


class GraphBoundIntendedUseEvaluator:
    """Compute observations from the graph, then attach immutable evidence."""

    name = "graph_bound_intended_use_evaluator"

    def __init__(
        self,
        *,
        runner: GraphWorkloadRunner | None = None,
        metrics: IntendedUseMetricsEvaluator | None = None,
    ) -> None:
        self.runner = runner or GraphWorkloadRunner()
        self.metrics = metrics or IntendedUseMetricsEvaluator()

    def evaluate(
        self,
        graph: PropertyGraph,
        suite: VersionedGraphEvaluationSuite,
    ) -> tuple[IntendedUseEvaluationReport, GraphBoundExecution]:
        execution = self.runner.run(graph, suite)
        raw = self.metrics.evaluate(execution.inputs)
        families = {}
        for name, family in raw.families.items():
            query_hashes = execution.query_hashes[name]
            latencies = execution.observed_latencies_ms[name]
            families[name] = family.model_copy(
                update={
                    "workload_support": family.evaluated_cases,
                    "graph_sha256": execution.graph_sha256,
                    "fixture_sha256": execution.fixture_sha256,
                    "graph_backend_id": execution.graph_backend_id,
                    "graph_snapshot_id": execution.graph_snapshot_id,
                    "query_sha256": query_hashes,
                    "observed_latency_ms": latencies,
                    "metrics": {
                        **family.metrics,
                        "workload_support": float(family.evaluated_cases),
                    },
                }
            )
        report = IntendedUseEvaluationReport(
            families=families,
            executed_uses=raw.executed_uses,
            missing_uses=raw.missing_uses,
            passed=(
                raw.passed
                and (
                    execution.formation_gold_passed
                    if suite.construction_profile in {"structured", "mixed"}
                    else True
                )
            ),
            suite_version=suite.version,
            graph_sha256=execution.graph_sha256,
            fixture_sha256=execution.fixture_sha256,
            graph_backend_id=execution.graph_backend_id,
            graph_snapshot_id=execution.graph_snapshot_id,
            graph_bound=True,
            construction_profile=suite.construction_profile,
            training_graph_sha256=execution.training_graph_sha256,
            held_out_edge_ids=list(execution.held_out_edge_ids),
            derived_popularity_strata=execution.derived_popularity_strata,
            formation_gold_executed=execution.formation_gold_executed,
            formation_gold_passed=execution.formation_gold_passed,
            formation_gold_metrics=execution.formation_gold_metrics,
            formation_gold_assertions=execution.formation_gold_assertions,
        )
        return report, execution


def evaluation_fixture_sha256(
    suite: VersionedGraphEvaluationSuite,
) -> str:
    payload = suite.model_dump(
        mode="json",
        exclude={"fixture_sha256"},
    )
    return _payload_sha256(payload)


def property_graph_sha256(graph: PropertyGraph) -> str:
    """Hash a graph snapshot independently of list insertion order."""

    payload = {
        "nodes": sorted(
            (node.model_dump(mode="json") for node in graph.nodes),
            key=lambda item: str(item["node_id"]),
        ),
        "edges": sorted(
            (edge.model_dump(mode="json") for edge in graph.edges),
            key=lambda item: str(item["edge_id"]),
        ),
        "metadata": graph.metadata,
    }
    return _payload_sha256(payload)


def predicted_quality_mappings(
    batch: ConstructionBatch,
) -> tuple[dict[str, str], dict[str, str | None]]:
    """Derive occurrence-bound coreference/linking observations.

    Provider-generated mention IDs are not gold identities: they can change
    when a locked provider implementation changes even when its exact
    observation does not.  Evaluation keys therefore bind to the immutable
    source snapshot, document version, offsets, and detected type.
    """

    coreference: dict[str, str] = {}
    links: dict[str, str | None] = {}
    for entity in batch.entities:
        if entity.entity_type != "Mention":
            continue
        key = _mention_occurrence_key(entity)
        if key is None:
            continue
        cluster = entity.properties.get("coreference_cluster_id")
        coreference[key] = str(cluster or key)
        linked = entity.properties.get("linked_canonical_entity_id")
        links[key] = None if linked in (None, "") else str(linked)
    return coreference, links


def _mention_occurrence_key(entity: Any) -> str | None:
    detected_type = entity.properties.get("detected_entity_type")
    document_version = entity.properties.get("document_version")
    for evidence in entity.evidence:
        start = evidence.location.get("start")
        end = evidence.location.get("end")
        if (
            evidence.location.get("kind") == "span"
            and isinstance(start, int)
            and isinstance(end, int)
            and detected_type not in (None, "")
            and document_version not in (None, "")
        ):
            return canonical_json(
                [
                    evidence.source_id,
                    str(document_version),
                    start,
                    end,
                    str(detected_type).casefold(),
                ]
            )
    return None


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _select_nodes(
    graph: PropertyGraph,
    selector: GraphNodeSelector,
) -> list[PropertyNode]:
    node_ids = set(selector.node_ids)
    labels = set(selector.labels)
    selected = [
        node
        for node in graph.nodes
        if (not node_ids or node.node_id in node_ids)
        and labels.issubset(node.labels)
        and all(node.properties.get(key) == value for key, value in selector.properties.items())
    ]
    return sorted(selected, key=lambda node: node.node_id)


def _project_node(node: PropertyNode, expression: str) -> object:
    if expression == "$id":
        return node.node_id
    if expression == "$labels":
        return sorted(node.labels)
    if expression.startswith("$property."):
        return node.properties.get(expression.removeprefix("$property."))
    raise ValueError(
        "projection expressions must be '$id', '$labels', or '$property.<name>'"
    )


def _edge_steps(
    graph: PropertyGraph,
    *,
    relationship_types: set[str],
    direction: Literal["outgoing", "incoming", "both"],
) -> dict[str, list[tuple[str, str]]]:
    steps: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for edge in graph.edges:
        if edge.type not in relationship_types:
            continue
        if direction in {"outgoing", "both"}:
            steps[edge.source_id].append((edge.type, edge.target_id))
        if direction in {"incoming", "both"}:
            steps[edge.target_id].append((edge.type, edge.source_id))
    for node_steps in steps.values():
        node_steps.sort()
    return steps


def _traverse(
    graph: PropertyGraph,
    starts: Sequence[str],
    *,
    relationship_types: set[str],
    max_depth: int,
    direction: Literal["outgoing", "incoming", "both"],
) -> tuple[list[str], list[list[str]]]:
    steps = _edge_steps(
        graph,
        relationship_types=relationship_types,
        direction=direction,
    )
    visited = set(starts)
    paths: list[list[str]] = []
    queue: deque[tuple[str, int, list[str]]] = deque(
        (start, 0, [start]) for start in sorted(starts)
    )
    while queue:
        node_id, depth, path = queue.popleft()
        if depth >= max_depth:
            continue
        for edge_type, target in steps.get(node_id, []):
            next_path = [*path, edge_type, target]
            paths.append(next_path)
            if target not in visited:
                visited.add(target)
                queue.append((target, depth + 1, next_path))
    return sorted(visited), paths


def _typed_adjacency(
    graph: PropertyGraph,
    *,
    relationship_type: str,
    direction: Literal["outgoing", "incoming"],
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.type != relationship_type:
            continue
        if direction == "outgoing":
            adjacency[edge.source_id].add(edge.target_id)
        else:
            adjacency[edge.target_id].add(edge.source_id)
    return adjacency


def _resolve_neighbor_scope(
    graph: PropertyGraph,
    query: NeighborRankingQuery,
    *,
    family: str,
    case_id: str,
) -> tuple[str, set[str]]:
    source_ids = [
        node.node_id for node in _select_nodes(graph, query.source)
    ]
    if len(source_ids) != 1:
        raise ValueError(
            f"{family} case {case_id!r} must resolve exactly one source node; "
            f"resolved={source_ids}"
        )
    candidate_ids = {
        node.node_id for node in _select_nodes(graph, query.candidates)
    }
    if len(candidate_ids) < 3:
        raise ValueError(
            f"{family} case {case_id!r} requires at least three candidates"
        )
    return source_ids[0], candidate_ids


def _known_neighbor_truths(
    graph: PropertyGraph,
    *,
    source_id: str,
    candidate_ids: set[str],
    relationship_type: str,
    direction: Literal["outgoing", "incoming"],
) -> set[str]:
    adjacency = _typed_adjacency(
        graph,
        relationship_type=relationship_type,
        direction=direction,
    )
    return adjacency.get(source_id, set()) & candidate_ids


def _matching_edge_ids(
    graph: PropertyGraph,
    *,
    source_id: str,
    target_ids: set[str],
    relationship_type: str,
    direction: Literal["outgoing", "incoming"],
) -> set[str]:
    matches: set[str] = set()
    for edge in graph.edges:
        if edge.type != relationship_type:
            continue
        source_matches = (
            edge.source_id == source_id
            and edge.target_id in target_ids
            if direction == "outgoing"
            else edge.target_id == source_id
            and edge.source_id in target_ids
        )
        if source_matches:
            matches.add(edge.edge_id)
    return matches


def _training_popularity_strata(
    graph: PropertyGraph,
    entity_ids: set[str],
) -> dict[str, Literal["head", "torso", "tail"]]:
    if len(entity_ids) < 3:
        raise ValueError(
            "popularity stratification requires at least three candidate entities"
        )
    frequency: dict[str, int] = {entity_id: 0 for entity_id in entity_ids}
    for edge in graph.edges:
        if edge.source_id in frequency:
            frequency[edge.source_id] += 1
        if edge.target_id in frequency:
            frequency[edge.target_id] += 1
    ordered = sorted(
        entity_ids,
        key=lambda entity_id: (-frequency[entity_id], entity_id),
    )
    labels: tuple[Literal["head", "torso", "tail"], ...] = (
        "head",
        "torso",
        "tail",
    )
    return {
        entity_id: labels[min(2, index * 3 // len(ordered))]
        for index, entity_id in enumerate(ordered)
    }


def _evaluate_structured_formation(
    graph: PropertyGraph,
    gold: StructuredFormationGold | None,
) -> tuple[bool, bool, dict[str, float], dict[str, bool]]:
    if gold is None:
        return False, False, {}, {}

    nodes_by_label = {
        label: [node for node in graph.nodes if label in node.labels]
        for label in gold.expected_node_counts
    }
    relationship_counts = defaultdict(int)
    for edge in graph.edges:
        relationship_counts[edge.type] += 1

    assertions: dict[str, bool] = {
        f"record_count:{gold.record_label}": (
            len(nodes_by_label[gold.record_label])
            == gold.expected_record_count
        )
    }
    assertions.update(
        {
            f"node_count:{label}": len(nodes_by_label[label]) == expected
            for label, expected in sorted(gold.expected_node_counts.items())
        }
    )
    assertions.update(
        {
            f"relationship_count:{relationship_type}": (
                relationship_counts[relationship_type] == expected
            )
            for relationship_type, expected in sorted(
                gold.expected_relationship_counts.items()
            )
        }
    )

    required_total = 0
    required_present = 0
    for label, properties in sorted(gold.required_properties.items()):
        for property_name in properties:
            label_nodes = nodes_by_label[label]
            present = sum(
                node.properties.get(property_name) not in (None, "")
                for node in label_nodes
            )
            required_total += len(label_nodes)
            required_present += present
            assertions[f"required_property:{label}.{property_name}"] = (
                bool(label_nodes) and present == len(label_nodes)
            )

    forbidden_total = 0
    forbidden_absent = 0
    for label, properties in sorted(gold.forbidden_properties.items()):
        for property_name in properties:
            label_nodes = nodes_by_label[label]
            absent = sum(
                node.properties.get(property_name) in (None, "")
                for node in label_nodes
            )
            forbidden_total += len(label_nodes)
            forbidden_absent += absent
            assertions[f"forbidden_property:{label}.{property_name}"] = (
                bool(label_nodes) and absent == len(label_nodes)
            )

    identity_total = 0
    identity_complete = 0
    unique_label_count = 0
    for label, key_properties in sorted(gold.identity_keys.items()):
        label_nodes = nodes_by_label[label]
        complete_keys = [
            tuple(
                canonical_json(node.properties.get(property_name))
                for property_name in key_properties
            )
            for node in label_nodes
            if all(
                node.properties.get(property_name) not in (None, "")
                for property_name in key_properties
            )
        ]
        identity_total += len(label_nodes)
        identity_complete += len(complete_keys)
        complete = bool(label_nodes) and len(complete_keys) == len(label_nodes)
        unique = complete and len(set(complete_keys)) == len(label_nodes)
        unique_label_count += int(unique)
        assertions[f"identity_complete:{label}"] = complete
        assertions[f"identity_unique:{label}"] = unique

    expected_counts = [
        *gold.expected_node_counts.values(),
        *gold.expected_relationship_counts.values(),
    ]
    metrics = {
        "structured_formation_gold_support": float(
            sum(expected_counts)
            + required_total
            + forbidden_total
            + identity_total
        ),
        "record_count_exact": float(
            assertions[f"record_count:{gold.record_label}"]
        ),
        "node_count_exact": (
            sum(
                assertions[f"node_count:{label}"]
                for label in gold.expected_node_counts
            )
            / len(gold.expected_node_counts)
        ),
        "relationship_count_exact": (
            sum(
                assertions[f"relationship_count:{relationship_type}"]
                for relationship_type in gold.expected_relationship_counts
            )
            / len(gold.expected_relationship_counts)
        ),
        "required_property_coverage": (
            required_present / required_total if required_total else 0.0
        ),
        "forbidden_property_absence": (
            forbidden_absent / forbidden_total if forbidden_total else 0.0
        ),
        "identity_key_completeness": (
            identity_complete / identity_total if identity_total else 0.0
        ),
        "identity_key_uniqueness": (
            unique_label_count / len(gold.identity_keys)
        ),
    }
    return True, all(assertions.values()), metrics, assertions


def _node_search_text(node: PropertyNode) -> str:
    return " ".join(
        [
            node.node_id,
            *node.labels,
            *(
                str(value)
                for key, value in sorted(node.properties.items())
                if key
                not in {
                    "candidate_provenance",
                    "evidence_support",
                    "relation_provenance",
                }
            ),
        ]
    )


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(value)]


def _text_vector(value: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(value):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return vector


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _catalog_popularity(
    graph: PropertyGraph,
    catalog_ids: Sequence[str],
) -> dict[str, float]:
    incoming = {item_id: 0 for item_id in catalog_ids}
    for edge in graph.edges:
        if edge.target_id in incoming:
            incoming[edge.target_id] += 1
    total = sum(incoming.values())
    if total:
        return {item_id: count / total for item_id, count in incoming.items()}
    uniform = 1.0 / len(catalog_ids) if catalog_ids else 0.0
    return {item_id: uniform for item_id in catalog_ids}


def _node_features(node: PropertyNode, properties: Iterable[str]) -> list[str]:
    features = {f"label:{label}" for label in node.labels}
    for property_name in properties:
        value = node.properties.get(property_name)
        for item in _as_list(value):
            features.add(f"{property_name}:{item}")
    return sorted(features)


def _as_list(value: object) -> list[object]:
    if isinstance(value, list | tuple | set | frozenset):
        return list(value)
    if value in (None, ""):
        return []
    return [value]


__all__ = [
    "GRAPH_EVALUATION_SUITE_VERSION",
    "BoundGraphRAGCase",
    "BoundGraphRAGSuite",
    "BoundLinkPredictionCase",
    "BoundLinkPredictionSuite",
    "BoundNoSQLCase",
    "BoundNoSQLSuite",
    "BoundReasoningCase",
    "BoundReasoningSuite",
    "BoundRecommendationCase",
    "BoundRecommendationSuite",
    "BoundSQLLikeCase",
    "BoundSQLLikeSuite",
    "BoundVectorCase",
    "BoundVectorSuite",
    "DeterministicVectorWorkloadBackend",
    "EvidenceGroundedPathAnswerBackend",
    "GraphAnswerBackend",
    "GraphAnswerObservation",
    "GraphBoundExecution",
    "GraphBoundIntendedUseEvaluator",
    "GraphBoundQualityGold",
    "GraphNodeSelector",
    "GraphTraversalQuery",
    "GraphWorkloadRunner",
    "LexicalRetrievalQuery",
    "NeighborRankingQuery",
    "Neo4jVectorWorkloadBackend",
    "PathQuery",
    "ReasoningPathQuery",
    "SQLNodeQuery",
    "StructuredFormationGold",
    "VectorSearchQuery",
    "VectorWorkloadBackend",
    "VectorWorkloadObservation",
    "VersionedGraphEvaluationSuite",
    "evaluation_fixture_sha256",
    "predicted_quality_mappings",
    "property_graph_sha256",
]
