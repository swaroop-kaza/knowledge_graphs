"""Evaluate the quality of a constructed knowledge graph and gate a release.

Computes a quality profile over the graph across four dimensions:

- **conciseness** (freedom from redundancy): ``categorical_duplication_rate``
  (0 when each dimension value is a single shared node), plus
  ``literal_that_should_be_node_rate`` and ``duplicate_node_rate``;
- **completeness**: ``schema_completeness``, ``property_completeness``,
  ``relationship_completeness``, and per-class population;
- **consistency**: domain/range and cardinality violations against the ontology (a
  property-graph equivalent of SHACL), plus an optional pySHACL violation count; and
- **structure**: node/edge counts, average degree, orphan rate, connected
  components, and class/relationship/attribute richness.

Categorical duplication is policy-aware: only columns the policy promotes to nodes
are checked, so a value kept as a property by the supernode guard is not wrongly
flagged as duplication.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.domain.modeling import ColumnRole, OntologySchema
from kg_construction.infrastructure.resolution.common import resolution_key

JsonDict = dict[str, object]

logger = logging.getLogger(__name__)

_NON_DOMAIN_LABELS = {"Fact", "RelationshipFact", "Literal", "Resource"}
_PROVENANCE_PREFIXES = (
    "_", "canonical", "identity", "evidence", "source_", "merged", "merge_", "resolution",
    "survivor", "candidate", "property_sources", "assertion", "semantic_relation",
)
_PROVENANCE_KEYS = frozenset(
    {
        "name", "confidence", "extractor_name", "technique", "schema_version", "created_at",
        "resolver", "review_required", "link_status", "knowledge_base", "property_conflicts",
    }
)


def _is_domain_property(key: str) -> bool:
    return not (key in _PROVENANCE_KEYS or key.startswith(_PROVENANCE_PREFIXES))


class SchemaQualityGateSettings(BaseModel):
    """Release gates for schema/instance quality."""

    model_config = ConfigDict(extra="forbid")

    maximum_categorical_duplication_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_duplicate_node_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_orphan_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    maximum_consistency_violation_count: int = Field(default=0, ge=0)


class SchemaQualityGate(BaseModel):
    """One evaluated schema-quality gate."""

    model_config = ConfigDict(extra="forbid")

    gate: str
    metric: str
    operator: str
    threshold: float
    actual: float
    passed: bool


class SchemaQualityReport(BaseModel):
    """Reported schema/instance quality metrics and gate results."""

    model_config = ConfigDict(extra="forbid")

    # conciseness
    categorical_duplication_rate: float | None = None
    literal_that_should_be_node_rate: float | None = None
    duplicate_node_rate: float = 0.0
    # completeness
    schema_completeness: float | None = None
    property_completeness: float = 1.0
    relationship_completeness: float = 1.0
    class_population: dict[str, int] = Field(default_factory=dict)
    # consistency
    domain_range_violation_rate: float | None = None
    cardinality_violation_rate: float | None = None
    shacl_violation_count: int | None = None
    consistency_violation_count: int = 0
    # structure (OntoQA)
    node_count: int = 0
    edge_count: int = 0
    average_degree: float = 0.0
    orphan_rate: float = 0.0
    connected_component_count: int = 0
    class_richness: float | None = None
    distinct_class_count: int = 0
    relationship_richness: float = 0.0
    attribute_richness: float = 0.0
    # gates
    gates: list[SchemaQualityGate] = Field(default_factory=list)
    passed: bool = True

    def metric_payload(self) -> dict[str, float]:
        """Return the numeric metrics usable as benchmark hard-gate inputs."""

        payload: dict[str, float] = {
            "duplicate_node_rate": self.duplicate_node_rate,
            "orphan_rate": self.orphan_rate,
            "consistency_violation_count": float(self.consistency_violation_count),
            "average_degree": self.average_degree,
            "relationship_richness": self.relationship_richness,
            "attribute_richness": self.attribute_richness,
        }
        if self.categorical_duplication_rate is not None:
            payload["categorical_duplication_rate"] = self.categorical_duplication_rate
        return payload


class SchemaQualityEvaluator:
    """Compute the schema/instance quality profile and gate a release."""

    name = "schema_quality_evaluator"

    def evaluate(
        self,
        graph: PropertyGraph,
        *,
        policy: object | None = None,
        source_records: list[JsonDict] | None = None,
        ontology: OntologySchema | None = None,
        shacl_violation_count: int | None = None,
        gate_settings: SchemaQualityGateSettings | None = None,
    ) -> SchemaQualityReport:
        domain_nodes = self._domain_nodes(graph)
        report = self._base_report(graph, domain_nodes)
        report.property_completeness = self._property_completeness(domain_nodes)

        if policy is not None and source_records is not None:
            duplication, literal_rate, relationship_completeness = self._policy_conciseness(
                graph, policy, source_records
            )
            report.categorical_duplication_rate = duplication
            report.literal_that_should_be_node_rate = literal_rate
            report.relationship_completeness = relationship_completeness

        if ontology is not None:
            report.schema_completeness = self._schema_completeness(graph, domain_nodes, ontology)
            dr_count, dr_rate, card_count, card_rate = self._consistency(graph, ontology)
            report.domain_range_violation_rate = dr_rate
            report.cardinality_violation_rate = card_rate
            report.class_richness = self._class_richness(domain_nodes, ontology)
            report.consistency_violation_count += dr_count + card_count
        if shacl_violation_count is not None:
            report.shacl_violation_count = shacl_violation_count
            report.consistency_violation_count += shacl_violation_count

        report.gates = self._gates(report, gate_settings or SchemaQualityGateSettings())
        report.passed = all(gate.passed for gate in report.gates)
        logger.info(
            "Schema quality: categorical_duplication=%s duplicate_nodes=%.3f "
            "orphan_rate=%.3f consistency_violations=%d passed=%s",
            report.categorical_duplication_rate,
            report.duplicate_node_rate,
            report.orphan_rate,
            report.consistency_violation_count,
            report.passed,
        )
        return report

    def structural_metrics(self, graph: PropertyGraph) -> dict[str, float]:
        """Graph-only metrics for the benchmark runner (no policy/ontology needed)."""

        return self._base_report(graph, self._domain_nodes(graph)).metric_payload()

    # -- structure -------------------------------------------------------------

    @staticmethod
    def _domain_nodes(graph: PropertyGraph) -> list[PropertyNode]:
        return [node for node in graph.nodes if not (_NON_DOMAIN_LABELS & set(node.labels))]

    def _base_report(
        self, graph: PropertyGraph, domain_nodes: list[PropertyNode]
    ) -> SchemaQualityReport:
        node_ids = {node.node_id for node in domain_nodes}
        edges = [edge for edge in graph.edges if edge.source_id in node_ids and edge.target_id in node_ids]
        degree: Counter[str] = Counter()
        for edge in edges:
            degree[edge.source_id] += 1
            degree[edge.target_id] += 1
        node_count = len(domain_nodes)
        orphans = sum(1 for node in domain_nodes if degree[node.node_id] == 0)
        population = self._class_population(domain_nodes)
        report = SchemaQualityReport()
        report.node_count = node_count
        report.edge_count = len(edges)
        report.average_degree = (2 * len(edges) / node_count) if node_count else 0.0
        report.orphan_rate = (orphans / node_count) if node_count else 0.0
        report.connected_component_count = self._connected_components(node_ids, edges)
        report.relationship_richness = self._relationship_richness(domain_nodes, edges)
        report.attribute_richness = self._attribute_richness(domain_nodes)
        report.duplicate_node_rate = self._duplicate_node_rate(domain_nodes)
        report.class_population = population
        report.distinct_class_count = len(population)
        return report

    @staticmethod
    def _connected_components(node_ids: set[str], edges: list[PropertyEdge]) -> int:
        parent = {node_id: node_id for node_id in node_ids}

        def find(node_id: str) -> str:
            root = node_id
            while parent[root] != root:
                root = parent[root]
            while parent[node_id] != root:
                parent[node_id], node_id = root, parent[node_id]
            return root

        for edge in edges:
            parent[find(edge.source_id)] = find(edge.target_id)
        return len({find(node_id) for node_id in node_ids})

    def _relationship_richness(self, domain_nodes: list[PropertyNode], edges: list[PropertyEdge]) -> float:
        attribute_values = sum(self._domain_attribute_count(node) for node in domain_nodes)
        total = len(edges) + attribute_values
        return (len(edges) / total) if total else 0.0

    def _attribute_richness(self, domain_nodes: list[PropertyNode]) -> float:
        if not domain_nodes:
            return 0.0
        return sum(self._domain_attribute_count(node) for node in domain_nodes) / len(domain_nodes)

    @staticmethod
    def _domain_attribute_count(node: PropertyNode) -> int:
        return sum(
            1
            for key, value in node.properties.items()
            if _is_domain_property(key) and value not in (None, "")
        )

    # -- conciseness -----------------------------------------------------------

    @staticmethod
    def _duplicate_node_rate(domain_nodes: list[PropertyNode]) -> float:
        if not domain_nodes:
            return 0.0
        signatures = {
            (
                tuple(sorted(label for label in node.labels if label != "Entity")),
                resolution_key(str(node.properties.get("name", ""))),
            )
            for node in domain_nodes
        }
        return 1.0 - (len(signatures) / len(domain_nodes))

    def _policy_conciseness(
        self, graph: PropertyGraph, policy: object, source_records: list[JsonDict]
    ) -> tuple[float, float, float]:
        record_type = getattr(policy, "record_type", "Record")
        record_nodes = [node for node in graph.nodes if record_type in node.labels]
        node_columns = self._node_columns(policy, source_records)

        total_mentions = 0
        redundant = 0
        literal_cells = 0
        for column in node_columns:
            mentions = sum(1 for row in source_records if row.get(column) not in (None, ""))
            total_mentions += mentions
            values = [
                str(node.properties[column])
                for node in record_nodes
                if column in node.properties and node.properties[column] not in (None, "")
            ]
            if values:
                literal_cells += len(values)
                counts = Counter(values)
                redundant += sum(count - 1 for count in counts.values())
        duplication = (redundant / total_mentions) if total_mentions else 0.0
        literal_rate = (literal_cells / total_mentions) if total_mentions else 0.0
        connected_records = sum(
            1 for node in record_nodes
            if any(edge.source_id == node.node_id for edge in graph.edges)
        )
        relationship_completeness = (connected_records / len(record_nodes)) if record_nodes else 1.0
        return duplication, literal_rate, relationship_completeness

    @staticmethod
    def _node_columns(policy: object, source_records: list[JsonDict]) -> list[str]:
        columns = getattr(policy, "columns", [])
        thresholds = getattr(policy, "thresholds", None)
        keep_as_property = getattr(thresholds, "keep_as_property_max_distinct", 2)
        promote = getattr(thresholds, "promote_categoricals_to_nodes", True)
        result: list[str] = []
        for column_policy in columns:
            role = getattr(column_policy, "role", None)
            name = getattr(column_policy, "column", "")
            if role == ColumnRole.ENTITY_REFERENCE:
                result.append(name)
            elif role == ColumnRole.CATEGORICAL_DIMENSION:
                distinct = len({str(row.get(name)) for row in source_records if row.get(name) not in (None, "")})
                if promote and distinct > keep_as_property:
                    result.append(name)
        return result

    # -- completeness ----------------------------------------------------------

    @staticmethod
    def _class_population(domain_nodes: list[PropertyNode]) -> dict[str, int]:
        population: Counter[str] = Counter()
        for node in domain_nodes:
            for label in node.labels:
                if label != "Entity":
                    population[label] += 1
        return dict(sorted(population.items()))

    def _property_completeness(self, domain_nodes: list[PropertyNode]) -> float:
        filled = 0
        possible = 0
        for node in domain_nodes:
            domain_keys = [key for key in node.properties if _is_domain_property(key)]
            possible += max(1, len(domain_keys))
            filled += sum(1 for key in domain_keys if node.properties[key] not in (None, ""))
        return (filled / possible) if possible else 1.0

    def _schema_completeness(
        self, graph: PropertyGraph, domain_nodes: list[PropertyNode], ontology: OntologySchema
    ) -> float:
        present_classes = set(self._class_population(domain_nodes))
        present_predicates = {edge.type for edge in graph.edges}
        for node in domain_nodes:
            present_predicates.update(key for key in node.properties if _is_domain_property(key))
        declared = [item.name for item in ontology.classes] + [item.name for item in ontology.predicates]
        if not declared:
            return 1.0
        populated = sum(
            1
            for name in [item.name for item in ontology.classes]
            if name in present_classes
        ) + sum(
            1
            for name in [item.name for item in ontology.predicates]
            if resolution_key(name) in {resolution_key(term) for term in present_predicates}
        )
        return populated / len(declared)

    @staticmethod
    def _class_richness(domain_nodes: list[PropertyNode], ontology: OntologySchema) -> float:
        declared = {item.name for item in ontology.classes}
        if not declared:
            return 1.0
        present = {label for node in domain_nodes for label in node.labels if label != "Entity"}
        return len(declared & present) / len(declared)

    # -- consistency -----------------------------------------------------------

    def _consistency(
        self, graph: PropertyGraph, ontology: OntologySchema
    ) -> tuple[int, float, int, float]:
        node_labels = {
            node.node_id: {label for label in node.labels if label != "Entity"}
            for node in graph.nodes
        }
        object_predicates = {
            resolution_key(predicate.name): predicate
            for predicate in ontology.predicates
            if predicate.kind == "object"
        }
        domain_violations = 0
        checked = 0
        for edge in graph.edges:
            predicate = object_predicates.get(resolution_key(edge.type))
            if predicate is None:
                continue
            checked += 1
            source_ok = predicate.domain in node_labels.get(edge.source_id, set())
            target_ok = predicate.range in node_labels.get(edge.target_id, set())
            if not (source_ok and target_ok):
                domain_violations += 1
        domain_range_rate = (domain_violations / checked) if checked else 0.0

        cardinality_violations = 0
        cardinality_checked = 0
        outgoing: dict[tuple[str, str], int] = defaultdict(int)
        for edge in graph.edges:
            outgoing[(edge.source_id, resolution_key(edge.type))] += 1
        for node in graph.nodes:
            labels = node_labels.get(node.node_id, set())
            for predicate in ontology.predicates:
                if predicate.kind != "object" or predicate.domain not in labels:
                    continue
                cardinality_checked += 1
                count = outgoing.get((node.node_id, resolution_key(predicate.name)), 0)
                below = count < predicate.cardinality.min_count
                above = predicate.cardinality.max_count is not None and count > predicate.cardinality.max_count
                if below or above:
                    cardinality_violations += 1
        cardinality_rate = (cardinality_violations / cardinality_checked) if cardinality_checked else 0.0
        return domain_violations, domain_range_rate, cardinality_violations, cardinality_rate

    # -- gates -----------------------------------------------------------------

    @staticmethod
    def _gates(report: SchemaQualityReport, settings: SchemaQualityGateSettings) -> list[SchemaQualityGate]:
        gates: list[SchemaQualityGate] = [
            SchemaQualityGate(
                gate="node_conciseness", metric="duplicate_node_rate", operator="<=",
                threshold=settings.maximum_duplicate_node_rate, actual=report.duplicate_node_rate,
                passed=report.duplicate_node_rate <= settings.maximum_duplicate_node_rate,
            ),
            SchemaQualityGate(
                gate="connectivity", metric="orphan_rate", operator="<=",
                threshold=settings.maximum_orphan_rate, actual=report.orphan_rate,
                passed=report.orphan_rate <= settings.maximum_orphan_rate,
            ),
            SchemaQualityGate(
                gate="consistency", metric="consistency_violation_count", operator="<=",
                threshold=float(settings.maximum_consistency_violation_count),
                actual=float(report.consistency_violation_count),
                passed=report.consistency_violation_count <= settings.maximum_consistency_violation_count,
            ),
        ]
        if report.categorical_duplication_rate is not None:
            gates.append(
                SchemaQualityGate(
                    gate="categorical_conciseness", metric="categorical_duplication_rate", operator="<=",
                    threshold=settings.maximum_categorical_duplication_rate,
                    actual=report.categorical_duplication_rate,
                    passed=report.categorical_duplication_rate <= settings.maximum_categorical_duplication_rate,
                )
            )
        return gates


__all__ = [
    "SchemaQualityEvaluator",
    "SchemaQualityGate",
    "SchemaQualityGateSettings",
    "SchemaQualityReport",
]
