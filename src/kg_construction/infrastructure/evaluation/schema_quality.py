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
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.domain.modeling import (
    ColumnRole,
    OntologyPredicate,
    OntologySchema,
)
from kg_construction.infrastructure.resolution.common import resolution_key

JsonDict = dict[str, object]

logger = logging.getLogger(__name__)

_NON_DOMAIN_LABELS = {"Fact", "RelationshipFact", "Literal", "Resource"}
_PROVENANCE_PREFIXES = (
    "_", "canonical", "identity", "evidence", "source_", "merged", "merge_", "resolution",
    "survivor", "candidate", "property_sources", "assertion", "semantic_relation", "raw_",
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

    thresholds_version: str = "schema-quality-v2"
    maximum_categorical_duplication_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    minimum_categorical_source_occurrence_coverage: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )
    maximum_duplicate_node_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    maximum_orphan_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    maximum_consistency_violation_count: int = Field(default=0, ge=0)
    minimum_property_completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_relationship_completeness: float = Field(default=1.0, ge=0.0, le=1.0)
    maximum_shacl_violation_count: int = Field(default=0, ge=0)
    require_shacl_execution: Literal[True] = True


class SchemaQualityGate(BaseModel):
    """One evaluated schema-quality gate."""

    model_config = ConfigDict(extra="forbid")

    gate: str
    metric: str
    operator: Literal[">=", "<="]
    threshold: float
    actual: float
    passed: bool

    @model_validator(mode="after")
    def _pass_state_matches_observation(self) -> SchemaQualityGate:
        expected = (
            self.actual >= self.threshold
            if self.operator == ">="
            else self.actual <= self.threshold
        )
        if self.passed != expected:
            raise ValueError(
                "schema-quality gate pass state must match its actual value, "
                "operator, and threshold"
            )
        return self


class SchemaQualityReport(BaseModel):
    """Reported schema/instance quality metrics and gate results."""

    model_config = ConfigDict(extra="forbid")

    thresholds_version: str = "schema-quality-v2"
    # conciseness
    categorical_duplication_rate: float | None = None
    categorical_source_occurrence_coverage: float | None = None
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
    shacl_executed: bool = False
    ontology_evaluated: bool = False
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
    passed: bool = False

    @model_validator(mode="after")
    def _pass_state_is_conjunctive(self) -> SchemaQualityReport:
        if not self.gates:
            if self.passed:
                raise ValueError(
                    "a schema-quality report without evaluated gates cannot pass"
                )
            return self
        expected = all(gate.passed for gate in self.gates)
        if self.passed != expected:
            raise ValueError(
                "schema-quality report pass state must equal the conjunction "
                "of every evaluated gate"
            )
        return self

    def metric_payload(self) -> dict[str, float]:
        """Return the numeric metrics usable as benchmark hard-gate inputs."""

        payload: dict[str, float] = {
            "duplicate_node_rate": self.duplicate_node_rate,
            "orphan_rate": self.orphan_rate,
            "consistency_violation_count": float(self.consistency_violation_count),
            "average_degree": self.average_degree,
            "relationship_richness": self.relationship_richness,
            "attribute_richness": self.attribute_richness,
            "property_completeness": self.property_completeness,
            "relationship_completeness": self.relationship_completeness,
            "shacl_executed": float(self.shacl_executed),
        }
        if self.categorical_duplication_rate is not None:
            payload["categorical_duplication_rate"] = self.categorical_duplication_rate
        if self.categorical_source_occurrence_coverage is not None:
            payload["categorical_source_occurrence_coverage"] = (
                self.categorical_source_occurrence_coverage
            )
        if self.schema_completeness is not None:
            payload["schema_completeness"] = self.schema_completeness
        if self.shacl_violation_count is not None:
            payload["shacl_violation_count"] = float(self.shacl_violation_count)
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
        shacl_executed: bool = False,
        gate_settings: SchemaQualityGateSettings | None = None,
    ) -> SchemaQualityReport:
        domain_nodes = self._domain_nodes(graph)
        settings = gate_settings or SchemaQualityGateSettings()
        report = self._base_report(graph, domain_nodes, ontology=ontology)
        report.thresholds_version = settings.thresholds_version
        report.property_completeness = self._property_completeness(domain_nodes, ontology)

        if policy is not None and source_records is not None:
            (
                duplication,
                literal_rate,
                relationship_completeness,
                occurrence_coverage,
            ) = self._policy_conciseness(
                graph, policy, source_records
            )
            report.categorical_duplication_rate = duplication
            report.categorical_source_occurrence_coverage = occurrence_coverage
            report.literal_that_should_be_node_rate = literal_rate
            if ontology is None:
                report.relationship_completeness = relationship_completeness

        if ontology is not None:
            report.ontology_evaluated = True
            report.schema_completeness = self._schema_completeness(graph, domain_nodes, ontology)
            report.relationship_completeness = self._relationship_completeness(
                graph, ontology
            )
            dr_count, dr_rate, card_count, card_rate = self._consistency(graph, ontology)
            report.domain_range_violation_rate = dr_rate
            report.cardinality_violation_rate = card_rate
            report.class_richness = self._class_richness(domain_nodes, ontology)
            report.consistency_violation_count += dr_count + card_count
        report.shacl_executed = shacl_executed or shacl_violation_count is not None
        if shacl_violation_count is not None:
            report.shacl_violation_count = shacl_violation_count
            report.consistency_violation_count += shacl_violation_count

        report.gates = self._gates(report, settings)
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
        # Mutating the local draft keeps the metric implementation readable.
        # Re-validation here makes the externally returned artifact immutable
        # with respect to the conjunctive pass-state contract.
        return SchemaQualityReport.model_validate(report.model_dump())

    def structural_metrics(self, graph: PropertyGraph) -> dict[str, float]:
        """Graph-only metrics for the benchmark runner (no policy/ontology needed)."""

        return self._base_report(
            graph, self._domain_nodes(graph), ontology=None
        ).metric_payload()

    # -- structure -------------------------------------------------------------

    @staticmethod
    def _domain_nodes(graph: PropertyGraph) -> list[PropertyNode]:
        return [node for node in graph.nodes if not (_NON_DOMAIN_LABELS & set(node.labels))]

    def _base_report(
        self,
        graph: PropertyGraph,
        domain_nodes: list[PropertyNode],
        *,
        ontology: OntologySchema | None,
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
        report.duplicate_node_rate = self._duplicate_node_rate(domain_nodes, ontology)
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
    def _duplicate_node_rate(
        domain_nodes: list[PropertyNode],
        ontology: OntologySchema | None,
    ) -> float:
        """Return duplicate rate using declared/business identity, never display name."""

        if not domain_nodes:
            return 0.0
        class_by_name = {
            item.name: item for item in ontology.classes
        } if ontology is not None else {}
        signatures: list[tuple[str, tuple[tuple[str, str], ...]]] = []
        for node in domain_nodes:
            class_names = [
                label for label in node.labels if label != "Entity" and label in class_by_name
            ]
            if not class_names:
                class_names = [label for label in node.labels if label != "Entity"]
            class_name = sorted(class_names)[0] if class_names else ""
            ontology_class = class_by_name.get(class_name)
            if (
                node.properties.get("concept_scheme") not in (None, "")
                and (
                    node.properties.get("code") not in (None, "")
                    or node.properties.get("canonical_value") not in (None, "")
                )
            ):
                code = node.properties.get(
                    "code",
                    node.properties.get("canonical_value"),
                )
                signatures.append(
                    (
                        class_name,
                        (
                            (
                                "concept_scheme",
                                resolution_key(
                                    str(node.properties["concept_scheme"])
                                ),
                            ),
                            ("code", resolution_key(str(code))),
                        ),
                    )
                )
                continue
            key_names = list(ontology_class.key_properties) if ontology_class else []
            if not key_names:
                key_names = [
                    key
                    for key in ("identity_key", "canonical_entity_id", "canonical_value")
                    if node.properties.get(key) not in (None, "")
                ]
            if (
                not key_names
                and ontology_class is not None
                and ontology_class.role == "reference"
                and node.properties.get("value") not in (None, "")
            ):
                key_names = ["value"]
            if not key_names or any(node.properties.get(key) in (None, "") for key in key_names):
                continue
            signatures.append(
                (
                    class_name,
                    tuple(
                        (key, resolution_key(str(node.properties[key])))
                        for key in key_names
                    ),
                )
            )
        if not signatures:
            return 0.0
        duplicate_count = sum(count - 1 for count in Counter(signatures).values())
        return duplicate_count / len(signatures)

    def _policy_conciseness(
        self, graph: PropertyGraph, policy: object, source_records: list[JsonDict]
    ) -> tuple[float, float, float, float]:
        record_type = getattr(
            policy, "record_class", getattr(policy, "record_type", "Record")
        )
        record_nodes = [node for node in graph.nodes if record_type in node.labels]
        node_columns = self._node_columns(policy, source_records)

        total_mentions = 0
        redundant = 0
        literal_cells = 0
        covered_occurrences: set[tuple[str, int, str]] = set()
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
            for node in graph.nodes:
                supported_locations = False
                for support in node.properties.get("evidence_support", []):
                    if not isinstance(support, dict):
                        continue
                    location = support.get("location")
                    if (
                        isinstance(location, dict)
                        and location.get("kind") == "cell"
                        and location.get("column") == column
                        and isinstance(location.get("row_index"), int)
                    ):
                        supported_locations = True
                        covered_occurrences.add(
                            (
                                str(support.get("source_id", "")),
                                int(location["row_index"]),
                                column,
                            )
                        )
                if supported_locations:
                    continue
                for location in node.properties.get("evidence_locations", []):
                    if (
                        isinstance(location, dict)
                        and location.get("kind") == "cell"
                        and location.get("column") == column
                        and isinstance(location.get("row_index"), int)
                    ):
                        covered_occurrences.add(
                            (
                                "",
                                int(location["row_index"]),
                                column,
                            )
                        )

        concept_signatures: Counter[tuple[str, str, str]] = Counter()
        node_targets = self._node_targets(policy, source_records)
        for target in node_targets.values():
            for node in graph.nodes:
                if target not in node.labels:
                    continue
                scheme = node.properties.get("concept_scheme", target)
                code = node.properties.get(
                    "code",
                    node.properties.get("canonical_value"),
                )
                if code in (None, ""):
                    continue
                concept_signatures[
                    (
                        target,
                        resolution_key(str(scheme)),
                        resolution_key(str(code)),
                    )
                ] += 1
        redundant += sum(
            count - 1 for count in concept_signatures.values()
        )
        duplication = (
            min(1.0, redundant / total_mentions)
            if total_mentions
            else 0.0
        )
        literal_rate = (literal_cells / total_mentions) if total_mentions else 0.0
        occurrence_coverage = (
            min(1.0, len(covered_occurrences) / total_mentions)
            if total_mentions
            else 1.0
        )
        connected_records = sum(
            1 for node in record_nodes
            if any(edge.source_id == node.node_id for edge in graph.edges)
        )
        relationship_completeness = (connected_records / len(record_nodes)) if record_nodes else 1.0
        return (
            duplication,
            literal_rate,
            relationship_completeness,
            occurrence_coverage,
        )

    @staticmethod
    def _node_columns(policy: object, source_records: list[JsonDict]) -> list[str]:
        resolved_attributes = getattr(policy, "attributes", None)
        if resolved_attributes is not None:
            materialized_columns: list[str] = []
            for attribute in resolved_attributes:
                column = str(getattr(attribute, "column", "")).strip()
                materialization = str(
                    getattr(attribute, "materialization", "")
                )
                if (
                    column
                    and materialization == "node"
                    and getattr(attribute, "semantic_role", None)
                    == ColumnRole.CATEGORICAL_DIMENSION
                ):
                    materialized_columns.append(column)
            return materialized_columns
        columns = getattr(policy, "columns", [])
        thresholds = getattr(policy, "thresholds", None)
        keep_as_property = getattr(thresholds, "keep_as_property_max_distinct", 2)
        promote = getattr(thresholds, "promote_categoricals_to_nodes", True)
        result: list[str] = []
        for column_policy in columns:
            role = getattr(column_policy, "role", None)
            name = getattr(column_policy, "column", "")
            if role == ColumnRole.CATEGORICAL_DIMENSION:
                distinct = len({str(row.get(name)) for row in source_records if row.get(name) not in (None, "")})
                if promote and distinct > keep_as_property:
                    result.append(name)
        return result

    @staticmethod
    def _node_targets(
        policy: object,
        source_records: list[JsonDict],
    ) -> dict[str, str]:
        resolved_attributes = getattr(policy, "attributes", None)
        if resolved_attributes is not None:
            return {
                str(attribute.column): str(attribute.target)
                for attribute in resolved_attributes
                if str(getattr(attribute, "materialization", "")) == "node"
                and getattr(attribute, "target", None)
            }
        columns = getattr(policy, "columns", [])
        promoted = set(
            SchemaQualityEvaluator._node_columns(policy, source_records)
        )
        return {
            str(column.column): str(column.reference_type)
            for column in columns
            if str(getattr(column, "column", "")) in promoted
            and getattr(column, "reference_type", None)
            and getattr(column, "role", None)
            == ColumnRole.CATEGORICAL_DIMENSION
        }

    # -- completeness ----------------------------------------------------------

    @staticmethod
    def _class_population(domain_nodes: list[PropertyNode]) -> dict[str, int]:
        population: Counter[str] = Counter()
        for node in domain_nodes:
            for label in node.labels:
                if label != "Entity":
                    population[label] += 1
        return dict(sorted(population.items()))

    def _property_completeness(
        self,
        domain_nodes: list[PropertyNode],
        ontology: OntologySchema | None,
    ) -> float:
        """Measure declared datatype obligations, including properties that are absent."""

        if ontology is None:
            return 1.0
        predicates = [item for item in ontology.predicates if item.kind == "datatype"]
        filled = 0
        possible = 0
        for node in domain_nodes:
            labels = set(node.labels)
            applicable = [item for item in predicates if item.domain in labels]
            possible += len(applicable)
            filled += sum(
                1 for item in applicable if node.properties.get(item.name) not in (None, "")
            )
        return (filled / possible) if possible else 1.0

    @staticmethod
    def _relationship_completeness(
        graph: PropertyGraph,
        ontology: OntologySchema,
    ) -> float:
        """Measure required object-predicate obligations from declared minCount."""

        required = [
            item
            for item in ontology.predicates
            if item.kind == "object" and item.cardinality.min_count >= 1
        ]
        if not required:
            return 1.0
        outgoing: Counter[tuple[str, str]] = Counter(
            (edge.source_id, resolution_key(edge.type)) for edge in graph.edges
        )
        satisfied = 0
        possible = 0
        for node in graph.nodes:
            labels = set(node.labels)
            for predicate in required:
                if predicate.domain not in labels:
                    continue
                possible += 1
                if (
                    outgoing[(node.node_id, resolution_key(predicate.name))]
                    >= predicate.cardinality.min_count
                ):
                    satisfied += 1
        return satisfied / possible if possible else 1.0

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
        object_predicates: dict[str, list[OntologyPredicate]] = defaultdict(list)
        datatype_predicates: dict[str, list[OntologyPredicate]] = defaultdict(list)
        for predicate in ontology.predicates:
            target = object_predicates if predicate.kind == "object" else datatype_predicates
            target[resolution_key(predicate.name)].append(predicate)
        domain_violations = 0
        checked = 0
        for edge in graph.edges:
            candidates = object_predicates.get(resolution_key(edge.type), [])
            if not candidates:
                continue
            checked += 1
            source_labels = node_labels.get(edge.source_id, set())
            target_labels = node_labels.get(edge.target_id, set())
            if not any(
                predicate.domain in source_labels
                and predicate.range in target_labels
                for predicate in candidates
            ):
                domain_violations += 1
        for node in graph.nodes:
            labels = node_labels.get(node.node_id, set())
            for key, value in node.properties.items():
                candidates = datatype_predicates.get(resolution_key(key), [])
                if not candidates or value in (None, ""):
                    continue
                # Generic graph metadata such as ``name`` may share a local key
                # with a declared predicate on another class.  It is not an
                # off-domain assertion unless it is a domain property here.
                applicable = [
                    predicate
                    for predicate in candidates
                    if predicate.domain in labels
                ]
                if not applicable and not _is_domain_property(key):
                    continue
                values = value if isinstance(value, list) else [value]
                for item in values:
                    checked += 1
                    if not applicable or not any(
                        self._matches_datatype(item, predicate.range)
                        for predicate in applicable
                    ):
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
                if predicate.domain not in labels:
                    continue
                cardinality_checked += 1
                if predicate.kind == "object":
                    count = outgoing.get(
                        (node.node_id, resolution_key(predicate.name)), 0
                    )
                else:
                    count = self._value_count(node.properties.get(predicate.name))
                below = count < predicate.cardinality.min_count
                above = predicate.cardinality.max_count is not None and count > predicate.cardinality.max_count
                if below or above:
                    cardinality_violations += 1
        cardinality_rate = (cardinality_violations / cardinality_checked) if cardinality_checked else 0.0
        return domain_violations, domain_range_rate, cardinality_violations, cardinality_rate

    @staticmethod
    def _value_count(value: object) -> int:
        if value in (None, ""):
            return 0
        return len(value) if isinstance(value, list) else 1

    @staticmethod
    def _matches_datatype(value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "object":
            return isinstance(value, dict)
        if expected == "array":
            return isinstance(value, list)
        if expected == "date":
            if isinstance(value, date) and not isinstance(value, datetime):
                return True
            if isinstance(value, str):
                try:
                    date.fromisoformat(value)
                except ValueError:
                    return False
                return True
            return False
        if expected == "datetime":
            if isinstance(value, datetime):
                return True
            if isinstance(value, str):
                try:
                    datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    return False
                return True
            return False
        return False

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
        if report.categorical_source_occurrence_coverage is not None:
            gates.append(
                SchemaQualityGate(
                    gate="categorical_occurrence_reconciliation",
                    metric="categorical_source_occurrence_coverage",
                    operator=">=",
                    threshold=(
                        settings.minimum_categorical_source_occurrence_coverage
                    ),
                    actual=report.categorical_source_occurrence_coverage,
                    passed=(
                        report.categorical_source_occurrence_coverage
                        >= settings.minimum_categorical_source_occurrence_coverage
                    ),
                )
            )
        if report.ontology_evaluated:
            gates.extend(
                [
                    SchemaQualityGate(
                        gate="property_completeness",
                        metric="property_completeness",
                        operator=">=",
                        threshold=settings.minimum_property_completeness,
                        actual=report.property_completeness,
                        passed=(
                            report.property_completeness
                            >= settings.minimum_property_completeness
                        ),
                    ),
                    SchemaQualityGate(
                        gate="relationship_completeness",
                        metric="relationship_completeness",
                        operator=">=",
                        threshold=settings.minimum_relationship_completeness,
                        actual=report.relationship_completeness,
                        passed=(
                            report.relationship_completeness
                            >= settings.minimum_relationship_completeness
                        ),
                    ),
                ]
            )
        # SHACL is a universal release condition. Keeping it outside the
        # ontology branch prevents a graph-only invocation from reporting a
        # successful schema evaluation merely because validation was omitted.
        gates.extend(
            [
                SchemaQualityGate(
                    gate="shacl_execution",
                    metric="shacl_executed",
                    operator=">=",
                    threshold=1.0,
                    actual=float(report.shacl_executed),
                    passed=report.shacl_executed,
                ),
                SchemaQualityGate(
                    gate="shacl_conformance",
                    metric="shacl_violation_count",
                    operator="<=",
                    threshold=float(settings.maximum_shacl_violation_count),
                    # Missing SHACL execution is an observed failure, not a
                    # vacuous zero-violation result.
                    actual=(
                        float(report.shacl_violation_count or 0)
                        if report.shacl_executed
                        else float(settings.maximum_shacl_violation_count + 1)
                    ),
                    passed=(
                        report.shacl_executed
                        and (report.shacl_violation_count or 0)
                        <= settings.maximum_shacl_violation_count
                    ),
                ),
            ]
        )
        return gates


__all__ = [
    "SchemaQualityEvaluator",
    "SchemaQualityGate",
    "SchemaQualityGateSettings",
    "SchemaQualityReport",
]
