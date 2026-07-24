"""Production graph database ingestion adapters."""

from __future__ import annotations

import json
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode, RDFGraph
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured
from kg_construction.infrastructure.production.dead_letter import DeadLetterQueue
from kg_construction.infrastructure.production.retry import (
    BatchRetryPolicy,
    RetryExhaustedError,
)

JsonDict = dict[str, Any]
_SAFE_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]")


def _safe_identifier(value: str, *, fallback: str) -> str:
    cleaned = _SAFE_IDENTIFIER_RE.sub("_", value).strip("_")
    if not cleaned:
        cleaned = fallback
    if not cleaned[0].isalpha() and cleaned[0] != "_":
        cleaned = f"_{cleaned}"
    return cleaned


def _json_value(value: object) -> object:
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _json_dict(value: JsonDict) -> JsonDict:
    normalized = _json_value(value)
    return normalized if isinstance(normalized, dict) else {}


def _chunks(items: Sequence[Any], batch_size: int) -> Iterable[list[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def _consume(result: Any) -> None:
    consume = getattr(result, "consume", None)
    if callable(consume):
        consume()


def _graph_payload(graph: PropertyGraph | RDFGraph) -> JsonDict:
    return {"graph": graph.to_dict()}


class IngestionWriteReport(BaseModel):
    """Write report for a graph ingestion operation."""

    model_config = ConfigDict(extra="forbid")

    store_name: str
    graph_model: str
    node_count: int = 0
    edge_count: int = 0
    triple_count: int = 0
    statement_count: int = 0
    transaction_count: int = 0
    attempted_batches: int = 0
    attempt_count: int = 1
    retry_count: int = 0
    dead_letter_id: str | None = None
    metadata: JsonDict = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Neo4jIngestionConfig:
    """Configuration for Neo4j transactional property graph ingestion."""

    uri: str = "bolt://localhost:7687"
    user: str | None = None
    password: str | None = None
    database: str | None = None
    batch_size: int = 500
    ensure_constraints: bool = True
    raise_on_failure: bool = True
    enable_neosemantics: bool = False
    neosemantics_format: str = "Turtle"


@dataclass(frozen=True, slots=True)
class LadybugIngestionConfig:
    """Configuration for Ladybug embedded property-graph ingestion."""

    database_path: str = "data/processed/ladybug_graph"
    node_table: str = "KGNode"
    edge_table: str = "KGEdge"
    batch_size: int = 500
    create_schema: bool = True
    use_manual_transactions: bool = True
    checkpoint_after_write: bool = False
    raise_on_failure: bool = True
    bulk_import_threshold: int = 10000


@dataclass(frozen=True, slots=True)
class _WriteStats:
    statement_count: int = 0
    transaction_count: int = 1
    attempted_batches: int = 0


class Neo4jIngestionAdapter:
    """Write property graphs to Neo4j using managed write transactions."""

    name = "neo4j_ingestion_adapter"

    def __init__(
        self,
        config: Neo4jIngestionConfig | None = None,
        *,
        driver: Any | None = None,
        retry_policy: BatchRetryPolicy | None = None,
        dead_letter_queue: DeadLetterQueue | None = None,
    ) -> None:
        self.config = config or Neo4jIngestionConfig()
        self._driver = driver
        self._owns_driver = driver is None
        self.retry_policy = retry_policy or BatchRetryPolicy()
        self.dead_letter_queue = dead_letter_queue

    def write_property_graph(self, graph: PropertyGraph) -> IngestionWriteReport:
        operation_name = "neo4j_property_graph_write"
        try:
            outcome = self.retry_policy.run(
                lambda: self._write_property_graph_once(graph),
                operation_name=operation_name,
            )
            return outcome.value.model_copy(
                update={
                    "attempt_count": outcome.attempts,
                    "retry_count": outcome.retry_count,
                }
            )
        except RetryExhaustedError as exc:
            record = self._record_dead_letter(
                graph,
                operation_name=operation_name,
                error=exc.last_error,
                attempt_count=exc.attempts,
            )
            if self.config.raise_on_failure:
                raise exc.last_error from exc
            return IngestionWriteReport(
                store_name=self.name,
                graph_model="property_graph",
                node_count=len(graph.nodes),
                edge_count=len(graph.edges),
                dead_letter_id=record.dead_letter_id if record else None,
                attempt_count=exc.attempts,
                retry_count=max(exc.attempts - 1, 0),
                metadata={"failed": True, "error_type": type(exc.last_error).__name__},
            )

    def write_rdf_graph(self, graph: RDFGraph) -> IngestionWriteReport:
        if not self.config.enable_neosemantics:
            raise AdapterNotConfigured(
                "Neo4j RDF ingestion requires enable_neosemantics=True and the neosemantics plugin. "
                f"Triples: {len(graph.triples)}"
            )
        operation_name = "neo4j_rdf_neosemantics_write"
        outcome = self.retry_policy.run(
            lambda: self._write_rdf_graph_once(graph),
            operation_name=operation_name,
        )
        return outcome.value.model_copy(
            update={"attempt_count": outcome.attempts, "retry_count": outcome.retry_count}
        )

    def close(self) -> None:
        close = getattr(self._driver, "close", None)
        if self._owns_driver and callable(close):
            close()

    def _driver_instance(self) -> Any:
        if self._driver is not None:
            return self._driver
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise AdapterNotConfigured(
                "Neo4j is a mandatory dependency. Run the repository provisioning "
                "step or inject a configured driver."
            ) from exc
        auth = None
        if self.config.user is not None or self.config.password is not None:
            auth = (self.config.user or "", self.config.password or "")
        self._driver = GraphDatabase.driver(self.config.uri, auth=auth)
        return self._driver

    def _write_property_graph_once(self, graph: PropertyGraph) -> IngestionWriteReport:
        driver = self._driver_instance()
        session_kwargs = {"database": self.config.database} if self.config.database else {}
        with driver.session(**session_kwargs) as session:
            stats = session.execute_write(self._write_property_graph_tx, graph)
        return IngestionWriteReport(
            store_name=self.name,
            graph_model="property_graph",
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            statement_count=stats.statement_count,
            transaction_count=stats.transaction_count,
            attempted_batches=stats.attempted_batches,
            metadata={
                "database": self.config.database,
                "batch_size": self.config.batch_size,
                "ensure_constraints": self.config.ensure_constraints,
            },
        )

    def _write_rdf_graph_once(self, graph: RDFGraph) -> IngestionWriteReport:
        driver = self._driver_instance()
        rdf_payload = graph.serialize_turtle()
        session_kwargs = {"database": self.config.database} if self.config.database else {}
        with driver.session(**session_kwargs) as session:
            session.execute_write(self._write_rdf_neosemantics_tx, rdf_payload)
        return IngestionWriteReport(
            store_name=self.name,
            graph_model="rdf_graph",
            triple_count=len(graph.triples),
            statement_count=1,
            transaction_count=1,
            metadata={
                "rdf_import": "neosemantics",
                "format": self.config.neosemantics_format,
                "database": self.config.database,
            },
        )

    def _write_rdf_neosemantics_tx(self, tx: Any, rdf_payload: str) -> _WriteStats:
        query = "CALL n10s.rdf.import.inline($rdf, $format, $config)"
        _consume(
            tx.run(
                query,
                rdf=rdf_payload,
                format=self.config.neosemantics_format,
                config={"handleVocabUris": "KEEP", "keepLangTag": True},
            )
        )
        return _WriteStats(statement_count=1, transaction_count=1, attempted_batches=1)

    def _write_property_graph_tx(self, tx: Any, graph: PropertyGraph) -> _WriteStats:
        statement_count = 0
        attempted_batches = 0
        if self.config.ensure_constraints:
            for label in self._node_labels(graph.nodes):
                query = f"CREATE CONSTRAINT kg_{label}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
                _consume(tx.run(query))
                statement_count += 1

        for labels, nodes in self._nodes_by_label_set(graph.nodes).items():
            label_clause = ":".join(labels)
            query = (
                f"UNWIND $rows AS row MERGE (n:{label_clause} {{id: row.id}}) "
                "SET n += row.properties"
            )
            for batch in _chunks(nodes, self.config.batch_size):
                rows = [self._node_row(node) for node in batch]
                _consume(tx.run(query, rows=rows))
                statement_count += 1
                attempted_batches += 1

        for rel_type, edges in self._edges_by_type(graph.edges).items():
            query = (
                "UNWIND $rows AS row "
                "MATCH (s {id: row.source_id}) "
                "MATCH (t {id: row.target_id}) "
                f"MERGE (s)-[r:{rel_type} {{id: row.id}}]->(t) "
                "SET r += row.properties"
            )
            for batch in _chunks(edges, self.config.batch_size):
                rows = [self._edge_row(edge) for edge in batch]
                _consume(tx.run(query, rows=rows))
                statement_count += 1
                attempted_batches += 1

        return _WriteStats(
            statement_count=statement_count,
            attempted_batches=attempted_batches,
            transaction_count=1,
        )

    def _record_dead_letter(
        self,
        graph: PropertyGraph,
        *,
        operation_name: str,
        error: BaseException,
        attempt_count: int,
    ):
        if self.dead_letter_queue is None:
            return None
        return self.dead_letter_queue.record_failure(
            stage="graph_store_ingestion",
            store_name=self.name,
            operation_name=operation_name,
            graph_model="property_graph",
            error=error,
            attempt_count=attempt_count,
            payload=_graph_payload(graph),
            metadata={"uri": self.config.uri, "database": self.config.database},
        )

    @staticmethod
    def _node_labels(nodes: list[PropertyNode]) -> list[str]:
        labels = {
            _safe_identifier(label, fallback="Entity")
            for node in nodes
            for label in (node.labels or ["Entity"])
        }
        return sorted(labels)

    @staticmethod
    def _nodes_by_label_set(nodes: list[PropertyNode]) -> dict[tuple[str, ...], list[PropertyNode]]:
        grouped: dict[tuple[str, ...], list[PropertyNode]] = defaultdict(list)
        for node in nodes:
            labels = tuple(
                sorted({_safe_identifier(label, fallback="Entity") for label in (node.labels or ["Entity"])})
            )
            grouped[labels].append(node)
        return dict(grouped)

    @staticmethod
    def _edges_by_type(edges: list[PropertyEdge]) -> dict[str, list[PropertyEdge]]:
        grouped: dict[str, list[PropertyEdge]] = defaultdict(list)
        for edge in edges:
            grouped[_safe_identifier(edge.type, fallback="RELATED_TO").upper()].append(edge)
        return dict(grouped)

    @staticmethod
    def _node_row(node: PropertyNode) -> JsonDict:
        return {
            "id": node.node_id,
            "properties": _json_dict(
                {
                    **node.properties,
                    "id": node.node_id,
                    "labels": node.labels,
                }
            ),
        }

    @staticmethod
    def _edge_row(edge: PropertyEdge) -> JsonDict:
        return {
            "id": edge.edge_id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "properties": _json_dict(
                {
                    **edge.properties,
                    "id": edge.edge_id,
                    "type": edge.type,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                }
            ),
        }


class LadybugIngestionAdapter:
    """Write property graphs to Ladybug through a generic typed graph schema."""

    name = "ladybug_ingestion_adapter"

    def __init__(
        self,
        config: LadybugIngestionConfig | None = None,
        *,
        connection: Any | None = None,
        retry_policy: BatchRetryPolicy | None = None,
        dead_letter_queue: DeadLetterQueue | None = None,
    ) -> None:
        self.config = config or LadybugIngestionConfig()
        self._connection = connection
        self.retry_policy = retry_policy or BatchRetryPolicy()
        self.dead_letter_queue = dead_letter_queue

    def write_property_graph(self, graph: PropertyGraph) -> IngestionWriteReport:
        operation_name = "ladybug_property_graph_write"
        try:
            outcome = self.retry_policy.run(
                lambda: self._write_property_graph_once(graph),
                operation_name=operation_name,
            )
            return outcome.value.model_copy(
                update={
                    "attempt_count": outcome.attempts,
                    "retry_count": outcome.retry_count,
                }
            )
        except RetryExhaustedError as exc:
            record = self._record_dead_letter(
                graph,
                operation_name=operation_name,
                error=exc.last_error,
                attempt_count=exc.attempts,
            )
            if self.config.raise_on_failure:
                raise exc.last_error from exc
            return IngestionWriteReport(
                store_name=self.name,
                graph_model="property_graph",
                node_count=len(graph.nodes),
                edge_count=len(graph.edges),
                dead_letter_id=record.dead_letter_id if record else None,
                attempt_count=exc.attempts,
                retry_count=max(exc.attempts - 1, 0),
                metadata={"failed": True, "error_type": type(exc.last_error).__name__},
            )

    def write_rdf_graph(self, graph: RDFGraph) -> IngestionWriteReport:
        raise AdapterNotConfigured(
            "Ladybug ingestion currently targets property graphs. "
            f"RDF triples received: {len(graph.triples)}"
        )

    def bulk_import_property_graph(self, graph: PropertyGraph, output_dir: str | Path | None = None) -> IngestionWriteReport:
        """Write large graphs through CSV files and Ladybug ``COPY`` statements."""

        directory = (
            Path(output_dir)
            if output_dir
            else Path(tempfile.mkdtemp(prefix="kg_ladybug_bulk_"))
        )
        directory.mkdir(parents=True, exist_ok=True)
        node_path = directory / f"{self._node_table}.csv"
        edge_path = directory / f"{self._edge_table}.csv"
        self._write_bulk_csv(graph, node_path, edge_path)
        connection = self._connection_instance()
        statement_count = 0
        if self.config.create_schema:
            statement_count += self._ensure_schema(connection)
        self._execute(connection, f"COPY {self._node_table} FROM '{node_path}' (HEADER=true)")
        self._execute(connection, f"COPY {self._edge_table} FROM '{edge_path}' (HEADER=true)")
        statement_count += 2
        return IngestionWriteReport(
            store_name=self.name,
            graph_model="property_graph",
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            statement_count=statement_count,
            attempted_batches=2,
            transaction_count=2,
            metadata={
                "bulk_import": True,
                "node_csv": str(node_path),
                "edge_csv": str(edge_path),
                "database_path": self.config.database_path,
            },
        )

    def _connection_instance(self) -> Any:
        if self._connection is not None:
            return self._connection
        try:
            import ladybug as lb
        except ImportError as exc:
            raise AdapterNotConfigured(
                "Ladybug is a mandatory dependency. Run the repository provisioning "
                "step or inject a configured connection."
            ) from exc
        database = lb.Database(self.config.database_path)
        self._connection = lb.Connection(database)
        return self._connection

    def _write_property_graph_once(self, graph: PropertyGraph) -> IngestionWriteReport:
        connection = self._connection_instance()
        statement_count = 0
        if self.config.create_schema:
            statement_count += self._ensure_schema(connection)
        if self.config.use_manual_transactions:
            self._execute(connection, "BEGIN TRANSACTION")
            statement_count += 1
        try:
            for node in graph.nodes:
                self._execute(connection, self._node_merge_query(), self._node_parameters(node))
                statement_count += 1
            for edge in graph.edges:
                self._execute(connection, self._edge_merge_query(), self._edge_parameters(edge))
                statement_count += 1
            if self.config.use_manual_transactions:
                self._execute(connection, "COMMIT")
                statement_count += 1
        except Exception:
            if self.config.use_manual_transactions:
                self._execute(connection, "ROLLBACK")
            raise
        if self.config.checkpoint_after_write:
            self._execute(connection, "CHECKPOINT")
            statement_count += 1

        item_count = len(graph.nodes) + len(graph.edges)
        return IngestionWriteReport(
            store_name=self.name,
            graph_model="property_graph",
            node_count=len(graph.nodes),
            edge_count=len(graph.edges),
            statement_count=statement_count,
            transaction_count=1 if self.config.use_manual_transactions else item_count,
            attempted_batches=ceil(item_count / self.config.batch_size) if item_count else 0,
            metadata={
                "database_path": self.config.database_path,
                "node_table": self._node_table,
                "edge_table": self._edge_table,
                "batch_size": self.config.batch_size,
            },
        )

    def _ensure_schema(self, connection: Any) -> int:
        statements = [
            (
                f"CREATE NODE TABLE {self._node_table}("
                "id STRING PRIMARY KEY, labels STRING, properties STRING)"
            ),
            (
                f"CREATE REL TABLE {self._edge_table}("
                f"FROM {self._node_table} TO {self._node_table}, "
                "id STRING, type STRING, properties STRING)"
            ),
        ]
        count = 0
        for statement in statements:
            try:
                self._execute(connection, statement)
            except Exception as exc:
                message = str(exc).casefold()
                if "exist" not in message and "already" not in message:
                    raise
            count += 1
        return count

    def _record_dead_letter(
        self,
        graph: PropertyGraph,
        *,
        operation_name: str,
        error: BaseException,
        attempt_count: int,
    ):
        if self.dead_letter_queue is None:
            return None
        return self.dead_letter_queue.record_failure(
            stage="graph_store_ingestion",
            store_name=self.name,
            operation_name=operation_name,
            graph_model="property_graph",
            error=error,
            attempt_count=attempt_count,
            payload=_graph_payload(graph),
            metadata={"database_path": self.config.database_path},
        )

    @property
    def _node_table(self) -> str:
        return _safe_identifier(self.config.node_table, fallback="KGNode")

    @property
    def _edge_table(self) -> str:
        return _safe_identifier(self.config.edge_table, fallback="KGEdge")

    def _node_merge_query(self) -> str:
        return (
            f"MERGE (n:{self._node_table} {{id: $id}}) "
            "ON CREATE SET n.labels = $labels, n.properties = $properties "
            "ON MATCH SET n.labels = $labels, n.properties = $properties"
        )

    def _edge_merge_query(self) -> str:
        return (
            f"MATCH (s:{self._node_table} {{id: $source_id}}), "
            f"(t:{self._node_table} {{id: $target_id}}) "
            f"MERGE (s)-[r:{self._edge_table} {{id: $id}}]->(t) "
            "ON CREATE SET r.type = $type, r.properties = $properties "
            "ON MATCH SET r.type = $type, r.properties = $properties"
        )

    @staticmethod
    def _node_parameters(node: PropertyNode) -> JsonDict:
        return {
            "id": node.node_id,
            "labels": json.dumps(node.labels, sort_keys=True),
            "properties": json.dumps(_json_dict(node.properties), sort_keys=True),
        }

    @staticmethod
    def _edge_parameters(edge: PropertyEdge) -> JsonDict:
        return {
            "id": edge.edge_id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "type": edge.type,
            "properties": json.dumps(_json_dict(edge.properties), sort_keys=True),
        }

    @staticmethod
    def _execute(connection: Any, query: str, parameters: JsonDict | None = None) -> Any:
        if parameters is None:
            return connection.execute(query)
        try:
            return connection.execute(query, parameters=parameters)
        except TypeError:
            return connection.execute(query, parameters)

    @staticmethod
    def _write_bulk_csv(graph: PropertyGraph, node_path: Path, edge_path: Path) -> None:
        import csv

        with node_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "labels", "properties"])
            writer.writeheader()
            for node in graph.nodes:
                writer.writerow(LadybugIngestionAdapter._node_parameters(node))
        with edge_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "source_id", "target_id", "type", "properties"])
            writer.writeheader()
            for edge in graph.edges:
                writer.writerow(LadybugIngestionAdapter._edge_parameters(edge))


# Compatibility aliases for callers migrating from the database's former name.
# The release registry, configuration, documentation, and certification surface
# use Ladybug exclusively.
KuzuIngestionConfig = LadybugIngestionConfig
KuzuIngestionAdapter = LadybugIngestionAdapter


__all__ = [
    "IngestionWriteReport",
    "LadybugIngestionAdapter",
    "LadybugIngestionConfig",
    "KuzuIngestionAdapter",
    "KuzuIngestionConfig",
    "Neo4jIngestionAdapter",
    "Neo4jIngestionConfig",
]
