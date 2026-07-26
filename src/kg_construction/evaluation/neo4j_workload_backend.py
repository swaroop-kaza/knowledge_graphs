"""Canonical certifiable Neo4j boundary for intended-use workloads.

The workload algorithms remain deterministic and fixture-driven, but their
input is never the caller's in-memory object in a release run.  This backend
writes an immutable, namespaced snapshot to Neo4j, reads the complete portable
graph back, verifies its SHA-256, and performs a live count attestation before
each of the seven workload families executes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kg_construction.domain.graph_objects import (
    PropertyEdge,
    PropertyGraph,
    PropertyNode,
)
from kg_construction.evaluation.graph_bound_suite import (
    property_graph_sha256,
)
from kg_construction.evaluation.intended_use_metrics import (
    IntendedUse,
)
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured

_NAMESPACE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_NODE_LABEL = "KGEvaluationSnapshotNode"
_META_LABEL = "KGEvaluationSnapshot"
_EDGE_TYPE = "KG_EVALUATION_SNAPSHOT_EDGE"


@dataclass(frozen=True, slots=True)
class Neo4jGraphWorkloadConfig:
    """Connection and isolation contract for release evaluation snapshots."""

    uri: str = "bolt://localhost:7687"
    user: str | None = "neo4j"
    password: str | None = "certification"
    database: str | None = None
    snapshot_namespace: str = "kg_publication"

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("Neo4j workload URI must be non-empty")
        if not _NAMESPACE_RE.fullmatch(self.snapshot_namespace):
            raise ValueError(
                "snapshot_namespace must start with a letter and contain only "
                "letters, digits, underscores, or hyphens"
            )


class Neo4jGraphWorkloadBackend:
    """Materialize and attest exact graph snapshots through live Neo4j."""

    certifiable = True

    def __init__(
        self,
        config: Neo4jGraphWorkloadConfig | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        self.config = config or Neo4jGraphWorkloadConfig()
        self._driver = driver
        self._owns_driver = driver is None
        self._readbacks: dict[str, PropertyGraph] = {}
        self._executed_families: set[IntendedUse] = set()
        self.backend_id = (
            "neo4j-exact-readback:"
            f"{self.config.snapshot_namespace}:"
            f"{self.config.database or 'default'}"
        )

    def snapshot_id(self, graph_sha256: str) -> str:
        return f"{self.backend_id}:{graph_sha256}"

    def materialize(self, graph: PropertyGraph) -> PropertyGraph:
        """Write and read the graph, rejecting any byte-semantic difference."""

        expected_hash = property_graph_sha256(graph)
        cached = self._readbacks.get(expected_hash)
        if cached is not None:
            return cached.model_copy(deep=True)

        node_rows = [
            {
                "snapshot_key": self._snapshot_key(
                    expected_hash,
                    node.node_id,
                ),
                "node_id": node.node_id,
                "payload_json": _payload_json(node.model_dump(mode="json")),
            }
            for node in graph.nodes
        ]
        edge_rows = [
            {
                "snapshot_key": self._snapshot_key(
                    expected_hash,
                    edge.edge_id,
                ),
                "source_key": self._snapshot_key(
                    expected_hash,
                    edge.source_id,
                ),
                "target_key": self._snapshot_key(
                    expected_hash,
                    edge.target_id,
                ),
                "payload_json": _payload_json(edge.model_dump(mode="json")),
            }
            for edge in graph.edges
        ]
        session_kwargs = (
            {"database": self.config.database}
            if self.config.database
            else {}
        )
        with self._driver_instance().session(**session_kwargs) as session:
            session.execute_write(self._ensure_snapshot_schema)
            session.execute_write(
                self._write_snapshot,
                namespace=self.config.snapshot_namespace,
                graph_sha256=expected_hash,
                metadata_json=_payload_json(graph.metadata),
                node_rows=node_rows,
                edge_rows=edge_rows,
            )
            readback = self._read_snapshot(session, expected_hash)

        observed_hash = property_graph_sha256(readback)
        if observed_hash != expected_hash:
            raise RuntimeError(
                "Neo4j workload snapshot changed graph semantics: "
                f"expected={expected_hash}, observed={observed_hash}"
            )
        self._readbacks[expected_hash] = readback.model_copy(deep=True)
        return readback

    def execute(
        self,
        family: IntendedUse,
        graph: PropertyGraph,
        operation: Callable[[], Any],
    ) -> Any:
        """Attest the live snapshot immediately before a workload executes."""

        graph_hash = property_graph_sha256(graph)
        readback = self.materialize(graph)
        if property_graph_sha256(readback) != graph_hash:
            raise RuntimeError("Neo4j workload read-back hash changed before execution")
        observed_nodes, observed_edges = self._live_counts(graph_hash)
        if observed_nodes != len(graph.nodes) or observed_edges != len(graph.edges):
            raise RuntimeError(
                "Neo4j workload snapshot count mismatch before "
                f"{family}: expected=({len(graph.nodes)}, {len(graph.edges)}), "
                f"observed=({observed_nodes}, {observed_edges})"
            )
        value = operation()
        self._executed_families.add(family)
        return value

    def executed_families(self) -> tuple[IntendedUse, ...]:
        return tuple(sorted(self._executed_families))

    def cleanup(self) -> None:
        """Remove only this backend's namespaced evaluation snapshots."""

        session_kwargs = (
            {"database": self.config.database}
            if self.config.database
            else {}
        )
        with self._driver_instance().session(**session_kwargs) as session:
            session.run(
                f"MATCH (n:{_NODE_LABEL} {{namespace: $namespace}}) "
                "DETACH DELETE n",
                namespace=self.config.snapshot_namespace,
            ).consume()
            session.run(
                f"MATCH (m:{_META_LABEL} {{namespace: $namespace}}) "
                "DETACH DELETE m",
                namespace=self.config.snapshot_namespace,
            ).consume()
        self._readbacks.clear()
        self._executed_families.clear()

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
                "Neo4j is mandatory for certifiable graph workloads"
            ) from exc
        auth = None
        if self.config.user is not None or self.config.password is not None:
            auth = (
                self.config.user or "",
                self.config.password or "",
            )
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=auth,
        )
        return self._driver

    def _snapshot_key(self, graph_sha256: str, object_id: str) -> str:
        return (
            f"{self.config.snapshot_namespace}:"
            f"{graph_sha256}:{object_id}"
        )

    @staticmethod
    def _ensure_snapshot_schema(tx: Any) -> None:
        """Create snapshot constraints in a schema-only transaction."""

        tx.run(
            f"CREATE CONSTRAINT kg_evaluation_snapshot_node_key IF NOT EXISTS "
            f"FOR (n:{_NODE_LABEL}) REQUIRE n.snapshot_key IS UNIQUE"
        ).consume()
        tx.run(
            f"CREATE CONSTRAINT kg_evaluation_snapshot_meta_key IF NOT EXISTS "
            f"FOR (m:{_META_LABEL}) REQUIRE m.snapshot_id IS UNIQUE"
        ).consume()

    @staticmethod
    def _write_snapshot(
        tx: Any,
        *,
        namespace: str,
        graph_sha256: str,
        metadata_json: str,
        node_rows: list[dict[str, str]],
        edge_rows: list[dict[str, str]],
    ) -> None:
        tx.run(
            f"MERGE (m:{_META_LABEL} {{snapshot_id: $snapshot_id}}) "
            "SET m.namespace = $namespace, "
            "m.graph_sha256 = $graph_sha256, "
            "m.metadata_json = $metadata_json",
            snapshot_id=f"{namespace}:{graph_sha256}",
            namespace=namespace,
            graph_sha256=graph_sha256,
            metadata_json=metadata_json,
        ).consume()
        tx.run(
            f"UNWIND $rows AS row "
            f"MERGE (n:{_NODE_LABEL} {{snapshot_key: row.snapshot_key}}) "
            "SET n.namespace = $namespace, "
            "n.graph_sha256 = $graph_sha256, "
            "n.node_id = row.node_id, "
            "n.payload_json = row.payload_json",
            rows=node_rows,
            namespace=namespace,
            graph_sha256=graph_sha256,
        ).consume()
        tx.run(
            f"UNWIND $rows AS row "
            f"MATCH (s:{_NODE_LABEL} {{snapshot_key: row.source_key}}) "
            f"MATCH (t:{_NODE_LABEL} {{snapshot_key: row.target_key}}) "
            f"MERGE (s)-[r:{_EDGE_TYPE} "
            "{snapshot_key: row.snapshot_key}]->(t) "
            "SET r.namespace = $namespace, "
            "r.graph_sha256 = $graph_sha256, "
            "r.payload_json = row.payload_json",
            rows=edge_rows,
            namespace=namespace,
            graph_sha256=graph_sha256,
        ).consume()

    def _read_snapshot(
        self,
        session: Any,
        graph_sha256: str,
    ) -> PropertyGraph:
        metadata_record = session.run(
            f"MATCH (m:{_META_LABEL} "
            "{namespace: $namespace, graph_sha256: $graph_sha256}) "
            "RETURN m.metadata_json AS payload",
            namespace=self.config.snapshot_namespace,
            graph_sha256=graph_sha256,
        ).single()
        if metadata_record is None:
            raise RuntimeError("Neo4j did not return snapshot metadata")
        node_records = session.run(
            f"MATCH (n:{_NODE_LABEL} "
            "{namespace: $namespace, graph_sha256: $graph_sha256}) "
            "RETURN n.payload_json AS payload ORDER BY n.node_id",
            namespace=self.config.snapshot_namespace,
            graph_sha256=graph_sha256,
        )
        edge_records = session.run(
            f"MATCH ()-[r:{_EDGE_TYPE} "
            "{namespace: $namespace, graph_sha256: $graph_sha256}]->() "
            "RETURN r.payload_json AS payload ORDER BY r.snapshot_key",
            namespace=self.config.snapshot_namespace,
            graph_sha256=graph_sha256,
        )
        return PropertyGraph(
            nodes=[
                PropertyNode.model_validate(
                    _payload_object(record["payload"])
                )
                for record in node_records
            ],
            edges=[
                PropertyEdge.model_validate(
                    _payload_object(record["payload"])
                )
                for record in edge_records
            ],
            metadata=_payload_object(metadata_record["payload"]),
        )

    def _live_counts(self, graph_sha256: str) -> tuple[int, int]:
        session_kwargs = (
            {"database": self.config.database}
            if self.config.database
            else {}
        )
        with self._driver_instance().session(**session_kwargs) as session:
            node_record = session.run(
                f"MATCH (n:{_NODE_LABEL} "
                "{namespace: $namespace, graph_sha256: $graph_sha256}) "
                "RETURN count(n) AS count",
                namespace=self.config.snapshot_namespace,
                graph_sha256=graph_sha256,
            ).single()
            edge_record = session.run(
                f"MATCH ()-[r:{_EDGE_TYPE} "
                "{namespace: $namespace, graph_sha256: $graph_sha256}]->() "
                "RETURN count(r) AS count",
                namespace=self.config.snapshot_namespace,
                graph_sha256=graph_sha256,
            ).single()
        if node_record is None or edge_record is None:
            raise RuntimeError("Neo4j did not return workload snapshot counts")
        return int(node_record["count"]), int(edge_record["count"])


def _payload_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _payload_object(value: object) -> dict[str, Any]:
    if not isinstance(value, str):
        raise RuntimeError("Neo4j snapshot payload must be a JSON string")
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Neo4j snapshot payload must contain a JSON object")
    return {str(key): item for key, item in payload.items()}


__all__ = [
    "Neo4jGraphWorkloadBackend",
    "Neo4jGraphWorkloadConfig",
]
