"""Exact write/read contracts for the certifiable Neo4j workload backend."""

from __future__ import annotations

from typing import Any

import pytest

from kg_construction.domain.graph_objects import (
    PropertyEdge,
    PropertyGraph,
    PropertyNode,
)
from kg_construction.evaluation import property_graph_sha256
from kg_construction.evaluation.neo4j_workload_backend import (
    Neo4jGraphWorkloadBackend,
    Neo4jGraphWorkloadConfig,
)


class _Result:
    def __init__(
        self,
        rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.rows = list(rows or [])

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None

    def consume(self) -> None:
        return None


class _SnapshotStore:
    def __init__(self) -> None:
        self.metadata: dict[str, str] = {}
        self.nodes: dict[str, list[dict[str, str]]] = {}
        self.edges: dict[str, list[dict[str, str]]] = {}
        self.write_operations: list[str] = []


class _Transaction:
    def __init__(self, store: _SnapshotStore) -> None:
        self.store = store

    def run(self, query: str, **parameters: Any) -> _Result:
        graph_hash = str(parameters.get("graph_sha256", ""))
        if "metadata_json = $metadata_json" in query:
            self.store.metadata[graph_hash] = str(parameters["metadata_json"])
        elif "n.payload_json = row.payload_json" in query:
            self.store.nodes[graph_hash] = list(parameters["rows"])
        elif "r.payload_json = row.payload_json" in query:
            self.store.edges[graph_hash] = list(parameters["rows"])
        return _Result()


class _Session:
    def __init__(self, store: _SnapshotStore) -> None:
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute_write(self, operation, **kwargs: Any):
        self.store.write_operations.append(operation.__name__)
        return operation(_Transaction(self.store), **kwargs)

    def run(self, query: str, **parameters: Any) -> _Result:
        graph_hash = str(parameters.get("graph_sha256", ""))
        if "RETURN m.metadata_json AS payload" in query:
            payload = self.store.metadata.get(graph_hash)
            return _Result([] if payload is None else [{"payload": payload}])
        if "RETURN n.payload_json AS payload" in query:
            return _Result(
                [
                    {"payload": row["payload_json"]}
                    for row in self.store.nodes.get(graph_hash, [])
                ]
            )
        if "RETURN r.payload_json AS payload" in query:
            return _Result(
                [
                    {"payload": row["payload_json"]}
                    for row in self.store.edges.get(graph_hash, [])
                ]
            )
        if "RETURN count(n) AS count" in query:
            return _Result(
                [{"count": len(self.store.nodes.get(graph_hash, []))}]
            )
        if "RETURN count(r) AS count" in query:
            return _Result(
                [{"count": len(self.store.edges.get(graph_hash, []))}]
            )
        if "DETACH DELETE n" in query:
            self.store.nodes.clear()
            self.store.edges.clear()
        if "DETACH DELETE m" in query:
            self.store.metadata.clear()
        return _Result()


class _Driver:
    def __init__(self) -> None:
        self.store = _SnapshotStore()

    def session(self, **kwargs: object) -> _Session:
        del kwargs
        return _Session(self.store)


def _graph() -> PropertyGraph:
    return PropertyGraph(
        nodes=[
            PropertyNode(
                node_id="ada",
                labels=["Person"],
                properties={"name": "Ada", "source_ids": ["source-1"]},
            ),
            PropertyNode(
                node_id="acme",
                labels=["Organization"],
                properties={"name": "Acme"},
            ),
        ],
        edges=[
            PropertyEdge(
                edge_id="founded",
                source_id="ada",
                target_id="acme",
                type="FOUNDED",
                properties={"year": 1843},
            )
        ],
        metadata={"fixture": "neo4j-readback"},
    )


def test_neo4j_backend_reads_exact_graph_and_attests_every_family() -> None:
    driver = _Driver()
    backend = Neo4jGraphWorkloadBackend(
        Neo4jGraphWorkloadConfig(snapshot_namespace="unit_readback"),
        driver=driver,
    )
    graph = _graph()

    readback = backend.materialize(graph)

    assert property_graph_sha256(readback) == property_graph_sha256(graph)
    assert driver.store.write_operations[:2] == [
        "_ensure_snapshot_schema",
        "_write_snapshot",
    ]
    for family in (
        "sql_like",
        "nosql",
        "graphrag",
        "vector",
        "link_prediction",
        "recommendation",
        "reasoning",
    ):
        def return_family(value: str = family) -> str:
            return value

        assert (
            backend.execute(
                family,
                readback,
                return_family,
            )
            == family
        )
    assert backend.executed_families() == (
        "graphrag",
        "link_prediction",
        "nosql",
        "reasoning",
        "recommendation",
        "sql_like",
        "vector",
    )
    assert backend.certifiable
    assert property_graph_sha256(graph) in backend.snapshot_id(
        property_graph_sha256(graph)
    )

    backend.cleanup()
    assert not driver.store.nodes
    assert not driver.store.edges
    assert not driver.store.metadata


def test_neo4j_backend_rejects_changed_readback_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = _Driver()
    backend = Neo4jGraphWorkloadBackend(
        Neo4jGraphWorkloadConfig(snapshot_namespace="unit_tamper"),
        driver=driver,
    )
    graph = _graph()
    original_execute_write = _Session.execute_write

    def tampering_execute_write(self, operation, **kwargs):
        result = original_execute_write(self, operation, **kwargs)
        if "graph_sha256" in kwargs:
            graph_hash = str(kwargs["graph_sha256"])
            self.store.nodes[graph_hash][0]["payload_json"] = (
                '{"node_id":"tampered","labels":["Person"],"properties":{}}'
            )
        return result

    monkeypatch.setattr(_Session, "execute_write", tampering_execute_write)
    with pytest.raises(RuntimeError, match="changed graph semantics"):
        backend.materialize(graph)
