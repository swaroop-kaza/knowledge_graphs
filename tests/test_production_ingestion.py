from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from kg_construction.application.registry import default_registry
from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.infrastructure.graph_stores import (
    KuzuIngestionAdapter,
    KuzuIngestionConfig,
    Neo4jIngestionAdapter,
    Neo4jIngestionConfig,
)
from kg_construction.infrastructure.production import (
    BatchRetryPolicy,
    DeadLetterQueue,
    DeadLetterReplayTool,
    RetryPolicyConfig,
)
from kg_construction.interfaces.cli.replay_dead_letters import main as replay_dead_letters_main


class TransientError(RuntimeError):
    pass


class FakeResult:
    def consume(self) -> None:
        return None


class FakeNeo4jTx:
    def __init__(self) -> None:
        self.runs: list[tuple[str, dict[str, object]]] = []

    def run(self, query: str, **parameters: object) -> FakeResult:
        self.runs.append((query, parameters))
        return FakeResult()


class FakeNeo4jSession:
    def __init__(self, driver: FakeNeo4jDriver) -> None:
        self.driver = driver
        self.tx = FakeNeo4jTx()

    def __enter__(self) -> FakeNeo4jSession:
        return self

    def __exit__(self, _exc_type: object, exc: object, _tb: object) -> None:
        return None

    def execute_write(self, callback: object, graph: PropertyGraph) -> object:
        self.driver.execute_write_calls += 1
        if self.driver.failures_before_success > 0:
            self.driver.failures_before_success -= 1
            raise TransientError("temporary service unavailable")
        result = callback(self.tx, graph)  # type: ignore[misc]
        self.driver.runs.extend(self.tx.runs)
        return result


class FakeNeo4jDriver:
    def __init__(self, *, failures_before_success: int = 0) -> None:
        self.failures_before_success = failures_before_success
        self.execute_write_calls = 0
        self.session_databases: list[str | None] = []
        self.runs: list[tuple[str, dict[str, object]]] = []

    def session(self, **kwargs: object) -> FakeNeo4jSession:
        database = kwargs.get("database")
        self.session_databases.append(database if isinstance(database, str) else None)
        return FakeNeo4jSession(self)


class FakeKuzuConnection:
    def __init__(self, *, fail_on_query_fragment: str | None = None) -> None:
        self.fail_on_query_fragment = fail_on_query_fragment
        self.executions: list[tuple[str, dict[str, object] | None]] = []

    def execute(self, query: str, parameters: dict[str, object] | None = None) -> FakeResult:
        self.executions.append((query, parameters))
        if self.fail_on_query_fragment and self.fail_on_query_fragment in query:
            raise TransientError("temporary Kuzu write timeout")
        return FakeResult()


class FakeWriter:
    name = "neo4j_ingestion_adapter"

    def __init__(self) -> None:
        self.graphs: list[PropertyGraph] = []

    def write_property_graph(self, graph: PropertyGraph) -> object:
        self.graphs.append(graph)
        return {"written": len(graph.nodes)}

    def write_rdf_graph(self, graph: object) -> object:
        _ = graph
        raise AssertionError("RDF replay was not expected")


def sample_graph() -> PropertyGraph:
    return PropertyGraph(
        nodes=[
            PropertyNode(
                node_id="svc:recommendation",
                labels=["Entity", "System"],
                properties={"name": "Recommendation API"},
            ),
            PropertyNode(
                node_id="incident:1042",
                labels=["Entity", "Incident"],
                properties={"name": "INCIDENT-1042"},
            ),
        ],
        edges=[
            PropertyEdge(
                edge_id="edge:affected",
                source_id="incident:1042",
                target_id="svc:recommendation",
                type="AFFECTED",
                properties={"confidence": 0.97},
            )
        ],
    )


class ProductionIngestionTests(unittest.TestCase):
    def test_neo4j_adapter_uses_managed_write_transactions_and_parameterized_batches(self) -> None:
        driver = FakeNeo4jDriver()
        adapter = Neo4jIngestionAdapter(
            Neo4jIngestionConfig(database="neo4j", batch_size=10),
            driver=driver,
            retry_policy=BatchRetryPolicy(RetryPolicyConfig(max_attempts=1, sleep=False)),
        )

        report = adapter.write_property_graph(sample_graph())

        self.assertEqual(report.store_name, "neo4j_ingestion_adapter")
        self.assertEqual(report.node_count, 2)
        self.assertEqual(report.edge_count, 1)
        self.assertEqual(report.transaction_count, 1)
        self.assertEqual(driver.execute_write_calls, 1)
        self.assertEqual(driver.session_databases, ["neo4j"])
        self.assertTrue(any("UNWIND $rows AS row MERGE" in query for query, _ in driver.runs))
        self.assertTrue(any("MERGE (s)-[r:AFFECTED" in query for query, _ in driver.runs))
        row_batches = [params["rows"] for _, params in driver.runs if "rows" in params]
        self.assertTrue(row_batches)

    def test_retry_policy_retries_transient_write_failures(self) -> None:
        driver = FakeNeo4jDriver(failures_before_success=1)
        adapter = Neo4jIngestionAdapter(
            Neo4jIngestionConfig(ensure_constraints=False),
            driver=driver,
            retry_policy=BatchRetryPolicy(RetryPolicyConfig(max_attempts=2, sleep=False)),
        )

        report = adapter.write_property_graph(sample_graph())

        self.assertEqual(report.attempt_count, 2)
        self.assertEqual(report.retry_count, 1)
        self.assertEqual(driver.execute_write_calls, 2)

    def test_dead_letter_queue_records_exhausted_ingestion_failures_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = DeadLetterQueue(Path(tmpdir) / "dead_letters.jsonl")
            adapter = Neo4jIngestionAdapter(
                Neo4jIngestionConfig(raise_on_failure=False),
                driver=FakeNeo4jDriver(failures_before_success=3),
                retry_policy=BatchRetryPolicy(RetryPolicyConfig(max_attempts=2, sleep=False)),
                dead_letter_queue=queue,
            )

            failed_report = adapter.write_property_graph(sample_graph())
            records = queue.load_records(status="pending")
            writer = FakeWriter()
            replay_report = DeadLetterReplayTool(queue).replay(
                {"neo4j_ingestion_adapter": writer}
            )

            replayed_records = queue.load_records(status="replayed")

        self.assertIsNotNone(failed_report.dead_letter_id)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].attempt_count, 2)
        self.assertEqual(replay_report.replayed, 1)
        self.assertEqual(len(writer.graphs), 1)
        self.assertEqual(len(replayed_records), 1)

    def test_dead_letter_replay_dry_run_does_not_mutate_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            queue = DeadLetterQueue(Path(tmpdir) / "dead_letters.jsonl")
            queue.record_failure(
                stage="graph_store_ingestion",
                store_name="neo4j_ingestion_adapter",
                operation_name="neo4j_property_graph_write",
                graph_model="property_graph",
                error=TransientError("temporary"),
                attempt_count=1,
                payload={"graph": sample_graph().to_dict()},
            )

            report = DeadLetterReplayTool(queue).replay({}, dry_run=True)
            pending = queue.load_records(status="pending")

        self.assertEqual(report.attempted, 1)
        self.assertEqual(report.skipped, 1)
        self.assertEqual(len(pending), 1)

    def test_dead_letter_replay_cli_supports_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dead_letters.jsonl"
            DeadLetterQueue(path).record_failure(
                stage="graph_store_ingestion",
                store_name="kuzu_ingestion_adapter",
                operation_name="kuzu_property_graph_write",
                graph_model="property_graph",
                error=TransientError("temporary"),
                attempt_count=1,
                payload={"graph": sample_graph().to_dict()},
            )

            code = replay_dead_letters_main(
                [
                    "--dead-letter-path",
                    str(path),
                    "--store",
                    "kuzu_ingestion_adapter",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)

    def test_kuzu_adapter_creates_schema_and_writes_inside_manual_transaction(self) -> None:
        connection = FakeKuzuConnection()
        adapter = KuzuIngestionAdapter(
            KuzuIngestionConfig(database_path="memory", batch_size=10),
            connection=connection,
            retry_policy=BatchRetryPolicy(RetryPolicyConfig(max_attempts=1, sleep=False)),
        )

        report = adapter.write_property_graph(sample_graph())
        queries = [query for query, _ in connection.executions]

        self.assertEqual(report.store_name, "kuzu_ingestion_adapter")
        self.assertEqual(report.node_count, 2)
        self.assertEqual(report.edge_count, 1)
        self.assertIn("CREATE NODE TABLE KGNode", queries[0])
        self.assertIn("CREATE REL TABLE KGEdge", queries[1])
        self.assertIn("BEGIN TRANSACTION", queries)
        self.assertIn("COMMIT", queries)
        self.assertTrue(any("MERGE (n:KGNode" in query for query in queries))
        self.assertTrue(any("MERGE (s)-[r:KGEdge" in query for query in queries))
        parameterized = [params for _, params in connection.executions if params]
        self.assertTrue(parameterized)

    def test_registry_includes_production_ingestion_capabilities(self) -> None:
        registry = default_registry()
        for name in {
            "batch_retry_policy",
            "dead_letter_queue",
            "dead_letter_replay_tool",
            "kuzu_ingestion_adapter",
            "neo4j_ingestion_adapter",
        }:
            self.assertEqual(registry.get(name).status, "implemented")


if __name__ == "__main__":
    unittest.main()
