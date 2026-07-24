"""Production contract tests for the mandatory Neo4j vector store."""

from __future__ import annotations

import math
import unittest
from collections.abc import Iterator
from typing import Any

from kg_construction.application.registry import default_registry
from kg_construction.infrastructure.graph_stores import (
    Neo4jVectorStoreAdapter,
    Neo4jVectorStoreConfig,
    VectorIndexConfigurationError,
    VectorRecord,
    VectorStoreValidationError,
)


class FakeResult:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self.rows = rows or []
        self.consumed = False

    def consume(self) -> None:
        self.consumed = True

    def single(self) -> dict[str, object] | None:
        return self.rows[0] if self.rows else None

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.rows)


class FakeTransaction:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver

    def run(self, query: str, **parameters: object) -> FakeResult:
        self.driver.runs.append((query, parameters))
        return FakeResult()


class FakeSession:
    def __init__(self, driver: FakeDriver, database: str | None) -> None:
        self.driver = driver
        self.database = database

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        return None

    def run(self, query: str, **parameters: object) -> FakeResult:
        self.driver.runs.append((query, parameters))
        if query.startswith("SHOW VECTOR INDEXES"):
            return FakeResult([self.driver.index_record])
        if "db.index.vector.queryNodes" in query:
            return FakeResult(self.driver.search_rows)
        return FakeResult()

    def execute_write(self, callback: Any, rows: list[dict[str, object]]) -> None:
        self.driver.write_transactions += 1
        callback(FakeTransaction(self.driver), rows)


class FakeDriver:
    def __init__(
        self,
        *,
        index_record: dict[str, object] | None = None,
        search_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.index_record = index_record or online_index_record()
        self.search_rows = search_rows or []
        self.runs: list[tuple[str, dict[str, object]]] = []
        self.databases: list[str | None] = []
        self.write_transactions = 0
        self.closed = False

    def session(self, **kwargs: object) -> FakeSession:
        database = kwargs.get("database")
        selected = database if isinstance(database, str) else None
        self.databases.append(selected)
        return FakeSession(self, selected)

    def close(self) -> None:
        self.closed = True


def online_index_record(
    *,
    dimensions: int = 3,
    state: str = "ONLINE",
    label: str = "KGVector",
    property_name: str = "embedding",
    similarity: str = "COSINE",
) -> dict[str, object]:
    return {
        "name": "kg_vector_index",
        "state": state,
        "populationPercent": 100.0,
        "entityType": "NODE",
        "labelsOrTypes": [label],
        "properties": [property_name],
        "options": {
            "indexConfig": {
                "vector.dimensions": dimensions,
                "vector.similarity_function": similarity,
            }
        },
    }


def configured_adapter(driver: FakeDriver) -> Neo4jVectorStoreAdapter:
    return Neo4jVectorStoreAdapter(
        Neo4jVectorStoreConfig(
            database="neo4j",
            dimensions=3,
            batch_size=1,
            filter_candidate_multiplier=20,
            max_top_k=100,
            max_filter_candidates=2000,
        ),
        driver=driver,
    )


class Neo4jVectorStoreTests(unittest.TestCase):
    def test_ensure_index_creates_constraint_and_current_vector_index_then_verifies_it(
        self,
    ) -> None:
        driver = FakeDriver()
        adapter = configured_adapter(driver)

        report = adapter.ensure_index()

        self.assertEqual(report.state, "ONLINE")
        self.assertEqual(report.dimensions, 3)
        self.assertEqual(report.similarity, "cosine")
        self.assertEqual(driver.databases, ["neo4j"])
        constraint = next(query for query, _ in driver.runs if "CREATE CONSTRAINT" in query)
        create = next(query for query, _ in driver.runs if "CREATE VECTOR INDEX" in query)
        create_parameters = next(
            parameters
            for query, parameters in driver.runs
            if "CREATE VECTOR INDEX" in query
        )
        self.assertIn("REQUIRE n.`id` IS UNIQUE", constraint)
        self.assertIn("FOR (n:`KGVector`) ON (n.`embedding`)", create)
        self.assertIn("`vector.dimensions`: $dimensions", create)
        self.assertEqual(
            {"dimensions": 3, "similarity": "cosine"},
            create_parameters,
        )
        self.assertTrue(any("db.awaitIndex" in query for query, _ in driver.runs))

    def test_upsert_validates_entire_batch_and_uses_transactional_vector_procedure(
        self,
    ) -> None:
        driver = FakeDriver()
        adapter = configured_adapter(driver)
        records = [
            VectorRecord(
                vector_id="doc-1",
                embedding=[1.0, 0.0, 0.0],
                properties={"tenant": "blue", "chunk": 1},
            ),
            VectorRecord(
                vector_id="doc-2",
                embedding=[0.0, 1.0, 0.0],
                properties={"tenant": "green", "chunk": 2},
            ),
        ]

        report = adapter.upsert(records)

        self.assertEqual(report.record_count, 2)
        self.assertEqual(report.batch_count, 2)
        self.assertEqual(driver.write_transactions, 2)
        writes = [
            (query, parameters)
            for query, parameters in driver.runs
            if "db.create.setNodeVectorProperty" in query
        ]
        self.assertEqual(len(writes), 2)
        self.assertTrue(all("$rows" in query for query, _ in writes))
        self.assertEqual(writes[0][1]["embedding_property"], "embedding")
        self.assertEqual(
            {
                "id": "doc-1",
                "embedding": [1.0, 0.0, 0.0],
                "properties": {"tenant": "blue", "chunk": 1},
            },
            writes[0][1]["rows"][0],  # type: ignore[index]
        )

    def test_invalid_batch_fails_before_any_database_call(self) -> None:
        invalid_records = {
            "dimension": [
                VectorRecord(vector_id="doc", embedding=[1.0, 0.0])
            ],
            "nan": [
                VectorRecord(vector_id="doc", embedding=[1.0, math.nan, 0.0])
            ],
            "zero_cosine": [
                VectorRecord(vector_id="doc", embedding=[0.0, 0.0, 0.0])
            ],
            "duplicate_id": [
                VectorRecord(vector_id="doc", embedding=[1.0, 0.0, 0.0]),
                VectorRecord(vector_id="doc", embedding=[0.0, 1.0, 0.0]),
            ],
            "reserved_property": [
                VectorRecord(
                    vector_id="doc",
                    embedding=[1.0, 0.0, 0.0],
                    properties={"embedding": "shadow"},
                )
            ],
            "unsafe_property": [
                VectorRecord(
                    vector_id="doc",
                    embedding=[1.0, 0.0, 0.0],
                    properties={"tenant) MATCH (n": "blue"},
                )
            ],
        }
        for name, records in invalid_records.items():
            with self.subTest(name=name):
                driver = FakeDriver()
                with self.assertRaises(VectorStoreValidationError):
                    configured_adapter(driver).upsert(records)
                self.assertFalse(driver.runs)
                self.assertEqual(driver.write_transactions, 0)

    def test_query_uses_index_scores_and_parameterized_safe_equality_filters(self) -> None:
        driver = FakeDriver(
            search_rows=[
                {
                    "vector_id": "doc-1",
                    "score": 0.99,
                    "properties": {
                        "id": "doc-1",
                        "embedding": [1.0, 0.0, 0.0],
                        "tenant": "blue",
                        "chunk": 1,
                    },
                },
                {
                    "vector_id": "doc-2",
                    "score": 0.75,
                    "properties": {
                        "id": "doc-2",
                        "embedding": [0.7, 0.3, 0.0],
                        "tenant": "blue",
                        "chunk": 2,
                    },
                },
            ]
        )
        adapter = configured_adapter(driver)

        hits = adapter.query(
            [1.0, 0.0, 0.0],
            top_k=2,
            filters={"tenant": "blue"},
        )

        self.assertEqual([hit.vector_id for hit in hits], ["doc-1", "doc-2"])
        self.assertEqual([hit.score for hit in hits], [0.99, 0.75])
        self.assertEqual(hits[0].properties, {"tenant": "blue", "chunk": 1})
        query, parameters = driver.runs[-1]
        self.assertIn("db.index.vector.queryNodes", query)
        self.assertIn("node[key] = $filters[key]", query)
        self.assertNotIn("blue", query)
        self.assertEqual(parameters["filters"], {"tenant": "blue"})
        self.assertEqual(parameters["candidate_k"], 40)
        self.assertEqual(parameters["top_k"], 2)

    def test_query_rejects_unsafe_or_malformed_inputs_before_database_call(self) -> None:
        invalid_calls: list[tuple[list[float], int, dict[str, object]]] = [
            ([1.0, 0.0], 2, {}),
            ([1.0, math.nan, 0.0], 2, {}),
            ([0.0, 0.0, 0.0], 2, {}),
            ([1.0, 0.0, 0.0], 0, {}),
            ([1.0, 0.0, 0.0], 2, {"tenant) DELETE n": "blue"}),
            ([1.0, 0.0, 0.0], 2, {"tenant": ["blue"]}),
        ]
        for embedding, top_k, filters in invalid_calls:
            with self.subTest(embedding=embedding, top_k=top_k, filters=filters):
                driver = FakeDriver()
                with self.assertRaises(VectorStoreValidationError):
                    configured_adapter(driver).query(
                        embedding,
                        top_k=top_k,
                        filters=filters,
                    )
                self.assertFalse(driver.runs)

    def test_index_schema_mismatch_fails_closed(self) -> None:
        for name, record in {
            "dimension": online_index_record(dimensions=4),
            "state": online_index_record(state="POPULATING"),
            "label": online_index_record(label="WrongLabel"),
            "property": online_index_record(property_name="wrong_embedding"),
            "similarity": online_index_record(similarity="EUCLIDEAN"),
        }.items():
            with self.subTest(name=name):
                driver = FakeDriver(index_record=record)
                with self.assertRaises(VectorIndexConfigurationError):
                    configured_adapter(driver).ensure_index()

    def test_invalid_live_score_fails_closed(self) -> None:
        driver = FakeDriver(
            search_rows=[
                {
                    "vector_id": "doc-1",
                    "score": math.nan,
                    "properties": {"id": "doc-1", "embedding": [1.0, 0.0, 0.0]},
                }
            ]
        )
        with self.assertRaises(VectorStoreValidationError):
            configured_adapter(driver).query([1.0, 0.0, 0.0])

    def test_configuration_rejects_unsafe_identifiers_and_invalid_dimensions(self) -> None:
        for kwargs in (
            {"index_name": "index`) DROP DATABASE neo4j"},
            {"node_label": "Vector Node"},
            {"id_property": "n.id"},
            {"embedding_property": ""},
            {"dimensions": 0},
            {"dimensions": 4097},
            {"id_property": "embedding", "embedding_property": "embedding"},
        ):
            with (
                self.subTest(kwargs=kwargs),
                self.assertRaises(VectorStoreValidationError),
            ):
                Neo4jVectorStoreConfig(**kwargs)

    def test_registry_declares_vector_store_mandatory_on_every_profile(self) -> None:
        descriptor = default_registry().get("neo4j_vector_store_adapter")

        self.assertEqual(descriptor.status, "implemented")
        self.assertEqual(descriptor.public_support_level, "mandatory")
        self.assertEqual(
            descriptor.profiles,
            ("darwin-arm64", "linux-cpu", "linux-cuda"),
        )
        self.assertIs(descriptor.adapter, Neo4jVectorStoreAdapter)

    def test_injected_driver_is_not_closed_by_adapter(self) -> None:
        driver = FakeDriver()
        adapter = configured_adapter(driver)

        adapter.close()

        self.assertFalse(driver.closed)


if __name__ == "__main__":
    unittest.main()
