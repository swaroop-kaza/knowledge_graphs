"""Strict Neo4j vector-index storage and retrieval.

The adapter targets the repository's locked Neo4j 5.26 service contract.  It
uses the declarative ``CREATE VECTOR INDEX`` syntax, the efficient
``db.create.setNodeVectorProperty`` write procedure, and
``db.index.vector.queryNodes`` for indexed approximate-nearest-neighbour
queries.  Validation happens before any database call so malformed batches
cannot be partially written.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured

JsonDict = dict[str, Any]
VectorSimilarity = Literal["cosine", "euclidean"]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SIGNED_INT64_MIN = -(2**63)
_SIGNED_INT64_MAX = 2**63 - 1
_IEEE754_FLOAT32_MAX = 3.4028234663852886e38


class VectorStoreValidationError(ValueError):
    """Input or configuration violates the vector-store contract."""


class VectorIndexConfigurationError(RuntimeError):
    """The live index does not match the declared vector schema."""


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise VectorStoreValidationError(
            f"{field_name} must match {_IDENTIFIER.pattern!r}; unsafe identifier {value!r}"
        )
    return value


def _quoted_identifier(value: str) -> str:
    """Quote an identifier only after strict validation."""

    return f"`{value}`"


@dataclass(frozen=True, slots=True)
class Neo4jVectorStoreConfig:
    """Fixed-schema configuration for a Neo4j node vector index."""

    uri: str = "bolt://localhost:7687"
    user: str | None = None
    password: str | None = None
    database: str | None = None
    index_name: str = "kg_vector_index"
    node_label: str = "KGVector"
    id_property: str = "id"
    embedding_property: str = "embedding"
    dimensions: int = 384
    similarity: VectorSimilarity = "cosine"
    batch_size: int = 250
    await_index_timeout_seconds: int = 300
    max_top_k: int = 1000
    filter_candidate_multiplier: int = 20
    max_filter_candidates: int = 10_000

    def __post_init__(self) -> None:
        if not isinstance(self.uri, str) or not self.uri.strip():
            raise VectorStoreValidationError("uri must be a non-empty string")
        for name in ("index_name", "node_label", "id_property", "embedding_property"):
            _validate_identifier(str(getattr(self, name)), field_name=name)
        if self.id_property == self.embedding_property:
            raise VectorStoreValidationError(
                "id_property and embedding_property must be different"
            )
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 4096:
            raise VectorStoreValidationError(
                "dimensions must be an integer between 1 and 4096"
            )
        if self.similarity not in {"cosine", "euclidean"}:
            raise VectorStoreValidationError(
                "similarity must be either 'cosine' or 'euclidean'"
            )
        for name in (
            "batch_size",
            "await_index_timeout_seconds",
            "max_top_k",
            "filter_candidate_multiplier",
            "max_filter_candidates",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise VectorStoreValidationError(f"{name} must be a positive integer")
        if self.max_filter_candidates < self.max_top_k:
            raise VectorStoreValidationError(
                "max_filter_candidates must be greater than or equal to max_top_k"
            )

    @property
    def constraint_name(self) -> str:
        """Stable uniqueness-constraint name for concurrency-safe upserts."""

        return f"{self.index_name}_id_unique"


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One identified vector and its filterable Neo4j properties."""

    vector_id: str
    embedding: Sequence[float]
    properties: Mapping[str, object] = field(default_factory=dict)


class VectorIndexReport(BaseModel):
    """Verified state of the live Neo4j vector index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index_name: str
    state: str
    population_percent: float
    node_label: str
    embedding_property: str
    dimensions: int
    similarity: VectorSimilarity


class VectorUpsertReport(BaseModel):
    """Transactional vector upsert outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    store_name: str
    index_name: str
    node_label: str
    dimensions: int
    record_count: int
    batch_count: int


class VectorSearchHit(BaseModel):
    """One scored Neo4j vector-search result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vector_id: str
    score: float = Field(ge=0.0, le=1.0)
    properties: JsonDict = Field(default_factory=dict)


class Neo4jVectorStoreAdapter:
    """Create, write, and query a fixed-dimension Neo4j vector index."""

    name = "neo4j_vector_store_adapter"

    def __init__(
        self,
        config: Neo4jVectorStoreConfig | None = None,
        *,
        driver: Any | None = None,
    ) -> None:
        self.config = config or Neo4jVectorStoreConfig()
        self._driver = driver
        self._owns_driver = driver is None

    def ensure_index(self) -> VectorIndexReport:
        """Create the uniqueness constraint/vector index and verify its schema."""

        config = self.config
        label = _quoted_identifier(config.node_label)
        id_property = _quoted_identifier(config.id_property)
        embedding_property = _quoted_identifier(config.embedding_property)
        constraint_name = _quoted_identifier(config.constraint_name)
        index_name = _quoted_identifier(config.index_name)
        constraint_query = (
            f"CREATE CONSTRAINT {constraint_name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{id_property} IS UNIQUE"
        )
        index_query = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{embedding_property}) "
            "OPTIONS {indexConfig: {"
            "`vector.dimensions`: $dimensions, "
            "`vector.similarity_function`: $similarity"
            "}}"
        )
        session_kwargs = {"database": config.database} if config.database else {}
        with self._driver_instance().session(**session_kwargs) as session:
            _consume(session.run(constraint_query))
            _consume(
                session.run(
                    index_query,
                    dimensions=config.dimensions,
                    similarity=config.similarity,
                )
            )
            _consume(
                session.run(
                    "CALL db.awaitIndex($index_name, $timeout_seconds)",
                    index_name=config.index_name,
                    timeout_seconds=config.await_index_timeout_seconds,
                )
            )
            record = session.run(
                "SHOW VECTOR INDEXES "
                "YIELD name, state, populationPercent, entityType, "
                "labelsOrTypes, properties, options "
                "WHERE name = $index_name "
                "RETURN name, state, populationPercent, entityType, "
                "labelsOrTypes, properties, options",
                index_name=config.index_name,
            ).single()
        if record is None:
            raise VectorIndexConfigurationError(
                f"Neo4j did not report vector index {config.index_name!r} after creation"
            )
        return self._verify_index_record(record)

    def upsert(self, records: Sequence[VectorRecord]) -> VectorUpsertReport:
        """Validate the complete batch, then transactionally merge all vectors."""

        rows = self._validated_rows(records)
        # Index and ID uniqueness are mandatory preconditions, not caller-selected options.
        self.ensure_index()
        if not rows:
            return VectorUpsertReport(
                store_name=self.name,
                index_name=self.config.index_name,
                node_label=self.config.node_label,
                dimensions=self.config.dimensions,
                record_count=0,
                batch_count=0,
            )

        batches = list(_chunks(rows, self.config.batch_size))
        session_kwargs = (
            {"database": self.config.database} if self.config.database else {}
        )
        with self._driver_instance().session(**session_kwargs) as session:
            for batch in batches:
                session.execute_write(self._upsert_tx, batch)
        return VectorUpsertReport(
            store_name=self.name,
            index_name=self.config.index_name,
            node_label=self.config.node_label,
            dimensions=self.config.dimensions,
            record_count=len(rows),
            batch_count=len(batches),
        )

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 10,
        filters: Mapping[str, object] | None = None,
    ) -> list[VectorSearchHit]:
        """Return scored ANN hits, with parameterized equality filters when supplied."""

        if type(top_k) is not int or not 1 <= top_k <= self.config.max_top_k:
            raise VectorStoreValidationError(
                f"top_k must be an integer between 1 and {self.config.max_top_k}"
            )
        vector = _validate_vector(
            query_embedding,
            dimensions=self.config.dimensions,
            similarity=self.config.similarity,
            field_name="query_embedding",
        )
        normalized_filters = _validate_filters({} if filters is None else filters)
        candidate_k = top_k
        if normalized_filters:
            candidate_k = min(
                self.config.max_filter_candidates,
                top_k * self.config.filter_candidate_multiplier,
            )
        query = (
            "CALL db.index.vector.queryNodes("
            "$index_name, $candidate_k, $query_vector"
            ") "
            "YIELD node, score "
            "WHERE all(key IN keys($filters) WHERE node[key] = $filters[key]) "
            "RETURN node[$id_property] AS vector_id, score, "
            "properties(node) AS properties "
            "ORDER BY score DESC, vector_id ASC "
            "LIMIT $top_k"
        )
        session_kwargs = (
            {"database": self.config.database} if self.config.database else {}
        )
        with self._driver_instance().session(**session_kwargs) as session:
            result = session.run(
                query,
                index_name=self.config.index_name,
                candidate_k=candidate_k,
                query_vector=vector,
                filters=normalized_filters,
                id_property=self.config.id_property,
                top_k=top_k,
            )
            raw_rows = list(result)

        hits: list[VectorSearchHit] = []
        for raw in raw_rows:
            vector_id = _record_get(raw, "vector_id")
            score_raw = _record_get(raw, "score")
            properties_raw = _record_get(raw, "properties")
            if not isinstance(vector_id, str) or not vector_id:
                raise VectorIndexConfigurationError(
                    "Neo4j vector result is missing a non-empty vector ID"
                )
            score = _finite_float(score_raw, field_name="score")
            if not 0.0 <= score <= 1.0:
                raise VectorIndexConfigurationError(
                    f"Neo4j vector score must be within [0, 1], received {score!r}"
                )
            if not isinstance(properties_raw, Mapping):
                raise VectorIndexConfigurationError(
                    "Neo4j vector result properties must be a mapping"
                )
            properties = {str(key): value for key, value in properties_raw.items()}
            properties.pop(self.config.embedding_property, None)
            properties.pop(self.config.id_property, None)
            hits.append(
                VectorSearchHit(
                    vector_id=vector_id,
                    score=score,
                    properties=properties,
                )
            )
        return hits

    def close(self) -> None:
        """Close a driver created by this adapter."""

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
                "Neo4j is a mandatory dependency. Run repository provisioning "
                "or inject a configured Neo4j driver."
            ) from exc
        auth = None
        if self.config.user is not None or self.config.password is not None:
            auth = (self.config.user or "", self.config.password or "")
        self._driver = GraphDatabase.driver(self.config.uri, auth=auth)
        return self._driver

    def _verify_index_record(self, record: object) -> VectorIndexReport:
        config = self.config
        name = _record_get(record, "name")
        state = str(_record_get(record, "state") or "").upper()
        population = _finite_float(
            _record_get(record, "populationPercent"),
            field_name="populationPercent",
        )
        entity_type = str(_record_get(record, "entityType") or "").upper()
        labels = _record_get(record, "labelsOrTypes")
        properties = _record_get(record, "properties")
        options = _record_get(record, "options")
        index_config = (
            options.get("indexConfig")
            if isinstance(options, Mapping)
            else None
        )
        dimension = (
            index_config.get("vector.dimensions")
            if isinstance(index_config, Mapping)
            else None
        )
        similarity = (
            index_config.get("vector.similarity_function")
            if isinstance(index_config, Mapping)
            else None
        )
        mismatches: list[str] = []
        if name != config.index_name:
            mismatches.append(f"name={name!r}")
        if state != "ONLINE":
            mismatches.append(f"state={state!r}")
        if entity_type != "NODE":
            mismatches.append(f"entityType={entity_type!r}")
        if not isinstance(labels, Sequence) or isinstance(labels, str | bytes):
            mismatches.append(f"labelsOrTypes={labels!r}")
        elif list(labels) != [config.node_label]:
            mismatches.append(f"labelsOrTypes={list(labels)!r}")
        if not isinstance(properties, Sequence) or isinstance(properties, str | bytes):
            mismatches.append(f"properties={properties!r}")
        elif list(properties) != [config.embedding_property]:
            mismatches.append(f"properties={list(properties)!r}")
        if type(dimension) is not int or dimension != config.dimensions:
            mismatches.append(f"vector.dimensions={dimension!r}")
        if str(similarity or "").casefold() != config.similarity:
            mismatches.append(f"vector.similarity_function={similarity!r}")
        if population < 100.0:
            mismatches.append(f"populationPercent={population!r}")
        if mismatches:
            raise VectorIndexConfigurationError(
                f"vector index {config.index_name!r} violates its schema: "
                + ", ".join(mismatches)
            )
        return VectorIndexReport(
            index_name=config.index_name,
            state=state,
            population_percent=population,
            node_label=config.node_label,
            embedding_property=config.embedding_property,
            dimensions=config.dimensions,
            similarity=config.similarity,
        )

    def _validated_rows(self, records: Sequence[VectorRecord]) -> list[JsonDict]:
        if isinstance(records, str | bytes) or not isinstance(records, Sequence):
            raise VectorStoreValidationError("records must be a sequence of VectorRecord")
        rows: list[JsonDict] = []
        seen: set[str] = set()
        reserved = {self.config.id_property, self.config.embedding_property}
        for position, record in enumerate(records):
            if not isinstance(record, VectorRecord):
                raise VectorStoreValidationError(
                    f"records[{position}] must be a VectorRecord"
                )
            if not isinstance(record.vector_id, str) or not record.vector_id.strip():
                raise VectorStoreValidationError(
                    f"records[{position}].vector_id must be a non-empty string"
                )
            if record.vector_id in seen:
                raise VectorStoreValidationError(
                    f"duplicate vector_id {record.vector_id!r} in one upsert batch"
                )
            seen.add(record.vector_id)
            embedding = _validate_vector(
                record.embedding,
                dimensions=self.config.dimensions,
                similarity=self.config.similarity,
                field_name=f"records[{position}].embedding",
            )
            if not isinstance(record.properties, Mapping):
                raise VectorStoreValidationError(
                    f"records[{position}].properties must be a mapping"
                )
            properties: JsonDict = {}
            for key, value in record.properties.items():
                _validate_identifier(key, field_name=f"records[{position}].properties key")
                if key in reserved:
                    raise VectorStoreValidationError(
                        f"record property {key!r} is reserved by the vector schema"
                    )
                properties[key] = _normalize_property_value(
                    value,
                    field_name=f"records[{position}].properties[{key!r}]",
                )
            rows.append(
                {
                    "id": record.vector_id,
                    "embedding": embedding,
                    "properties": properties,
                }
            )
        return rows

    def _upsert_tx(self, tx: Any, rows: list[JsonDict]) -> None:
        label = _quoted_identifier(self.config.node_label)
        id_property = _quoted_identifier(self.config.id_property)
        query = (
            "UNWIND $rows AS row "
            f"MERGE (n:{label} {{{id_property}: row.id}}) "
            "SET n += row.properties "
            "CALL db.create.setNodeVectorProperty("
            "n, $embedding_property, row.embedding"
            ") "
            "RETURN count(n) AS written"
        )
        _consume(
            tx.run(
                query,
                rows=rows,
                embedding_property=self.config.embedding_property,
            )
        )


def _chunks(rows: Sequence[JsonDict], batch_size: int) -> list[list[JsonDict]]:
    return [
        list(rows[start : start + batch_size])
        for start in range(0, len(rows), batch_size)
    ]


def _consume(result: object) -> None:
    consume = getattr(result, "consume", None)
    if callable(consume):
        consume()


def _record_get(record: object, key: str) -> object:
    if isinstance(record, Mapping):
        return record.get(key)
    try:
        return record[key]  # type: ignore[index]
    except (KeyError, TypeError, IndexError, AttributeError):
        getter = getattr(record, "get", None)
        return getter(key) if callable(getter) else None


def _finite_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise VectorStoreValidationError(f"{field_name} must be numeric, not boolean")
    try:
        normalized = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise VectorStoreValidationError(f"{field_name} must be numeric") from exc
    if not math.isfinite(normalized):
        raise VectorStoreValidationError(f"{field_name} must be finite")
    return normalized


def _validate_vector(
    values: Sequence[float],
    *,
    dimensions: int,
    similarity: VectorSimilarity,
    field_name: str,
) -> list[float]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise VectorStoreValidationError(f"{field_name} must be a numeric sequence")
    if len(values) != dimensions:
        raise VectorStoreValidationError(
            f"{field_name} has dimension {len(values)}; expected exactly {dimensions}"
        )
    normalized = [
        _finite_float(value, field_name=f"{field_name}[{position}]")
        for position, value in enumerate(values)
    ]
    if similarity == "cosine":
        norm = math.hypot(*normalized)
        if not math.isfinite(norm) or norm == 0.0:
            raise VectorStoreValidationError(
                f"{field_name} must have a finite, non-zero L2 norm for cosine similarity"
            )
    elif any(abs(value) > _IEEE754_FLOAT32_MAX for value in normalized):
        raise VectorStoreValidationError(
            f"{field_name} components must fit finite IEEE 754 single precision "
            "for euclidean similarity"
        )
    return normalized


def _normalize_property_value(value: object, *, field_name: str) -> object:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        if not _SIGNED_INT64_MIN <= value <= _SIGNED_INT64_MAX:
            raise VectorStoreValidationError(
                f"{field_name} integer must fit Neo4j signed 64-bit storage"
            )
        return value
    if isinstance(value, float):
        return _finite_float(value, field_name=field_name)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        normalized = [
            _normalize_property_value(item, field_name=f"{field_name}[{position}]")
            for position, item in enumerate(value)
        ]
        if any(isinstance(item, list) or item is None for item in normalized):
            raise VectorStoreValidationError(
                f"{field_name} must be a flat list without null values"
            )
        type_families = {
            "number" if isinstance(item, int | float) and not isinstance(item, bool) else type(item)
            for item in normalized
        }
        if len(type_families) > 1:
            raise VectorStoreValidationError(
                f"{field_name} list values must have one Neo4j property type"
            )
        return normalized
    raise VectorStoreValidationError(
        f"{field_name} must be a Neo4j scalar or flat homogeneous scalar list"
    )


def _validate_filters(filters: Mapping[str, object]) -> JsonDict:
    if not isinstance(filters, Mapping):
        raise VectorStoreValidationError("filters must be a mapping")
    normalized: JsonDict = {}
    for key, value in filters.items():
        _validate_identifier(key, field_name="filter property")
        if value is None or isinstance(value, Sequence) and not isinstance(value, str | bytes):
            raise VectorStoreValidationError(
                f"filter {key!r} must use one non-null scalar equality value"
            )
        item = _normalize_property_value(value, field_name=f"filter[{key!r}]")
        if isinstance(item, list):
            raise VectorStoreValidationError(
                f"filter {key!r} must use one scalar equality value"
            )
        normalized[key] = item
    return normalized


__all__ = [
    "Neo4jVectorStoreAdapter",
    "Neo4jVectorStoreConfig",
    "VectorIndexConfigurationError",
    "VectorIndexReport",
    "VectorRecord",
    "VectorSearchHit",
    "VectorSimilarity",
    "VectorStoreValidationError",
    "VectorUpsertReport",
]
