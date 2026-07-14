"""Schema registry infrastructure."""

from __future__ import annotations

from kg_construction.domain.schema import GraphSchema, SchemaRegistry


class InMemorySchemaRegistry(SchemaRegistry):
    """In-memory implementation suitable for tests, batch jobs, and local pipelines."""

    def __init__(self, schema: GraphSchema | None = None) -> None:
        super().__init__(schema)


__all__ = ["InMemorySchemaRegistry"]
