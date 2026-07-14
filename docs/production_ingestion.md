# Production Graph Ingestion

This repository now supports production-oriented property graph ingestion into Neo4j and Kuzu, with retry and dead-letter replay around the write path.

## Research Anchors

- Neo4j Python driver recommends managed transactions such as `session.execute_write(...)` for retryable write work.
- Neo4j Cypher supports parameterized values, but labels and relationship types must be part of the query text, so this implementation sanitizes labels/types and parameterizes row payloads.
- Kuzu is an embedded graph database with a Python API built around `Database`, `Connection`, and `Connection.execute(...)`.
- Kuzu requires schema before writes, so the implementation creates a generic property graph schema and stores arbitrary labels/properties as JSON strings.
- Kuzu supports transactional Cypher semantics, so writes can be wrapped in `BEGIN TRANSACTION`, `COMMIT`, and `ROLLBACK`.

## Implemented Capabilities

| Capability | Class / CLI | What it does |
|---|---|---|
| Neo4j write transactions | `Neo4jIngestionAdapter` | Writes property graph nodes/edges through managed Neo4j write transactions. |
| Kuzu write adapter | `KuzuIngestionAdapter` | Creates generic Kuzu graph schema and writes nodes/edges with parameterized MERGE. |
| Batch retry | `BatchRetryPolicy` | Retries idempotent writes with bounded exponential backoff. |
| Dead-letter queue | `DeadLetterQueue` | Persists failed graph write batches as JSONL. |
| Replay tool | `DeadLetterReplayTool` / `kg-construction-replay-deadletters` | Replays pending failed batches into the configured writer. |

## Neo4j Write Path

- Groups nodes by label set.
- Groups relationships by relationship type.
- Sanitizes labels and relationship types because Cypher does not parameterize schema tokens.
- Uses parameterized batches:

```cypher
UNWIND $rows AS row
MERGE (n:Entity:System {id: row.id})
SET n += row.properties
```

```cypher
UNWIND $rows AS row
MATCH (s {id: row.source_id})
MATCH (t {id: row.target_id})
MERGE (s)-[r:AFFECTED {id: row.id}]->(t)
SET r += row.properties
```

- Executes write work through `session.execute_write(...)`.
- Optionally creates per-label `id` uniqueness constraints.
- Produces `IngestionWriteReport` with counts, attempts, retries, batches, and statements.

## Kuzu Write Path

Kuzu uses a schema-first storage model. To preserve arbitrary property graph data without generating a table for every label/type, the adapter creates a generic schema:

```cypher
CREATE NODE TABLE KGNode(
  id STRING PRIMARY KEY,
  labels STRING,
  properties STRING
)
```

```cypher
CREATE REL TABLE KGEdge(
  FROM KGNode TO KGNode,
  id STRING,
  type STRING,
  properties STRING
)
```

Then it writes nodes and edges with parameterized `MERGE` statements inside a transaction. Labels and properties are JSON strings, which keeps the adapter reusable across many graph shapes.

## Retry And Dead-Letter Flow

```text
PropertyGraph
-> graph store writer
-> BatchRetryPolicy
   -> success: IngestionWriteReport
   -> exhausted: DeadLetterQueue record
-> replay later with DeadLetterReplayTool
```

The retry policy is intentionally generic:

- Retries by known transient exception names.
- Also retries when messages contain fragments such as `timeout`, `transient`, `unavailable`, or `deadlock`.
- Can disable sleep for deterministic tests.
- Raises `RetryExhaustedError` when attempts are exhausted.

Dead-letter records include:

- store name
- operation name
- graph model
- serialized graph payload
- attempt count
- error type and message
- status: `pending`, `replayed`, `failed`, or `skipped`

## Replay CLI

Dry-run Neo4j replay:

```bash
kg-construction-replay-deadletters \
  --dead-letter-path data/processed/graph_ingestion_dead_letters.jsonl \
  --store neo4j_ingestion_adapter \
  --dry-run
```

Replay into Kuzu:

```bash
kg-construction-replay-deadletters \
  --dead-letter-path data/processed/graph_ingestion_dead_letters.jsonl \
  --store kuzu_ingestion_adapter \
  --kuzu-database data/processed/kuzu_graph
```

## Resource And Latency Notes

- Neo4j uses fewer statements because rows are batched with `UNWIND`.
- Kuzu currently executes one parameterized write per node/edge inside a transaction, which is simple and reliable for small/medium graphs.
- Kuzu large-graph ingestion should later add a bulk import path.
- Retry improves resilience but can increase tail latency.
- Dead-letter replay prevents silent data loss, but replayed links should still be idempotent.

## Current Limitations

- The implemented ingestion path targets property graphs, not RDF import.
- Neo4j RDF import should be added through an explicit neosemantics integration when required.
- Kuzu uses a generic schema for portability; domain-specific Kuzu tables can be generated later from `GraphSchema`.
- Tests use fake drivers/connections. Live database integration tests should be added under optional profiles.

## References

- Neo4j Python Driver Manual: transactions: https://neo4j.com/docs/python-manual/current/transactions/
- Neo4j Python Driver Manual: query parameters: https://neo4j.com/docs/python-manual/current/query-simple/#query-parameters
- Kuzu Python API documentation: https://docs.kuzudb.com/client-apis/python/
- Kuzu transactions documentation: https://docs.kuzudb.com/cypher/transactions/
- Kuzu data definition documentation: https://docs.kuzudb.com/cypher/data-definition/
