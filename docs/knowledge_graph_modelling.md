# Knowledge Graph Modelling

The quality of a graph comes from how it is *formed*. This framework focuses on
three things, and every one of them is measured, not assumed.

The single rule behind all of it: **store each real-world thing once, give it a
stable identity, declare what it is, and check the result.**

---

## 1. Structured data → graph

**Goal: no categorical duplication, and relationships formed only where they belong.**

Every source column is resolved into two separate decisions — its **semantic role**
(what it means) and its **physical materialization** (how it is stored):

| Materialization | Becomes |
| --- | --- |
| identity | the node's business key |
| node | one **shared concept node** + a typed edge (deduplicated) |
| edge | a typed **edge** to the referenced entity (foreign key) |
| property | a literal value on the node |
| excluded | dropped |

So a value like `"Engineering"` on 10,000 rows becomes **one** `Department` node
that 10,000 employees link to — never a literal repeated 10,000 times. A foreign
key such as `manager_id` becomes an `Employee -[:manager]-> Employee` edge. A
low-cardinality value with no identity (e.g. a two-value `status`) stays a
property — a scalar enumeration, not a node — because role and materialization are
decided separately, not by cardinality alone. A foreign key whose target table is
absent is **retained as a value, never fabricated** as a placeholder entity.
Business-entity identity is kept separate from source-row identity, and equal
business keys across files merge or stay distinct according to the declared
**identity scope**.

This resolved plan is the single input to construction, and six schema
projections are generated from it and checked to agree: the property-graph
schema, a JSON-LD context, an OWL 2 RL ontology, SHACL shapes, Neo4j constraints,
and evaluation expectations.

Code:

- `application/semantic_profiler.py` — classify columns into roles and infer keys with confidence (`SemanticColumnProfiler`).
- `application/resolved_plan.py` + `application/plan_resolution.py` — the single **resolved contract** and its six agreeing projections.
- `application/source_catalog.py` — accept declared structure (database constraints, CSVW, R2RML, or an explicit policy) and fall back to profiling; resolve foreign keys across tables (`SourceCatalog`).
- `application/semantic_table_mapper.py` — build the graph from the resolved plan (`SemanticTableMapper`).

`construct_kg` uses this semantic path by default for structured sources; the
`DirectTableMapper` baseline is retained only for explicit comparison
(`pipeline.structured_mapping = "direct"`).

```python
from kg_construction import construct_kg

# CSV / Excel / SQL are mapped semantically by default.
graph = construct_kg("employees.csv").property_graph

# With declared structure, resolve foreign keys across tables:
from kg_construction.application.source_catalog import (
    CatalogTableMapper, DeclaredForeignKey, SourceCatalog, TableDeclaration,
)
from kg_construction.application.workflow import ConstructionWorkflow

catalog = SourceCatalog().declare(
    TableDeclaration(
        table_name="employees.csv", record_type="Employee",
        primary_key=["emp_id"],
        foreign_keys=[DeclaredForeignKey(columns=["manager_id"], target_table="employees")],
    )
)
graph = ConstructionWorkflow(mappers=[CatalogTableMapper(catalog)]).run(sources).property_graph
```

## 2. Unstructured data → graph

**Goal: entities identified properly, and intent not missed.**

Extracted mentions are canonicalised and linked as a default step, then coverage is
measured so "we didn't miss anything" becomes a number:

- **Canonicalisation / linking** (`infrastructure/resolution/canonicalization.py`) —
  groups mentions of the same entity (coreference), folds partial names into their
  fuller form, and links to a knowledge base when a linker is configured; otherwise
  marks the entity `nil`. Every entity records its `link_status`.
- **Ontology alignment** (`infrastructure/ontology/ontology_alignment.py`) — checks
  extracted types against the ontology and flags off-ontology (potentially
  hallucinated) terms.
- **Coverage** (`infrastructure/evaluation/extraction_coverage.py`) — reports
  `entity_coverage` (fraction of salient mentions captured), `nil_rate`, and
  `claim_coverage`, with an optional LLM-as-judge intent check.

## 3. Evaluate the graph schema

**Goal: quality is a set of numbers with a pass/fail gate.**

`infrastructure/evaluation/schema_quality.py` (`SchemaQualityEvaluator`) computes a
quality profile over the graph and gates a release on it:

- **conciseness** — `categorical_duplication_rate` (0 when every dimension value is
  one shared node), `duplicate_node_rate`, `literal_that_should_be_node_rate`;
- **completeness** — schema, property, and relationship completeness, per-class population;
- **consistency** — domain/range and cardinality violations against the ontology
  (also enforced as SHACL via the release gate in
  `infrastructure/ontology/structured_ontology.py`);
- **structure** — connectivity, orphan rate, connected components, and richness.

Release gates: `categorical_duplication_rate = 0`, `orphan_rate ≤ threshold`, and
zero consistency/SHACL violations.

```bash
make schema-quality   # writes reports/schema_quality.md
```

---

## How it is verified

Small gold fixtures pin the expected output, and CI checks them on every change:

- a structured table with its expected graph (shared nodes, edges, zero duplication), and
- an unstructured document with gold salient entities and a minimum coverage.

```bash
make modelling-correctness   # writes reports/modelling_correctness.md; fails on regression
```

Fixtures live in `tests/fixtures/gold_standards/`; the checks are in
`application/modelling_correctness.py` and run in CI.
