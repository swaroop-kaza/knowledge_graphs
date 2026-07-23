# Knowledge Graph Modelling

The quality of a graph comes from how it is *formed*. This framework focuses on
three things, and every one of them is measured, not assumed.

The single rule behind all of it: **store each real-world thing once, give it a
stable identity, declare what it is, and check the result.**

---

## 1. Structured data → graph

**Goal: no categorical duplication, and relationships formed only where they belong.**

Each source column is first assigned a modelling role, then mapped accordingly:

| Column role | Becomes |
| --- | --- |
| identifier | the node's identity |
| categorical dimension | one **shared node** + a typed edge (deduplicated) |
| entity reference (foreign key) | a typed **edge** to the referenced entity |
| measure / temporal / free text | a node **property** |
| ignore | excluded |

So a value like `"Engineering"` on 10,000 rows becomes **one** `Department` node
that 10,000 employees link to — never a literal repeated 10,000 times. A
foreign key such as `manager_id` becomes a `Employee -[:manager]-> Employee` edge,
not a copied value. A **supernode guard** keeps very low-cardinality categoricals
(e.g. a two-value `status`) as properties instead of nodes.

Code:

- `application/semantic_profiler.py` — classify columns into roles (`SemanticColumnProfiler`).
- `application/modeling_policy.py` — the declarative policy the mapper executes.
- `application/semantic_table_mapper.py` — build the graph (`SemanticTableMapper`).

```python
from kg_construction.application import SemanticColumnProfiler, SemanticTableMapper
from kg_construction.application.workflow import ConstructionWorkflow
from kg_construction.domain.models import SourceRecord

policy = SemanticColumnProfiler().profile(rows, record_type="Employee").to_modeling_policy()
graph = ConstructionWorkflow(mappers=[SemanticTableMapper(policy)]).run(
    [SourceRecord.from_rows(rows, source_type="csv", source_name="employees.csv")]
).property_graph
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
