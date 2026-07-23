# Knowledge-Graph Modelling Rules (Normative)

Status: **Stage 0 foundation** of the modelling redesign. This document is the
normative contract that every later stage (semantic profiling, the semantic
mapper, ontology governance, and evaluation) is measured against. The words
**MUST**, **MUST NOT**, **SHOULD**, and **MAY** carry their usual RFC 2119 force.

The contract is encoded as executable models:

- `kg_construction.domain.modeling` — `ColumnRole`, `Cardinality`,
  `ControlledVocabulary`, `ReferenceNode`, `OntologyClass`, `OntologyPredicate`,
  `OntologySchema`.
- `kg_construction.application.modeling_policy` — `PromotionThresholds`,
  `CanonicalizationRule`, `ColumnPolicy`, `RelationshipDeclaration`,
  `ModelingPolicy`, `default_modeling_policy`.

Nothing consumes these models yet; Stage 0 defines the rules. Stages 1–2 make the
structured pipeline obey them, Stage 3 governs the ontology, and Stage 5 turns
each rule into a measured metric.

---

## 1. Purpose and the one principle

A knowledge graph is only useful — as an SQL-like store, a NoSQL-like store, or a
vector store — if it is *modelled properly*. Proper modelling means the graph
represents **each real thing once**, connects things with **typed, meaningful
relationships**, and carries a **declared schema** you can validate against.

The single principle behind every rule below:

> Store each real-world thing (entity, dimension value, relationship) **exactly
> once**, give it a **stable identity independent of how it is displayed**, and
> **declare** what it is.

This directly serves the intrinsic quality dimensions of the linked-data quality
literature — **conciseness** (no redundancy), **completeness** (nothing missing),
**semantic accuracy**, and **consistency** (Zaveri et al., 2016).

---

## 2. The core decision: node vs. property vs. relationship

Every source field becomes one of: node identity, a shared node, a relationship,
or a literal property. The decision is **query-driven**, not automatic
(Neo4j graph-data-modelling guidance).

- **M1.** A field that *uniquely identifies* the row's entity (a primary key)
  **MUST** define node identity, not a duplicated node.
- **M2.** A field that *references another entity* (a foreign key or a
  dimensional category you traverse through) **SHOULD** become a **relationship**
  to a **shared node**, not a repeated literal.
- **M3.** A field that is a *measurement, timestamp, or free text* **SHOULD**
  stay a **literal property** on its entity node.
- **M4.** A value **MUST NOT** be promoted to a node when doing so would create a
  **supernode** without query benefit (e.g. a boolean linking every row to one of
  two nodes). Low-cardinality, dense, non-traversed values stay properties.
- **M5.** Identity **MUST** be independent of mutable display values. A node's ID
  is a deterministic function of its type and canonical value/key, never of its
  current label.

## 3. Column roles

Each source column is assigned exactly one `ColumnRole`, which fixes its graph
mapping:

| Role | Meaning | Graph mapping |
|---|---|---|
| `identifier` | Primary key of the row entity | Node identity |
| `entity_reference` | Foreign reference to another entity | Typed **edge** to that entity |
| `categorical_dimension` | A dimension value (department, status, category) | Typed **edge** to a shared **`ReferenceNode`** |
| `measure` | A numeric quantity | Literal **property** |
| `temporal` | A date/time/interval | Literal **property** (or occurrence qualifier) |
| `free_text` | Unbounded natural language | Literal **property** (may seed extraction) |
| `ignore` | Noise / internal / prohibited | Excluded from the graph |

- **M6.** Every included column **MUST** be assigned a role. Columns of role
  `entity_reference` or `categorical_dimension` **MUST** declare a
  `reference_type` (the class of the node they resolve to). Columns of other
  roles **MUST NOT** declare one. (Enforced by `ColumnPolicy`.)

## 4. The conciseness rule: categorical values become shared nodes

This is the rule that eliminates categorical duplication.

- **M7.** All rows whose canonical value for a `categorical_dimension` /
  `entity_reference` column is the same **MUST** link to **one** `ReferenceNode`.
  The node identity is `stable_id("ref", reference_type, canonical_value)` — a
  function of `(type, canonical value)` **only**. Two surface forms of the same
  value (`"Engineering"`, `"  ENGINEERING "`) therefore resolve to the *same*
  node.
- **M8.** A canonical value **MUST** be produced by an explicit
  `CanonicalizationRule` (Unicode NFKC, whitespace normalisation, optional
  case-folding, and an alias table). Surface variants are retained as
  `raw_values` for provenance but **MUST NOT** affect identity.
- **M9.** Promotion to a node is governed by `PromotionThresholds`:
  - a column is categorical-like when its **distinct/row ratio** is at or below
    `categorical_max_distinct_ratio` (default `0.5`) and its absolute distinct
    count is at or below `categorical_max_distinct_absolute` (once at least
    `min_rows_for_ratio` rows are observed);
  - a column with distinct count at or below `keep_as_property_max_distinct`
    (default `2`) **SHOULD** stay a property (supernode avoidance, per M4);
  - a string column whose average length exceeds `free_text_min_avg_length`
    (default `40`) **SHOULD** be treated as `free_text`, not a category.
- **M10.** A reference node whose in-degree would exceed
  `supernode_fanin_warning` **MUST** be handled per `supernode_policy`
  (`warn` | `keep_property` | `allow`) — never silently created as a supernode.

## 5. Relationship formation

- **M11.** A relationship **MUST** be typed (a meaningful predicate name), and
  **SHOULD** carry a declared **direction**, **domain**, and **range**.
- **M12.** A foreign-key column **MUST** form an edge to the referenced entity,
  not a literal copy of the foreign value.
- **M13.** Every predicate **SHOULD** declare a `Cardinality` (min/max). A
  single-valued predicate is *functional* (`max_count = 1`); a required predicate
  has `min_count >= 1`.
- **M14.** A relationship that describes a *qualified* or *n-ary* fact (a role,
  a validity interval, an event) **SHOULD** be reified — kept as an occurrence
  node/edge with its qualifiers — rather than collapsed, so distinct occurrences
  between the same endpoints are not merged. (Consumed later by the existing
  occurrence-aware property-graph builder.)

## 6. Identity, canonicalisation, and controlled vocabularies

- **M15.** Categorical domains **SHOULD** be backed by a `ControlledVocabulary`
  (permissible values + aliases). A **closed** vocabulary **MUST** reject values
  outside the set (route them to review/quarantine); an **open** vocabulary
  admits new values but still alias-normalises them.
- **M16.** Node and edge identities **MUST** be deterministic and reproducible:
  the same input and configuration always yield the same IDs.

## 7. Ontology governance

The ontology (`OntologySchema`) declares the classes and predicates the graph is
allowed to contain.

- **M17.** Every class **MUST** declare a `role`: `record` (a row entity),
  `reference` (a shared dimension), or `event` (an occurrence).
- **M18.** Every predicate **MUST** declare `domain`, `range`, `kind`
  (`object` for entity→entity, `datatype` for entity→literal), and
  `Cardinality`.
- **M19.** `OntologySchema` **MUST** lower losslessly-enough into the existing
  `GraphSchema` (`to_graph_schema()`), so the repository's SHACL/OWL emitters and
  validators are reused rather than duplicated. Object-predicate cardinality
  beyond `required` is retained on the `OntologySchema` and in schema metadata
  (the legacy SHACL emitter is extended in Stage 3).
- **M20.** New classes/predicates **SHOULD** enter through the existing
  schema proposal → quarantine → promotion lifecycle before being treated as
  governed.

## 8. How the contract is encoded (example)

```python
from kg_construction.application import ColumnPolicy, ModelingPolicy
from kg_construction.domain import ColumnRole, ControlledVocabulary

policy = ModelingPolicy(
    table_name="employees",
    record_type="Employee",
    id_columns=["emp_id"],
    columns=[
        ColumnPolicy(column="emp_id", role=ColumnRole.IDENTIFIER),
        # A categorical dimension: "Engineering" becomes ONE shared Department node.
        ColumnPolicy(column="department", role=ColumnRole.CATEGORICAL_DIMENSION,
                     reference_type="Department", relationship_type="IN_DEPARTMENT"),
        # A foreign reference: manager_id becomes an edge to another Employee.
        ColumnPolicy(column="manager_id", role=ColumnRole.ENTITY_REFERENCE,
                     reference_type="Employee", relationship_type="REPORTS_TO"),
        ColumnPolicy(column="salary", role=ColumnRole.MEASURE),   # literal property
        ColumnPolicy(column="hired_on", role=ColumnRole.TEMPORAL),
        ColumnPolicy(column="notes", role=ColumnRole.FREE_TEXT),
        ColumnPolicy(column="_internal", role=ColumnRole.IGNORE),
    ],
    controlled_vocabularies=[
        ControlledVocabulary(name="Department",
                             permissible_values=["Engineering", "Sales"], closed=False),
    ],
)

ontology = policy.to_ontology_schema()   # classes + predicates + cardinality
graph_schema = ontology.to_graph_schema()  # bridges to the existing registry/SHACL
```

## 9. Grounding in existing standards ("multiple implementations")

The abstractions are deliberately faithful to established mapping and schema
languages so the contract is not invented in a vacuum:

| Concept here | Comes from |
|---|---|
| `ColumnPolicy` (column → term map) | W3C **R2RML** term maps; W3C **CSVW** column `datatype`/`aboutUrl`/`valueUrl` |
| `RelationshipDeclaration` (FK → edge with join) | W3C **R2RML** *referencing object map* + join condition |
| `ColumnRole` / class role | **LinkML** slot roles, `identifier`; dimensional modelling (fact vs. dimension) |
| `ControlledVocabulary` | **LinkML** `enum` / `permissible_values`; SHACL `sh:in` |
| `Cardinality` | SHACL `sh:minCount` / `sh:maxCount`; LinkML `required` / `multivalued` |
| `OntologyClass` / `OntologyPredicate` domain/range | RDFS/OWL `rdfs:domain` / `rdfs:range`; **LinkML** class/slot |
| `PromotionThresholds` (node vs. property, supernode guard) | **Neo4j** graph-data-modelling guidance |
| The one principle (store once, stable identity, declare it) | **Zaveri et al.** intrinsic dimensions: conciseness, completeness, accuracy, consistency |

## 10. How each rule will be measured (forward reference to Stage 5)

Every rule above becomes a metric so "properly modelled" is a number, not an
opinion:

- **M7–M10 (conciseness)** → *categorical-duplication rate* (target `0`),
  *duplicate-node rate*, *literal-that-should-be-a-node rate*.
- **M11–M14 (relationships)** → *foreign-key edge coverage*, *typed-edge rate*,
  *orphan/isolated-node rate*, connectivity.
- **M6, M17–M19 (schema)** → *schema conformance* (SHACL violations = 0),
  *class/relationship/attribute richness*, *population per class*.
- **M15–M16 (identity)** → *controlled-vocabulary conformance*, *replay
  determinism*.

## References

- A. Zaveri et al., *Quality Assessment for Linked Data: A Survey* — quality dimensions (accuracy, consistency, **conciseness**, **completeness**). <https://www.emse.fr/~zimmermann/KGBook/Multifile/quality-assessment/>
- Knowledge Graph Completeness: A Systematic Literature Review. <https://hal.science/hal-03621495v1/file/Knowledge_Graph_Completeness_A_Systematic_Literature_Review.pdf>
- Neo4j — *Graph Data Modeling: Categorical Variables* (node vs. property, supernodes). <https://medium.com/neo4j/graph-data-modeling-categorical-variables-dd8a2845d5e0>
- W3C **R2RML** (RDB to RDF Mapping Language) and **CSVW** (CSV on the Web).
- **LinkML** (Linked data Modeling Language) — classes, slots, enums.
- W3C **SHACL** (Shapes Constraint Language) — cardinality and value constraints.
