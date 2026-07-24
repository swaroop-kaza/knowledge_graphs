# Knowledge-graph formation and evaluation contract

The repository is deliberately centred on three product pillars:

1. structured data to a correctly identified and related graph;
2. unstructured data to evidence-grounded entities, claims, and events; and
3. evaluation of both intrinsic graph quality and fitness for the intended use.

Graph storage is a projection of that model. Neo4j, Ladybug, RDF, NetworkX, Memgraph, and vector indexes cannot repair a graph whose identities or relationships were formed incorrectly.

## Modelling doctrine

The implementation follows these invariants.

- One real-world entity or governed concept has one canonical identity in its declared scope. Source rows, mentions, canonical entities, documents, claims, and events are different resources.
- A property describes its owner. A relationship connects independently identifiable things. A qualified, temporal, attributed, or n-ary relationship is reified as an association/event resource.
- A repeated scalar is not automatically an entity. Cardinality is evidence, not semantics.
- Construction never invents a missing foreign-key target or forces an uncertain mention to an existing canonical entity.
- Every promoted extracted claim quotes an immutable source version with exact end-exclusive offsets and complete provider/model/configuration provenance.
- A transformation that changes source text or rows creates a new source identity/document version. Its lineage names the parent and root versions; redaction never reuses the unredacted snapshot ID.
- OWL describes meaning and entailment; SHACL checks closed operational constraints. A graph is not accepted merely because it parses.
- Quality is reported as named metrics with support counts, thresholds, and an executed flag. “Not measured” cannot become a passing zero.

These choices align the structured path with [R2RML](https://www.w3.org/TR/r2rml/), [CSV on the Web](https://www.w3.org/TR/csv2rdf/), and the [relational direct mapping](https://www.w3.org/TR/rdb-direct-mapping/); controlled vocabularies with [SKOS](https://www.w3.org/TR/skos-reference/); validation with [SHACL](https://www.w3.org/TR/shacl/); provenance with [PROV-O](https://www.w3.org/TR/prov-o/); exact textual selectors with the [Web Annotation selector model](https://www.w3.org/TR/selectors-states/); and scalable inference with [OWL 2 RL](https://www.w3.org/TR/owl2-profiles/). The overall lifecycle also follows the acquisition, refinement, and evolution separation synthesized by the [ACM survey of more than 300 automatic KG-construction methods](https://doi.org/10.1145/3618295). Here, ingestion/extraction is acquisition, identity resolution and ontology/SHACL promotion are refinement, and versioned source identities plus deterministic re-ingestion provide the basis for evolution.

Two details from those standards are release invariants here, rather than documentation-only guidance. The W3C direct mapping explicitly supports foreign keys that reference candidate keys and identifies keyless rows by distinct blank nodes; therefore a child alternate-key value must resolve to the target row's canonical identity, and a row number cannot be used as a persistent substitute for missing identity. The [CSVW tabular data model](https://www.w3.org/TR/tabular-data-model/) likewise defines a primary key as the cells whose values uniquely identify a row and models foreign keys as references to actual rows.

## Pillar 1: structured data to knowledge graph

### Stage S0 — source inventory and declarations

For each table or tabular source, capture:

- source and table identity;
- column names, native datatypes, nullability, and array/object structure;
- declared primary, alternate, and composite keys;
- foreign keys including target table and target candidate key;
- row counts, distinct counts, missingness, and value samples;
- update/watermark columns;
- governed controlled vocabularies; and
- whether a table is an entity, event, association, or pure edge table.

`SourceCatalog` accepts explicit `TableDeclaration` objects and resolves all tables together before mapping. Declarations are authoritative. Profiling is a proposal mechanism when declarations are absent; its evidence and confidence are retained.

Exit gate:

- every required declaration is present when `structured.require_declarations=true`;
- table names are unique;
- primary and foreign-key arities are valid; and
- every source has a reproducible content identity.

### Stage S1 — semantic role and materialization

Every column receives two independent decisions.

| Semantic role | Typical meaning |
| --- | --- |
| identifier | primary, alternate, or composite business identity |
| entity reference | foreign key or governed reference to another entity |
| categorical dimension | reusable concept or scalar enumeration |
| measure | numeric value with an owner and, where relevant, a unit |
| temporal | instant, interval, validity, or event time |
| free text | literal content and possible input to the unstructured path |
| metadata | source/processing information |

| Materialization | Graph result |
| --- | --- |
| identity | stable node identifier / key |
| node | one shared entity or concept node |
| edge | typed relationship to an existing target |
| property | typed literal on the owner |
| excluded | intentionally omitted with a recorded rule |

`SemanticColumnProfiler` does not create a node merely because a column has few values. A value such as `status=active` normally stays a scalar property. `department=Engineering` can become a shared `Department` concept because the dimension is independently reusable across employees, projects, costs, and policies.

Exit gate:

- every included column has both decisions;
- datatypes and null semantics are explicit;
- low cardinality alone never promotes a column; and
- excluded data has an auditable rule.

### Stage S2 — identity design

Identity is resolved before edges.

- Declared business keys are preferred to row numbers.
- Composite keys are serialized as canonical JSON arrays, not delimiter-joined strings, so `["US", "A|1"]` cannot collide with `["US|A", "1"]`.
- `source` scope keeps equal keys from independent sources distinct.
- `global` scope is permitted only for a governed key.
- `cross_source` scope requires an explicit resolver and retains its confidence and evidence.
- Missing/incomplete/redacted keys receive source-row identities; names are not silently promoted to global identifiers.

`ResolvedModelingPlan` makes identity keys and scope part of the schema contract. `ExactKeyEntityResolver` reconciles aliases only after all source mappers have emitted their observations.

Exit gate:

- key uniqueness is checked in its declared scope;
- collision tests pass for every composite key;
- false merges and false splits are measured on gold cross-source pairs; and
- all edges are rewritten to current canonical IDs.

### Stage S3 — categorical values and controlled vocabularies

The mapper distinguishes:

1. scalar enumeration — a literal property (`status`, `priority`);
2. governed concept — one SKOS-like concept keyed by `(scheme, code)` with preferred/alternate labels; and
3. business entity — a node with independent identity and lifecycle.

Canonicalization uses Unicode normalization, trimming, and case folding according to the declared rule while retaining `raw_value`. A closed vocabulary rejects or quarantines an unknown code. The concept ID contains the scheme and code, so the same label in two schemes does not collapse.

`SemanticTableMapper` creates one shared concept candidate and qualifier-aware edges from all owners. Resolution and graph builders deduplicate it.

Exit gate:

- categorical duplication rate is exactly `0`;
- each concept has one preferred label per language;
- unknown closed-scheme codes are not promoted; and
- scalar enumerations remain properties.

### Stage S4 — relationship formation

Relationships are formed from evidence, not column-name coincidence.

- A resolved foreign key becomes a typed edge to the actual target-table entity.
- A missing target retains the raw foreign-key value and emits no fabricated placeholder.
- A pure two-foreign-key join table becomes one direct relationship.
- A join table with attributes, temporal validity, provenance, or more than two roles becomes an association/event node.
- Relationship direction, domain, range, cardinality, source columns, target columns, and missing-target policy are part of the plan.
- Repeated observations with different qualifiers are not deduplicated as if they were the same assertion.

Exit gate:

- dangling edge rate is `0`;
- foreign-key join precision/recall meet the gold threshold;
- pure joins do not create redundant row nodes;
- qualified joins preserve every qualifier; and
- domain/range and cardinality constraints pass.

### Stage S5 — one plan, six schema projections

The resolved plan is the single source for:

1. property-graph schema;
2. JSON-LD context;
3. OWL 2 RL ontology;
4. SHACL shapes;
5. Neo4j uniqueness/existence constraints; and
6. evaluation expectations.

`ResolvedModelingPlan.assert_schema_agreement()` parses and compares these projections. This prevents the common failure where mapper code, documentation, database constraints, and evaluation fixtures describe different graphs.

Exit gate:

- all six projections agree on classes, keys, predicates, datatypes, domain/range, and cardinality;
- OWL and SHACL artifacts parse; and
- Neo4j node-key/uniqueness constraints exist for governed identities.

### Stage S6 — materialize, validate, and publish

Construction emits property, RDF, and evidence graphs from the same `ConstructionBatch`. RDF assertions point to PROV evidence resources; property nodes and edges retain source selectors and extractor metadata.

SHACL runs against the generated RDF graph. Referential integrity and duplicate checks run on the property graph. Invalid candidates are rejected or quarantined according to policy; they are not silently copied to a store.

## Pillar 2: unstructured data to knowledge graph

### Stage U0 — immutable document versions

`DocumentRef` identifies the source name, type, content digest, text length, and document version. Every mention, evidence span, and claim must reference that exact version.

Exit gate: a changed document creates a new version and cannot satisfy evidence from the previous version.

### Stage U1 — structural segmentation

Documents are split into paragraphs/sentences/sections while retaining absolute character offsets, exact quote, prefix, and suffix. Start is inclusive; end is exclusive.

Exit gate: every segment reproduces `document_text[start:end]` exactly.

### Stage U2 — mandatory mention ensemble

Every applicable profile runs:

- deterministic exact-pattern detection;
- GLiNER zero-shot entity detection;
- Transformers token-classification NER;
- spaCy statistical NER; and
- domain labels declared in the configuration.

Provider results are normalized only after offset validation. The union is retained with provider-agreement ratios; a disagreement is information, not a reason to erase a supported mention. Any mandatory provider failure fails the stage after the remaining providers have also been attempted.

Exit gates:

- entity and exact-span precision/recall/F1;
- label accuracy;
- boundary-error rate;
- provider agreement/support; and
- complete model revision and configuration provenance.

Exact-span evaluation is intentionally separate from label/name evaluation. End-to-end entity-linking research has shown that aggregate scores can conceal mention-definition, ambiguity, NIL, and benchmark-bias errors; the release contract therefore uses exact `(document version, start, end, type, occurrence)` matching and reports NIL behavior explicitly. See the [EMNLP 2023 in-depth entity-linking evaluation](https://aclanthology.org/2023.emnlp-main.411/). GLiNER remains a separate mandatory provider—not a synonym for LLM generation—because its [NAACL 2024 evaluation](https://aclanthology.org/2024.naacl-long.300/) targets open-label, parallel NER. Provider union and agreement therefore expose what the specialist recognizer contributed independently of spaCy and token classification.

### Stage U3 — coreference and identity layers

The graph keeps separate:

- `Mention`: one textual occurrence;
- observed/coreference identity: a document-local cluster; and
- `CanonicalEntity`: an identity created by a resolver or external linker.

Conservative rules resolve exact safe cases. Ambiguous cases remain NIL/unresolved and can be quarantined. They are never forced to the nearest name.

Exit gates: pairwise/cluster F1, false-merge rate, false-split rate, NIL accuracy, and record-universe coverage.

### Stage U4 — claims, relations, events, facts, and intent

Every platform runs deterministic patterns, GLiNER RelEx, and structured Qwen3 generation:

- Transformers Qwen3 on all profiles;
- MLX Qwen3 additionally on Apple Silicon; and
- the locked Qwen3 Ollama model on all profiles.

The structured response schema covers subject, predicate, object, event roles, evidence, negation, modality, attribution, time, quantity, and confidence. Values and cues require exact spans. Relation/fact/intent claims require an object; events require typed roles.

The intermediate `Claim` is not flattened prematurely. It is an immutable, reified assertion with provenance and review state. The joint GLiNER RelEx provider follows the entity-plus-relation formulation described by [GLiNER-Relex](https://arxiv.org/abs/2605.10108), while the independent Qwen and deterministic providers prevent the release contract from relying on one model family or one decoding path.

Exit gates:

- relation and triple precision/recall/F1;
- event-role F1;
- separate intent and fact F1;
- negation, modality, temporal, quantity, and attribution F1; and
- unsupported-claim rate.

Document-level held-out evaluation is essential: datasets such as [DocRED](https://aclanthology.org/P19-1074/) demonstrate that relations can cross sentence boundaries and require supporting evidence, so sentence-only success is not treated as complete extraction.

### Stage U5 — evidence and provenance promotion gate

A claim is promotable only when:

- the document digest/version matches;
- every evidence and semantic span is exact;
- extractor, provider, model revision, configuration ID, and confidence exist;
- confidence meets policy;
- the predicate is aligned to the governed ontology; and
- validation passes.

Anything else is quarantined. Exact evidence coverage and provenance coverage are both required to be `1.0` in the release fixture.

### Stage U6 — layered graph materialization

The final batch materializes document, mention, observed identity, canonical identity, claim, and event resources separately, then adds safe direct semantic edges for convenient querying. Reified claims preserve qualifications that a bare `(subject, predicate, object)` edge would lose.

## Pillar 3: evaluation and metric gates

### Stage E0 — declare the use and gold contract

Every evaluation run records dataset version/hash, source inventory, train/dev/test split, annotation guidelines, adjudication status, model/service locks, code tree, and random seeds. Gold extraction, entity-resolution, query, retrieval, and recommendation judgments are separate fixtures.

Do not tune thresholds on the test split. Report support counts and confidence intervals alongside point estimates.

### Stage E1 — universal formation metrics

These gates apply before any downstream benchmark.

| Dimension | Metrics |
| --- | --- |
| identity/conciseness | duplicate-node rate, categorical-duplication rate, literal-that-should-be-node rate |
| integrity | dangling-edge rate, referential-integrity score, duplicate-edge rate |
| semantic consistency | SHACL violations, datatype violations, domain/range violations, cardinality violations |
| evidence/trust | exact-evidence coverage, provenance coverage, unsupported-extraction rate |
| completeness | class/property/relationship completeness and source-to-graph coverage |
| structure | orphan rate, component count, degree distribution, relationship/attribute richness |
| operations | latency percentiles, throughput, memory, graph-write time, retry/error rate |

Hard defaults include zero categorical duplicates, dangling edges, unsupported promoted claims, and SHACL violations.

### Stage E2 — task metrics

Gold construction:

- entity/span/relation/event-role/intent/fact precision, recall, and F1;
- exact and relaxed triple match;
- qualifier preservation; and
- claim coverage versus unsupported-claim rate.

Entity resolution:

- pairwise and cluster precision/recall/F1;
- false-merge and false-split rates;
- NIL precision/recall; and
- prediction/gold record-universe overlap.

Schema and data integration:

- key and datatype preservation;
- foreign-key/join accuracy;
- association qualifier preservation;
- source-field coverage; and
- structured round-trip query correctness.

### Stage E3 — evaluate the actual intended use

Intrinsic metrics do not prove usefulness. The same frozen graph is evaluated on each declared consumer.

| Intended use | Required metrics |
| --- | --- |
| SQL-like/property-graph querying | query success, result precision/recall, aggregation accuracy, p50/p95 latency |
| NoSQL traversal | path-answer accuracy, traversal coverage, fan-out/latency, write/read correctness |
| GraphRAG / QA | Recall@k, MRR/nDCG, path relevance, multi-hop success, answer faithfulness, citation accuracy |
| vector store | index coverage, Recall@k, MRR, nDCG, ANN latency/memory, filtered-retrieval correctness |
| link prediction | filtered MRR and Hits@k, with relation/entity popularity strata |
| recommendation | Precision/Recall/nDCG@k, coverage, novelty/diversity, and downstream task success |
| analytics/reasoning | competency-question accuracy, inferred-fact precision/recall, rule consistency, runtime |

[KGrEaT](https://arxiv.org/abs/2308.10537) motivates fixed downstream tasks in addition to correctness/completeness. Standard KGE MRR and Hits@k are retained, but results are stratified because [popularity can bias those metrics](https://proceedings.mlr.press/v124/mohamed20a.html). Neo4j vector search is approximate, so exact-neighbour recall and filtered retrieval are measured rather than inferred from index availability. The GraphRAG fixture separately judges construction, retrieval, multi-hop paths, answers, and citations, following the end-to-end decomposition emphasized by [GraphRAG-Bench](https://arxiv.org/abs/2506.02404). Requiring both supporting paths and answer evidence also reflects [MultiHop-RAG](https://arxiv.org/abs/2401.15391), where multi-hop retrieval and answering are evaluated against associated evidence rather than final-answer accuracy alone.

For vector use, the live evaluator fixes the embedding dimension, waits for an online index, executes ANN retrieval, checks equality filters, and measures Recall@k, latency, and memory. This follows Neo4j's current guidance that declaring [`vector.dimensions`](https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/) adds validation and that vector lookup is approximate.

### Stage E4 — statistical confidence and regression policy

- Bootstrap intervals are used for continuous/sample means.
- Wilson intervals are used for proportions.
- Metrics are sliced by source, class, predicate, document length, entity popularity, and confidence band.
- Candidate comparisons use paired examples and report deltas, not only ranks.
- A release fails on a hard-gate regression even when a weighted aggregate score rises.

### Stage E5 — executable report

Each observation contains:

- metric and purpose;
- value and support;
- threshold and comparison operator;
- `executed`;
- pass/fail;
- fixture and output hashes; and
- failure reason.

The release result is the conjunction of mandatory gates. Weighted rankings can help compare research candidates but cannot override a correctness gate. This evidence model is deliberately compatible with the W3C [Data Quality Vocabulary](https://www.w3.org/TR/vocab-dqv/) distinction between quality dimensions, metrics, measurements, policies, and certificates: the versioned threshold contract is the policy, each named observation is a measurement, and the nonce-bound hard-pass report is the certificate.

This separation of intrinsic formation quality from ability/downstream quality follows the practical distinction made in [Steps to Knowledge Graphs Quality Assessment](https://arxiv.org/abs/2208.07779) and [KGrEaT](https://arxiv.org/abs/2308.10537). It also prevents a circular benchmark: expected judgments are versioned inputs, while predictions are computed by workload runners from the frozen graph snapshot and are never accepted from the caller.

## Mandatory runtime stage

`models.manifest.yaml` defines mutually exclusive platform profiles. The product dependency set has no extras. Provisioning generates immutable dependency, service-image, model-file, and Ollama-content locks. Certification then runs with dependency/model registries disabled (while live remote-linking checks remain live) and requires:

```text
declared = configured = instantiated = executed = evaluated = passed
```

No capability is credited for importability alone. The dispatcher creates a fresh nonce, runs the capability against a fixture, requires semantic metrics and named assertions, and rejects skipped/fallback output.

External reconciliation follows the [Reconciliation Service API protocol](https://reconciliation-api.github.io/specs/latest/) through the Wikimedia Cloud Wikidata service. Apple generation uses the locked local snapshot through [MLX-LM](https://github.com/ml-explore/mlx-lm); it is a required `darwin-arm64` execution backend, not a fallback for Transformers or Ollama. All Hugging Face-backed providers receive exact local snapshot paths under registry-disconnected execution, consistent with Transformers' [documented offline mode and `local_files_only`](https://huggingface.co/docs/transformers/v4.49.0/en/installation).

## Implementation map

| Contract | Main implementation |
| --- | --- |
| structured profiling and role/materialization | `application/semantic_profiler.py` |
| resolved plan and six projections | `application/resolved_plan.py` |
| cross-table keys/foreign keys | `application/source_catalog.py` |
| semantic materialization | `application/semantic_table_mapper.py` |
| unstructured evidence pipeline | `application/unstructured_pipeline.py` |
| claim and identity contracts | `domain/claim.py` |
| mandatory local model ensemble | `infrastructure/extractors/model_backed/` |
| graph construction | `infrastructure/graph_builders/` |
| SHACL/ontology | `infrastructure/ontology/` |
| quality and use-specific metrics | `infrastructure/evaluation/` |
| trusted release fixture/config/source binding | `application/publication_contract.py`, `resources/publication_contracts_v1.yaml` |
| immutable hard pass | `certification/`, `scripts/certify.py`, `run_all_approaches.sh` |

Regression fixtures live in `tests/fixtures/gold_standards/`. Run:

```bash
make modelling-correctness
make schema-quality
make test
make quality
bash run_all_approaches.sh
```
