# Multi-Source Knowledge Graph Correctness and Benchmark Plan

Status: implementation plan, reviewed against the repository on 2026-07-21.

This document defines how this repository should construct, evaluate, and publish
knowledge graphs from heterogeneous sources without silently collapsing distinct
real-world things. It is intentionally stricter than a feature list. It separates:

- what the repository implements now;
- what the current tests and reports actually prove;
- what must be added before a dataset can be called a repository benchmark; and
- what must be demonstrated before a graph can be recommended for a domain.

The central rule is not simply “remove duplicates.” A trustworthy graph must merge
records only when there is sufficient evidence that they denote the same thing,
retain independent assertions and their provenance, and preserve distinct events or
namesakes. False merges are usually more damaging than visible false splits because
they can transfer attributes and relationships to the wrong entity.

## 1. What “Best Graph” Means

There is no universally best graph for a dataset. A graph is best only relative to a
declared identity policy, ontology, use case, gold standard, and risk budget. This
repository should select a **publishable winner** in the following order. A separate
diagnostic ranking may retain every weak or failed baseline so failures stay visible,
but an ineligible baseline must not be presented as production-safe:

1. Reject every candidate that fails a non-negotiable correctness or safety gate.
2. Compare the remaining candidates on gold-labelled task metrics.
3. Compare downstream competency-query or retrieval utility.
4. Compare measured latency, throughput, memory, and operating cost.
5. Prefer the simplest reproducible candidate when results are statistically tied.

A high weighted score must never override a false-merge, referential-integrity,
evidence, privacy, or schema-conformance gate. A deterministic smoke leaderboard is
useful for regression testing, but it is not evidence that a technique is accurate.

The current publication contract is documented in
[`benchmarking_and_publication.md`](benchmarking_and_publication.md). The committed
gold correctness starting point is
[`../benchmarks/cross_source_identity/v1/`](../benchmarks/cross_source_identity/v1/),
with published results in
[`../reports/cross_source_identity_benchmark.md`](../reports/cross_source_identity_benchmark.md).

## 2. Terms and Identity Layers

The implementation and benchmark fixtures must distinguish the following objects.
Using one identifier for all of them is a common cause of unsafe deduplication.

| Layer | Meaning | Required identity |
|---|---|---|
| Source system | CRM, billing, publication index, API, file collection | Governed source-system ID |
| Source snapshot | One observed version of a file, table, response, or document | Content/config/time-qualified snapshot ID |
| Source record | One row, object, message, document span, or event in a source | Source-scoped record ID |
| Candidate | A mapper/extractor assertion that a record denotes an entity or relation | Deterministic candidate ID plus evidence |
| Canonical entity | A governed cluster of candidates believed to denote one real thing | Stable canonical ID independent of display name |
| Semantic assertion | A timeless or validity-qualified statement between canonical entities | Canonical endpoints, predicate, qualifiers |
| Occurrence | A purchase, message, observation, transaction, or other event instance | Event/occurrence ID or conservative separate identity |
| Evidence | The exact source location supporting a candidate or assertion | Content-addressed evidence ID |
| Resolution decision | A merge, non-merge, split, or review outcome | Versioned decision ID, policy/model version, rationale |

“Duplicate” must therefore always be qualified. Duplicate source snapshots, duplicate
candidate emissions, duplicate semantic assertions, and co-referent entity records
are different problems. Repeated independent evidence for the same semantic assertion
must be consolidated as support, not discarded as noise.

## 3. Repository Implementation Map

The current end-to-end path is:

```text
load/profile
  -> source-type mapping and/or extraction
  -> candidate entities, relations, facts, and evidence
  -> one or more entity resolvers
  -> transitive endpoint reconciliation
  -> evidence and referential-integrity validation
  -> property, RDF, and evidence graph materialization
  -> gold metrics, hard gates, reports, and recommendation
```

Important implementation locations are:

| Concern | Current implementation |
|---|---|
| Public loading and heterogeneous routing | `application/public_api.py` |
| Workflow ordering and default validation | `application/workflow.py` |
| Source, evidence, candidate, fact, and batch models | `domain/models.py` |
| Structured row identity | `infrastructure/mapping/direct_table_mapper.py` |
| R2RML/RML/CSVW mapping subset | `infrastructure/mapping/standards.py` |
| Exact, normalized, alias, fuzzy, probabilistic/adapter resolution | `infrastructure/resolution/` |
| Canonical endpoint reconciliation and merge audit | `infrastructure/resolution/common.py` |
| Evidence and referential-integrity rules | `infrastructure/ontology/validation.py` |
| SHACL execution | `infrastructure/ontology/shacl.py` |
| Canonical property graph construction | `infrastructure/graph_builders/property_graph.py` |
| RDF plus provenance named graphs | `infrastructure/graph_builders/rdf_graph.py` |
| Source/evidence/candidate audit graph | `infrastructure/graph_builders/evidence_graph.py` |
| Gold manifests, metrics, hard gates, run metadata | `application/benchmark_runner.py` |
| ER and graph-hygiene metrics | `infrastructure/evaluation/` |
| Markdown, CSV, and JSON publications | `infrastructure/reports/` and `interfaces/cli/benchmark.py` |
| Synthetic cross-source identity generator | `scripts/generate_cross_source_identity_benchmark.py` |

### 3.1 Implemented now

The following capabilities have executable code and regression coverage now:

- canonical JSON/source hashing that is independent of object-key order, and for row
  snapshots independent of row order;
- source-record de-duplication when pipeline component batches are merged;
- per-source mapping/extraction routing for heterogeneous directories;
- conservative source-scoped identity keys for default structured row mapping;
- evidence-backed candidates, facts, and source locations;
- exact, normalized-string, alias-table, and fuzzy resolution paths, plus optional
  Splink, Dedupe, embedding, cross-encoder, OpenRefine, Wikidata, and DBpedia adapters;
- merge audit metadata, canonical entity IDs, and transitive reconciliation of
  relations and facts after sequential resolvers;
- referential-integrity validation before graph construction;
- deterministic canonical-node field survivorship with value conflicts, field
  sources, candidate provenance, and evidence union;
- semantic relation consolidation while preserving event-like occurrences with
  occurrence qualifiers;
- property, RDF, and evidence graph outputs;
- an RDF builder that uses `prov:wasDerivedFrom` and separates data and evidence
  named graphs;
- versioned gold manifests, input/config/gold hashes, environment/run metadata,
  gold task support counts, explicit hard gates, and Markdown/CSV/JSON output;
- pairwise ER precision/recall/F1, false-merge and false-split rates, extraction and
  triple metrics, retrieval metrics when predictions exist, and graph hygiene
  metrics; and
- a committed three-source identity fixture with aliases and a deliberate namesake,
  showing that name-only baselines fail while the scoped trusted-identifier reference
  passes this synthetic test.

Regression proof points include `tests/test_resolution_integrity.py`,
`tests/test_canonical_graph_correctness.py`, and
`tests/test_publishable_benchmarks.py`.

### 3.2 Not implemented or not yet proven

The following must not be inferred from the current implementation:

- The repository does **not** guarantee a globally duplicate-free graph. Identity is
  domain- and policy-dependent. Exact-name fallback is source-scoped by default,
  while explicit global-name, normalized, and alias baselines can still merge
  namesakes and therefore require labelled evaluation.
- There is no durable, versioned identity registry that survives independent runs,
  supports explicit cannot-link constraints, or performs governed split/unmerge
  operations.
- Resolver adapters being registered does not mean they have been calibrated on a
  domain. Splink, Dedupe, embeddings, and cross-encoders require data-specific
  features, thresholds, training or review, and leakage-safe evaluation.
- Default structured mapping is intentionally conservative, but source-local keys are
  not automatically promoted into issuer-qualified enterprise identifiers.
- Loader PK/FK metadata and configured mapper foreign keys do not yet form a complete
  automatic multi-table semantic model or global ontology mapping.
- Field survivorship is deterministic, but the default policy is confidence/evidence/
  candidate-order based; it is not yet a declarative per-field authority and temporal
  policy.
- The RDF output uses part of PROV-O, but the workflow does not yet serialize full
  `prov:Activity`, `prov:Agent`, qualified derivations, resolution decisions, or DQV
  quality measurements.
- RDF output is de-duplicated and serializable, but the repository does not implement
  or claim conformance with W3C RDFC-1.0 canonicalization.
- SHACL execution exists, but domain shapes and SHACL conformance are not part of every
  default workflow or every committed benchmark.
- R2RML/RML/CSVW functionality is an executable subset, not a completed W3C/RML
  conformance certification.
- The small cross-source fixture does not test scale, class imbalance, blocking
  recall, multilingual values, field survivorship, updates, deletes, temporal
  validity, cluster repair, or domain utility.
- Current deterministic runtime controls are regression inputs, not measured speed or
  resource results.
- No public domain pack listed later in this document is a repository release
  benchmark until its manifest, license review, frozen data, gold labels, tests, and
  published reports are committed.

## 4. Normative Data-Modelling and Identity Invariants

The words MUST, MUST NOT, SHOULD, and MAY below define the target construction
contract. New domain packs should turn each applicable invariant into an executable
test or hard gate.

### 4.1 Source and record identity

1. Every source system MUST have a stable governed identifier separate from a file
   path or display label.
2. Every source snapshot MUST record source ID, retrieval time, content checksum,
   parser version, mapping version, schema version, and permitted-use metadata.
3. A source record MUST have an identity scoped to the source system and logical
   collection. A bare integer, email address, name, or redaction token is not a global
   identity.
4. Canonical serialization MUST make snapshot and candidate identity independent of
   dictionary insertion order and ingestion order where order has no domain meaning.
5. Replaying the same snapshot and configuration MUST be idempotent.
6. Changed content without an explicit monotonic watermark MUST be treated as a new
   snapshot, not compared lexicographically as though a hash were a timestamp.

### 4.2 Identifier policy

7. Every identifier used for cross-source resolution MUST declare an issuer or
   namespace, entity type, normalization rule, uniqueness scope, reuse policy, and
   temporal validity.
8. Display names MUST be treated as matching features, never as sufficient identity
   proof for people, organizations, products, accounts, or places.
9. A trusted identifier conflict MUST create a cannot-link constraint or review item;
   a weaker name similarity MUST NOT override it.
10. Missing or redacted identifier values MUST NOT collapse into one constant
    identity. Conservative splitting is preferred until stronger evidence exists.
11. Identifier reuse over time, mergers, spin-offs, household/shared contact fields,
    and many-to-one source mappings MUST be represented explicitly.

### 4.3 Candidate, canonical entity, and resolution decision

12. Candidate IDs MUST remain traceable after resolution. A merge changes cluster
    membership; it must not erase the source candidate.
13. Canonical IDs MUST be independent of mutable display names and winning field
    values.
14. Each merge MUST retain members, resolver/policy/model version, score, threshold,
    features or rule reason, timestamp, reviewer when applicable, and evidence.
15. Each resolver chain MUST flatten aliases and rewrite every relation/fact endpoint
    to current canonical IDs before materialization.
16. Resolution MUST support positive links and hard negative/cannot-link constraints.
17. Clustering MUST enforce transitivity without allowing one weak bridge to violate a
    cannot-link or trusted-identifier constraint.
18. Merge decisions SHOULD be reversible; split/unmerge operations MUST produce a new
    cluster version and repair affected relationships without losing history.

### 4.4 Assertions, relations, and events

19. Candidate assertions, canonical semantic relations, and event occurrences MUST be
    separate concepts.
20. Equivalent semantic relations MAY consolidate only after endpoints are canonical
    and semantic qualifiers match.
21. Independent evidence for a consolidated relation MUST be unioned and countable.
22. Event-like relations MUST retain an occurrence ID or temporal/transaction
    qualifier. When occurrence identity is unknown, the system MUST keep candidates
    separate rather than over-merge them.
23. Conflicting facts MUST remain attributable and queryable. A winning value MUST NOT
    delete losing assertions.
24. Temporal facts SHOULD use explicit valid-time and observed/transaction-time fields
    rather than overwriting prior values.

### 4.5 Schema and ontology governance

25. Every domain benchmark MUST pin ontology/mapping/shape versions.
26. Entity types, predicates, domains, ranges, cardinalities, required properties,
    controlled values, and datatype/unit rules MUST be declared or explicitly marked
    open-world.
27. Source schema changes MUST be detected before silently changing mappings.
28. New labels and predicates MUST enter a proposal/quarantine path before promotion
    into a governed namespace.
29. RDF mappings SHOULD be tested against W3C R2RML/direct-mapping cases where
    applicable; RDF domain outputs SHOULD pass SHACL shapes.
30. Property-graph and RDF outputs from the same batch SHOULD satisfy equivalent
    identity, assertion, and evidence invariants.

### 4.6 Provenance, quality, and privacy

31. Every published entity, relation, and fact MUST have source-locatable evidence or
    an explicit approved exception.
32. Provenance MUST distinguish source entities, construction activities, software or
    human agents, generated candidates, and resolution decisions.
33. Field-level winners and conflicts MUST identify supporting candidates and sources.
34. Quality measurements MUST identify metric definition, value, evaluated dataset,
    computation activity, time, and responsible agent.
35. PII handling MUST occur before unsafe logs, reports, embeddings, or external
    services are invoked. Raw secrets and PII MUST NOT appear in committed benchmark
    artifacts.
36. Redaction and hashing MUST not be described as anonymization unless the threat
    model and re-identification risk have been evaluated.
37. Source withdrawal or deletion MUST be able to remove its unsupported assertions
    and recompute cluster/field support.

## 5. Target Construction Architecture

The target pipeline should retain the current evidence-first batch model while adding
governed identity and benchmark boundaries.

### Stage A — Source contract and profiling

- Register source system, owner, license, sensitivity, retention, and trust tier.
- Freeze a source manifest with checksums and retrieval instructions.
- Profile columns/paths/types/nulls/distinctness and compare them with the prior
  schema version.
- Decide which fields are identifiers, quasi-identifiers, matching features,
  assertions, event qualifiers, and prohibited data.
- Route each source independently; never choose one mapper for a heterogeneous batch.

Exit criteria: all source records have deterministic source-scoped IDs, evidence
locations, parser versions, and no prohibited data in diagnostic output.

### Stage B — Semantic mapping and candidate generation

- Use R2RML/RML/CSVW/JSONPath or domain mapping specs where possible.
- Emit candidates without prematurely choosing global canonical IDs.
- Preserve raw-to-normalized transformations and mapping-rule IDs.
- Run extraction models only on source types and fields for which they were validated.
- Reject unsupported model output or send it to review; never fabricate evidence.

Exit criteria: mapping coverage is measured, required schema rules pass, and every
candidate is attributable to a source location and construction activity.

### Stage C — Identity resolution

Use a tiered policy rather than one universal matcher:

1. Validate and normalize trusted issuer-qualified identifiers.
2. Apply deterministic match and cannot-link rules.
3. Generate candidate pairs with multiple blocking rules and measure blocking recall.
4. Score remaining pairs with calibrated probabilistic or supervised models.
5. Partition decisions into auto-link, review, and non-link bands.
6. Cluster with entity-type, identifier, cannot-link, temporal, and transitivity
   constraints.
7. Version cluster membership and emit a complete decision audit.

The matching feature set must be domain-specific. For example, organization matching
may combine LEI, country, legal name, registered address, and parent relationships;
person matching must treat shared contact details and namesakes as hard risks; product
matching should use brand, model, GTIN/MPN, capacity, and category compatibility.

Exit criteria: gold pair/cluster metrics and subgroup slices meet gates, all high-risk
conflicts are reviewed or kept split, and record order does not change clusters.

### Stage D — Canonicalization and survivorship

- Assign stable canonical IDs from the identity registry, not from mutable winners.
- Choose field values with a declarative rule per field: source authority, freshness,
  completeness, validation status, and confidence.
- Preserve all supporting and conflicting assertions.
- Consolidate timeless semantic edges; preserve distinct event occurrences.
- Recompute endpoints after every merge/split decision.

Exit criteria: source order, batch order, and full-versus-incremental construction
produce equivalent canonical graphs under the declared comparison function.

### Stage E — Validation and publication

- Run evidence, referential-integrity, schema registry, datatype, cardinality, and
  domain SHACL validation.
- Evaluate identity, extraction, mapping, graph hygiene, provenance, temporal replay,
  competency queries, retrieval, and actual operational performance separately.
- Apply hard gates before selecting a publishable winner; keep an all-baseline
  weighted ranking explicitly labelled as diagnostic or smoke output.
- Publish failed baselines alongside passing ones and state task support explicitly.

Exit criteria: a versioned manifest, frozen gold, hashes, environment metadata,
confidence intervals, limitations, and Markdown/CSV/JSON reports can be regenerated.

## 6. Evaluation Framework

### 6.1 Evaluation layers and metrics

| Layer | Required metrics | Notes |
|---|---|---|
| Source ingestion | source/record coverage, parse failures, schema drift, checksum/replay equality | Measure against expected source inventory and rows/documents, not emitted node count alone. |
| Mapping | subject/predicate/object accuracy, datatype/language/unit accuracy, join accuracy, mapping coverage | Add W3C/KROWN conformance cases; report unsupported mapping features. |
| Entity extraction/linking | exact and relaxed entity/span precision, recall, F1; type accuracy; NIL accuracy | Slice by entity type, source, language, and seen/unseen entity. |
| Relation/fact extraction | relation and triple precision, recall, F1; direction; qualifier/temporal accuracy | Evidence-location accuracy should be scored separately. |
| Entity resolution | pairwise P/R/F1, false-merge and false-split rates, B-cubed P/R/F1, cluster exact match, blocking recall | Pairwise metrics exist now; B-cubed, exact-cluster, and blocking metrics are next. |
| Canonical fields | per-field value accuracy, completeness, conflict retention, source-attribution accuracy | Evaluate winners and retained alternatives. |
| Graph hygiene | dangling-edge rate, semantic duplicate-edge rate, orphan rate, SHACL/schema violation rate | A legitimate repeated event is not a duplicate edge. |
| Provenance | evidence coverage, resolvable locator rate, field lineage accuracy, decision-audit completeness | Add PROV/DQV conformance and provenance competency queries. |
| Change behavior | replay idempotence, source-order invariance, batch/incremental equivalence, deletion propagation, cluster churn | Evaluate snapshots as sequences, not independent static files. |
| Downstream utility | competency-query precision/recall, answer accuracy, Recall@k, MRR/NDCG, path relevance, citation accuracy | Retrieval/GraphRAG utility complements, not replaces, graph correctness. |
| Operations | measured throughput, p50/p95/p99 latency, peak memory, graph write time, update lag, cost | Record hardware, backend, warm-up, repetitions, and variance. |
| Safety/fairness | PII leakage, subgroup false-merge/split rates, review load, abstention quality | Domain and legal review are required for people data. |

All proportions must include numerator, denominator, and fixture/task version. Report
95% confidence intervals when support is large enough. For paired candidate
comparisons, use bootstrap or an appropriate paired test and report effect size; do not
declare a winner from an insignificant decimal difference.

### 6.2 Split and leakage policy

Each domain pack must state which generalization it tests:

- **record split:** useful only when the same entities may appear across splits;
- **entity-disjoint split:** tests unseen entities and prevents entity memorization;
- **source-disjoint split:** tests transfer to a new source/system;
- **time split:** trains/configures on earlier snapshots and tests later changes;
- **schema-version split:** tests mapping robustness to changed source structure; and
- **hard-negative challenge split:** concentrates namesakes, shared addresses,
  aliases, reused identifiers, and near-duplicate events.

Gold records, cluster IDs, target labels, and direct derivatives MUST NOT be present in
model inputs. Thresholds and mapping rules are selected on development data and frozen
before test evaluation. Any manual inspection of test labels must be recorded as test
set exposure and trigger a new held-out test version for future claims.

### 6.3 Hard gates

The generic runner currently supports these defaults from
`configs/evaluation.example.yaml`:

| Gate | Current default |
|---|---:|
| Compatible gold task evaluated | required |
| Dangling-edge rate | 0 |
| Semantic duplicate-edge rate | 0 |
| Evidence coverage | at least 0.95 |
| Validation failure rate | at most 0.05 |
| False-merge rate when ER gold exists | at most 0.02 |
| Entity F1 when entity gold exists | at least 0.80 |
| Relation F1 when relation gold exists | at least 0.80 |
| Exact triple match when triple gold exists | at least 0.80 |
| Pairwise ER recall when ER gold exists | at least 0.80 |
| Retrieval Recall@k when retrieval gold and predictions exist | at least 0.70 |

These are framework defaults, not universal production thresholds. Each domain pack
must justify stricter or additional gates. Recommended next mandatory gates are:

- trusted-identifier conflicts: zero auto-merges;
- namesake/shared-contact challenge false merges: zero in the reviewed challenge set;
- source-order and batch-order invariance: exact pass;
- full rebuild versus incremental replay equivalence: exact pass after canonical
  normalization;
- source withdrawal: zero remaining assertions supported only by the withdrawn source;
- SHACL violations at severity `Violation`: zero for release graphs;
- evidence locator resolution and decision-audit completeness: 100%; and
- subgroup false-merge rate: no subgroup exceeds its risk budget.

### 6.4 Baselines

Every identity benchmark should publish at least:

1. no-merge baseline;
2. exact normalized name/type baseline;
3. controlled alias or legal-suffix baseline;
4. trusted-identifier deterministic baseline where identifiers exist;
5. probabilistic record-linkage baseline, such as
   [Splink](https://github.com/moj-analytical-services/splink);
6. supervised/active-learning baseline, such as
   [Dedupe](https://docs.dedupe.io/en/latest/), when labelled data and the optional
   dependency are available; and
7. the proposed resolver configuration.

Weak baselines stay in the report when they fail. This demonstrates why an apparently
reasonable name-based deduplicator is unsafe and prevents selective reporting.

## 7. Benchmark Pack Contract

### 7.1 Repository layout

Each benchmark should use a versioned, self-contained control directory:

```text
benchmarks/<benchmark_id>/v<major>/
  README.md
  benchmark_manifest.yaml
  schemas/                  # mapping, ontology, JSON Schema, SHACL
  source_manifests/         # URLs, retrieval date, license, checksums
  fixtures/                 # only redistributable small/frozen inputs
  gold/                     # independent labels and adjudication metadata
  configs/                  # frozen candidate/baseline configurations
  expected/                 # deterministic invariant outputs, not claimed accuracy
scripts/generate_<benchmark_id>.py
reports/<benchmark_id>.md
reports/<benchmark_id>.csv
reports/<benchmark_id>.json
```

Large or restricted inputs must be downloaded into ignored locations. Commit retrieval
instructions, exact checksums, transformations, and licence/terms references instead
of copying data into Git. Public access does not automatically permit redistribution.

### 7.2 Required manifest fields

Extend the current `GoldDatasetManifest` contract so every release pack declares:

- schema version, benchmark ID, semantic version, domain, task, and status;
- description, maintainers, publication date, license, citations, and source terms;
- source snapshots, checksums, record counts, and redistribution mode;
- ontology, mapping, shapes, normalization, identity-policy, and config hashes;
- train/dev/test or time/source split IDs and leakage policy;
- gold fixture paths, annotation guide, annotator/adjudicator counts, agreement, and
  unresolved labels;
- hard gates, primary metric, secondary metrics, subgroup slices, and confidence
  interval method;
- baselines and exact software/model/dependency versions;
- expected hardware/resource profile for scale runs;
- known limitations, intended use, and prohibited use; and
- generation command and expected report paths.

### 7.3 Benchmark status

Use explicit status instead of calling every fixture a benchmark:

| Status | Meaning | Required evidence |
|---|---|---|
| `smoke` | Executability/regression only | Deterministic inputs and expected code-path behavior |
| `gold-candidate` | Separate gold labels exist, but independence, adjudication, scale, or domain coverage is incomplete | Manifest, hashes, gold metrics, limitations |
| `publishable-gold` | Reproducible correctness result on the declared fixture | Passing publication contract, regenerated reports, independent gold |
| `domain-release` | Representative enough for a scoped domain recommendation | License review, leakage-safe splits, domain review, hard negatives, temporal and utility tests |
| `scale-release` | Reproducible operational comparison | Frozen generator/data, hardware profile, repeated measured runs, variance |

The existing cross-source identity v1 fixture is published as a `gold-candidate`
correctness/CI fixture. Its README correctly states that it is not a production-scale
leaderboard. It can be promoted only after the acceptance items below are independently
reviewed; passing its trusted-identifier reference is not evidence of resolver
generalization.

### 7.4 Acceptance criteria for adding a benchmark badge

A pull request may mark a pack `publishable-gold` only when:

- all source and gold files have stable IDs/checksums and documented licenses;
- gold is independent of predictions and leakage checks pass;
- the fixture includes positives, easy negatives, and adversarial hard negatives;
- at least one intentionally weak baseline and one reference baseline are reported;
- every metric includes support counts and exact definitions;
- correctness hard gates are explicit and failed rows remain visible;
- source-order/replay tests pass where applicable;
- Markdown, CSV, and JSON reports are generated from one logical run;
- the generator is deterministic apart from explicitly identified environment fields;
- the command, environment, Git state, input/config/gold hashes, and limitations are
  present in the report; and
- CI regenerates or semantically compares the artifacts.

`domain-release` additionally requires domain-expert sign-off, a representative test
split, temporal/update cases, competency queries, subgroup analysis, and a documented
decision about operational thresholds.

## 8. Benchmark Portfolio Across Domains

The suite should be layered. No public dataset below covers heterogeneous loading,
mapping, extraction, identity, provenance, validation, temporal behavior, and GraphRAG
utility by itself.

### Pack 0 — Cross-source identity invariant pack (implemented)

- **Data:** committed synthetic CRM, billing, and support CSV files.
- **Gold:** eight co-reference pairs over twelve records, including an `Alex Kim`
  namesake hard negative.
- **Purpose:** identity-key wiring, false-merge accounting, source-order invariance,
  and honest baseline publication.
- **Next version:** add trusted-ID conflict, shared email/phone, identifier reuse,
  temporal updates, deletions, field survivorship, cannot-link, and unmerge cases.
- **Release target:** remain the fast CI correctness pack, not a scale claim.

### Pack 1 — Mapping and materialization conformance

- **Data:** [W3C RDB2RDF test cases](https://www.w3.org/2001/sw/rdb2rdf/test-cases/)
  plus [KROWN](https://github.com/kg-construct/KROWN) generated scenarios.
- **Purpose:** R2RML/direct-mapping correctness, nulls, duplicate values, datatype and
  language terms, joins, named graphs, heterogeneous sources, and scale.
- **Gold:** canonical expected RDF datasets supplied by conformance cases or KROWN
  scenario definitions.
- **Primary metrics:** test-case pass rate, triple/quad exact match after RDFC-1.0 or an
  explicitly equivalent comparison, join accuracy, and unsupported-feature count.
- **Implementation prerequisite:** a conformance runner and actual RDF dataset
  canonicalization; the current mapper subset is not enough for a conformance claim.

### Pack 2 — Commerce/product identity

- **Data:** [WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html),
  a real multi-source entity-matching benchmark with corner-case and unseen-entity
  dimensions.
- **Graph model:** `Offer -> Product -> Brand/Category`, with source offer assertions
  retained independently.
- **Identity risks:** title similarity, model variants, bundles, refurbished products,
  unit/capacity differences, seller-specific SKUs, and missing GTIN/MPN.
- **Gold/tasks:** offer-to-product clusters, hard negatives, field normalization,
  relation mapping, and product competency queries.
- **Primary gates:** high pairwise precision/low false merges, blocking recall, no
  cross-variant merge, and source-disjoint test performance.
- **Data policy:** confirm the exact WDC subset terms before redistribution; prefer a
  checksum-backed downloader and small derived test fixtures where terms permit.

### Pack 3 — Financial/legal-entity graph

- **Data:** [GLEIF Golden Copy and Delta files](https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-golden-copy)
  for authoritative LEIs/relationships and
  [SEC EDGAR submissions/XBRL APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
  for CIKs, names, former names, filings, and company facts.
- **Graph model:** legal entities, registered names/addresses, direct/ultimate parents,
  filers, filings, reporting periods, and facts.
- **Identity risks:** LEI-versus-CIK crosswalk uncertainty, former names, corporate
  actions, branches/funds/subsidiaries, duplicate tickers, and temporal parentage.
- **Gold/tasks:** authoritative within-source ID integrity; a separately curated,
  adjudicated LEI–CIK crosswalk test set; parent relationship and temporal fact checks.
- **Primary gates:** zero conflicting-authority auto-merges, exact identifier scope,
  validity-interval accuracy, delta/full equivalence, and source attribution.
- **Important limitation:** GLEIF and SEC each provide strong internal identifiers but
  do not by themselves supply a complete cross-source LEI–CIK gold alignment. That
  crosswalk must be independently curated, not inferred and then reused as gold.

### Pack 4 — Scholarly authors and publications

- **Data:** [S2AND](https://github.com/allenai/S2AND/) for author-name
  disambiguation, with a licence-compatible publication/venue/affiliation snapshot.
- **Graph model:** authors, papers, venues, affiliations, citations, and authorship
  occurrences.
- **Identity risks:** identical names, transliteration, initials, name changes,
  affiliation changes, large collaborations, and missing ORCID.
- **Gold/tasks:** author clusters, authorship relations, field survivorship, and
  time/entity-disjoint evaluation.
- **Primary gates:** namesake false merges, B-cubed and pairwise scores, subgroup slices
  by name frequency/script, and author-order preservation.
- **Data policy:** record the S2AND and any companion source licences separately.

### Pack 5 — Biomedical literature graph

- **Data:** NCBI’s [BioRED](https://github.com/ncbi/BioRED), which provides annotated
  biomedical entities and document-level relations over PubMed abstracts.
- **Graph model:** genes/proteins, diseases, chemicals, variants, species, cell lines,
  relations, novelty status, publications, and supporting passages.
- **Identity risks:** abbreviations, synonyms, family-versus-member concepts, database
  namespace mismatch, negation, relation direction, and background-versus-novel facts.
- **Gold/tasks:** entity spans/types, relation types/direction, novelty, evidence span,
  and ontology/identifier linking when an independent gold mapping is available.
- **Primary gates:** entity/relation F1, direction/novelty accuracy, 100% evidence
  linkage, schema/SHACL conformance, and no unsupported clinical-use claim.
- **Safety:** this is literature benchmarking, not patient or diagnostic data.

### Pack 6 — Cybersecurity knowledge integration

- **Data:** versioned [MITRE ATT&CK STIX 2.1](https://github.com/mitre-attack/attack-stix-data)
  releases and the [NVD CVE API](https://nvd.nist.gov/developers/vulnerabilities).
- **Graph model:** techniques, tactics, groups, software, mitigations, data sources,
  CVEs, CPE products, references, and release/snapshot provenance.
- **Identity risks:** ATT&CK object versions, revoked/deprecated objects, aliases,
  software/product naming, CVE/CPE applicability, and changing NVD analysis status.
- **Gold/tasks:** STIX identifier/version preservation, ATT&CK relationship mapping,
  CVE/CPE integrity, update/deprecation replay, and cross-source links only where an
  authoritative reference or reviewed gold exists.
- **Primary gates:** no conflation of object versions, exact revocation/deprecation
  behavior, delta/full equivalence, referential integrity, and provenance to release.
- **Data policy:** pin release tags/API time windows and capture each source’s terms.

### Pack 7 — Table-to-KG and graph alignment challenge pack

- **Data:** [SemTab](https://www.cs.ox.ac.uk/isg/challenges/sem-tab/) tasks for cell
  entity, column type, and property annotation, and the
  [OAEI Knowledge Graph track](https://oaei.ontologymatching.org/2025/knowledgegraph/index.html)
  for instance and schema alignment.
- **Purpose:** evaluate semantic annotation and ontology/instance alignment separately
  from construction mechanics.
- **Primary metrics:** official challenge precision/recall/F1 and coverage, plus NIL,
  ambiguity, and partial-gold limitations.
- **Important limitation:** challenge editions and gold coverage change. Pin the exact
  edition/version and do not compare scores across incompatible editions.

### Optional text-to-KG research pack

[Text2KGBench](https://github.com/cenguix/Text2KGBench) can seed an ontology-driven
text-to-KG comparison, but it should be independently reviewed for ontology, annotation
quality, licence, literal normalization, and leakage before receiving repository
benchmark status. It is complementary to BioRED and does not test entity resolution
across source systems.

## 9. Phased Implementation Backlog

The backlog is ordered by correctness dependency, not novelty.

### Phase 0 — Current correctness foundation

Status: implemented in the current worktree; keep protected by regression tests.

- `P0.1` Canonical source hashing and idempotent source-record batch merge.
- `P0.2` Heterogeneous source routing in the public default workflow.
- `P0.3` Source-scoped default structured identity keys, including safe redaction
  behavior.
- `P0.4` Transitive resolution lineage and relation/fact endpoint reconciliation.
- `P0.5` Default referential-integrity validation.
- `P0.6` Order-independent canonical node/edge construction, conflict retention, and
  event occurrence preservation.
- `P0.7` Gold manifest loading, real gold metrics, hard publication gates, and
  Markdown/CSV/JSON report generation.
- `P0.8` Versioned cross-source identity pack and reproducible publication script.

Acceptance: targeted correctness tests, full unit suite, static checks, report
regeneration, and no claim broader than the synthetic fixture.

### Phase 1 — Governed identity contract

Priority: highest next phase.

- `P1.1` Add `SourceSystem`, `SourceSnapshot`, `IdentityKey`, `IdentityNamespace`,
  `ResolutionDecision`, `ClusterVersion`, and `CannotLink` domain models.
- `P1.2` Add a manifest schema for identifier issuer, scope, normalization, reuse, and
  validity policy.
- `P1.3` Promote the source-scoped name fallback into a versioned identity-policy
  configuration and retain global name matching only as an explicit baseline or
  reviewed domain policy.
- `P1.4` Add a durable identity registry port with an in-memory reference
  implementation and backend adapters.
- `P1.5` Implement constrained clustering, split/unmerge, cluster versioning, and
  relationship repair.
- `P1.6` Add declarative per-field survivorship and source-authority policies.
- `P1.7` Add blocking recall, B-cubed, exact-cluster, cluster-churn, and field-winner
  metrics.
- `P1.8` Extend synthetic v2 with conflicts, temporal reuse, updates, deletes, and
  unmerge cases.

Acceptance: zero hard-negative false merges, exact order/replay invariance, reversible
merge/split history, and independently evaluated field survivorship.

### Phase 2 — Standards-complete provenance and validation

- `P2.1` Model construction, mapping, extraction, validation, resolution, and review as
  PROV-O activities and software/human agents.
- `P2.2` Publish metric definitions and values using DQV-compatible resources linked
  to run provenance.
- `P2.3` Add field/assertion-level qualified derivation and source withdrawal support.
- `P2.4` Make domain SHACL shapes and schema version checks mandatory in release
  benchmark profiles.
- `P2.5` Implement RDFC-1.0 via a conformant library/adapter and use canonical N-Quads
  for RDF equality/hashing.
- `P2.6` Run W3C R2RML/direct-mapping tests; publish supported/unsupported feature
  matrices for R2RML, RML, CSVW, JSON-LD, and OBDA.

Acceptance: provenance competency queries pass, DQV measures are traceable to runs,
SHACL violations are gated, and conformance results are reproducible.

### Phase 3 — General benchmark harness and CI publication

- `P3.1` Version and validate an expanded benchmark manifest JSON Schema.
- `P3.2` Add downloader/checksum/licence-manifest tooling without committing restricted
  or oversized data.
- `P3.3` Standardize fixture adapters from external record IDs to candidate IDs.
- `P3.4` Add source/entity/time/schema split utilities and automated leakage checks;
  extend extraction gold with document/source IDs so repeated multi-document mentions
  and offsets are evaluated independently.
- `P3.5` Add confidence intervals, paired candidate comparison, subgroup slices, and
  minimum-support warnings.
- `P3.6` Separate deterministic smoke timing from repeated measured scale runs.
- `P3.7` Generate a docs index of benchmark cards and status badges from manifests.
- `P3.8` Add CI jobs for quick invariant packs, nightly domain candidates, and manually
  triggered scale releases.

Acceptance: one command per pack produces semantically stable reports; CI detects
stale artifacts; unsupported tasks are marked unavailable rather than scored as zero
or success.

### Phase 4 — Domain packs

Recommended order:

1. mapping/KROWN conformance, because it validates construction mechanics;
2. WDC Products, because it stresses real multi-source ER;
3. GLEIF/SEC, because authoritative identifiers and temporal changes test identity
   governance;
4. S2AND, because namesakes stress person resolution;
5. BioRED, because it adds evidence-grounded entity/relation extraction; and
6. ATT&CK/NVD, because versioned STIX and changing vulnerability data stress replay.

For each pack: complete legal/terms review, commit the manifest/generator/gold or
download instructions, run weak and reference baselines, obtain domain review, publish
reports, and assign status only after its acceptance checklist passes.

### Phase 5 — Utility and operational release

- `P5.1` Define domain competency queries before inspecting candidate graph answers.
- `P5.2` Add GraphRAG retrieval and answer evaluation with independent contexts,
  citations, and answer labels.
- `P5.3` Benchmark Neo4j/Kuzu/RDF backends with the same canonical graph and measured
  hardware profile.
- `P5.4` Add update lag, deletion propagation, recovery, dead-letter replay, and
  back-pressure tests.
- `P5.5` Publish Pareto views for correctness, utility, latency, memory, and cost; do
  not collapse them into one unexplained “best” score.

Acceptance: a domain-release candidate passes correctness and utility gates; a
scale-release candidate additionally has repeated measured performance with variance.

## 10. CI and Documentation Publication Plan

Keep three lanes so fast correctness tests are never weakened by slow external data:

| Lane | Trigger | Contents | Publication |
|---|---|---|---|
| Quick | every pull request | unit tests, synthetic identity, canonicalization/order/replay invariants, small gold fixtures | required check; committed concise reports |
| Domain | nightly/manual | downloaded checksum-pinned public-access packs, optional resolver/model adapters | candidate reports/artifacts with expiry and exact dataset versions |
| Scale | manual release | KROWN and graph-store scale runs on declared hardware | versioned release assets, not ordinary PR fixtures |

Current report commands are:

```bash
make benchmark-leaderboard
make gold-benchmark-story
make cross-source-identity-benchmark
make registry-reports
```

Before a release, also run the repository’s configuration, notebook, quality, unit,
and package checks documented in [`../README.md`](../README.md). Generated reports must link
back to their benchmark card and generation command. Benchmark cards must link to
Markdown, CSV, and JSON outputs so GitHub Pages or another docs renderer can publish
the same results without manual copying.

## 11. Research and Standards Basis

The plan uses the following primary or project-authoritative references:

| Reference | How it informs this repository |
|---|---|
| [W3C PROV-O](https://www.w3.org/TR/prov-o/) | Model source/generated entities, construction activities, agents, usage, generation, and qualified provenance. |
| [W3C SHACL](https://www.w3.org/TR/shacl/) | Express and execute graph constraints and publish validation reports. |
| [W3C Data Quality Vocabulary](https://www.w3.org/TR/vocab-dqv/) | Publish quality dimensions, metrics, measurements, annotations, and provenance. |
| [W3C RDF Dataset Canonicalization 1.0](https://www.w3.org/TR/rdf-canon/) | Produce deterministic RDF dataset serializations and stable graph hashes, including blank nodes. |
| [W3C R2RML](https://www.w3.org/TR/r2rml/) | Define customized relational-to-RDF mappings. |
| [W3C RDB2RDF test cases](https://www.w3.org/2001/sw/rdb2rdf/test-cases/) | Provide standards conformance fixtures and expected output datasets. |
| [KROWN](https://github.com/kg-construct/KROWN) | Benchmark RDF materialization over raw data, duplicates/empty values, mappings, joins, heterogeneous sources, and scale. |
| [Splink](https://github.com/moj-analytical-services/splink) | Probabilistic record-linkage baseline for data without unique identifiers. |
| [Dedupe](https://docs.dedupe.io/en/latest/) | Active-learning structured deduplication and record-linkage baseline. |
| [WDC Products](https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html) | Real-world, multi-source product matching with corner cases and unseen-entity dimensions. |
| [SemTab](https://www.cs.ox.ac.uk/isg/challenges/sem-tab/) | Reproducible table-to-KG semantic annotation tasks. |
| [OAEI Knowledge Graph track](https://oaei.ontologymatching.org/2025/knowledgegraph/index.html) | Instance and schema alignment with gold correspondences and official metrics. |
| [S2AND](https://github.com/allenai/S2AND/) | Author-name disambiguation datasets and evaluation. |
| [BioRED](https://github.com/ncbi/BioRED) | Human-annotated biomedical entity and document-level relation extraction. |
| [GLEIF Golden Copy](https://www.gleif.org/en/lei-data/gleif-golden-copy/download-the-golden-copy) | Authoritative LEI reference, relationship, and delta data. |
| [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Public submissions and XBRL company facts keyed by CIK. |
| [MITRE ATT&CK STIX data](https://github.com/mitre-attack/attack-stix-data) | Versioned cybersecurity KG data in STIX 2.1. |
| [NVD vulnerability API](https://nvd.nist.gov/developers/vulnerabilities) | Versioned CVE/CPE vulnerability records and source/status metadata. |

## 12. Immediate Definition of Done

The next implementation milestone is complete only when all of the following are true:

- the identity policy is explicit and display names are disabled as implicit global
  keys in conservative production profiles;
- resolution decisions and cannot-link constraints are first-class, versioned data;
- source order, component order, replay, and full/incremental equivalence tests pass;
- field survivorship and conflicts are evaluated against gold, not only made
  deterministic;
- PROV-O activities/agents and DQV measurements are emitted and queryable;
- SHACL and referential-integrity violations are publication gates;
- the synthetic identity pack v2 covers updates, deletes, trusted-ID conflicts, shared
  contacts, and unmerge;
- at least one standards pack and one real multi-source domain pack reach
  `publishable-gold`;
- benchmark manifests and reports are indexed in repository docs; and
- every “best” claim names the dataset version, task, metric, gate set, uncertainty,
  and limitations.

Until those conditions are met, the correct public description is: this repository
provides an evidence-grounded, duplicate-resistant multi-source KG construction and
benchmarking framework with a published synthetic identity `gold-candidate` fixture. It
does not yet prove universal deduplication or domain-production readiness.
