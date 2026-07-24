# Testing every technique

The repo ships **437 tests** plus report generators, gold fixtures, and an
offline certification gate. This guide shows how to test each technique family
and which test pins it.

Contents:

1. [Running the suite](#1-running-the-suite)
2. [Test ↔ technique map](#2-test--technique-map)
3. [Gold fixtures](#3-gold-fixtures)
4. [Report-driven gates](#4-report-driven-gates)
5. [The doctor and certification hard pass](#5-the-doctor-and-certification-hard-pass)
6. [Quality (lint / type / dead-code / config / notebooks)](#6-quality-lint--type--dead-code--config--notebooks)
7. [Testing a technique end to end](#7-testing-a-technique-end-to-end)

---

## 1. Running the suite

```bash
make test          # PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q  (compiles first)
make verify        # test + smoke construction/benchmark flows
```

Run a single file or test:

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_semantic_table_mapper.py -q
PYTHONPATH=src .venv/bin/python -m pytest tests/test_structured_conversion.py -q -k employees
```

The suite also runs under stdlib unittest without pytest installed:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Count and list what will run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src python3 -m pytest --collect-only -q | tail -3
```

---

## 2. Test ↔ technique map

### Pillar 1 — structured → graph

| Technique / subsystem | Test file | What it pins |
| --- | --- | --- |
| Semantic column profiler (roles + materialization) | `tests/test_semantic_column_profiler.py` | role assignment, promotion thresholds, key/composite-key detection |
| Resolved plan + six projections | `tests/test_resolved_modeling_plan.py`, `tests/test_modeling_contract.py` | plan resolution + `assert_schema_agreement` across all 6 artifacts |
| Semantic table mapper | `tests/test_semantic_table_mapper.py` | shared concept nodes, FK edges, no literal duplication |
| Cross-table / catalog conversion (exit gate) | `tests/test_structured_conversion.py` | employees 8/3/8/7, 100% key/FK coverage, zero re-ingestion delta, identity scope |
| Adversarial structured invariants | `tests/test_structured_adversarial_invariants.py` | no fabricated FK targets, composite-key collisions, supernode guard |
| SHACL / ontology gate (structured) | `tests/test_structured_ontology_gate.py` | RDF conforms; gate rejects violations |
| Direct mapper + graph builders | `tests/test_canonical_graph_correctness.py`, `tests/test_data_to_kg_construction.py` | baseline mapping + property/RDF/evidence graph correctness |

### Pillar 2 — unstructured → graph

| Technique / subsystem | Test file | What it pins |
| --- | --- | --- |
| Evidence-first pipeline (11 stages) | `tests/test_unstructured_pipeline.py` | segmentation offsets, promotion gate, claim suite hard gate |
| Mandatory composed runtime | `tests/test_mandatory_unstructured_runtime.py` | every locked provider participates; NIL stays observed |
| Model-backed providers | `tests/test_mandatory_model_providers.py` | GLiNER / Transformers NER / spaCy / Qwen3 contracts |
| Relation extraction | `tests/test_relation_extraction.py` | pattern + GLiNER RelEx + Qwen claim signatures |
| Coverage / occurrence accuracy | `tests/test_unstructured_coverage.py`, `tests/test_occurrence_accuracy_metrics.py` | exact-span vs entity F1 separation, occurrence counting |
| Entity resolution / linking | `tests/test_resolution_integrity.py`, `tests/test_entity_linking.py` | false-merge/split, Wikidata/DBpedia/reconciliation |

### Pillar 3 — evaluation & gates

| Technique / subsystem | Test file | What it pins |
| --- | --- | --- |
| Formation quality gates (E1) | `tests/test_knowledge_graph_quality.py` | evidence/provenance/dangling/duplicate/SHACL gates |
| Schema quality (4 dimensions) | `tests/test_schema_quality.py` | conciseness/completeness/consistency/structure + `shacl_executed` gate |
| Task metrics (E2) | `tests/test_occurrence_accuracy_metrics.py`, `tests/test_resolution_integrity.py` | P/R/F1, ER pairs |
| Intended use (E3) | `tests/test_intended_use_metrics.py`, `tests/test_graph_bound_evaluation.py` | 7 consumers, graph-bound evidence, popularity strata |
| Weighted ranking / leaderboard | `tests/test_benchmark_leaderboard.py`, `tests/test_publishable_benchmarks.py`, `tests/test_technique_evaluation.py` | scoring engine, reproducible rankings |
| Modelling correctness (CI gate) | `tests/test_modelling_correctness.py` | gold-fixture structured + unstructured correctness |
| Baseline defect characterization | `tests/test_modelling_baseline_defects.py` | the confirmed defects (some now closed, some `expectedFailure`) |

### Platform, registry, production

| Subsystem | Test file |
| --- | --- |
| Registry ↔ capability manifest bijection | `tests/test_capability_manifest.py`, `tests/test_registry_support_levels.py` |
| Public API (`construct_kg`, `evaluate_techniques`) | `tests/test_public_api.py`, `tests/test_quickstart.py` |
| Mandatory product config | `tests/test_mandatory_product_config.py` |
| Certification substrate (offline invariants) | `tests/test_certification_substrate.py`, `tests/test_semantic_capability_certifiers.py` |
| Model provisioning / locks | `tests/test_model_provisioning.py` |
| Doctor CLI | `tests/test_doctor_cli.py` |
| Loaders (enterprise) | `tests/test_enterprise_loaders.py` |
| Neo4j vector store | `tests/test_neo4j_vector_store.py` |
| Production hardening (retry / dead-letter / PII / watermark) | `tests/test_production_hardening.py`, `tests/test_production_ingestion.py` |
| Graph-ML enrichment (research) | `tests/test_graph_ml_enrichment.py` |

---

## 3. Gold fixtures

Correctness is pinned by small, versioned fixtures under
`tests/fixtures/gold_standards/`. Because they are tiny and deterministic, they
are the ground truth the CI gate enforces (`make modelling-correctness`).

| Fixture | Feeds |
| --- | --- |
| `column_roles.json` | semantic profiler role assignments |
| `modelling_correctness_structured.json` | structured gold graph (employees/departments) |
| `modelling_correctness_unstructured.json` | unstructured gold claims/intents |
| `unstructured_claims.json` | claim evaluation suite (entity/relation/attributes) |
| `unstructured_coverage.json` | extraction coverage |
| `ner_spans.json` | exact-span NER scoring |
| `gold_triples.json`, `relation_sample.json` | relation/triple F1 |
| `entity_resolution_clusters.json` | ER pairwise/cluster + false-merge/split |
| `retrieval_cases.json` | GraphRAG / retrieval judgments |

Other fixture roots: `tests/fixtures/golden_benchmarks/` (leaderboard),
`tests/fixtures/enterprise_loaders/` (loaders).

---

## 4. Report-driven gates

Several gates are exercised by regenerating a committed report and asserting it
matches. Regenerate them with:

```bash
make modelling-correctness   # reports/modelling_correctness.md   (CI-enforced)
make schema-quality          # reports/schema_quality.md
make benchmark-leaderboard   # reports/benchmark_leaderboard.md
make registry-reports        # reports/current_registry_*.{csv,json,md}
make cross-source-identity-benchmark
make gold-benchmark-story
```

The corresponding tests (`test_modelling_correctness.py`,
`test_schema_quality.py`, `test_benchmark_leaderboard.py`) fail if the code
drifts from the pinned report.

---

## 5. The doctor and certification hard pass

`doctor` verifies the *selected runtime* — it has only `ok`/`fail`, never a
downgrade:

```bash
uv run kg-construction doctor --strict
uv run kg-construction doctor --strict --json
```

The full **zero-skip** gate provisions locks, then re-runs everything with
package/model/container registries disconnected:

```bash
make release-locks   # one-time, online: uv.lock + models.lock + services.lock
make setup           # hydrate models + services, strict doctor
bash run_all_approaches.sh
```

`run_all_approaches.sh` sets `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, dispatches every capability certifier with a fresh
nonce, and requires
`declared = configured = instantiated = executed = evaluated = passed`. Live
remote identity/reconciliation certifiers still use their documented endpoints.

> The three lock files are network-gated: generate them once on a networked
> machine. Without them, `make lock-check` / `certify-preflight` fail fast — by
> design.

---

## 6. Quality (lint / type / dead-code / config / notebooks)

```bash
make quality
```

Runs, in order: `ruff check`, `mypy src/kg_construction`, `vulture`,
`check_config_examples.py` (every `configs/*.example.yaml` parses and matches
the schema), `check_notebook_hygiene.py`, and `check_notebook_coverage.py`
(each of the 11 notebooks maps to a covered capability).

---

## 7. Testing a technique end to end

Beyond unit tests, exercise a technique through the CLI on the sample data and
inspect the emitted graphs and gate reports:

```bash
# Structured (semantic mapper default) — writes property/RDF/evidence/Neo4j + gate JSON
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.construct \
  --input examples/data_to_kg/sample.csv --source-type csv \
  --config configs/data_to_kg_complete.example.yaml \
  --evaluation-suite configs/graph_evaluation_suite.sample_csv.yaml \
  --output /tmp/kg/sample.property.json

# Unstructured (evidence-first pipeline)
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.construct \
  --input examples/data_to_kg/sample.txt --source-type text \
  --config configs/data_to_kg_complete.example.yaml \
  --evaluation-suite configs/graph_evaluation_suite.sample_text.yaml \
  --output /tmp/kg/text.property.json

# Compare many techniques on one input (ranked report)
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.benchmark \
  --input examples/data_to_kg/sample.txt --source-type text \
  --config configs/evaluation.example.yaml --techniques applicable \
  --output /tmp/kg/report.md --json-output /tmp/kg/report.json
```

A construction succeeds only if `result.validation_report.valid` and
`result.publication_gate.passed` — a failing gate is a failing run, which is the
whole point. See the full command catalogue in the
[README run guide](../README.md#running-every-technique).
