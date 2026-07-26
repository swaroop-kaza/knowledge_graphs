# Testing construction, evaluation, and certification

[Documentation home](README.md) · [Getting started](getting_started.md) · [Evaluation](evaluation_metrics.md) · [Stage-by-stage implementation](stage_by_stage_implementation.md)

- **Product checks** cover structured construction, unstructured construction, evaluation, and the zero-skip certification substrate.
- **Research checks** cover comparisons and leaderboards separately; they are not publication gates.

## On this page

1. [Running the suite](#1-running-the-suite)
2. [Test-to-technique map](#2-test-to-technique-map)
3. [Gold fixtures](#3-gold-fixtures)
4. [Report-driven gates](#4-report-driven-gates)
5. [The doctor and certification hard pass](#5-the-doctor-and-certification-hard-pass)
6. [Quality (lint / type / dead-code / config / notebooks)](#6-quality-lint--type--dead-code--config--notebooks)
7. [Testing a technique end to end](#7-testing-a-technique-end-to-end)
8. [Related guides](#related-guides)

---

## 1. Running the suite

| Need | Command | Scope |
| --- | --- | --- |
| Unit and contract tests | `make test` | Compiles first, then runs pytest |
| Product construction smoke | `make product-smoke` | Structured and unstructured flows |
| Product verification | `make verify` | Tests plus product smoke |
| Research comparison smoke | `make research-smoke` | Non-authoritative comparisons |
| Complete repository check | `make all-checks` | Quality, product verification, and research smoke |

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

## 2. Test-to-technique map

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

These tests target implementations under `kg_construction.evaluation`. Legacy `kg_construction.infrastructure.evaluation.*` modules are named compatibility shims, not a second implementation tree or a wildcard façade.

| Technique / subsystem | Test file | What it pins |
| --- | --- | --- |
| Formation quality gates (E1) | `tests/test_knowledge_graph_quality.py` | evidence/provenance/dangling/duplicate/SHACL gates |
| Schema quality (4 dimensions) | `tests/test_schema_quality.py` | conciseness/completeness/consistency/structure + `shacl_executed` gate |
| Task metrics (E2) | `tests/test_occurrence_accuracy_metrics.py`, `tests/test_resolution_integrity.py` | P/R/F1, ER pairs |
| Intended use (E3) | `tests/test_intended_use_metrics.py`, `tests/test_graph_bound_evaluation.py` | 7 consumers, graph-bound evidence, popularity strata |
| Modelling correctness (CI gate) | `tests/test_modelling_correctness.py` | gold-fixture structured + unstructured correctness |
| Baseline defect characterization | `tests/test_modelling_baseline_defects.py` | the confirmed defects (some now closed, some `expectedFailure`) |

### Research boundary

| Research surface | Test file | What it proves |
| --- | --- | --- |
| Simple candidate benchmark ownership | `tests/test_evaluation_package_contract.py` | canonical research ownership and legacy import identity |
| Weighted technique ranking | `tests/test_benchmark_leaderboard.py`, `tests/test_technique_evaluation.py` | deterministic comparison behavior only |
| Research-report hygiene | `tests/test_research_benchmarks.py` | research artifacts are reproducible and honestly labelled, never treated as certificates |
| Graph ML enrichment | `tests/test_graph_ml_enrichment.py` | experimental enrichment behavior |

### Platform, registry, production

| Subsystem | Test file |
| --- | --- |
| Registry ↔ capability manifest bijection | `tests/test_capability_manifest.py`, `tests/test_registry_support_levels.py` |
| Product public API (`convert`, compatibility `construct_kg`) | `tests/test_public_api.py`, `tests/test_quickstart.py` |
| Research API (`kg_construction.research.evaluate_techniques`) | `tests/test_public_api.py`, `tests/test_technique_evaluation.py` |
| Mandatory product config | `tests/test_mandatory_product_config.py` |
| Certification substrate (offline invariants) | `tests/test_certification_substrate.py`, `tests/test_semantic_capability_certifiers.py` |
| Model provisioning / locks | `tests/test_model_provisioning.py` |
| Doctor CLI | `tests/test_doctor_cli.py` |
| Loaders (enterprise) | `tests/test_enterprise_loaders.py` |
| Neo4j vector store | `tests/test_neo4j_vector_store.py` |
| Production hardening (retry / dead-letter / PII / watermark) | `tests/test_production_hardening.py`, `tests/test_production_ingestion.py` |

---

## 3. Gold fixtures

- **Small, versioned fixtures** live under `tests/fixtures/gold_standards/`.
- **Tiny and deterministic** — they are the ground truth the CI gate enforces (`make modelling-correctness`).

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

Other fixture roots: `tests/fixtures/golden_benchmarks/` (research comparison) and `tests/fixtures/enterprise_loaders/` (loader contracts). The `benchmarks/cross_source_identity/` fixture is a research comparison of resolvers; core false-merge/false-split metrics remain part of entity-resolution evaluation.

---

## 4. Report-driven gates

Core gates are exercised by regenerating committed reports and asserting they match:

```bash
make modelling-correctness   # reports/modelling_correctness.md   (CI-enforced)
make schema-quality          # reports/schema_quality.md
make registry-reports        # reports/current_registry_*.{csv,json,md}
```

Research reports are separate:

```bash
make research-smoke
make cross-source-identity-benchmark
make gold-benchmark-story
```

- **Product drift is caught** — `test_modelling_correctness.py` and `test_schema_quality.py` pin the core outputs.
- **Research drift is labelled separately** — `test_benchmark_leaderboard.py` pins a comparison report, not publishability.

---

## 5. The doctor and certification hard pass

`doctor` verifies the *selected runtime* — it has only `ok`/`fail`, never a downgrade:

```bash
uv run kg-construction doctor --strict
uv run kg-construction doctor --strict --json
```

The full **zero-skip** gate provisions locks, then re-runs everything with package/model/container registries disconnected:

```bash
make release-locks   # one-time, online: uv.lock + models.lock + services.lock
make setup           # hydrate models + services, strict doctor
bash run_all_approaches.sh
```

`run_all_approaches.sh`:

- **Forces offline** — sets `UV_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.
- **Dispatches every certifier** with a fresh nonce.
- **Requires the full chain** — `declared = configured = instantiated = executed = evaluated = passed`.
- **Live endpoints stay live** — remote identity/reconciliation certifiers still use their documented endpoints.

> The three lock files are network-gated: generate them once on a networked machine. Without them, `make lock-check` / `certify-preflight` fail fast — by design.

---

## 6. Quality (lint / type / dead-code / config / notebooks)

```bash
make quality
```

Runs, in order:

- **`ruff check`** — lint.
- **`mypy src/kg_construction`** — types.
- **`vulture`** — dead code.
- **`check_config_examples.py`** — every `configs/*.example.yaml` parses and matches its schema.
- **`check_notebook_hygiene.py`** + **`check_notebook_coverage.py`** — product demonstrations and research illustrations remain output-free and correctly classified.

---

## 7. Testing a technique end to end

Beyond unit tests, exercise the product through the CLI and inspect the emitted draft graphs and observations:

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

# Research only: compare techniques on one input
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.benchmark \
  --input examples/data_to_kg/sample.txt --source-type text \
  --config configs/evaluation.example.yaml --techniques applicable \
  --output /tmp/kg/report.md --json-output /tmp/kg/report.json
```

- **Draft output is inspectable** — the CLI writes graphs plus metric reports so a failed family can be diagnosed.
- **Publishability is gated** — only the zero-skip certified run may produce a publication claim, and any required failed/unexecuted observation fails it.
- **Feature boundaries** — product tests target `kg_construction.structured`, `kg_construction.unstructured`, and `kg_construction.evaluation`; comparison tests target `kg_construction.research`.
- **Full command catalogue** — see the [README](../README.md).

## Related guides

- [Getting started](getting_started.md) covers provisioning, strict doctor, and the zero-skip hard pass.
- [Configuration](configuration.md) identifies which settings are draft, certified, or research-only.
- [Evaluation metrics](evaluation_metrics.md) defines the calculations and thresholds asserted by these tests.
- [Certification](../certification/README.md) defines the final evidence and release decision.
