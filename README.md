# Data-to-Knowledge Graph Construction Framework

`kg_construction` is an evidence-grounded, multi-source knowledge graph construction framework with duplicate-resistant identity resolution, provenance, validation, and reproducible benchmarks.

## Run it in under a minute

You need Python 3.11 or 3.12 and `git`. From the repo root:

```bash
make setup       # creates .venv, installs the package (offline), runs a health check
make quickstart  # builds sample CSV/JSON/text graphs and prints a summary
```

`make setup` creates the virtual environment for you — no manual `venv` or activation, and no PEP 668 "externally-managed-environment" errors on Homebrew/Debian Python. Run `make help` to list all tasks.

To use the `kg-construction` command directly, activate the env first: `source .venv/bin/activate` (Windows: `.\.venv\Scripts\Activate.ps1`). After that, `kg-construction …` and `import kg_construction` work directly — you do **not** need `PYTHONPATH=src`. Prefer to manage your own environment? `python3 -m venv .venv && source .venv/bin/activate && pip install -e .` works too.

Nothing installed yet? The from-zero walkthrough (macOS/Linux/Windows, plus Ollama/Neo4j/Kuzu/GLiNER setup) is **[docs/getting_started.md](docs/getting_started.md)**.

It is built for teams that need to combine structured, semi-structured, and unstructured data without silently collapsing namesakes, discarding conflicting source claims, or selecting a graph pipeline from unverified proxy metrics.

The core flow:

```text
Sources -> Evidence -> Candidates -> Identity Decisions -> Canonical Graph -> Validation -> Benchmarks
```

The default structured-data policy is deliberately conservative: source records receive source-scoped identity keys, and cross-source co-reference must be established by an explicit resolver. "Duplicate-resistant" does not mean that uncertain real-world identity can be solved without labels or evidence; it means that automatic merges, canonical claims, provenance, and benchmark decisions are designed to be auditable and testable.

The framework remains technique-centered. A dataset can be profiled, multiple applicable techniques can be run side by side, and recommendations can be based on gold correctness, validation, duplicate-control, provenance, runtime, and cost measurements when those benchmark inputs are available.

See the small committed smoke leaderboard for a reproducible pipeline comparison:

- [reports/benchmark_leaderboard.md](reports/benchmark_leaderboard.md)
- [reports/benchmark_leaderboard.csv](reports/benchmark_leaderboard.csv)

See the gold-standard benchmark story for component-level correctness checks:

- [reports/gold_standard_benchmark_story.md](reports/gold_standard_benchmark_story.md)
- [reports/gold_standard_benchmark_story.csv](reports/gold_standard_benchmark_story.csv)
- [reports/gold_standard_benchmark_story.json](reports/gold_standard_benchmark_story.json)

See the versioned cross-source identity fixture for the duplicate-control baseline,
including an intentional namesake and explicit false-merge gates:

- [reports/cross_source_identity_benchmark.md](reports/cross_source_identity_benchmark.md)
- [reports/cross_source_identity_benchmark.csv](reports/cross_source_identity_benchmark.csv)
- [reports/cross_source_identity_benchmark.json](reports/cross_source_identity_benchmark.json)

The research-backed architecture, domain portfolio, and phased implementation plan is
[docs/multisource_kg_benchmark_plan.md](docs/multisource_kg_benchmark_plan.md).

## Why This Exists

Enterprise data already contains relationships, but they are usually hidden across tables, JSON payloads, logs, documents, PDFs, emails, reports, and APIs. The same real entity may appear under different identifiers, while two unrelated entities may share a name. This repo preserves source evidence, creates traceable candidates and identity decisions, validates property/RDF/evidence graph outputs, and publishes the measurements needed to compare construction strategies honestly.

## 60-Second Data Scientist Quickstart

> **New here, or have nothing installed yet?** Follow the complete from-zero
> walkthrough in [docs/getting_started.md](docs/getting_started.md). It covers
> installing Python and the framework, and step-by-step setup for every optional
> service — Ollama (local LLM), Neo4j, Kuzu, and GLiNER — plus how to run each path.

This path runs locally with the core install. It does not require Neo4j, Kuzu, Ollama, GLiNER, PyTorch, or graph ML extras.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python3 examples/quickstart_60_second.py
```

The same flow as Python:

```python
from kg_construction import construct_kg, evaluate_techniques, profile_dataset
from kg_construction.infrastructure.graph_stores import Neo4jExporter, NetworkXExporter, RDFExporter

# 1. CSV -> Knowledge Graph
csv_result = construct_kg("examples/data_to_kg/sample.csv")
print(len(csv_result.property_graph.nodes), len(csv_result.property_graph.edges))

# 2. Text -> Knowledge Graph
text_result = construct_kg("examples/data_to_kg/sample.txt", source_type="text")
print([entity.name for entity in text_result.batch.entities])

# 3. JSON -> Knowledge Graph
json_result = construct_kg("examples/data_to_kg/sample.json", source_type="json")
print(len(json_result.property_graph.nodes))

# 4. Benchmark construction techniques
report = evaluate_techniques(
    "examples/data_to_kg/sample.txt",
    source_type="text",
    techniques="regex_entity_extractor,hybrid_rules_gazetteer_regex",
)
print(report.best_technique)

# 5. Export without live external services
rdf_payload = RDFExporter().export_rdf_graph(csv_result.rdf_graph)
neo4j_cypher = Neo4jExporter().export_property_graph(csv_result.property_graph)
networkx_payload = NetworkXExporter().export_property_graph(csv_result.property_graph)

print(rdf_payload["turtle"][:200])
print(neo4j_cypher[:2])
print(networkx_payload["nodes"][:2])
```

What this proves:

- CSV rows can become graph nodes and RDF facts.
- Heterogeneous directories route each source to its matching mapper or extractor.
- Structured rows use conservative source-scoped identity keys by default.
- Text can become evidence-backed entity candidates.
- JSON can become graph candidates through JSONPath mapping.
- Technique benchmarking can rank construction paths.
- RDF export produces Turtle/N-Quads payloads.
- Neo4j export produces Cypher statements without opening a database connection.
- NetworkX export produces a node-link dictionary without requiring the `networkx` package.

## Developer Quickstart

Install developer, RDF, graph, and notebook extras when you want the full test and notebook environment:

```bash
python -m pip install -e ".[dev,rdf,graph,notebook]"
make verify
```

Run a CLI construction:

```bash
kg-construction \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --output data/processed/data_to_kg_text_property_graph.json \
  --rdf-output data/processed/data_to_kg_text_rdf.json \
  --evidence-output data/processed/data_to_kg_text_evidence_graph.json
```

Run a CLI technique benchmark:

```bash
kg-construction-benchmark \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --config configs/evaluation.example.yaml \
  --techniques applicable \
  --output data/processed/data_to_kg_technique_evaluation_report.md \
  --json-output data/processed/data_to_kg_technique_evaluation_report.json \
  --csv-output data/processed/data_to_kg_technique_evaluation_report.csv
```

Use the Python API in notebooks or scripts:

```python
from kg_construction import construct_kg, evaluate_techniques, load_sources, profile_dataset

sources = load_sources("examples/data_to_kg/sample.csv")
profile = profile_dataset(sources)

result = construct_kg("examples/data_to_kg/sample.csv")
property_graph = result.property_graph
rdf_graph = result.rdf_graph
evidence_graph = result.evidence_graph

report = evaluate_techniques(
    "examples/data_to_kg/sample.txt",
    source_type="text",
    techniques="applicable",
)
print(report.best_technique)
```

## Doctor CLI

Use doctor when a local install, optional dependency, config file, or runtime service is not behaving as expected:

```bash
kg-construction doctor
kg-construction doctor --json
kg-construction doctor --skip-services
```

The command checks:

- Python version and package distribution metadata.
- Core imports and inferred optional extras.
- Optional imports for notebook, GLiNER, LLM, graph, RDF, linking, dedup, and ML paths.
- Ollama availability through `OLLAMA_BASE_URL` or `http://localhost:11434`.
- Neo4j availability through `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`.
- Kuzu embedded database availability.
- RDFLib/pySHACL smoke validation.
- Notebook dependencies.
- YAML config validity under `configs/`.

`kg-construction doctor` is read-only. Missing optional services are reported as skipped unless they are explicitly configured and fail.

## Package

```text
src/kg_construction/
  domain/                 Pydantic source, evidence, entity, relation, fact, schema, graph models
  ports/                  Protocol interfaces for plugin-style extension
  application/            config, workflow, registry, profiler, selector, scoring, recommendations
  infrastructure/
    loaders/              CSV, JSON, text, Excel, SQL, API, XML, PDF, email, chat loaders
    mapping/              direct table, JSONPath, JSON-LD, OBDA, R2RML, RML, CSVW
    extractors/           regex, gazetteer, GLiNER, OpenIE, dependency, Ollama, REBEL adapters
    resolution/           exact resolver plus advanced resolver adapters
    ontology/             schema registry and validation
    graph_builders/       property graph, RDF graph, evidence graph builders
    graph_stores/         Neo4j, Memgraph, Kuzu, NetworkX, RDF exports and ingestion
    evaluation/           accuracy, validation, coverage, ER, retrieval, runtime, confidence metrics
    reports/              JSON and Markdown reports
    production/           logging, manifests, watermarks, retry, dead letters, review, PII policy
  interfaces/cli/         construction, benchmark, and dead-letter replay CLIs
```

## Support Levels

The project is currently `0.1.x`, so the package as a whole is still early-stage. Individual techniques are labeled more precisely in the registry, README, docs, and reports.

| Public support level | Registry signal | What it means | Examples |
| --- | --- | --- | --- |
| Stable | `implemented` | Safe to use now through the documented API and covered by normal tests. | CSV/JSON/text loaders, direct table mapping, regex/gazetteer extraction, RDF/property/evidence graph builders, Cypher/RDF/NetworkX exports |
| Optional adapter | `adapter`, or `implemented` with optional service/dependency tags | Works when the dependency, model, service, or project-specific wiring is configured. Missing setup should fail with an explicit message. | GLiNER model extraction, Ollama extraction, REBEL/transformer adapters, Neo4j/Kuzu ingestion, Splink/Dedupe, Wikidata/DBpedia/OpenRefine |
| Experimental | `implemented` research-heavy capability | Usable for experiments and benchmarks, but API, scoring, model behavior, or performance assumptions may change. | Graph ML, embeddings, link prediction, retrieval benchmarks, LLM-heavy hybrid workflows |
| Planned | `stub` | Registered roadmap idea or extension point, not ready for normal use. | future spaCy/coreference/POS adapters, full XMLPath/API response mappers, additional validators |

Programmatic access:

```python
from kg_construction import default_registry

registry = default_registry()
technique = registry.get("gliner_entity_extractor")
print(technique.status)                 # internal implementation state
print(technique.public_support_level)   # public maturity level
print(technique.public_support_label)   # display label
```

See `docs/support_levels.md` and `reports/current_registry_matrix.csv` for the complete registry view.

## Implemented Capabilities

- Evidence-first Pydantic models with deterministic IDs.
- Stable top-level Python API: `load_sources`, `profile_dataset`, `construct_kg`, `evaluate_techniques`, `DataToKGConfig`, `ConstructionWorkflow`, `ConstructionResult`, and `default_registry`.
- CSV, JSON, text, Excel, SQL, API response, XML, HTML, web page, log, Markdown, report, scientific paper, PDF, email, and chat loading.
- Direct table-to-graph mapping with schema profiling and key inference.
- Dotted-path JSON flattening and real JSONPath mapping with array-derived edges.
- HTML metadata mapping, event payload graph mapping, trace/span correlation, log template mining, and log event graph mapping.
- Executable standards mapping for JSON-LD, OBDA rewriting, R2RML relational rows, RML CSV/JSON/XML sources, and CSVW tabular metadata.
- R2RML/RML referencing object maps with join conditions and cross-source joins.
- CSVW dialect parsing and validation for required columns, primary keys, and metadata coverage.
- Structured data to schema proposal with SHACL and OWL-lite vocabulary output.
- Regex, gazetteer, GLiNER, OpenIE, dependency-pattern, and Ollama graph extraction.
- Ollama label proposal followed by GLiNER span extraction.
- GLiNER entity spans plus Ollama relation typing with evidence-gated relationship candidates.
- REBEL/transformer relation extraction adapter with injectable Hugging Face-style pipeline.
- Ollama provider with structured JSON retry/repair.
- Ollama graph extractor that returns candidates only and never writes directly to a graph.
- Source-scoped exact-name fallback plus explicit global-name, normalized string, RapidFuzz, and alias-table auditable entity-resolution paths.
- Transitive canonical-ID handling and referential-integrity checks across sequential resolvers.
- Wikidata, DBpedia, OpenRefine, embedding, and cross-encoder entity linking/resolution.
- Splink and Dedupe adapter surfaces for probabilistic and active-learning entity resolution.
- Evidence-required, schema-registry, confidence, and pySHACL validation.
- Property graph, RDF triple/quad graph, and evidence graph builders.
- Neo4j/Memgraph Cypher, Kuzu manifest, NetworkX node-link, Turtle, N-Quads, and RDFLib-compatible export.
- Production graph ingestion: Neo4j managed write transactions, Kuzu write adapter, batch retry policy, and dead-letter replay.
- Production ingestion extensions: live Neo4j/Kuzu integration profiles, graph-store batch benchmark fixtures, RDF-to-Neo4j neosemantics import, and Kuzu bulk import.
- Graph ML enrichment: train/validation splits, degree features, PageRank, communities, connected components, node2vec, KG embeddings, node embeddings, link prediction, retrieval benchmarks, large-graph backends, and optional GraphSAGE adapter.
- Gold benchmark fixtures and evaluator for NER spans, RDF triples, human-reviewed relation extraction, entity resolution clusters, and GraphRAG/retrieval cases.
- Versioned benchmark manifests and committed Markdown/CSV/JSON reports for reproducible publication.
- Technique registry covering internal statuses and public support levels.
- Dataset profiling and applicability scoring.
- Weighted mathematical technique ranking.
- Deterministic benchmark mode with explicit tie-breaking policy.
- Best single, best hybrid, production-safe, high-recall, low-cost, and GraphRAG-ready recommendations.
- Summary-safe JSON, Markdown, and CSV benchmark reports with explicit publication gates.
- Gold-standard benchmark story for NER spans, relation extraction, triples, entity resolution, and GraphRAG/retrieval.
- Production hardening: structured logging, run manifests, config validation, watermarks, quarantine, retry, dead-letter replay, human review queue, PII redaction, CI optional-extra matrix.

## Construction CLI

```bash
kg-construction \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --output data/processed/data_to_kg_text_property_graph.json \
  --rdf-output data/processed/data_to_kg_text_rdf.json \
  --evidence-output data/processed/data_to_kg_text_evidence_graph.json \
  --neo4j-output data/processed/data_to_kg_text_neo4j.cypher
```

## Technique Evaluation CLI

```bash
kg-construction-benchmark \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --config configs/evaluation.example.yaml \
  --techniques applicable \
  --output data/processed/data_to_kg_technique_evaluation_report.md \
  --json-output data/processed/data_to_kg_technique_evaluation_report.json \
  --csv-output data/processed/data_to_kg_technique_evaluation_report.csv \
  --candidate-output-dir data/processed/data_to_kg_candidates
```

The benchmark command:

- profiles the input data
- computes applicability for registered techniques
- runs executable candidate pipelines side by side
- validates every candidate graph
- computes accuracy, validation, coverage, entity-resolution, retrieval, runtime, cost, and confidence metrics
- ranks techniques with configurable weights and penalties
- writes candidate graph outputs separately
- produces summary-safe JSON, Markdown, and CSV reports explaining why a technique won or failed

JSON reports omit candidate batches and source-derived graphs by default. Candidate
graph export is opt-in and can contain raw values or evidence; review it before
publishing. Use `--include-sensitive-json-artifacts` only for controlled local output.

## Dead-Letter Replay CLI

```bash
kg-construction-replay-deadletters \
  --dead-letter-path data/processed/graph_ingestion_dead_letters.jsonl \
  --store kuzu_ingestion_adapter \
  --kuzu-database data/processed/kuzu_graph
```

Use `--dry-run` to inspect pending replay records without mutating their status.

## Configuration

Primary examples:

```text
configs/data_to_kg_complete.example.yaml
configs/evaluation.example.yaml
configs/ollama_gliner_label_proposal.example.yaml
configs/gliner_ollama_relation_typing.example.yaml
configs/graph_ml.example.yaml
configs/production_ingestion.example.yaml
configs/integration_profiles.example.yaml
```

## Examples

```text
examples/data_to_kg/sample.csv
examples/data_to_kg/sample.json
examples/data_to_kg/sample.txt
```

## Experimentation Notebooks

The repo includes local-first notebooks for exercising the current implementation:

```text
notebooks/01_data_loading_and_profiling.ipynb
notebooks/02_structured_schema_mapping_rdf.ipynb
notebooks/03_jsonpath_xml_semistructured_mapping.ipynb
notebooks/04_text_extraction_gliner_ollama_labels.ipynb
notebooks/05_evaluation_resolution_production_hardening.ipynb
notebooks/06_graph_ml_capabilities.ipynb
notebooks/07_standards_mapping_and_virtual_kg.ipynb
notebooks/08_relation_extraction_and_entity_linking.ipynb
notebooks/09_enterprise_events_and_observability.ipynb
notebooks/10_graph_store_ingestion_and_exports.ipynb
notebooks/11_gold_benchmarks_and_retrieval.ipynb
notebooks/capability_coverage.yaml
```

For implementation coverage and current capability reports, see:

```text
docs/README.md
docs/getting_started.md
docs/setup_running_configuration_guide.md
docs/repo_implementation_inventory.md
docs/next_stage_capabilities.md
docs/graph_ml_capabilities.md
docs/production_ingestion.md
docs/benchmarking_and_publication.md
docs/multisource_kg_benchmark_plan.md
docs/identity_v2_migration.md
docs/packaging_and_notebook_policy.md
notebooks/README.md
reports/current_repo_capability_summary.md
reports/current_registry_summary.json
reports/current_registry_matrix.csv
docs/github_open_source_readiness_plan.md
```

## Generated Artifacts Policy

The repo keeps source code, configs, examples, reusable notebooks, and concise current reports. It does not keep stale article outputs, cache folders, smoke-test graphs, or high-volume dataset graph dumps.

Generated workspaces:

```text
data/processed/
reports/experiment_datasets/
```

Run `make smoke` to regenerate local smoke outputs. Run `scripts/run_experiment_datasets.py` when you need full dataset experiment reports and graph exports.

## Verification

```bash
make quality
make verify
make package-check
make benchmark-leaderboard
make gold-benchmark-story
make cross-source-identity-benchmark
```

`make config-check` validates every YAML config example.

`make quality` runs Ruff, mypy, vulture, config validation, notebook hygiene, and registry-to-notebook coverage checks.

`make verify` runs unit tests and smoke tests for construction plus technique evaluation.

`make package-check` builds source/wheel distributions with `python -m build` and validates them with `python -m twine check dist/*`.

`make benchmark-leaderboard` regenerates the small committed technique leaderboard from tiny golden fixtures.

`make cross-source-identity-benchmark` regenerates the versioned multi-source identity
Markdown, CSV, and JSON publication. It is a small synthetic correctness fixture, not
a production-scale or universal de-duplication claim.

`make notebook-check` verifies that notebooks are output-free, execution-count-free, path-portable, free of stale generated-artifact references, and cover every implemented or adapter technique in the registry.

## Community And Project Info

- License: Apache-2.0, see `LICENSE`.
- Contributing: see `CONTRIBUTING.md`.
- Security reporting: see `SECURITY.md`.
- Support: see `SUPPORT.md`.
- Citation: see `CITATION.cff`.
- Changelog: see `CHANGELOG.md`.
- Packaging and notebook policy: see `docs/packaging_and_notebook_policy.md`.

## Extension Checklist

- Add a loader by implementing `SourceLoader.load()` and registering it in `kg_construction.application.registry`.
- Add a mapper by returning `ConstructionBatch` from `map_source()` with row/cell/span evidence.
- Add an extractor by returning candidates only; extractors must not write directly to graph stores.
- Add a resolver by attaching canonical IDs and reversible audit metadata.
- Add a validator by returning `ValidationReport` findings with a policy.
- Add a graph store by exporting or writing validated `PropertyGraph` or `RDFGraph` objects.
- Add an evaluator by comparing `ConstructionBatch` outputs on the same dataset.
- Add an Ollama prompt by versioning it and validating its JSON before graph build.

## Implementation Inventory

The current implementation inventory is maintained in `docs/repo_implementation_inventory.md`.
