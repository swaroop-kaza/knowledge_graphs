# Data-to-Knowledge Graph Construction Framework

Repo status date: 2026-07-14

`kg_construction` is a production-oriented Python framework for converting structured, semi-structured, and unstructured data into evidence-grounded knowledge graphs.

It is built for people who want to compare multiple graph construction techniques side by side instead of hard-coding one extraction path.

The core flow:

```text
Data -> Evidence -> Entities -> Relations -> Validated Graph -> Evaluation -> Recommendation
```

The framework is technique-centered, not use-case-centered. A dataset can be profiled, multiple applicable techniques can be run side by side, and the final recommendation is based on measured validation, coverage, runtime, cost, and weighted scoring.

See the small committed benchmark leaderboard for a quick answer to “which technique should I try first?”:

- [reports/benchmark_leaderboard.md](reports/benchmark_leaderboard.md)
- [reports/benchmark_leaderboard.csv](reports/benchmark_leaderboard.csv)

See the gold-standard benchmark story for the small correctness proof behind the framework:

- [reports/gold_standard_benchmark_story.md](reports/gold_standard_benchmark_story.md)
- [reports/gold_standard_benchmark_story.csv](reports/gold_standard_benchmark_story.csv)
- [reports/gold_standard_benchmark_story.json](reports/gold_standard_benchmark_story.json)

## Why This Exists

Enterprise data already contains relationships, but they are usually hidden across tables, JSON payloads, logs, documents, PDFs, emails, reports, and APIs. This repo turns those hidden relationships into traceable graph candidates, validates them, materializes them as property graph/RDF/evidence graph outputs, and ranks which construction strategy is safest for the data.

## 60-Second Data Scientist Quickstart

This path runs locally with the core install. It does not require Neo4j, Kuzu, Ollama, GLiNER, PyTorch, or graph ML extras.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
PYTHONPATH=src python examples/quickstart_60_second.py
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
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.construct \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --output data/processed/data_to_kg_text_property_graph.json \
  --rdf-output data/processed/data_to_kg_text_rdf.json \
  --evidence-output data/processed/data_to_kg_text_evidence_graph.json
```

Run a CLI technique benchmark:

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.benchmark \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --config configs/evaluation.example.yaml \
  --techniques applicable \
  --output data/processed/data_to_kg_technique_evaluation_report.md \
  --json-output data/processed/data_to_kg_technique_evaluation_report.json
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
- Exact, normalized string, RapidFuzz, and alias-table auditable entity resolution.
- Wikidata, DBpedia, OpenRefine, embedding, and cross-encoder entity linking/resolution.
- Splink and Dedupe adapter surfaces for probabilistic and active-learning entity resolution.
- Evidence-required, schema-registry, confidence, and pySHACL validation.
- Property graph, RDF triple/quad graph, and evidence graph builders.
- Neo4j/Memgraph Cypher, Kuzu manifest, NetworkX node-link, Turtle, N-Quads, and RDFLib-compatible export.
- Production graph ingestion: Neo4j managed write transactions, Kuzu write adapter, batch retry policy, and dead-letter replay.
- Production ingestion extensions: live Neo4j/Kuzu integration profiles, graph-store batch benchmark fixtures, RDF-to-Neo4j neosemantics import, and Kuzu bulk import.
- Graph ML enrichment: train/validation splits, degree features, PageRank, communities, connected components, node2vec, KG embeddings, node embeddings, link prediction, retrieval benchmarks, large-graph backends, and optional GraphSAGE adapter.
- Gold benchmark fixtures and evaluator for NER spans, RDF triples, human-reviewed relation extraction, entity resolution clusters, and GraphRAG/retrieval cases.
- Technique registry covering internal statuses and public support levels.
- Dataset profiling and applicability scoring.
- Weighted mathematical technique ranking.
- Deterministic benchmark mode with explicit tie-breaking policy.
- Best single, best hybrid, production-safe, high-recall, low-cost, and GraphRAG-ready recommendations.
- JSON and Markdown benchmark reports.
- Gold-standard benchmark story for NER spans, relation extraction, triples, entity resolution, and GraphRAG/retrieval.
- Production hardening: structured logging, run manifests, config validation, watermarks, quarantine, retry, dead-letter replay, human review queue, PII redaction, CI optional-extra matrix.

## Construction CLI

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.construct \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --output data/processed/data_to_kg_text_property_graph.json \
  --rdf-output data/processed/data_to_kg_text_rdf.json \
  --evidence-output data/processed/data_to_kg_text_evidence_graph.json \
  --neo4j-output data/processed/data_to_kg_text_neo4j.cypher
```

## Technique Evaluation CLI

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.benchmark \
  --input examples/data_to_kg/sample.txt \
  --source-type text \
  --config configs/evaluation.example.yaml \
  --techniques applicable \
  --output data/processed/data_to_kg_technique_evaluation_report.md \
  --json-output data/processed/data_to_kg_technique_evaluation_report.json \
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
- produces JSON and Markdown reports explaining why a technique won or failed

## Dead-Letter Replay CLI

```bash
PYTHONPATH=src python3 -m kg_construction.interfaces.cli.replay_dead_letters \
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
docs/setup_running_configuration_guide.md
docs/repo_implementation_inventory.md
docs/next_stage_capabilities.md
docs/graph_ml_capabilities.md
docs/production_ingestion.md
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
```

`make config-check` validates every YAML config example.

`make quality` runs Ruff, mypy, vulture, config validation, notebook hygiene, and registry-to-notebook coverage checks.

`make verify` runs unit tests and smoke tests for construction plus technique evaluation.

`make package-check` builds source/wheel distributions with `python -m build` and validates them with `python -m twine check dist/*`.

`make benchmark-leaderboard` regenerates the small committed technique leaderboard from tiny golden fixtures.

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
