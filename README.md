# Data-to-Knowledge Graph Construction Framework

`kg_construction` turns structured and unstructured data into **well-formed** knowledge graphs: deduplicated dimension nodes, relationships formed where they belong, canonicalised entities, and a schema whose quality is measured rather than assumed.

## Provision and run the complete profile

You need Python 3.11 or 3.12, `git`, `uv`, and Docker. The first provisioning
run resolves immutable dependency/model/service locks and downloads all artifacts:

```bash
make release-locks
make setup
bash run_all_approaches.sh
```

There is no reduced install or successful model fallback. On Apple Silicon the
profile runs Torch MPS and MLX; Linux selects the declared CPU or CUDA profile.
Every applicable profile also runs GLiNER NER, GLiNER RelEx, Transformers NER,
spaCy, Qwen3 through Transformers and Ollama, Neo4j, Memgraph, Ladybug, RDF, and
the evaluation gates. A missing provider makes the run fail after all providers
have been attempted.

The complete macOS/Linux walkthrough, lock semantics, disk requirements, and
failure diagnostics are in **[docs/getting_started.md](docs/getting_started.md)**.

## What it does: three pillars

The framework focuses on getting the graph *formation* right — and measures each
part. The modelling rules are in
**[docs/knowledge_graph_modelling.md](docs/knowledge_graph_modelling.md)**; the
implementation and release order is in
**[docs/stage_by_stage_implementation.md](docs/stage_by_stage_implementation.md)**.

**1. Structured data → graph.** Each column is resolved into a separate semantic role and physical materialization: a categorical value like `"Engineering"` collapses into one shared `Department` node that every row links to (never a literal repeated thousands of times), a foreign key such as `manager_id` becomes a typed edge, and a scalar value with no identity stays a property. Foreign keys resolve across tables from declared structure (database constraints, CSVW, R2RML, or explicit policy) with a profiler fallback, missing targets are never fabricated, and business keys merge across files by declared identity scope.

**2. Unstructured data → graph.** Text runs through an evidence-first pipeline (segment → mention/typing → coreference → relation/event/claim → temporal/negation/modality → evidence → identity → ontology alignment → SHACL → promote/quarantine). Every accepted claim carries exact supporting spans and complete provenance, uncertain mentions are never forced into an entity, and eight evaluations (entity, relation, event-role, intent/fact, negation, modality, temporal, attribution) run on held-out data.

**3. Formation and intended-use evaluation.** The resulting graph is scored on
identity, conciseness, completeness, consistency, evidence, and structure, then
evaluated for the declared consumer: SQL-like queries, traversals, GraphRAG,
vector retrieval, link prediction, recommendation, or reasoning. Hard gates
require zero categorical duplicates, dangling edges, unsupported promoted
claims, and SHACL violations.

Small gold fixtures pin the expected output, and CI enforces it (`make modelling-correctness`).

## Running every technique

The mandatory release surface is **105 certifiable techniques** across loaders,
mappers, extractors, resolvers, validators, graph stores, and evaluators (drawn
from a broader research inventory of 165 — 123 implemented, plus adapters and
roadmap stubs). You reach them three ways: construct a single graph, compare
many techniques on one input, or run the full hard pass. Reproduce the live
counts with `make registry-reports`.

**1. Construct one graph** (`kg-construction` / `interfaces.cli.construct`). The
same command handles all 20 source types via `--source-type`; structured sources
use the semantic mapper by default, text runs the evidence-first pipeline.

```bash
# Structured — semantic mapper (default). Also: excel, sql, json, xml, jsonld, html, logs, ...
uv run kg-construction \
  --input examples/data_to_kg/sample.csv --source-type csv \
  --config configs/data_to_kg_complete.example.yaml \
  --evaluation-suite configs/graph_evaluation_suite.sample_csv.yaml \
  --output data/processed/sample.property.json

# Unstructured — runs GLiNER, Transformers NER, spaCy, GLiNER RelEx, Qwen3 (Transformers/Ollama/MLX)
uv run kg-construction \
  --input examples/data_to_kg/sample.txt --source-type text \
  --config configs/data_to_kg_complete.example.yaml \
  --evaluation-suite configs/graph_evaluation_suite.sample_text.yaml \
  --output data/processed/text.property.json
```

Select a specific technique by editing the config: structured
`pipeline.structured_mapping` is `semantic` (default) or `direct` (baseline);
mapping standards, extractor providers, and resolvers are declared under their
config sections (see `configs/data_to_kg_complete.example.yaml`).

**2. Compare techniques on one input** (`interfaces.cli.benchmark`). Runs every
*applicable* technique, scores them, and emits a ranked report.

```bash
uv run python -m kg_construction.interfaces.cli.benchmark \
  --input examples/data_to_kg/sample.txt --source-type text \
  --config configs/evaluation.example.yaml \
  --techniques applicable \
  --output data/processed/report.md --json-output data/processed/report.json
```

`--techniques` accepts `applicable`, `all`, or a comma-separated list of
technique names.

**3. Run the complete, zero-skip hard pass** — every locked model, service,
store, and evaluation gate, offline:

```bash
make release-locks && make setup   # one-time, online
bash run_all_approaches.sh
```

| To run… | Command |
| --- | --- |
| A single graph (any of 20 source types) | `uv run kg-construction --source-type <type> ...` |
| Every applicable technique, ranked | `python -m kg_construction.interfaces.cli.benchmark --techniques applicable ...` |
| Structured baseline vs semantic | set `pipeline.structured_mapping=direct` / `semantic` in the config |
| The unit test suite (437 tests) | `make test` |
| Smoke construction + benchmark | `make smoke` |
| Formation / schema / leaderboard reports | `make modelling-correctness` · `make schema-quality` · `make benchmark-leaderboard` |
| Verify the selected runtime | `uv run kg-construction doctor --strict` |
| The full offline certification gate | `bash run_all_approaches.sh` |

Per-technique explanations and the exact commands to test each are in the
[documentation map](#documentation-map) below.

## Python usage

Build a graph from a CSV, JSON, or text file:

```python
from kg_construction import construct_kg

result = construct_kg(
    "examples/data_to_kg/sample.csv",
    source_type="csv",
    config="configs/data_to_kg_complete.example.yaml",
    evaluation_suite_path="configs/graph_evaluation_suite.sample_csv.yaml",
)
print(len(result.property_graph.nodes), len(result.property_graph.edges))

text_result = construct_kg(
    "examples/data_to_kg/sample.txt",
    source_type="text",
    config="configs/data_to_kg_complete.example.yaml",
    evaluation_suite_path="configs/graph_evaluation_suite.sample_text.yaml",
)
print([entity.name for entity in text_result.batch.entities])
print(text_result.quality_report)
print(text_result.intended_use_report)
```

Every public construction requires a sealed, dataset-specific evaluation suite
and executes formation plus all seven intended-use evaluators against the frozen
constructed graph. A self-computed fixture hash is integrity, not approval:
`resources/publication_contracts_v1.yaml` separately allowlists the reviewed
suite bytes, source snapshot, semantic configuration, model lock, governed
schema, held-out gold, builder contract, and policy hash. Repository-owned
threshold floors and ceilings cannot be weakened by editing and re-sealing a
suite. Custom suites remain valid for research workflows, but the public API
labels them research-only and refuses to publish. The suite contains gold
judgments and declarative queries, never caller-supplied predictions. Missing
support is recorded as `executed=false`, and no graph is published unless every
applicable gate passes.

Structured sources (CSV/Excel/SQL) are mapped semantically by default. To resolve
foreign keys across tables, declare their structure in a source catalog:

```python
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

Export without any live database:

```python
from kg_construction.infrastructure.graph_stores import Neo4jExporter, RDFExporter

turtle = RDFExporter().export_rdf_graph(result.rdf_graph)["turtle"]
cypher = Neo4jExporter().export_property_graph(result.property_graph)
```

## Package layout

```text
src/kg_construction/
  domain/           Pydantic models: sources, evidence, entities, relations, graphs, schema
  ports/            Protocol interfaces for plugin-style extension
  application/      column profiling, resolved modelling contract, source catalog, semantic mapper, workflow, public API
  infrastructure/
    loaders/        CSV, JSON, text, Excel, SQL, API, XML, PDF, email, chat, ...
    mapping/        direct table, JSONPath, and standards mappings (R2RML/RML/CSVW)
    extractors/     deterministic evidence, GLiNER/RelEx, spaCy, Transformers, MLX, Ollama
    resolution/     canonicalisation and entity resolution/linking
    ontology/       ontology proposal and SHACL validation gate
    evaluation/     formation, extraction, SQL/NoSQL, GraphRAG, vector, link-prediction, recommendation, and reasoning metrics
    graph_builders/ property, RDF, and evidence graph builders
    graph_stores/   Neo4j/vector, Memgraph, Ladybug, NetworkX, and RDF stores/exports
  interfaces/cli/   construct and doctor commands
```

Check the entire selected runtime with
`uv run kg-construction doctor --strict`. The doctor reports only pass or fail;
it does not downgrade missing models or services to warnings.

## Verify

```bash
make test        # unit tests
make quality     # ruff, mypy, vulture, config and notebook checks
```

## Documentation map

| Read this to… | Document |
| --- | --- |
| See the whole system: hexagonal layers, data flow, registry, hard pass (with diagrams) | [docs/architecture.md](docs/architecture.md) |
| Understand every **structured → graph** technique technically | [docs/techniques_structured_to_kg.md](docs/techniques_structured_to_kg.md) |
| Understand every **unstructured → graph** technique technically | [docs/techniques_unstructured_to_kg.md](docs/techniques_unstructured_to_kg.md) |
| Know the **metrics, weighting, and thresholds** for every evaluation | [docs/evaluation_metrics.md](docs/evaluation_metrics.md) |
| **Test** each technique (test ↔ technique map, fixtures, gates) | [docs/testing_guide.md](docs/testing_guide.md) |
| Provision the mandatory runtime and run the hard pass | [docs/getting_started.md](docs/getting_started.md) |
| The modelling doctrine and stage-by-stage exit gates | [docs/knowledge_graph_modelling.md](docs/knowledge_graph_modelling.md) · [docs/stage_by_stage_implementation.md](docs/stage_by_stage_implementation.md) |

## Project info

- License: Apache-2.0 (`LICENSE`).
- Contributing: `CONTRIBUTING.md` · Security: `SECURITY.md` · Support: `SUPPORT.md` · Citation: `CITATION.cff` · Changelog: `CHANGELOG.md`.
