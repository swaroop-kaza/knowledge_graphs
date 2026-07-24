# Data-to-Knowledge Graph Construction Framework

`kg_construction` turns structured and unstructured data into **well-formed** knowledge graphs: deduplicated dimension nodes, relationships formed where they belong, canonicalised entities, and a schema whose quality is measured rather than assumed.

## Run it in under a minute

You need Python 3.11 or 3.12 and `git`. From the repo root:

```bash
make setup       # creates .venv, installs the package (offline), runs a health check
make quickstart  # builds sample CSV/JSON/text graphs and prints a summary
```

`make setup` creates the virtual environment for you — no manual `venv` or activation, and no PEP 668 "externally-managed-environment" errors on Homebrew/Debian Python. Run `make help` to list all tasks.

To use the `kg-construction` command directly, activate the env first: `source .venv/bin/activate` (Windows: `.\.venv\Scripts\Activate.ps1`). Prefer to manage your own environment? `python3 -m venv .venv && source .venv/bin/activate && pip install -e .` works too.

Nothing installed yet? The from-zero walkthrough (macOS/Linux/Windows, plus Ollama/Neo4j/Kuzu/GLiNER setup) is **[docs/getting_started.md](docs/getting_started.md)**.

## What it does: three pillars

The framework focuses on getting the graph *formation* right — and measures each part. Full detail and code entry points are in **[docs/knowledge_graph_modelling.md](docs/knowledge_graph_modelling.md)**.

**1. Structured data → graph.** Each column is resolved into a separate semantic role and physical materialization: a categorical value like `"Engineering"` collapses into one shared `Department` node that every row links to (never a literal repeated thousands of times), a foreign key such as `manager_id` becomes a typed edge, and a scalar value with no identity stays a property. Foreign keys resolve across tables from declared structure (database constraints, CSVW, R2RML, or explicit policy) with a profiler fallback, missing targets are never fabricated, and business keys merge across files by declared identity scope.

**2. Unstructured data → graph.** Extracted mentions are canonicalised and linked — coreference, NIL handling, and alignment to the ontology to flag off-ontology terms — and coverage is reported so missed intent becomes a number, not a guess.

**3. Schema evaluation.** The resulting graph is scored on conciseness, completeness, consistency, and structure, and gated: categorical duplication must be `0`, with zero SHACL/consistency violations.

Small gold fixtures pin the expected output, and CI enforces it (`make modelling-correctness`).

## Python usage

Build a graph from a CSV, JSON, or text file:

```python
from kg_construction import construct_kg

result = construct_kg("examples/data_to_kg/sample.csv")
print(len(result.property_graph.nodes), len(result.property_graph.edges))

text_result = construct_kg("examples/data_to_kg/sample.txt", source_type="text")
print([entity.name for entity in text_result.batch.entities])
```

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
    extractors/     regex, gazetteer, GLiNER, OpenIE, Ollama, REBEL adapters
    resolution/     canonicalisation and entity resolution/linking
    ontology/       ontology proposal and SHACL validation gate
    evaluation/     schema-quality and extraction-coverage metrics
    graph_builders/ property, RDF, and evidence graph builders
    graph_stores/   Neo4j, Kuzu, NetworkX, and RDF exports
  interfaces/cli/   construct and doctor commands
```

Check a local install with `kg-construction doctor`.

## Verify

```bash
make test        # unit tests
make quality     # ruff, mypy, vulture, config and notebook checks
```

## Project info

- License: Apache-2.0 (`LICENSE`).
- Contributing: `CONTRIBUTING.md` · Security: `SECURITY.md` · Support: `SUPPORT.md` · Citation: `CITATION.cff` · Changelog: `CHANGELOG.md`.
