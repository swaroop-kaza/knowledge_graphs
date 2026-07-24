# Notebooks

These notebooks are public-facing examples for the current `kg_construction` implementation. They are intentionally output-free so GitHub diffs stay readable.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,rdf,graph,notebook]"
jupyter lab
```

Optional notebooks or live integrations may need additional extras:

```bash
python -m pip install -e ".[gliner,llm,linking,dedup,ml]"
```

## Suggested order

| Notebook | What it demonstrates |
| --- | --- |
| `01_data_loading_and_profiling.ipynb` | Load CSV, JSON, text, Excel, SQL, API, XML, Markdown, report, PDF, email, and chat fixtures into `SourceRecord` objects, then profile them. |
| `02_structured_schema_mapping_rdf.ipynb` | Map structured rows into graph candidates, propose schema, validate, and build property/RDF/evidence graphs. |
| `03_jsonpath_xml_semistructured_mapping.ipynb` | Use JSONPath, dotted-path JSON flattening, XML loading, and source-path evidence. |
| `04_text_extraction_gliner_ollama_labels.ipynb` | Compare regex, gazetteer, fake Ollama label proposal, and fake GLiNER span extraction offline. |
| `05_evaluation_resolution_production_hardening.ipynb` | Run deterministic technique evaluation, entity resolution, manifests, structured logging, watermarks, quarantine, human review, and PII redaction. |
| `06_graph_ml_capabilities.ipynb` | Run graph analytics, PageRank, community detection, node embeddings, and link prediction. |
| `07_standards_mapping_and_virtual_kg.ipynb` | Run R2RML, RML, CSVW, JSON-LD, and OBDA/virtual graph mapping examples. |
| `08_relation_extraction_and_entity_linking.ipynb` | Run OpenIE, dependency-pattern relations, GLiNER plus Ollama relation typing, REBEL adapters, external linkers, embedding/cross-encoder resolution, and Splink/Dedupe adapter boundaries. |
| `09_enterprise_events_and_observability.ipynb` | Run web/HTML/log/event/scientific/JSON-LD loaders plus HTML metadata, log event, trace/span, and event graph mappers. |
| `10_graph_store_ingestion_and_exports.ipynb` | Run graph store exporters, Neo4j/Ladybug ingestion, RDF-to-Neo4j neosemantics calls, Ladybug bulk import, retries, and dead-letter replay. |
| `11_gold_benchmarks_and_retrieval.ipynb` | Run gold benchmark evaluation, graph train/validation splits, node2vec/KG embeddings, GraphSAGE prep, large-graph backends, retrieval comparison, and graph-store benchmark fixtures. |

## Capability coverage

`capability_coverage.yaml` maps registry techniques to notebooks. `make notebook-check` verifies:

- every implemented or adapter technique in `default_registry()` has notebook coverage;
- every stub technique is documented as an extension point;
- every referenced notebook exists;
- notebook files remain output-free and path-portable.

## Hygiene policy

- Do not commit notebook outputs.
- Do not commit execution counts.
- Do not commit absolute local paths.
- Do not write generated graphs or reports into the notebook directory.
- Use `data/processed/` for local generated outputs.
- Run `make notebook-check` before committing notebook changes.
