# Knowledge Graph Construction Repo Implementation Inventory

Last inspected: 2026-07-13

This document captures what is currently implemented in this repository after the recent file cleanup. It is not a Medium article. It is a repo-facing inventory of implemented capabilities, adapter boundaries, stubs, configs, notebooks, tests, generated reports, and experiment evidence.

## 1. Current repo purpose

- Package name: `kg-construction`
- Main Python package: `src/kg_construction`
- Primary goal: convert structured, semi-structured, and unstructured data into evidence-backed knowledge graph outputs.
- Core workflow:

```text
Source data
-> SourceRecord
-> EvidenceRef
-> EntityCandidate / RelationCandidate / GraphFact
-> validation
-> property graph / RDF graph / evidence graph
-> metrics
-> ranking
-> recommendation
```

- Design style:
  - Technique-centered, not use-case-centered.
  - Evidence-first: every candidate should trace back to source evidence.
  - Config-driven: YAML controls pipeline, LLM, evaluation, and production options.
  - Extensible: ports and registry make new loaders, mappers, extractors, resolvers, validators, stores, and evaluators pluggable.
  - Production-aware: run manifests, deterministic benchmarks, structured logging, watermarking, quarantine, human review, and PII redaction are implemented.

## 2. Current top-level files

| Path | Purpose |
| --- | --- |
| `README.md` | Repo overview, package layout, commands, implemented capability summary. |
| `pyproject.toml` | Package metadata, dependencies, optional extras, scripts, lint/type/test config. |
| `Makefile` | `compile`, `test`, `smoke`, `quality`, `verify`, `package-check`, `benchmark-leaderboard`, registry report, and `clean` targets. |
| `configs/` | Example YAML configs for complete pipeline, evaluation, Ollama-label-plus-GLiNER, GLiNER-plus-Ollama relation typing, and graph ML enrichment. |
| `src/kg_construction/` | Main implementation. |
| `tests/` | Unit and integration-style tests for construction, loaders, relation extraction, evaluation, production hardening. |
| `examples/data_to_kg/` | Small sample CSV, JSON, and text inputs. |
| `examples/quickstart_60_second.py` | Core-install data scientist quickstart covering CSV, text, JSON, benchmarking, RDF export, Neo4j Cypher export, and NetworkX node-link export without live external services. |
| `notebooks/` | Notebook entry points plus a notebook usage/hygiene guide. |
| `scripts/` | Dataset download, dataset experiment runner, registry report, and benchmark leaderboard generator scripts. |
| `data/external/` | README-only external data workspace; downloaded datasets are generated locally and ignored. |
| `data/processed/` | Generated local smoke-test and benchmark workspace; only the workspace README is kept. |
| `reports/` | Concise current capability reports plus a small committed benchmark leaderboard generated from tiny fixtures. |
| `docs/` | Documentation hub plus setup/running/configuration, current implementation, graph ML, production ingestion, packaging/notebook, GitHub readiness, and next-stage capability docs. |
| GitHub community files | `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, `SUPPORT.md`, `CHANGELOG.md`, `CITATION.cff`, issue templates, PR template, Dependabot, CodeQL. |

## 3. Package dependencies and optional extras

### Required dependencies

- `pydantic`: strict domain/config models.
- `PyYAML`: YAML config loading.
- `jsonpath-ng`: real JSONPath selector support.
- `openpyxl`: Excel loader.
- `SQLAlchemy`: SQL database loader.
- `pypdf`: PDF loader.
- `rapidfuzz`: fuzzy entity resolution.
- `rdflib`: RDF graph serialization and RDFLib graph/dataset conversion.
- `pyshacl`: standards-based SHACL validation.

### Optional extras

| Extra | Purpose |
| --- | --- |
| `dev` | Build, mypy, pytest, ruff, vulture, type stubs. |
| `gliner` | Real GLiNER model loading. |
| `llm` | LangChain/Ollama/Neo4j-related optional LLM integrations. |
| `bert` | Transformer/torch path for future NER adapters. |
| `spacy` | spaCy extraction adapters. |
| `graph` | NetworkX, NumPy, Neo4j driver, and Kuzu embedded graph database support. |
| `rdf` | RDFLib and pySHACL, also included in required deps now. |
| `linking` | SentenceTransformers embedding and cross-encoder models for semantic entity resolution. |
| `dedup` | Splink/Dedupe and duplicate resolution dependencies. |
| `ml` | NumPy, Torch Geometric, and PyKEEN for optional graph ML and embedding paths. |

## 4. Registry coverage

The technique registry currently contains these public support levels:

| Public support level | Count | Meaning |
| --- | ---: | --- |
| Stable | 76 | Safe to use now through the documented API and normal test coverage. |
| Optional adapter | 29 | Works when the dependency, model, service, or project-specific wiring is configured. |
| Experimental | 16 | Implemented for research/prototyping, but API, scoring, or model behavior may change. |
| Planned | 29 | Registered roadmap idea or extension point, not ready for normal use. |
| Total | 150 | All registered techniques across categories. |

The internal registry status is still retained for planners, reports, and tests:

| Internal status | Count | Meaning |
| --- | ---: | --- |
| Implemented | 104 | Working code exists and is registered. |
| Adapter | 17 | Boundary exists for a real implementation or optional workflow, but requires extra configuration, dependency, or project-specific wiring. |
| Stub | 29 | Named future capability exists as an explicit extension point. |
| Total | 150 | All registered techniques across categories. |

Category counts:

| Category | Count |
| --- | ---: |
| loader | 20 |
| mapper | 22 |
| extractor | 20 |
| schema | 8 |
| resolution | 16 |
| validation | 9 |
| graph_builder | 3 |
| graph_store | 8 |
| llm | 1 |
| enrichment | 24 |
| evaluation | 2 |
| hybrid | 8 |
| production | 9 |

## 5. Domain model layer

Location:

- `src/kg_construction/domain/models.py`
- `src/kg_construction/domain/evidence.py`
- `src/kg_construction/domain/graph_objects.py`
- `src/kg_construction/domain/schema.py`

Implemented pointers:

- Deterministic text normalization and hashing:
  - `normalize_text`
  - `content_hash`
  - `stable_id`
- Source abstraction:
  - `SourceRecord.from_text`
  - `SourceRecord.from_rows`
  - `SourceRecord.from_json`
- Evidence abstraction:
  - `EvidenceRef`
  - `row_evidence`
  - `cell_evidence`
  - `span_evidence`
- Candidate objects:
  - `EntityCandidate`
  - `RelationCandidate`
  - `GraphFact`
  - `ConstructionBatch`
- Graph outputs:
  - `PropertyNode`
  - `PropertyEdge`
  - `PropertyGraph`
  - `RDFTriple`
  - `RDFQuad`
  - `RDFGraph`
- Schema objects:
  - `PropertyDefinition`
  - `NodeTypeDefinition`
  - `RelationshipTypeDefinition`
  - `GraphSchema`
  - `SchemaDiffReport`
  - `SchemaRegistry`
- Schema lifecycle states:
  - `proposed`
  - `approved`
  - `rejected`
  - `promoted`
- GraphSchema can emit:
  - SHACL Turtle.
  - OWL-lite/RDFS vocabulary Turtle.
  - lifecycle status propagation.
  - schema diffs across runs.

## 6. Port interfaces

Location:

- `src/kg_construction/ports/`

Implemented protocol surfaces:

| File | Protocol |
| --- | --- |
| `loaders.py` | `SourceLoader` |
| `mappers.py` | `SourceMapper` |
| `extractors.py` | `EntityExtractor`, `RelationExtractor` |
| `resolvers.py` | `EntityResolver` |
| `validators.py` | `GraphValidator` |
| `graph_store.py` | `GraphWriter`, `GraphStore` |
| `evaluators.py` | `ConstructionEvaluator` |
| `enrichers.py` | `GraphEnricher` |
| `preprocessors.py` | `SourcePreprocessor` |

Why this matters:

- New techniques can be added without changing the core workflow.
- Implementations return construction batches or graph objects instead of directly mutating storage.
- Extractors produce candidates only; graph builders and validators decide what gets materialized.

## 7. Configuration system

Location:

- `src/kg_construction/application/config.py`
- `configs/data_to_kg_complete.example.yaml`
- `configs/evaluation.example.yaml`
- `configs/ollama_gliner_label_proposal.example.yaml`
- `configs/gliner_ollama_relation_typing.example.yaml`
- `configs/graph_ml.example.yaml`
- `configs/production_ingestion.example.yaml`
- `configs/integration_profiles.example.yaml`

Implemented config models:

- `PipelineConfig`
- `OllamaSettings`
- `TechniqueConfig`
- `StructuredLoggingSettings`
- `RunManifestSettings`
- `IncrementalSyncSettings`
- `ErrorQuarantineSettings`
- `HumanReviewQueueSettings`
- `PIIRedactionSettings`
- `ProductionSettings`
- `DataToKGConfig`

Implemented validation:

- Pipeline name and schema version cannot be empty.
- Ollama base URL must be HTTP/HTTPS.
- Unknown technique names are rejected.
- `ollama_graph` requires `llm.provider = ollama`.
- `pipeline.evidence_required = true` requires evidence validation to be enabled.
- Enabled production paths must be non-empty.

Example configs currently present:

- `configs/data_to_kg_complete.example.yaml`
  - Enables the full conceptual pipeline across loaders, mappers, extractors, resolution, validation, graph stores, enrichment, evaluation, and production.
- `configs/evaluation.example.yaml`
  - Enables deterministic benchmark mode with fixed runtime/cost values.
  - Configures weights, penalties, production thresholds, and Ollama settings.
- `configs/ollama_gliner_label_proposal.example.yaml`
  - Configures LLM label proposal, prompt template, label filters, manual review, GLiNER settings, and baseline comparisons.
- `configs/gliner_ollama_relation_typing.example.yaml`
  - Configures GLiNER entity labels, allowed relationship types, confidence threshold, and schema version for LLM relation typing.
- `configs/graph_ml.example.yaml`
  - Configures graph analytics, PageRank, community detection, connected components, node embeddings, and link prediction.
- `configs/production_ingestion.example.yaml`
  - Documents retry, dead-letter, Neo4j ingestion, Kuzu ingestion, and replay CLI settings.
- `configs/integration_profiles.example.yaml`
  - Documents opt-in Neo4j/Kuzu live integration profiles and graph-store batch benchmark sizing.

## 8. Application orchestration

Location:

- `src/kg_construction/application/workflow.py`
- `src/kg_construction/application/pipeline.py`
- `src/kg_construction/application/public_api.py`
- `src/kg_construction/application/production.py`
- `src/kg_construction/application/planner.py`
- `src/kg_construction/__init__.py`

Implemented pointers:

- `ConstructionWorkflow`
  - Runs mappers and extractors over each source.
  - Merges outputs into a `ConstructionBatch`.
  - Applies entity resolvers.
  - Applies composite validation.
  - Builds property graph, RDF graph, and evidence graph.
  - Computes metrics.
  - Writes run manifests when configured.
  - Applies incremental sync when configured.
  - Applies PII redaction before processing when configured.
  - Quarantines exceptions and validation findings when configured.
  - Enqueues human review items when configured.
  - Supports `continue_on_error`.
- `ConstructionResult`
  - Returns the batch, validation report, three graph outputs, metrics, run manifest, quarantine records, human review items, and skipped sources.
- Stable top-level public API:
  - `load_sources`: loads paths, directories, raw text, `SourceRecord`, or source-record sequences into evidence-backed source records.
  - `profile_dataset`: profiles paths or source records for technique selection.
  - `construct_kg`: runs a conservative local-first construction workflow and returns `ConstructionResult`.
  - `evaluate_techniques`: profiles, runs, scores, and ranks construction techniques for a path.
  - Re-exports `DataToKGConfig`, `ConstructionWorkflow`, `ConstructionResult`, and `default_registry` from `kg_construction`.
- `DataToKGPipeline`
  - Builds workflows from `DataToKGConfig`.
- `TechniquePlanner`
  - Orders techniques, preferring deterministic methods before expensive or human-review-heavy paths.
- `build_production_services`
  - Converts production config into structured logger, manifest writer, watermark store, quarantine store, human review queue, and PII policy objects.

## 9. Input loaders

Location:

- `src/kg_construction/infrastructure/loaders/`

Implemented loaders:

| Loader | Input | What it emits |
| --- | --- | --- |
| `CsvLoader` | CSV | One `SourceRecord` with normalized rows and source metadata. |
| `JsonLoader` | JSON | `SourceRecord` with original JSON payload preserved. |
| `TextLoader` | Plain text | `SourceRecord` with text content. |
| `ExcelLoader` | XLSX/XLSM | One `SourceRecord` per worksheet with headers, rows, sheet metadata, source evidence. |
| `SQLDatabaseLoader` | SQLite/SQLAlchemy database or query | One `SourceRecord` per table/query with columns, PKs, FKs, row count, source evidence. |
| `APIResponseLoader` | Captured JSON file or live URL | Structured rows from `data` array when possible; otherwise text JSON payload. |
| `StructuredAPILoader` | API-like structured payloads | Alias-style loader for structured API records. |
| `XMLLoader` | XML | Flattened XML records with row paths and source evidence. |
| `MarkdownLoader` | Markdown | Text content with heading metadata and source evidence. |
| `ReportLoader` | Plain text or Markdown-style reports | Text content with inferred sections and source evidence. |
| `PDFLoader` | Digitally-born PDF | Extracted text, page metadata, document metadata; includes fallback string extraction. |
| `EmailLoader` | `.eml` | Email body plus headers, subject, message id, attachment count. |
| `ChatTranscriptLoader` | Chat transcript text | Turn-level rows with speaker, timestamp, message, line number. |
| `HTMLLoader` | Local HTML | Visible text, title, meta tags, canonical links, anchors, headings, JSON-LD, source evidence. |
| `WebPageLoader` | URL or local HTML fixture | Fetches or reads HTML, preserves final URL/status/content type when remote, emits webpage source metadata. |
| `LogLoader` | `.log` text | Parses JSON logs, Apache common/combined logs, key-value logs, and generic timestamp/level/message lines. |
| `ScientificPaperLoader` | JATS/TEI/XML, PDF, text | Extracts title, abstract, authors, DOI, journal, publication date, sections, citations, and source evidence. |

Loader stubs:

- `DataWarehouseExportLoader`
- `JSONLDLoader`
- `EventPayloadLoader`

Test coverage:

- `tests/test_enterprise_loaders.py`
  - Excel stable source IDs/evidence.
  - SQL table records and schema metadata.
  - API response row extraction.
  - XML repeated-element flattening with path evidence.
  - Markdown heading preservation.
  - Report section detection.
  - PDF text/page evidence.
  - Email header/body extraction.
  - Chat transcript turn records.
  - HTML metadata, links, JSON-LD, headings, visible text.
  - Web page source typing for local fixtures and URL-ready loader path.
  - JSON, Apache, key-value, and generic log parsing.
  - Scientific paper JATS metadata, sections, citations, DOI, and authors.
  - HTML metadata graph mapping.
  - Log event graph mapping.
  - Registry/profiler recognition of enterprise loaders.

Fixtures:

- `tests/fixtures/enterprise_loaders/service_inventory.xlsx`
- `tests/fixtures/enterprise_loaders/enterprise.sqlite`
- `tests/fixtures/enterprise_loaders/api_response.json`
- `tests/fixtures/enterprise_loaders/services.xml`
- `tests/fixtures/enterprise_loaders/incident_review.md`
- `tests/fixtures/enterprise_loaders/ops_report.report`
- `tests/fixtures/enterprise_loaders/incident_brief.pdf`
- `tests/fixtures/enterprise_loaders/incident_email.eml`
- `tests/fixtures/enterprise_loaders/support_chat.chat`
- `tests/fixtures/enterprise_loaders/recommendation_runbook.html`
- `tests/fixtures/enterprise_loaders/service_events.log`
- `tests/fixtures/enterprise_loaders/kg_construction_article.jats`

## 10. Dataset profiling and technique selection

Location:

- `src/kg_construction/application/dataset_profiler.py`
- `src/kg_construction/application/technique_selector.py`

Implemented dataset profiling:

- Data type detection:
  - structured
  - semi-structured
  - unstructured
  - mixed
  - unknown
- File format inference.
- Source count.
- Record count.
- Character count.
- Schema presence.
- Primary key inference.
- Foreign key inference.
- Column type profile.
- Null rates.
- Duplicate rate.
- Text density.
- Table density.
- Nestedness score.
- Schema drift placeholder score.
- Document length.
- Chunkability score.
- Language estimate.
- Entity density score.
- Relation density score.
- OCR/layout complexity score.
- Ontology alignment score.
- Source reliability score.
- Expected graph size.
- Structuredness score.
- Schema quality score.
- Text complexity score.
- Evidence quality score.
- Data volume score.
- Warnings.

Implemented applicability scoring:

- `ApplicabilityWeights`
  - format match
  - schema match
  - structure match
  - text suitability
  - ontology suitability
  - scale suitability
  - cost suitability
- `TechniqueSelector`
  - Scores each registered technique against the dataset profile.
  - Applies threshold filtering.
  - Supports force-all mode.
  - Returns explainable score components and reason strings.

## 11. Structured data mapping

Location:

- `src/kg_construction/infrastructure/mapping/direct_table_mapper.py`

Implemented pointers:

- `DirectTableMapper`
  - Maps rows to entity candidates.
  - Uses configured `id_columns` for stable identity.
  - Uses configured `name_column` or falls back to common columns such as `name`, `title`, `id`, `email`.
  - Maps row columns to entity properties.
  - Adds `_source_table`, `_row_index`, and `_stable_row_id`.
  - Emits row-level evidence.
  - Builds relationships from configured foreign keys.
- `DirectTableMappingConfig`
  - table name
  - node type
  - ID columns
  - name column
  - property columns
  - foreign key mappings
  - schema version
- `ForeignKeyMappingConfig`
  - source column
  - target column
  - relationship type
  - optional target node type
- `profile_records`
  - column list
  - row count
  - inferred column types
  - null counts
  - distinct counts
- `infer_column_type`
  - integer
  - number
  - boolean
  - string
  - unknown
- Potential key column detection.

Registered implemented mapper techniques backed by `DirectTableMapper`:

- `direct_table_mapper`
- `table_to_node_mapping`
- `row_to_entity_mapping`
- `primary_key_to_id_mapping`
- `foreign_key_to_edge_mapping`
- `column_to_property_mapping`
- `schema_profiling`
- `column_type_inference`

Registered adapter:

- `relationship_inference_from_keys`

Test coverage:

- CSV direct mapping builds entities, relations, and evidence.
- Profiler detects structured keys and scores.
- Applicability prefers direct mapping for CSV.

## 12. Semi-structured JSON mapping

Location:

- `src/kg_construction/infrastructure/mapping/semi_structured.py`

Implemented mappers:

### DottedPathJsonMapper

- Lightweight mapper that flattens nested JSON into dotted property names.
- Uses `flatten_json`.
- Converts flattened JSON records through `DirectTableMapper`.
- Best for quick nested JSON inspection.
- Not a full JSONPath query engine.

### JsonPathMapper

- Uses `jsonpath-ng` for real JSONPath selectors.
- Supports:
  - root path selection.
  - name path selection.
  - configured property paths.
  - required property paths.
  - scalar leaf extraction.
  - exact JSON source-path evidence.
  - configured array-to-edge mappings.
  - automatic array-to-edge mappings.
  - child entity creation from object arrays.
  - relationship label generation from JSON paths.
  - entity type generation from JSON paths.
  - missing path tracking.
- Evidence location uses:

```json
{"kind": "jsonpath", "path": "$.service.errors[0].code"}
```

Config objects:

- `JsonPropertyMapping`
- `JsonArrayEdgeMapping`
- `JsonPathMappingConfig`

Registered implemented mapper techniques:

- `jsonpath_mapper`
- `dotted_path_json_mapper`
- `log_event_mapper`
- `html_metadata_mapper`

Registered stubs:

- `xmlpath_mapper`
- `api_response_mapper`
- `event_graph_mapper`

Test coverage:

- JSONPath mapping preserves source paths.
- JSONPath mapping creates array-derived edges.
- Repeated array objects stay distinct.
- Missing required paths are tracked.
- HTML metadata mapping emits document, meta tag, link, and JSON-LD graph candidates.
- Log event mapping emits event nodes and links to systems, loggers, client IPs, endpoints, and levels.

## 13. RDF and standards mapping

Location:

- `src/kg_construction/infrastructure/mapping/standards.py`
- `src/kg_construction/infrastructure/mapping/rdf_mapping_stubs.py`

Implemented standards mappers:

- `r2rml_mapper`
- `rml_mapper`
- `csvw_mapper`

R2RML support:

- Parses Turtle mapping documents.
- Supports logical tables through `rr:logicalTable`, `rr:tableName`, and already-loaded relational rows.
- Supports subject maps through `rr:subjectMap`, `rr:template`, `rr:class`, and shortcut `rr:subject`.
- Supports predicate-object maps through `rr:predicateObjectMap`.
- Supports `rr:predicate`, `rr:predicateMap`, `rr:object`, and `rr:objectMap`.
- Supports `rr:column`, `rr:constant`, `rr:template`, `rr:termType`, `rr:datatype`, and `rr:language`.
- Emits evidence-backed `GraphFact` objects that materialize as real RDF triples.

RML support:

- Parses Turtle mapping documents.
- Supports `rml:logicalSource`, `rml:source`, `rml:iterator`, `rml:referenceFormulation`, and `rml:reference`.
- Supports CSV row references.
- Supports JSONPath iterators and JSON references for JSON payloads.
- Supports XML-style logical sources over flattened XML loader rows and simple XML text contexts.
- Reuses the R2RML triples-map model for subject and predicate-object maps.

CSVW support:

- Loads CSVW metadata JSON from a dict or file path.
- Supports table matching by CSVW `url`.
- Supports table-level `aboutUrl` URI templates.
- Supports `tableSchema.columns`.
- Supports `name`, `titles`, `datatype`, `propertyUrl`, `valueUrl`, `aboutUrl`, `suppressOutput`, and `required` column metadata.
- Supports `primaryKey`.
- Supports `foreignKeys` with `columnReference` and referenced `resource`/`columnReference`.
- Supports CSVW dialect parsing and validation.
- Emits row entities, RDF facts, and relation candidates for foreign keys.

Additional implemented standards adapters:

- `jsonld_mapper`
- `obda_virtual_graph_mapper`
- R2RML/RML referencing object maps with join conditions.
- Cross-source joins through `map_sources(...)`.

Purpose:

- Provide executable standards-backed RDF generation for JSON-LD, virtual OBDA plans, relational rows, heterogeneous RML sources, and CSVW tabular metadata.

## 14. Text extraction

Location:

- `src/kg_construction/infrastructure/extractors/`

### RegexEntityExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/regex.py`

Default extracted entity types:

- `EmailAddress`
- `URL`
- `Date`
- `Identifier`
- `Version`

Implemented behavior:

- Uses compiled regex patterns.
- Preserves exact character spans.
- Deduplicates same type/span matches.
- Emits span evidence.
- Sets high confidence by pattern.

### GazetteerEntityExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/gazetteer.py`

Implemented behavior:

- Uses configured dictionary of `entity_type -> terms`.
- Case-insensitive exact term matching with word boundaries.
- Emits span evidence.
- Preserves gazetteer term, start, and end in properties.

### GlinerEntityExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/gliner.py`

Implemented behavior:

- Supports real GLiNER if optional package/model weights are installed.
- Also supports injected model protocol for tests and offline experiments.
- Requires dynamic labels.
- Normalizes label names into graph-friendly names.
- Emits span evidence.
- Preserves GLiNER label, model name, threshold, start/end offsets.
- Raises `AdapterNotConfigured` if no model or optional dependency is available.

### OllamaGraphExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/ollama_graph.py`

Implemented behavior:

- Uses local Ollama provider.
- Builds schema-constrained graph extraction prompt.
- Requires structured JSON response.
- Validates JSON shape before converting to candidates.
- Emits entities and relations as candidates only.
- Requires `evidence_text`; unsupported candidates are dropped when evidence is required.
- Maps LLM node `id_hint` to generated entity IDs for relation construction.
- Enforces allowed node types and relationship types.
- Stores prompt version in properties/metadata.

### OpenIERelationExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/relations.py`

Implemented behavior:

- Emits open-domain subject-relation-object triples as `RelationCandidate` objects.
- Supports an injected Stanford/CoreNLP/OpenIE-style backend protocol.
- Provides an offline deterministic fallback for common enterprise relation phrases such as `works at`, `depends on`, `reports to`, `uses`, `affected`, and `calls`.
- Creates endpoint entity candidates when source/target spans are present in the text.
- Preserves relation phrase, backend name, confidence, and source-span evidence.

### DependencyPatternRelationExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/relations.py`

Implemented behavior:

- Supports an injected dependency backend for spaCy/Stanza-style dependency matchers.
- Provides deterministic source/target patterns for `REPORTS_TO`, `DEPENDS_ON`, `USES`, `AFFECTED`, and `CALLS`.
- Emits evidence-backed relation candidates and endpoint entities.
- Preserves the matched pattern name in relation properties.

### GlinerOllamaRelationTypingExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/relations.py`

Implemented behavior:

- Runs GLiNER first to extract evidence-backed entity spans.
- Sends only those extracted entities to Ollama for relation typing.
- Uses the relation-typing prompt version `ollama_relation_typing_v1`.
- Loads settings from `configs/gliner_ollama_relation_typing.example.yaml` through `RelationTypingConfig.from_yaml(...)`.
- Enforces allowed relationship types and minimum confidence.
- Drops LLM relationships that do not contain source evidence.
- Returns GLiNER entities plus LLM-typed relation candidates in one batch.

### RebelRelationExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/relations.py`

Implemented behavior:

- Acts as a REBEL/transformer relation extraction adapter.
- Supports injected Hugging Face-style text-to-text generation pipeline for deterministic tests.
- Lazily loads `transformers` only when live model inference is requested.
- Parses REBEL linearized triples in the form `<triplet> subject <subj> object <obj> relation`.
- Converts generated triples into evidence-backed entity and relation candidates.

### TransformerRelationExtractor

Implemented in:

- `src/kg_construction/infrastructure/extractors/relations.py`

Implemented behavior:

- Generic transformer relation extraction adapter using the same REBEL-style decoding path.

Registered text extractor stubs:

- `spacy_entity_ruler_extractor`
- `sentence_splitter_adapter`
- `pos_tagging_adapter`
- `coreference_adapter`
- `spacy_ner_extractor`
- `transformer_ner_extractor`
- `pairwise_relation_classifier_adapter`
- `joint_entity_relation_extractor`
- `supervised_relation_extractor`

Test coverage:

- Regex preserves span evidence.
- Ollama provider retries invalid JSON.
- Ollama graph extractor returns candidates.
- GLiNER extractor works through injected model protocol.
- OpenIE extracts evidence-backed relation candidates.
- Dependency-pattern extraction emits configured relation types.
- GLiNER entities plus Ollama relation typing emits typed edges with evidence.
- REBEL adapter parses generated triples into graph candidates.

## 15. LLM provider and prompt layer

Location:

- `src/kg_construction/infrastructure/llm/ollama.py`
- `src/kg_construction/infrastructure/llm/prompts.py`
- `src/kg_construction/infrastructure/llm/label_proposal.py`

### OllamaProvider

Implemented behavior:

- Calls local Ollama `/api/chat`.
- Uses `format = json` when structured output is enabled.
- Allows injected transport for deterministic tests.
- Extracts JSON from fenced or mixed responses.
- Retries invalid JSON with a repair prompt.
- Supports max retry configuration.
- Raises `OllamaOutputError` after final invalid response.

### Prompt helpers

Implemented prompt builders:

- `build_graph_extraction_prompt`
- `build_repair_prompt`
- `build_label_proposal_prompt`
- `build_relation_typing_prompt`

Implemented validators:

- `validate_graph_payload`
- `validate_label_payload`
- `validate_relation_typing_payload`

### OllamaLabelProposalExtractor

Implemented behavior:

- Uses Ollama to propose entity labels for a GLiNER pass.
- Loads label proposal config from YAML.
- Supports:
  - prompt template.
  - allowed label count.
  - seed labels.
  - include filters.
  - exclude filters.
  - minimum confidence.
  - auto-approval.
- Normalizes labels.
- Deduplicates labels.
- Applies deterministic review.
- Returns proposed labels and reviewed labels.

Registered LLM technique:

- `ollama_provider`

## 16. Hybrid GLiNER plus LLM-label workflow

Location:

- `src/kg_construction/application/hybrid_gliner.py`
- `configs/ollama_gliner_label_proposal.example.yaml`

Implemented workflow:

```text
sample text
-> Ollama proposes entity labels
-> deterministic/manual label review
-> GLiNER extracts spans using reviewed labels
-> regex/gazetteer/Ollama baselines optionally run
-> notebook-friendly comparison rows
```

Key function:

- `extract_entities_with_ollama_labels_and_gliner`

Config models:

- `HybridGlinerPipelineConfig`
- `LabelReviewSettings`
- `ExtractionComparisonRow`
- `HybridGlinerExtractionResult`

Implemented helpers:

- `apply_manual_label_review`
- `compare_entity_extraction_strategies`
- `comparison_row`

Baselines supported:

- Regex entity extraction.
- Gazetteer entity extraction.
- Optional Ollama graph extraction.

Registered hybrid techniques:

Implemented:

- `hybrid_rules_gazetteer_regex`

Adapter surfaces:

- `hybrid_rules_ml_ollama`
- `hybrid_structured_then_text`
- `hybrid_gliner_then_ollama_relations`
- `hybrid_ollama_schema_then_gliner`
- `hybrid_openie_then_ollama_canonicalization`
- `hybrid_deterministic_then_llm_enrichment`
- `hybrid_human_reviewed_schema`

Experiment evidence:

- Current report includes `ollama_label_proposal_plus_gliner_offline_adapter`.
- GLiNER optional package/model weights were not installed during the run.
- The implemented GLiNER model protocol was exercised with WikiANN gold spans.
- Label proposal logic was exercised with an offline static Ollama provider.
- Output from current report:
  - proposed labels: `Person`, `Organization`, `Location`
  - reviewed labels: `Person`, `Organization`, `Location`
  - entities: 359
  - RDF triples: 6061
  - RDF quads: 8574
  - evidence coverage: 1.0
  - unsupported extraction rate: 0.0
  - SHACL conforms: true

## 17. Structured data to schema and ontology

Location:

- `src/kg_construction/infrastructure/ontology/schema_proposal.py`
- `src/kg_construction/domain/schema.py`
- `src/kg_construction/infrastructure/ontology/schema_registry.py`

Implemented workflow:

- `SchemaProposalWorkflow`
  - Accepts a `ConstructionBatch` or list of `SourceRecord`.
  - Profiles structured records.
  - Infers table name.
  - Infers node type.
  - Infers properties from columns.
  - Infers primary keys.
  - Infers foreign keys.
  - Infers relationship types.
  - Emits `GraphSchema`.
  - Emits SHACL Turtle.
  - Emits OWL-lite/RDFS vocabulary Turtle.
  - Emits proposal evidence describing tables, primary keys, FKs, column types, null rates, and profiles.
  - Emits schema diff report when previous schema is supplied.

Implemented inference helpers:

- `table_name_from_source`
- `infer_node_type`
- `infer_property`
- `null_rates_from_profile`
- `column_types_from_profile`
- `infer_primary_keys`
- `infer_foreign_keys`
- `infer_relationships`
- `relationship_type_from_fk`
- `to_json_type`
- `node_confidence`
- `property_confidence`
- `dedupe_node_types`
- `dedupe_relationship_types`
- `singularize`

Human review states:

- `SchemaProposalWorkflow.approve`
- `SchemaProposalWorkflow.reject`
- `SchemaProposalWorkflow.promote`
- `SchemaProposalWorkflow.diff`

Registry integration:

- `InMemorySchemaRegistry`
- `SchemaRegistryValidator`

Registered implemented schema techniques:

- `in_memory_schema_registry`
- `schema_proposal_workflow`
- `schema_quarantine_namespace`
- `schema_promotion_workflow`

Registered schema stubs:

- `owl_ontology_loader`
- `ollama_ontology_proposal`
- `ollama_entity_label_proposal`
- `ollama_relationship_type_proposal`

Test coverage:

- Sample CSV can produce candidate `Person` node schema.
- `REPORTS_TO` relationship schema is inferred from manager/reporting key patterns.
- SHACL and OWL-lite artifacts are generated.
- Schema registry validation uses generated schema.
- Schema diff reports identify changes across runs.

## 18. Validation

Location:

- `src/kg_construction/infrastructure/ontology/validation.py`
- `src/kg_construction/infrastructure/ontology/shacl.py`

Implemented validators:

| Validator | Purpose |
| --- | --- |
| `EvidenceRequiredValidator` | Ensures entities, relations, and facts have evidence. |
| `SchemaRegistryValidator` | Validates labels, required properties, and relationship domain/range against schema registry. |
| `ConfidenceThresholdValidator` | Rejects/quarantines candidates below a confidence threshold. |
| `CompositeValidator` | Runs multiple validators and merges reports. |
| `ShaclValidator` | Uses pySHACL to validate RDF output against generated or supplied SHACL shapes. |
| `HumanReviewQueue` | Registered as human validation/review pathway. |

Validation objects:

- `ValidationFinding`
- `ValidationReport`
- `ValidationSeverity`
- `ValidationPolicy`
- `ShaclValidationResult`

Policies:

- `reject`
- `quarantine`
- `repair`
- `send_to_human_review`

Registered adapter:

- `hallucination_filter`
  - Uses evidence-required logic as a practical unsupported-extraction guard.

Registered stubs:

- `json_schema_validator`
- `property_graph_validator`
- `schema_drift_validator`

Important metric distinction:

- The code computes `unsupported_extraction_rate`.
- It also maps this to `hallucination_risk_proxy` and legacy `hallucination_rate` fields for scoring compatibility.
- Evidence coverage is not treated as a complete hallucination metric; it is used as a practical risk proxy.

Test coverage:

- Candidates without evidence are rejected.
- Unsupported extraction rate is measured.
- RDF output validates against generated SHACL shapes.

## 19. Entity resolution and linking

Location:

- `src/kg_construction/infrastructure/resolution/exact.py`
- `src/kg_construction/infrastructure/resolution/common.py`
- `src/kg_construction/infrastructure/resolution/entity_resolution.py`
- `src/kg_construction/infrastructure/resolution/linking.py`
- `src/kg_construction/infrastructure/resolution/stubs.py`

Implemented resolvers:

| Resolver | Technique | What it does |
| --- | --- | --- |
| `ExactKeyEntityResolver` | exact deterministic matching | Resolves exact entity keys. |
| `NormalizedStringResolver` | normalized string matching | Case, whitespace, punctuation, Unicode, and suffix normalization. |
| `AliasTableResolver` | alias table matching | Maps configured aliases to canonical names. |
| `RapidFuzzResolver` | fuzzy string matching | Uses RapidFuzz scorers and thresholds for noisy duplicates. |
| `WikidataLinker` | Wikidata external KB linking | Calls/injects Wikidata `wbsearchentities`, scores labels, and attaches external link candidates. |
| `DBpediaLinker` | DBpedia external KB linking | Calls/injects DBpedia Lookup, scores labels, and attaches DBpedia resource links. |
| `OpenRefineReconciler` | OpenRefine/W3C reconciliation | Calls/injects reconciliation endpoint, attaches matches, and can merge entities by reconciliation ID. |
| `EmbeddingResolver` | embedding similarity matching | Uses injected or SentenceTransformers embedding model and cosine similarity to merge semantic duplicates. |
| `CrossEncoderResolver` | cross-encoder pair scoring | Uses injected or SentenceTransformers cross-encoder model to score entity pairs before merge audit. |

Implemented adapter boundaries:

| Adapter | Status |
| --- | --- |
| `SplinkProbabilisticResolver` | Checks optional dependency and requires project-specific Splink settings. |
| `DedupeActiveLearningResolver` | Checks optional dependency or accepts injected backend for active-learning decisions. |

Implemented merge audit:

- `MergeDecision`
- `resolution_key`
- `compact_resolution_key`
- `canonical_entity_id`
- `current_canonical_name`
- `apply_merge_decisions`
- `merge_audit`
- `decisions_from_groups`

Audit fields added:

- `canonical_entity_id`
- `canonical_id`
- `canonical_name`
- `merged_from`
- `resolver`
- `resolution_method`
- `resolution_confidence`
- `review_required`
- `merge_group_size`
- `merge_reason`

Relation handling:

- Resolved relations are rewritten to canonical source/target IDs.
- Original source and target IDs are preserved in relation properties.

Registered resolver stubs:

- `blocking_resolver`
- `graph_neighborhood_resolver`
- `human_review_resolver`

Experiment evidence:

- Current dataset report includes adapter checks:
  - `normalized_string_resolver`: ok, 3 audit clusters.
  - `alias_table_resolver`: ok, 2 audit clusters.
  - `rapidfuzz_resolver`: ok, 3 audit clusters.
  - `splink_probabilistic_resolver`: requires optional configuration.
  - `dedupe_active_learning_resolver`: requires optional configuration.

Test coverage:

- Recommendation API / recommendation api / aliases / noisy duplicates resolve with confidence and audit trail.
- Wikidata linker attaches external candidates and audit rows through injected transport.
- DBpedia linker attaches Lookup candidates and audit rows through injected transport.
- OpenRefine reconciler attaches external matches and merges matching IDs with merge audit.
- Embedding resolver merges semantic duplicates with injected model vectors.
- Cross-encoder resolver merges high-scoring pairs with injected pair scorer.
- Splink and Dedupe adapters explain missing optional configuration.

## 20. Graph builders

Location:

- `src/kg_construction/infrastructure/graph_builders/property_graph.py`
- `src/kg_construction/infrastructure/graph_builders/rdf_graph.py`
- `src/kg_construction/infrastructure/graph_builders/evidence_graph.py`

### PropertyGraphBuilder

Implemented behavior:

- Builds GraphRAG-ready property graphs from validated candidates.
- Skips invalid IDs from validation report.
- Uses canonical entity IDs when entity resolution has run.
- Merges duplicate canonical entity nodes.
- Adds candidate IDs, names, confidence, extractor, technique, schema version, created_at.
- Embeds evidence IDs, source IDs, source names, evidence texts, locations, and source paths.
- Builds relationship edges.
- Optionally creates fact nodes for relationship facts.
- Adds metadata:
  - builder
  - validated
  - finding count
  - node count
  - edge count
  - graph model
  - `graphrag_ready = true`

### RDFGraphBuilder

Implemented behavior:

- Builds RDF triples and named-graph quads.
- Uses RDF, RDFS, PROV, XSD namespaces.
- Creates class/type triples.
- Creates labels.
- Converts entity properties to RDF literals with datatype inference.
- Links graph resources to evidence with `prov:wasDerivedFrom`.
- Emits evidence quads into an evidence named graph.
- Emits data quads into a data named graph.
- Deduplicates triples and quads.
- Produces RDFLib-serializable graph and dataset.

Named graph context:

- Data graph default: `urn:kg:graph:data`
- Evidence graph default: `urn:kg:graph:evidence`

### EvidenceGraphBuilder

Implemented behavior:

- Builds an audit graph connecting:
  - sources
  - evidence nodes
  - entity candidates
  - relation candidates
  - facts
- Creates:
  - `HAS_EVIDENCE` edges from source to evidence.
  - `EVIDENCED_BY` edges from candidate/fact to evidence.

Test coverage:

- Property graph, RDF graph, and evidence graph include provenance.
- RDF output validates against generated SHACL shapes.

## 21. Graph store and export layer

Location:

- `src/kg_construction/infrastructure/graph_stores/exporters.py`
- `src/kg_construction/infrastructure/graph_stores/ingestion.py`

Implemented exporters:

| Exporter | Output |
| --- | --- |
| `Neo4jExporter` | Idempotent Cypher `MERGE` statements for property graphs; RDF as Turtle/N-Quads lines. |
| `MemgraphExporter` | Memgraph-compatible Cypher via Neo4j exporter behavior. |
| `KuzuExporter` | Kuzu ingestion manifest with nodes and relationships. |
| `NetworkXExporter` | NetworkX-compatible node-link dictionary. |
| `RDFExporter` | Turtle and N-Quads strings. |
| `RDFLibExporter` | RDFLib `Graph` and RDFLib `Dataset`. |

Implemented ingestion adapters:

- `Neo4jIngestionAdapter`
  - Uses Neo4j managed write transactions through `session.execute_write(...)`.
  - Groups nodes by label set and relationships by type.
  - Uses parameterized `UNWIND $rows AS row` Cypher batches.
  - Optionally creates per-label `id` uniqueness constraints.
  - Integrates `BatchRetryPolicy`.
  - Writes exhausted failures to `DeadLetterQueue` when configured.
- `KuzuIngestionAdapter`
  - Uses a generic Kuzu schema:
    - `KGNode(id STRING PRIMARY KEY, labels STRING, properties STRING)`
    - `KGEdge(FROM KGNode TO KGNode, id STRING, type STRING, properties STRING)`
  - Creates schema when configured.
  - Uses parameterized `MERGE` statements.
  - Wraps writes in `BEGIN TRANSACTION` / `COMMIT` / `ROLLBACK`.
  - Stores labels and properties as JSON strings to preserve arbitrary property graph payloads.
  - Integrates retry and dead-letter behavior.

Ingestion support objects:

- `Neo4jIngestionConfig`
- `KuzuIngestionConfig`
- `IngestionWriteReport`

Important behavior:

- Live database dependencies are optional.
- Tests use injectable fake drivers/connections, so CI does not require Neo4j or Kuzu.
- RDF ingestion into Neo4j/Kuzu remains explicit and guarded; the current implemented write path is property graph ingestion.

## 22. Evaluation and scoring

Location:

- `src/kg_construction/application/benchmark_runner.py`
- `src/kg_construction/application/scoring.py`
- `src/kg_construction/application/recommendation_engine.py`
- `src/kg_construction/infrastructure/evaluation/`
- `src/kg_construction/infrastructure/reports/`

### TechniqueEvaluationRunner

Implemented behavior:

- Loads input sources from files or directories.
- Profiles dataset.
- Computes technique applicability.
- Selects candidate techniques.
- Builds candidate workflows.
- Executes candidates.
- Measures runtime or applies deterministic runtime settings.
- Computes evaluation metrics.
- Scores candidates.
- Applies deterministic tie-breaking.
- Produces recommendations.
- Saves candidate graph outputs when requested.
- Stores reproducibility metadata.

Benchmark metadata includes:

- run ID
- timestamp
- config hash
- git commit
- input hash
- source type
- selected technique mode
- deterministic runtime flag
- tie-breaking policy
- model settings

Tie-breaking policy:

```text
final_score_desc
production_thresholds_pass_desc
latency_asc
maintainability_desc
technique_name_asc
```

### ScoringEngine

Implemented score dimensions:

- accuracy
- validation
- coverage
- entity resolution
- retrieval
- scalability
- cost efficiency
- maintainability

Implemented penalties:

- unsupported extraction
- schema violation
- latency
- human review

Final score is split into:

- `quality_score`
- `runtime_score`
- `final_score`

This separation prevents runtime noise from silently moving quality comparisons when deterministic mode is enabled.

### RecommendationEngine

Recommendation slots:

- best single technique
- best hybrid technique
- production-safe technique
- high-recall technique
- low-cost technique
- GraphRAG-ready technique
- fallback recommendation
- limitations

Production-safe thresholds:

- evidence coverage
- schema validity
- unsupported extraction rate
- validation failure rate

### Evaluation metric modules

| Module | Metrics |
| --- | --- |
| `accuracy_metrics.py` | entity precision/recall/F1, relation precision/recall/F1, triple exact/relaxed match, span F1. |
| `coverage_metrics.py` | entity, relation, source, and attribute coverage. |
| `validation_metrics.py` | schema validity, RDF/SHACL validity, evidence coverage, provenance coverage, duplicate edge rate, orphan node rate, validation failure, unsupported extraction. |
| `entity_resolution_metrics.py` | merge-pair based entity resolution metrics. |
| `retrieval_metrics.py` | retrieval and GraphRAG-readiness metrics. |
| `operational_metrics.py` | latency, throughput, memory, graph write time, retry rate, token/cost proxy, scalability/cost scores. |
| `statistical_confidence.py` | Wilson score interval and bootstrap mean interval. |
| `metrics.py` | generic precision/recall/F1 and construction metrics evaluator. |
| `benchmark.py` | dependency-light benchmark runner. |

### Report writers

Location:

- `src/kg_construction/infrastructure/reports/json_report.py`
- `src/kg_construction/infrastructure/reports/markdown_report.py`
- `src/kg_construction/infrastructure/reports/comparison_matrix.py`

Implemented report outputs:

- JSON report writer.
- Markdown report writer.
- Comparison matrix builder.

Test coverage:

- Weighted scoring penalizes unsupported extraction and latency.
- Wilson interval is bounded.
- Runner compares three or more text techniques.
- Deterministic benchmark mode repeats rankings and scores.
- Golden benchmark rankings cover text, CSV, JSON, and mixed input.
- Markdown report matches ranked JSON order and score fields.
- Benchmark CLI writes JSON and Markdown reports.

## 23. Production hardening

Location:

- `src/kg_construction/infrastructure/production/`
- `src/kg_construction/application/production.py`

Implemented components:

### Structured logging

- `StructuredLoggingConfig`
- `JsonLogFormatter`
- `StructuredLogger`
- `configure_structured_logging`

Behavior:

- JSON log lines.
- Bound context.
- Optional PII redaction.
- Exception details.

### Run manifests

- `RunManifest`
- `RunManifestWriter`
- `canonical_hash`
- `git_commit`

Behavior:

- Captures run status.
- Captures started/completed timestamps.
- Captures config hash.
- Captures input hash.
- Captures git commit when available.
- Captures source IDs/names.
- Captures entity/relation/fact/validation counts.
- Captures failures and error stages.

### Incremental sync and watermarking

- `WatermarkState`
- `IncrementalSyncResult`
- `WatermarkStore`
- `IncrementalSyncPolicy`

Behavior:

- JSON-backed high-watermark store.
- Filters sources by metadata watermark field.
- Falls back to source ID when needed.
- Commits processed watermarks.

### Error quarantine

- `QuarantineRecord`
- `ErrorQuarantine`

Behavior:

- Append-only JSONL quarantine store.
- Stores component exceptions.
- Stores validation findings.
- Links quarantine records to source/candidate when possible.

### Batch retry policy

- `RetryPolicyConfig`
- `RetryOutcome`
- `RetryExhaustedError`
- `BatchRetryPolicy`

Behavior:

- Retries idempotent batch operations.
- Uses bounded exponential backoff.
- Retries by exception type and retryable message fragments.
- Can disable sleep for deterministic tests.
- Returns attempt and retry counts on success.
- Raises `RetryExhaustedError` with original error and attempt count on failure.

### Dead-letter queue and replay

- `DeadLetterRecord`
- `DeadLetterQueue`
- `DeadLetterReplayReport`
- `DeadLetterReplayTool`

Behavior:

- Persists failed graph write batches as JSONL.
- Stores graph payload, store name, operation name, graph model, error type, message, and attempt count.
- Supports status updates: `pending`, `replayed`, `failed`, `skipped`.
- Rehydrates `PropertyGraph` / `RDFGraph` payloads during replay.
- Dry-run replay inspects pending records without mutating status.

### Human review queue

- `HumanReviewItem`
- `HumanReviewQueue`

Behavior:

- File-backed review queue.
- Enqueues validation findings.
- Supports pending list.
- Supports approve/reject decisions with reviewer and note.

### PII redaction

- `PIIRedactionPolicy`

Behavior:

- Redacts emails.
- Redacts phone numbers.
- Redacts SSNs.
- Redacts IP addresses.
- Redacts API keys/tokens/secrets/passwords.
- Recursively redacts mappings/lists/tuples.
- Redacts `SourceRecord` content, records, and metadata.

Registered production techniques:

- `structured_logging`
- `run_manifest_writer`
- `batch_retry_policy`
- `dead_letter_queue`
- `dead_letter_replay_tool`
- `incremental_sync_policy`
- `watermark_store`
- `error_quarantine`
- `pii_redaction_policy`

Test coverage:

- PII redaction covers source records, metadata, and structured logs.
- Watermark store filters and commits incremental sources.
- Quarantine and human review queue persist validation findings.
- Workflow writes manifest and quarantines component errors.
- Production services are config-driven.
- Config validation catches bad values and unknown techniques.
- Registry includes production hardening capabilities.
- Graph ingestion retry, dead-letter, and replay are covered with fake database writers.

## 24. Graph enrichment

Location:

- `src/kg_construction/infrastructure/enrichment/graph_ml.py`
- `src/kg_construction/infrastructure/enrichment/stubs.py`
- `src/kg_construction/application/graph_ml.py`

Implemented:

- `GraphAnalyticsEnricher`
  - Adds `degree`, `in_degree`, and `out_degree` features to property graph nodes.
- `PageRankEnricher`
  - Runs directed PageRank through NetworkX when available.
  - Falls back to an internal power-iteration implementation.
  - Adds `pagerank` to each node and stores top-node metadata.
- `CommunityDetectionEnricher`
  - Runs greedy modularity community detection on an undirected graph projection when NetworkX is available.
  - Falls back to weak components when NetworkX or graph edges are unavailable.
  - Adds `community_id`, `community_size`, and `community_method`.
- `ConnectedComponentsEnricher`
  - Adds weak `component_id` and `component_size`.
- `NodeEmbeddingEnricher`
  - Supports `spectral` embeddings through normalized adjacency eigenvectors when NumPy is available.
  - Supports deterministic `structural_hash` embeddings without ML dependencies.
  - Supports optional `graphsage` embedding through `PyTorchGeometricGraphSAGEAdapter`.
  - Adds `node_embedding` vectors to nodes.
- `LinkPredictionEnricher`
  - Scores missing node pairs with resource allocation, Jaccard, common neighbors, preferential attachment, and optional embedding cosine.
  - Supports a hybrid topology-plus-embedding score.
  - Stores predictions in graph metadata and can optionally materialize `PREDICTED_LINK` edges.
- `GraphMLWorkflow`
  - Runs degree features, PageRank, communities, components, embeddings, and link prediction in one stable notebook/script call.
- `graph_ml_workflow_config_from_dict`
  - Converts YAML/JSON config dictionaries into `GraphMLWorkflowConfig`.
- `PyTorchGeometricGraphSAGEAdapter`
  - Optional adapter for GraphSAGE message passing embeddings when `torch-geometric` is installed.

Registered graph ML adapters:

- `graphsage_pyg_adapter`

Registered enrichment stubs that remain:

- `ChunkEmbeddingEnricher`
- `EntityEmbeddingEnricher`
- `ExternalKGLinkEnricher`
- `EntitySummaryEnricher`
- `CommunitySummaryEnricher`
- `TemporalEventEnricher`

Purpose:

- Adds graph topology and ML-ready signals after graph construction.
- Supports retrieval ranking, GraphRAG community context, entity prioritization, missing-link discovery, and downstream ML feature handoff.
- Keeps heavy GNN dependencies optional while providing deterministic graph ML baselines in the core repo.

## 25. CLI interfaces

Location:

- `src/kg_construction/interfaces/cli/construct.py`
- `src/kg_construction/interfaces/cli/benchmark.py`
- `src/kg_construction/interfaces/cli/replay_dead_letters.py`
- `src/kg_construction/interfaces/cli/doctor.py`

Console scripts from `pyproject.toml`:

- `kg-construction`
- `kg-construction-benchmark`
- `kg-construction-replay-deadletters`

Implemented construction CLI:

- Input path.
- Source type.
- Property graph output.
- RDF JSON output.
- Evidence graph output.
- Neo4j Cypher output.
- Loads CSV, JSON, text, and supported enterprise-like source types through the runner path.

Implemented benchmark CLI:

- Input path.
- Source type.
- Config path.
- Technique mode.
- Markdown output.
- JSON output.
- Candidate output directory.

Implemented dead-letter replay CLI:

- Dead-letter JSONL path.
- Target store name.
- Neo4j connection settings.
- Kuzu database path.
- Limit.
- Dry-run mode.
- JSON replay report output.

Implemented doctor CLI:

- `kg-construction doctor` subcommand through the main construction console script.
- Python version and package distribution metadata check.
- Core dependency import check.
- Inferred optional extras and optional import checks.
- Ollama, Neo4j, and Kuzu availability checks.
- RDFLib/pySHACL smoke validation.
- Notebook dependency check.
- YAML config validity check.
- JSON output mode for automated diagnostics.
- Service-skip mode for CI and offline environments.

Makefile targets:

- `make compile`
- `make test`
- `make smoke`
- `make quality`
- `make verify`
- `make package-check`
- `make benchmark-leaderboard`
- `make clean`

CI packaging checks:

- Existing matrix still runs editable installs with `make quality` and `make verify`.
- Package job runs `python -m build`.
- Package job runs `python -m twine check dist/*`.
- Wheel-install job downloads the built wheel artifact.
- Wheel-install job installs from the wheel without editable mode or `PYTHONPATH=src`.
- Wheel-install job runs console-script smoke checks and focused import tests.

## 26. Notebooks

Location:

- `notebooks/`

Current notebooks:

| Notebook | Purpose |
| --- | --- |
| `01_data_loading_and_profiling.ipynb` | Calls loaders and profiler. |
| `02_structured_schema_mapping_rdf.ipynb` | Calls structured schema proposal, mapping, property graph, RDF, and SHACL paths. |
| `03_jsonpath_xml_semistructured_mapping.ipynb` | Calls JSONPath/dotted-path and XML/semi-structured mapping capabilities. |
| `04_text_extraction_gliner_ollama_labels.ipynb` | Calls regex, gazetteer, Ollama label proposal, and GLiNER extraction workflow. |
| `05_evaluation_resolution_production_hardening.ipynb` | Calls evaluation, entity resolution, structured logging, watermarks, manifests, quarantine, human review, and PII redaction. |
| `06_graph_ml_capabilities.ipynb` | Calls graph analytics, PageRank, community detection, node embeddings, and link prediction. |
| `07_standards_mapping_and_virtual_kg.ipynb` | Calls R2RML, RML, CSVW, JSON-LD, and OBDA/virtual graph mapping examples. |
| `08_relation_extraction_and_entity_linking.ipynb` | Calls OpenIE, dependency-pattern relation extraction, GLiNER plus Ollama relation typing, REBEL adapters, external linkers, semantic resolvers, and Splink/Dedupe adapter boundaries. |
| `09_enterprise_events_and_observability.ipynb` | Calls HTML/web/log/event/scientific/JSON-LD loaders plus HTML metadata, log event, event graph, trace/span, and causality mapping. |
| `10_graph_store_ingestion_and_exports.ipynb` | Calls graph store exporters, offline Neo4j/Kuzu ingestion, neosemantics RDF import path, Kuzu bulk import, retries, and dead-letter replay. |
| `11_gold_benchmarks_and_retrieval.ipynb` | Calls gold benchmark evaluation, graph splits, node2vec, KG embeddings, GraphSAGE prep, large-graph backends, retrieval comparison, and benchmark fixtures. |

Notebook coverage guard:

- `notebooks/capability_coverage.yaml` maps implemented and adapter techniques to notebook examples.
- `scripts/check_notebook_coverage.py` checks the coverage map against `default_registry()`.
- `make notebook-check` runs notebook hygiene plus registry coverage validation.

Notebook inventory from current files:

- `01_data_loading_and_profiling.ipynb`: 3 code cells.
- `02_structured_schema_mapping_rdf.ipynb`: 4 code cells.
- `03_jsonpath_xml_semistructured_mapping.ipynb`: 5 code cells.
- `04_text_extraction_gliner_ollama_labels.ipynb`: 3 code cells.
- `05_evaluation_resolution_production_hardening.ipynb`: 5 code cells.
- `06_graph_ml_capabilities.ipynb`: 5 code cells.
- `07_standards_mapping_and_virtual_kg.ipynb`: 5 code cells.
- `08_relation_extraction_and_entity_linking.ipynb`: 5 code cells.
- `09_enterprise_events_and_observability.ipynb`: 5 code cells.
- `10_graph_store_ingestion_and_exports.ipynb`: 5 code cells.
- `11_gold_benchmarks_and_retrieval.ipynb`: 5 code cells.

## 27. Dataset download and experiment scripts

Location:

- `scripts/download_experiment_datasets.py`
- `scripts/run_experiment_datasets.py`

### Dataset downloader

Implemented dataset download/sample functions:

- Chinook SQLite sample database.
- W3C RDB2RDF D001 fixture.
- NYC 311 service requests bounded CSV sample.
- SEC EDGAR Apple companyfacts JSON.
- PubMed baseline XML bounded sample.
- DBLP publication API JSON sample.
- Loghub HDFS raw logs and structured CSV.
- WikiANN English NER sample.
- Enron-derived bounded email/text sample.

Outputs:

- Downloaded files under ignored local workspace `data/external/experiment_datasets/`.
- Manifest under `reports/experiment_datasets/` when the downloader is run.

### Dataset experiment runner

Implemented behavior:

- Loads each dataset using repo loaders.
- Runs applicable construction techniques by dataset type.
- Writes property graph, RDF JSON, Turtle, N-Quads, and evidence graph artifacts.
- Generates runtime schema from batches.
- Generates SHACL shapes.
- Runs pySHACL validation.
- Proposes structured schemas where possible.
- Runs deterministic benchmark summaries.
- Runs offline Ollama graph adapter flow.
- Runs GLiNER-label pipeline with offline/static provider and model protocol.
- Runs entity resolution adapter checks.
- Writes dataset technique matrix CSV.
- Writes Markdown experiment report.
- Writes machine-readable experiment JSON.

Key outputs:

- `reports/experiment_datasets/dataset_experiment_report.md`
- `reports/experiment_datasets/dataset_experiment_results.json`
- `reports/experiment_datasets/dataset_technique_matrix.csv`
- `reports/experiment_datasets/graphs/`
- `reports/experiment_datasets/schemas/`
- `reports/experiment_datasets/benchmarks/`

These outputs are intentionally treated as generated run artifacts. They are useful for experiments, but they can become large and stale, so the cleaned repo keeps only current registry-level reports by default.

## 28. External dataset policy

Generated location:

- `data/external/experiment_datasets/`

Current checked-in state:

- Only `data/external/README.md` is kept in the repository.
- Downloaded external datasets are ignored by `.gitignore`.
- Tiny synthetic fixtures remain under `examples/` and `tests/fixtures/`.
- Public dataset samples can be recreated with `scripts/download_experiment_datasets.py`.
- Full experiment outputs can be recreated with `scripts/run_experiment_datasets.py`.

Reason:

- Avoid committing large generated artifacts.
- Avoid redistributing third-party datasets without explicitly documenting each license.
- Avoid accidentally publishing raw data, PII, secrets, or proprietary text.

## 29. Current reports and generated run artifacts

Current checked-in report files:

- `reports/README.md`
- `reports/benchmark_leaderboard.md`
- `reports/benchmark_leaderboard.csv`
- `reports/gold_standard_benchmark_story.md`
- `reports/gold_standard_benchmark_story.csv`
- `reports/gold_standard_benchmark_story.json`
- `reports/current_repo_capability_summary.md`
- `reports/current_registry_summary.json`
- `reports/current_registry_matrix.csv`

Current generated workspace policy:

- `data/processed/` is for local smoke-test graph outputs, candidate graphs, benchmark reports, dead letters, and temporary graph-store paths.
- `reports/experiment_datasets/` is for high-volume dataset experiment outputs such as property graphs, RDF JSON, Turtle, N-Quads, evidence graphs, SHACL artifacts, and benchmark summaries.
- Legacy article-specific artifacts were removed because they did not represent the current implementation standards.
- Old diagram PNGs were removed from `docs/` because they did not represent the current repo standards.

Current registry proof is generated from `kg_construction.application.registry.default_registry()`:

- Total registered techniques: 150.
- Implemented techniques: 104.
- Adapter boundaries: 17.
- Planned stubs: 29.
- Public Stable techniques: 76.
- Public Optional adapter techniques: 29.
- Public Experimental techniques: 16.
- Public Planned techniques: 29.

## 30. Tests

Location:

- `tests/test_data_to_kg_construction.py`
- `tests/test_enterprise_loaders.py`
- `tests/test_public_api.py`
- `tests/test_quickstart.py`
- `tests/test_doctor_cli.py`
- `tests/test_benchmark_leaderboard.py`
- `tests/test_registry_support_levels.py`
- `tests/test_production_hardening.py`
- `tests/test_relation_extraction.py`
- `tests/test_technique_evaluation.py`

Current test inventory:

### Data-to-KG construction tests

- Deterministic source and evidence IDs.
- CSV direct mapping builds entities, relations, and evidence.
- Schema proposal workflow generates schema, shapes, and registry validation.
- Schema proposal workflow reports schema diffs across runs.
- JSONPath mapping preserves source paths and array edges.
- JSONPath repeated array objects stay distinct.
- R2RML Turtle mapping generates evidence-backed RDF facts.
- RML mapping supports CSV, JSON, and XML logical sources.
- CSVW mapping uses primary keys, column metadata, and foreign keys.
- Regex extractor preserves span evidence.
- Validation rejects candidates without evidence.
- Validation metrics report unsupported extraction rate.
- Entity resolution handles normalized aliases and noisy duplicates with audit.
- Splink and Dedupe adapters explain required configuration.
- Property/RDF/evidence graphs include provenance.
- RDF output validates against generated SHACL shapes.
- Ollama provider retries invalid JSON and extractor returns candidates.
- Ollama label proposal plus GLiNER extracts evidence spans.
- Registry covers required categories and techniques.
- Planner prefers deterministic before LLM and human review.
- Complete example config loads.
- Construction CLI builds sample text graph.

### Public API tests

- Top-level exports include the stable framework facade.
- `load_sources` and `profile_dataset` work from CSV inputs.
- `construct_kg` works from a CSV path.
- `construct_kg` works from raw text.
- `construct_kg` accepts preloaded `SourceRecord` objects.
- `evaluate_techniques` returns ranked benchmark results from a path.

### Quickstart tests

- Executes `examples/quickstart_60_second.py`.
- Verifies CSV, text, and JSON construction produce graph output.
- Verifies benchmark ranking returns an expected lightweight technique.
- Verifies RDF, Neo4j Cypher, and NetworkX node-link exports are populated.

### Doctor CLI tests

- Verifies doctor reports core install, RDF/SHACL, config validity, and skipped services.
- Verifies `kg-construction doctor --json` routing through the main console entrypoint.
- Verifies invalid extra config files are reported as doctor failures.

### Benchmark leaderboard tests

- Verifies the committed CSV contains required data-science decision columns.
- Verifies the leaderboard covers text, CSV, JSON, and mixed fixture types.
- Verifies regenerated leaderboard rows match the committed CSV for core reproducibility fields.
- Verifies the Markdown report stays small and documents generated-output policy.

### Registry support-level tests

- Verifies every registered technique exposes a public support label.
- Verifies Stable, Optional adapter, Experimental, and Planned levels all exist.
- Verifies internal `implemented`, `adapter`, and `stub` statuses map to public-facing maturity labels.
- Verifies service-backed and research-heavy techniques are not mislabeled as core-stable.

### Enterprise loader tests

- Excel loader.
- SQL loader.
- API response loader.
- XML loader.
- Markdown loader.
- Report loader.
- PDF loader.
- Email loader.
- Chat transcript loader.
- Profiler/registry coverage for enterprise loaders.

### Production hardening tests

- PII redaction.
- Watermark filtering and commit.
- Quarantine and human review persistence.
- Workflow manifest writing and component error quarantine.
- Production services from config.
- Config validation for bad values and unknown techniques.
- Registry production coverage.

### Relation extraction tests

- OpenIE relation extraction emits evidence-backed triples.
- Dependency-pattern relation extraction emits configured relation types.
- GLiNER entity spans plus Ollama relation typing emits evidence-backed relation candidates.
- REBEL adapter parses generated transformer output into relation candidates.
- Registry marks relation extraction capabilities as implemented or adapter-backed.

### Technique evaluation tests

- Profiler structured key detection and scoring.
- Applicability direct mapping for CSV.
- Weighted scoring penalties.
- Wilson interval bounds.
- Runner compares three or more text techniques.
- Deterministic benchmark repeatability.
- Golden benchmark rankings for text, CSV, JSON, mixed input.
- Markdown/JSON report order consistency.
- Benchmark CLI output.

Golden benchmark fixtures:

- `tests/fixtures/golden_benchmarks/text/input.txt`
- `tests/fixtures/golden_benchmarks/csv/input.csv`
- `tests/fixtures/golden_benchmarks/json/input.json`
- `tests/fixtures/golden_benchmarks/mixed/input.csv`
- `tests/fixtures/golden_benchmarks/mixed/input.json`
- `tests/fixtures/golden_benchmarks/mixed/input.txt`
- `tests/fixtures/golden_benchmarks/expected_rankings.json`

## 31. Current registry by category

This section lists every registered technique so future work can see exactly what exists as implemented, adapter, or stub.

### Loaders

Implemented:

- `csv_loader`
- `excel_loader`
- `sql_database_loader`
- `structured_api_loader`
- `json_loader`
- `xml_loader`
- `html_loader`
- `webpage_loader`
- `log_loader`
- `api_response_loader`
- `text_loader`
- `markdown_loader`
- `pdf_loader`
- `report_loader`
- `email_loader`
- `chat_transcript_loader`
- `scientific_paper_loader`

Stubs:

- `data_warehouse_export_loader`
- `jsonld_loader`
- `event_payload_loader`

### Mappers

Implemented:

- `direct_table_mapper`
- `table_to_node_mapping`
- `row_to_entity_mapping`
- `primary_key_to_id_mapping`
- `foreign_key_to_edge_mapping`
- `column_to_property_mapping`
- `schema_profiling`
- `column_type_inference`
- `jsonpath_mapper`
- `dotted_path_json_mapper`
- `r2rml_mapper`
- `rml_mapper`
- `csvw_mapper`
- `log_event_mapper`
- `html_metadata_mapper`

Adapters:

- `relationship_inference_from_keys`

Stubs:

- `incremental_watermark_sync`
- `jsonld_mapper`
- `obda_virtual_graph_mapper`
- `xmlpath_mapper`
- `api_response_mapper`
- `event_graph_mapper`

### Extractors

Implemented:

- `regex_entity_extractor`
- `gazetteer_entity_extractor`
- `dependency_pattern_relation_extractor`
- `gliner_entity_extractor`
- `openie_relation_extractor`
- `ollama_graph_extractor`
- `gliner_ollama_relation_typing_extractor`

Adapters:

- `rebel_relation_extractor`
- `transformer_relation_extractor`
- `ollama_schema_guided_extraction`
- `ollama_schema_later_extraction`

Stubs:

- `spacy_entity_ruler_extractor`
- `sentence_splitter_adapter`
- `pos_tagging_adapter`
- `coreference_adapter`
- `spacy_ner_extractor`
- `transformer_ner_extractor`
- `pairwise_relation_classifier_adapter`
- `joint_entity_relation_extractor`
- `supervised_relation_extractor`

### Schema

Implemented:

- `in_memory_schema_registry`
- `schema_proposal_workflow`
- `schema_quarantine_namespace`
- `schema_promotion_workflow`

Stubs:

- `owl_ontology_loader`
- `ollama_ontology_proposal`
- `ollama_entity_label_proposal`
- `ollama_relationship_type_proposal`

### Resolution

Implemented:

- `exact_key_entity_resolver`
- `normalized_string_resolver`
- `fuzzy_string_resolver`
- `rapidfuzz_resolver`
- `alias_table_resolver`
- `embedding_resolver`
- `cross_encoder_resolver`
- `wikidata_linker`
- `dbpedia_linker`
- `openrefine_reconciler`

Adapters:

- `probabilistic_linkage_resolver`
- `splink_probabilistic_resolver`
- `dedupe_active_learning_resolver`

Stubs:

- `blocking_resolver`
- `graph_neighborhood_resolver`
- `human_review_resolver`

### Validation

Implemented:

- `evidence_required_validator`
- `schema_registry_validator`
- `confidence_threshold_validator`
- `shacl_validator`
- `human_review_queue`

Adapter:

- `hallucination_filter`

Stubs:

- `json_schema_validator`
- `property_graph_validator`
- `schema_drift_validator`

### Graph builders

Implemented:

- `property_graph_builder`
- `rdf_graph_builder`
- `evidence_graph_builder`

### Graph stores/exporters

Implemented:

- `neo4j_exporter`
- `neo4j_ingestion_adapter`
- `memgraph_exporter`
- `kuzu_exporter`
- `kuzu_ingestion_adapter`
- `networkx_exporter`
- `rdf_exporter`

Adapter:

- `rdflib_exporter`

Stub:

- `neo4j_ingestion_adapter`

### LLM

Implemented:

- `ollama_provider`

### Enrichment

Implemented:

- `graph_analytics_enricher`
- `pagerank_enricher`
- `community_detection_enricher`
- `connected_components_enricher`
- `node_embedding_enricher`
- `spectral_node_embedding_enricher`
- `structural_hash_node_embedding_enricher`
- `link_prediction_enricher`
- `resource_allocation_link_prediction`
- `jaccard_link_prediction`
- `hybrid_topology_embedding_link_prediction`

Adapters:

- `graphsage_pyg_adapter`

Stubs:

- `chunk_embedding_enricher`
- `entity_embedding_enricher`
- `external_kg_link_enricher`
- `entity_summary_enricher`
- `community_summary_enricher`
- `temporal_event_enricher`

### Evaluation

Implemented:

- `construction_metrics_evaluator`
- `benchmark_runner`

### Hybrid

Implemented:

- `hybrid_rules_gazetteer_regex`

Adapters:

- `hybrid_rules_ml_ollama`
- `hybrid_structured_then_text`
- `hybrid_gliner_then_ollama_relations`
- `hybrid_ollama_schema_then_gliner`
- `hybrid_openie_then_ollama_canonicalization`
- `hybrid_deterministic_then_llm_enrichment`
- `hybrid_human_reviewed_schema`

### Production

Implemented:

- `structured_logging`
- `run_manifest_writer`
- `incremental_sync_policy`
- `watermark_store`
- `error_quarantine`
- `pii_redaction_policy`

## 32. Current docs

Current `docs/` content:

- `docs/README.md`
- `docs/setup_running_configuration_guide.md`
- `docs/repo_implementation_inventory.md`
- `docs/next_stage_capabilities.md`
- `docs/graph_ml_capabilities.md`
- `docs/production_ingestion.md`
- `docs/packaging_and_notebook_policy.md`
- `docs/github_open_source_readiness_plan.md`

## 33. Current implementation proof points

Code proof:

- 104 registered implemented techniques.
- 17 registered adapter boundaries.
- 29 registered stubs.
- 76 public Stable techniques.
- 29 public Optional adapter techniques.
- 16 public Experimental techniques.
- 29 public Planned techniques.
- Strict Pydantic models for domain, schema, config, metrics, manifests, review, and quarantine.
- 107 unit tests cover loaders, mapping, extraction, relation extraction, validation, RDF/SHACL, entity linking/resolution, graph ML, production ingestion, CLI, doctor diagnostics, benchmark leaderboard reproducibility, gold-standard benchmark story reproducibility, registry support levels, and evaluation.

Runtime/report proof:

- `reports/current_repo_capability_summary.md` summarizes current capabilities from the live registry.
- `reports/current_registry_summary.json` stores machine-readable registry counts, support-level counts, and technique metadata.
- `reports/current_registry_matrix.csv` stores the registry in tabular form with both internal status and public support level.
- `reports/benchmark_leaderboard.md` and `.csv` provide a small deterministic fixture benchmark for technique selection.
- `reports/gold_standard_benchmark_story.md`, `.csv`, and `.json` provide a small public correctness story for NER spans, relation extraction, triples, entity resolution, and GraphRAG/retrieval.
- `data/processed/README.md` documents the generated smoke-output workspace.
- `reports/README.md` documents the report artifact policy.

## 34. What is implemented vs what is not yet complete

The list below includes Stable, Optional adapter, and Experimental capabilities. Use the support-level tables above to decide what is core-ready, what needs extra setup, and what should be treated as research/prototype surface.

Implemented and usable now:

- Source loading for CSV, JSON, text, Excel, SQL, API responses, XML, HTML, web pages, logs, Markdown, reports, scientific papers, PDF, email, chat.
- Direct table-to-graph mapping.
- Dotted JSON flattening.
- Real JSONPath selection with source-path evidence.
- JSON object-array edges.
- HTML metadata mapping into document, meta tag, link, and JSON-LD graph candidates.
- Event payload loading and event graph mapping with trace/span correlation and causality.
- Log template mining and log event mapping into event, system/logger/client/endpoint/level/template graph candidates.
- JSON-LD loading, expansion, and RDF dataset fact generation.
- OBDA virtual KG rewriting into SQL plans.
- R2RML relational row mapping from Turtle triples maps.
- RML CSV/JSON/XML logical-source mapping from Turtle triples maps.
- R2RML/RML referencing object maps with cross-source joins.
- CSVW dialect parsing, validation, tabular metadata mapping with primary keys, column metadata, and foreign keys.
- Regex entity extraction.
- Gazetteer entity extraction.
- GLiNER extraction through optional real model or injected model protocol.
- OpenIE relation extraction with injected backend support and deterministic fallback.
- Dependency-pattern relation extraction with injected backend support and deterministic fallback.
- Ollama graph extraction with evidence-gated candidate generation.
- Ollama label proposal for GLiNER labels.
- Hybrid Ollama-label-plus-GLiNER notebook-friendly pipeline.
- GLiNER entity spans plus Ollama relation typing.
- REBEL/transformer relation extraction adapter.
- Structured schema proposal from tables.
- SHACL generation.
- OWL-lite/RDFS vocabulary generation.
- Schema lifecycle and diffing.
- Schema registry validation.
- Evidence validation.
- Confidence threshold validation.
- pySHACL validation.
- Exact, normalized, alias, and RapidFuzz entity resolution.
- Wikidata, DBpedia, and OpenRefine external entity linking/reconciliation.
- Embedding and cross-encoder semantic entity resolution.
- Splink/Dedupe adapter boundaries.
- Property graph builder.
- RDF graph builder with triples/quads/named graph context.
- Evidence graph builder.
- Neo4j/Memgraph/Kuzu/NetworkX/RDF/RDFLib export surfaces.
- Neo4j and Kuzu transactional ingestion.
- Neo4j RDF-to-graph import through neosemantics when explicitly enabled.
- Kuzu bulk CSV import path for large graphs.
- Graph ML train/validation splits, node2vec, KG embeddings, large-graph backends, and retrieval benchmarks.
- Gold benchmark evaluator and gold fixtures for NER spans, triples, relation extraction, entity resolution clusters, and GraphRAG/retrieval cases.
- Dataset profiling.
- Applicability scoring.
- Deterministic benchmark mode.
- Weighted scoring and ranking.
- Tie-breaking policy.
- Recommendation slots.
- Runtime/cost/provenance/coverage/validation/entity-resolution/retrieval metrics.
- JSON/Markdown report generation.
- Structured logging.
- Run manifests.
- Incremental sync/watermarks.
- Error quarantine.
- Human review queue.
- PII redaction.
- Dataset download and multi-dataset experiment scripts.
- Capability notebooks.
- CLI construction, benchmark, doctor, and dead-letter replay commands.

Adapter or stub only:

- Data warehouse loader.
- Full XMLPath mapper.
- API response mapper beyond loader support.
- spaCy/entity ruler/coreference/POS/sentence splitting.
- Transformer NER.
- Supervised, pairwise, and joint relation extraction.
- Ollama ontology/entity-label/relationship-type proposal as schema registry techniques.
- Splink/Dedupe full project-trained resolution.
- Graph-neighborhood entity resolution.
- JSON Schema/property graph/schema drift validators.
- Always-on live database CI jobs; current Neo4j/Kuzu live profiles are opt-in by environment.
- Chunk/entity text embeddings, external KG enrichment, graph summaries, temporal enrichment.

## 35. Recommended next implementation priorities

The repo is already broad. The most useful next implementation steps are:

1. Add live optional integration jobs:
   - Run Neo4j/Kuzu profiles in CI only when service credentials are present.
   - Store integration benchmark history.

2. Add trained model paths:
   - Real GraphSAGE training loop when PyG is installed.
   - Real PyKEEN pipeline execution and artifact metadata.

3. Add production-scale import optimizations:
   - Domain-specific Kuzu schema generation from `GraphSchema`.
   - Neo4j bulk import/export profiles for very large graphs.

4. Add larger public gold benchmarks:
   - CoNLL/GLiNER-compatible NER spans.
   - W3C mapping test-suite fixtures.
   - Human-reviewed relation-extraction corpora.

## 36. Verification commands

Useful local checks:

```bash
python3 -m compileall -q src tests
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m ruff check src tests
python3 -m mypy src/kg_construction
make smoke
make verify
```

Useful inventory checks:

```bash
PYTHONPATH=src python3 - <<'PY'
from collections import Counter
from kg_construction.application.registry import default_registry

registry = default_registry()
print(len(registry.techniques))
print(Counter(item.status for item in registry.techniques))
print(Counter(item.public_support_level for item in registry.techniques))
print(Counter(item.category for item in registry.techniques))
PY
```

Useful report checks:

```bash
sed -n '1,120p' reports/current_repo_capability_summary.md
sed -n '1,40p' reports/current_registry_matrix.csv
python3 -m json.tool reports/current_registry_summary.json | sed -n '1,80p'
```
