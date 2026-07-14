# Next Stage Capabilities

This document summarizes the latest implementation wave across standards mapping, observability graphs, graph ML, production ingestion, and gold-standard benchmarks.

## Standards Mapping

| Capability | Implementation | Notes |
|---|---|---|
| JSON-LD expansion | `JSONLDMapper.expand` | Uses `pyld` when installed, with lightweight fallback. |
| JSON-LD RDF dataset generation | `JSONLDMapper.map_source` | Uses RDFLib JSON-LD parsing and emits evidence-backed `GraphFact` records. |
| OBDA virtual KG rewriting | `OBDAVirtualGraphMapper.rewrite_select` | Rewrites simple SPARQL triple patterns into SQL plans from predicate mappings. |
| R2RML/RML referencing object maps | `parent_triples_map` + `join_conditions` | Supports child/parent joins and emits joined RDF facts. |
| Cross-source joins | `R2RMLMapper.map_sources`, `RMLMapper.map_sources` | Joins rows across multiple `SourceRecord`s. |
| CSVW dialect parsing | `CSVWDialectParser` | Supports delimiter, quote char, skip rows/columns, headers, comments, trim, encoding. |
| CSVW validation | `CSVWValidator` | Checks required columns, missing columns, unexpected columns, primary key null/unique rules. |

## Event And Observability Graphs

| Capability | Implementation | Notes |
|---|---|---|
| Event payload loader | `EventPayloadLoader` | Loads `events`, `data`, `records`, `items`, or a single event object. |
| Event graph mapper | `EventGraphMapper` | Emits Event, Actor, Object, System, Trace, and Span candidates. |
| Trace/span correlation | `EventGraphMapper` | Adds `IN_TRACE`, `IN_SPAN`, and `PARENT_OF_SPAN` relations. |
| Event causality | `EventGraphMapper` | Adds `CAUSED` edges from `caused_by` / `parent_event_id`. |
| Log template mining | `LogLoader._mine_template` | Normalizes IDs, numbers, UUIDs, hex values into reusable templates. |
| Log template graph nodes | `LogEventMapper` | Adds `LogTemplate` nodes and `MATCHES_TEMPLATE` edges. |

## Graph ML

| Capability | Implementation | Notes |
|---|---|---|
| Train/validation/test splits | `GraphTrainValidationSplitter` | Deterministic edge splits with optional negative samples. |
| GraphSAGE preparation | `GraphSageLinkPredictionTrainer` | Produces split payload plus baseline link prediction; PyG remains optional. |
| Node2Vec adapter | `Node2VecEmbeddingAdapter` | NetworkX random-walk hash fallback for deterministic embeddings. |
| Knowledge graph embeddings | `KnowledgeGraphEmbeddingAdapter` | Deterministic TransE-style fallback with PyKEEN adapter-ready metadata. |
| Large-graph community backend | `LargeGraphAnalyticsBackend` | Uses `igraph`/`leidenalg` when installed; fallback metadata otherwise. |
| Scalable PageRank backend | `LargeGraphAnalyticsBackend.pagerank` | Routes through existing PageRank implementation and marks large-graph backend. |
| Retrieval benchmarks | `GraphRetrievalBenchmark` | Compares text-only, vector, and GraphRAG-style retrieval signals. |

## Production Ingestion

| Capability | Implementation | Notes |
|---|---|---|
| Live integration profiles | `configs/integration_profiles.example.yaml` | Defines Neo4j/Kuzu opt-in environment profiles. |
| High-volume graph fixtures | `generate_graph_store_benchmark_fixture` | Deterministic synthetic graphs for batch sizing tests. |
| RDF-to-Neo4j neosemantics | `Neo4jIngestionAdapter.write_rdf_graph` | Calls `n10s.rdf.import.inline` only when explicitly enabled. |
| Kuzu bulk import | `KuzuIngestionAdapter.bulk_import_property_graph` | Writes node/edge CSV files and executes Kuzu `COPY`. |

## Gold Benchmarks

| Capability | Implementation | Notes |
|---|---|---|
| Gold NER spans | `tests/fixtures/gold_standards/ner_spans.json` | Human-reviewed entity span fixture. |
| Gold triples | `tests/fixtures/gold_standards/gold_triples.json` | W3C/domain-style triple fixture. |
| Gold relation sample | `tests/fixtures/gold_standards/relation_sample.json` | Human-reviewed relation extraction fixture. |
| Gold entity resolution clusters | `tests/fixtures/gold_standards/entity_resolution_clusters.json` | Human-reviewed duplicate clusters converted to pairwise ER gold pairs. |
| Gold GraphRAG/retrieval cases | `tests/fixtures/gold_standards/retrieval_cases.json` | Queries with relevant contexts, graph paths, citations, and answer hints. |
| Gold evaluator | `GoldBenchmarkEvaluator` | Computes precision, recall, F1, TP, FP, FN for spans, relations, and triples. |
| Gold benchmark story | `reports/gold_standard_benchmark_story.md` | Public report covering NER, relation extraction, triples, ER, and GraphRAG/retrieval. |

## References

- W3C JSON-LD 1.1 API: https://www.w3.org/TR/json-ld11-api/
- W3C CSV on the Web Metadata: https://www.w3.org/TR/tabular-metadata/
- W3C R2RML: https://www.w3.org/TR/r2rml/
- RML mapping language: https://rml.io/specs/rml/
- OpenTelemetry trace semantic conventions: https://opentelemetry.io/docs/specs/semconv/
- Drain log template mining paper: https://jiemingzhu.github.io/pub/pjhe_icws2017.pdf
- PyTorch Geometric `RandomLinkSplit`: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.transforms.RandomLinkSplit.html
- PyTorch Geometric `Node2Vec`: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.Node2Vec.html
- PyKEEN: https://pykeen.readthedocs.io/
- Leiden algorithm package: https://leidenalg.readthedocs.io/
- Neo4j neosemantics: https://neo4j.com/labs/neosemantics/
- Kuzu `COPY`: https://docs.kuzudb.com/import/copy-from-csv/
