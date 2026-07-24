# Current Repo Capability Summary

Report date: 2026-07-24

This report is generated from the live `kg_construction.application.registry.default_registry()` implementation registry. It replaces old article-specific and run-specific outputs with a compact source-of-truth view of the current repo capabilities.

## Registry totals

- Total registered techniques: 105
- implemented: 105

## Public support levels

The package is still `0.1.x` and classified as Alpha, but each registered technique has a user-facing support level:

| Support level | Count | Meaning |
| --- | ---: | --- |
| Mandatory | 105 | Part of the executable release contract; certification permits no skip or fallback. |
| Research inventory | 0 | Implemented or adapted outside the focused release contract; retained for comparative research. |
| Roadmap | 0 | A non-executable extension point kept outside the release registry. |

## Categories

| Category | Count |
| --- | ---: |
| evaluation | 6 |
| extractor | 17 |
| graph_builder | 3 |
| graph_store | 9 |
| hybrid | 1 |
| llm | 1 |
| loader | 19 |
| mapper | 21 |
| production | 9 |
| resolution | 10 |
| schema | 4 |
| validation | 5 |

## Method types

| Method type | Count |
| --- | ---: |
| deterministic | 37 |
| embedding | 2 |
| evaluation | 6 |
| human | 5 |
| hybrid | 5 |
| llm | 6 |
| ml | 4 |
| nlp | 2 |
| observability | 1 |
| ontology | 12 |
| privacy | 1 |
| reproducibility | 1 |
| resilience | 4 |
| rules | 19 |

## Capability groups

### evaluation

- Mandatory: benchmark_runner, claim_evaluation_suite, construction_metrics_evaluator, intended_use_metrics_evaluator, knowledge_graph_quality_evaluator, schema_quality_evaluator
- Research inventory: none
- Roadmap: none

### extractor

- Mandatory: gazetteer_entity_extractor, regex_entity_extractor, dependency_pattern_relation_extractor, evidence_first_unstructured_pipeline, spacy_ner_extractor, openie_relation_extractor, gliner_entity_extractor, transformer_ner_extractor, gliner_relex_claim_extractor, mlx_qwen_claim_extractor, ollama_qwen_claim_extractor, transformers_qwen_claim_extractor, mandatory_claim_ensemble, mandatory_mention_ensemble, mandatory_composed_unstructured_runtime, gliner_ollama_relation_typing_extractor, ollama_graph_extractor
- Research inventory: none
- Roadmap: none

### graph_builder

- Mandatory: evidence_graph_builder, property_graph_builder, rdf_graph_builder
- Research inventory: none
- Roadmap: none

### graph_store

- Mandatory: ladybug_exporter, memgraph_exporter, neo4j_exporter, networkx_exporter, rdf_exporter, rdflib_exporter, ladybug_ingestion_adapter, neo4j_ingestion_adapter, neo4j_vector_store_adapter
- Research inventory: none
- Roadmap: none

### hybrid

- Mandatory: hybrid_gliner_then_ollama_relations
- Research inventory: none
- Roadmap: none

### llm

- Mandatory: ollama_provider
- Research inventory: none
- Roadmap: none

### loader

- Mandatory: csv_loader, json_loader, markdown_loader, text_loader, api_response_loader, event_payload_loader, excel_loader, jsonld_loader, log_loader, sql_database_loader, xml_loader, structured_api_loader, html_loader, chat_transcript_loader, email_loader, pdf_loader, report_loader, webpage_loader, scientific_paper_loader
- Research inventory: none
- Roadmap: none

### mapper

- Mandatory: semantic_column_profiler, semantic_table_mapper, source_catalog, column_to_property_mapping, column_type_inference, direct_table_mapper, foreign_key_to_edge_mapping, primary_key_to_id_mapping, row_to_entity_mapping, schema_profiling, table_to_node_mapping, csvw_mapper, event_graph_mapper, jsonld_mapper, jsonpath_mapper, log_event_mapper, r2rml_mapper, rml_mapper, dotted_path_json_mapper, obda_virtual_graph_mapper, html_metadata_mapper
- Research inventory: none
- Roadmap: none

### production

- Mandatory: run_manifest_writer, structured_logging, batch_retry_policy, dead_letter_queue, pii_redaction_policy, dead_letter_replay_tool, error_quarantine, incremental_sync_policy, watermark_store
- Research inventory: none
- Roadmap: none

### resolution

- Mandatory: exact_key_entity_resolver, alias_table_resolver, normalized_string_resolver, fuzzy_string_resolver, rapidfuzz_resolver, dbpedia_linker, wikidata_linker, embedding_resolver, cross_encoder_resolver, openrefine_reconciler
- Research inventory: none
- Roadmap: none

### schema

- Mandatory: in_memory_schema_registry, schema_promotion_workflow, schema_proposal_workflow, schema_quarantine_namespace
- Research inventory: none
- Roadmap: none

### validation

- Mandatory: evidence_required_validator, confidence_threshold_validator, schema_registry_validator, shacl_validator, human_review_queue
- Research inventory: none
- Roadmap: none

## Artifact policy

- Keep source code, configs, examples, notebooks, and concise current reports in the repo.
- Treat `data/processed/` as a generated smoke-output workspace.
- Treat `reports/experiment_datasets/` as a generated high-volume experiment-output workspace.
- Regenerate bulky experiment outputs with `scripts/run_experiment_datasets.py` instead of storing stale graph dumps.
- Keep article-specific narrative assets outside the repo unless they directly document current implementation behavior.

## Current report files

- `reports/current_repo_capability_summary.md`
- `reports/current_registry_summary.json`
- `reports/current_registry_matrix.csv`
