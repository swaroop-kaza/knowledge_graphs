# Current Repo Capability Summary

Report date: 2026-07-14

This report is generated from the live `kg_construction.application.registry.default_registry()` implementation registry. It replaces old article-specific and run-specific outputs with a compact source-of-truth view of the current repo capabilities.

## Registry totals

- Total registered techniques: 150
- adapter: 17
- implemented: 104
- stub: 29

## Public support levels

The package is still `0.1.x` and classified as Alpha, but each registered technique has a user-facing support level:

| Support level | Count | Meaning |
| --- | ---: | --- |
| Stable | 76 | Safe to use now with the documented API and normal test coverage. |
| Optional adapter | 29 | Works when the required dependency, model, service, or project wiring is configured. |
| Experimental | 16 | Implemented for research/prototyping, but API, scoring, or model behavior may change. |
| Planned | 29 | Registered as a roadmap extension point, but not implemented for normal use. |

## Categories

| Category | Count |
| --- | ---: |
| enrichment | 24 |
| evaluation | 2 |
| extractor | 20 |
| graph_builder | 3 |
| graph_store | 8 |
| hybrid | 8 |
| llm | 1 |
| loader | 20 |
| mapper | 22 |
| production | 9 |
| resolution | 16 |
| schema | 8 |
| validation | 9 |

## Method types

| Method type | Count |
| --- | ---: |
| deterministic | 44 |
| embedding | 8 |
| evaluation | 3 |
| human | 7 |
| hybrid | 1 |
| llm | 14 |
| ml | 23 |
| nlp | 5 |
| observability | 1 |
| ontology | 13 |
| privacy | 1 |
| reproducibility | 1 |
| resilience | 4 |
| rules | 25 |

## Capability groups

### enrichment

- Stable: graph_analytics_enricher
- Optional adapter: graphsage_pyg_adapter
- Experimental: community_detection_enricher, connected_components_enricher, pagerank_enricher, graph_train_validation_splitter, jaccard_link_prediction, link_prediction_enricher, node_embedding_enricher, resource_allocation_link_prediction, spectral_node_embedding_enricher, structural_hash_node_embedding_enricher, knowledge_graph_embedding_adapter, node2vec_embedding_adapter, hybrid_topology_embedding_link_prediction, graphsage_link_prediction_trainer, large_graph_analytics_backend, graph_retrieval_benchmark
- Planned: external_kg_link_enricher, temporal_event_enricher, chunk_embedding_enricher, entity_embedding_enricher, community_summary_enricher, entity_summary_enricher

### evaluation

- Stable: benchmark_runner, construction_metrics_evaluator
- Optional adapter: none
- Experimental: none
- Planned: none

### extractor

- Stable: gazetteer_entity_extractor, regex_entity_extractor, dependency_pattern_relation_extractor, openie_relation_extractor
- Optional adapter: gliner_entity_extractor, rebel_relation_extractor, transformer_relation_extractor, gliner_ollama_relation_typing_extractor, ollama_graph_extractor, ollama_schema_guided_extraction, ollama_schema_later_extraction
- Experimental: none
- Planned: spacy_entity_ruler_extractor, pos_tagging_adapter, sentence_splitter_adapter, spacy_ner_extractor, coreference_adapter, transformer_ner_extractor, joint_entity_relation_extractor, pairwise_relation_classifier_adapter, supervised_relation_extractor

### graph_builder

- Stable: evidence_graph_builder, property_graph_builder, rdf_graph_builder
- Optional adapter: none
- Experimental: none
- Planned: none

### graph_store

- Stable: kuzu_exporter, memgraph_exporter, neo4j_exporter, networkx_exporter, rdf_exporter
- Optional adapter: rdflib_exporter, kuzu_ingestion_adapter, neo4j_ingestion_adapter
- Experimental: none
- Planned: none

### hybrid

- Stable: hybrid_rules_gazetteer_regex
- Optional adapter: hybrid_structured_then_text, hybrid_rules_ml_ollama, hybrid_deterministic_then_llm_enrichment, hybrid_gliner_then_ollama_relations, hybrid_ollama_schema_then_gliner, hybrid_openie_then_ollama_canonicalization, hybrid_human_reviewed_schema
- Experimental: none
- Planned: none

### llm

- Stable: none
- Optional adapter: ollama_provider
- Experimental: none
- Planned: none

### loader

- Stable: csv_loader, json_loader, markdown_loader, text_loader, api_response_loader, event_payload_loader, excel_loader, jsonld_loader, log_loader, sql_database_loader, xml_loader, structured_api_loader, html_loader, chat_transcript_loader, email_loader, pdf_loader, report_loader, webpage_loader, scientific_paper_loader
- Optional adapter: none
- Experimental: none
- Planned: data_warehouse_export_loader

### mapper

- Stable: column_to_property_mapping, column_type_inference, direct_table_mapper, foreign_key_to_edge_mapping, primary_key_to_id_mapping, row_to_entity_mapping, schema_profiling, table_to_node_mapping, csvw_mapper, event_graph_mapper, jsonld_mapper, jsonpath_mapper, log_event_mapper, r2rml_mapper, rml_mapper, dotted_path_json_mapper, obda_virtual_graph_mapper, html_metadata_mapper
- Optional adapter: relationship_inference_from_keys
- Experimental: none
- Planned: api_response_mapper, xmlpath_mapper, incremental_watermark_sync

### production

- Stable: run_manifest_writer, structured_logging, batch_retry_policy, dead_letter_queue, pii_redaction_policy, dead_letter_replay_tool, error_quarantine, incremental_sync_policy, watermark_store
- Optional adapter: none
- Experimental: none
- Planned: none

### resolution

- Stable: exact_key_entity_resolver, alias_table_resolver, normalized_string_resolver, fuzzy_string_resolver, rapidfuzz_resolver
- Optional adapter: dbpedia_linker, wikidata_linker, probabilistic_linkage_resolver, splink_probabilistic_resolver, dedupe_active_learning_resolver, embedding_resolver, cross_encoder_resolver, openrefine_reconciler
- Experimental: none
- Planned: blocking_resolver, graph_neighborhood_resolver, human_review_resolver

### schema

- Stable: in_memory_schema_registry, schema_promotion_workflow, schema_proposal_workflow, schema_quarantine_namespace
- Optional adapter: none
- Experimental: none
- Planned: owl_ontology_loader, ollama_entity_label_proposal, ollama_ontology_proposal, ollama_relationship_type_proposal

### validation

- Stable: evidence_required_validator, confidence_threshold_validator, schema_registry_validator, shacl_validator, human_review_queue
- Optional adapter: hallucination_filter
- Experimental: none
- Planned: json_schema_validator, property_graph_validator, schema_drift_validator

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
