"""Certification-owned capability registry and broader research inventory."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

from kg_construction.evaluation import (
    ConstructionMetricsEvaluator,
    IntendedUseMetricsEvaluator,
    KnowledgeGraphQualityEvaluator,
    SchemaQualityEvaluator,
)
from kg_construction.evaluation.claim_evaluations import ClaimEvaluationSuite
from kg_construction.infrastructure.enrichment import (
    ChunkEmbeddingEnricher,
    CommunityDetectionEnricher,
    CommunitySummaryEnricher,
    ConnectedComponentsEnricher,
    EntityEmbeddingEnricher,
    EntitySummaryEnricher,
    ExternalKGLinkEnricher,
    GraphAnalyticsEnricher,
    GraphRetrievalBenchmark,
    GraphSageLinkPredictionTrainer,
    GraphTrainValidationSplitter,
    KnowledgeGraphEmbeddingAdapter,
    LargeGraphAnalyticsBackend,
    LinkPredictionEnricher,
    Node2VecEmbeddingAdapter,
    NodeEmbeddingEnricher,
    PageRankEnricher,
    PyTorchGeometricGraphSAGEAdapter,
    TemporalEventEnricher,
)
from kg_construction.infrastructure.extractors import (
    CoreferenceAdapter,
    DependencyPatternRelationExtractor,
    GazetteerEntityExtractor,
    GlinerOllamaRelationTypingExtractor,
    JointEntityRelationExtractor,
    OllamaGraphExtractor,
    OpenIERelationExtractor,
    PairwiseRelationClassifierAdapter,
    PosTaggingAdapter,
    RegexEntityExtractor,
    SentenceSplitterAdapter,
    SpacyEntityRulerExtractor,
    SupervisedRelationExtractor,
    TransformerRelationExtractor,
)
from kg_construction.infrastructure.extractors.model_backed import (
    GlinerMentionDetector,
    GlinerRelExClaimExtractor,
    MandatoryClaimEnsemble,
    MandatoryMentionEnsemble,
    MlxQwenClaimExtractor,
    OllamaQwenClaimExtractor,
    SpacyMentionDetector,
    TransformersMentionDetector,
    TransformersQwenClaimExtractor,
)
from kg_construction.infrastructure.graph_builders import (
    EvidenceGraphBuilder,
    PropertyGraphBuilder,
    RDFGraphBuilder,
)
from kg_construction.infrastructure.graph_stores import (
    LadybugExporter,
    LadybugIngestionAdapter,
    MemgraphExporter,
    Neo4jExporter,
    Neo4jIngestionAdapter,
    Neo4jVectorStoreAdapter,
    NetworkXExporter,
    RDFExporter,
    RDFLibExporter,
)
from kg_construction.infrastructure.llm import OllamaProvider
from kg_construction.infrastructure.loaders import (
    APIResponseLoader,
    ChatTranscriptLoader,
    CsvLoader,
    DataWarehouseExportLoader,
    EmailLoader,
    EventPayloadLoader,
    ExcelLoader,
    HTMLLoader,
    JSONLDLoader,
    JsonLoader,
    LogLoader,
    MarkdownLoader,
    PDFLoader,
    ReportLoader,
    ScientificPaperLoader,
    SQLDatabaseLoader,
    StructuredAPILoader,
    TextLoader,
    WebPageLoader,
    XMLLoader,
)
from kg_construction.infrastructure.mapping import (
    APIResponseMapper,
    CSVWMapper,
    DirectTableMapper,
    DottedPathJsonMapper,
    EventGraphMapper,
    HTMLMetadataMapper,
    JSONLDMapper,
    JsonPathMapper,
    LogEventMapper,
    OBDAVirtualGraphMapper,
    R2RMLMapper,
    RMLMapper,
    XMLPathMapper,
)
from kg_construction.infrastructure.ontology import (
    ConfidenceThresholdValidator,
    EvidenceRequiredValidator,
    HumanReviewQueue,
    InMemorySchemaRegistry,
    JSONSchemaValidator,
    OwlOntologyLoader,
    PropertyGraphValidator,
    SchemaDriftValidator,
    SchemaProposalWorkflow,
    SchemaRegistryValidator,
    ShaclValidator,
)
from kg_construction.infrastructure.production import (
    BatchRetryPolicy,
    DeadLetterQueue,
    DeadLetterReplayTool,
    ErrorQuarantine,
    IncrementalSyncPolicy,
    PIIRedactionPolicy,
    RunManifestWriter,
    StructuredLogger,
    WatermarkStore,
)
from kg_construction.infrastructure.resolution import (
    AliasTableResolver,
    BlockingResolver,
    CrossEncoderResolver,
    DBpediaLinker,
    DedupeActiveLearningResolver,
    EmbeddingResolver,
    ExactKeyEntityResolver,
    GraphNeighborhoodResolver,
    HumanReviewResolver,
    NormalizedStringResolver,
    OpenRefineReconciler,
    RapidFuzzResolver,
    SplinkProbabilisticResolver,
    WikidataLinker,
)
from kg_construction.research.simple_benchmark import BenchmarkRunner
from kg_construction.structured.semantic_profiler import SemanticColumnProfiler
from kg_construction.structured.source_catalog import CatalogTableMapper, SourceCatalog
from kg_construction.unstructured.pipeline import UnstructuredPipeline
from kg_construction.unstructured.runtime import (
    MandatoryUnstructuredRuntime,
)

TechniqueStatus = Literal["implemented", "adapter", "stub"]
SupportLevel = Literal["mandatory", "research", "roadmap"]

SUPPORT_LEVEL_DEFINITIONS: dict[SupportLevel, str] = {
    "mandatory": (
        "Part of the executable release contract; certification permits no skip or fallback."
    ),
    "research": (
        "Implemented or adapted outside the focused release contract; retained for comparative research."
    ),
    "roadmap": "A non-executable extension point kept outside the release registry.",
}

SUPPORT_LEVEL_LABELS: dict[SupportLevel, str] = {
    "mandatory": "Mandatory",
    "research": "Research inventory",
    "roadmap": "Roadmap",
}

_EXPERIMENTAL_METHOD_TYPES = {"embedding", "llm", "ml"}


@dataclass(frozen=True, slots=True)
class TechniqueDescriptor:
    """Registry metadata for a Data-to-KG technique."""

    name: str
    category: str
    method_type: str
    status: TechniqueStatus
    priority: int
    adapter: type | None = None
    input_types: tuple[str, ...] = ()
    notes: str = ""
    tags: tuple[str, ...] = ()
    support_level: SupportLevel | None = None
    profiles: tuple[str, ...] = ("darwin-arm64", "linux-cpu", "linux-cuda")

    @property
    def public_support_level(self) -> SupportLevel:
        """Return the user-facing maturity level for README/docs/reports."""

        if self.support_level is not None:
            return self.support_level
        if self.status == "stub":
            return "roadmap"
        if self.status == "adapter":
            return "research"
        if self.category == "enrichment":
            return "research"
        if "experimental" in self.tags or self.method_type in _EXPERIMENTAL_METHOD_TYPES:
            return "research"
        return "mandatory"

    @property
    def public_support_label(self) -> str:
        """Return the display label for the user-facing support level."""

        return SUPPORT_LEVEL_LABELS[self.public_support_level]

    @property
    def public_support_description(self) -> str:
        """Return the display description for the user-facing support level."""

        return SUPPORT_LEVEL_DEFINITIONS[self.public_support_level]


@dataclass
class PluginRegistry:
    """Searchable registry of all construction techniques."""

    techniques: list[TechniqueDescriptor] = field(default_factory=list)

    def register(self, descriptor: TechniqueDescriptor) -> None:
        self.techniques.append(descriptor)

    def by_category(self, category: str) -> list[TechniqueDescriptor]:
        return [item for item in self.techniques if item.category == category]

    def get(self, name: str) -> TechniqueDescriptor:
        for item in self.techniques:
            if item.name == name:
                return item
        raise KeyError(name)

    def categories(self) -> set[str]:
        return {item.category for item in self.techniques}

    def names(self) -> set[str]:
        return {item.name for item in self.techniques}


def _d(
    name: str,
    category: str,
    method_type: str,
    status: TechniqueStatus,
    priority: int,
    adapter: type | None = None,
    input_types: tuple[str, ...] = (),
    notes: str = "",
    tags: tuple[str, ...] = (),
    support_level: SupportLevel | None = None,
    profiles: tuple[str, ...] = ("darwin-arm64", "linux-cpu", "linux-cuda"),
) -> TechniqueDescriptor:
    return TechniqueDescriptor(
        name=name,
        category=category,
        method_type=method_type,
        status=status,
        priority=priority,
        adapter=adapter,
        input_types=input_types,
        notes=notes,
        tags=tags,
        support_level=support_level,
        profiles=profiles,
    )


def research_registry() -> PluginRegistry:
    """Return the complete implementation, adapter, and roadmap inventory.

    This broad catalogue exists for comparative research and future work. It is
    intentionally not the release surface: stubs, adapter-only entries, and
    downstream graph-enrichment experiments must never be counted as mandatory
    construction capabilities.
    """

    registry = PluginRegistry()
    descriptors = [
        _d("csv_loader", "loader", "deterministic", "implemented", 10, CsvLoader, ("csv",)),
        _d("excel_loader", "loader", "deterministic", "implemented", 12, ExcelLoader, ("excel",)),
        _d("sql_database_loader", "loader", "deterministic", "implemented", 12, SQLDatabaseLoader, ("sql", "relational_table")),
        _d("data_warehouse_export_loader", "loader", "deterministic", "stub", 12, DataWarehouseExportLoader, ("warehouse",)),
        _d("structured_api_loader", "loader", "deterministic", "implemented", 14, StructuredAPILoader, ("api",)),
        _d("json_loader", "loader", "deterministic", "implemented", 10, JsonLoader, ("json",)),
        _d("xml_loader", "loader", "deterministic", "implemented", 12, XMLLoader, ("xml",)),
        _d("jsonld_loader", "loader", "ontology", "implemented", 12, JSONLDLoader, ("jsonld",)),
        _d("html_loader", "loader", "rules", "implemented", 16, HTMLLoader, ("html",)),
        _d("webpage_loader", "loader", "rules", "implemented", 18, WebPageLoader, ("webpage",)),
        _d("log_loader", "loader", "rules", "implemented", 12, LogLoader, ("logs",)),
        _d("event_payload_loader", "loader", "deterministic", "implemented", 12, EventPayloadLoader, ("event_payload",)),
        _d("api_response_loader", "loader", "deterministic", "implemented", 12, APIResponseLoader, ("api_response",)),
        _d("text_loader", "loader", "deterministic", "implemented", 10, TextLoader, ("text", "long_form_document")),
        _d("markdown_loader", "loader", "deterministic", "implemented", 10, MarkdownLoader, ("markdown",)),
        _d("pdf_loader", "loader", "rules", "implemented", 18, PDFLoader, ("pdf",)),
        _d("report_loader", "loader", "rules", "implemented", 18, ReportLoader, ("report",)),
        _d("email_loader", "loader", "rules", "implemented", 18, EmailLoader, ("email",)),
        _d("chat_transcript_loader", "loader", "rules", "implemented", 18, ChatTranscriptLoader, ("chat_transcript",)),
        _d("scientific_paper_loader", "loader", "rules", "implemented", 20, ScientificPaperLoader, ("scientific_paper",)),
        _d("semantic_column_profiler", "mapper", "deterministic", "implemented", 9, SemanticColumnProfiler, ("csv", "sql", "excel")),
        _d("semantic_table_mapper", "mapper", "deterministic", "implemented", 9, CatalogTableMapper, ("csv", "sql", "excel")),
        _d("source_catalog", "mapper", "deterministic", "implemented", 9, SourceCatalog, ("csv", "sql", "excel")),
        _d("direct_table_mapper", "mapper", "deterministic", "implemented", 10, DirectTableMapper, ("csv", "sql", "excel")),
        _d("table_to_node_mapping", "mapper", "deterministic", "implemented", 10, DirectTableMapper),
        _d("row_to_entity_mapping", "mapper", "deterministic", "implemented", 10, DirectTableMapper),
        _d("primary_key_to_id_mapping", "mapper", "deterministic", "implemented", 10, DirectTableMapper),
        _d("foreign_key_to_edge_mapping", "mapper", "deterministic", "implemented", 10, DirectTableMapper),
        _d("column_to_property_mapping", "mapper", "deterministic", "implemented", 10, DirectTableMapper),
        _d("schema_profiling", "mapper", "deterministic", "implemented", 10, DirectTableMapper),
        _d("column_type_inference", "mapper", "deterministic", "implemented", 10, DirectTableMapper),
        _d("relationship_inference_from_keys", "mapper", "deterministic", "adapter", 14, DirectTableMapper),
        _d("incremental_watermark_sync", "mapper", "deterministic", "stub", 14, None),
        _d("r2rml_mapper", "mapper", "ontology", "implemented", 12, R2RMLMapper, ("sql",)),
        _d("rml_mapper", "mapper", "ontology", "implemented", 12, RMLMapper, ("csv", "json", "xml")),
        _d("csvw_mapper", "mapper", "ontology", "implemented", 12, CSVWMapper, ("csv",)),
        _d("jsonld_mapper", "mapper", "ontology", "implemented", 12, JSONLDMapper, ("jsonld",)),
        _d("obda_virtual_graph_mapper", "mapper", "ontology", "implemented", 14, OBDAVirtualGraphMapper, ("sql",)),
        _d("jsonpath_mapper", "mapper", "deterministic", "implemented", 12, JsonPathMapper, ("json",)),
        _d("dotted_path_json_mapper", "mapper", "deterministic", "implemented", 13, DottedPathJsonMapper, ("json",)),
        _d("xmlpath_mapper", "mapper", "deterministic", "stub", 12, XMLPathMapper, ("xml",)),
        _d("api_response_mapper", "mapper", "deterministic", "stub", 12, APIResponseMapper, ("api_response",)),
        _d("event_graph_mapper", "mapper", "deterministic", "implemented", 12, EventGraphMapper, ("event_payload", "logs")),
        _d("log_event_mapper", "mapper", "rules", "implemented", 12, LogEventMapper, ("logs",)),
        _d("html_metadata_mapper", "mapper", "rules", "implemented", 16, HTMLMetadataMapper, ("html", "webpage")),
        _d("regex_entity_extractor", "extractor", "rules", "implemented", 20, RegexEntityExtractor, ("text",)),
        _d("gazetteer_entity_extractor", "extractor", "rules", "implemented", 20, GazetteerEntityExtractor, ("text",)),
        _d("spacy_entity_ruler_extractor", "extractor", "rules", "stub", 22, SpacyEntityRulerExtractor, ("text",)),
        _d("dependency_pattern_relation_extractor", "extractor", "rules", "implemented", 22, DependencyPatternRelationExtractor, ("text",)),
        _d("sentence_splitter_adapter", "extractor", "nlp", "stub", 25, SentenceSplitterAdapter, ("text",)),
        _d("pos_tagging_adapter", "extractor", "nlp", "stub", 25, PosTaggingAdapter, ("text",)),
        _d("coreference_adapter", "extractor", "nlp", "stub", 26, CoreferenceAdapter, ("text",)),
        _d(
            "spacy_ner_extractor",
            "extractor",
            "nlp",
            "implemented",
            25,
            SpacyMentionDetector,
            ("text",),
        ),
        _d(
            "transformer_ner_extractor",
            "extractor",
            "ml",
            "implemented",
            30,
            TransformersMentionDetector,
            ("text",),
        ),
        _d(
            "gliner_entity_extractor",
            "extractor",
            "ml",
            "implemented",
            30,
            GlinerMentionDetector,
            ("text",),
        ),
        _d(
            "gliner_relex_claim_extractor",
            "extractor",
            "ml",
            "implemented",
            31,
            GlinerRelExClaimExtractor,
            ("text",),
        ),
        _d(
            "transformers_qwen_claim_extractor",
            "extractor",
            "llm",
            "implemented",
            32,
            TransformersQwenClaimExtractor,
            ("text",),
        ),
        _d(
            "mlx_qwen_claim_extractor",
            "extractor",
            "llm",
            "implemented",
            32,
            MlxQwenClaimExtractor,
            ("text",),
            profiles=("darwin-arm64",),
        ),
        _d(
            "ollama_qwen_claim_extractor",
            "extractor",
            "llm",
            "implemented",
            32,
            OllamaQwenClaimExtractor,
            ("text",),
        ),
        _d(
            "mandatory_mention_ensemble",
            "extractor",
            "hybrid",
            "implemented",
            33,
            MandatoryMentionEnsemble,
            ("text",),
        ),
        _d(
            "mandatory_claim_ensemble",
            "extractor",
            "hybrid",
            "implemented",
            33,
            MandatoryClaimEnsemble,
            ("text",),
        ),
        _d(
            "mandatory_composed_unstructured_runtime",
            "extractor",
            "hybrid",
            "implemented",
            34,
            MandatoryUnstructuredRuntime,
            ("text", "html"),
            notes=(
                "Single governed release path: every locked extractor and identity/"
                "linking provider participates, NIL stays observed, claims pass "
                "ontology/SHACL, and versioned held-out metrics gate publication."
            ),
        ),
        _d("openie_relation_extractor", "extractor", "nlp", "implemented", 28, OpenIERelationExtractor, ("text",)),
        _d("pairwise_relation_classifier_adapter", "extractor", "ml", "stub", 32, PairwiseRelationClassifierAdapter, ("text",)),
        _d("joint_entity_relation_extractor", "extractor", "ml", "stub", 32, JointEntityRelationExtractor, ("text",)),
        _d("evidence_first_unstructured_pipeline", "extractor", "hybrid", "implemented", 24, UnstructuredPipeline, ("text",)),
        _d("transformer_relation_extractor", "extractor", "ml", "adapter", 32, TransformerRelationExtractor, ("text",)),
        _d("supervised_relation_extractor", "extractor", "ml", "stub", 32, SupervisedRelationExtractor, ("text",)),
        _d(
            "ollama_graph_extractor",
            "extractor",
            "llm",
            "implemented",
            40,
            OllamaGraphExtractor,
            ("text",),
        ),
        _d(
            "gliner_ollama_relation_typing_extractor",
            "extractor",
            "hybrid",
            "implemented",
            40,
            GlinerOllamaRelationTypingExtractor,
            ("text",),
        ),
        _d("ollama_schema_guided_extraction", "extractor", "llm", "adapter", 40, OllamaGraphExtractor, ("text",)),
        _d("ollama_schema_later_extraction", "extractor", "llm", "adapter", 42, OllamaGraphExtractor, ("text",)),
        _d("ollama_ontology_proposal", "schema", "llm", "stub", 45, SchemaProposalWorkflow),
        _d("ollama_entity_label_proposal", "schema", "llm", "stub", 45, SchemaProposalWorkflow),
        _d("ollama_relationship_type_proposal", "schema", "llm", "stub", 45, SchemaProposalWorkflow),
        _d("exact_key_entity_resolver", "resolution", "deterministic", "implemented", 10, ExactKeyEntityResolver),
        _d("normalized_string_resolver", "resolution", "rules", "implemented", 20, NormalizedStringResolver),
        _d("fuzzy_string_resolver", "resolution", "rules", "implemented", 22, RapidFuzzResolver),
        _d("rapidfuzz_resolver", "resolution", "rules", "implemented", 22, RapidFuzzResolver),
        _d("alias_table_resolver", "resolution", "rules", "implemented", 20, AliasTableResolver),
        _d("blocking_resolver", "resolution", "rules", "stub", 24, BlockingResolver),
        _d("probabilistic_linkage_resolver", "resolution", "ml", "adapter", 30, SplinkProbabilisticResolver),
        _d("splink_probabilistic_resolver", "resolution", "ml", "adapter", 30, SplinkProbabilisticResolver),
        _d("dedupe_active_learning_resolver", "resolution", "ml", "adapter", 32, DedupeActiveLearningResolver),
        _d("embedding_resolver", "resolution", "embedding", "implemented", 35, EmbeddingResolver),
        _d("cross_encoder_resolver", "resolution", "ml", "implemented", 36, CrossEncoderResolver),
        _d("graph_neighborhood_resolver", "resolution", "ml", "stub", 36, GraphNeighborhoodResolver),
        _d("wikidata_linker", "resolution", "rules", "implemented", 28, WikidataLinker),
        _d("dbpedia_linker", "resolution", "rules", "implemented", 28, DBpediaLinker),
        _d("openrefine_reconciler", "resolution", "human", "implemented", 50, OpenRefineReconciler),
        _d("human_review_resolver", "resolution", "human", "stub", 50, HumanReviewResolver),
        _d("in_memory_schema_registry", "schema", "ontology", "implemented", 15, InMemorySchemaRegistry),
        _d("owl_ontology_loader", "schema", "ontology", "stub", 16, OwlOntologyLoader),
        _d("schema_proposal_workflow", "schema", "human", "implemented", 50, SchemaProposalWorkflow),
        _d("schema_quarantine_namespace", "schema", "human", "implemented", 50, SchemaProposalWorkflow),
        _d("schema_promotion_workflow", "schema", "human", "implemented", 50, SchemaProposalWorkflow),
        _d("evidence_required_validator", "validation", "deterministic", "implemented", 10, EvidenceRequiredValidator),
        _d("schema_registry_validator", "validation", "ontology", "implemented", 15, SchemaRegistryValidator),
        _d("confidence_threshold_validator", "validation", "deterministic", "implemented", 15, ConfidenceThresholdValidator),
        _d("json_schema_validator", "validation", "deterministic", "stub", 12, JSONSchemaValidator),
        _d("shacl_validator", "validation", "ontology", "implemented", 16, ShaclValidator),
        _d("property_graph_validator", "validation", "deterministic", "stub", 16, PropertyGraphValidator),
        _d("schema_drift_validator", "validation", "deterministic", "stub", 16, SchemaDriftValidator),
        _d("hallucination_filter", "validation", "rules", "adapter", 16, EvidenceRequiredValidator),
        _d("human_review_queue", "validation", "human", "implemented", 50, HumanReviewQueue),
        _d("property_graph_builder", "graph_builder", "deterministic", "implemented", 10, PropertyGraphBuilder),
        _d("rdf_graph_builder", "graph_builder", "ontology", "implemented", 15, RDFGraphBuilder),
        _d("evidence_graph_builder", "graph_builder", "deterministic", "implemented", 10, EvidenceGraphBuilder),
        _d("neo4j_exporter", "graph_store", "deterministic", "implemented", 12, Neo4jExporter),
        _d(
            "neo4j_ingestion_adapter",
            "graph_store",
            "deterministic",
            "implemented",
            14,
            Neo4jIngestionAdapter,
        ),
        _d(
            "neo4j_vector_store_adapter",
            "graph_store",
            "embedding",
            "implemented",
            14,
            Neo4jVectorStoreAdapter,
            notes=(
                "Fixed-dimension Neo4j vector index with validated transactional "
                "upserts and parameterized equality-filtered ANN search."
            ),
            support_level="mandatory",
        ),
        _d("memgraph_exporter", "graph_store", "deterministic", "implemented", 12, MemgraphExporter),
        _d(
            "ladybug_exporter",
            "graph_store",
            "deterministic",
            "implemented",
            12,
            LadybugExporter,
        ),
        _d(
            "ladybug_ingestion_adapter",
            "graph_store",
            "deterministic",
            "implemented",
            14,
            LadybugIngestionAdapter,
        ),
        _d("networkx_exporter", "graph_store", "deterministic", "implemented", 12, NetworkXExporter),
        _d("rdf_exporter", "graph_store", "ontology", "implemented", 12, RDFExporter),
        _d("rdflib_exporter", "graph_store", "ontology", "implemented", 12, RDFLibExporter),
        _d("ollama_provider", "llm", "llm", "implemented", 40, OllamaProvider),
        _d("graph_analytics_enricher", "enrichment", "deterministic", "implemented", 20, GraphAnalyticsEnricher),
        _d("graph_train_validation_splitter", "enrichment", "ml", "implemented", 34, GraphTrainValidationSplitter),
        _d("node_embedding_enricher", "enrichment", "embedding", "implemented", 35, NodeEmbeddingEnricher),
        _d("spectral_node_embedding_enricher", "enrichment", "embedding", "implemented", 35, NodeEmbeddingEnricher),
        _d("structural_hash_node_embedding_enricher", "enrichment", "embedding", "implemented", 35, NodeEmbeddingEnricher),
        _d("node2vec_embedding_adapter", "enrichment", "embedding", "implemented", 36, Node2VecEmbeddingAdapter),
        _d("knowledge_graph_embedding_adapter", "enrichment", "embedding", "implemented", 36, KnowledgeGraphEmbeddingAdapter),
        _d("graphsage_pyg_adapter", "enrichment", "ml", "adapter", 38, PyTorchGeometricGraphSAGEAdapter),
        _d("graphsage_link_prediction_trainer", "enrichment", "ml", "implemented", 38, GraphSageLinkPredictionTrainer),
        _d("large_graph_analytics_backend", "enrichment", "ml", "implemented", 38, LargeGraphAnalyticsBackend),
        _d("chunk_embedding_enricher", "enrichment", "embedding", "stub", 35, ChunkEmbeddingEnricher),
        _d("entity_embedding_enricher", "enrichment", "embedding", "stub", 35, EntityEmbeddingEnricher),
        _d("external_kg_link_enricher", "enrichment", "rules", "stub", 28, ExternalKGLinkEnricher),
        _d("pagerank_enricher", "enrichment", "ml", "implemented", 30, PageRankEnricher),
        _d("community_detection_enricher", "enrichment", "ml", "implemented", 30, CommunityDetectionEnricher),
        _d("connected_components_enricher", "enrichment", "ml", "implemented", 30, ConnectedComponentsEnricher),
        _d("link_prediction_enricher", "enrichment", "ml", "implemented", 35, LinkPredictionEnricher),
        _d("resource_allocation_link_prediction", "enrichment", "ml", "implemented", 35, LinkPredictionEnricher),
        _d("jaccard_link_prediction", "enrichment", "ml", "implemented", 35, LinkPredictionEnricher),
        _d("hybrid_topology_embedding_link_prediction", "enrichment", "ml", "implemented", 37, LinkPredictionEnricher),
        _d(
            "graph_retrieval_benchmark",
            "enrichment",
            "evaluation",
            "implemented",
            39,
            GraphRetrievalBenchmark,
            tags=("experimental",),
        ),
        _d("entity_summary_enricher", "enrichment", "llm", "stub", 45, EntitySummaryEnricher),
        _d("community_summary_enricher", "enrichment", "llm", "stub", 45, CommunitySummaryEnricher),
        _d("temporal_event_enricher", "enrichment", "rules", "stub", 30, TemporalEventEnricher),
        _d("construction_metrics_evaluator", "evaluation", "evaluation", "implemented", 60, ConstructionMetricsEvaluator),
        _d(
            "benchmark_runner",
            "evaluation",
            "evaluation",
            "implemented",
            60,
            BenchmarkRunner,
            support_level="research",
        ),
        _d("claim_evaluation_suite", "evaluation", "evaluation", "implemented", 60, ClaimEvaluationSuite),
        _d(
            "knowledge_graph_quality_evaluator",
            "evaluation",
            "evaluation",
            "implemented",
            60,
            KnowledgeGraphQualityEvaluator,
        ),
        _d(
            "schema_quality_evaluator",
            "evaluation",
            "evaluation",
            "implemented",
            60,
            SchemaQualityEvaluator,
        ),
        _d(
            "intended_use_metrics_evaluator",
            "evaluation",
            "evaluation",
            "implemented",
            60,
            IntendedUseMetricsEvaluator,
        ),
        _d(
            "hybrid_rules_gazetteer_regex",
            "hybrid",
            "rules",
            "implemented",
            22,
            None,
            ("text",),
            support_level="research",
        ),
        _d("hybrid_rules_ml_ollama", "hybrid", "llm", "adapter", 35, None),
        _d("hybrid_structured_then_text", "hybrid", "deterministic", "adapter", 18, None),
        _d(
            "hybrid_gliner_then_ollama_relations",
            "hybrid",
            "llm",
            "implemented",
            38,
            GlinerOllamaRelationTypingExtractor,
            ("text",),
        ),
        _d("hybrid_ollama_schema_then_gliner", "hybrid", "llm", "adapter", 38, None),
        _d("hybrid_openie_then_ollama_canonicalization", "hybrid", "llm", "adapter", 38, None),
        _d("hybrid_deterministic_then_llm_enrichment", "hybrid", "llm", "adapter", 38, None),
        _d("hybrid_human_reviewed_schema", "hybrid", "human", "adapter", 50, None),
        _d("structured_logging", "production", "observability", "implemented", 5, StructuredLogger),
        _d("run_manifest_writer", "production", "reproducibility", "implemented", 5, RunManifestWriter),
        _d("batch_retry_policy", "production", "resilience", "implemented", 6, BatchRetryPolicy),
        _d("dead_letter_queue", "production", "resilience", "implemented", 6, DeadLetterQueue),
        _d("dead_letter_replay_tool", "production", "resilience", "implemented", 7, DeadLetterReplayTool),
        _d("incremental_sync_policy", "production", "deterministic", "implemented", 8, IncrementalSyncPolicy),
        _d("watermark_store", "production", "deterministic", "implemented", 8, WatermarkStore),
        _d("error_quarantine", "production", "resilience", "implemented", 8, ErrorQuarantine),
        _d("pii_redaction_policy", "production", "privacy", "implemented", 6, PIIRedactionPolicy),
    ]
    for descriptor in descriptors:
        registry.register(descriptor)
    return registry


def default_registry() -> PluginRegistry:
    """Return only mandatory executable Data-to-KG release capabilities.

    The product is deliberately centred on structured construction, unstructured
    construction, evaluation/validation, graph materialisation, and the production
    services needed to execute those paths. Downstream graph-ML enrichment belongs
    to :func:`research_registry`, as do every stub and adapter-only declaration.
    """

    release = PluginRegistry()
    for descriptor in research_registry().techniques:
        if descriptor.status != "implemented":
            continue
        if descriptor.support_level in {"research", "roadmap"}:
            continue
        if descriptor.category == "enrichment" or descriptor.adapter is None:
            continue
        release.register(
            replace(
                descriptor,
                support_level="mandatory",
            )
        )
    return release


__all__ = [
    "PluginRegistry",
    "SUPPORT_LEVEL_DEFINITIONS",
    "SUPPORT_LEVEL_LABELS",
    "SupportLevel",
    "TechniqueDescriptor",
    "TechniqueStatus",
    "default_registry",
    "research_registry",
]
