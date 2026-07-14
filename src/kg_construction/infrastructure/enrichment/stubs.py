"""Graph enrichment implementations and adapters."""

from __future__ import annotations

from kg_construction.domain.graph_objects import PropertyGraph
from kg_construction.infrastructure.enrichment.graph_ml import (
    CommunityDetectionConfig,
    CommunityDetectionEnricher,
    ConnectedComponentsConfig,
    ConnectedComponentsEnricher,
    GraphAnalyticsEnricher,
    GraphSAGEConfig,
    LinkPredictionConfig,
    LinkPredictionEnricher,
    NodeEmbeddingConfig,
    NodeEmbeddingEnricher,
    PageRankConfig,
    PageRankEnricher,
    PyTorchGeometricGraphSAGEAdapter,
)
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured


class ExtensibleEnricherStub:
    """Base class for enrichment adapters represented in the registry."""

    name = "enricher_stub"
    technique = "enricher_stub"
    dependency_hint = "Configure enrichment dependency."

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        raise AdapterNotConfigured(
            f"{self.name} is represented as an extensible enricher. "
            f"{self.dependency_hint} Nodes: {len(graph.nodes)}"
        )


class ChunkEmbeddingEnricher(ExtensibleEnricherStub):
    name = "chunk_embedding_enricher"
    technique = "chunk_embeddings"
    dependency_hint = "Provide text embedding model for source chunks."


class EntityEmbeddingEnricher(ExtensibleEnricherStub):
    name = "entity_embedding_enricher"
    technique = "entity_embeddings"
    dependency_hint = "Provide canonical entity text representation and embedding model."


class ExternalKGLinkEnricher(ExtensibleEnricherStub):
    name = "external_kg_link_enricher"
    technique = "external_kg_linking"
    dependency_hint = "Configure Wikidata/DBpedia/custom KG linking policy."


class EntitySummaryEnricher(ExtensibleEnricherStub):
    name = "entity_summary_enricher"
    technique = "ollama_entity_summaries"
    dependency_hint = "Configure Ollama summary prompt and citation policy."


class CommunitySummaryEnricher(ExtensibleEnricherStub):
    name = "community_summary_enricher"
    technique = "ollama_community_summaries"
    dependency_hint = "Configure Ollama community summarization prompt."


class TemporalEventEnricher(ExtensibleEnricherStub):
    name = "temporal_event_enricher"
    technique = "temporal_event_graph_enrichment"
    dependency_hint = "Configure event time extraction, ordering, and causality rules."


__all__ = [
    "ChunkEmbeddingEnricher",
    "CommunityDetectionConfig",
    "CommunityDetectionEnricher",
    "CommunitySummaryEnricher",
    "ConnectedComponentsConfig",
    "ConnectedComponentsEnricher",
    "EntityEmbeddingEnricher",
    "EntitySummaryEnricher",
    "ExtensibleEnricherStub",
    "ExternalKGLinkEnricher",
    "GraphAnalyticsEnricher",
    "GraphSAGEConfig",
    "LinkPredictionConfig",
    "LinkPredictionEnricher",
    "NodeEmbeddingConfig",
    "NodeEmbeddingEnricher",
    "PageRankConfig",
    "PageRankEnricher",
    "PyTorchGeometricGraphSAGEAdapter",
    "TemporalEventEnricher",
]
