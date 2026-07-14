"""Extensible entity resolution and external linking adapters."""

from __future__ import annotations

from kg_construction.domain.models import ConstructionBatch
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured


class ExtensibleResolverStub:
    """Base class for resolution techniques represented by the framework."""

    name = "resolver_stub"
    technique = "resolver_stub"
    dependency_hint = "Configure the resolver-specific dependency."

    def resolve(self, batch: ConstructionBatch) -> ConstructionBatch:
        raise AdapterNotConfigured(
            f"{self.name} is represented as an extensible resolver. "
            f"{self.dependency_hint} Entity candidates: {len(batch.entities)}"
        )


class BlockingResolver(ExtensibleResolverStub):
    name = "blocking_resolver"
    technique = "blocking_canopy_clustering"
    dependency_hint = "Configure blocking keys and canopy similarity thresholds."


class EmbeddingResolver(ExtensibleResolverStub):
    name = "embedding_resolver"
    technique = "embedding_similarity_matching"
    dependency_hint = "Provide embedding model, vector index, and threshold policy."


class CrossEncoderResolver(ExtensibleResolverStub):
    name = "cross_encoder_resolver"
    technique = "cross_encoder_pair_scoring"
    dependency_hint = "Provide pair generator and cross-encoder checkpoint."


class GraphNeighborhoodResolver(ExtensibleResolverStub):
    name = "graph_neighborhood_resolver"
    technique = "graph_neighborhood_matching"
    dependency_hint = "Provide graph features and neighborhood similarity scorer."


class WikidataLinker(ExtensibleResolverStub):
    name = "wikidata_linker"
    technique = "wikidata_external_kb_linking"
    dependency_hint = "Configure Wikidata search/SPARQL policy and caching."


class DBpediaLinker(ExtensibleResolverStub):
    name = "dbpedia_linker"
    technique = "dbpedia_external_kb_linking"
    dependency_hint = "Configure DBpedia lookup/SPARQL policy and caching."


class OpenRefineReconciler(ExtensibleResolverStub):
    name = "openrefine_reconciler"
    technique = "openrefine_style_reconciliation"
    dependency_hint = "Configure reconciliation endpoint and candidate scoring."


class HumanReviewResolver(ExtensibleResolverStub):
    name = "human_review_resolver"
    technique = "human_in_the_loop_merge_review"
    dependency_hint = "Configure queue storage, reviewer workflow, and merge audit policy."


__all__ = [
    "BlockingResolver",
    "CrossEncoderResolver",
    "DBpediaLinker",
    "EmbeddingResolver",
    "ExtensibleResolverStub",
    "GraphNeighborhoodResolver",
    "HumanReviewResolver",
    "OpenRefineReconciler",
    "WikidataLinker",
]
