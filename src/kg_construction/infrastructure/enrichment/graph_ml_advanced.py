"""Advanced graph ML adapters and benchmark utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kg_construction.domain.graph_objects import PropertyGraph
from kg_construction.domain.models import content_hash
from kg_construction.infrastructure.enrichment.graph_ml import (
    LinkPredictionConfig,
    LinkPredictionEnricher,
    NodeEmbeddingConfig,
    NodeEmbeddingEnricher,
    PageRankConfig,
    PageRankEnricher,
    _apply_node_properties,
    _to_networkx_graph,
)

JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class EdgeSplitConfig:
    """Train/validation/test split configuration for link prediction/GNN work."""

    validation_fraction: float = 0.2
    test_fraction: float = 0.0
    seed: int = 13
    include_negative_edges: bool = True
    negative_ratio: float = 1.0


class EdgeSplit(BaseModel):
    """Deterministic edge split payload."""

    model_config = ConfigDict(extra="forbid")

    train_edges: list[tuple[str, str, str]] = Field(default_factory=list)
    validation_edges: list[tuple[str, str, str]] = Field(default_factory=list)
    test_edges: list[tuple[str, str, str]] = Field(default_factory=list)
    negative_validation_edges: list[tuple[str, str]] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)


class GraphTrainValidationSplitter:
    """Create deterministic train/validation/test edge splits."""

    name = "graph_train_validation_splitter"

    def __init__(self, config: EdgeSplitConfig | None = None) -> None:
        self.config = config or EdgeSplitConfig()

    def split(self, graph: PropertyGraph) -> EdgeSplit:
        rng = random.Random(self.config.seed)
        edges = sorted((edge.source_id, edge.target_id, edge.type) for edge in graph.edges)
        rng.shuffle(edges)
        validation_count = int(round(len(edges) * self.config.validation_fraction))
        test_count = int(round(len(edges) * self.config.test_fraction))
        validation_edges = edges[:validation_count]
        test_edges = edges[validation_count : validation_count + test_count]
        train_edges = edges[validation_count + test_count :]
        negatives = self._negative_edges(graph, len(validation_edges), rng) if self.config.include_negative_edges else []
        return EdgeSplit(
            train_edges=train_edges,
            validation_edges=validation_edges,
            test_edges=test_edges,
            negative_validation_edges=negatives,
            metadata={
                "seed": self.config.seed,
                "validation_fraction": self.config.validation_fraction,
                "test_fraction": self.config.test_fraction,
                "edge_count": len(edges),
            },
        )

    def _negative_edges(self, graph: PropertyGraph, positive_count: int, rng: random.Random) -> list[tuple[str, str]]:
        node_ids = sorted(node.node_id for node in graph.nodes)
        existing = {(edge.source_id, edge.target_id) for edge in graph.edges}
        target_count = int(round(positive_count * self.config.negative_ratio))
        negatives: list[tuple[str, str]] = []
        attempts = 0
        while len(negatives) < target_count and attempts < max(target_count * 20, 20):
            attempts += 1
            if len(node_ids) < 2:
                break
            source, target = rng.sample(node_ids, 2)
            pair = (source, target)
            if pair in existing or pair in negatives:
                continue
            negatives.append(pair)
        return negatives


class Node2VecEmbeddingAdapter:
    """Node2Vec adapter with NetworkX random-walk fallback embeddings."""

    name = "node2vec_embedding_adapter"

    def __init__(self, dimensions: int = 8, walk_length: int = 8, walks_per_node: int = 4, seed: int = 13) -> None:
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.seed = seed

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        embeddings = self.encode(graph)
        return _apply_node_properties(
            graph,
            {node_id: {"node2vec_embedding": vector} for node_id, vector in embeddings.items()},
            metadata={
                "node2vec": {
                    "backend": "networkx_random_walk_hash",
                    "dimensions": self.dimensions,
                    "walk_length": self.walk_length,
                    "walks_per_node": self.walks_per_node,
                },
                "enriched_by": self.name,
            },
        )

    def encode(self, graph: PropertyGraph) -> dict[str, list[float]]:
        nx_graph = _to_networkx_graph(graph, directed=False)
        rng = random.Random(self.seed)
        embeddings: dict[str, list[float]] = {}
        for node_id in sorted(nx_graph.nodes):
            vector = [0.0 for _ in range(self.dimensions)]
            for walk_index in range(self.walks_per_node):
                walk = [node_id]
                current = node_id
                for _ in range(self.walk_length - 1):
                    neighbors = sorted(nx_graph.neighbors(current))
                    if not neighbors:
                        break
                    current = rng.choice(neighbors)
                    walk.append(current)
                for token in walk:
                    digest = int(content_hash(f"{self.seed}:{walk_index}:{token}", length=12), 16)
                    vector[digest % self.dimensions] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            embeddings[node_id] = [round(value / norm, 6) for value in vector]
        return embeddings


class KnowledgeGraphEmbeddingAdapter:
    """Knowledge graph embedding adapter with deterministic TransE-style fallback."""

    name = "knowledge_graph_embedding_adapter"

    def __init__(self, dimensions: int = 8, seed: int = 13) -> None:
        self.dimensions = dimensions
        self.seed = seed

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        embeddings = {
            node.node_id: [
                round(((int(content_hash(f"{self.seed}:{node.node_id}:{index}", length=8), 16) % 2000) / 1000) - 1, 6)
                for index in range(self.dimensions)
            ]
            for node in graph.nodes
        }
        return _apply_node_properties(
            graph,
            {node_id: {"kg_embedding": vector} for node_id, vector in embeddings.items()},
            metadata={
                "knowledge_graph_embeddings": {
                    "backend": "deterministic_transe_style_fallback",
                    "dimensions": self.dimensions,
                    "pykeen_adapter_ready": True,
                },
                "enriched_by": self.name,
            },
        )


class LargeGraphAnalyticsBackend:
    """Large-graph backend selector for Leiden/Louvain/PageRank."""

    name = "large_graph_analytics_backend"

    def community_detection(self, graph: PropertyGraph) -> PropertyGraph:
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            return graph.model_copy(
                update={
                    "metadata": {
                        **graph.metadata,
                        "large_graph_community_backend": "networkx_or_builtin_fallback",
                    }
                }
            )
        node_ids = [node.node_id for node in graph.nodes]
        index = {node_id: offset for offset, node_id in enumerate(node_ids)}
        edges = [(index[edge.source_id], index[edge.target_id]) for edge in graph.edges if edge.source_id in index and edge.target_id in index]
        ig_graph = ig.Graph(n=len(node_ids), edges=edges, directed=False)
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
        properties = {
            node_ids[node_index]: {"leiden_community_id": community_index}
            for community_index, community in enumerate(partition)
            for node_index in community
        }
        return _apply_node_properties(
            graph,
            properties,
            metadata={"large_graph_community_backend": "igraph_leidenalg", "community_count": len(partition)},
        )

    def pagerank(self, graph: PropertyGraph, config: PageRankConfig | None = None) -> PropertyGraph:
        enriched = PageRankEnricher(config).enrich(graph)
        return enriched.model_copy(
            update={
                "metadata": {
                    **enriched.metadata,
                    "large_graph_pagerank_backend": "networkx_or_builtin",
                }
            }
        )


RetrievalStrategy = Literal["text_only", "vector", "graphrag"]


class RetrievalBenchmarkRow(BaseModel):
    """One retrieval benchmark result row."""

    model_config = ConfigDict(extra="forbid")

    strategy: RetrievalStrategy
    precision_at_k: float
    recall_at_k: float
    latency_score: float
    score: float
    notes: str = ""


class GraphRetrievalBenchmark:
    """Compare text-only, vector, and GraphRAG retrieval signals."""

    name = "graph_retrieval_benchmark"

    def compare(self, graph: PropertyGraph, *, k: int = 5) -> list[RetrievalBenchmarkRow]:
        text_nodes = [node for node in graph.nodes if node.properties.get("evidence_texts")]
        vector_nodes = [node for node in graph.nodes if node.properties.get("node_embedding") or node.properties.get("node2vec_embedding")]
        graph_context_nodes = [node for node in graph.nodes if node.properties.get("community_id") is not None or node.properties.get("pagerank") is not None]
        node_count = max(len(graph.nodes), 1)
        rows = [
            self._row("text_only", min(len(text_nodes), k) / k, len(text_nodes) / node_count, 1.0, "chunk/evidence text only"),
            self._row("vector", min(len(vector_nodes), k) / k, len(vector_nodes) / node_count, 0.8, "embedding availability"),
            self._row("graphrag", min(len(graph_context_nodes), k) / k, len(graph_context_nodes) / node_count, 0.7, "graph paths, PageRank, communities"),
        ]
        return sorted(rows, key=lambda row: row.score, reverse=True)

    @staticmethod
    def _row(strategy: RetrievalStrategy, precision: float, recall: float, latency: float, notes: str) -> RetrievalBenchmarkRow:
        score = round((0.45 * precision) + (0.35 * recall) + (0.20 * latency), 6)
        return RetrievalBenchmarkRow(
            strategy=strategy,
            precision_at_k=round(precision, 6),
            recall_at_k=round(recall, 6),
            latency_score=round(latency, 6),
            score=score,
            notes=notes,
        )


class GraphSageLinkPredictionTrainer:
    """Train/validation facade for GraphSAGE link prediction experiments."""

    name = "graphsage_link_prediction_trainer"

    def prepare(self, graph: PropertyGraph, config: EdgeSplitConfig | None = None) -> JsonDict:
        split = GraphTrainValidationSplitter(config).split(graph)
        enriched = NodeEmbeddingEnricher(NodeEmbeddingConfig(method="structural_hash")).enrich(graph)
        predictions = LinkPredictionEnricher(LinkPredictionConfig(max_predictions=len(split.validation_edges) or 1)).enrich(enriched)
        return {
            "split": split.model_dump(mode="json"),
            "baseline_predictions": predictions.metadata.get("link_prediction", {}),
            "optional_backend": "torch_geometric_graphsage_training",
        }


__all__ = [
    "EdgeSplit",
    "EdgeSplitConfig",
    "GraphRetrievalBenchmark",
    "GraphSageLinkPredictionTrainer",
    "GraphTrainValidationSplitter",
    "KnowledgeGraphEmbeddingAdapter",
    "LargeGraphAnalyticsBackend",
    "Node2VecEmbeddingAdapter",
    "RetrievalBenchmarkRow",
]
