"""Graph ML and topology enrichment for constructed property graphs."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from typing import Any, Literal

from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.domain.models import content_hash, stable_id
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured

JsonDict = dict[str, Any]
EmbeddingMethod = Literal["structural_hash", "spectral", "graphsage"]
LinkPredictionMethod = Literal[
    "resource_allocation",
    "jaccard",
    "common_neighbors",
    "preferential_attachment",
    "hybrid",
]


@dataclass(frozen=True, slots=True)
class PageRankConfig:
    """Configuration for PageRank enrichment."""

    alpha: float = 0.85
    max_iter: int = 100
    tol: float = 1e-6
    property_name: str = "pagerank"
    precision: int = 8


@dataclass(frozen=True, slots=True)
class CommunityDetectionConfig:
    """Configuration for community detection over an undirected projection."""

    property_name: str = "community_id"
    size_property_name: str = "community_size"
    resolution: float = 1.0
    precision: int = 6


@dataclass(frozen=True, slots=True)
class ConnectedComponentsConfig:
    """Configuration for weakly connected components."""

    property_name: str = "component_id"
    size_property_name: str = "component_size"


@dataclass(frozen=True, slots=True)
class NodeEmbeddingConfig:
    """Configuration for deterministic node embedding enrichment."""

    dimensions: int = 8
    method: EmbeddingMethod = "spectral"
    property_name: str = "node_embedding"
    precision: int = 6
    max_nodes_for_spectral: int = 500
    include_reverse_edges: bool = True
    seed: int = 13


@dataclass(frozen=True, slots=True)
class LinkPredictionConfig:
    """Configuration for topology and embedding-based link prediction."""

    method: LinkPredictionMethod = "hybrid"
    max_predictions: int = 10
    max_candidate_pairs: int = 5000
    min_score: float = 0.0
    only_with_common_neighbors: bool = True
    add_edges: bool = False
    edge_type: str = "PREDICTED_LINK"
    embedding_property_name: str = "node_embedding"
    precision: int = 6


@dataclass(frozen=True, slots=True)
class GraphSAGEConfig:
    """Configuration for the research-only PyTorch Geometric GraphSAGE adapter."""

    input_dimensions: int = 16
    hidden_dimensions: int = 16
    output_dimensions: int = 8
    seed: int = 13
    include_reverse_edges: bool = True
    normalize: bool = True


def _round(value: float, precision: int) -> float:
    return round(float(value), precision)


def _normalize_vector(values: list[float]) -> list[float]:
    norm = sqrt(sum(value * value for value in values))
    if norm <= 0:
        return values
    return [value / norm for value in values]


def _hash_int(*parts: object, length: int = 16) -> int:
    return int(content_hash("\x1f".join(str(part) for part in parts), length=length), 16)


def _hash_signed_float(*parts: object) -> float:
    raw = _hash_int(*parts)
    max_value = float(16**16 - 1)
    return (raw / max_value) * 2.0 - 1.0


def _safe_dimensions(dimensions: int) -> int:
    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be greater than zero.")
    return dimensions


def _node_lookup(graph: PropertyGraph) -> dict[str, PropertyNode]:
    return {node.node_id: node for node in graph.nodes}


def _valid_edge_pairs(graph: PropertyGraph) -> list[tuple[str, str]]:
    node_ids = set(_node_lookup(graph))
    return [
        (edge.source_id, edge.target_id)
        for edge in graph.edges
        if edge.source_id in node_ids and edge.target_id in node_ids and edge.source_id != edge.target_id
    ]


def _degree_maps(graph: PropertyGraph) -> tuple[Counter[str], Counter[str], Counter[str]]:
    in_degree: Counter[str] = Counter()
    out_degree: Counter[str] = Counter()
    total_degree: Counter[str] = Counter()
    for source, target in _valid_edge_pairs(graph):
        out_degree[source] += 1
        in_degree[target] += 1
        total_degree[source] += 1
        total_degree[target] += 1
    return in_degree, out_degree, total_degree


def _undirected_adjacency(graph: PropertyGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    for source, target in _valid_edge_pairs(graph):
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def _directed_out_adjacency(graph: PropertyGraph) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}
    for source, target in _valid_edge_pairs(graph):
        adjacency[source].add(target)
    return adjacency


def _weak_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    unseen = set(adjacency)
    components: list[set[str]] = []
    while unseen:
        start = min(unseen)
        queue: deque[str] = deque([start])
        component: set[str] = set()
        unseen.remove(start)
        while queue:
            node_id = queue.popleft()
            component.add(node_id)
            for neighbor in sorted(adjacency[node_id]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(component)
    components.sort(key=lambda item: (-len(item), sorted(item)[0] if item else ""))
    return components


def _nx_or_none() -> Any | None:
    try:
        import networkx as nx

        return nx
    except ImportError:
        return None


def _to_networkx_graph(graph: PropertyGraph, *, directed: bool) -> Any:
    nx = _nx_or_none()
    if nx is None:
        raise AdapterNotConfigured("NetworkX is required for this graph ML backend.")
    nx_graph = nx.DiGraph() if directed else nx.Graph()
    for node in graph.nodes:
        nx_graph.add_node(node.node_id, labels=node.labels, **node.properties)
    for edge in graph.edges:
        if edge.source_id == edge.target_id:
            continue
        if nx_graph.has_edge(edge.source_id, edge.target_id):
            nx_graph[edge.source_id][edge.target_id]["weight"] += 1.0
        else:
            nx_graph.add_edge(edge.source_id, edge.target_id, weight=1.0, type=edge.type)
    return nx_graph


def _pagerank_fallback(graph: PropertyGraph, config: PageRankConfig) -> dict[str, float]:
    node_ids = [node.node_id for node in graph.nodes]
    node_count = len(node_ids)
    if node_count == 0:
        return {}
    out_adjacency = _directed_out_adjacency(graph)
    scores = {node_id: 1.0 / node_count for node_id in node_ids}
    teleport = (1.0 - config.alpha) / node_count
    for _ in range(config.max_iter):
        next_scores = dict.fromkeys(node_ids, teleport)
        dangling_score = sum(scores[node_id] for node_id in node_ids if not out_adjacency[node_id])
        dangling_share = config.alpha * dangling_score / node_count
        for node_id in node_ids:
            next_scores[node_id] += dangling_share
            neighbors = out_adjacency[node_id]
            if not neighbors:
                continue
            share = config.alpha * scores[node_id] / len(neighbors)
            for neighbor in neighbors:
                next_scores[neighbor] += share
        delta = sum(abs(next_scores[node_id] - scores[node_id]) for node_id in node_ids)
        scores = next_scores
        if delta <= node_count * config.tol:
            break
    return scores


def _apply_node_properties(
    graph: PropertyGraph,
    properties_by_node: dict[str, JsonDict],
    *,
    metadata: JsonDict,
) -> PropertyGraph:
    nodes = [
        node.model_copy(update={"properties": {**node.properties, **properties_by_node.get(node.node_id, {})}})
        for node in graph.nodes
    ]
    return graph.model_copy(update={"nodes": nodes, "metadata": {**graph.metadata, **metadata}})


def _top_nodes(scores: dict[str, float], limit: int = 5) -> list[dict[str, object]]:
    return [
        {"node_id": node_id, "score": score}
        for node_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


class GraphAnalyticsEnricher:
    """Add degree features without external graph dependencies."""

    name = "graph_analytics_enricher"
    technique = "stdlib_degree_features"

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        in_degree, out_degree, total_degree = _degree_maps(graph)
        properties = {
            node.node_id: {
                "in_degree": in_degree.get(node.node_id, 0),
                "out_degree": out_degree.get(node.node_id, 0),
                "degree": total_degree.get(node.node_id, 0),
            }
            for node in graph.nodes
        }
        return _apply_node_properties(
            graph,
            properties,
            metadata={
                "graph_analytics": {
                    "method": self.technique,
                    "node_count": len(graph.nodes),
                    "edge_count": len(graph.edges),
                },
                "enriched_by": self.name,
            },
        )


class PageRankEnricher:
    """Add PageRank scores over the directed property-graph projection."""

    name = "pagerank_enricher"
    technique = "pagerank"

    def __init__(self, config: PageRankConfig | None = None) -> None:
        self.config = config or PageRankConfig()

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        nx = _nx_or_none()
        backend = "stdlib_power_iteration"
        if nx is None:
            scores = _pagerank_fallback(graph, self.config)
        else:
            nx_graph = _to_networkx_graph(graph, directed=True)
            if nx_graph.number_of_nodes() == 0:
                scores = {}
            else:
                scores = nx.pagerank(
                    nx_graph,
                    alpha=self.config.alpha,
                    max_iter=self.config.max_iter,
                    tol=self.config.tol,
                    weight="weight",
                )
            backend = "networkx_pagerank"

        rounded_scores = {
            node_id: _round(score, self.config.precision) for node_id, score in scores.items()
        }
        properties = {
            node_id: {self.config.property_name: score}
            for node_id, score in rounded_scores.items()
        }
        return _apply_node_properties(
            graph,
            properties,
            metadata={
                "pagerank": {
                    "backend": backend,
                    "alpha": self.config.alpha,
                    "top_nodes": _top_nodes(rounded_scores),
                },
                "enriched_by": self.name,
            },
        )


class CommunityDetectionEnricher:
    """Detect communities using greedy modularity when NetworkX is available."""

    name = "community_detection_enricher"
    technique = "greedy_modularity_communities"

    def __init__(self, config: CommunityDetectionConfig | None = None) -> None:
        self.config = config or CommunityDetectionConfig()

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        nx = _nx_or_none()
        backend = "weak_components_fallback"
        adjacency = _undirected_adjacency(graph)
        communities: list[set[str]]
        modularity: float | None = None
        if not graph.nodes:
            communities = []
        elif nx is None or not any(adjacency.values()):
            communities = _weak_components(adjacency)
        else:
            nx_graph = _to_networkx_graph(graph, directed=False)
            raw_communities = nx.community.greedy_modularity_communities(
                nx_graph,
                weight="weight",
                resolution=self.config.resolution,
            )
            communities = [set(item) for item in raw_communities]
            communities.sort(key=lambda item: (-len(item), sorted(item)[0] if item else ""))
            modularity = nx.community.modularity(nx_graph, communities, weight="weight")
            backend = "networkx_greedy_modularity"

        properties: dict[str, JsonDict] = {}
        for index, community in enumerate(communities):
            for node_id in community:
                properties[node_id] = {
                    self.config.property_name: index,
                    self.config.size_property_name: len(community),
                    "community_method": backend,
                }
        metadata: JsonDict = {
            "community_detection": {
                "backend": backend,
                "community_count": len(communities),
                "community_sizes": [len(item) for item in communities],
            },
            "enriched_by": self.name,
        }
        if modularity is not None:
            metadata["community_detection"]["modularity"] = _round(
                modularity, self.config.precision
            )
        return _apply_node_properties(graph, properties, metadata=metadata)


class ConnectedComponentsEnricher:
    """Add weak component IDs over the undirected graph projection."""

    name = "connected_components_enricher"
    technique = "weak_connected_components"

    def __init__(self, config: ConnectedComponentsConfig | None = None) -> None:
        self.config = config or ConnectedComponentsConfig()

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        components = _weak_components(_undirected_adjacency(graph))
        properties: dict[str, JsonDict] = {}
        for index, component in enumerate(components):
            for node_id in component:
                properties[node_id] = {
                    self.config.property_name: index,
                    self.config.size_property_name: len(component),
                }
        return _apply_node_properties(
            graph,
            properties,
            metadata={
                "connected_components": {
                    "component_count": len(components),
                    "component_sizes": [len(item) for item in components],
                },
                "enriched_by": self.name,
            },
        )


def _structural_embedding(
    graph: PropertyGraph,
    *,
    config: NodeEmbeddingConfig,
) -> dict[str, list[float]]:
    dimensions = _safe_dimensions(config.dimensions)
    in_degree, out_degree, total_degree = _degree_maps(graph)
    max_degree = max(total_degree.values(), default=1)
    vectors: dict[str, list[float]] = {}
    for node in graph.nodes:
        vector = [0.0 for _ in range(dimensions)]
        numeric_features = [
            in_degree.get(node.node_id, 0) / max_degree,
            out_degree.get(node.node_id, 0) / max_degree,
            total_degree.get(node.node_id, 0) / max_degree,
            len(node.labels) / 10.0,
        ]
        for index, value in enumerate(numeric_features[:dimensions]):
            vector[index] += value

        tokens = [node.node_id, *node.labels]
        name = node.properties.get("name")
        if name is not None:
            tokens.append(str(name))
        for key, value in sorted(node.properties.items()):
            if isinstance(value, str | int | float | bool):
                tokens.append(f"{key}={value}")
        for token in tokens:
            index = _hash_int(config.seed, "index", token) % dimensions
            vector[index] += _hash_signed_float(config.seed, "value", token)

        vectors[node.node_id] = [
            _round(value, config.precision) for value in _normalize_vector(vector)
        ]
    return vectors


def _spectral_embedding(
    graph: PropertyGraph,
    *,
    config: NodeEmbeddingConfig,
) -> tuple[dict[str, list[float]], str]:
    if len(graph.nodes) > config.max_nodes_for_spectral:
        return _structural_embedding(graph, config=config), "structural_hash_size_fallback"
    try:
        import numpy as np
    except ImportError:
        return _structural_embedding(graph, config=config), "structural_hash_dependency_fallback"

    dimensions = _safe_dimensions(config.dimensions)
    node_ids = sorted(node.node_id for node in graph.nodes)
    if not node_ids:
        return {}, "spectral"
    index_by_node = {node_id: index for index, node_id in enumerate(node_ids)}
    adjacency = np.zeros((len(node_ids), len(node_ids)), dtype=float)
    for source, target in _valid_edge_pairs(graph):
        source_index = index_by_node[source]
        target_index = index_by_node[target]
        adjacency[source_index, target_index] += 1.0
        if config.include_reverse_edges:
            adjacency[target_index, source_index] += 1.0
    adjacency += np.eye(len(node_ids), dtype=float)
    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    normalized = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1][:dimensions]
    selected = eigenvectors[:, order]
    if selected.shape[1] < dimensions:
        selected = np.pad(selected, ((0, 0), (0, dimensions - selected.shape[1])))

    vectors: dict[str, list[float]] = {}
    for column_index in range(selected.shape[1]):
        column = selected[:, column_index]
        anchor_index = int(np.argmax(np.abs(column)))
        if column[anchor_index] < 0:
            selected[:, column_index] = -column
    for node_id, row in zip(node_ids, selected, strict=True):
        values = [float(value) for value in row[:dimensions]]
        vectors[node_id] = [
            _round(value, config.precision) for value in _normalize_vector(values)
        ]
    return vectors, "spectral"


class NodeEmbeddingEnricher:
    """Add deterministic node embeddings for retrieval and ML feature handoff."""

    name = "node_embedding_enricher"
    technique = "node_embeddings"

    def __init__(self, config: NodeEmbeddingConfig | None = None) -> None:
        self.config = config or NodeEmbeddingConfig()

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        if self.config.method == "graphsage":
            embeddings = PyTorchGeometricGraphSAGEAdapter(
                GraphSAGEConfig(
                    output_dimensions=self.config.dimensions,
                    seed=self.config.seed,
                    include_reverse_edges=self.config.include_reverse_edges,
                )
            ).encode(graph)
            backend = "pytorch_geometric_graphsage"
        elif self.config.method == "spectral":
            embeddings, backend = _spectral_embedding(graph, config=self.config)
        else:
            embeddings = _structural_embedding(graph, config=self.config)
            backend = "structural_hash"

        properties = {
            node_id: {self.config.property_name: vector}
            for node_id, vector in embeddings.items()
        }
        return _apply_node_properties(
            graph,
            properties,
            metadata={
                "node_embeddings": {
                    "backend": backend,
                    "dimensions": self.config.dimensions,
                    "property_name": self.config.property_name,
                    "node_count": len(embeddings),
                },
                "enriched_by": self.name,
            },
        )


def _embedding_cosine(left: Iterable[Any] | None, right: Iterable[Any] | None) -> float | None:
    if left is None or right is None:
        return None
    try:
        left_values = [float(value) for value in left]
        right_values = [float(value) for value in right]
    except (TypeError, ValueError):
        return None
    if len(left_values) != len(right_values) or not left_values:
        return None
    left_norm = sqrt(sum(value * value for value in left_values))
    right_norm = sqrt(sum(value * value for value in right_values))
    if left_norm <= 0 or right_norm <= 0:
        return None
    return sum(a * b for a, b in zip(left_values, right_values, strict=True)) / (
        left_norm * right_norm
    )


def _candidate_pairs(
    node_ids: list[str],
    adjacency: dict[str, set[str]],
    *,
    config: LinkPredictionConfig,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for left, right in combinations(sorted(node_ids), 2):
        if right in adjacency[left]:
            continue
        if config.only_with_common_neighbors and not adjacency[left].intersection(adjacency[right]):
            continue
        pairs.append((left, right))
        if len(pairs) >= config.max_candidate_pairs:
            break
    return pairs


def _normalizer(values: Iterable[float]) -> float:
    return max([abs(value) for value in values] or [1.0]) or 1.0


class LinkPredictionEnricher:
    """Score plausible missing links using topology and configured embeddings."""

    name = "link_prediction_enricher"
    technique = "topology_link_prediction"

    def __init__(self, config: LinkPredictionConfig | None = None) -> None:
        self.config = config or LinkPredictionConfig()

    def enrich(self, graph: PropertyGraph) -> PropertyGraph:
        adjacency = _undirected_adjacency(graph)
        node_lookup = _node_lookup(graph)
        candidates = _candidate_pairs(list(node_lookup), adjacency, config=self.config)
        raw_predictions: list[JsonDict] = []
        for source_id, target_id in candidates:
            source_neighbors = adjacency[source_id]
            target_neighbors = adjacency[target_id]
            common = source_neighbors.intersection(target_neighbors)
            union = source_neighbors.union(target_neighbors)
            resource_allocation = sum(1.0 / max(len(adjacency[neighbor]), 1) for neighbor in common)
            jaccard = len(common) / len(union) if union else 0.0
            preferential = len(source_neighbors) * len(target_neighbors)
            cosine = _embedding_cosine(
                node_lookup[source_id].properties.get(self.config.embedding_property_name),
                node_lookup[target_id].properties.get(self.config.embedding_property_name),
            )
            raw_predictions.append(
                {
                    "source_id": source_id,
                    "target_id": target_id,
                    "source_name": node_lookup[source_id].properties.get("name"),
                    "target_name": node_lookup[target_id].properties.get("name"),
                    "common_neighbor_count": len(common),
                    "common_neighbors": sorted(common),
                    "resource_allocation": resource_allocation,
                    "jaccard": jaccard,
                    "preferential_attachment": preferential,
                    "embedding_cosine": cosine,
                }
            )

        ra_norm = _normalizer(item["resource_allocation"] for item in raw_predictions)
        cn_norm = _normalizer(item["common_neighbor_count"] for item in raw_predictions)
        pa_norm = _normalizer(item["preferential_attachment"] for item in raw_predictions)
        predictions: list[JsonDict] = []
        for item in raw_predictions:
            cosine = item["embedding_cosine"]
            cosine_score = 0.0 if cosine is None else (float(cosine) + 1.0) / 2.0
            if self.config.method == "resource_allocation":
                score = float(item["resource_allocation"]) / ra_norm
            elif self.config.method == "jaccard":
                score = float(item["jaccard"])
            elif self.config.method == "common_neighbors":
                score = float(item["common_neighbor_count"]) / cn_norm
            elif self.config.method == "preferential_attachment":
                score = float(item["preferential_attachment"]) / pa_norm
            else:
                score = (
                    0.35 * (float(item["resource_allocation"]) / ra_norm)
                    + 0.25 * float(item["jaccard"])
                    + 0.20 * (float(item["common_neighbor_count"]) / cn_norm)
                    + 0.20 * cosine_score
                )
            if score < self.config.min_score:
                continue
            predictions.append(
                {
                    **item,
                    "score": _round(score, self.config.precision),
                    "method": self.config.method,
                    "requires_validation": True,
                    "evidence_type": "topology_and_embedding_features",
                }
            )

        predictions.sort(
            key=lambda item: (-float(item["score"]), str(item["source_id"]), str(item["target_id"]))
        )
        predictions = predictions[: self.config.max_predictions]

        edges = list(graph.edges)
        if self.config.add_edges:
            for prediction in predictions:
                edge_id = stable_id(
                    "predicted_edge",
                    prediction["source_id"],
                    self.config.edge_type,
                    prediction["target_id"],
                    self.config.method,
                )
                edges.append(
                    PropertyEdge(
                        edge_id=edge_id,
                        source_id=str(prediction["source_id"]),
                        target_id=str(prediction["target_id"]),
                        type=self.config.edge_type,
                        properties={
                            "prediction_score": prediction["score"],
                            "prediction_method": self.config.method,
                            "requires_validation": True,
                            "is_inferred": True,
                            "common_neighbors": prediction["common_neighbors"],
                            "resource_allocation": _round(
                                float(prediction["resource_allocation"]), self.config.precision
                            ),
                            "jaccard": _round(float(prediction["jaccard"]), self.config.precision),
                        },
                    )
                )

        return graph.model_copy(
            update={
                "edges": edges,
                "metadata": {
                    **graph.metadata,
                    "link_prediction": {
                        "method": self.config.method,
                        "candidate_pair_count": len(candidates),
                        "prediction_count": len(predictions),
                        "predictions": predictions,
                        "added_edges": self.config.add_edges,
                    },
                    "enriched_by": self.name,
                },
            }
        )


def _node_feature_matrix(graph: PropertyGraph, dimensions: int, seed: int) -> list[list[float]]:
    config = NodeEmbeddingConfig(
        dimensions=dimensions,
        method="structural_hash",
        precision=8,
        seed=seed,
    )
    embeddings = _structural_embedding(graph, config=config)
    return [embeddings[node.node_id] for node in graph.nodes]


class PyTorchGeometricGraphSAGEAdapter:
    """Research-only GraphSAGE adapter backed by mandatory repository dependencies."""

    name = "pytorch_geometric_graphsage_adapter"
    technique = "graphsage_pyg_adapter"

    def __init__(self, config: GraphSAGEConfig | None = None) -> None:
        self.config = config or GraphSAGEConfig()

    def encode(self, graph: PropertyGraph) -> dict[str, list[float]]:
        try:
            import torch
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise AdapterNotConfigured(
                "The mandatory torch and torch-geometric dependencies are missing."
            ) from exc

        if not graph.nodes:
            return {}
        torch.manual_seed(self.config.seed)
        node_ids = [node.node_id for node in graph.nodes]
        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        features = torch.tensor(
            _node_feature_matrix(graph, self.config.input_dimensions, self.config.seed),
            dtype=torch.float32,
        )
        edge_indices: list[tuple[int, int]] = []
        for source, target in _valid_edge_pairs(graph):
            edge_indices.append((node_index[source], node_index[target]))
            if self.config.include_reverse_edges:
                edge_indices.append((node_index[target], node_index[source]))
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        class _GraphSAGEEncoder(torch.nn.Module):
            def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
                super().__init__()
                self.conv1 = SAGEConv(input_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, output_dim, normalize=True)

            def forward(self, x: Any, edges: Any) -> Any:
                x = self.conv1(x, edges).relu()
                return self.conv2(x, edges)

        model = _GraphSAGEEncoder(
            self.config.input_dimensions,
            self.config.hidden_dimensions,
            self.config.output_dimensions,
        )
        model.eval()
        with torch.no_grad():
            encoded = model(features, edge_index)
            if self.config.normalize:
                encoded = torch.nn.functional.normalize(encoded, p=2, dim=1)
        return {
            node_id: [round(float(value), 6) for value in encoded[index].tolist()]
            for node_id, index in node_index.items()
        }


__all__ = [
    "CommunityDetectionConfig",
    "CommunityDetectionEnricher",
    "ConnectedComponentsConfig",
    "ConnectedComponentsEnricher",
    "GraphAnalyticsEnricher",
    "GraphSAGEConfig",
    "LinkPredictionConfig",
    "LinkPredictionEnricher",
    "NodeEmbeddingConfig",
    "NodeEmbeddingEnricher",
    "PageRankConfig",
    "PageRankEnricher",
    "PyTorchGeometricGraphSAGEAdapter",
]
