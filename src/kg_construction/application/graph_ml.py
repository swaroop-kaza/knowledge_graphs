"""Application workflow for graph ML enrichment over constructed graphs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from kg_construction.domain.graph_objects import PropertyGraph
from kg_construction.infrastructure.enrichment import (
    CommunityDetectionConfig,
    CommunityDetectionEnricher,
    ConnectedComponentsConfig,
    ConnectedComponentsEnricher,
    GraphAnalyticsEnricher,
    LinkPredictionConfig,
    LinkPredictionEnricher,
    NodeEmbeddingConfig,
    NodeEmbeddingEnricher,
    PageRankConfig,
    PageRankEnricher,
)

JsonDict = dict[str, Any]


@dataclass(frozen=True, slots=True)
class GraphMLWorkflowConfig:
    """Config-driven switchboard for graph ML capabilities."""

    enable_degree_features: bool = True
    enable_pagerank: bool = True
    enable_communities: bool = True
    enable_components: bool = True
    enable_embeddings: bool = True
    enable_link_prediction: bool = True
    pagerank: PageRankConfig = field(default_factory=PageRankConfig)
    communities: CommunityDetectionConfig = field(default_factory=CommunityDetectionConfig)
    components: ConnectedComponentsConfig = field(default_factory=ConnectedComponentsConfig)
    embeddings: NodeEmbeddingConfig = field(default_factory=NodeEmbeddingConfig)
    link_prediction: LinkPredictionConfig = field(default_factory=LinkPredictionConfig)


class GraphMLWorkflow:
    """Run graph analytics, embeddings, and link prediction in a stable order."""

    name = "graph_ml_workflow"

    def __init__(self, config: GraphMLWorkflowConfig | None = None) -> None:
        self.config = config or GraphMLWorkflowConfig()

    def run(self, graph: PropertyGraph) -> PropertyGraph:
        enriched = graph
        steps: list[str] = []
        if self.config.enable_degree_features:
            enriched = GraphAnalyticsEnricher().enrich(enriched)
            steps.append(GraphAnalyticsEnricher.name)
        if self.config.enable_pagerank:
            enriched = PageRankEnricher(self.config.pagerank).enrich(enriched)
            steps.append(PageRankEnricher.name)
        if self.config.enable_communities:
            enriched = CommunityDetectionEnricher(self.config.communities).enrich(enriched)
            steps.append(CommunityDetectionEnricher.name)
        if self.config.enable_components:
            enriched = ConnectedComponentsEnricher(self.config.components).enrich(enriched)
            steps.append(ConnectedComponentsEnricher.name)
        if self.config.enable_embeddings:
            enriched = NodeEmbeddingEnricher(self.config.embeddings).enrich(enriched)
            steps.append(NodeEmbeddingEnricher.name)
        if self.config.enable_link_prediction:
            enriched = LinkPredictionEnricher(self.config.link_prediction).enrich(enriched)
            steps.append(LinkPredictionEnricher.name)

        return enriched.model_copy(
            update={
                "metadata": {
                    **enriched.metadata,
                    "graph_ml_workflow": {
                        "steps": steps,
                        "node_count": len(enriched.nodes),
                        "edge_count": len(enriched.edges),
                    },
                }
            }
        )


def run_graph_ml_capabilities(
    graph: PropertyGraph,
    *,
    config: GraphMLWorkflowConfig | None = None,
) -> PropertyGraph:
    """One-call facade for notebooks, scripts, and tests."""

    return GraphMLWorkflow(config).run(graph)


def _dataclass_from_mapping(model: type[Any], payload: JsonDict) -> Any:
    allowed = {item.name for item in fields(model)}
    return model(**{key: value for key, value in payload.items() if key in allowed})


def graph_ml_workflow_config_from_dict(payload: JsonDict) -> GraphMLWorkflowConfig:
    """Build workflow config from a YAML/JSON dictionary."""

    root = payload.get("graph_ml", payload)
    if not isinstance(root, dict):
        raise ValueError("Graph ML config must be a mapping.")
    return GraphMLWorkflowConfig(
        enable_degree_features=bool(root.get("enable_degree_features", True)),
        enable_pagerank=bool(root.get("enable_pagerank", True)),
        enable_communities=bool(root.get("enable_communities", True)),
        enable_components=bool(root.get("enable_components", True)),
        enable_embeddings=bool(root.get("enable_embeddings", True)),
        enable_link_prediction=bool(root.get("enable_link_prediction", True)),
        pagerank=_dataclass_from_mapping(PageRankConfig, root.get("pagerank", {})),
        communities=_dataclass_from_mapping(CommunityDetectionConfig, root.get("communities", {})),
        components=_dataclass_from_mapping(ConnectedComponentsConfig, root.get("components", {})),
        embeddings=_dataclass_from_mapping(NodeEmbeddingConfig, root.get("embeddings", {})),
        link_prediction=_dataclass_from_mapping(LinkPredictionConfig, root.get("link_prediction", {})),
    )


__all__ = [
    "GraphMLWorkflow",
    "GraphMLWorkflowConfig",
    "graph_ml_workflow_config_from_dict",
    "run_graph_ml_capabilities",
]
