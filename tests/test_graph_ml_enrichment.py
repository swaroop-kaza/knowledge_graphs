from __future__ import annotations

import unittest

from kg_construction.application.graph_ml import (
    GraphMLWorkflow,
    graph_ml_workflow_config_from_dict,
    run_graph_ml_capabilities,
)
from kg_construction.application.registry import research_registry
from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.infrastructure.enrichment import (
    CommunityDetectionEnricher,
    ConnectedComponentsEnricher,
    LinkPredictionConfig,
    LinkPredictionEnricher,
    NodeEmbeddingConfig,
    NodeEmbeddingEnricher,
    PageRankEnricher,
    PyTorchGeometricGraphSAGEAdapter,
)
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured


def sample_property_graph() -> PropertyGraph:
    nodes = [
        PropertyNode(node_id="svc:recommendation", labels=["Entity", "System"], properties={"name": "Recommendation API"}),
        PropertyNode(node_id="incident:1042", labels=["Entity", "Incident"], properties={"name": "INCIDENT-1042"}),
        PropertyNode(node_id="team:platform", labels=["Entity", "Team"], properties={"name": "Platform Team"}),
        PropertyNode(node_id="svc:gateway", labels=["Entity", "System"], properties={"name": "Gateway API"}),
    ]
    edges = [
        PropertyEdge(
            edge_id="e1",
            source_id="incident:1042",
            target_id="svc:recommendation",
            type="AFFECTED",
        ),
        PropertyEdge(
            edge_id="e2",
            source_id="team:platform",
            target_id="svc:recommendation",
            type="OWNS",
        ),
        PropertyEdge(
            edge_id="e3",
            source_id="svc:gateway",
            target_id="svc:recommendation",
            type="CALLS",
        ),
    ]
    return PropertyGraph(nodes=nodes, edges=edges, metadata={"fixture": "graph_ml"})


class GraphMLEnrichmentTests(unittest.TestCase):
    def test_pagerank_communities_components_and_embeddings_enrich_nodes(self) -> None:
        graph = sample_property_graph()
        graph = PageRankEnricher().enrich(graph)
        graph = CommunityDetectionEnricher().enrich(graph)
        graph = ConnectedComponentsEnricher().enrich(graph)
        graph = NodeEmbeddingEnricher(NodeEmbeddingConfig(dimensions=4)).enrich(graph)

        pagerank_sum = sum(node.properties["pagerank"] for node in graph.nodes)
        self.assertAlmostEqual(pagerank_sum, 1.0, places=5)
        self.assertTrue(all("community_id" in node.properties for node in graph.nodes))
        self.assertTrue(all(node.properties["component_id"] == 0 for node in graph.nodes))
        self.assertTrue(all(len(node.properties["node_embedding"]) == 4 for node in graph.nodes))
        self.assertEqual(graph.metadata["node_embeddings"]["dimensions"], 4)

    def test_link_prediction_scores_missing_edges_without_overwriting_facts(self) -> None:
        graph = NodeEmbeddingEnricher(NodeEmbeddingConfig(dimensions=4)).enrich(sample_property_graph())
        enriched = LinkPredictionEnricher(
            LinkPredictionConfig(method="hybrid", max_predictions=3)
        ).enrich(graph)

        predictions = enriched.metadata["link_prediction"]["predictions"]
        self.assertTrue(predictions)
        self.assertEqual(len(enriched.edges), len(graph.edges))
        self.assertTrue(all(item["requires_validation"] for item in predictions))
        existing_pairs = {
            tuple(sorted((edge.source_id, edge.target_id)))
            for edge in graph.edges
        }
        predicted_pairs = {
            tuple(sorted((item["source_id"], item["target_id"])))
            for item in predictions
        }
        self.assertTrue(predicted_pairs.isdisjoint(existing_pairs))

    def test_link_prediction_can_materialize_predicted_edges_when_requested(self) -> None:
        enriched = LinkPredictionEnricher(
            LinkPredictionConfig(method="resource_allocation", max_predictions=2, add_edges=True)
        ).enrich(sample_property_graph())

        predicted_edges = [edge for edge in enriched.edges if edge.type == "PREDICTED_LINK"]
        self.assertTrue(predicted_edges)
        self.assertTrue(all(edge.properties["requires_validation"] for edge in predicted_edges))

    def test_workflow_and_config_parser_run_all_default_graph_ml_steps(self) -> None:
        config = graph_ml_workflow_config_from_dict(
            {
                "graph_ml": {
                    "embeddings": {"dimensions": 5, "method": "structural_hash"},
                    "link_prediction": {"max_predictions": 2},
                }
            }
        )
        graph = GraphMLWorkflow(config).run(sample_property_graph())
        facade_graph = run_graph_ml_capabilities(sample_property_graph(), config=config)

        self.assertEqual(graph.metadata["graph_ml_workflow"]["steps"][-1], "link_prediction_enricher")
        self.assertEqual(facade_graph.metadata["node_embeddings"]["dimensions"], 5)
        self.assertIn("link_prediction", facade_graph.metadata)

    def test_graphsage_adapter_is_optional_and_explicit(self) -> None:
        adapter = PyTorchGeometricGraphSAGEAdapter()
        try:
            embeddings = adapter.encode(sample_property_graph())
        except AdapterNotConfigured as exc:
            self.assertIn("torch-geometric", str(exc))
        else:
            self.assertEqual(set(embeddings), {node.node_id for node in sample_property_graph().nodes})

    def test_registry_marks_graph_ml_capabilities_as_implemented(self) -> None:
        registry = research_registry()
        self.assertEqual(registry.get("pagerank_enricher").status, "implemented")
        self.assertEqual(registry.get("community_detection_enricher").status, "implemented")
        self.assertEqual(registry.get("connected_components_enricher").status, "implemented")
        self.assertEqual(registry.get("node_embedding_enricher").status, "implemented")
        self.assertEqual(registry.get("link_prediction_enricher").status, "implemented")
        self.assertEqual(registry.get("graphsage_pyg_adapter").status, "adapter")


if __name__ == "__main__":
    unittest.main()
