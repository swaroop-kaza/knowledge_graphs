# Graph ML Capabilities

This repository now treats graph machine learning as a downstream enrichment layer over the constructed property graph. The construction pipeline creates evidence-backed nodes and edges; graph ML then adds topology-aware signals for ranking, retrieval, clustering, and inferred-link discovery.

## Research Anchors

- GraphSAGE introduced inductive node representation learning by aggregating neighbor features instead of learning one fixed embedding per node.
- PyTorch Geometric exposes GraphSAGE through `SAGEConv`, which makes it a practical adapter target when `torch-geometric` is installed.
- PageRank is a stable baseline for graph importance scoring and is available through NetworkX and many graph databases.
- NetworkX also provides greedy modularity community detection and classic link-prediction heuristics such as resource allocation and Jaccard coefficient.
- Node2Vec and knowledge graph embedding libraries such as PyKEEN remain useful future extensions, especially when training/validation splits are available.

## Implemented Techniques

| Technique | Implementation | Best use | Output |
|---|---|---|---|
| Degree analytics | `GraphAnalyticsEnricher` | Fast structural features | `degree`, `in_degree`, `out_degree` |
| PageRank | `PageRankEnricher` | Rank important entities, systems, documents, or facts | `pagerank` on each node |
| Community detection | `CommunityDetectionEnricher` | Find graph neighborhoods for GraphRAG context and summarization | `community_id`, `community_size` |
| Connected components | `ConnectedComponentsEnricher` | Identify isolated subgraphs and ingestion gaps | `component_id`, `component_size` |
| Node embeddings | `NodeEmbeddingEnricher` | Retrieval features, resolver features, ML handoff | `node_embedding` vector |
| Link prediction | `LinkPredictionEnricher` | Suggest missing relationships for review | `metadata["link_prediction"]["predictions"]` |
| GraphSAGE adapter | `PyTorchGeometricGraphSAGEAdapter` | Optional PyG message-passing embeddings | Node vectors when `torch-geometric` is installed |

## How The Logic Works

### PageRank

- Builds a directed projection from the property graph.
- Uses NetworkX PageRank when available.
- Falls back to an internal power-iteration implementation when NetworkX is unavailable.
- Writes stable scores onto node properties.
- Useful for prioritizing graph facts during retrieval because high-score nodes are often central entities, services, people, or documents.

### Community Detection

- Builds an undirected projection from the property graph.
- Uses NetworkX greedy modularity communities when available.
- Falls back to weak components for dependency-light operation.
- Writes community IDs and sizes onto nodes.
- Useful for GraphRAG because retrieval can move from a single node to a meaningful neighborhood or community summary.

### Node Embeddings

- Supports `spectral` embeddings using normalized adjacency eigenvectors when NumPy is available and the graph is within the configured node limit.
- Supports `structural_hash` embeddings as a deterministic, dependency-light fallback using labels, node properties, and degree features.
- Supports `graphsage` through an optional PyTorch Geometric adapter.
- Stores vectors directly on nodes so later modules can use them for retrieval, entity resolution, or link prediction.

### Link Prediction

- Builds candidate pairs from non-adjacent node pairs.
- Optionally restricts candidates to pairs with common neighbors.
- Scores candidates using:
  - Resource allocation
  - Jaccard coefficient
  - Common-neighbor count
  - Preferential attachment
  - Embedding cosine similarity when node embeddings exist
- The default `hybrid` score combines topology and embedding signals.
- Predicted links are not treated as facts by default. They are written to graph metadata and marked `requires_validation`.
- If `add_edges=true`, predicted edges are materialized as `PREDICTED_LINK` edges with validation metadata.

## Notebook Call Pattern

```python
from kg_construction.application.graph_ml import (
    graph_ml_workflow_config_from_dict,
    run_graph_ml_capabilities,
)

config = graph_ml_workflow_config_from_dict({
    "graph_ml": {
        "embeddings": {"method": "spectral", "dimensions": 8},
        "link_prediction": {"method": "hybrid", "max_predictions": 10},
    }
})

enriched_graph = run_graph_ml_capabilities(property_graph, config=config)
enriched_graph.metadata["link_prediction"]["predictions"][:3]
```

## Evaluation Strategy

| Capability | Evaluation method | Production signal |
|---|---|---|
| PageRank | Stability across runs, top-k sanity checks, latency | Central nodes should match domain expectations |
| Community detection | Modularity, component/community size distribution, manual review | Communities should map to meaningful topics or operational domains |
| Node embeddings | Similarity sanity checks, retrieval metrics, downstream resolver quality | Similar nodes should retrieve together without losing provenance |
| Link prediction | Hold out known edges, then compute precision@k, recall@k, MRR, Hits@K, AUC | Suggested links should be useful enough for human review or validator promotion |
| GraphSAGE | Same as embeddings plus train/validation split quality | Only promote trained models with reproducible metadata |

## Resource And Latency Notes

- Degree features and connected components are cheap and can run on every graph construction batch.
- PageRank is iterative and depends on edge count and convergence tolerance, but it is usually a strong low-cost ranking baseline.
- Greedy modularity community detection is useful for medium graphs, but very large production graphs should move to graph database procedures or scalable Leiden/Louvain implementations.
- Spectral embeddings use dense matrices and are limited by `max_nodes_for_spectral`; larger graphs fall back to structural hashing unless a graph ML backend is configured.
- Link prediction can become quadratic in node count, so candidate generation is capped by `max_candidate_pairs`.
- GraphSAGE requires optional ML dependencies and should be evaluated with held-out edges before using it as a production signal.

## Files Added Or Updated

- `src/kg_construction/infrastructure/enrichment/graph_ml.py`
- `src/kg_construction/application/graph_ml.py`
- `configs/graph_ml.example.yaml`
- `tests/test_graph_ml_enrichment.py`
- `notebooks/06_graph_ml_capabilities.ipynb`

## Current Limitations

- The GraphSAGE adapter provides a PyTorch Geometric encoding path, but supervised or self-supervised training loops are still a future extension.
- Link prediction is candidate generation, not automatic fact promotion.
- Spectral embeddings are intended for small and medium graphs; large production graphs need sampling, GNN training, or an external graph ML system.
- PageRank and community scores should be stored with run metadata when used in production decisioning.
