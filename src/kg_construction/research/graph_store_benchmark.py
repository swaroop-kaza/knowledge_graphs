"""Graph-store benchmark fixture generation for research."""

from __future__ import annotations

from kg_construction.domain.graph_objects import PropertyEdge, PropertyGraph, PropertyNode
from kg_construction.domain.models import stable_id


def generate_graph_store_benchmark_fixture(
    *,
    node_count: int = 1000,
    edge_factor: int = 3,
) -> PropertyGraph:
    """Generate a deterministic synthetic property graph for batch-size tests."""

    nodes = [
        PropertyNode(
            node_id=f"bench_node_{index}",
            labels=["Entity", "BenchmarkNode"],
            properties={"name": f"Benchmark Node {index}", "partition": index % 10},
        )
        for index in range(node_count)
    ]
    edge_count = max(node_count * edge_factor, 0)
    edges: list[PropertyEdge] = []
    for index in range(edge_count):
        source = f"bench_node_{index % node_count}"
        target = f"bench_node_{(index * 7 + 3) % node_count}"
        if source == target:
            target = f"bench_node_{(index + 1) % node_count}"
        edges.append(
            PropertyEdge(
                edge_id=stable_id("bench_edge", index, source, target),
                source_id=source,
                target_id=target,
                type="BENCH_LINK",
                properties={"weight": (index % 100) / 100},
            )
        )
    return PropertyGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "fixture": "graph_store_batch_benchmark",
            "node_count": node_count,
            "edge_count": len(edges),
            "edge_factor": edge_factor,
        },
    )


__all__ = ["generate_graph_store_benchmark_fixture"]
