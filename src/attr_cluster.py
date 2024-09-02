import networkx as nx
from utils import *
import pandas as pd
from graspologic.partition import hierarchical_leiden


def attribute_hierarchical_clustering(
    weighted_graph: nx.graph, attributes: pd.DataFrame
):
    node_to_community_map = compute_leiden_communities(
        graph=weighted_graph,
        max_cluster_size=10,
        use_lcc=True,
        seed=0xDEADBEEF,
    )

    levels = sorted(node_to_community_map.keys())
    results_by_level: dict[int, dict[str, list[str]]] = {}
    for level in levels:
        result = {}
        results_by_level[level] = result
        for node_id, raw_community_id in node_to_community_map[level].items():
            community_id = str(raw_community_id)
            if community_id not in result:
                result[community_id] = []
            result[community_id].append(node_id)
    return results_by_level


def compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph, max_cluster_size: int, use_lcc: bool, seed=0xDEADBEE
):
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed, is_weighted=True
    )
    results: dict[int, dict[str, int]] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster
    return results


if __name__ == "__main__":
    base_path = "/home/wangshu/rag/graphrag/ragtest/output/20240813-220313/artifacts"

    graph, final_entities, final_relationships = read_graph_nx(base_path)
    cos_graph = compute_distance(graph=graph)
    attribute_hierarchical_clustering(cos_graph, final_entities)
