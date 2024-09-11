import networkx as nx
from utils import *
import pandas as pd
from graspologic.partition import hierarchical_leiden, leiden


def attribute_hierarchical_clustering(
    weighted_graph: nx.graph, attributes: pd.DataFrame
):
    cluster_info, community_mapping = compute_leiden_communities(
        graph=weighted_graph,
        max_cluster_size=10,
        use_lcc=True,
        seed=0xDEADBEEF,
    )

    hier_tree: dict[str, str] = {}
    cluster_node_map: dict[str, list[str]] = {}
    for community_id, info in cluster_info.items():

        cluster_node_map[community_id] = info["nodes"]
        parent_community_id = str(info["parent_cluster"])
        if parent_community_id is not None:
            hier_tree[community_id] = parent_community_id

    community_level = calculate_community_levels(hier_tree)
    results_by_level: dict[int, dict[str, list[str]]] = {}

    for community_id, level in community_level.items():
        if level not in results_by_level:
            results_by_level[level] = {}
        results_by_level[level][community_id] = cluster_node_map[community_id]
    return results_by_level


def calculate_community_levels(hier_tree):
    # Initialize a dictionary to store the level of each community
    community_levels = {}

    # Function to recursively calculate the level of a community
    def calculate_level(community_id):
        # If the level is already calculated, return it
        if community_id in community_levels:
            return community_levels[community_id]

        # Find all communities that have this community_id as their parent (children nodes)
        children = [
            comm_id
            for comm_id, parent_id in hier_tree.items()
            if parent_id == community_id
        ]

        # If there are no children, it's a leaf node, so its level is 0
        if not children:
            community_levels[community_id] = 0
            return 0

        # Otherwise, calculate the level as 1 + max level of all child communities
        level = 1 + max(calculate_level(child) for child in children)

        # Store the calculated level
        community_levels[community_id] = level
        return level

    # Calculate levels for all communities, excluding None
    all_communities = set(hier_tree.keys()).union(
        set(parent_id for parent_id in hier_tree.values() if parent_id is not None)
    )
    for community_id in all_communities:
        calculate_level(community_id)

    # Before returning, remove any entry with None as a key, if it exists
    community_levels.pop("None", None)

    return community_levels


def compute_leiden_communities(
    graph: nx.Graph | nx.DiGraph, max_cluster_size: int, use_lcc: bool, seed=0xDEADBEE
):
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed, is_weighted=True
    )
    community_info: dict[str, dict] = {}

    for partition in community_mapping:
        commuinity_id = str(partition.cluster)
        if commuinity_id not in community_info:
            community_info[commuinity_id] = {
                "level": partition.level,
                "nodes": [],
                "is_final_cluster": partition.is_final_cluster,
                "parent_cluster": partition.parent_cluster,
            }
        community_info[commuinity_id]["nodes"].append(partition.node)
    return community_info, community_mapping


def compute_leiden(graph: nx.Graph, seed: int):
    # 使用 leiden 算法计算一层
    community_mapping = leiden(
        graph,
        partition_kwargs={"weight": "weight"},
        random_seed=seed,
    )

    # 将结果转换为字典格式，方便处理
    node_to_community = {node: community for node, community in community_mapping}

    return node_to_community


if __name__ == "__main__":
    base_path = "/home/wangshu/rag/graphrag/ragtest/output/20240813-220313/artifacts"

    graph, final_entities, final_relationships = read_graph_nx(base_path)
    cos_graph = compute_distance(graph=graph)
    results_by_level = attribute_hierarchical_clustering(cos_graph, final_entities)
    for level, communities in results_by_level.items():
        print(f"Create community report for level: {level} ")
        print(f"Number of communities in this level: {len(communities)}")
        for community_id, node_list in communities.items():
            print(f"Community {community_id}:")
