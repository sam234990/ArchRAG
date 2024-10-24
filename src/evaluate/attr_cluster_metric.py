import os
import ast
import math
import numpy as np
import networkx as nx
import pandas as pd
from graspologic.partition import hierarchical_leiden, leiden
from src.utils import *
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score


def attribute_hierarchical_clustering(
    weighted_graph: nx.graph, attributes: pd.DataFrame
):
    cluster_info, community_mapping = compute_leiden_communities(
        graph=weighted_graph,
        max_cluster_size=10,
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
    graph: nx.Graph | nx.DiGraph, max_cluster_size: int, seed=0xDEADBEEF
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


# attr cluster method 2:
# each time, we first compute the leiden community for the given cos graph
# then, we get the community report and the corresponding embedding
# finally, we reconstruct the graph with the community information
# and use this graph for the next level community computation
def compute_leiden(graph: nx.Graph, seed=0xDEADBEEF) -> dict[str, list[int]]:
    # 使用 leiden 算法计算一层
    community_mapping = leiden(
        graph,
        is_weighted=True,
        random_seed=seed,
    )
    c_n_mapping: dict[str, list[int]] = {}

    for node, community in community_mapping.items():
        community_id = str(community)
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(node)

    return c_n_mapping


def compute_leiden_max_size(graph: nx.Graph, max_cluster_size: int, seed=0xDEADBEEF):
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed, is_weighted=True
    )
    c_n_mapping: dict[str, list[int]] = {}

    for partition in community_mapping:
        if not partition.is_final_cluster:
            continue
        community_id = str(partition.cluster)
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(partition.node)

    return c_n_mapping


def community_id_node_resize(
    c_n_mapping: dict[str, list[str]], community_df: pd.DataFrame
):
    if not community_df.empty:
        # 先将 community_id 字段转换为数值类型，遇到非数值的情况使用 NaN，然后忽略 NaN 值
        community_df["community_id_numeric"] = pd.to_numeric(
            community_df["community_id"], errors="coerce"
        )

        # 找到 community_id 中的最大值
        cur_max_id = community_df["community_id_numeric"].max() + 1
    else:
        cur_max_id = 0

    community_df["community_nodes"] = community_df["community_nodes"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    # 创建一个新的字典，存放更新后的 community_id 和对应的社区节点
    updated_c_n_mapping = {}
    c_c_mapping = {}

    # 遍历 c_n_mapping 中的每个社区
    for community_id, node_list in c_n_mapping.items():
        # 获取 node_list 中每个 node 对应的 title 列表
        community_nodes = []
        for community_id in node_list:
            ct_nodes = community_df.loc[
                community_df["community_id"] == community_id, "community_nodes"
            ]
            if not ct_nodes.empty:
                # ct_nodes 是一个 Series，取第一个值
                community_nodes_list = ct_nodes.iloc[0]

                if isinstance(community_nodes_list, list):
                    community_nodes.extend(community_nodes_list)
                else:
                    print(
                        f"Warning: {community_id} does not have a list of community_nodes"
                    )
            else:
                print(f"Warning: {community_id} not found in community_df")

        # 去重处理
        unique_nodes = list(set(community_nodes))

        updated_c_n_mapping[str(cur_max_id)] = unique_nodes
        c_c_mapping[str(cur_max_id)] = node_list

        # 增加 cur_max_id，为下一个社区准备新的编号
        cur_max_id += 1

    return updated_c_n_mapping


def eval_cluster_res(graph, c_n_mapping):
    pred_labels = []
    node = []
    for key, value in c_n_mapping.items():
        for a_node in value:
            # print(graph.nodes[a_node]['embedding'])
            # print(type(graph.nodes[a_node]['embedding']))
            node.append(graph.nodes[a_node]["embedding"])
        current_label = [int(key) for _ in range(len(value))]
        pred_labels += current_label
    print(len(node), len(pred_labels))

    pred_labels = np.array(pred_labels)
    node = np.array(node)

    silhouette_score_ = silhouette_score(node, pred_labels)
    calinski_harabasz_score_ = calinski_harabasz_score(node, pred_labels)
    return silhouette_score_, calinski_harabasz_score_


def attr_cluster(
    init_graph: nx.Graph,
    final_entities,
    final_relationships,
    args,
    max_level=4,
    min_clusters=5,
):
    level = 1
    graph = init_graph
    community_df = pd.DataFrame()
    print(f"Start clustering for level {level}")

    # 计算余弦距离图
    cos_graph = compute_distance(
        graph,
        x_percentile=args.wx_weight,
        search_k=args.search_k,
        m_du_sacle=args.m_du_scale,
    )

    # 使用 Leiden 算法进行聚类
    if args.max_cluster_size != 0:
        c_n_mapping = compute_leiden_max_size(
            cos_graph, args.max_cluster_size, args.seed
        )
    else:
        c_n_mapping = compute_leiden(cos_graph, args.seed)

    # 使用xx方法进行聚类

    # evaluating
    silhouette_score, calinski_harabasz_score = eval_cluster_res(
        graph, c_n_mapping
    )
    return silhouette_score, calinski_harabasz_score


# attr cluster method 2:


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    args.max_cluster_size = 0

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
    # print(graph.nodes[0]['embedding'])
    # quit()
    eval_score1, eval_score2 = community_df = attr_cluster(
        init_graph=graph,
        final_entities=final_entities,
        final_relationships=final_relationships,
        args=args,
        max_level=args.max_level,
        min_clusters=args.min_clusters,
    )
    print("silhouette_score:", eval_score1)
    print("calinski_harabasz_score:", eval_score2)

    # output_path = "/home/taotao/hier_graph_rag/datasets_io/communities.csv"
    # community_df.to_csv(output_path, index=False)

    # results_by_level = attribute_hierarchical_clustering(cos_graph, final_entities)

    # for level, communities in results_by_level.items():
    #     print(f"Create community report for level: {level} ")
    #     print(f"Number of communities in this level: {len(communities)}")
    #     for community_id, node_list in communities.items():
    #         print(f"Community {community_id}:")
