import os
import ast
import math
import numpy as np
import networkx as nx
import pandas as pd
from graspologic.partition import hierarchical_leiden, leiden
from utils import *
from community_report import community_report_batch


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


# attr cluster method 2:
# each time, we first compute the leiden community for the given cos graph
# then, we get the community report and the corresponding embedding
# finally, we reconstruct the graph with the community information
# and use this graph for the next level community computation
def compute_leiden(graph: nx.Graph, seed=0xDEADBEE):
    # 使用 leiden 算法计算一层
    community_mapping = leiden(
        graph,
        is_weighted=True,
        random_seed=seed,
    )
    c_n_mapping: dict[str, list[str]] = {}

    for node, community in community_mapping.items():
        community_id = str(community)
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(node)

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
        for community_title in node_list:
            ct_nodes = community_df.loc[
                community_df["title"] == community_title, "community_nodes"
            ]
            if not ct_nodes.empty:
                # ct_nodes 是一个 Series，取第一个值
                nodes_list = ct_nodes.iloc[0]
                
                if isinstance(nodes_list, list):
                    community_nodes.extend(nodes_list)
                else:
                    print(f"Warning: {community_title} does not have a list of community_nodes")
            else:
                print(f"Warning: {community_title} not found in community_df")

        # 去重处理
        unique_nodes = list(set(community_nodes))

        updated_c_n_mapping[str(cur_max_id)] = unique_nodes
        c_c_mapping[str(cur_max_id)] = node_list

        # 增加 cur_max_id，为下一个社区准备新的编号
        cur_max_id += 1

    return updated_c_n_mapping


def reconstruct_graph(community_df, final_relationships):
    graph = nx.Graph()

    node_community_map = {}

    # add nodes
    for idx, row in community_df.iterrows():
        # Only add the 'embedding' attribute to the graph
        embedding = row["embedding"] if "embedding" in row else None
        if isinstance(embedding, str):
            try:
                embedding = ast.literal_eval(embedding)
            except (ValueError, SyntaxError):
                print(f"Warning: Unable to parse embedding for {row['title']}")

        if not pd.notna(row["title"]):
            new_title = "CommunityID" + str(row["community_id"])
            community_df.loc[idx, "title"] = new_title  # 修改原始 DataFrame 中的 title
            row["title"] = new_title  # 更新当前迭代的 row 中的 title
        graph.add_node(row["title"], embedding=embedding)

        community_nodes = row["community_nodes"]
        if isinstance(community_nodes, str):
            try:
                community_nodes = ast.literal_eval(community_nodes)
            except (ValueError, SyntaxError):
                print(f"Warning: Unable to parse community_nodes for {row['title']}")
                community_nodes = []

        for nodes in community_nodes:
            node_community_map[nodes] = row["title"]

    for _, row in final_relationships.iterrows():

        if row["source"] in node_community_map:
            source = node_community_map[row["source"]]
        else:
            continue

        if row["target"] in node_community_map:
            target = node_community_map[row["target"]]
        else:
            continue

        if source == target:
            continue

            # If the edge already exists, increment the weight; otherwise, add the edge with weight 1
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += 1
        else:
            graph.add_edge(source, target, weight=1)

            # 检查边中是否有 NaN
    nan_edges = [
        (u, v)
        for u, v in graph.edges
        if u is None
        or v is None
        or (isinstance(u, float) and math.isnan(u))
        or (isinstance(v, float) and math.isnan(v))
    ]

    if nan_edges:
        print(f"Detected NaN edges: {nan_edges}")
    else:
        print("No NaN edges detected.")
        # 检查节点中是否有 NaN
    nan_nodes = [
        node
        for node in graph.nodes
        if node is None or (isinstance(node, float) and math.isnan(node))
    ]

    if nan_nodes:
        print(f"Detected NaN nodes: {nan_nodes}")
    else:
        print("No NaN nodes detected.")

    self_loops = list(nx.selfloop_edges(graph))
    if self_loops:
        graph.remove_edges_from(self_loops)

    return graph, community_df


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
    while level <= max_level:
        print(f"Start clustering for level {level}")

        # 计算余弦距离图
        cos_graph = compute_distance(graph)

        # 使用 Leiden 算法进行聚类
        c_n_mapping = compute_leiden(cos_graph, args.seed)

        # 如果不是第一层，需要调整 community_id
        if level > 1:
            updated_c_n_mapping = community_id_node_resize(
                c_n_mapping=c_n_mapping, community_df=community_df
            )
        else:
            updated_c_n_mapping = c_n_mapping

        print(f"Number of communities: {len(updated_c_n_mapping)}")
        c_id_list = list(updated_c_n_mapping.keys())
        print(f"Community id list: {c_id_list}")

        # 构建 level_dict，记录每个社区对应的 level
        level_dict = {
            community_id: level for community_id in updated_c_n_mapping.keys()
        }

        tmp_comunity_df_result = os.path.join(
            args.output_dir, f"tmp_community_df_{level}.csv"
        )

        if os.path.exists(tmp_comunity_df_result):
            print(
                f"File {tmp_comunity_df_result} already exists. Loading existing data."
            )
            new_community_df = pd.read_csv(tmp_comunity_df_result)
            new_community_df["embedding"] = new_community_df["embedding"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            new_community_df["community_nodes"] = new_community_df[
                "community_nodes"
            ].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else:
            print("Generating new community report.")
            new_community_df = community_report_batch(
                communities=updated_c_n_mapping,
                final_entities=final_entities,
                final_relationships=final_relationships,
                level_dict=level_dict,
                args=args,
            )
            new_community_df = pd.DataFrame(new_community_df)

        # update
        graph, new_community_df = reconstruct_graph(
            new_community_df, final_relationships
        )
        new_community_df.to_csv(tmp_comunity_df_result, index=False)
        community_df = pd.concat([community_df, new_community_df], ignore_index=True)
        level += 1

        # check for finish
        number_of_clusters = len(c_n_mapping)
        if number_of_clusters < min_clusters:
            break

    return community_df


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
    cos_graph = compute_distance(graph=graph)
    print("finish compute cos graph")
    community_df = attr_cluster(
        init_graph=graph,
        final_entities=final_entities,
        final_relationships=final_relationships,
        args=args,
        max_level=args.max_level,
        min_clusters=args.min_clusters,
    )

    output_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/communities.csv"
    community_df.to_csv(output_path, index=False)

    # results_by_level = attribute_hierarchical_clustering(cos_graph, final_entities)

    # for level, communities in results_by_level.items():
    #     print(f"Create community report for level: {level} ")
    #     print(f"Number of communities in this level: {len(communities)}")
    #     for community_id, node_list in communities.items():
    #         print(f"Community {community_id}:")
