import os
import ast
import math
import numpy as np
import networkx as nx
import pandas as pd
import tqdm
import cupy as cp
import scipy.sparse as sp
from graspologic.partition import hierarchical_leiden, leiden
from sklearn.cluster import SpectralClustering
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from cupy.sparse.linalg import eigsh


from src.utils import *
from src.community_report import community_report_batch


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
def compute_leiden(
    graph: nx.Graph, seed=0xDEADBEEF, weighted=True
) -> dict[str, list[int]]:
    # 使用 leiden 算法计算一层
    community_mapping = leiden(
        graph,
        is_weighted=weighted,
        random_seed=seed,
    )
    c_n_mapping: dict[str, list[int]] = {}

    for node, community in community_mapping.items():
        community_id = str(community)
        if community_id not in c_n_mapping:
            c_n_mapping[community_id] = []
        c_n_mapping[community_id].append(node)

    return c_n_mapping


def compute_leiden_max_size(
    graph: nx.Graph, max_cluster_size: int, seed=0xDEADBEEF, weighted=True
):
    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed, is_weighted=weighted
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


def spectralClustering(graph: nx.graph, seed, l, is_weighted):
    c_n_mapping: dict[str, list[int]] = {}
    # 转换成sklearn中SpectralClustering的输入格式
    num_nodes = len(graph.nodes())

    index = np.array([node for node in graph.nodes()])
    # 谱聚类

    num_worker_spec = 32
    if not is_weighted:
        # 非加权图
        adj_matrix = nx.to_scipy_sparse_array(graph, dtype=np.int32, format="csr")
        adj_matrix.indices = adj_matrix.indices.astype(np.int32, casting="same_kind")
        adj_matrix.indptr = adj_matrix.indptr.astype(np.int32, casting="same_kind")
        sc = SpectralClustering(
            affinity="precomputed",
            assign_labels="discretize",
            random_state=seed,
            n_clusters=l,
            n_jobs=num_worker_spec,
            verbose=True,
        )
    else:
        # 加权图
        adj_matrix = [[0 for _ in range(num_nodes)] for _ in range(num_nodes)]
        for node in tqdm.tqdm(graph.adj):
            indice_node = np.where(index == node)[0][0]
            for neighbor in graph.adj[node]:
                indice_neighbor = np.where(index == neighbor)[0][0]
                adj_matrix[indice_node][indice_neighbor] = graph.adj[node][neighbor][
                    "weight"
                ]

        adj_matrix = csr_matrix(adj_matrix)
        sc = SpectralClustering(
            affinity="precomputed",
            assign_labels="discretize",
            random_state=seed,
            n_clusters=l,
            n_jobs=num_worker_spec,
            verbose=True,
        )

    print("Start fit predict")

    # 获取聚类结果
    cluster_result = sc.fit_predict(adj_matrix)

    # 转换成c_n_mapping的格式
    for node, label in enumerate(cluster_result):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])
    return c_n_mapping


def spectral_clustering_cupy(graph: nx.graph, seed, number_cluster, is_weighted):
    if number_cluster < 1:
        number_cluster = 1

    c_n_mapping: dict[str, list[int]] = {}
    index = np.array([node for node in graph.nodes()])
    if is_weighted:
        # 如果是加权图，使用边的权重
        adj_matrix = nx.adjacency_matrix(graph, weight="weight")
    else:
        # 如果是无权图，权重为1
        adj_matrix = nx.adjacency_matrix(graph)

    # transform the adjacency to sparse matrix
    adj_matrix = adj_matrix.astype(float)
    adj_matrix = sp.csr_matrix(adj_matrix)

    print("finish compute_laplacian_matrix")

    # 将邻接矩阵转换为CuPy的稀疏矩阵
    adj_matrix_gpu = cp.sparse.csr_matrix(adj_matrix)

    # 计算度矩阵D
    degrees = cp.array(adj_matrix_gpu.sum(axis=1)).flatten()  # 行和作为度数
    degree_matrix = cp.sparse.diags(degrees)

    # 计算拉普拉斯矩阵L = D - A
    laplacian_matrix = degree_matrix - adj_matrix_gpu

    top_k_eig = min(number_cluster, 200)
    try:
        # 使用CuPy计算拉普拉斯矩阵的前k个特征值和特征向量
        eigvals, eigvecs = eigsh(laplacian_matrix, k=top_k_eig, which="LA")
    except Exception as e:
        print(f"Error during eigendecomposition: {e}")
    finally:
        # release cuda memory
        cp.get_default_memory_pool().free_all_blocks()
    print("finish cp.linalg.eigh, number of eigvals:", len(eigvals))
    print("eigvecs shape:", eigvecs.shape)

    # 选择前k个最小特征值对应的特征向量
    eigvecs_selected = eigvecs[:, :top_k_eig]

    # 对特征向量进行k-means聚类
    kmeans = KMeans(n_clusters=number_cluster, random_state=seed)
    clusters = kmeans.fit_predict(eigvecs_selected.get())
        
    for node, label in enumerate(clusters):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])

    print("number of clusters:", len(set(c_n_mapping)))
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
    for _, node_list in c_n_mapping.items():
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

    return updated_c_n_mapping, c_c_mapping


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
                print(
                    f"Warning: Unable to parse embedding for {row['title']} - {row['community_id']}"
                )

        graph.add_node(row["community_id"], embedding=embedding)

        community_nodes = row["community_nodes"]
        if isinstance(community_nodes, str):
            try:
                community_nodes = ast.literal_eval(community_nodes)
            except (ValueError, SyntaxError):
                print(
                    f"Warning: Unable to parse community_nodes for {row['title']} - {row['community_id']}"
                )
                community_nodes = []

        for nodes in community_nodes:
            node_community_map[nodes] = row["community_id"]

    for _, row in final_relationships.iterrows():
        if row["head_id"] in node_community_map:
            source = node_community_map[row["head_id"]]
        else:
            continue

        if row["tail_id"] in node_community_map:
            target = node_community_map[row["tail_id"]]
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
    all_token = 0
    while level <= max_level:
        print(f"Start clustering for level {level}")

        # 1. augment graph and compute weight
        if args.augment_graph is True:
            # 计算余弦距离图
            cos_graph = compute_distance(
                graph,
                x_percentile=args.wx_weight,
                search_k=args.search_k,
                m_du_sacle=args.m_du_scale,
            )
        else:
            cos_graph = graph

        # 2. clustering
        if args.augment_graph is True:
            if args.cluster_method == "weighted_leiden":
                c_n_mapping = compute_leiden_max_size(
                    cos_graph, args.max_cluster_size, args.seed
                )
            else:
                num_c = int(cos_graph.number_of_nodes() / (args.max_cluster_size))
                c_n_mapping = spectral_clustering_cupy(
                    cos_graph, args.seed, num_c, True
                )
        else:
            if args.cluster_method == "weighted_leiden":
                c_n_mapping = compute_leiden_max_size(
                    cos_graph, args.max_cluster_size, args.seed, False
                )
            else:
                num_c = int(cos_graph.number_of_nodes() / (args.max_cluster_size))
                c_n_mapping = spectral_clustering_cupy(
                    cos_graph, args.seed, num_c, False
                )

        # # 使用 Leiden 算法进行聚类
        # if args.max_cluster_size != 0:

        # else:
        #     c_n_mapping = compute_leiden(cos_graph, args.seed)

        # check for finish
        number_of_clusters = len(c_n_mapping)
        if number_of_clusters < min_clusters:
            break

        # 如果不是第一层，需要调整 community_id
        if level > 1:
            updated_c_n_mapping, c_c_mapping = community_id_node_resize(
                c_n_mapping=c_n_mapping, community_df=community_df
            )
        else:
            updated_c_n_mapping = c_n_mapping
            c_c_mapping = {}

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

        tmp_comunity_df_error = os.path.join(
            args.output_dir, f"tmp_community_df_{level}_error.csv"
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
            new_community_df, cur_token = community_report_batch(
                communities=updated_c_n_mapping,
                c_c_mapping=c_c_mapping,
                final_entities=final_entities,
                final_relationships=final_relationships,
                exist_community_df=community_df,
                level_dict=level_dict,
                error_save_path=tmp_comunity_df_error,
                args=args,
            )
            all_token += cur_token
            print(f"cur token usage for current level: {cur_token}")

        # update
        graph, new_community_df = reconstruct_graph(
            new_community_df, final_relationships
        )
        new_community_df.to_csv(tmp_comunity_df_result, index=False)
        community_df = pd.concat([community_df, new_community_df], ignore_index=True)
        level += 1

    return community_df, all_token


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    print_args(args)

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
    community_df, all_token = attr_cluster(
        init_graph=graph,
        final_entities=final_entities,
        final_relationships=final_relationships,
        args=args,
        max_level=args.max_level,
        min_clusters=args.min_clusters,
    )

    output_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/communities.csv"
    community_df.to_csv(output_path, index=False)
    print(f"Community report saved to {output_path}")
    print(f"Total token usage: {all_token}")

    # results_by_level = attribute_hierarchical_clustering(cos_graph, final_entities)

    # for level, communities in results_by_level.items():
    #     print(f"Create community report for level: {level} ")
    #     print(f"Number of communities in this level: {len(communities)}")
    #     for community_id, node_list in communities.items():
    #         print(f"Community {community_id}:")
