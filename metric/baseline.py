from src.utils import *
from sklearn.cluster import KMeans, SpectralClustering
from metric.metric import (
    calculate_silhouette_score,
    calculate_davies_bouldin_score,
    calculate_our_similarity,
    calculate_clustering_entropy,
    evaluation,
)
import networkx as nx
from metric.scan import scan
from scipy.sparse import csr_matrix
from graspologic.partition import hierarchical_leiden, leiden
import tqdm


# 基于structure similarity的方法
# 基于structure similarity的方法
from metric.scan import scan
from metric.scan_weight import scan_weight


def SCAN(graph: nx.graph, seed, epsilon=0.7, mu=2, is_weight=False):
    c_n_mapping: dict[str, list[int]] = {}
    index = np.array([node for node in graph.nodes()])
    num_nodes = len(graph.nodes())
    # 将格式转换成SCAN的输入
    if not is_weight:
        # 无权图
        rows = []
        cols = []
        for u, v in graph.edges:
            rows.append(np.where(index == u)[0][0])
            cols.append(np.where(index == v)[0][0])
        data = np.ones(len(rows))
        input_graph = csr_matrix(
            (data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.int32
        )
        # SCAN聚类结果
        cluster_result = scan(input_graph, epsilon, mu)
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

        input_graph = csr_matrix(adj_matrix)

        cluster_result = scan_weight(input_graph, epsilon, mu)

    # 构建c_n_mapping
    max_num = -10
    for node, label in enumerate(cluster_result):
        if label > max_num:
            max_num = label
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])

    # 将离群点这种类型的点归为一类
    if "-1" in c_n_mapping and "-2" in c_n_mapping and "-3" in c_n_mapping:
        c_n_mapping[str(max_num + 1)] = (
            c_n_mapping["-1"] + c_n_mapping["-2"] + c_n_mapping["-3"]
        )
        del c_n_mapping["-1"]
        del c_n_mapping["-2"]
        del c_n_mapping["-3"]
    if "-1" in c_n_mapping and "-2" not in c_n_mapping and "-3" not in c_n_mapping:
        c_n_mapping[str(max_num + 1)] = c_n_mapping["-1"]
        del c_n_mapping["-1"]
    if "-1" not in c_n_mapping and "-2" in c_n_mapping and "-3" not in c_n_mapping:
        c_n_mapping[str(max_num + 1)] = c_n_mapping["-2"]
        del c_n_mapping["-2"]
    if "-1" not in c_n_mapping and "-2" not in c_n_mapping and "-3" in c_n_mapping:
        c_n_mapping[str(max_num + 1)] = c_n_mapping["-3"]
        del c_n_mapping["-3"]
    if "-1" in c_n_mapping and "-2" in c_n_mapping and "-3" not in c_n_mapping:
        c_n_mapping[str(max_num + 1)] = c_n_mapping["-1"] + c_n_mapping["-2"]
        del c_n_mapping["-1"]
        del c_n_mapping["-2"]
    if "-1" not in c_n_mapping and "-2" in c_n_mapping and "-3" in c_n_mapping:
        c_n_mapping[str(max_num + 1)] = c_n_mapping["-2"] + c_n_mapping["-3"]
        del c_n_mapping["-2"]
        del c_n_mapping["-3"]
    if "-1" in c_n_mapping and "-2" not in c_n_mapping and "-3" in c_n_mapping:
        c_n_mapping[str(max_num + 1)] = c_n_mapping["-1"] + c_n_mapping["-3"]
        del c_n_mapping["-1"]
        del c_n_mapping["-3"]
    return c_n_mapping


# 谱聚类
# 谱聚类
def spectralClustering(graph: nx.graph, seed, l, is_weighted):
    c_n_mapping: dict[str, list[int]] = {}
    # 转换成sklearn中SpectralClustering的输入格式
    num_nodes = len(graph.nodes())

    index = np.array([node for node in graph.nodes()])
    # 谱聚类

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
        )

    # 获取聚类结果
    cluster_result = sc.fit_predict(adj_matrix)

    # 转换成c_n_mapping的格式

    for node, label in enumerate(cluster_result):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])
    return c_n_mapping


def compute_leiden_max_size(
    graph: nx.Graph, max_cluster_size: int, seed=0xDEADBEEF, weighted=False
):
    print(weighted)
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


def compute_leiden(
    graph: nx.Graph, seed=0xDEADBEEF, weighted=False
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
