import os
import ast
import math
import numpy as np
import networkx as nx
import pandas as pd
# from graspologic.partition import hierarchical_leiden, leiden
# from sklearn.metrics.pairwise import cosine_similarity
from src.utils import read_graph_nx, embedding_similarity, create_inference_arg_parser
import multiprocessing as mp
# from community_report import community_report_batch
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.cluster import KMeans
# import pymetis
from concurrent.futures import ThreadPoolExecutor
import time


def create_content_batch_edges(graph, k, similarity_func, batch):
    new_edges = []
    for node in batch:
        node_data = graph.nodes[node]['embedding']
        similarities = []
        for other_node in graph.nodes:
            if other_node != node:
                cos_sim = similarity_func(node_data, graph.nodes[other_node]['embedding'])
                similarities.append((other_node, cos_sim))
        k_nearest_neighbors = sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
        new_edges.extend([(node, neighbor) for neighbor, _ in k_nearest_neighbors])
    return new_edges

def create_content_edges(init_graph: nx.Graph, k, similarity_func, num_workers):
    """
    使用networkx创建内容边
    参数：
    init_graph: networkx的Graph对象
    k: 每个节点的最近内容邻居数量
    similarity_func: 相似度计算函数
    num_workers: 并行任务的数量
    返回：
    Gc: 包含内容边的Graph对象
    """
    Gc = nx.Graph(init_graph)  # 创建一个空的图用于存储内容边
    nodes = list(init_graph.nodes(data=True))
    edge_num = 0
    ts_1 = time.time()

    print("Using parallel processing to compute new edges")
    all_new_edges = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        print(f"Using ThreadPoolExecutor to compute new edges with {num_workers} workers")
        node_batches = np.array_split(list(init_graph.nodes()), num_workers)
        futures = [
            executor.submit(create_content_batch_edges, init_graph, k, similarity_func, batch)
            for batch in node_batches
        ]
        for future in futures:
            all_new_edges.extend(future.result())

    for u, v in all_new_edges:
        Gc.add_edge(u, v)
        edge_num += 1
        
    ts_2=time.time()
    print("node num:", len(nodes))
    print("original edge num:", len(init_graph.edges()))
    print("added edge num:", edge_num)
    print("total edge num:", len(Gc.edges()))
    print("time:", ts_2-ts_1)

    return Gc

def calculate_topological_similarity(graph: nx.Graph, node, neighbors):
    """
    计算拓扑相似度
    参数：
    graph: networkx的Graph对象
    node: 节点索引
    neighbors: 邻居节点索引列表
    返回：
    topological_sim: 拓扑相似度向量
    """
    node_neighbors = set(graph.neighbors(node))
    topological_sim = [
        len(node_neighbors & set(graph.neighbors(neighbor))) / len(node_neighbors | set(graph.neighbors(neighbor)))
        if len(node_neighbors | set(graph.neighbors(neighbor))) != 0 else 0
        for neighbor in neighbors
    ]
    return np.array(topological_sim)  # 确保返回的是 NumPy 数组

def calculate_content_similarity(graph: nx.Graph, node, neighbors, similarity_func):
    """
    计算内容相似度
    参数：
    graph: networkx的Graph对象
    node: 节点索引
    neighbors: 邻居节点索引列表
    similarity_func: 相似度计算函数
    返回：
    content_sim: 内容相似度向量
    """
    node_embedding = graph.nodes[node]['embedding']
    content_sim = [
        similarity_func(node_embedding, graph.nodes[neighbor]['embedding'])
        for neighbor in neighbors
    ]
    return np.array(content_sim)

def normalize(arr: np.array):
    min_value = np.min(arr)
    max_value = np.max(arr)
    if max_value == 0:
        return arr
    else:
        return (arr - min_value) / (max_value - min_value)

def biased_edge_sampling(graph: nx.Graph, init_graph: nx.Graph, alpha, similarity_func, num_workers=4):
    """
    进行有偏边采样
    参数：
    graph: networkx的Graph对象
    init_graph: 初始的networkx的Graph对象
    alpha: 权重参数
    similarity_func: 相似度计算函数
    num_workers: 并行任务的数量
    返回：
    Esample: 采样后的边列表
    """
    Esample = []

    def process_node(i):
        neighbors = list(graph.neighbors(i))
        topological_sim = calculate_topological_similarity(init_graph, i, neighbors)
        topological_sim_norm = normalize(topological_sim)
        content_sim = calculate_content_similarity(graph, i, neighbors, similarity_func)
        content_sim_norm = normalize(content_sim)
        top_sim = np.array([alpha * topological_sim_norm[i] for i in range(len(topological_sim_norm))])
        cont_sim = np.array([(1 - alpha) * content_sim_norm[i] for i in range(len(content_sim_norm))])
        sim = np.add(top_sim, cont_sim)
        # sim = alpha * topological_sim_norm + (1 - alpha) * content_sim_norm
        sorted_sim_index = np.argsort(sim)[::-1]
        neighbors = np.array(neighbors)
        sorted_neighbors = neighbors[sorted_sim_index]
        return [(i, neighbor) for neighbor in sorted_neighbors[:len(sorted_neighbors)//2]]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_node, i) for i in graph.nodes()]
        for future in futures:
            Esample.extend(future.result())

    return Esample


def k_means_for_graph_clusering(Gsample: nx.Graph, l):
    # 聚类的输出
    c_n_mapping: dict[str, list[int]] = {}
    
    # k-menas聚类
    # 提取特征，以节点的度为例
    features = np.array([nx.degree(Gsample)[node] for node in Gsample.nodes()])
    # 调用sklearn做聚类
    kmeans = KMeans(n_clusters=l, random_state=0)
    kmeans.fit(features.reshape(-1, 1))  # 因为features是一维的，需要reshape
    # 获取聚类结果
    labels = kmeans.labels_
    for node, label in enumerate(labels):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(node)
    return c_n_mapping

# def PyMetis_for_graph_clusering(Gsample: nx.Graph, l):
#     # 聚类的输出
#     c_n_mapping: dict[str, list[int]] = {}
    
#     # metis的输入（邻接表）
#     adjacency_list = []
    
#     # 获取networkx中graph的邻接表
#     adjll=[(n, nbrdict) for n, nbrdict in Gsample.adjacency()]
#     # 转换成metis的输入格式（使用numpy）
#     for pair in adjll:
#         taget_node = pair[0]
#         neighbor_dict = pair[1]
#         neighbor_list = []
#         for key in neighbor_dict.keys():
#             neighbor_list.append(key)
#         adjacency_list.append(np.array(neighbor_list))
#     # 使用pymetis做聚类
#     n_cuts, membership = pymetis.part_graph(l, adjacency=adjacency_list)
    
#     # 将结果转换成c_n_mapping
#     for node, label in enumerate(membership):
#         cluster_label = str(label)
#         if cluster_label not in c_n_mapping:
#             c_n_mapping[cluster_label] = []
#         c_n_mapping[cluster_label].append(node)
#     return c_n_mapping
    

def CODICIL(
    init_graph: nx.Graph,
    # max_level=4,
    # min_clusters=5,
    k=50,
    alpha=0.5,
    num_workers=4,
):
    # 创建内容边
    print("workers num:", num_workers)
    Gu = create_content_edges(init_graph, k, embedding_similarity, num_workers)
    print(Gu)
    # 采样
    Esample = biased_edge_sampling(Gu, init_graph, alpha, embedding_similarity)
    # (any)聚类算法，这里暂时先用k-means
    Gsample = nx.Graph(init_graph)
    Gsample.add_edges_from(Esample)
    
    # c_n_mapping = PyMetis_for_graph_clusering(Gsample, l=200)
    # c_n_mapping = k_means_for_graph_clusering(Gsample, l=200)
    
    print("node num:", len(init_graph.nodes()))
    print("original edge num:", len(init_graph.edges()))
    print("added edge num:", len(Gsample.edges()) - len(init_graph.edges()))
    print("total edge num:", len(Gsample.edges()))
    
    return Gsample

