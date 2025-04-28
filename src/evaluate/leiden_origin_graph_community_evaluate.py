from metric.metric import (
    calculate_clustering_entropy,
    calculate_silhouette_score,
    calculate_davies_bouldin_score,
    calculate_our_similarity,
)
from src.utils import create_inference_arg_parser, read_graph_nx
from metric.baseline import *
from metric.CODICIL import *
import pandas as pd
import time
import pdb
from src.attr_cluster import spectral_clustering_cupy


def process_topk_graph(graph_path):
    read_graph = nx.read_gml(graph_path)
    cos_graph = nx.Graph()
    # 创建一个映射，将旧节点映射到新节点
    node_mapping = {
        old_node: int(old_node) for idx, old_node in enumerate(read_graph.nodes())
    }
    for old_node, new_node in node_mapping.items():
        cos_graph.add_node(new_node, **read_graph.nodes[old_node])

    # 添加边到新图中，并保留边属性
    for old_node1, old_node2, attrs in read_graph.edges(data=True):
        new_node1 = node_mapping[old_node1]
        new_node2 = node_mapping[old_node2]
        cos_graph.add_edge(new_node1, new_node2, **attrs)

    for node in cos_graph.nodes():
        embedding_str = cos_graph.nodes[node]["embedding"]
        embedding_list = embedding_str.strip("[]").split()
        cos_graph.nodes[node]["embedding"] = np.array(embedding_list, dtype=float)
    print("finished")
    return cos_graph


# def process_cos_graph(graph_path):
#     read_graph = nx.read_gml(graph_path)
#     cos_graph = nx.Graph()
#     pdb.set_trace()
#     # 创建一个映射，将旧节点映射到新节点
#     node_mapping = {old_node: int(old_node) for idx, old_node in enumerate(read_graph.nodes())}
#     for old_node, new_node in node_mapping.items():
#         cos_graph.add_node(new_node, **read_graph.nodes[old_node])

#     # 添加边到新图中，并保留边属性
#     for old_node1, old_node2, attrs in read_graph.edges(data=True):
#         new_node1 = node_mapping[old_node1]
#         new_node2 = node_mapping[old_node2]
#         cos_graph.add_edge(new_node1, new_node2, **attrs)

#     for node in cos_graph.nodes():
#         embedding_str = cos_graph.nodes[node]['embedding']
#         embedding_list = embedding_str.strip('[]').split()
#         cos_graph.nodes[node]['embedding'] = np.array(embedding_list, dtype=float)
#     print("finished")
#     return cos_graph


if __name__ == "__main__":
    parser = create_inference_arg_parser()
    args, _ = parser.parse_known_args()
    args.max_cluster_size = 10
    print(args)

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)

    # weighted = args.weighted = False
    level = 1
    community_df = pd.DataFrame()
    # print(f"------------------weighted:{weighted}---------------------")
    print(f"Start clustering for level {level}")

    # 图增强方法
    # 1. cos graph
    # cos_graph = compute_distance(graph)  # cos graph

    # 2. topk graph
    # k=3
    # alpha=0.5
    # num_workers=64

    # print("workers num:", num_workers)
    # Gu = create_content_edges(graph, k, embedding_similarity, num_workers)
    # # 采样
    # Esample = biased_edge_sampling(Gu, graph, alpha, embedding_similarity)
    # # (any)聚类算法，这里暂时先用k-means
    # cos_graph = nx.Graph(graph)
    # cos_graph.add_edges_from(Esample) # final topk graph
    # # save topk graph
    # saved_graph = nx.Graph(cos_graph)
    # for node in saved_graph.nodes():
    #     saved_graph.nodes[node]['embedding'] = str(saved_graph.nodes[node]['embedding'])
    # nx.write_gml(saved_graph, f"/mnt/data/wangshu/hcarag/MultiHop-RAG/hcarag/graph/{args.dataset_name}_topk_graph_edges_{len(saved_graph.edges())}_k_{k}.gml")

    ########################################################################
    # 从建好的图直接读取
    ######## topk graph #######
    # /mnt/data/wangshu/hcarag/MultiHop-RAG/hcarag/graph/multihopqa_topk_graph_edges_594928_k_50.gml
    # /mnt/data/wangshu/hcarag/HotpotQA/hcarag/graph/hotpotqa_topk_graph_edges_943747_k_50.gml
    ######## cos graph #######
    # 这里cos graph算的比较快，可以直接用 cos_graph = compute_distance(graph)
    # /mnt/data/wangshu/hcarag/MultiHop-RAG/hcarag/graph/MultihopQA_cos_graph.gml
    # /mnt/data/wangshu/hcarag/HotpotQA/hcarag/graph/hotpotQA_cos_graph.gml

    # graph_path = "/mnt/data/wangshu/hcarag/MultiHop-RAG/hcarag/graph/multihopqa_topk_graph_edges_594928_k_50.gml"
    # # topk graph
    # cos_graph = process_topk_graph(graph_path)
    
    # 原始图
    cos_graph = graph

    weighted = False  # cos graph:True   topk_graph: False 

    # 使用 Leiden 算法进行聚类
    t1 = time.time()
    if args.max_cluster_size != 0:
        c_n_mapping = compute_leiden_max_size(
            cos_graph, args.max_cluster_size, args.seed, weighted
        )
    else:
        c_n_mapping = compute_leiden(cos_graph, args.seed, weighted)
    t2 = time.time()

    print("cluster num:", len(c_n_mapping))
    # evaluating
    l = len(c_n_mapping)  # 7845 4773 用于kmeans 和 spectral clustering

    cross_entropy = calculate_clustering_entropy(graph, c_n_mapping)
    silhouette_score_, calinski_harabasz_score_ = calculate_silhouette_score(
        graph, c_n_mapping
    )
    dav_score = calculate_davies_bouldin_score(graph, c_n_mapping)
    our_sim = calculate_our_similarity(graph, c_n_mapping)
    time_leiden = t2 - t1
    print("-------------Leidan---------------")
    print("silhouette_score:", silhouette_score_)
    print("calinski_harabasz_score:", calinski_harabasz_score_)
    print("davies_bouldin_score:", dav_score)
    print("similarity:", our_sim)
    print("clustering_entropy_score:", cross_entropy)
    print("time:", time_leiden)


    