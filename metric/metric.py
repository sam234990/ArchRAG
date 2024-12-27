import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from src.utils import embedding_similarity
import networkx as nx
import tqdm
# import torch.nn.functional as F
# import torch
# from torch.nn.parameter import Parameter


# 轮廓系数和ch系数， 轮廓系数是适用于kmeans和层次化聚类的
# 轮廓系数越大越好，取值范围为[-1,1]
# ch系数越大越好
def calculate_silhouette_score(graph:nx.Graph, c_n_mapping):
    pred_labels=[]
    node=[]
    for key, value in c_n_mapping.items():
        for a_node in value:
            node.append(graph.nodes[a_node]['embedding'])
        current_label=[int(key) for _ in range(len(value))]
        pred_labels+=current_label
    
    pred_labels=np.array(pred_labels)
    node=np.array(node)
    # 轮廓系数和ch系数
    silhouette_score_=silhouette_score(node, pred_labels)
    calinski_harabasz_score_=calinski_harabasz_score(node, pred_labels)
    return silhouette_score_, calinski_harabasz_score_

# 戴维森堡丁指数(分类适确性指标)
# 越大越好
def calculate_davies_bouldin_score(graph:nx.Graph, c_n_mapping):
    pred_labels=[]
    node=[]
    for key, value in c_n_mapping.items():
        for a_node in value:
            node.append(graph.nodes[a_node]['embedding'])
        current_label=[int(key) for _ in range(len(value))]
        pred_labels+=current_label
    
    pred_labels=np.array(pred_labels)
    node=np.array(node)
    
    davies_bouldin_score_ = davies_bouldin_score(node, pred_labels)
    return davies_bouldin_score_

# 聚类熵 
# paper: Community detection in graphs 
# authpr: Santo Fortunato
# 这里P_ij应该是概率，但是我们的方法没有这个概率值，这里用cosine similarity代替P_ij
def calculate_clustering_entropy(graph:nx.Graph, c_n_mapping):
    clustering_entropy=0
    # 遍历图中的每一条边
    for u, v in tqdm.tqdm(graph.edges):
        # 判断u,v是否在同一cluster中
        for key, value in c_n_mapping.items():
            if u in value:
                if v in value:
                    prob = embedding_similarity(graph.nodes[u]['embedding'], graph.nodes[v]['embedding'])
                    if prob<=0 or prob>=1:
                        continue
                    clustering_entropy += prob *np.log(prob) + (1 - prob) * np.log(1-prob)
                    continue
    clustering_entropy = - clustering_entropy/graph.number_of_edges()
    
    return clustering_entropy

# 基于文本相似度的评价指标
# 每个社区用内部所有节点的文本emb平均值作为社区的emb
def calculate_our_similarity(graph:nx.Graph, c_n_mapping):
    num_nodes=len(graph.nodes())
    cluster_similarities = [0 for _ in range(num_nodes)]
    
    # 计算community的emb
    for key, value in c_n_mapping.items():
        cluster_num = int(key)
        node_in_cluster = []
        for a_node in value:
            node_in_cluster.append(graph.nodes[a_node]['embedding'])
        node_emb_in_cluster = np.array(node_in_cluster)
        
        # community的emb
        cluster_node_emb = node_emb_in_cluster.mean(axis=0)
        
        # 计算 cosine similarity
        a_cluster_similarity = 0
        for i in range(len(node_in_cluster)):
            a_cluster_similarity += embedding_similarity(cluster_node_emb, node_emb_in_cluster[i])
        cluster_similarities[cluster_num] = a_cluster_similarity
    
    # 所有的simlarity球平均值
    similarity = sum(cluster_similarities)/len(cluster_similarities)
    return similarity

def evaluation(graph:nx.Graph, c_n_mapping):
    sil_score, ch_score=calculate_silhouette_score(graph, c_n_mapping)
    dav_score = calculate_davies_bouldin_score(graph, c_n_mapping)
    our_similarity = calculate_our_similarity(graph, c_n_mapping)
    cluster_entropy = calculate_clustering_entropy(graph, c_n_mapping)
    # cluster_kl_div = calculate_kl_div(graph, c_n_mapping)
    return sil_score, ch_score, dav_score, our_similarity, cluster_entropy #, cluster_kl_div