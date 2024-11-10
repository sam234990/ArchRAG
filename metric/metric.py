import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from utils import embedding_similarity
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

# 使用Attributed Graph Clustering: A Deep Attentional Embedding Approach 文章中的计算p,q的方式计算kl散度
# https://github.com/Tiger101010/DAEGC/blob/main/DAEGC/daegc.py line 41-49

def get_Q(cluster_layer, z, v):
    # q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - cluster_layer, 2), 2) / v)
    # q = q.pow((v + 1.0) / 2.0)
    # q = (q.t() / torch.sum(q, 1)).t()
    distances = np.sum(np.square(z - cluster_layer), axis=1)
    # 计算q
    q = 1.0 / (1.0 + distances / v)
    q = np.power(q, (v + 1.0) / 2.0)
    # 归一化q
    q = q / np.sum(q, axis=1)[:, np.newaxis]
    return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def calculate_kl_div(graph:nx.Graph, c_n_mapping):
    
    # 簇中心, 用每个community的每个节点的均值代替
    cluster_centers={}
    v=1
    for key, value in c_n_mapping.items():
        node_in_cluster = []
        for a_node in value:
            node_in_cluster.append(graph.nodes[a_node]['embedding'])
        node_emb_in_cluster = np.array(node_in_cluster)
        
        # community的emb
        cluster_node_emb = node_emb_in_cluster.mean(axis=0)
        cluster_centers[key] = cluster_node_emb
    
    # 计算q
    qs: dict[str, list[int]] = {}
    for key, value in c_n_mapping.items():
        qk = []
        for z in value:
            a_distance = np.sum(np.square(graph.nodes[z]['embedding'] - cluster_centers[key]), axis=1)
            target_q = 1.0 / (1.0 + a_distance / v)
            target_q = np.power(target_q, (v + 1.0) / 2.0)
            distances = np.array([0 for _ in range(len(graph.nodes[z]['embedding']))])
            for s in cluster_centers:
                distances += 1.0 / (1.0 + np.sum(np.square(graph.nodes[z]['embedding'] - cluster_centers[s]), axis=1))
            
            q = target_q / np.sum(distances, axis=1)
            qk.append(q)
        qs[key] = qk
    
    # 计算p
    ps: dict[str, list[int]] = {}
    total_weight = np.array([0 for _ in range(len(graph.nodes[z]['embedding']))])
    for key, value in qs.item():
        a_qs = np.array(value)
        pk=[]
        for q in a_qs:
            weight = q**2 / np.sum(a_qs, axis=1)
            total_weight += weight
            pk.append(weight)
        ps[key] = pk

    for key, value in ps.item():
        a_qs = np.array(value)
        new_pk=[]
        for q in a_qs:
            final_q = q / total_weight
            new_pk.append(final_q)
        ps[key] = new_pk
    # 计算kl散度
    kl_div = []
    for key, value in qs.item():
        for i in range(len(value)):
            a_kl_arr = ps[key][i] * np.log2(ps[key][i] / qs[key][i])
            kl_div.append(np.sum(a_kl_arr))
    #############################################
    # 这里的指标需要考虑好最后需要时这么算吗
    #############################################
    return sum(kl_div) / len(kl_div)

def evaluation(graph:nx.Graph, c_n_mapping):
    sil_score, ch_score=calculate_silhouette_score(graph, c_n_mapping)
    dav_score = calculate_davies_bouldin_score(graph, c_n_mapping)
    our_similarity = calculate_our_similarity(graph, c_n_mapping)
    cluster_entropy = calculate_clustering_entropy(graph, c_n_mapping)
    # cluster_kl_div = calculate_kl_div(graph, c_n_mapping)
    return sil_score, ch_score, dav_score, our_similarity, cluster_entropy #, cluster_kl_div