from utils import *
from sklearn.cluster import KMeans, SpectralClustering
from metric import calculate_silhouette_score, calculate_davies_bouldin_score, calculate_our_similarity, calculate_clustering_entropy, evaluation
import networkx as nx
from node2vec import Node2Vec
from scan import scan
from scipy.sparse import csr_matrix


# 基于structure similarity的方法
def SCAN(graph:nx.graph, seed, epsilon=0.7, mu=2):
    c_n_mapping: dict[str, list[int]] = {}
    # 将格式转换成SCAN的输入
    rows=[]
    cols=[]
    num_nodes= len(graph.nodes())
    index = np.array([node for node in graph.nodes()])
    for u, v in graph.edges:
        rows.append(np.where(index == u)[0][0])
        cols.append(np.where(index == v)[0][0])
    data = np.ones(len(rows))
    input_graph = csr_matrix((data,(rows,cols)),shape=(num_nodes, num_nodes), dtype=np.int32)
    
    # SCAN聚类结果
    cluster_result = scan(input_graph, epsilon, mu)
    
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
    c_n_mapping[str(max_num+1)] = c_n_mapping["-1"] + c_n_mapping["-2"] + c_n_mapping["-3"]
    del c_n_mapping["-1"]
    del c_n_mapping["-2"]
    del c_n_mapping["-3"]
    
    return c_n_mapping

# 谱聚类
def spectralClustering(graph:nx.graph, seed, l):
    c_n_mapping: dict[str, list[int]] = {}
    # 转换成sklearn中SpectralClustering的输入格式
    adj = nx.to_scipy_sparse_array(graph, dtype=np.int32, format='csr')
    adj.indices = adj.indices.astype(np.int32, casting='same_kind')
    adj.indptr = adj.indptr.astype(np.int32, casting='same_kind')
    
    # 谱聚类
    # if l==0:
    #     sc = SpectralClustering(affinity='precomputed', assign_labels='discretize',random_state=seed)
    # else:
    #     sc = SpectralClustering(affinity='precomputed', assign_labels='discretize',random_state=seed, n_clusters=l)
    sc = SpectralClustering(affinity='precomputed', assign_labels='discretize',random_state=seed, n_clusters=l)
    # 获取聚类结果
    cluster_result = sc.fit_predict(adj)
    
    # 转换成c_n_mapping的格式
    index = np.array([node for node in graph.nodes()])
    for node, label in enumerate(cluster_result):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])
    return c_n_mapping

# kmeans，用文本emb做聚类
def kmeans_text_emb(graph:nx.graph, seed, l):
    c_n_mapping: dict[str, list[int]] = {}
    
    # 提取graph中存储的文本emb
    features = np.array([graph.nodes[node]['embedding'] for node in graph.nodes()])
    
    # 调用kmeans做聚类
    # if l==0:
    #     kmeans = KMeans(random_state=seed)
    # else:
    #     kmeans = KMeans(random_state=seed, n_clusters=l)
    kmeans = KMeans(random_state=seed, n_clusters=l)
    kmeans.fit(features)
    
    # 获取聚类结果
    cluster_result = kmeans.labels_
    
    # 转换成c_n_mapping的格式
    index = np.array([node for node in graph.nodes()])
    for node, label in enumerate(cluster_result):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])
    return c_n_mapping

# 用node2vec做为emb进行聚类，聚类方法采用kmeans
def node2vec_clustering(graph:nx.graph, seed, l):
    c_n_mapping: dict[str, list[int]] = {}
    ########################################
    # 这里的node2vec可能需要写成多线程的形式
    ########################################
    # 用node2vec计算structure emb
    node2vec = Node2Vec(graph, dimensions=768)
    model = node2vec.fit()
    
    # 训练模型并获取嵌入
    # embeddings 是个numpy array
    embeddings = model.wv.vectors
    
    # 调用KMeans做聚类
    # if l==0:
    #     kmeans = KMeans(random_state=seed)
    # else:
    #     kmeans = KMeans(random_state=seed, n_clusters=l)
    kmeans = KMeans(random_state=seed, n_clusters=l)
    kmeans.fit(embeddings)
    
    # 获取聚类结果
    cluster_result = kmeans.labels_
    
    # 转换成c_n_mapping的格式
    index = np.array([node for node in graph.nodes()])
    for node, label in enumerate(cluster_result):
        cluster_label = str(label)
        if cluster_label not in c_n_mapping:
            c_n_mapping[cluster_label] = []
        c_n_mapping[cluster_label].append(index[node])
    return c_n_mapping

if __name__ == "__main__":
    parser = create_inference_arg_parser()
    args = parser.parse_args()
    args.max_cluster_size = 10
    # torch.manual_seed(args.seed)
    

    graph, final_entities, final_relationships = read_graph_nx(args.base_path,args.relationship_filename, args.entity_filename)
    cos_graph = compute_distance(
        graph,
        x_percentile=args.wx_weight,
        search_k=args.search_k,
        m_du_sacle=args.m_du_scale,
    )
    
    scale = 0.9
    # l = int(len(graph.nodes()) // (args.max_cluster_size * scale))
    l =1800
    print(scale)
    print(l)
    
    
    print("-------------kmeans_text---------------")
    kmeans_text_c_n_mapping = kmeans_text_emb(graph, args.seed, l)
    kmeans_text_sil_score, kmeans_text_cal_score, kmeans_text_dav_score, kmeans_our_sim, kmeans_clustering_entropy=evaluation(graph, kmeans_text_c_n_mapping)
    
    print("silhouette_score:", kmeans_text_sil_score)
    print("calinski_harabasz_score:", kmeans_text_cal_score)
    print("davies_bouldin_score:", kmeans_text_dav_score)
    print("our_similarity:", kmeans_our_sim)
    print("clustering_entropy:", kmeans_clustering_entropy)
    # print("kmeans_clusering_kl_dic:", kmeans_clustering_entropy)
    
    
    print("-------------node2vec_clustering---------------")
    node2vec_clustering_c_n_mapping = node2vec_clustering(cos_graph, args.seed, l)
    node2vec_sil_score, node2vec_cal_score, node2vec_dav_score, node2vec_our_sim, node2vec_clustering_entropy=evaluation(graph, node2vec_clustering_c_n_mapping)
    
    print("silhouette_score:", node2vec_sil_score)
    print("calinski_harabasz_score:", node2vec_cal_score)
    print("davies_bouldin_score:", node2vec_dav_score)
    print("our_similarity:", node2vec_our_sim)
    print("clustering_entropy:", node2vec_clustering_entropy)
    
    print("-------------spectral_clustering---------------")
    spectral_clustering_c_n_mapping = spectralClustering(cos_graph, args.seed, l)
    spectral_clustering_sil_score, spectral_clustering_cal_score, spectral_clustering_dav_score, spectral_clustering_our_sim, spectral_clustering_clustering_entropy=evaluation(graph, spectral_clustering_c_n_mapping)
    
    print("silhouette_score:", spectral_clustering_sil_score)
    print("calinski_harabasz_score:", spectral_clustering_cal_score)
    print("davies_bouldin_score:", spectral_clustering_dav_score)
    print("our_similarity:", spectral_clustering_our_sim)
    print("clustering_entropy:", spectral_clustering_clustering_entropy)
    
    print("-------------SCAN---------------")
    # SCAN算法的参数
    epsilon = 0.7
    mu = 2
    SCAN_c_n_mapping = SCAN(cos_graph, args.seed, epsilon, mu)
    SCAN_sil_score, SCAN_cal_score, SCAN_dav_score, SCAN_our_sim, SCAN_clustering_entropy=evaluation(graph, SCAN_c_n_mapping)
    
    print("silhouette_score:", SCAN_sil_score)
    print("calinski_harabasz_score:", SCAN_cal_score)
    print("davies_bouldin_score:", SCAN_dav_score)
    print("our_similarity:", SCAN_our_sim)
    print("clustering_entropy:", SCAN_clustering_entropy)