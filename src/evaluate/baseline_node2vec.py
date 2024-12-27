from utils import *
from sklearn.cluster import KMeans
from metric import calculate_silhouette_score, calculate_davies_bouldin_score, calculate_our_similarity, calculate_clustering_entropy, evaluation
import networkx as nx
from scan import scan
from scipy.sparse import csr_matrix
import tqdm

# 用node2vec做为emb进行聚类，聚类方法采用kmeans
# 这里将node2vec拆分
from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec

from utils import create_inference_arg_parser

if __name__ == "__main__":
    ##########################################################
    # 需要再python =3.6上run，需要新创建一个环境（用的包会冲突）
    ##########################################################
    parser = create_inference_arg_parser()
    args, _ = parser.parse_known_args()
    args.max_cluster_size = 10
    # torch.manual_seed(args.seed)

    # cos_graph = nx.read_gml("fb15k_cos_graph.gml")
    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
    # graph, final_entities, final_relationships = read_graph_nx(args.base_path,args.relationship_filename, args.entity_filename)

    scale = 0.9
    # l = int(len(graph.nodes()) // (args.max_cluster_size * scale))
    l = 6293
    print(scale)
    print(l)

    cos_graph = nx.read_gml("hotpotqa_topk_graph_edges_943747_k_50.gml")
    G = StellarGraph.from_networkx(cos_graph)

    print(G.info())
    rw = BiasedRandomWalk(G)

    print("generateing weighted_walks")
    weighted_walks = rw.run(
        nodes=G.nodes(),  # root nodes
        length=80,  # maximum length of a random walk
        n=10,  # number of random walks per root node
        p=0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=False,  # for weighted random walks
        seed=args.seed,  # random seed fixed for reproducibility
    )
    print("Number of random walks: {}".format(len(weighted_walks)))

    print("generateing weighted_model")
    weighted_model = Word2Vec(
        weighted_walks, vector_size=768, window=5, min_count=0, sg=1, workers=16
    )
    embeddings = weighted_model.wv.vectors

    node2vec_clustering_c_n_mapping = {}
    kmeans = KMeans(random_state=args.seed, n_clusters=l)
    kmeans.fit(embeddings)

    # 获取聚类结果
    cluster_result = kmeans.labels_

    # 转换成c_n_mapping的格式
    index = np.array([node for node in graph.nodes()])
    for node, label in enumerate(cluster_result):
        cluster_label = str(label)
        if cluster_label not in node2vec_clustering_c_n_mapping:
            node2vec_clustering_c_n_mapping[cluster_label] = []
        node2vec_clustering_c_n_mapping[cluster_label].append(index[node])
        
    print("-------------node2vec_clustering---------------")
    node2vec_sil_score, node2vec_cal_score, node2vec_dav_score, node2vec_our_sim, node2vec_clustering_entropy=evaluation(graph, node2vec_clustering_c_n_mapping)


    print("silhouette_score:", node2vec_sil_score)
    print("calinski_harabasz_score:", node2vec_cal_score)
    print("davies_bouldin_score:", node2vec_dav_score)
    print("our_similarity:", node2vec_our_sim)
    print("clustering_entropy:", node2vec_clustering_entropy)