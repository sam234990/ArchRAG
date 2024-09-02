import networkx as nx
from pathlib import Path
import pandas as pd
from scipy.spatial.distance import cosine


def read_graph_nx(file_path: str):
    """Read graph from file."""
    data_path = Path(file_path)
    final_nodes = pd.read_parquet(data_path / "create_final_nodes.parquet")
    final_text_units = pd.read_parquet(data_path / "create_final_text_units.parquet")
    final_relationships = pd.read_parquet(
        data_path / "create_final_relationships.parquet"
    )

    final_entities = pd.read_parquet(data_path / "create_final_entities.parquet")

    print(final_entities.head())
    graph = nx.Graph()
    for _, row in final_entities.iterrows():
        graph.add_node(row["name"], **row.to_dict())

    for _, row in final_relationships.iterrows():
        graph.add_edge(row["source"], row["target"], weight=row["weight"])

    # 将embedding添加到图的节点属性中
    for _, row in final_entities.iterrows():
        graph.nodes[row["name"]]["embedding"] = row["description_embedding"]

    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    ent_embedding_sample = final_entities.loc[0, "description_embedding"]
    print(f"embedding sample:{ent_embedding_sample.shape}")

    return graph, final_entities, final_relationships


def embedding_distance(emb_1, emb_2):
    """Compute embedding distance between two nodes."""
    cos_similarity = 1 - cosine(emb_1, emb_2)  # 得到余弦相似度
    cos_distance = 1 - cos_similarity  # 转换为余弦距离
    return cos_distance


def compute_distance(graph):
    """Compute distance between all nodes in the graph."""
    res_graph = nx.Graph()

    res_graph.add_nodes_from(graph.nodes)

    for n1 in graph.nodes():
        n1_emb = graph.nodes[n1]["embedding"]
        for neighbor in graph.neighbors(n1):
            if not res_graph.has_edge(n1, neighbor):
                nei_emb = graph.nodes[neighbor]["embedding"]
                cos_res = embedding_distance(n1_emb, nei_emb)
                # 将边和余弦相似度结果添加到新图中
                res_graph.add_edge(n1, neighbor, weight=cos_res)
        

    return res_graph


if __name__ == "__main__":
    base_path = "/home/wangshu/rag/graphrag/ragtest/output/20240813-220313/artifacts"

    graph, final_entities, final_relationships = read_graph_nx(base_path)
    cos_graph = compute_distance(graph=graph)
    