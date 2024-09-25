import tiktoken
import argparse
import random
import faiss
import networkx as nx
import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import cosine
from pathlib import Path


docker_list = [
    "http://localhost:8876/v1",
    "http://localhost:8877/v1",
    "http://localhost:8878/v1",
    "http://localhost:8879/v1",
]


def num_tokens(text: str, token_encoder: tiktoken.Encoding | None = None) -> int:
    """Return the number of tokens in the given text."""
    if token_encoder is None:
        token_encoder = tiktoken.get_encoding("cl100k_base")
    return len(token_encoder.encode(text))  # type: ignore


def read_graph_nx(
    file_path: str,
    relationship_filename: str = "create_final_relationships.parquet",
    entity_filename: str = "create_final_entities.parquet",
) -> nx.Graph:
    """Read graph from file."""
    data_path = Path(file_path)

    # Determine the file extension
    relationship_file_path = data_path / relationship_filename
    entity_file_path = data_path / entity_filename

    if relationship_file_path.suffix in [".csv", ".txt"]:
        relationships = pd.read_csv(relationship_file_path)
    else:
        relationships = pd.read_parquet(relationship_file_path)

    if entity_file_path.suffix in [".csv", ".txt"]:
        final_entities = pd.read_csv(entity_file_path)
    else:
        final_entities = pd.read_parquet(entity_file_path)

    if "head_id" not in relationships.columns:
        name_to_id = dict(
            zip(final_entities["name"], final_entities["human_readable_id"])
        )
        relationships["head_id"] = relationships["source"].map(name_to_id)
        relationships["tail_id"] = relationships["target"].map(name_to_id)

    # Create a NetworkX graph
    graph = nx.Graph()
    for _, row in final_entities.iterrows():
        # graph.add_node(row["name"], **row.to_dict())
        graph.add_node(row["human_readable_id"], **row.to_dict())

    for _, row in relationships.iterrows():
        # graph.add_edge(row["source"], row["target"], weight=row["weight"])
        if "weight" in row:
            graph.add_edge(row["head_id"], row["tail_id"], weight=row["weight"])
        else:
            graph.add_edge(row["head_id"], row["tail_id"])

    # process embedding
    final_entities["description_embedding"] = final_entities[
        "description_embedding"
    ].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
    # add embedding to graph
    for _, row in final_entities.iterrows():
        # graph.nodes[row["name"]]["embedding"] = row["description_embedding"]
        graph.nodes[row["human_readable_id"]]["embedding"] = row[
            "description_embedding"
        ]

    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    ent_embedding_sample = final_entities.loc[0, "description_embedding"]
    print(f"embedding sample:{ent_embedding_sample.shape}")

    return graph, final_entities, relationships


def embedding_similarity(emb_1, emb_2):
    return 1 - cosine(emb_1, emb_2)


def build_faiss_hnsw_index(graph, embedding_dim):
    """
    Build a Faiss HNSW index for the graph node embeddings.

    :param graph: A NetworkX graph where each node has an 'embedding' attribute.
    :param embedding_dim: The dimension of the node embeddings.
    :return: A Faiss HNSW index with node embeddings.
    """
    # Create the HNSW index
    hnsw_index = faiss.IndexHNSWFlat(
        embedding_dim, 32
    )  # HNSW index, 32 is the default M parameter

    # Prepare node embeddings for the index
    embeddings = []
    node_list = []
    for node, data in graph.nodes(data=True):
        embeddings.append(data["embedding"])
        node_list.append(node)

    # Convert embeddings to a numpy array
    embeddings = np.array(embeddings, dtype=np.float32)

    # Add the embeddings to the HNSW index
    hnsw_index.add(embeddings)

    return hnsw_index, node_list


def compute_distance(graph, x_percentile=0.7, search_k=1.5, m_du_sacle=1):
    """
    Compute the distance between all nodes in the graph and enhance the graph by adding more edges.

    :param graph: Original graph where nodes have 'embedding' attributes.
    :param x_percentile: The percentile for the edge weight threshold.
    :param search_k: Scaling factor to determine how many nodes to search for adding edges.
    :return: A new graph with enhanced connections
    """
    res_graph = nx.Graph()

    res_graph.add_nodes_from(graph.nodes)

    # Step 1: Compute the initial graph with cosine similarity as weights.
    all_weights = []
    for n1 in graph.nodes():
        n1_emb = graph.nodes[n1]["embedding"]
        for neighbor in graph.neighbors(n1):
            if not res_graph.has_edge(n1, neighbor):
                nei_emb = graph.nodes[neighbor]["embedding"]
                cos_res = embedding_similarity(n1_emb, nei_emb)
                # 将边和余弦相似度结果添加到新图中
                res_graph.add_edge(n1, neighbor, weight=cos_res)
                all_weights.append(cos_res)

    original_edges_count = res_graph.number_of_edges()
    initial_isolates = len(list(nx.isolates(graph)))

    # Step 2: Calculate x-percentile weight wx
    wx = np.percentile(all_weights, x_percentile * 100)

    # Step 3: Calculate the average degree (m_du) of non-isolated nodes
    degrees = [deg for node, deg in res_graph.degree() if deg > 0]
    if degrees:
        m_du = int(sum(degrees) / len(degrees))
    else:
        m_du = int(np.sqrt(len(graph.nodes)) / 2)

    m_du = int(m_du * m_du_sacle)
    search_nodes = int(search_k * m_du) + 1

    print(f"Adding up to {m_du} edges to each node, searching {search_nodes} nodes")

    # Step 4: Enhance the graph by adding new edges
    embedding_dim = len(
        graph.nodes[next(iter(graph.nodes))]["embedding"]
    )  # Get embedding dimension
    hnsw_index, node_list = build_faiss_hnsw_index(graph, embedding_dim)

    for u in graph.nodes():
        u_emb = np.array(graph.nodes[u]["embedding"], dtype=np.float32)

        # Ensure u_emb is (1, dim) shape for Faiss
        u_emb = np.expand_dims(u_emb, axis=0)  # Convert (dim,) to (1, dim)

        # Use Faiss HNSW to find the k+1 nearest neighbors (including itself)
        D, I = hnsw_index.search(u_emb, search_nodes)
        neighbor_indices = I[0]

        # Filter out the node itself from neighbors
        neighbor_indices = [
            node_idx for node_idx in neighbor_indices if node_idx != node_list.index(u)
        ]

        # Calculate cosine similarity and add edges if similarity > wx
        new_edges = []
        for v_idx in neighbor_indices:
            v = node_list[v_idx]
            if not res_graph.has_edge(u, v):
                v_emb = graph.nodes[v]["embedding"]
                cos_sim = embedding_similarity(u_emb.flatten(), v_emb)

                if cos_sim > wx:
                    new_edges.append((u, v, cos_sim))

        # Sort the new edges by similarity and add up to m_du edges
        new_edges = sorted(new_edges, key=lambda x: x[2], reverse=True)
        for i in range(min(len(new_edges), int(m_du))):
            u, v, weight = new_edges[i]
            res_graph.add_edge(u, v, weight=weight)

    # Print the final graph statistics
    final_edges_count = res_graph.number_of_edges()
    total_added_edges = final_edges_count - original_edges_count

    print(f"Original edges count: {original_edges_count}", end=", ")
    print(f"New edges added: {total_added_edges}", end=", ")
    print(f"Total edges in final graph: {final_edges_count}")

    final_isolates = len(list(nx.isolates(res_graph)))
    print(f"Initial number of isolated nodes: {initial_isolates}", end=", ")
    print(f"Final number of isolated nodes: {final_isolates}")

    return res_graph


def create_arg_parser():
    parser = argparse.ArgumentParser(
        description="All the arguments needed for the project."
    )

    parser.add_argument(
        "--base_path",
        type=str,
        # required=True,
        default="/home/wangshu/rag/graphrag/ragtest/output/20240813-220313/artifacts",
        help="Base path to the directory containing the graph data.",
    )

    parser.add_argument(
        "--relationship_filename",
        type=str,
        default="create_final_relationships.parquet",
        help="Filename for the relationship data.",
    )

    parser.add_argument(
        "--entity_filename",
        type=str,
        default="create_final_entities.parquet",
        help="Filename for the entity data.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        default="/home/wangshu/rag/hier_graph_rag/test/debug_file",
        help="Output dir path for index",
    )

    parser.add_argument(
        "--wx_weight",
        type=float,
        default=0.7,
        help="The percentile for the edge weight threshold",
    )

    parser.add_argument(
        "--search_k",
        type=float,
        default=1.5,
        help="Scaling factor to determine how many nodes to search for adding edges",
    )

    parser.add_argument(
        "--m_du_scale",
        type=float,
        default=1,
        help="Scaling factor to determine the average degree of non-isolated nodes",
    )

    # attr clustering parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=0xDEADBEEF,
        help="Seed for reproducibility in leiden clustering \
            (default is hex 0xDEADBEEF, input can be any valid integer)",
    )

    parser.add_argument(
        "--max_level",
        type=int,
        default=4,
        help="Set the maximum level for attribute clustering",
    )

    parser.add_argument(
        "--min_clusters",
        type=int,
        default=5,
        help="Set the number of minimum cluster in the top level for attribute clustering",
    )

    parser.add_argument(
        "--max_cluster_size",
        type=int,
        default=15,
        help="Set the maximum size of the cluster",
    )

    # add LLM parameters
    parser.add_argument(
        "--api_key",
        type=str,
        # required=True,
        help="API key for accessing the service",
        default="ollama",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        # required=True,
        help="Base URL for the API service",
        default="http://localhost:11434/v1",
    )

    parser.add_argument(
        "--engine",
        type=str,
        # required=True,
        default="llama3.1:8b",
        help="Model engine to be used. Example values: 'gpt-3.5-turbo', 'gpt-4', 'davinci', 'curie', 'llama3'",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4000, help="Maximum tokens to generate"
    )

    parser.add_argument(
        "--max_community_tokens",
        type=int,
        default=4000,
        help="Maximum community report tokens ",
    )

    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Maximum number of retries to make in case of failure",
    )

    parser.add_argument(
        "--embedding_local",
        type=bool,
        default=False,
        help="Whether to use local embeddings or not",
    )

    parser.add_argument(
        "--embedding_model_local",
        type=str,
        default="nomic-embed-text-v1",
        help="Model engine to be used for embeddings. Example values: 'sbert', 'contriever'",
    )

    parser.add_argument(
        "--embedding_model",
        type=str,
        # required=True,
        default="nomic-embed-text",
        help="Model engine to be used for embeddings. Example values: 'text-embedding-ada-002', 'text-embedding-ada-003', 'text-embedding-ada-004'",
    )
    parser.add_argument(
        "--embedding_api_key",
        type=str,
        # required=True,
        default="ollama",
        help="API key for accessing the service",
    )
    parser.add_argument(
        "--embedding_api_base",
        type=str,
        # required=True,
        default="http://localhost:11434/v1",
        help="Base URL for the API service",
    )

    parser.add_argument(
        "--entity_second_embedding",
        type=bool,
        default=True,
        help="Whether to use second entity embedding or not",
    )

    return parser


if __name__ == "__main__":
    base_path = "/home/wangshu/rag/graphrag/ragtest/output/20240813-220313/artifacts"

    graph, final_entities, final_relationships = read_graph_nx(base_path)
    cos_graph = compute_distance(graph=graph)
