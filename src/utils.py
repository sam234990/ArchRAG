import tiktoken
import argparse
from src.lm_emb import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp

# import random
import faiss
import logging
import networkx as nx
import pandas as pd
import numpy as np
import ast
import json
import re
from json_repair import repair_json
from scipy.spatial.distance import cosine
from pathlib import Path


log = logging.getLogger(__name__)


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

    print(f"Number of entities: {final_entities.shape[0]}")

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
    ent_embedding_sample = final_entities.iloc[0]["description_embedding"]
    print(f"embedding sample shape: {ent_embedding_sample.shape}")

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


def compute_new_edges_batch(
    node_batch, hnsw_index, node_list, graph, wx, m_du, search_nodes
):
    batch_new_edges = []
    for idx, u in enumerate(node_batch):
        u_emb = np.array(graph.nodes[u]["embedding"], dtype=np.float32)
        u_emb = np.expand_dims(u_emb, axis=0)  # Convert (dim,) to (1, dim)

        # Use Faiss HNSW to find the k+1 nearest neighbors (including itself)
        D, I = hnsw_index.search(u_emb, search_nodes)
        neighbor_indices = I[0]

        # Filter out the node itself from neighbors
        neighbor_indices = [
            node_idx for node_idx in neighbor_indices if node_idx != node_list.index(u)
        ]

        new_edges = []
        for v_idx in neighbor_indices:
            v = node_list[v_idx]
            if not graph.has_edge(u, v):
                v_emb = graph.nodes[v]["embedding"]
                cos_sim = embedding_similarity(u_emb.flatten(), v_emb)

                if cos_sim > wx:
                    new_edges.append((u, v, cos_sim))
        # Sort the new edges by similarity and limit to m_du edges
        new_edges = sorted(new_edges, key=lambda x: x[2], reverse=True)
        max_num = min(len(new_edges), int(m_du))
        batch_new_edges.extend(new_edges[:max_num])
        if idx % (len(node_batch) / 3) == 0:
            print(f"Processed {idx} nodes in the batch")

    return batch_new_edges  # Return only the top m_du edges


def compute_distance(
    graph, x_percentile=0.7, search_k=1.5, m_du_sacle=1, num_workers=32
):
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

    if graph.number_of_nodes() > 100:
        print("Using parallel processing to compute new edges")
        all_new_edges = []
        with mp.Pool(processes=num_workers) as pool:
            print(
                f"Using multiprocessing Pool to compute new edges with {num_workers} workers"
            )
            # Split nodes into batches
            node_batches = np.array_split(list(graph.nodes()), num_workers)
            results = list(
                pool.starmap(
                    compute_new_edges_batch,
                    [
                        (batch, hnsw_index, node_list, graph, wx, m_du, search_nodes)
                        for batch in node_batches
                    ],
                )
            )

            for new_edges in results:
                all_new_edges.extend(new_edges)

        # Add new edges to the res_graph after collecting all of them
        for u, v, weight in all_new_edges:
            res_graph.add_edge(u, v, weight=weight)
    else:
        for u in graph.nodes():
            u_emb = np.array(graph.nodes[u]["embedding"], dtype=np.float32)

            # Ensure u_emb is (1, dim) shape for Faiss
            u_emb = np.expand_dims(u_emb, axis=0)  # Convert (dim,) to (1, dim)

            # Use Faiss HNSW to find the k+1 nearest neighbors (including itself)
            D, I = hnsw_index.search(u_emb, search_nodes)
            neighbor_indices = I[0]

            # Filter out the node itself from neighbors
            neighbor_indices = [
                node_idx
                for node_idx in neighbor_indices
                if node_idx != node_list.index(u)
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


def try_parse_json_object(input: str) -> tuple[str, dict]:
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        log.info("Warning: Error decoding faulty json, attempting repair")

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```json"):
        input = input[len("```json") :]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        input = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:
            result = json.loads(input)
        except json.JSONDecodeError:
            log.exception("error loading json, json=%s", input)
            return input, {}
        else:
            if not isinstance(result, dict):
                log.exception("not expected dict type. type=%s:", type(result))
                return input, {}
            return input, result
    else:
        return input, result


def entity_embedding(
    entity_df: pd.DataFrame, args, embed_colname="embedding", num_workers=28
):

    if embed_colname in entity_df.columns:
        print(f"column name :{embed_colname} existing")
        return entity_df

    print(f"local is {args.embedding_local}")

    if args.embedding_local:
        print("Loading local embedding model")
        model, tokenizer, device = load_sbert(args.embedding_model_local)
        entity_df["embedding_context"] = entity_df.apply(
            lambda x: (
                x["name"] + " " + x["description"]
                if x["description"] is not None
                else x["name"]
            ),
            axis=1,
        )
        texts = entity_df["embedding_context"].tolist()
        entity_df[embed_colname] = text_to_embedding_batch(
            model, tokenizer, device, texts
        )
        entity_df = entity_df.drop(columns=["embedding_context"])
    else:
        # Define a function that applies openai_embedding to each description
        def compute_embedding(row):
            # Replace community_text with the description in each row
            if row["description"] is None:
                row_content = row["name"]
            else:
                row_content = row["name"] + " " + row["description"]
            return openai_embedding(
                row_content,  # Pass the description as input text
                args.embedding_api_key,
                args.embedding_api_base,
                args.embedding_model,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            rows = [row for _, row in entity_df.iterrows()]
            # 使用 tqdm 包装可迭代对象，以显示进度条
            embeddings = list(
                tqdm(
                    executor.map(compute_embedding, rows),
                    total=len(rows),
                    desc="Computing embeddings",
                )
            )

        entity_df[embed_colname] = embeddings
    return entity_df


def compute_embedding(input_a_d):
    args, description = input_a_d
    return openai_embedding(
        description,  # Pass the description as input text
        args.embedding_api_key,
        args.embedding_api_base,
        args.embedding_model,
    )


def relation_embedding(
    relation_df: pd.DataFrame,
    args,
    e_colname="description",
    embed_colname="embedding",
    num_workers=32,
):
    if embed_colname in relation_df.columns:
        print(f"column name :{embed_colname} existing")
        return relation_df

    relation_df[e_colname] = relation_df[e_colname].fillna("N")
    # 提取唯一描述
    unique_descriptions = relation_df[e_colname].unique()

    print(f"local is {args.embedding_local}")
    print(f"the number of unique {e_colname} is {len(unique_descriptions)}")
    if args.embedding_local:
        print("Loading local embedding model")
        model, tokenizer, device = load_sbert(args.embedding_model_local)

        embeddings = text_to_embedding_batch(
            model, tokenizer, device, unique_descriptions
        )
    else:
        args_list = [(args, desc) for desc in unique_descriptions]
        with mp.Pool(processes=num_workers) as pool:
            embeddings = list(
                tqdm(
                    pool.imap(compute_embedding, args_list),
                    total=len(unique_descriptions),
                    desc="Computing embeddings",
                    leave=True,  # 保持进度条显示在输出中
                )
            )
    # 创建一个映射从描述到嵌入
    embedding_mapping = dict(zip(unique_descriptions, embeddings))

    # 将嵌入合并回原 DataFrame
    relation_df[embed_colname] = relation_df[e_colname].map(embedding_mapping)

    return relation_df


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
        default="http://localhost:5000/forward",
    )

    parser.add_argument(
        "--engine",
        type=str,
        # required=True,
        default="llama3.1:8b4k",
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
        default="http://localhost:5000/forward",
        help="Base URL for the API service",
    )

    parser.add_argument(
        "--entity_second_embedding",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use second entity embedding or not",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help="Number of workers to use for parallel processing",
    )

    # log
    parser.add_argument(
        "--print_log",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to print log or not",
    )

    parser.add_argument(
        "--debug_flag",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to use debug flag or not",
    )

    return parser


def create_inference_arg_parser():
    parser = argparse.ArgumentParser(
        description="All the arguments needed for inference."
    )

    # index
    parser.add_argument(
        "--base_path",
        type=str,
        # required=True,
        default="/mnt/data/wangshu/hcarag/FB15k/KG",
        help="Base path to the directory containing the graph data.",
    )

    parser.add_argument(
        "--relationship_filename",
        type=str,
        default="relationships.csv",
        help="Filename for the relationship data.",
    )

    parser.add_argument(
        "--entity_filename",
        type=str,
        default="entity_df.csv",
        help="Filename for the entity data.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        # required=True,
        default="/mnt/data/wangshu/hcarag/FB15k/hc_index_8b",
        help="Output dir path for index",
    )

    # dataset info
    parser.add_argument(
        "--dataset_path",
        type=str,
        # required=True,
        default="/mnt/data/wangshu/hcarag/FB15k/webqa/webqa.json",
        help="dataset path for index",
    )
    parser.add_argument(
        "--inference_output_dir",
        type=str,
        # required=True,
        default="/mnt/data/wangshu/hcarag/FB15k/hc_index_8b/qa",
        help="Output dir path for dataset output",
    )

    # attr clustering parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=0xDEADBEEF,
        help="Seed for reproducibility in leiden clustering \
            (default is hex 0xDEADBEEF, input can be any valid integer)",
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
        default="http://localhost:5000/forward",
    )

    parser.add_argument(
        "--engine",
        type=str,
        # required=True,
        default="llama3.1:8b4k",
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
        default="http://localhost:5000/forward",
        help="Base URL for the API service",
    )

    parser.add_argument(
        "--entity_second_embedding",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to use second entity embedding or not",
    )

    # parallel processing
    parser.add_argument(
        "--num_workers",
        type=int,
        default=24,
        help="Number of workers to use for parallel processing",
    )

    # query and inference parameters
    parser.add_argument(
        "--strategy",
        type=str,
        default="global",
        help="Strategy for inference",
    )

    parser.add_argument(
        "--k_each_level",
        type=int,
        default=5,
        help="Number of k for each level",
    )

    parser.add_argument(
        "--k_final",
        type=int,
        default=15,
        help="Number of k for final",
    )

    parser.add_argument(
        "--all_k_inference",
        type=int,
        default=15,
        help="Number of k for all inference",
    )

    parser.add_argument(
        "--generate_strategy",
        type=str,
        default="direct",
        help="Strategy for generation",
    )

    parser.add_argument(
        "--response_type",
        type=str,
        default="QA",
        help="Type of response for generate answer",
    )

    # log
    parser.add_argument(
        "--print_log",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Whether to print log or not",
    )

    parser.add_argument(
        "--debug_flag",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to use debug flag or not",
    )

    return parser


def print_args(args, print_str="Parsed Arguments:"):
    print(print_str)
    if type(args) == argparse.Namespace:
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
    elif type(args) == dict:
        for arg, value in args.items():
            print(f"{arg}: {value}")
    

if __name__ == "__main__":
    base_path = "/home/wangshu/rag/graphrag/ragtest/output/20240813-220313/artifacts"

    graph, final_entities, final_relationships = read_graph_nx(base_path)
    cos_graph = compute_distance(graph=graph)
