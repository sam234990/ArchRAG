from lm_emb import *
from utils import *
from tqdm import tqdm
import faiss
import os
import pandas as pd
import ast
from concurrent.futures import ProcessPoolExecutor


def entity_embedding(final_entities, args, embed_colname="embedding"):

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
        
    print(f"local is {args.embedding_local}")
    
    if args.embedding_local:
        print("Loading local embedding model")
        model, tokenizer, device = load_sbert(args.embedding_model_local)
        final_entities["embedding_context"] = final_entities.apply(
            lambda x: (
                x["name"] + " " + x["description"]
                if x["description"] is not None
                else x["name"]
            ),
            axis=1,
        )
        texts = final_entities["embedding_context"].tolist()
        final_entities[embed_colname] = text_to_embedding_batch(
            model, tokenizer, device, texts
        )
        final_entities = final_entities.drop(columns=["embedding_context"])
    else:
        # Initialize the progress bar
        tqdm.pandas()
        # Apply the compute_embedding function to each row in the dataframe with a progress bar
        final_entities[embed_colname] = final_entities.progress_apply(
            compute_embedding, axis=1
        )

    return final_entities


def save_index(index, index_dir: str, index_name: str):
    if not os.path.exists(index_dir):
        os.makedirs(index_dir)
    index_path = os.path.join(index_dir, index_name)
    faiss.write_index(index, index_path)

    print(f"Index saved to {index_path}")


def read_index(index_dir: str, index_name: str):
    index_path = os.path.join(index_dir, index_name)
    index = faiss.read_index(index_path)
    return index


def get_vector_hchnsw(community_df, entity_df):
    level_counts = community_df["level"].value_counts()
    print(level_counts)

    community_df["level"] = (
        pd.to_numeric(community_df["level"], errors="coerce").fillna(0).astype(int)
    )

    level_unique = community_df["level"].unique()
    if 0 in level_unique:
        community_df["level"] = community_df["level"] + 1

    # Convert 'embedding' from string representation to a list of floats
    community_df["embedding"] = community_df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
    )
    entity_df["embedding"] = entity_df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
    )

    # Embeddings from community_df and entity_df (assuming they are stored as lists or arrays)
    community_embeddings = np.vstack(community_df["embedding"].values)
    entity_embeddings = np.vstack(entity_df["embedding"].values)

    # Combine both embeddings into one array
    combined_embeddings = np.concatenate(
        (community_embeddings, entity_embeddings), axis=0
    )

    # Create a level vector: use community_df's level and set entity_df's level to 0
    community_levels = community_df["level"].values
    entity_levels = np.zeros(len(entity_df), dtype=int)

    # Combine both levels into one array
    combined_levels = np.concatenate((community_levels, entity_levels), axis=0)

    # Add index_id to community_df and entity_df
    community_df["index_id"] = np.arange(len(community_df))
    entity_df["index_id"] = np.arange(
        len(community_df), len(community_df) + len(entity_df)
    )

    return combined_embeddings, combined_levels, community_df, entity_df


def create_hchnsw_index(community_df, entity_df, save_path):

    embeddings, levels, community_df, entity_df = get_vector_hchnsw(
        community_df, entity_df
    )

    ML = int(max(levels))
    M = 32
    dim = embeddings.shape[1]
    vector_size = embeddings.shape[0]
    efSearch = 40
    efConstruction = 16
    print(dim, ML, M, 1, vector_size)

    index = faiss.IndexHCHNSWFlat(dim, ML, M, 1, vector_size)
    index.set_vector_level(levels)
    index.hchnsw.efSearch = efSearch
    index.hchnsw.efConstruction = efConstruction

    index.add(embeddings)

    save_index(index, save_path, "hchnsw.index")
    return community_df, entity_df


if __name__ == "__main__":

    parser = create_arg_parser()
    args = parser.parse_args()

    # graph, final_entities, final_relationships = read_graph_nx(args.base_path)

    # final_entities = entity_embedding(final_entities, args=args)
    # print("Entity embedding done")
    entity_output_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/entity.csv"
    # final_entities.to_csv(entity_output_path, index=False)

    entity_df = pd.read_csv(entity_output_path)
    print(entity_df.shape)
    print(entity_df.columns)
    c_df_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/communities.csv"
    community_df = pd.read_csv(c_df_path)
    print(community_df.shape)
    print(community_df.columns)
    print(type(community_df["embedding"][0]))
    create_hchnsw_index(community_df, entity_df, args.output_dir)
