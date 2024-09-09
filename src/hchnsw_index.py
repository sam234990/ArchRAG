from lm_emb import *
from utils import *
from tqdm import tqdm


def entity_embedding(final_entities, args):
    # Initialize the progress bar
    tqdm.pandas()

    # Define a function that applies openai_embedding to each description
    def compute_embedding(row):
        # Replace community_text with the description in each row
        return openai_embedding(
            row["description"],  # Pass the description as input text
            args.embedding_api_key,
            args.embedding_api_base,
            args.embedding_model,
        )

    # Apply the compute_embedding function to each row in the dataframe with a progress bar
    final_entities["embedding"] = final_entities.progress_apply(
        compute_embedding, axis=1
    )

    return final_entities


if __name__ == "__main__":

    parser = create_arg_parser()
    args = parser.parse_args()

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)

    final_entities = entity_embedding(final_entities, args=args)
    print("Entity embedding done")
    entity_output_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/entity.csv"
    final_entities.to_csv(entity_output_path, index=False)
