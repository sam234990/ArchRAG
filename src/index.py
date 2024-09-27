import os
import pandas as pd
from utils import create_arg_parser, read_graph_nx, print_args
from attr_cluster import attr_cluster
from hchnsw_index import entity_embedding, create_hchnsw_index
from client_reasoning import level_summary
import time


def make_hc_index(args):
    overall_start_time = time.time()
    graph, entities_df, final_relationships = read_graph_nx(
        file_path=args.base_path,
        entity_filename=args.entity_filename,
        relationship_filename=args.relationship_filename,
    )
    print(
        f"Finished reading graph in {time.time() - overall_start_time:.2f} seconds ()"
    )

    community_df: pd.DataFrame = attr_cluster(
        init_graph=graph,
        final_entities=entities_df,
        final_relationships=final_relationships,
        args=args,
        max_level=args.max_level,
        min_clusters=args.min_clusters,
    )
    print("finish compute hierarchical clusters")
    print(
        f"Finished computing hierarchical clusters in {time.time() - overall_start_time:.2f} seconds ()"
    )

    c_df_save_path = os.path.join(args.output_dir, "community_df_intermediate.csv")
    community_df.to_csv(c_df_save_path, index=False)

    if args.entity_second_embedding:
        entities_df: pd.DataFrame = entity_embedding(entities_df, args=args)
    else:
        entities_df["embedding"] = entities_df["description_embedding"]

    final_community_df, final_entity_df = create_hchnsw_index(
        community_df=community_df, entity_df=entities_df, save_path=args.output_dir
    )
    print("finish compute HC HNSW")

    make_level_summary(community_df, args.output_dir, args)
    print(
        f"Finished making level summary in {time.time() - overall_start_time:.2f} seconds ()"
    )

    print("finish make level summary")

    f_c_save_path = os.path.join(args.output_dir, "community_df_index.csv")
    final_community_df.to_csv(f_c_save_path, index=False)

    f_e_save_path = os.path.join(args.output_dir, "entity_df_index.csv")
    final_entity_df.to_csv(f_e_save_path, index=False)
    print("finish save community_df and entity_df")
    
    entity_mapping = dict(zip(entities_df["human_readable_id"], entities_df["index_id"]))
    final_relationships['source_index_id'] = final_relationships['head_id'].map(entity_mapping)
    final_relationships['target_index_id'] = final_relationships['tail_id'].map(entity_mapping)
    
    f_r_save_path = os.path.join(args.output_dir, "relationship_df_index.csv")
    final_relationships.to_csv(f_r_save_path, index=False)
    print("finish save relationship_df")
    
    print("finish compute HCa RAG index")
    print(
        f"Finished computing HCa RAG index in {time.time() - overall_start_time:.2f} seconds ()"
    )


def make_level_summary(community_df, save_path, args):
    max_level = community_df["level"].max()
    print(f"The maximum level is: {max_level}")
    level_summary_df = level_summary(community_df, max_level, args)

    level_summary_path = os.path.join(save_path, "level_summary.csv")
    level_summary_df.to_csv(level_summary_path, index=False)


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    print_args(args)
    make_hc_index(args=args)
