import numpy as np
import os
import pandas as pd
from utils import create_arg_parser, read_graph_nx
from attr_cluster import attr_cluster
from hchnsw_index import entity_embedding, create_hchnsw_index


def make_hc_index(args):
    graph, entities_df, final_relationships = read_graph_nx(args.base_path)
    community_df: pd.DataFrame = attr_cluster(
        init_graph=graph,
        final_entities=entities_df,
        final_relationships=final_relationships,
        args=args,
        max_level=args.max_level,
        min_clusters=args.min_clusters,
    )
    print("finish compute hierarchical clusters")

    c_df_save_path = os.path.join(args.output_dir, "community_df_intermediate.csv")
    community_df.to_csv(c_df_save_path, index=False)

    entities_df: pd.DataFrame = entity_embedding(entities_df, args=args)
    final_entity_df, final_community_df = create_hchnsw_index(
        community_df=community_df, entity_df=entities_df, save_path=args.output_dir
    )
    print("finish compute HC HNSW")

    f_c_save_path = os.path.join(args.output_dir, "community_df_index.csv")
    final_community_df.to_csv(f_c_save_path, index=False)

    f_e_save_path = os.path.join(args.output_dir, "entity_df_index.csv")
    final_entity_df.to_csv(f_e_save_path, index=False)
    print("finish compute HCa RAG index")


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    make_hc_index(args=args)
