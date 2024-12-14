import pandas as pd
import os
import sys
import argparse


project_path = "/home/wangshu/rag/hier_graph_rag"
sys.path.append(os.path.abspath(project_path))
print(project_path)

from src.utils import entity_embedding

class Args:
    def __init__(self):
        self.api_key = "ollama"
        self.api_base = "http://localhost:5000/forward"
        self.embedding_local = False
        self.embedding_model_local= "nomic-embed-text"
        self.embedding_api_key="ollama"
        self.embedding_api_base="http://localhost:5000/forward"
        self.embedding_model="nomic-embed-text"
        self.embedding_num_workers=16
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--entity_path",
        type=str,
        default="/mnt/data/wangshu/hcarag/narrativeqa/data/train/0/hcarag/create_final_entities.parquet",
        help="Path to the entity file",
    )
    parser.add_argument(
        "--entity_save_path",
        type=str,
        default="/mnt/data/wangshu/hcarag/narrativeqa/data/train/0/hcarag/embeded_entities.parquet",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # input_entity_df = pd.read_parquet(args.entity_path)
    
    
    # embedding_args = Args()
    # entities_df = entity_embedding(
    #     input_entity_df,
    #     args=embedding_args,
    #     num_workers=embedding_args.embedding_num_workers,
    #     embed_colname="embedding",
    # )
    # entities_df.to_parquet(args.entity_save_path, index=False)
    # print(f"Embedding saved at {args.entity_save_path}")

    output_entity_df = pd.read_parquet(args.entity_save_path)
    output_entity_df['description_embedding'] = output_entity_df['embedding']
    output_entity_df.to_parquet(args.entity_save_path)
    print(f"finish add description_embedding column at {args.entity_save_path}")
        