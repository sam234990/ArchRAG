import sys
import os

# 添加 src 文件夹到 sys.path
src_path = "/home/wangshu/rag/hier_graph_rag"
sys.path.append(os.path.abspath(src_path))

from src.lm_emb import openai_embedding
from src.evaluate.evaluate import *
import multiprocessing as mp
import pandas as pd
import numpy as np
import argparse
import faiss

class Args:
    def __init__(self):
        self.api_key = "ollama"
        self.api_base = "http://localhost:5000/forward"
        self.embedding_local = False
        self.embedding_model_local = "nomic-embed-text"
        self.embedding_api_key = "ollama"
        self.embedding_api_base = "http://localhost:5000/forward"
        self.embedding_model = "nomic-embed-text"
        self.embedding_num_workers = 16


def embedder_worker(dataset_part: pd.DataFrame, process_id, args):
    results = []
    all_token = 0
    for i in range(len(dataset_part)):
        data = dataset_part.iloc[i]
        embed_content = data["title"] + " " + data["content"]
        embedding = openai_embedding(
            embed_content,
            args.embedding_api_key,
            args.embedding_api_base,
            args.embedding_model,
        )

        tmp_res = {
            "id": data["id"],
            "title": data["title"],
            "content": data["content"],
            "embedding": embedding,
        }
        results.append(tmp_res)
        if i % (len(dataset_part) / 3) == 0:
            print(f"Process {process_id}: Processing {i}/{len(dataset_part)}")

    return results


def process_corpus_embedding(corpus_data: pd.DataFrame):
    args = Args()
    # parallel processing
    num_workers = args.embedding_num_workers
    dataset_parts = np.array_split(corpus_data, num_workers)
    pool = mp.Pool(num_workers)
    results = pool.starmap(
        embedder_worker,
        [(dataset_part, i, args) for i, dataset_part in enumerate(dataset_parts)],
    )
    pool.close()
    pool.join()
    res_df = pd.DataFrame([item for sublist in results for item in sublist])
    print(res_df.shape)
    print("Embedding finished")
    return res_df
    
def compute_ann_index(embed_dataset, save_path):
    
    # faiss index
    dimension = len(embed_dataset["embedding"][0])
    print(f"Dimension: {dimension}")
    index = faiss.IndexHNSWFlat(dimension, 32)
    
    # get all embeddings form the dataset
    embeddings = np.stack(embed_dataset["embedding"].values)
    index.add(embeddings)
    
    # test the index
    D, I = index.search(embeddings[:5], 5)
    print(I)
        
    # save the index
    faiss.write_index(index, save_path)
    print("Index saved")
    return
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()
    dataset_path = args.dataset_path
    save_path = args.save_path
    
    # dataset_path = "/mnt/data/wangshu/hcarag/HotpotQA/dataset/rag_hotpotqa_corpus.json"
    # save_path = "/mnt/data/wangshu/hcarag/HotpotQA/dataset/rag_hotpotqa_corpus.index"
    
    # dataset_path = "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_corpus.json"
    # save_path = "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_corpus.index"

    dataset = pd.read_json(dataset_path, lines=True)
    print(dataset.shape)
    
    embed_dataset = process_corpus_embedding(dataset)

    compute_ann_index(embed_dataset, save_path)
    