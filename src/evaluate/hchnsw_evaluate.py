import faiss
from src.inference import *
from src.utils import create_inference_arg_parser
from src.evaluate.test_qa import load_datasets
from src.evaluate.evaluate import *
import pdb
import random
import tqdm
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity


def create_sampled_query_embeddings(args):
    dataset_path = dataset_name_path[args.dataset_name]
    qa_df = load_datasets(dataset_path)
    random.seed(args.seed)
    questions = qa_df["question"]
    _500_questions = questions.sample(n=500, random_state=args.seed)
    sampled_indices = _500_questions.index.to_numpy()
    # print(sampled_indices)
    indices_path = args.output_dir + f"/{args.dataset_name}_sampled_indices.npy"
    np.save(indices_path, sampled_indices)
    query_embedding_path = args.output_dir +f"/{args.dataset_name}_embedding.csv"
    # 1. dataset question embedding select 500 question
    query_embeddings = []
    for i in tqdm.tqdm(range(500)):
        question = _500_questions.iloc[i]
        query_embedding = openai_embedding(
                question,
                args.embedding_api_key,
                args.embedding_api_base,
                args.embedding_model,
            )
        query_embeddings.append(query_embedding)
        
    query_embeddings = pd.DataFrame(query_embeddings)
    query_embeddings.to_csv(query_embedding_path)

def load_sampled_query_embeddings(args):
    
    # embedding, ground_truth (entity_df idx)
    
    query_embedding_path = args.output_dir +f"/{args.dataset_name}_embedding.csv"
    query_embeddings = pd.read_csv(query_embedding_path)
    return query_embeddings.values

def load_sampled_query_indices(args):
    indices_path = args.output_dir + f"/{args.dataset_name}_sampled_indices.npy"
    query_indices = np.load(indices_path)
    return query_indices

def compute_ann_index(save_path):
    entity_path = os.path.join(args.output_dir, "entity_df_index.csv")
    entity_df = pd.read_csv(entity_path)
    
    entity_df["embedding"] = entity_df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
    )
    # pdb.set_trace()
    # faiss index
    dimension = len(entity_df["embedding"][0])
    print(f"Dimension: {dimension}")
    index = faiss.IndexHNSWFlat(dimension, 32)

    # get all embeddings form the dataset
    embeddings = np.stack(entity_df["embedding"].values)
    # pdb.set_trace()
    index.add(embeddings)

    # test the index
    D, I = index.search(embeddings[:5], 5)
    print(I)
    # "hnsw.index"
    # save the index
    faiss.write_index(index, save_path)
    print("Index saved")
    return

if __name__ == "__main__":
    parser = create_inference_arg_parser()
    args = parser.parse_args()
    print_args(args=args)
    
    # # # create and save sampled query embeddings and hnsw index
    # create_sampled_query_embeddings(args)
    # hnsw_index_path = args.output_dir +f"/hnsw.index"
    # compute_ann_index(hnsw_index_path)
    
    # load questions and index
    query_embeddings = load_sampled_query_embeddings(args)
    query_indices = load_sampled_query_indices(args)
    
    hchnsw_index = read_index(args.output_dir, "hchnsw.index")
    
    hnsw_index_path = os.path.join(args.output_dir, "hnsw.index")
    hnsw_index = faiss.read_index(hnsw_index_path)
    print(f"level of index: {hnsw_index.hnsw.max_level}")
    
    entity_path = os.path.join(args.output_dir, "entity_df_index.csv")
    entity_df = pd.read_csv(entity_path)
    entity_df["embedding"] = entity_df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
    )
    # pdb.set_trace()
    
    
    # 2 ground_truch each question ground entity
    # for each entity in entity-df find the entity with the highest similarity -- top 10
    '''
       [[    0  8943 11713  9266 19935]
        [    1  4591 16242     2  4575]
        [    2  4575 10377 16242     1]
        ...
        [23350  5504  5503  5514  5501]
        [23351 11641 12573 11631  7399]
        [23352  7011  8219 20744  1989]]
    '''
    embeddings = np.stack(entity_df["embedding"].values)
    
    # query_embedding -- entity_embedding

    similarities = cosine_similarity(query_embeddings[:,1:], embeddings)
    # hnsw index 采用内积（cosine similarity）作为相似度度量
    # 对每个查询嵌入，找到相似度最高的前10个实体
    top_k = 10
    top_k_indices = np.argsort(-similarities, axis=1)
    print(top_k_indices[:, :top_k])
    
    
    
    # check query embeddings
    checked_query_embeddings = []
    for query_embedding in query_embeddings:
        if isinstance(query_embedding, list):
            query_embedding = np.array(query_embedding, dtype=np.float32)
        elif not isinstance(query_embedding, np.ndarray):
            raise ValueError(
                "query_embedding is not in a valid format. It should be a list or numpy array."
            )
        if query_embedding.ndim == 1:
            query_embedding = np.expand_dims(query_embedding, axis=0)

        checked_query_embeddings.append(query_embedding[:,1:])
    checked_query_embeddings= np.array(checked_query_embeddings)
    checked_query_embeddings = np.squeeze(checked_query_embeddings, axis=1)
    print(checked_query_embeddings.shape)
    # pdb.set_trace()

    # test hchnsw recall
    topks = [1, 2, 3, 4, 5, 10]
    level = 0
    # level = 3
    # level = hnsw_index.hnsw.max_level
    print()
    for topk in topks:
        ground_truths = top_k_indices[:, :topk]
        all_results_hchnsw = []
        all_results_hnsw = []
        saerch_params = faiss.SearchParametersHCHNSW()
        saerch_params.search_level = level
        
        ###################################################
        # hchnsw recall
        t1 = time.time()
        distances_hchnsw, preds_hchnsw = hchnsw_index.search(checked_query_embeddings, k=topk, params=saerch_params)
        t2 = time.time()
        
        entity_index_list = entity_df.index_id.tolist()
        
        hchnsw_TP = 0
        
        for i, pred in tqdm.tqdm(enumerate(preds_hchnsw)):
            for j in pred:
                index = entity_index_list.index(j)
                if index in ground_truths[i]:
                    hchnsw_TP += 1
        hchnsw_recall = hchnsw_TP / (topk * len(preds_hchnsw))
        
        ###################################################
        # hnsw recall
        t3 = time.time()
        distances_hnsw, preds_hnsw = hnsw_index.search(checked_query_embeddings, k=topk)
        t4 = time.time()
        
        hnsw_TP = 0
        # pdb.set_trace()
        for i, pred in tqdm.tqdm(enumerate(preds_hnsw)):
            for j in pred:
                # index = entity_index_list.index(j)
                if j in ground_truths[i]:
                    hnsw_TP += 1
        hnsw_recall = hnsw_TP / (topk * len(preds_hnsw))
        
        # distances_flat = distances.flatten()
        # preds_flat = preds.flatten()
        # for dist, pred in zip(distances_flat, preds_flat):
        #     all_results.append((dist, pred))

        # final_results = all_results
        # # 提取最终的预测值（实体索引）

        # final_predictions = [pred for _, pred in final_results]

        # # 用于存储 top-k 的实体和社区
        # topk_entity = entity_df[entity_df["index_id"].isin(final_predictions)]
        
        print("-----------------top{}-------------------".format(topk))
        print("dataset:", args.dataset_name)
        print("hchnsw pred time:", t2-t1)
        print("hchnsw recall :", hchnsw_recall)
        print("hnsw pred time:", t4-t3)
        print("hnsw recall :", hnsw_recall)
        print()
    

    
    