import pandas as pd
import os
import faiss
from utils import *
from llm import llm_invoker
from lm_emb import openai_embedding
from hchnsw_index import read_index


def hcarag_inference(
    query_content,
    hc_index: faiss.IndexHCHNSWFlat,
    entity_df,
    community_df,
    query_paras,
    args,
):

    query_embedding = openai_embedding(
        query_content,
        args.embedding_api_key,
        args.embedding_api_base,
        args.embedding_model,
    )

    # 检查 query_embedding 是否为 list，若是则转换为 np.float32 的 numpy array
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding, dtype=np.float32)
    elif not isinstance(query_embedding, np.ndarray):
        raise ValueError(
            "query_embedding is not in a valid format. It should be a list or numpy array."
        )

    hc_level = hc_index.hchnsw.max_level
    final_k, k_per_level = load_strategy(query_paras, hc_level)

    all_results = []

    for level in range(hc_level + 1):
        saerch_params = faiss.SearchParametersHCHNSW()
        saerch_params.search_level = level
        distances, preds = hc_index.search(
            query_embedding, k=k_per_level[level], params=saerch_params
        )

        # 将每一层的结果添加到 all_results 中
        for dist, pred in zip(distances, preds):
            all_results.append((dist, pred))

    # 根据距离排序，选择距离最小的 final_k 个结果
    all_results = sorted(all_results, key=lambda x: x[0])

    # 获取最终的 top-k 结果
    final_results = all_results[:final_k]

    # 提取最终的预测值（实体索引）
    final_predictions = [pred for _, pred in final_results]

    # 用于存储提取出的 entity_context 和 community_context
    entity_context = []
    community_context = []

    # 用于存储 top-k 的实体和社区
    topk_entity = entity_df[entity_df["index_id"].isin(final_predictions)]
    topk_community = community_df[community_df["index_id"].isin(final_predictions)]

    # 提取 entity_df 中的 name 和 description 列构成 entity_context
    for _, row in topk_entity.iterrows():
        entity_context.append(f"{row['name']} - {row['description']}")

    # 提取 community_df 中的 title 和 summary 列构成 community_context
    for _, row in topk_community.iterrows():
        community_context.append(f"{row['title']} - {row['summary']}")

    pass


def load_strategy(query_paras, all_levels):

    strategy = query_paras["strategy"]
    if strategy == "global":
        k_each_level = query_paras["k_each_level"]
        k_final = query_paras["k_final"]
        k_per_level = [k_each_level] * all_levels

        return k_final, k_per_level
    elif strategy == "inference":
        k_final = query_paras["k_final"]
        #  TODO 实现 inference 策略和每层的level数值
        times = query_paras["inference_search_times"]
        all_k = times * k_final
        k_per_level = [all_k] * all_levels
        return k_final, k_per_level
    else:
        raise ValueError("Invalid strategy.")


def load_index(args):

    hc_index = read_index(args.output_dir, "hchnsw.index")

    entity_path = os.path.join(args.output_dir, "entity_df_index.csv")
    entity_df = pd.read_csv(entity_path)

    community_path = os.path.join(args.output_dir, "community_df_index.csv")
    community_df = pd.read_csv(community_path)

    return hc_index, entity_df, community_df


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    hc_index, entity_df, community_df = load_index(args)
    test_question = "What is the usage and value of TLB in an Operating System?"
    query_paras = {
        "strategy": "global",
        "k_each_level": 5,
        "k_final": 10,
        "inference_search_times": 2,
    }
