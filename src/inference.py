import pandas as pd
import os
import faiss
from utils import *
from llm import llm_invoker
from lm_emb import openai_embedding
from hchnsw_index import read_index
from client_reasoning import *


def hcarag(
    query_content,
    hc_index: faiss.IndexHCHNSWFlat,
    entity_df,
    community_df,
    query_paras,
    args,
):
    entity_context, community_context = hcarag_retrieval(
        query_content, hc_index, entity_df, community_df, query_paras, args
    )

    response_report = hcarag_inference(
        entity_context, community_context, query_content, args.max_retries, args
    )
    return response_report


def hcarag_retrieval(
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

    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)

    hc_level = hc_index.hchnsw.max_level
    final_k, k_per_level = load_strategy(query_paras, hc_level + 1)

    all_results = []

    for level in range(hc_level + 1):
        saerch_params = faiss.SearchParametersHCHNSW()
        saerch_params.search_level = level
        distances, preds = hc_index.search(
            query_embedding, k=k_per_level[level], params=saerch_params
        )

        # 将 numpy 数组展平成一维，并添加到 all_results 中
        distances_flat = distances.flatten()
        preds_flat = preds.flatten()
        
        for dist, pred in zip(distances_flat, preds_flat):
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
    for idx, row in topk_entity.iterrows():
        entity_context.append(f"{idx}, {row['name']}, {row['description']}")

    # 提取 community_df 中的 title 和 summary 列构成 community_context
    for idx, row in topk_community.iterrows():
        community_context.append(f"{idx}, {row['title']}, {row['summary']}")

    return entity_context, community_context


def hcarag_inference(
    entity_context,
    community_context,
    query,
    max_retries,
    args,
):

    content = prep_infer_content(
        entity_context=entity_context,
        community_context=community_context,
        query=query,
        max_tokens=args.max_tokens,
    )

    retries = 0
    success = False
    response = ""

    while not success and retries < max_retries:
        raw_result = llm_invoker(content, args, max_tokens=args.max_tokens, json=True)
        try:
            output = json.loads(raw_result)
            if "response" in output:
                response = output["response"]
                break

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        retries += 1

    response_report = {"response": response, "raw_result": raw_result}

    return response_report


def load_strategy(query_paras, number_levels):

    strategy = query_paras["strategy"]
    if strategy == "global":
        k_each_level = query_paras["k_each_level"]
        k_final = query_paras["k_final"]
        k_per_level = [k_each_level] * number_levels

        return k_final, k_per_level
    elif strategy == "inference":
        k_final = query_paras["k_final"]
        #  TODO 实现 inference 策略和每层的level数值
        times = query_paras["inference_search_times"]
        all_k = times * k_final
        k_per_level = [all_k] * number_levels
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
    response = hcarag(
        test_question, hc_index, entity_df, community_df, query_paras, args
    )
    print(response["response"])
