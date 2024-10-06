import pandas as pd
import os
import faiss
from src.utils import *
from src.llm import llm_invoker
from src.lm_emb import openai_embedding
from src.hchnsw_index import read_index
from src.client_reasoning import *
from sklearn.metrics.pairwise import cosine_similarity


def hcarag(
    query_content,
    hc_index: faiss.IndexHCHNSWFlat,
    entity_df,
    community_df,
    level_summary_df,
    relation_df,
    query_paras,
    args,
):

    topk_entity, topk_community, topk_related_r = hcarag_retrieval(
        query_content=query_content,
        hc_index=hc_index,
        entity_df=entity_df,
        community_df=community_df,
        level_summary_df=level_summary_df,
        relation_df=relation_df,
        query_paras=query_paras,
        args=args,
    )

    response_report = hcarag_inference(
        topk_entity,
        topk_community,
        topk_related_r,
        query_content,
        args.max_retries,
        args,
        query_paras=query_paras,
    )
    return response_report


def get_topk_related_r(query_embedding, relation_df, topk=10):
    embeddings = np.stack(relation_df["embedding"].values)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    similarities = cosine_similarity(embeddings, query_embedding).flatten()

    relation_df.loc[:, "similarity"] = similarities

    topk = min(topk, len(relation_df))
    topk_related_r = relation_df.nlargest(topk, "similarity")

    return topk_related_r


def hcarag_retrieval(
    query_content,
    hc_index: faiss.IndexHCHNSWFlat,
    entity_df,
    community_df,
    level_summary_df,
    relation_df,
    query_paras,
    args,
):
    query_paras["query_content"] = query_content
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
    final_k, k_per_level = load_strategy(
        query_paras=query_paras,
        number_levels=hc_level + 1,
        entity_df=entity_df,
        community_df=community_df,
        level_summary_df=level_summary_df,
        args=args,
    )

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

    # 用于存储 top-k 的实体和社区
    topk_entity = entity_df[entity_df["index_id"].isin(final_predictions)]
    topk_community = community_df[community_df["index_id"].isin(final_predictions)]

    sel_r_df = relation_df[
        relation_df["source_index_id"].isin(final_predictions)
    ].copy()
    if len(sel_r_df) == 0:
        topk_related_r = pd.DataFrame(columns=relation_df.columns)
    else:

        topk_related_r = get_topk_related_r(
            query_embedding, sel_r_df, topk=query_paras["topk_e"]
        )

    return topk_entity, topk_community, topk_related_r


def hcarag_inference(
    topk_entity, topk_community, topk_related_r, query, max_retries, args, query_paras
):
    if query_paras["generate_strategy"] == "direct":
        response_report = hcarag_inference_direct(
            topk_entity,
            topk_community,
            topk_related_r,
            query,
            max_retries,
            query_paras,
            args,
        )
    else:
        response_report = hcarag_inference_mr(
            topk_entity,
            topk_community,
            topk_related_r,
            query,
            query_paras,
            args,
        )

    if response_report["pred"] == "":
        response_report["pred"] = "No answer found."

    return response_report


def hcarag_inference_direct(
    topk_entity,
    topk_community,
    topk_related_r,
    query,
    max_retries,
    query_paras,
    args,
):

    content = prep_infer_content(
        entity_df=topk_entity,
        relation_df=topk_related_r,
        community_df=topk_community,
        query=query,
        max_tokens=args.max_tokens,
        response_type=query_paras["response_type"],
    )

    retries = 0
    direct_answer = ""
    raw_result = ""

    while retries < max_retries:
        raw_result = llm_invoker(content, args, max_tokens=args.max_tokens, json=False)

        success, direct_answer = qa_response_extract(raw_result)
        if success:
            break
        retries += 1

    response_report = {"pred": direct_answer, "raw_result": raw_result}

    return response_report


def hcarag_inference_mr(
    topk_entity,
    topk_community,
    topk_related_r,
    query,
    query_paras,
    args,
):
    map_res_df = map_inference(
        entity_df=topk_entity,
        community_df=topk_community,
        relation_df=topk_related_r,
        query=query,
        args=args,
    )
    response_report = reduce_inference(map_res_df, query, args)
    return response_report


def load_strategy(
    query_paras,
    number_levels,
    entity_df: pd.DataFrame,
    community_df: pd.DataFrame,
    level_summary_df: pd.DataFrame,
    args,
):

    strategy = query_paras["strategy"]
    if strategy == "global":
        k_each_level = query_paras["k_each_level"]
        k_final = query_paras["k_final"]
        k_per_level = [k_each_level] * number_levels

        return k_final, k_per_level
    elif strategy == "inference":
        k_final = query_paras["k_final"]

        level_weight, raw_result = problem_reasoning(
            query_content=query_paras["query_content"],
            entity_df=entity_df,
            community_df=community_df,
            level_summary_df=level_summary_df,
            max_level=number_levels - 1,
            max_retries=args.max_retries,
            args=args,
        )

        all_k = query_paras["all_k_inference"]

        k_per_level = calculate_k_per_level(level_weight, all_k)

        print("inference k per level is:")
        for k in k_per_level:
            print(k, end="; ")
        return k_final, k_per_level
    else:
        raise ValueError("Invalid strategy.")


def calculate_k_per_level(level_weight, all_k):
    total_weight = sum(level_weight)

    # 计算每层的 k 值并四舍五入为整数
    k_per_level = [round(weight / total_weight * all_k) for weight in level_weight]

    # 调整 k_per_level 以确保总和为 all_k
    current_sum = sum(k_per_level)
    while current_sum != all_k:
        # 找到需要增加或减少的数量
        difference = all_k - current_sum

        # 确保我们在调整时只对 k_per_level 中的某一层进行加一或减一
        if difference > 0:
            # 增加
            for i in range(len(k_per_level)):
                if difference <= 0:
                    break
                k_per_level[i] += 1
                difference -= 1
        else:
            # 减少
            for i in range(len(k_per_level)):
                if difference >= 0:
                    break
                if k_per_level[i] > 0:  # 确保不减到负数
                    k_per_level[i] -= 1
                    difference += 1

        current_sum = sum(k_per_level)

    return k_per_level


def load_index(args):

    hc_index = read_index(args.output_dir, "hchnsw.index")

    entity_path = os.path.join(args.output_dir, "entity_df_index.csv")
    entity_df = pd.read_csv(entity_path)

    community_path = os.path.join(args.output_dir, "community_df_index.csv")
    community_df = pd.read_csv(community_path)

    level_summary_path = os.path.join(args.output_dir, "level_summary.csv")
    level_summary_df = pd.read_csv(level_summary_path)

    relation_path = os.path.join(args.output_dir, "relationship_df_index.csv")
    relation_df = pd.read_csv(relation_path)

    # add relation embedding
    relation_embedding_path = os.path.join(
        args.output_dir, "relationship_embedding.csv"
    )
    relation_embedding_df = pd.read_csv(relation_embedding_path)
    relation_embedding_df["embedding"] = relation_embedding_df["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
    )
    idx_embed_map = dict(
        zip(relation_embedding_df["idx"], relation_embedding_df["embedding"])
    )

    relation_df["embedding"] = relation_df["embedding_idx"].map(idx_embed_map)
    print("Index loaded successfully.")
    return hc_index, entity_df, community_df, level_summary_df, relation_df


if __name__ == "__main__":
    parser = create_inference_arg_parser()
    args = parser.parse_args()
    hc_index, entity_df, community_df, level_summary_df, relation_df = load_index(args)
    # test_question = "What is the usage and value of TLB in an Operating System?"
    test_question = "what does jamaican people speak?"
    # query_paras = {
    #     "strategy": "global",
    #     "k_each_level": 5,
    #     "k_final": 10,
    #     "all_k_inference": 2,
    # }
    query_paras = {
        "strategy": "global",
        "k_each_level": 5,
        "k_final": 10,
        "topk_e": args.topk_e,
        "all_k_inference": 15,
        "generate_strategy": "mr",
        # "generate_strategy": "direct",
        "response_type": "QA",
    }
    response = hcarag(
        test_question,
        hc_index,
        entity_df,
        community_df,
        level_summary_df,
        relation_df,
        query_paras,
        args,
    )
    print(response["raw_result"])
    print(response["pred"])
