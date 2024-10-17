from inference import *


def ppr_serach(query_content,
    hc_index: faiss.IndexHCHNSWFlat,
    entity_df,
    community_df,
    level_summary_df,
    relation_df,
    query_paras,
    args):

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

    if query_paras["only_entity"] is True:
        query_max_levl = 1
    else:
        query_max_levl = hc_level + 1

    for level in range(query_max_levl):
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

    final_k = min(len(all_results), final_k)

    # 获取最终的 top-k 结果
    if query_paras["generate_strategy"] == "mr":
        # map-reduce use all the result
        final_results = all_results
    else:
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

    #...

    # return ppr_topk_entity, topk_community, ppr_topk_related_r
    
    return topk_entity, topk_community, topk_related_r






if __name__ == "__main__":
    parser = create_inference_arg_parser()
    args = parser.parse_args()

    graph, entities_df, final_relationships = read_graph_nx(
        file_path=args.base_path,
        entity_filename=args.entity_filename,
        relationship_filename=args.relationship_filename,
    )

    hc_index, entity_df, community_df, level_summary_df, relation_df = load_index(args)
    test_question = "what does jamaican people speak?"
    query_paras = {
        "strategy": "global",
        "only_entity":args.only_entity,
        "k_each_level": 5,
        "k_final": 10,
        "topk_e": args.topk_e,
        "all_k_inference": 15,
        "generate_strategy": "mr",
        # "generate_strategy": "direct",
        "response_type": "QA",
    }

    ppr_serach(test_question,
    hc_index,
    entity_df,
    community_df,
    level_summary_df,
    relation_df,
    query_paras,
    args)
