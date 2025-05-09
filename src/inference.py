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
    index_dict,
    query_paras,
    args,
):
    all_token = 0

    retrieval_dict, token_used = hcarag_retrieval(
        query_content=query_content,
        hc_index=index_dict["hc_index"],
        entity_df=index_dict["entity_df"],
        community_df=index_dict["community_df"],
        level_summary_df=index_dict["level_summary_df"],
        relation_df=index_dict["relation_df"],
        query_paras=query_paras,
        villa_index=index_dict["villa_index"],
        chunk_df=index_dict["chunk_df"],
        graph=index_dict["graph"],
        args=args,
    )

    all_token += token_used
    response_report, total_token = hcarag_inference(
        topk_entity=retrieval_dict["topk_entity"],
        topk_community=retrieval_dict["topk_community"],
        topk_related_r=retrieval_dict["topk_related_r"],
        topk_chunk=retrieval_dict["topk_chunk"],
        query=query_content,
        max_retries=args.max_retries,
        args=args,
        query_paras=query_paras,
    )
    all_token += total_token

    topk_enetity_str = df_to_str(retrieval_dict["topk_entity"])
    topk_community_str = df_to_str(retrieval_dict["topk_community"])
    topk_related_r_str = df_to_str(retrieval_dict["topk_related_r"])
    topk_chunk_str = df_to_str(retrieval_dict["topk_chunk"])
    response_report["topk_entity"] = topk_enetity_str
    response_report["topk_community"] = topk_community_str
    response_report["topk_related_r"] = topk_related_r_str
    response_report["topk_chunk"] = topk_chunk_str

    return response_report, all_token


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
    villa_index,
    chunk_df,
    query_paras,
    graph,
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
    final_k, k_per_level, token_used = load_strategy(
        query_paras=query_paras,
        number_levels=hc_level + 1,
        entity_df=entity_df,
        community_df=community_df,
        level_summary_df=level_summary_df,
        args=args,
    )

    # if query_paras['strategy'] == "all":
    #     # Use GraphRAG method
    #     print("Using GraphRAG method.")
        
    #     # get index_id in community_df if the level <= limit_level
    #     limit_level = int(query_paras['range_level'])
    #     selected_community = community_df[community_df['level'] <= limit_level]
    #     index_id_list = selected_community['index_id'].to_list()
        
    #     saerch_params = faiss.SearchParametersHCHNSW()
    #     saerch_params.search_level = 0
    #     _, preds = hc_index.search(
    #         query_embedding, k=query_paras['k_each_level'], params=saerch_params
    #     )

    #     # 将 numpy 数组展平成一维，并添加到 all_results 中
    #     preds_flat = preds.flatten()

    #     # combine preds_flat and index_id_list into final_predictions 
        
    #     final_predictions = index_id_list + preds_flat.tolist()
    # else:    
    all_results = []
    if query_paras["only_entity"] is True:
        query_max_levl = 1
    elif query_paras["wo_hierarchical"] is False:
        query_max_levl = 2
    else:
        query_max_levl = hc_level + 1

    if query_paras['strategy'] == "all":
        query_max_levl = int(query_paras['range_level']) + 1
        print("Using GraphRAG method.")


    for level in range(query_max_levl):
        saerch_params = faiss.SearchParametersHCHNSW()
        saerch_params.search_level = level
        distances, preds = hc_index.search(
            query_embedding, k=k_per_level[level], params=saerch_params
        )

        # 将 numpy 数组展平成一维，并添加到 all_results 中
        distances_flat = distances.flatten()
        preds_flat = preds.flatten()

        for dist, villa_pred in zip(distances_flat, preds_flat):
            all_results.append((dist, villa_pred))

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

    retrieval_dict = {
        "topk_entity": topk_entity,
        "topk_community": topk_community,
        "topk_related_r": topk_related_r,
        "topk_chunk": None,
    }

    if query_paras["topk_chunk"] > 0:
        topk = query_paras["topk_chunk"]
        _, villa_pred = villa_index.search(query_embedding, topk)
        retrieval_context_idx = villa_pred.flatten()
        topk_chunk_df = chunk_df.iloc[retrieval_context_idx]
        retrieval_dict["topk_chunk"] = topk_chunk_df

    if query_paras["ppr_refine"] is False:
        return retrieval_dict, token_used
    else:

        # 从 topk_entity 获取 ppr 所需的 personalization 信息
        siz = len(topk_entity["human_readable_id"])
        personalization = {id: 1.0 / siz for id in topk_entity["human_readable_id"]}

        # 调用 nx 库 pagerank 方法，获得图的 pagerank 字典
        pagerank = nx.pagerank(graph, personalization=personalization)

        # 从 pagerank 字典中找到 rank 前 ppr_topk 的元素，将其id加入 ppr_topk_id
        ppr_topk = query_paras["k_final"]
        ppr_topk_id = [
            id
            for id, value in sorted(
                pagerank.items(), key=lambda item: item[1], reverse=True
            )[:ppr_topk]
        ]

        # 提取最终的预测值（实体索引）
        ppr_final_predictions = [
            id
            for id in entity_df[
                entity_df["human_readable_id"].isin(ppr_topk_id)
            ].index_id
        ]

        # 用于存储 ppr top-k 的实体和社区
        ppr_topk_entity = entity_df[entity_df["index_id"].isin(ppr_final_predictions)]
        ppr_sel_r_df = relation_df[
            relation_df["source_index_id"].isin(ppr_final_predictions)
        ].copy()
        if len(ppr_sel_r_df) == 0:
            ppr_topk_related_r = pd.DataFrame(columns=relation_df.columns)
        else:
            ppr_topk_related_r = get_topk_related_r(
                query_embedding, ppr_sel_r_df, topk=query_paras["topk_e"]
            )

        retrieval_dict["topk_entity"] = ppr_topk_entity
        retrieval_dict["topk_related_r"] = ppr_topk_related_r

        return retrieval_dict, token_used
        # return ppr_topk_entity, topk_community, ppr_topk_related_r, token_used


def hcarag_inference(
    topk_entity,
    topk_community,
    topk_related_r,
    topk_chunk,
    query,
    max_retries,
    args,
    query_paras,
):
    if query_paras["generate_strategy"] == "direct":
        response_report, total_token = hcarag_inference_direct(
            topk_entity,
            topk_community,
            topk_related_r,
            query,
            max_retries,
            query_paras,
            args,
        )
    else:
        response_report, total_token = hcarag_inference_mr(
            topk_entity,
            topk_community,
            topk_related_r,
            topk_chunk,
            query,
            query_paras,
            args,
        )

    if response_report["pred"] == "":
        response_report["pred"] = "No answer found."

    return response_report, total_token


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
    total_token = 0

    while retries < max_retries:
        raw_result, cur_token = llm_invoker(
            content, args, max_tokens=args.max_tokens, json=False
        )
        total_token += cur_token
        success, direct_answer = qa_response_extract(raw_result)
        if success:
            break
        retries += 1

    response_report = {"pred": direct_answer, "raw_result": raw_result}

    return response_report, total_token


def hcarag_inference_mr(
    topk_entity,
    topk_community,
    topk_related_r,
    topk_chunk,
    query,
    query_paras,
    args,
):
    all_token = 0

    llm_query_content = query + "\nLet’s think step by step. \n Answer: "
    llm_res, cur_token = llm_invoker(
        llm_query_content, args=args, max_tokens=args.max_tokens, json=False
    )
    all_token += cur_token

    map_res_df, cur_token_map = map_inference(
        entity_df=topk_entity,
        community_df=topk_community,
        relation_df=topk_related_r,
        llm_res=llm_res,
        topk_chunk=topk_chunk,
        query=query,
        query_paras=query_paras,
        args=args,
    )
    all_token += cur_token_map

    response_report, cur_token_reduce = reduce_inference(map_res_df, query, args, response_type=query_paras["response_type"])
    all_token += cur_token_reduce

    map_res_str = df_to_str(map_res_df)
    response_report["map_res"] = map_res_str

    return response_report, all_token


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

        return k_final, k_per_level, 0
    elif strategy == "all":
        k_each_level = query_paras["k_each_level"]
        k_final = query_paras["k_final"]
        k_per_level = [k_each_level] + [200] * query_paras["range_level"]

        return k_final, k_per_level, 0
    
    elif strategy == "inference":
        k_final = query_paras["k_final"]

        level_weight, raw_result, all_token = problem_reasoning(
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
        return k_final, k_per_level, all_token
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

    villa_index, chunk_df = load_villa_index(args)

    graph, _, _ = read_graph_nx(
        file_path=args.base_path,
        entity_filename=args.entity_filename,
        relationship_filename=args.relationship_filename,
    )

    index_dict = {
        "hc_index": hc_index,
        "entity_df": entity_df,
        "community_df": community_df,
        "level_summary_df": level_summary_df,
        "relation_df": relation_df,
        "villa_index": villa_index,
        "chunk_df": chunk_df,
        "graph": graph,
    }

    print("Index loaded successfully.")

    # return hc_index, entity_df, community_df, level_summary_df, relation_df
    return index_dict


def load_villa_index(args):
    dataset_name = args.dataset_name
    corpus_path = {
        "hotpot": "/mnt/data/wangshu/hcarag/HotpotQA/dataset/rag_hotpotqa_corpus.json",
        # "multihop": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_corpus.json",
        "multihop": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_summary_corpus.json",
        "multihop_summary": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_summary_corpus.json",
        "narrativeqa_train": "/mnt/data/wangshu/hcarag/narrativeqa/data/train/{doc_idx}/qa_dataset/corpus_chunk.json",
        "narrativeqa_test": "/mnt/data/wangshu/hcarag/narrativeqa/data/test/{doc_idx}/qa_dataset/corpus_chunk.json",
        "lifestyle": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/lifestyle/Corpus.json",
        "recreation": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/recreation/Corpus.json",
        "science": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/science/Corpus.json",
        "technology": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/technology/Corpus.json",
        "writing": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/writing/Corpus.json",
    }
    index_path = {
        "hotpot": "/mnt/data/wangshu/hcarag/HotpotQA/dataset/rag_hotpotqa_corpus.index",
        # "multihop": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_corpus.index",
        "multihop": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_summary_corpus.index",
        "multihop_summary": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_summary_corpus.index",
        "narrativeqa_train": "/mnt/data/wangshu/hcarag/narrativeqa/data/train/{doc_idx}/qa_dataset/rag_corpus_chunk.index",
        "narrativeqa_test": "/mnt/data/wangshu/hcarag/narrativeqa/data/test/{doc_idx}/qa_dataset/rag_corpus_chunk.index",
        "lifestyle": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/lifestyle/vanilla.index",
        "recreation": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/recreation/vanilla.index",
        "science": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/science/vanilla.index",
        "technology": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/technology/vanilla.index",
        "writing": "/mnt/data/wangshu/hcarag/RAG-QA-Arena/writing/vanilla.index",
    }

    index_file = index_path[dataset_name]
    corpus_file = corpus_path[dataset_name]
    if "narrativeqa" in dataset_name:
        doc_idx = args.doc_idx
        index_file = index_file.format(doc_idx=doc_idx)
        corpus_file = corpus_file.format(doc_idx=doc_idx)

    index = faiss.read_index(index_file)
    corpus = pd.read_json(corpus_file, lines=True, orient="records")
    
    if "context" in corpus.columns:
        corpus.rename(columns={"context": "content"}, inplace=True)
    
    return index, corpus


def df_to_str(input_df: pd.DataFrame):
    if input_df is None:
        return "NoneType"
    row_sep = "<row-sep>\n"
    columns = [col for col in input_df.columns if "embedding" not in col]
    res_str = ",".join(columns) + row_sep
    for index, row in input_df.iterrows():
        res_str += ",".join([str(row[col]) for col in columns]) + row_sep
    return res_str


if __name__ == "__main__":
    parser = create_inference_arg_parser()
    args = parser.parse_args()

    index_dict = load_index(args)
    # {"question":"What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?","answers":"great-grandfather","label":"great-grandfather"}
    test_question = "What relationship does Fred Gehrke have to the 23rd overall pick in the 2010 Major League Baseball Draft?"
    query_paras = {
        "strategy": "global",
        "only_entity": args.only_entity,
        "wo_hierarchical": args.wo_hierarchical,
        "k_each_level": 5,
        "k_final": 10,
        "topk_e": args.topk_e,
        "all_k_inference": 15,
        "ppr_refine": args.ppr_refine,
        "generate_strategy": "mr",
        "response_type": "QA",
        "involve_llm_res": True,
        "topk_chunk": 2,
    }
    response, total_token = hcarag(
        query_content=test_question,
        index_dict=index_dict,
        query_paras=query_paras,
        args=args,
    )
    print(response["raw_result"])
    print(response["pred"])
    print(f"Total tokens: {total_token}")
    for key, value in response.items():
        print(f"{key}: {value}")
    print("Done.")
