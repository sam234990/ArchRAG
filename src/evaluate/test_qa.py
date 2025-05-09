import pandas as pd
import multiprocessing as mp
from functools import partial
import logging
import wandb
import os
import time

# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:7892"
os.environ["https_proxy"] = "http://127.0.0.1:7892"

from src.inference import *
from src.utils import create_inference_arg_parser
from src.evaluate.evaluate import *


def load_datasets(datasets_path) -> pd.DataFrame:
    qa_df = pd.read_json(datasets_path, orient="records", lines=True)
    print("test datasets size:")
    print(qa_df.shape)
    qa_df["id"] = range(len(qa_df))
    return qa_df


def process_question(
    idx,
    row,
    index_dict,
    query_paras,
    args,
):
    question = row["question"]
    if (idx % 100 == 0) and args.print_log:
        print(f"Processing question at index {idx}: {question}")
    try:
        response_report, total_token = hcarag(
            query_content=question,
            index_dict=index_dict,
            query_paras=query_paras,
            args=args,
        )
        return idx, response_report, total_token
    except Exception as e:
        logging.error(f"Error processing question at index {idx}: {e}")
        return idx, None, 0


def test_qa(query_paras, args):

    wandb.init(
        project=f"{args.project}",
        name=f"{args.dataset_name}_hcarag_{query_paras['generate_strategy']}",
        config=args,
    )

    # 1. load dataset and index
    index_dict = load_index(args)

    ragqa_list = ["lifestyle", "recreation", "technology", "science", "writing"]

    if args.dataset_name in ragqa_list:
        dataset_path = (
            f"/mnt/data/wangshu/hcarag/RAG-QA-Arena/{args.dataset_name}/Question.json"
        )
    else:
        dataset_path = dataset_name_path[args.dataset_name]
        if "narrative" in args.dataset_name:
            dataset_path = dataset_path.format(doc_idx=args.doc_idx)

    qa_df = load_datasets(dataset_path)

    # 重置索引，确保连续性
    qa_df.reset_index(drop=True, inplace=True)

    DEBUG_FLAG = args.debug_flag
    qa_df = qa_df.iloc[:10] if DEBUG_FLAG else qa_df

    save_file_str = "_".join([str(value) for value in query_paras.values()])
    # save_file_str += ".csv"
    save_file_str += "_newprompt.csv"
    if args.dataset_name == "multihop_summary":
        inference_output_dir = args.output_dir + "/summary"
    else:
        inference_output_dir = args.output_dir + "/qa"
    os.makedirs(inference_output_dir, exist_ok=True)
    save_file_qa = os.path.join(inference_output_dir, save_file_str)
    print(f"Save file: {save_file_qa}")

    print_args(query_paras, "Query Parameters:")

    number_works = args.num_workers if not DEBUG_FLAG else 2
    print(f"Number of workers: {number_works}")
    print(f"Number of questions: {len(qa_df)}")
    print(f"Number of questions per process: {len(qa_df) / number_works}")

    start_time = time.time()

    # 创建进程池
    with mp.Pool(processes=number_works) as pool:
        # 准备每个问题的输入参数
        process_func = partial(
            process_question,
            index_dict=index_dict,
            query_paras=query_paras,
            args=args,
        )

        # 使用 enumerate 处理每个问题，确保索引正确
        results = pool.starmap(
            process_func, [(idx, row) for idx, row in qa_df.iterrows()]
        )

    all_token = 0
    # 将结果合并回 qa_df
    for idx, response_report, total_token in results:
        # 确保索引有效
        all_token += total_token
        if isinstance(response_report, dict):
            qa_df.loc[idx, "raw_result"] = response_report.get("raw_result", "None")
            qa_df.loc[idx, "pred"] = response_report.get("pred", "None")
            for key, value in response_report.items():
                if key not in ["raw_result", "pred"]:
                    qa_df.loc[idx, key] = value
        else:
            qa_df.loc[idx, "raw_result"] = "None"
            qa_df.loc[idx, "pred"] = "None"

    print(f"Finish query Time: {time.time() - start_time:.2f} seconds")
    print(f"Total token: {all_token}")

    qa_df["pred"] = qa_df["pred"].fillna("No Answer", inplace=False)

    qa_df.to_csv(save_file_qa, index=False)

    # 2. Evaluation
    eval_inference(save_file_qa, args)


def eval_inference(prediction_path, args):
    if args.dataset_name == "multihop_summary":
        print("Summarization task will be evaluated by LLM, use summary_eval.py file.")
        return

    if args.eval_mode == "KGQA":
        acc = get_accuracy_webqsp_qa(prediction_path)
        print(f"Test Hit {acc}")

        print("-" * 30)
        print("Test Raw Result")
        acc_raw = get_accuracy_webqsp_qa(prediction_path, pred_col="raw_result")
        print(f"Test Hit Raw {acc_raw}")
        wandb.log({"Test Acc": acc_raw})
    elif args.eval_mode == "DocQA":
        if "narrative" in args.dataset_name:
            Blue_1 = get_bleu_doc_qa(prediction_path)
            print(f"Test Blue_1 {Blue_1}")
            print("-" * 30)
            print("Test Raw Result")
            Blue_1_raw = get_bleu_doc_qa(prediction_path, pred_col="raw_result")
            print(f"Test Hit Blue_1 {Blue_1_raw}")
            wandb.log({"Test Blue_1": Blue_1_raw})
        else:
            hit = get_accuracy_doc_qa(prediction_path)
            print(f"Test Hit {hit}")

            print("-" * 30)
            print("Test Raw Result")
            acc_raw = get_accuracy_doc_qa(prediction_path, pred_col="raw_result")
            print(f"Test Hit Raw {acc_raw}")
            wandb.log({"Test Acc": acc_raw})


def process_retrieval(
    idx,
    row,
    hc_index,
    entity_df,
    community_df,
    level_summary_df,
    relation_df,
    query_paras,
    args,
):
    question = row["question"]
    if (idx % 100 == 0) and args.print_log:
        print(f"Processing question at index {idx}: {question}")
    try:

        topk_entity, topk_community, topk_related_r = hcarag_retrieval(
            query_content=question,
            hc_index=hc_index,
            entity_df=entity_df,
            community_df=community_df,
            level_summary_df=level_summary_df,
            relation_df=relation_df,
            query_paras=query_paras,
            args=args,
        )
        res_row = row.copy()
        e_r_content = prep_e_r_content(
            topk_entity, topk_related_r, max_tokens=args.max_tokens
        )
        c_content = prep_community_content(topk_community, max_tokens=args.max_tokens)

        e_r_content = " ".join(e_r_content)
        c_content = " ".join(c_content)

        res_row["e_r_content"] = e_r_content
        res_row["c_content"] = c_content

        e_r_cnt = 0
        c_cnt = 0
        ans_list = row["answers"]

        for answer_item in ans_list:
            if answer_item in e_r_content:
                e_r_cnt += 1
                break

        for answer_item in ans_list:
            if answer_item in c_content:
                c_cnt += 1
                break

        res_row["e_r_cnt"] = e_r_cnt
        res_row["c_cnt"] = c_cnt
        res_row["cnt"] = 1 if (e_r_cnt + c_cnt) > 0 else 0
        return idx, res_row
    except Exception as e:
        logging.error(f"Error processing question at index {idx}: {e}")
        return idx, None


def eval_retrieval(query_paras, args):
    hc_index, entity_df, community_df, level_summary_df, relation_df = load_index(args)
    qa_df = load_datasets(args.dataset_path)

    # 重置索引，确保连续性
    qa_df.reset_index(drop=True, inplace=True)

    DEBUG_FLAG = args.debug_flag
    qa_df = qa_df.iloc[:10] if DEBUG_FLAG else qa_df

    save_file_str = (
        f"{query_paras['strategy']}_"
        f"{query_paras['k_each_level']}_"
        f"{query_paras['k_final']}_"
        f"{query_paras['topk_e']}_"
        f"{query_paras['all_k_inference']}_"
        f"{query_paras['generate_strategy']}_"
        f"{query_paras['response_type']}_"
    )
    save_file_str += ".csv"
    inference_output_dir = "/home/wangshu/rag/hier_graph_rag/test/debug_file"
    save_file_qa = os.path.join(inference_output_dir, save_file_str)
    print(f"Save file: {save_file_qa}")

    print_args(query_paras, "Query Parameters:")

    number_works = args.num_workers if not DEBUG_FLAG else 2
    print(f"Number of workers: {number_works}")
    print(f"Number of questions: {len(qa_df)}")
    print(f"Number of questions per process: {len(qa_df) / number_works}")

    # 创建进程池
    with mp.Pool(processes=number_works) as pool:
        # 准备每个问题的输入参数
        process_func = partial(
            process_retrieval,
            hc_index=hc_index,
            entity_df=entity_df,
            community_df=community_df,
            level_summary_df=level_summary_df,
            relation_df=relation_df,
            query_paras=query_paras,
            args=args,
        )

        # 使用 enumerate 处理每个问题，确保索引正确
        results = pool.starmap(
            process_func, [(idx, row) for idx, row in qa_df.iterrows()]
        )

    valid_results = [(idx, res_row) for idx, res_row in results if res_row is not None]

    # 提取索引和有效行的数据
    _, rows = zip(*valid_results)

    # 将有效行转换为 DataFrame
    res_df = pd.DataFrame(rows)
    all_question = len(res_df)
    sum_e_r = res_df["e_r_cnt"].sum()
    sum_c = res_df["c_cnt"].sum()
    sum_all = res_df["cnt"].sum()
    print(f"all question number is {all_question}")
    print(
        f"retrieval entity and relation with answer :{sum_e_r}, ratio: {sum_e_r / all_question}"
    )
    print(f"retrieval community with answer :{sum_c}, ratio: {sum_c / all_question}")
    print(
        f"retrieval information with answer :{sum_all}, ratio: {sum_all / all_question}"
    )


if __name__ == "__main__":
    parser = create_inference_arg_parser()
    args = parser.parse_args()
    print_args(args=args)

    query_paras = {
        "strategy": args.strategy,
        "only_entity": args.only_entity,
        "wo_hierarchical": args.wo_hierarchical,
        "k_each_level": args.k_each_level,
        "k_final": args.k_final,
        "topk_e": args.topk_e,
        "all_k_inference": args.all_k_inference,
        "ppr_refine": args.ppr_refine,
        "generate_strategy": args.generate_strategy,
        "response_type": args.response_type,
        "involve_llm_res": args.involve_llm_res,
        "topk_chunk": args.topk_chunk,
        "range_level": args.range_level,
    }
    # query_paras = {
    #     "strategy": "global",
    #     "k_each_level": 5,
    #     "k_final": 15,
    #     "all_k_inference": 15,
    #     "generate_strategy": "mr",
    #     "response_type": "QA",
    # }

    test_qa(query_paras, args=args)
    # eval_retrieval(query_paras, args=args)
