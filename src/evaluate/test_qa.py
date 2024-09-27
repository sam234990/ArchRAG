import pandas as pd
import multiprocessing as mp
from functools import partial
import logging

from ..inference import *
from ..utils import create_inference_arg_parser
from .evaluate import *


def load_datasets(datasets_path) -> pd.DataFrame:
    qa_df = pd.read_json(datasets_path)
    return qa_df
    pass


def process_question(
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
    try:
        response_report = hcarag(
            query_content=question,
            hc_index=hc_index,
            entity_df=entity_df,
            community_df=community_df,
            level_summary_df=level_summary_df,
            relation_df=relation_df,
            query_paras=query_paras,
            args=args,
        )
        return idx, response_report
    except Exception as e:
        logging.error(f"Error processing question at index {idx}: {e}")
        return idx, None


def test_qa(query_paras, args):
    hc_index, entity_df, community_df, level_summary_df, relation_df = load_index(args)
    qa_df = load_datasets(args.dataset_path)

    # 创建进程池
    with mp.Pool(processes=args.num_workers) as pool:
        # 准备每个问题的输入参数
        process_func = partial(
            process_question,
            hc_index=hc_index,
            entity_df=entity_df,
            community_df=community_df,
            level_summary_df=level_summary_df,
            relation_df=relation_df,
            query_paras=query_paras,
            args=args,
        )

        # 使用 starmap 并行处理每个问题
        results = pool.starmap(process_func, qa_df.iterrows())

    # 将结果合并回 qa_df
    for idx, response_report in results:
        if response_report:
            qa_df.loc[idx, "response_report"] = response_report
            qa_df.loc[idx, "pred"] = response_report["response"]

    qa_df["label"] = qa_df["answers"].apply(lambda x: "|".join(map(str, x)))
    save_file_str = (
        f"{query_paras['strategy']}_"
        f"{query_paras['k_each_level']}_"
        f"{query_paras['k_final']}_"
        f"{query_paras['inference_search_times']}_"
        f"{query_paras['generate_strategy']}"
    )
    save_file_str += ".json"
    save_file_qa = os.path.join(args.inference_output_dir, save_file_str)
    qa_df.to_csv(save_file_qa, index=False)

    acc = get_accuracy_webqsp_qa(save_file_qa)
    print(f"Test Acc {acc}")


if __name__ == "__main__":
    parser = create_inference_arg_parser()
    args = parser.parse_args()
    print_args(args=args)

    query_paras = {
        "strategy": "global",
        "k_each_level": 5,
        "k_final": 10,
        "inference_search_times": 2,
        "generate_strategy": "direct",
    }

    test_qa(query_paras, args=args)
