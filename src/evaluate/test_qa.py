import pandas as pd
import multiprocessing as mp
from functools import partial
import logging

from src.inference import *
from src.utils import create_inference_arg_parser
from src.evaluate.evaluate import *

DEBUG_FLAG = True


def load_datasets(datasets_path) -> pd.DataFrame:
    qa_df = pd.read_json(datasets_path, orient="records", lines=True)
    print("test datasets size:")
    print(qa_df.shape)
    return qa_df


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
    if (idx % 100 == 0) and args.print_log:
        print(f"Processing question at index {idx}: {question}")
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
    
    # 重置索引，确保连续性
    qa_df.reset_index(drop=True, inplace=True)
    
    qa_df = qa_df.iloc[:20] if DEBUG_FLAG else qa_df

    save_file_str = (
        f"{query_paras['strategy']}_"
        f"{query_paras['k_each_level']}_"
        f"{query_paras['k_final']}_"
        f"{query_paras['all_k_inference']}_"
        f"{query_paras['generate_strategy']}_"
        f"{query_paras['response_type']}_"
    )
    save_file_str += ".json"
    save_file_qa = os.path.join(args.inference_output_dir, save_file_str)

    number_works = args.num_workers if not DEBUG_FLAG else 20
    # 创建进程池
    with mp.Pool(processes=number_works) as pool:
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

        # 使用 enumerate 处理每个问题，确保索引正确
        results = pool.starmap(process_func, [(idx, row) for idx, row in qa_df.iterrows()])

    # 将结果合并回 qa_df
    for (idx, response_report) in results:
        # 确保索引有效
        if idx < len(qa_df):
            qa_df.loc[idx, "raw_result"] = (
                response_report["raw_result"]
                if response_report["raw_result"] != ""
                else "None"
            )
            qa_df.loc[idx, "pred"] = (
                response_report["pred"] if response_report["pred"] != "" else "None"
            )
        else:
            print(f"Index {idx} is out of range for qa_df.")

    qa_df['pred'] = qa_df['pred'].fillna("No Answer", inplace=False)
    qa_df["label"] = qa_df["answers"].apply(lambda x: "|".join(map(str, x)))

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
        "all_k_inference": 15,
        "generate_strategy": "direct",
        "response_type": "QA",
    }

    test_qa(query_paras, args=args)
