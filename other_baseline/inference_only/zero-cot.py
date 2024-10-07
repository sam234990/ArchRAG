import sys
import os

# 添加 src 文件夹到 sys.path
src_path = "/home/wangshu/rag/hier_graph_rag"
sys.path.append(os.path.abspath(src_path))

from src.llm import llm_invoker
from src.evaluate.evaluate import get_accuracy_webqsp_qa
import multiprocessing as mp
import pandas as pd
import numpy as np


def process_worker(dataset_part: pd.DataFrame, process_id, prompt, llm_invoker_args):
    results = []

    for i in range(len(dataset_part)):
        data = dataset_part.iloc[i]
        query_content = data["question"] + prompt
        llm_res = llm_invoker(query_content, args=llm_invoker_args)
        tmp_res = {
            "id": data["id"],
            "question": data["question"],
            "label": data["label"],
            "pred": llm_res,
        }
        results.append(tmp_res)
        if i % (len(dataset_part) / 3) == 0:
            print(f"Process {process_id}: Processing {i}/{len(dataset_part)}")

    return results


def run_zero_cot_llm(dataset: pd.DataFrame, strategy, save_dir, num_workers=24):
    print(f"Running {strategy} strategy")
    if strategy == "zero-shot":
        prompt = " \n Answer: "
    elif strategy == "cot":
        prompt = "Let’s think step by step. \n Answer: "

    dataset_parts = np.array_split(dataset, num_workers)
    print(f"dataset size: {len(dataset)}")
    print(f"split the dataset into {num_workers} subsets")
    print(f"subset size: {len(dataset_parts[0])}")

    llm_args = Args()

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(
            process_worker,
            [
                (dataset_part, idx, prompt, llm_args)
                for idx, dataset_part in enumerate(dataset_parts)
            ],
        )

    # 整理返回结果
    flattened_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(flattened_results)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{strategy}.csv")
    results_df.to_csv(save_path, index=False)

    acc = get_accuracy_webqsp_qa(save_path)
    print(f"Test Acc : {acc}")


class Args:
    def __init__(self):
        self.api_key = "ollama"
        self.api_base = "http://localhost:5000/forward"
        self.engine = "llama3.1:8b4k"


if __name__ == "__main__":
    dataset_path = "/mnt/data/wangshu/hcarag/mintaka/QA/mintaka_test_qa.json"
    dataset = pd.read_json(dataset_path, lines=True, orient="records")
    dataset.rename(columns={"answers": "label"}, inplace=True)
    dataset["id"] = range(len(dataset))

    strategy = "zero-shot"
    # strategy = "cot"

    save_dir = "/mnt/data/wangshu/hcarag/mintaka/QA/baseline"

    run_zero_cot_llm(dataset, strategy, save_dir=save_dir)
