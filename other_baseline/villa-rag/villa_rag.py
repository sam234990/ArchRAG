import sys
import os

# 添加 src 文件夹到 sys.path
src_path = "/home/wangshu/rag/hier_graph_rag"
sys.path.append(os.path.abspath(src_path))
# os.environ["WANDB_MODE"] = "offline"
# 设置代理
os.environ["http_proxy"] = "http://127.0.0.1:7892"
os.environ["https_proxy"] = "http://127.0.0.1:7892"


from src.llm import llm_invoker
from src.lm_emb import openai_embedding
from src.evaluate.evaluate import *
import multiprocessing as mp
import pandas as pd
import numpy as np
import argparse
import wandb
import faiss


def process_worker(
    dataset_part: pd.DataFrame,
    index,
    corpus,
    process_id,
    prompt,
    llm_invoker_args,
    topk=2,
):
    results = []
    all_token = 0
    for i in range(len(dataset_part)):
        data = dataset_part.iloc[i]

        query_embedding = openai_embedding(
            data["question"],
            llm_invoker_args.embedding_api_key,
            llm_invoker_args.embedding_api_base,
            llm_invoker_args.embedding_model,
        )
        query_embedding = np.array(query_embedding).reshape(1, -1)
        _, preds = index.search(query_embedding, topk)
        retrieval_context_idx = preds.flatten()
        retrieval_context = ""
        for idx in retrieval_context_idx:
            retrieval_context += (
                corpus.iloc[idx]["title"] + corpus.iloc[idx]["content"] + " "
            )

        query_content = prompt.format(
            question=data["question"], context=retrieval_context
        )

        llm_res, token = llm_invoker(query_content, args=llm_invoker_args)
        all_token += token
        tmp_res = {
            "id": data["id"],
            "question": data["question"],
            "label": data["label"],
            "pred": llm_res,
        }
        results.append(tmp_res)
        if i % (len(dataset_part) / 3) == 0:
            print(f"Process {process_id}: Processing {i}/{len(dataset_part)}")

    return results, all_token


def rag_llm(
    dataset: pd.DataFrame, index, corpus, save_dir, args=None, num_workers=12, topk=2
):
    if args.debug_flag:
        dataset = dataset.iloc[:20]

    prompt = """Qustion: {question}
{context} 
Answer: """
    wandb.init(
        project=f"{args.project}",
        name=f"{args.dataset_name}_villa_rag_{args.engine}_topk_{topk}",
        config=args,
    )

    dataset_parts = np.array_split(dataset, num_workers)
    print(f"dataset size: {len(dataset)}")
    print(f"split the dataset into {num_workers} subsets")
    print(f"subset size: {len(dataset_parts[0])}")

    llm_args = Args()
    llm_args.engine = args.engine
    print(f"llm engine: {llm_args.engine}")

    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(
            process_worker,
            [
                (dataset_part, index, corpus, idx, prompt, llm_args, topk)
                for idx, dataset_part in enumerate(dataset_parts)
            ],
        )

    # 整理返回结果和token使用量
    flattened_results = [item for sublist, _ in results for item in sublist]
    total_tokens = sum(token for _, token in results)
    results_df = pd.DataFrame(flattened_results)

    print(f"Total tokens used: {total_tokens}")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"villa_rag_{args.engine}_topk{topk}.csv")

    print(f"Saving results to {save_path}")

    results_df.to_csv(save_path, index=False)
    eval_res(save_path=save_path, eval_mode=args.eval_mode, args=args)


def eval_res(save_path, eval_mode, args=None):
    if eval_mode == "KGQA":
        hit = get_accuracy_webqsp_qa(save_path)
        print(f"Test Hit : {hit}")
        wandb.log({"Test Hit": hit})
    elif eval_mode == "DocQA":
        if args.dataset_name != "narrativeqa":
            hit = get_accuracy_doc_qa(save_path)
            print(f"Test Hit : {hit}")
            wandb.log({"Test Hit": hit})
        else:
            Blue_1 = get_bleu_doc_qa(save_path)
            print(f"Test Blue_1 : {Blue_1}")
            wandb.log({"Test Blue_1": Blue_1})


class Args:
    def __init__(self):
        self.api_key = "ollama"
        self.api_base = "http://localhost:5000/forward"
        self.embedding_local = False
        self.embedding_model_local = "nomic-embed-text"
        self.embedding_api_key = "ollama"
        self.embedding_api_base = "http://localhost:5000/forward"
        self.embedding_model = "nomic-embed-text"


def load_index(dataset_name):

    corpus_path = {
        "hotpot": "/mnt/data/wangshu/hcarag/HotpotQA/dataset/rag_hotpotqa_corpus.json",
        "multihop": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_corpus.json",
    }
    index_path = {
        "hotpot": "/mnt/data/wangshu/hcarag/HotpotQA/dataset/rag_hotpotqa_corpus.index",
        "multihop": "/mnt/data/wangshu/hcarag/MultiHop-RAG/dataset/rag_multihop_corpus.index",
    }

    index = faiss.read_index(index_path[dataset_name])
    corpus = pd.read_json(corpus_path[dataset_name], lines=True, orient="records")
    return index, corpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--project", type=str, default="hcarag")

    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=dataset_name_path.keys(),  # 只允许选择这两个数据集
        default="hotpot",
        help=f"Select the dataset name. Options are: {' '.join(dataset_name_path.keys())}",
    )

    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["KGQA", "DocQA"],
        default="DocQA",
        help="Evaluation mode for the dataset:['KGQA', 'DocQA']",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers for multiprocessing",
    )

    parser.add_argument(
        "--debug_flag",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Debug flag for testing",
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="llama3.1:8b4k",
        help="Engine name for LLM",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=2,
        help="Top k retrieval context",
    )

    args = parser.parse_args()

    dataset_path = dataset_name_path[args.dataset_name]
    save_dir = baseline_save_path_dict[args.dataset_name]

    # Read dataset
    dataset = pd.read_json(dataset_path, lines=True, orient="records")
    # dataset.rename(columns={"answers": "label"}, inplace=True)
    dataset["id"] = range(len(dataset))

    index, corpus = load_index(args.dataset_name)

    rag_llm(
        dataset=dataset,
        index=index,
        corpus=corpus,
        save_dir=save_dir,
        args=args,
        num_workers=args.num_workers,
        topk=args.topk,
    )
