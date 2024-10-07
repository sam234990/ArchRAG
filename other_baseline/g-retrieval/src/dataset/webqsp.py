import os
import sys

g_project_path = "/home/wangshu/rag/hier_graph_rag/other_baseline/g-retrieval"
sys.path.append(os.path.abspath(g_project_path))
print(g_project_path)

import pandas as pd
import numpy as np
import ast
from torch.utils.data import Dataset

from src.dataset.utils.retrieval import retrieval_via_pcst_hnsw
from faiss import IndexHNSWFlat
import multiprocessing as mp


path = "/mnt/data/wangshu/hcarag/FB15k/KG"

cached_desc = f"{path}/cached_desc"

dataset_path = "/mnt/data/wangshu/hcarag/FB15k/webqa/webqa_emb.json"


class WebQDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = "Please answer the given question."
        self.graph = None
        self.graph_type = "Knowledge Graph"
        self.dataset = pd.read_json(dataset_path, lines=True, orient="records")

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.iloc[index]
        question = f'Question: {data["question"]}\nAnswer: '
        desc = open(f"{cached_desc}/{index}.txt", "r").read()
        label = ("|").join(data["answers"]).lower()

        return {
            "id": index,
            "question": question,
            "label": label,
            "desc": desc,
        }


def process_question_worker(
    entity_df, relation_df, relation_embedding, dataset_split: pd.DataFrame
):

    topk = 3
    topk_e = 3
    dim = entity_df.iloc[0]["description_embedding"].shape[0]
    entity_index = IndexHNSWFlat(dim, 32)
    relation_index = IndexHNSWFlat(dim, 32)

    entity_embeddings = np.array(
        entity_df["description_embedding"].tolist(), dtype=np.float32
    )
    entity_index.add(entity_embeddings)

    relation_embeddings = np.array(
        relation_embedding["description_embedding"].tolist(), dtype=np.float32
    )
    relation_index.add(relation_embeddings)
    q_emb = np.vstack(dataset_split["embedding"].values).astype(np.float32)

    # 批量查询 entity 和 relation
    D_e, I_e = entity_index.search(q_emb, topk)
    D_r, I_r = relation_index.search(q_emb, topk_e)

    results = []
    for i, row in enumerate(dataset_split.itertuples(index=False)):
        if os.path.exists(f"{cached_desc}/{row.question_id}.txt"):
            continue
        # 可以根据 D_e, I_e 和 D_r, I_r 的结果进行后续处理
        res_desc = retrieval_via_pcst_hnsw(
            I_e=I_e[i],
            I_r=I_r[i],
            entity_df=entity_df,
            relation_df=relation_df,
            topk=topk,
            topk_e=topk_e,
        )
        result = {"question_id": row.question_id, "desc": res_desc}
        results.append(result)
        if len(results) % 100 == 0:
            print(f"Processed {len(results)} questions")

    return results  # 直接返回结果


def preprocess(entity_df, relation_df, relation_embedding, dataset, num_workers=32):
    os.makedirs(cached_desc, exist_ok=True)

    # 划分 dataset 为 num_workers 份
    dataset_splits = np.array_split(dataset, num_workers)
    print(f"Processing {len(dataset)} questions using {num_workers} workers")
    print(f"Each worker will process {len(dataset_splits[0])} questions")
    # 使用 Pool 并行处理
    with mp.Pool(processes=num_workers) as pool:
        results = pool.starmap(
            process_question_worker,
            [
                (entity_df, relation_df, relation_embedding, split)
                for split in dataset_splits
            ],
        )

    # 合并所有进程返回的结果
    all_results = [item for sublist in results for item in sublist]  # 扁平化列表

    # 保存结果
    for result in all_results:
        desc = result["desc"]
        question_id = result["question_id"]
        with open(f"{cached_desc}/{question_id}.txt", "w") as f:
            f.write(desc)

    return


def read_graph():
    entity_df = pd.read_csv(f"{path}/entity_df.csv")
    relation_df = pd.read_csv(f"{path}/relationships.csv")
    relation_embedding = pd.read_csv(f"{path}/relationships_embedding.csv")

    entity_df = entity_df[
        ["human_readable_id", "name", "node_description", "description_embedding"]
    ]
    relation_df = relation_df[
        ["human_readable_id", "head_id", "description", "tail_id", "embedding_idx"]
    ]

    entity_df["description_embedding"] = entity_df["description_embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
    )
    relation_embedding["description_embedding"] = relation_embedding[
        "description_embedding"
    ].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)
    print("finish read graph")

    entity_df["pcst_idx"] = range(len(entity_df))
    id_to_pcst_idx = dict(zip(entity_df["human_readable_id"], entity_df["pcst_idx"]))
    relation_df["head_pcst_idx"] = relation_df["head_id"].map(id_to_pcst_idx)
    relation_df["tail_pcst_idx"] = relation_df["tail_id"].map(id_to_pcst_idx)

    dataset = pd.read_json(dataset_path, lines=True, orient="records")
    dataset["embedding"] = dataset["embedding"].apply(
        lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x
    )
    dataset["question_id"] = range(len(dataset))

    return entity_df, relation_df, relation_embedding, dataset


if __name__ == "__main__":

    # entity_df, relation_df, relation_embedding, dataset = read_graph()
    # preprocess(entity_df, relation_df, relation_embedding, dataset, num_workers=64)

    dataset = WebQDataset()
    print(len(dataset))
    data = dataset[1]
    for k, v in data.items():
        print(f"{k}: {v}")

    # 将数据集拆分为多个子集，每个进程处理一个子集
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_size = dataset_size // 2
    from torch.utils.data import Subset

    # 生成数据子集
    subsets = [
        Subset(dataset, indices[i * split_size : (i + 1) * split_size])
        for i in range(2)
    ]
    for data in subsets[0]:
        print(data['desc'][:4000])
        print(data)
