import sys
import os

project_path = "/home/wangshu/rag/hier_graph_rag"
print(project_path)
sys.path.append(os.path.abspath(project_path))

from src.lm_emb import openai_embedding
from src.utils import *


def q_emb_compute(datasets, args):
    print(datasets.head(2))
    datasets = relation_embedding(
        datasets, args=args, e_colname="question", num_workers=28
    )
    return datasets


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
    dataset_path = "/mnt/data/wangshu/hcarag/FB15k/webqa/webqa.json"
    datasets = pd.read_json(dataset_path, lines=True, orient="records")
    embed_datasets = q_emb_compute(datasets, args)
    save_path = "/mnt/data/wangshu/hcarag/FB15k/webqa/webqa_emb.json"
    embed_datasets.to_json(save_path, orient="records", lines=True)
