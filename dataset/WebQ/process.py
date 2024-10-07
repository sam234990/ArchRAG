import sys
import os

# 添加 src 文件夹到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from hchnsw_index import entity_embedding

import pandas as pd
import os

base_path = "/mnt/data/wangshu/hcarag/FB15k"

node_df_raw = pd.read_csv(os.path.join(base_path, "./fb15k_description.tsv"), sep="\t")
relation_df_raw = pd.read_csv(
    os.path.join(base_path, "./freebase_mtr100_mte100-all.txt"), sep="\t", header=None
)
relation_df_raw.columns = ["head", "relation", "tail"]

print(node_df_raw.head(2))
relation_df_raw.head(2)
# 将\N替换为空字符串
node_df = node_df_raw.copy()
# node_df['node_description'] = node_df['node_description'].replace('\\N', '')
node_df["node_description"] = node_df["node_description"].replace("\\N", None)

# 删除node_description为空的行
node_df = node_df[node_df["node_description"].notnull()]

# 使用str.split方法按第一个逗号分割
node_df[["name", "description"]] = node_df["node_description"].str.split(
    ",", n=1, expand=True
)


# 将空字符串替换为None
node_df["name"] = node_df["name"].replace("", None)
node_df["description"] = node_df["description"].replace("", None)
node_df["node_description"] = node_df["node_description"].replace("", None)

# 统计name和description字段中空缺的数量
name_missing_count = node_df["name"].isnull().sum()
description_missing_count = node_df["description"].isnull().sum()

# 打印结果
print(node_df["node_description"].isnull().sum())
print(f"name字段空缺的数量: {name_missing_count}")
print(f"description字段空缺的数量: {description_missing_count}")

node_df


class Args:
    def __init__(self):
        self.embedding_local = True
        self.embedding_model_local = "nomic-embed-text-v1"
        # self.embedding_api_key = "ollama"
        # self.embedding_api_base = "http://localhost:11434/v1"
        # self.embedding_model = "nomic-embed-text"
        # 其他参数...


args = Args()
# 处理 node_df 中的 description 字段
node_df["description"] = node_df["description"].replace(
    [None, "", "None"], "."
)  # 替换 None 和空字符串为 “.”

node_df = entity_embedding(node_df, args, embed_colname="description_embedding")

print(node_df.head(2))
