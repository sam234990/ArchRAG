import json
import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from llm import llm_invoker
from utils import create_arg_parser, num_tokens
from prompts import *


def problem_reasoning(
    query_content: str,
    entity_df: pd.DataFrame,
    community_df: pd.DataFrame,
    level_summary_df: pd.DataFrame,
    max_level,
    max_retries: int,
    args,
) -> list[int]:
    infer_content = prep_level_infer_content(
        query_content,
        entity_df,
        community_df,
        level_summary_df,
        max_level,
        max_tokens=args.max_tokens,
    )

    reason_level = []
    retry = 0
    success = False

    while not success and retry < max_retries:
        reason_level = llm_invoker(infer_content, args, json=True)
        try:
            output: dict = json.loads(reason_level)
            raw_result, rate_list, success = extract_level(output, max_level)
            reason_level = rate_list
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        if success:
            break

        retry += 1

    # entity importance is the mean value
    if reason_level:
        # 计算平均值
        average_value = sum(reason_level) / len(reason_level)

        # 将平均值添加到 reason_level 的最前面
        reason_level.insert(0, average_value)
    else:
        reason_level = [5] * (max_level + 1)

    return reason_level, raw_result


def prep_level_infer_content(
    query_content, entity_df, community_df, level_summary_df, max_level, max_tokens=None
) -> str:
    index_conent = f"""
Max level: {max_level}
Indexed Data:

level_id, community_number, level_summary, community_example  
"""

    for level in range(1, max_level + 1):

        level_summary_content = level_summary_df[level_summary_df["level"] == level][
            "summary"
        ].values[0]
        community_number = level_summary_df[level_summary_df["level"] == level][
            "comunity_number"
        ].values[0]

        level_community_df = community_df[community_df["level"] == level]

        sampled_df = level_community_df.sample(n=1, random_state=42)
        # 构建字典并转换为字符串格式
        level_example_dict = {
            "title": sampled_df["title"].values[0],
            "summary": sampled_df["summary"].values[0],
        }

        # 将字典转换为字符串形式
        level_example_content = str(level_example_dict)
        level_content = f"{level}, {community_number}, {level_summary_content}, {level_example_content}\n"
        index_conent += level_content

    index_conent += f"Query: {query_content}\n"

    return LEVEL_INFERENCE_PROMPT.format(indexed_structure_data=index_conent)


def extract_level(level_output, num_level) -> Tuple[Dict[str, Any], bool]:
    infer_level_report = []
    success = True
    rate_list = []

    if "finds" in level_output:
        finds_data = level_output["finds"]

        # 确认 "finds" 是一个列表并且长度为 num_level
        if isinstance(finds_data, list) and len(finds_data) == num_level:
            for find in finds_data:
                level_id = find.get("id", None)
                rate = find.get("rate", 5.0)
                rating_explanation = find.get("rating_explanation", "")

                infer_level_report.append(
                    {
                        "id": level_id,
                        "rate": rate,
                        "rating_explanation": rating_explanation,
                    }
                )
                rate_list.append(float(rate))  # 添加 rate 到 rate_list

            # 如果 finds 中字典数量不足 num_level，用占位数据补齐
            while len(infer_level_report) < num_level:
                infer_level_report.append(
                    {"id": "N/A", "rate": 5.0, "rating_explanation": None}
                )
                rate_list.append(5.0)  # 补齐 rate_list

            # 如果 finds 中字典数量超过 num_level，截断多余部分
            if len(infer_level_report) > num_level:
                infer_level_report = infer_level_report[:num_level]
                rate_list = rate_list[:num_level]  # 截断 rate_list

        else:
            success = False
    else:
        success = False

    return infer_level_report, rate_list, success


def prep_level_content(
    level, max_level, community_df, sample_size=3, max_tokens=None
) -> str:
    level_content = f"""
Max level: {max_level}
Community level: {level}

Sample Communities:

community_id, title, summary \n"""
    # 选出 "level" 等于指定值的行
    level_df = community_df[community_df["level"] == level]

    # 随机选择 sample_size 个样本
    sampled_df = level_df.sample(n=sample_size, random_state=42)

    res_level_context = ""
    # 遍历 sampled_df 并填充 community_id, title, summary 到字符串中
    for _, row in sampled_df.iterrows():
        level_content += f"{row['community_id']}, {row['title']}, {row['summary']}\n"
        if max_tokens and num_tokens(level_content) > max_tokens:
            break
        else:
            res_level_context = level_content

    if res_level_context == "":
        # 找到 summary 字段长度最短的行
        shortest_summary_row = community_df.loc[
            community_df["summary"].str.len().idxmin()
        ]

        res_level_context = (
            level_content
            + f"{shortest_summary_row['community_id']}, {shortest_summary_row['title']}, {shortest_summary_row['summary']}\n"
        )

    res_level_context = LEVEL_SUMMARY_PROMPT.format(Community_text=res_level_context)
    return res_level_context


def generate_level_summary(content, max_retries, args):
    retries = 0
    success = False
    level_summary = ""

    while not success and retries < max_retries:
        raw_result = llm_invoker(content, args, max_tokens=args.max_tokens, json=True)
        try:
            output = json.loads(raw_result)
            if "summary" in output:
                level_summary = output["summary"]
                break

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        retries += 1

    level_report = {"summary": level_summary, "raw_result": raw_result}

    return level_report


def level_summary(community_df, max_level, args):

    level_summary = []
    for level in range(1, max_level + 1):
        level_content = prep_level_content(
            level, max_level, community_df, sample_size=3, max_tokens=args.max_tokens
        )
        report = generate_level_summary(level_content, args.max_retries, args)
        report["level"] = level
        level_comunity_number = community_df[community_df["level"] == level].shape[0]
        report["comunity_number"] = level_comunity_number

        level_summary.append(report)

    level_summary = pd.DataFrame(level_summary)
    return level_summary


def prep_infer_content(
    entity_context: list[str], community_context: list[str], query, max_tokens=None
) -> str:
    res_content = ""
    entity_str = """
ENTITY:
entity_id, name, description \n
"""
    community_str = """
COMMUNITY:
community_id, title, summary \n
"""

    if entity_context:
        for entity in entity_context:
            entity_str += entity + "\n"
            if max_tokens and num_tokens(entity_str) > max_tokens:
                break
            else:
                res_content = entity_str
        res_content += "\n"

    if community_context:
        community_str = res_content + community_str
        for community in community_context:
            community_str += community + "\n"
            if max_tokens and num_tokens(community_str) > max_tokens:
                break
            else:
                res_content = community_str
        res_content += "\n"

    res_content = GENERATION_PROMPT.format(context_data=res_content, user_query=query)
    return res_content.strip()


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
