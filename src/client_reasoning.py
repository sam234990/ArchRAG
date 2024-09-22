import json
import re
import pandas as pd
from typing import Tuple, Dict, Any
from llm import llm_invoker
from utils import create_arg_parser, num_tokens
from prompts import *


def problem_reasoning(
    query_content: str, index_info: str, num_level, max_retries: int, args
) -> list[int]:
    reason_level = []
    retry = 0
    success = False

    while not success and retry < max_retries:
        reason_level = llm_invoker(query_content, args, json=True)
        try:
            output: dict = json.loads(reason_level)
            extract_level, success = extract_level(output, num_level)
            reason_level = extract_level["level"]

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            retries += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1

    return reason_level


def extract_level(level_output, num_level) -> Tuple[Dict[str, Any], bool]:
    res_level = []
    res_description = ""
    success = True

    if "level" in level_output:
        level_data = level_output["level"]

        # 检查 'level' 是否为 list 类型且长度为 num_level
        if (
            isinstance(level_data, list)
            and len(level_data) == num_level
            and all(isinstance(i, (int, float)) for i in level_data)
        ):
            res_level = level_data
        else:
            # 尝试从非list的level_data中提取数字并转换成list
            if isinstance(level_data, str):
                # 提取数字
                level_values = re.findall(r"[-+]?\d*\.\d+|\d+", level_data)
                res_level = [float(x) if "." in x else int(x) for x in level_values]

            elif isinstance(level_data, (int, float)):
                res_level = [level_data]  # 单个数字转为列表

            # 检查转换后的list是否符合num_level的要求
            if len(res_level) != num_level:
                res_level = []
                success = False

    else:
        success = False

    # 检测 'description' 字段是否存在且为str类型
    if "description" in level_output and isinstance(level_output["description"], str):
        res_description = level_output["description"]
    else:
        success = False

    return {
        "level": res_level,
        "description": res_description if success else None,
    }, success


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
