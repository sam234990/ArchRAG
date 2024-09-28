import json
import re
import pandas as pd
import numpy as np
import math
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any
from src.llm import llm_invoker
from src.utils import *
from src.prompts import *


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
    sampled_df = level_df.sample(n=min(level_df.shape[0], sample_size), random_state=42)

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


def trim_e_r_content(community_nodes, relationships):
    entity_str = ""

    # 检测 community_nodes 是否为 Series，并转换为 DataFrame
    if isinstance(community_nodes, pd.Series):
        community_nodes = community_nodes.to_frame().T

    for _, row in community_nodes.iterrows():
        entity_str += f"{row['human_readable_id']},{row['name']},{row['description']}\n"

    relationships_str = ""

    # 检测 relationships 是否为 Series，并转换为 DataFrame
    if isinstance(relationships, pd.Series):
        relationships = relationships.to_frame().T

    for _, row in relationships.iterrows():
        relationships_str += f"{row['human_readable_id']},{row['source']},{row['target']},{row['description']}\n"

    context = COMMUNITY_CONTEXT.format(
        entity_df=entity_str, relationship_df=relationships_str
    )
    return context


def prep_e_r_content(entity_df, relation_df, max_tokens=None):
    if entity_df.empty:
        return [COMMUNITY_CONTEXT]

    if relation_df.empty:
        return [trim_e_r_content(entity_df, relation_df)]

    relationships_sorted = relation_df.copy()
    relationships_sorted["degree_sum"] = (
        relationships_sorted["source_degree"] + relationships_sorted["target_degree"]
    )
    relationships_sorted = relationships_sorted.sort_values(
        by="degree_sum", ascending=False
    )

    selected_relationships = pd.DataFrame(columns=relation_df.columns)
    selected_entities = pd.DataFrame(columns=entity_df.columns)

    new_string = ""
    res_string_list = []
    start = 0
    for i in range(len(relationships_sorted)):
        selected_relationships = relationships_sorted.iloc[start:i]

        # Filter entities involved in the selected relationships
        involved_entity_ids = pd.concat(
            [selected_relationships["head_id"], selected_relationships["tail_id"]]
        ).unique()
        selected_entities = entity_df[
            entity_df["human_readable_id"].isin(involved_entity_ids)
        ]

        if max_tokens:
            context = trim_e_r_content(selected_entities, selected_relationships)
            if num_tokens(context) > max_tokens:
                # Append the current new_string if it's not empty
                if new_string:
                    res_string_list.append(new_string)
                new_string = context  # Start a new context
                start = i + 1  # Move the start to the next relationship
            else:
                new_string = (
                    context  # Update new_string if it doesn't exceed max_tokens
                )

    # Append any remaining content in new_string to res_string_list
    if new_string:
        res_string_list.append(new_string)

    # If no valid context was generated, handle the case
    if not res_string_list:
        shortest_entity = entity_df.loc[entity_df["description"].str.len().idxmin()]
        # 获取 shortest_entity 中的所有 human_readable_id
        human_readable_ids = shortest_entity["human_readable_id"].tolist()

        related_relationships = relation_df[
            relation_df["head_id"].isin(human_readable_ids)
        ]
        new_string = trim_e_r_content(shortest_entity, related_relationships)
        if new_string:
            res_string_list.append(new_string)

    return res_string_list


def prep_infer_content(
    entity_df, relation_df, community_df, query, max_tokens=None, response_type="QA"
) -> str:

    res_content = ""
    if not entity_df.empty:
        e_r_content = prep_e_r_content(entity_df, relation_df, max_tokens=max_tokens)
        res_content += e_r_content[0]

    if not community_df.empty:
        c_content = prep_community_content(community_df, max_tokens)
        res_content += "\n\n"
        res_content += c_content[0]

    response_type_content = GENERATION_RESPONSE_FORMAT[response_type]

    res_content = GENERATION_PROMPT.format(
        context_data=res_content,
        user_query=query,
        response_format=response_type_content,
    )
    return res_content.strip()


def prep_community_content(community_df, max_tokens=None) -> list[str]:
    community_str = """
id|title|content|rank \n
"""
    res_list = []
    if not community_df.empty:
        community_chunk_str = ""
        chunk_str = community_str

        for idx, row in community_df.iterrows():
            new_entry = f"{row['community_id']}|{row['title']}|{row['summary']}|{row['rating']}\n"
            # Check if adding the new entry exceeds max_tokens
            if max_tokens and num_tokens(chunk_str + new_entry) > max_tokens:
                # Add current chunk to the results
                res_list.append(chunk_str)
                # Reset the chunk_str and start a new one
                chunk_str = community_str
            # Add the new entry to the chunk_str
            chunk_str += new_entry

        # Add the last chunk if it contains any entries
        if chunk_str != community_str:
            res_list.append(chunk_str)

    return res_list


def prep_map_content(
    entity_df: pd.DataFrame,
    relation_df: pd.DataFrame,
    community_df: pd.DataFrame,
    query,
    max_tokens=None,
) -> str:
    if not entity_df.empty:
        e_r_prefix = (
            "Here is a community consisting of the following entities and their "
            "associated relationships. In order to adapt to the specific task "
            "presented above, you should consider the information about this "
            "community as a cohesive whole.\n\n"
        )
        e_r_content = prep_e_r_content(entity_df, relation_df, max_tokens=max_tokens)
        er_chunk = [e_r_prefix + er for er in e_r_content]
        er_chunk = [
            GLOBAL_MAP_SYSTEM_PROMPT.format(context_data=er, user_query=query)
            for er in er_chunk
        ]

    community_chunk = prep_community_content(community_df, max_tokens)
    community_chunk = [
        GLOBAL_MAP_SYSTEM_PROMPT.format(context_data=c, user_query=query)
        for c in community_chunk
    ]
    all_chunks = er_chunk + community_chunk
    return all_chunks


def map_llm_worker(content, args):
    retries = 0
    points_data = []  # List to hold the descriptions and scores

    while retries < args.max_retries:
        raw_result = llm_invoker(content, args, max_tokens=args.max_tokens, json=True)
        output, json_output = try_parse_json_object(raw_result or "")

        if "points" in json_output and isinstance(json_output["points"], list):
            all_points_valid = True  # Flag to check if all points are valid

            for kv in json_output["points"]:
                if "description" in kv and "score" in kv:
                    score_value = (
                        float(kv["score"])
                        if isinstance(kv["score"], (int, float))
                        else str(kv["score"])
                    )
                    points_data.append(
                        {
                            "answer": kv["description"],
                            "score": score_value,  # Use the converted score
                        }
                    )
                else:
                    all_points_valid = False
                    break

            if all_points_valid and points_data:
                break

        retries += 1

    return points_data


def map_inference(entity_df, relation_df, community_df, query, args):
    all_chunks = prep_map_content(
        entity_df=entity_df,
        relation_df=relation_df,
        community_df=community_df,
        query=query,
        max_tokens=args.max_tokens,
    )

    max_workers = max(len(all_chunks), args.num_workers)

    with mp.Pool(processes=max_workers) as pool:
        results = pool.starmap(
            map_llm_worker,
            [(chunk, args) for chunk in all_chunks],
        )

    # Flatten the results to a 1D list
    flattened_results = [item for sublist in results for item in sublist]

    res_df = pd.DataFrame(flattened_results)
    return res_df


def prep_reduce_content(map_response_df: pd.DataFrame, max_tokens=None) -> str:
    # collect all key points into a single list to prepare for sorting
    key_points = []
    for index, row in map_response_df.iterrows():
        if (
            row["answer"] and row["score"] is not None
        ):  # Check if both are not empty or None
            key_points.append(
                {
                    "analyst": index,
                    "answer": row["answer"],
                    "score": row["score"],
                }
            )

    # filter response with score = 0 and rank responses by descending order of score
    filtered_key_points = [
        point for point in key_points if point["score"] > 0  # type: ignore
    ]

    if len(filtered_key_points) == 0:
        return ""

    filtered_key_points = sorted(
        filtered_key_points,
        key=lambda x: x["score"],  # type: ignore
        reverse=True,  # type: ignore
    )

    data = []
    total_tokens = 0
    for point in filtered_key_points:
        formatted_response_data = []
        formatted_response_data.append(f'----Analyst {point["analyst"] + 1}----')
        formatted_response_data.append(
            f'Importance Score: {point["score"]}'  # type: ignore
        )
        formatted_response_data.append(point["answer"])  # type: ignore
        formatted_response_text = "\n".join(formatted_response_data)
        if total_tokens + num_tokens(formatted_response_text) > max_tokens:
            break
        data.append(formatted_response_text)
        total_tokens += num_tokens(formatted_response_text)
    text_data = "\n\n".join(data)
    return text_data


def reduce_inference(map_res_df, query, args, response_type="QA"):
    reduce_context = prep_reduce_content(map_res_df, max_tokens=args.max_tokens)
    response_type_content = GENERATION_RESPONSE_FORMAT[response_type]
    reduce_prompt = GLOBAL_REDUCE_SYSTEM_PROMPT.format(
        report_data=reduce_context, user_query=query, response_format=response_type_content
    )
    if reduce_context == "":
        reduce_prompt += GENERAL_KNOWLEDGE_INSTRUCTION

    retries = 0
    response = ""

    while retries < args.max_retries:
        raw_result = llm_invoker(
            reduce_prompt, args, max_tokens=args.max_tokens, json=False
        )
        if raw_result != "":
            response = raw_result
            break
        retries += 1

    response_report = {"response": response, "raw_result": raw_result}

    return response_report


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()
