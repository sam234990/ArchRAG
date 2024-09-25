import json
import multiprocessing as mp
from functools import partial

from llm import llm_invoker
from prompts import COMMUNITY_REPORT_PROMPT, COMMUNITY_CONTEXT
from utils import *
from lm_emb import *


def trim_community_context(community_nodes, relationships):
    entity_str = ""
    for _, row in community_nodes.iterrows():
        entity_str += f"{row['human_readable_id']},{row['name']},{row['description']}\n"
    relationships_str = ""
    for _, row in relationships.iterrows():
        relationships_str += f"{row['human_readable_id']},{row['source']},{row['target']},{row['description']}\n"

    context = COMMUNITY_CONTEXT.format(
        entity_df=entity_str, relationship_df=relationships_str
    )
    return context


def prep_community_report_context(
    level, community_nodes, relationships, sub_communities_summary=None, max_tokens=None
):
    relationships_sorted = relationships.copy()
    relationships_sorted["degree_sum"] = (
        relationships_sorted["source_degree"] + relationships_sorted["target_degree"]
    )
    relationships_sorted = relationships_sorted.sort_values(
        by="degree_sum", ascending=False
    )

    selected_relationships = pd.DataFrame(columns=relationships.columns)
    selected_entities = pd.DataFrame(columns=community_nodes.columns)

    new_string = ""
    for i in range(len(relationships_sorted)):
        selected_relationships = relationships_sorted.iloc[:i]

        # Filter entities involved in the selected relationships
        involved_entity_ids = pd.concat(
            [selected_relationships["head_id"], selected_relationships["tail_id"]]
        ).unique()
        selected_entities = community_nodes[
            community_nodes["human_readable_id"].isin(involved_entity_ids)
        ]

        if max_tokens:
            context = trim_community_context(selected_entities, selected_relationships)
            if num_tokens(context) > max_tokens:
                break
            new_string = context

    if new_string == "":
        return trim_community_context(
            community_nodes=community_nodes, relationships=relationships
        )

    return new_string


def extract_community_report(result, community_id):
    """
    Extract the fields from result if valid.
    """
    required_fields = [
        ("title", str),
        ("summary", str),
        ("findings", list),
        ("rating", float),
        ("rating_explanation", str),
    ]
    # Validate the result
    if dict_has_keys_with_types(result, required_fields):
        return {
            "title": result["title"],
            "summary": result["summary"],
            "findings": result["findings"],
            "rating": result["rating"],
            "rating_explanation": result["rating_explanation"],
        }, True
    else:
        return {
            "title": "CommunityID" + str(community_id),
            "summary": None,
            "findings": None,
            "rating": None,
            "rating_explanation": None,
        }, False


def dict_has_keys_with_types(d, keys_with_types):
    """Check if dict `d` contains keys with expected types as defined in `keys_with_types`."""
    for key, expected_type in keys_with_types:
        if key not in d or not isinstance(d[key], expected_type):
            return False
    return True


def generate_community_report(community_text, args, community_id, max_generate=3):
    report_prompt = COMMUNITY_REPORT_PROMPT.format(input_text=community_text)
    retries = 0
    success = False
    raw_result = None
    extract_result = None

    while not success and retries < max_generate:
        raw_result = llm_invoker(
            report_prompt, args, max_tokens=args.max_tokens, json=True
        )
        try:
            output = json.loads(raw_result)

            extract_result, success = extract_community_report(output, community_id)
            if success:
                break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        retries += 1

    if success is False:
        print(f"Failed to generate community report for community:{community_id}")
        if extract_result is None:
            extract_result = {
                "title": "CommunityID" + str(community_id),
                "summary": None,
                "findings": None,
                "rating": None,
                "rating_explanation": None,
            }
        return None, extract_result

    return raw_result, extract_result


def report_embedding(community_report, community_text, args):
    if community_report["title"] is None or community_report["summary"] is None:
        text = community_text
        community_report["title"] = "CommunityID" + str(
            community_report["community_id"]
        )
    else:
        text = community_report["title"] + community_report["summary"]
    embedding = openai_embedding(
        text, args.embedding_api_key, args.embedding_api_base, args.embedding_model
    )
    community_report["embedding"] = embedding
    return community_report


def reprot_embedding_batch(community_df, args):
    def embedding_context(row):
        if row["summary"] is None:
            row["embedding_context"] = row["title"] + row["community_text"]
        else:
            row["embedding_context"] = row["title"] + row["summary"]

    community_df = community_df.apply(embedding_context, axis=1)

    if args.embedding_local:
        model, tokenizer, device = load_sbert(args.embedding_model_local)

        # 计算 embeddings
        texts = community_df["embedding_context"].tolist()
        community_df["embedding"] = text_to_embedding_batch(
            model, tokenizer, device, texts
        )
    else:
        for i, row in community_df.iterrows():
            community_df.at[i, "embedding"] = openai_embedding(
                row["embedding_context"],
                args.embedding_api_key,
                args.embedding_api_base,
                args.embedding_model,
            )

    community_df = community_df.drop(columns=["embedding_context"], inplace=True)
    return community_df


def community_report_worker(
    community_id, node_list, final_entities, final_relationships, args, level_dict
):
    # 处理单个社区的函数
    print(f"Community {community_id}:")
    # community_nodes = final_entities.loc[final_entities["name"].isin(node_list)]
    community_nodes = final_entities.loc[
        final_entities["human_readable_id"].isin(node_list)
    ]

    # community_relationships = final_relationships[
    #     final_relationships["source"].isin(node_list)
    # ]
    community_relationships = final_relationships[
        final_relationships["head_id"].isin(node_list)
    ]

    community_text = prep_community_report_context(
        0, community_nodes, community_relationships, max_tokens=args.max_tokens
    )

    raw_result, community_report = generate_community_report(
        community_text, args, community_id
    )

    community_report["community_id"] = community_id
    community_report["level"] = level_dict.get(
        community_id, None
    )  # 使用 level_dict 获取对应的 level
    community_report["community_nodes"] = node_list
    community_report["raw_result"] = raw_result
    community_report["community_text"] = community_text

    if not community_report.get("title"):
        community_report["title"] = f"CommunityID{community_id}"

    # if raw_result is None or community_report is None:
    #     # use the community text as the embedding alternatively
    #     community_report["embedding"] = openai_embedding(
    #         community_text,
    #         args.embedding_api_key,
    #         args.embedding_api_base,
    #         args.embedding_model,
    #     )
    # else:
    #     community_report = report_embedding(community_report, community_text, args)

    return community_report


def community_report_batch(
    communities: dict[str, list[str]],
    final_entities,
    final_relationships,
    args,
    level_dict: dict[str, int],
    num_workers=8,
):
    results_community = []

    # 将处理任务分发到各个进程
    pool = mp.Pool(processes=num_workers)

    # 使用partial来将固定的参数传递到worker中
    process_func = partial(
        community_report_worker,
        final_entities=final_entities,
        final_relationships=final_relationships,
        args=args,
        level_dict=level_dict,
    )

    # 并行处理每个社区
    results_community = pool.starmap(process_func, communities.items())

    pool.close()  # 关闭进程池
    pool.join()  # 等待所有进程完成

    community_df = pd.DataFrame(results_community)
    community_df = reprot_embedding_batch(community_df, args)
    
    return community_df


def community_report_for_level(
    results_by_level, args, final_entities, final_relationships
):
    results_community = []

    for level, communities in results_by_level.items():
        print(f"Create community report for level: {level} ")
        print(f"Number of communities in this level: {len(communities)}")
        # 构建一个 level_dict，key 为 community_id，value 为 level
        level_dict = {community_id: level for community_id in communities.keys()}

        res = community_report_batch(
            communities=communities,
            final_entities=final_entities,
            final_relationships=final_relationships,
            args=args,
            level_dict=level_dict,
        )

        results_community.append(res)  # 将 DataFrame 添加到列表中

    # 使用 pd.concat 合并所有 DataFrame
    community_df = pd.concat(results_community, ignore_index=True)
    return community_df


def confirm_community_result(results_by_level):
    levels = sorted(
        results_by_level.keys(), reverse=True
    )  # Start from the highest level
    lower_to_higher_assignments = (
        {}
    )  # To track the assignment of lower-level communities

    for i in range(len(levels) - 1):  # Compare each level with the next lower level
        higher_level = levels[i]
        lower_level = levels[i + 1]
        print(f"Checking level {higher_level} against level {lower_level}...")

        higher_level_communities = results_by_level[higher_level]
        lower_level_communities = results_by_level[lower_level]

        # Track which lower-level communities have already been assigned to a higher-level community
        lower_community_to_higher = {}
        for higher_community_id, higher_node_list in higher_level_communities.items():
            higher_node_set = set(higher_node_list)

            # Check each node in the higher-level community against the lower-level communities
            for lower_community_id, lower_node_list in lower_level_communities.items():
                lower_node_set = set(lower_node_list)

                # If any nodes from the lower-level community are in the higher-level community
                if higher_node_set.intersection(lower_node_set):
                    if lower_community_id in lower_community_to_higher:
                        # If this lower-level community is already assigned to a different higher-level community
                        assigned_higher_community = lower_community_to_higher[
                            lower_community_id
                        ]
                        if assigned_higher_community != higher_community_id:
                            print(
                                f"Error: Lower-level community {lower_community_id} is part of both "
                                f"higher-level community {higher_community_id} and {assigned_higher_community}!"
                            )
                    else:
                        # Assign this lower-level community to the current higher-level community
                        lower_community_to_higher[lower_community_id] = (
                            higher_community_id
                        )

                        # Store the assignment in the tracking dictionary
                        if lower_community_id not in lower_to_higher_assignments:
                            lower_to_higher_assignments[lower_community_id] = []
                        lower_to_higher_assignments[lower_community_id].append(
                            higher_community_id
                        )

        print(f"Finished checking level {higher_level} against level {lower_level}.\n")

    # Output the assignment results
    print("Assignment of lower-level communities to higher-level communities:")
    for lower_community_id, higher_communities in lower_to_higher_assignments.items():
        print(
            f"Lower-level community {lower_community_id} is assigned to higher-level communities: {higher_communities}"
        )

        # Optionally, output the contents (nodes) of the assigned communities
        for higher_level in higher_communities:
            print(
                f"  Higher-level community {higher_level} content: {results_by_level[levels[i]][higher_level]}"
            )
        print()

    print("Finished checking all levels.")


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
    cos_graph = compute_distance(graph=graph)
    # results_by_level = attribute_hierarchical_clustering(cos_graph, final_entities)

    # confirm_community_result(results_by_level)

    # community_df = community_report_for_level(
    #     results_by_level, args, final_entities, final_relationships
    # )

    # output_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/communities.csv"
    # community_df.to_csv(output_path, index=False)
