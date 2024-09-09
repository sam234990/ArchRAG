import json
from llm import llm_invoker
from prompts import COMMUNITY_REPORT_PROMPT, COMMUNITY_CONTEXT
from utils import *
from attr_cluster import *
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
            [selected_relationships["source"], selected_relationships["target"]]
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


def extract_community_report(result):
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
            "title": None,
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
    current_generated = 0
    retries = 0
    success = False
    raw_result = None

    while not success and retries < max_generate:
        raw_result = llm_invoker(
            report_prompt, args, max_tokens=args.max_tokens, json=True
        )
        try:
            output = json.loads(raw_result)

            extract_result, success = extract_community_report(output)
            if success:
                break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            retries += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1

    if success is False:
        print(f"Failed to generate community report for community:{community_id}")
        return None, extract_result

    return raw_result, extract_result


def community_report_embedding(community_report, args):
    text = community_report["title"] + community_report["summary"]
    embedding = openai_embedding(
        text, args.embedding_api_key, args.embedding_api_base, args.embedding_model
    )
    return embedding


def community_report(results_by_level, args, final_entities, final_relationships):
    results_community = []
    for level, communities in results_by_level.items():
        print(f"Create community report for level: {level} ")
        print(f"Number of communities in this level: {len(communities)}")
        for community_id, node_list in communities.items():
            # if community_id != "9":
            #     continue
            print(f"Community {community_id}:")
            # print(f"Nodes:{node_list}")
            community_nodes = final_entities.loc[final_entities["name"].isin(node_list)]
            community_relationships = final_relationships[
                final_relationships["source"].isin(node_list)
            ]
            community_text = prep_community_report_context(
                0, community_nodes, community_relationships, max_tokens=args.max_tokens
            )

            raw_result, community_report = generate_community_report(
                community_text, args, community_id
            )
            # print(community_report)

            community_report["community_id"] = community_id
            community_report["level"] = level
            community_report["community_nodes"] = node_list
            community_report["raw_result"] = raw_result

            if raw_result is None or community_report is None:
                # use the community text as the embedding alternatively
                community_report["embedding"] = openai_embedding(
                    community_text,
                    args.embedding_api_key,
                    args.embedding_api_base,
                    args.embedding_model,
                )
            else:
                community_report["embedding"] = community_report_embedding(
                    community_report, args
                )
            results_community.append(community_report)

    community_df = pd.DataFrame(results_community)
    return community_df


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    graph, final_entities, final_relationships = read_graph_nx(args.base_path)
    cos_graph = compute_distance(graph=graph)
    results_by_level = attribute_hierarchical_clustering(cos_graph, final_entities)
    community_df = community_report(
        results_by_level, args, final_entities, final_relationships
    )
    output_path = "/home/wangshu/rag/hier_graph_rag/datasets_io/communities.csv"
    community_df.to_csv(output_path, index=False)
